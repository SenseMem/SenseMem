"""Pinned vLLM Qwen3-VL reranking of text and original visual assets."""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, closing, nullcontext
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypedDict, cast

from sensemem._telemetry import mark_model_requests, record_unmetered_model_usage
from sensemem._validation import commit_revision, nonempty_text, optional_text, positive_int
from sensemem.exceptions import ModelError, ValidationError
from sensemem.models.base import ModelInput
from sensemem.models.reranking import reranking_inputs, reranking_scores
from sensemem.types import AssetRef, Modality

_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)
_RECIPE = "qwen3-vl-multimodal-vllm-pooling-v1"


class _ScoringEngine(Protocol):
    def score(
        self,
        query: ModelInput,
        documents: Sequence[ModelInput],
        *,
        max_length: int,
    ) -> object: ...

    def close(self) -> None: ...


class _TextPrompt(TypedDict):
    prompt: str


class _Prompt(_TextPrompt, total=False):
    multi_modal_data: dict[str, list[object]]


class _Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class _ClassificationOutput(Protocol):
    probs: list[float]


class _ClassificationRequestOutput(Protocol):
    outputs: _ClassificationOutput


class _EngineCore(Protocol):
    def shutdown(self) -> None: ...


class _LLMEngine(Protocol):
    engine_core: _EngineCore


class _LLM(Protocol):
    llm_engine: _LLMEngine

    def get_tokenizer(self) -> _Tokenizer: ...

    def classify(
        self, prompts: list[_Prompt], *, use_tqdm: bool
    ) -> Sequence[_ClassificationRequestOutput]: ...


class _LLMFactory(Protocol):
    def __call__(self, **kwargs: object) -> _LLM: ...


class _Image(Protocol):
    def close(self) -> None: ...


class _VideoFrames(Protocol):
    @property
    def nbytes(self) -> int: ...


_Video = tuple[_VideoFrames, dict[str, object]]


class _MediaLoader(Protocol):
    def fetch_image(self, image_url: str) -> _Image: ...

    def fetch_video(self, video_url: str, video_io_kwargs: dict[str, object]) -> _Video: ...


class _VideoCache:
    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._bytes = 0
        self._entries: OrderedDict[tuple[str, int, str], _Video] = OrderedDict()
        self._lock = threading.Lock()

    def load(self, asset: AssetRef, frames: int, loader: _MediaLoader) -> _Video:
        key = (cast(str, asset.sha256), frames, "opencv")
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached
        video = loader.fetch_video(
            cast(Path, asset.path).resolve().as_uri(),
            {"num_frames": frames, "video_backend": "opencv"},
        )
        size = video[0].nbytes
        if 0 < size <= self._max_bytes:
            with self._lock:
                # Another request may have decoded this asset while the lock was released.
                if key in self._entries:
                    self._entries.move_to_end(key)
                    return self._entries[key]
                while self._bytes + size > self._max_bytes:
                    _, evicted = self._entries.popitem(last=False)
                    self._bytes -= evicted[0].nbytes
                self._entries[key] = video
                self._bytes += size
        return video

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0


def _prompt(query: str, document: str) -> str:
    # The publisher's reranker_template.jinja; source data are never interpreted as a template.
    return (
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {_INSTRUCTION}<Query>:{query}\n"
        f"<Document>:{document}<|im_end|>\n<|im_start|>assistant\n"
    )


def _input_text(value: ModelInput) -> str:
    return (
        "".join(
            f"<|vision_start|><|{cast(Modality, asset.modality).value}_pad|><|vision_end|>"
            for asset in value.assets
        )
        + value.text
    )


class _VLLMScoringEngine:
    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        batch_size: int,
        max_length: int,
        gpu_memory_utilization: float,
        image_limit: int,
        video_limit: int,
        image_max_pixels: int,
        video_frames: int,
        video_cache_bytes: int,
    ) -> None:
        try:
            factory = cast(_LLMFactory, import_module("vllm").LLM)
        except (AttributeError, ImportError):
            raise ModelError(
                "vLLM is unavailable; install SenseMem with the reranking extra",
                stage="reranking",
            ) from None
        self._image_limit = image_limit
        self._video_limit = video_limit
        self._video_frames = video_frames
        self._batch_size = batch_size
        self._videos = _VideoCache(video_cache_bytes)
        self._inference_lock = threading.Lock()
        self._llm = factory(
            model=model_id,
            revision=revision,
            tokenizer_revision=revision,
            runner="pooling",
            dtype="auto",
            trust_remote_code=False,
            load_format="safetensors",
            hf_overrides={
                "architectures": ["Qwen3VLForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
            max_model_len=max_length,
            max_num_seqs=batch_size,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": image_limit, "video": video_limit},
            mm_processor_kwargs={"max_pixels": image_max_pixels, "do_sample_frames": False},
            disable_log_stats=True,
        )
        self._tokenizer = self._llm.get_tokenizer()

    def _load_media(
        self, value: ModelInput, loaded: dict[AssetRef, object], cleanup: ExitStack
    ) -> dict[str, list[object]]:
        media: dict[str, list[object]] = {}
        for asset in value.assets:
            kind = cast(Modality, asset.modality).value
            if asset not in loaded:
                loader = cast(_MediaLoader, import_module("vllm.multimodal.utils"))
                if asset.modality is Modality.IMAGE:
                    image = loader.fetch_image(cast(Path, asset.path).resolve().as_uri())
                    cleanup.callback(image.close)
                    loaded[asset] = image
                else:
                    loaded[asset] = self._videos.load(asset, self._video_frames, loader)
            media.setdefault(kind, []).append(loaded[asset])
        return media

    def _prepare_batch(
        self,
        documents: Sequence[ModelInput],
        prompts: list[str],
        query_media: dict[str, list[object]],
        query_loaded: dict[AssetRef, object],
    ) -> tuple[list[_Prompt], ExitStack]:
        with ExitStack() as cleanup:
            loaded = query_loaded.copy()
            rows: list[_Prompt] = []
            for document, prompt in zip(documents, prompts, strict=True):
                document_media = self._load_media(document, loaded, cleanup)
                row = _Prompt(prompt=prompt)
                media = {
                    kind: query_media.get(kind, []) + document_media.get(kind, [])
                    for kind in query_media.keys() | document_media.keys()
                }
                if media:
                    row["multi_modal_data"] = media
                rows.append(row)
            return rows, cleanup.pop_all()

    def _prepared_batches(
        self,
        documents: Sequence[ModelInput],
        prompts: list[str],
        query_media: dict[str, list[object]],
        query_loaded: dict[AssetRef, object],
    ) -> Generator[list[_Prompt], None, None]:
        # At most one batch is prepared ahead of the batch currently running on the GPU.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensemem-reranking")
        pending: Future[tuple[list[_Prompt], ExitStack]] | None = executor.submit(
            self._prepare_batch,
            documents[: self._batch_size],
            prompts[: self._batch_size],
            query_media,
            query_loaded,
        )
        next_start = self._batch_size
        try:
            while pending is not None:
                rows, cleanup = pending.result()
                with cleanup:
                    pending = (
                        executor.submit(
                            self._prepare_batch,
                            documents[next_start : next_start + self._batch_size],
                            prompts[next_start : next_start + self._batch_size],
                            query_media,
                            query_loaded,
                        )
                        if next_start < len(documents)
                        else None
                    )
                    next_start += self._batch_size
                    yield rows
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            if pending is not None and not pending.cancelled() and pending.exception() is None:
                pending.result()[1].close()

    def score(
        self,
        query: ModelInput,
        documents: Sequence[ModelInput],
        *,
        max_length: int,
    ) -> object:
        prompts: list[str] = []
        query_text = _input_text(query)
        for document in documents:
            for kind, limit in (
                (Modality.IMAGE, self._image_limit),
                (Modality.VIDEO, self._video_limit),
            ):
                if (
                    sum(asset.modality is kind for asset in (*query.assets, *document.assets))
                    > limit
                ):
                    raise ModelError(
                        "reranking pair exceeds its media limit",
                        reason="payload_too_large",
                        stage="reranking",
                    )
            prompt = _prompt(query_text, _input_text(document))
            if len(self._tokenizer.encode(prompt, add_special_tokens=False)) > max_length:
                raise ModelError(
                    "reranking pair exceeds its context budget",
                    reason="payload_too_large",
                    stage="reranking",
                )
            prompts.append(prompt)
        with ExitStack() as cleanup:
            loaded: dict[AssetRef, object] = {}
            query_media = self._load_media(query, loaded, cleanup)
            scores: list[float] = []
            request_count = 0
            with closing(
                self._prepared_batches(documents, prompts, query_media, loaded)
            ) as batches:
                for rows in batches:
                    with self._inference_lock:
                        request_count += 1
                        mark_model_requests(request_count, token_usage_expected=0)
                        outputs = self._llm.classify(rows, use_tqdm=False)
                    values = [reranking_scores(output.outputs.probs, 1)[0] for output in outputs]
                    scores.extend(reranking_scores(values, len(rows)))
                    record_unmetered_model_usage(request_count=request_count)
            return scores

    def close(self) -> None:
        self._videos.clear()
        self._llm.llm_engine.engine_core.shutdown()
        del self._llm


class Qwen3VLReranker:
    """Automatically score each pair's available text, images and videos in input order."""

    def __init__(
        self,
        engine: _ScoringEngine | None = None,
        *,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 8192,
        gpu_memory_utilization: float = 0.9,
        image_limit: int = 8,
        video_limit: int = 2,
        image_max_pixels: int = 262144,
        video_frames: int = 8,
        video_cache_bytes: int = 268435456,
    ) -> None:
        self._model_id = nonempty_text(model_id, "model_id")
        device = optional_text(device, "device")
        if device not in (None, "cuda"):
            raise ValidationError(
                "device must be cuda or None; select GPUs with CUDA_VISIBLE_DEVICES"
            )
        batch_size = positive_int(batch_size, "batch_size")
        self._max_length = positive_int(max_length, "max_length")
        image_limit = positive_int(image_limit, "image_limit")
        video_limit = positive_int(video_limit, "video_limit")
        self._image_max_pixels = positive_int(image_max_pixels, "image_max_pixels")
        self._video_frames = positive_int(video_frames, "video_frames")
        if not 2 <= self._video_frames <= 64:
            raise ValidationError("video_frames must be between 2 and 64")
        if (
            isinstance(video_cache_bytes, bool)
            or not isinstance(video_cache_bytes, int)
            or video_cache_bytes < 0
        ):
            raise ValidationError("video_cache_bytes must be a non-negative integer")
        if (
            isinstance(gpu_memory_utilization, bool)
            or not isinstance(gpu_memory_utilization, (int, float))
            or not 0 < gpu_memory_utilization <= 1
        ):
            raise ValidationError("gpu_memory_utilization must be greater than 0 and at most 1")
        if revision is None:
            try:
                revision = import_module("huggingface_hub").model_info(self._model_id).sha
            except Exception:
                raise ModelError(
                    "failed to resolve an immutable reranking model revision", stage="reranking"
                ) from None
        self._revision = commit_revision(revision)
        self._score_lock = threading.RLock() if engine is not None else nullcontext()
        self._native_engine = engine is None
        if engine is None:
            try:
                engine = _VLLMScoringEngine(
                    self._model_id,
                    self._revision,
                    batch_size=batch_size,
                    max_length=self._max_length,
                    gpu_memory_utilization=float(gpu_memory_utilization),
                    image_limit=image_limit,
                    video_limit=video_limit,
                    image_max_pixels=self._image_max_pixels,
                    video_frames=self._video_frames,
                    video_cache_bytes=video_cache_bytes,
                )
            except ModelError:
                raise
            except Exception:
                raise ModelError("failed to load the reranking model", stage="reranking") from None
        self._engine: _ScoringEngine | None = engine
        self._lock = threading.Condition()
        self._active_calls = 0
        self._closing = False

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 8192,
        gpu_memory_utilization: float = 0.9,
        image_limit: int = 8,
        video_limit: int = 2,
        image_max_pixels: int = 262144,
        video_frames: int = 8,
        video_cache_bytes: int = 268435456,
    ) -> Qwen3VLReranker:
        return cls(
            model_id=model_id,
            revision=revision,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            gpu_memory_utilization=gpu_memory_utilization,
            image_limit=image_limit,
            video_limit=video_limit,
            image_max_pixels=image_max_pixels,
            video_frames=video_frames,
            video_cache_bytes=video_cache_bytes,
        )

    @property
    def reranking_capabilities(self) -> frozenset[Modality]:
        return frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})

    @property
    def reranking_model(self) -> str:
        recipe = hashlib.sha256(
            f"{_RECIPE}{_SYSTEM}{_INSTRUCTION}:{self._image_max_pixels}:{self._video_frames}".encode()
        ).hexdigest()[:16]
        return f"{self._model_id}@{self._revision}:{recipe}:max_length={self._max_length}"

    def rerank(self, query: ModelInput, documents: Sequence[ModelInput]) -> tuple[float, ...]:
        batch = reranking_inputs(query, documents, self.reranking_capabilities)
        with self._lock:
            if self._engine is None:
                raise ModelError("reranking backend is closed", stage="reranking")
            if not batch:
                return ()
            engine = self._engine
            self._active_calls += 1
        try:
            try:
                with self._score_lock:
                    if not self._native_engine:
                        mark_model_requests(1, token_usage_expected=0)
                    values = engine.score(query, batch, max_length=self._max_length)
            except ModelError as error:
                raise ModelError(
                    "reranking model failed", reason=error.reason, stage="reranking"
                ) from None
            except ValueError as error:
                # vLLM exposes expanded context overflow as an unstructured ValueError.
                reason = (
                    "payload_too_large"
                    if re.match(
                        r"The decoder prompt \(length [0-9]+\) is longer than the maximum model "
                        r"length of [0-9]+\.",
                        str(error),
                    )
                    else "model_failed"
                )
                raise ModelError(
                    "reranking model failed", reason=reason, stage="reranking"
                ) from None
            except Exception:
                raise ModelError(
                    "reranking model failed", reason="model_failed", stage="reranking"
                ) from None
            scores = reranking_scores(values, len(batch))
            if any(not 0 <= value <= 1 for value in scores):
                raise ModelError(
                    "reranking model returned invalid probabilities",
                    reason="response_invalid",
                    stage="reranking",
                )
            if not self._native_engine:
                record_unmetered_model_usage()
            return scores
        finally:
            with self._lock:
                self._active_calls -= 1
                self._lock.notify_all()

    def close(self) -> None:
        with self._lock:
            self._lock.wait_for(lambda: not self._closing)
            engine = self._engine
            if engine is None:
                return
            self._closing = True
            self._engine = None
            self._lock.notify_all()
            self._lock.wait_for(lambda: self._active_calls == 0)
            try:
                engine.close()
            except Exception:
                raise ModelError("failed to close reranking model", stage="reranking") from None
            finally:
                self._closing = False
                self._lock.notify_all()
