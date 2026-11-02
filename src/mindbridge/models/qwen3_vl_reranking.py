"""Pinned vLLM Qwen3-VL reranking of text and original visual assets."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Sequence
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypedDict, cast

from mindbridge._telemetry import mark_model_requests, record_unmetered_model_usage
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import ModelInput
from mindbridge.models.reranking import _reranking_inputs, reranking_scores
from mindbridge.models.sentence_transformers import (
    _optional_text,
    _positive_integer,
    _revision,
    _text,
)
from mindbridge.types import AssetRef, Modality

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


class _MediaLoader(Protocol):
    def fetch_image(self, image_url: str) -> _Image: ...

    def fetch_video(self, video_url: str, video_io_kwargs: dict[str, object]) -> object: ...


def _prompt(query: str, document: str) -> str:
    # The publisher's reranker_template.jinja; source data are never interpreted as a template.
    return (
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {_INSTRUCTION}<Query>:{query}\n"
        f"<Document>:{document}<|im_end|>\n<|im_start|>assistant\n"
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
    ) -> None:
        try:
            factory = cast(_LLMFactory, import_module("vllm").LLM)
        except (AttributeError, ImportError):
            raise ModelError(
                "vLLM is unavailable; install MindBridge with the reranking extra",
                stage="reranking",
            ) from None
        self._image_limit = image_limit
        self._video_limit = video_limit
        self._video_frames = video_frames
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

    def score(
        self,
        query: ModelInput,
        documents: Sequence[ModelInput],
        *,
        max_length: int,
    ) -> object:
        tokenizer = self._llm.get_tokenizer()
        with ExitStack() as cleanup:
            loaded: dict[AssetRef, object] = {}

            def content(value: ModelInput) -> tuple[str, dict[str, list[object]]]:
                pieces: list[str] = []
                media: dict[str, list[object]] = {}
                for asset in value.assets:
                    kind = cast(Modality, asset.modality).value
                    pieces.append(f"<|vision_start|><|{kind}_pad|><|vision_end|>")
                    if asset not in loaded:
                        loader = cast(_MediaLoader, import_module("vllm.multimodal.utils"))
                        url = cast(Path, asset.path).resolve().as_uri()
                        if asset.modality is Modality.IMAGE:
                            image = loader.fetch_image(url)
                            cleanup.callback(image.close)
                            loaded[asset] = image
                        else:
                            loaded[asset] = loader.fetch_video(
                                url, {"num_frames": self._video_frames, "video_backend": "opencv"}
                            )
                    media.setdefault(kind, []).append(loaded[asset])
                pieces.append(value.text)
                return "".join(pieces), media

            query_text, query_media = content(query)
            prompts: list[_Prompt] = []
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
                document_text, document_media = content(document)
                prompt = _prompt(query_text, document_text)
                if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_length:
                    raise ModelError(
                        "reranking pair exceeds its context budget",
                        reason="payload_too_large",
                        stage="reranking",
                    )
                row = _Prompt(prompt=prompt)
                media = {
                    kind: query_media.get(kind, []) + document_media.get(kind, [])
                    for kind in query_media.keys() | document_media.keys()
                }
                if media:
                    row["multi_modal_data"] = media
                prompts.append(row)
            # vLLM schedules every pair together and checks the expanded visual context too.
            outputs = self._llm.classify(prompts, use_tqdm=False)
            return [reranking_scores(output.outputs.probs, 1)[0] for output in outputs]

    def close(self) -> None:
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
    ) -> None:
        self._model_id = _text(model_id, "model_id")
        device = _optional_text(device, "device")
        if device not in (None, "cuda"):
            raise ValidationError(
                "device must be cuda or None; select GPUs with CUDA_VISIBLE_DEVICES"
            )
        batch_size = _positive_integer(batch_size, "batch_size")
        self._max_length = _positive_integer(max_length, "max_length")
        image_limit = _positive_integer(image_limit, "image_limit")
        video_limit = _positive_integer(video_limit, "video_limit")
        self._image_max_pixels = _positive_integer(image_max_pixels, "image_max_pixels")
        self._video_frames = _positive_integer(video_frames, "video_frames")
        if not 2 <= self._video_frames <= 64:
            raise ValidationError("video_frames must be between 2 and 64")
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
        self._revision = _revision(revision)
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
                )
            except ModelError:
                raise
            except Exception:
                raise ModelError("failed to load the reranking model", stage="reranking") from None
        self._engine: _ScoringEngine | None = engine
        self._lock = threading.RLock()

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
        batch = _reranking_inputs(query, documents, self.reranking_capabilities)
        with self._lock:
            if self._engine is None:
                raise ModelError("reranking backend is closed", stage="reranking")
            if not batch:
                return ()
            mark_model_requests(1, token_usage_expected=0)
            try:
                values = self._engine.score(query, batch, max_length=self._max_length)
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
            record_unmetered_model_usage()
            return scores

    def close(self) -> None:
        with self._lock:
            engine = self._engine
            if engine is None:
                return
            self._engine = None
            try:
                engine.close()
            except Exception:
                raise ModelError("failed to close reranking model", stage="reranking") from None
