"""Pinned native Qwen3-VL yes/no reranking of query and source-text windows."""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

from mindbridge._telemetry import mark_model_requests, record_unmetered_model_usage
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models._media import fetch_videos
from mindbridge.models.base import ModelInput
from mindbridge.models.reranking import reranking_scores
from mindbridge.models.sentence_transformers import (
    _optional_text,
    _positive_integer,
    _revision,
    _text,
)
from mindbridge.types import Modality

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from torch import Tensor
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)
_RECIPE = "qwen3-vl-text-yes-no-float32-sigmoid-v1"
_SUFFIX_TOKENS = 5


@dataclass(frozen=True)
class _MediaSettings:
    max_length: int
    image_max_pixels: int
    video_max_pixels: int
    video_max_frames: int
    video_fps: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.video_fps, bool)
            or not isinstance(self.video_fps, (int, float))
            or not math.isfinite(self.video_fps)
            or self.video_fps <= 0
        ):
            raise ValidationError("video_fps must be positive and finite")
        if self.video_max_frames < 4:
            raise ValidationError("video_max_frames must be at least 4")


class _ScoringEngine(Protocol):
    def score(
        self, query: str, documents: Sequence[str], *, batch_size: int, max_length: int
    ) -> object: ...

    def close(self) -> None: ...


class _MediaScoringEngine(_ScoringEngine, Protocol):
    def score_media(
        self, query: ModelInput, documents: Sequence[ModelInput], settings: _MediaSettings
    ) -> object: ...


def _messages(query: str, document: str) -> list[dict[str, object]]:
    # This is the publisher's text-only conversation, with no media or benchmark-specific prompt.
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<Instruct>: " + _INSTRUCTION},
                {"type": "text", "text": "<Query>:"},
                {"type": "text", "text": query},
                {"type": "text", "text": "\n<Document>:"},
                {"type": "text", "text": document or "NULL"},
            ],
        },
    ]


def _media_parts(
    value: ModelInput, resources: ExitStack
) -> tuple[list[dict[str, object]], list[PILImage], list[str]]:
    """Verify local bytes and collect independently reusable query/document visual parts."""
    from PIL import Image

    parts: list[dict[str, object]] = []
    images: list[PILImage] = []
    videos: list[str] = []
    for asset in value.assets:
        assert asset.path is not None
        with asset.path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != asset.sha256:
            raise ModelError(
                "reranking media integrity mismatch", reason="response_invalid", stage="reranking"
            )
        if asset.modality is Modality.IMAGE:
            opened = resources.enter_context(Image.open(asset.path))
            image = opened.convert("RGB")
            resources.callback(image.close)
            images.append(image)
            parts.append({"type": "image", "image": str(asset.path)})
        elif asset.modality is Modality.VIDEO:
            videos.append(str(asset.path))
            parts.append({"type": "video", "video": str(asset.path)})
        else:
            raise ValidationError("Qwen reranking supports image and video assets")
    return parts, images, videos


def _truncate_tokens(tokens: list[int], max_length: int, special_ids: set[int]) -> list[int]:
    if len(tokens) <= max_length:
        return tokens
    prefix, suffix = tokens[:-_SUFFIX_TOKENS], tokens[-_SUFFIX_TOKENS:]
    # The publisher preserves the five-token generation suffix; it counts inside our hard limit.
    ordinary_budget = max_length - len(suffix) - sum(token in special_ids for token in prefix)
    if ordinary_budget < 0:
        raise ModelError(
            "reranking input exceeds its special-token budget",
            reason="payload_too_large",
            stage="reranking",
        )
    kept = []
    for token in prefix:
        if token in special_ids:
            kept.append(token)
        elif ordinary_budget:
            kept.append(token)
            ordinary_budget -= 1
    return kept + suffix


class _NativeScoringEngine:
    def __init__(self, model_id: str, revision: str, device: str | None) -> None:
        try:
            torch = import_module("torch")
            transformers = import_module("transformers")
            factory = transformers.Qwen3VLForConditionalGeneration
        except (AttributeError, ImportError):
            raise ModelError(
                "native Qwen3-VL is unavailable; install MindBridge with the local extra",
                stage="reranking",
            ) from None
        self._torch = torch
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float32
        if self._device.type == "cuda":
            with torch.cuda.device(self._device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._processor = cast(
            "Qwen3VLProcessor",
            transformers.AutoProcessor.from_pretrained(
                model_id, revision=revision, padding_side="left", trust_remote_code=False
            ),
        )
        vocabulary = self._processor.tokenizer.get_vocab()
        yes, no = vocabulary["yes"], vocabulary["no"]
        lm = cast(
            "Qwen3VLForConditionalGeneration",
            factory.from_pretrained(
                model_id,
                revision=revision,
                dtype=dtype,
                attn_implementation="sdpa",
                trust_remote_code=False,
                use_safetensors=True,
            ),
        )
        # Retain only the native backbone and the trained yes/no rows, not full vocabulary logits.
        self._weight = (lm.lm_head.weight[yes] - lm.lm_head.weight[no]).detach().unsqueeze(0)
        self._model = lm.model
        del lm
        self._model.to(self._device).eval()
        self._weight = self._weight.to(self._device)
        self._video_configured = False

    def score(
        self, query: str, documents: Sequence[str], *, batch_size: int, max_length: int
    ) -> object:
        torch = self._torch
        scores: list[float] = []
        special_ids = set(self._processor.tokenizer.all_special_ids)
        for start in range(0, len(documents), batch_size):
            pairs = [
                _messages(query, document) for document in documents[start : start + batch_size]
            ]
            # The native annotation omits its documented multimodal content-block format.
            text = self._processor.apply_chat_template(
                pairs,  # type: ignore[arg-type]
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self._processor(text=text, truncation=False, padding=False)
            ids = cast(list[list[int]], encoded["input_ids"])
            trimmed = [_truncate_tokens(tokens, max_length, special_ids) for tokens in ids]
            padded = self._processor.tokenizer.pad(
                {"input_ids": trimmed}, padding=True, return_tensors="pt"
            )
            inputs = {
                name: cast("Tensor", value).to(self._device) for name, value in padded.items()
            }
            with torch.inference_mode():
                hidden = self._model(**inputs, use_cache=False).last_hidden_state
                mask = inputs["attention_mask"]
                positions = torch.arange(mask.shape[1], device=self._device).expand_as(mask)
                last = positions.masked_fill(mask == 0, -1).max(dim=1).values
                selected = hidden[torch.arange(mask.shape[0], device=self._device), last]
                logits = torch.nn.functional.linear(selected, self._weight).squeeze(-1)
                scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
        return scores

    def score_media(
        self, query: ModelInput, documents: Sequence[ModelInput], settings: _MediaSettings
    ) -> object:
        """Use native visual tensors and timestamps; never trim expanded visual tokens."""
        from opentelemetry.trace import get_current_span

        torch = self._torch
        video_processor = cast("Qwen3VLVideoProcessor", self._processor.video_processor)
        if not self._video_configured:
            loader = import_module("transformers.video_utils").load_video
            vars(video_processor)["fetch_videos"] = partial(fetch_videos, loader)
            self._video_configured = True
        video_processor.max_frames = settings.video_max_frames
        vars(video_processor)["fps"] = settings.video_fps
        video_processor.size = {
            "shortest_edge": 32 * 32 * 32,
            "longest_edge": settings.video_max_pixels,
        }
        vars(video_processor)["cap_pixels_per_frame"] = True
        video_processor.max_video_tokens = max(
            32, settings.video_max_pixels // (settings.video_max_frames * 32 * 32)
        )
        scores: list[float] = []
        actual = {"image_pairs": 0, "video_pairs": 0, "video_frames": 0, "max_input_tokens": 0}
        for document in documents:
            messages = _messages(query.text, document.text)
            content = cast(list[dict[str, object]], messages[1]["content"])
            # Query attachments follow its text; document attachments follow its text.
            with ExitStack() as resources:
                query_parts, query_images, query_videos = _media_parts(query, resources)
                document_parts, document_images, document_videos = _media_parts(document, resources)
                content[3:3] = query_parts
                content.extend(document_parts)
                text = self._processor.apply_chat_template(
                    [messages],  # type: ignore[arg-type]
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = self._processor(
                    text=text,
                    images=query_images + document_images or None,
                    videos=query_videos + document_videos or None,
                    images_kwargs={
                        "size": {"shortest_edge": 4096, "longest_edge": settings.image_max_pixels}
                    },
                    truncation=False,
                    padding=True,
                    return_tensors="pt",
                )
            ids = cast("Tensor", encoded["input_ids"])
            actual["image_pairs"] += int("pixel_values" in encoded)
            actual["video_pairs"] += int("pixel_values_videos" in encoded)
            if "video_grid_thw" in encoded:
                grid = cast("Tensor", encoded["video_grid_thw"])
                actual["video_frames"] += int(grid[:, 0].sum().item()) * 2
            actual["max_input_tokens"] = max(actual["max_input_tokens"], int(ids.shape[1]))
            if ids.shape[1] > settings.max_length:
                raise ModelError(
                    "reranking media exceeds its token budget",
                    reason="payload_too_large",
                    stage="reranking",
                )
            inputs = {
                name: cast("Tensor", value).to(self._device) for name, value in encoded.items()
            }
            with torch.inference_mode():
                hidden = self._model(**inputs, use_cache=False).last_hidden_state[:, -1]
                logits = torch.nn.functional.linear(hidden, self._weight).squeeze(-1)
                scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
        span = get_current_span()
        for name, count in actual.items():
            span.set_attribute("mindbridge.reranking." + name, count)
        span.set_attribute("mindbridge.reranking.media_forward_count", len(scores))
        return scores

    def close(self) -> None:
        del self._model
        del self._weight
        del self._processor


class Qwen3VLReranker:
    """Use the native trained yes/no head on text windows, retaining caller-owned positions.

    Scores are float32 sigmoid transformations of query-local yes/no log odds. They are neither
    answer confidence nor a replacement for the retrieval relevance gate.
    """

    def __init__(
        self,
        engine: _ScoringEngine | None = None,
        *,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 1,
        max_length: int = 2048,
        multimodal: bool = False,
        media_candidates: int = 40,
        media_weight: float = 0.5,
        media_max_length: int = 4096,
        image_max_pixels: int = 1048576,
        video_max_pixels: int = 2097152,
        video_max_frames: int = 8,
        video_fps: float = 1.0,
    ) -> None:
        self._model_id = _text(model_id, "model_id")
        self._device = _optional_text(device, "device")
        self._batch_size = _positive_integer(batch_size, "batch_size")
        self._max_length = _positive_integer(max_length, "max_length")
        if not isinstance(multimodal, bool):
            raise ValidationError("multimodal must be a boolean")
        self._multimodal = multimodal
        self._media_candidates = _positive_integer(media_candidates, "media_candidates")
        if (
            isinstance(media_weight, bool)
            or not isinstance(media_weight, (int, float))
            or not 0 < media_weight <= 1
        ):
            raise ValidationError("media_weight must be in (0, 1]")
        self._media_weight = float(media_weight)
        self._media_settings = _MediaSettings(
            _positive_integer(media_max_length, "media_max_length"),
            _positive_integer(image_max_pixels, "image_max_pixels"),
            _positive_integer(video_max_pixels, "video_max_pixels"),
            _positive_integer(video_max_frames, "video_max_frames"),
            video_fps,
        )
        if self._max_length <= _SUFFIX_TOKENS:
            raise ValidationError("max_length must exceed the five-token generation suffix")
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
                engine = _NativeScoringEngine(self._model_id, self._revision, self._device)
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
        batch_size: int = 1,
        max_length: int = 2048,
        multimodal: bool = False,
        media_candidates: int = 40,
        media_weight: float = 0.5,
        media_max_length: int = 4096,
        image_max_pixels: int = 1048576,
        video_max_pixels: int = 2097152,
        video_max_frames: int = 8,
        video_fps: float = 1.0,
    ) -> Qwen3VLReranker:
        return cls(
            model_id=model_id,
            revision=revision,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            multimodal=multimodal,
            media_candidates=media_candidates,
            media_weight=media_weight,
            media_max_length=media_max_length,
            image_max_pixels=image_max_pixels,
            video_max_pixels=video_max_pixels,
            video_max_frames=video_max_frames,
            video_fps=video_fps,
        )

    @property
    def reranking_model(self) -> str:
        recipe = hashlib.sha256((_RECIPE + _SYSTEM + _INSTRUCTION).encode()).hexdigest()[:16]
        identity = f"{self._model_id}@{self._revision}:{recipe}:max_length={self._max_length}"
        if self._multimodal:
            settings = hashlib.sha256(
                repr((self._media_settings, self._media_candidates, self._media_weight)).encode()
            ).hexdigest()[:16]
            identity += f":native-image-video-v1:{settings}"
        return identity

    @property
    def reranking_capabilities(self) -> frozenset[Modality]:
        return frozenset(
            {Modality.TEXT, Modality.IMAGE, Modality.VIDEO} if self._multimodal else {Modality.TEXT}
        )

    def rerank_media(self, query: ModelInput, documents: Sequence[ModelInput]) -> tuple[float, ...]:
        """Refine the text shortlist using original pixels and native temporal video inputs."""
        from opentelemetry.trace import get_current_span

        batch = tuple(documents)
        with self._lock:
            scores = list(
                self.rerank(
                    query.text or "Retrieve candidates relevant to the query media.",
                    tuple(document.text or "Media evidence" for document in batch),
                )
            )
            if not self._multimodal:
                if query.assets or any(document.assets for document in batch):
                    raise ValidationError("multimodal reranking is not enabled")
                return tuple(scores)
            eligible = sorted(
                (index for index, document in enumerate(batch) if document.assets or query.assets),
                key=lambda index: (-scores[index], index),
            )[: self._media_candidates]
            if not eligible:
                return tuple(scores)
            selected = tuple(batch[index] for index in eligible)
            span = get_current_span()
            span.set_attribute(
                "mindbridge.input.modalities",
                tuple(
                    sorted(
                        {
                            Modality.TEXT.value,
                            *(
                                asset.modality.value
                                for source in (query, *selected)
                                for asset in source.assets
                                if asset.modality is not None
                            ),
                        }
                    )
                ),
            )
            span.set_attribute("mindbridge.reranking.media_candidates", len(selected))
            for kind in (Modality.IMAGE, Modality.VIDEO):
                span.set_attribute(
                    "mindbridge.reranking." + kind.value + "_exposures",
                    sum(asset.modality is kind for source in selected for asset in source.assets)
                    + len(selected) * sum(asset.modality is kind for asset in query.assets),
                )
            mark_model_requests(len(selected), token_usage_expected=0)
            try:
                values = cast("_MediaScoringEngine", self._engine).score_media(
                    query, selected, self._media_settings
                )
                refined = reranking_scores(values, len(selected))
                if any(not 0 <= value <= 1 for value in refined):
                    raise ModelError("invalid multimodal probability", reason="response_invalid")
            except ModelError as error:
                raise ModelError(
                    "multimodal reranking failed", reason=error.reason, stage="reranking"
                ) from None
            except Exception:
                raise ModelError(
                    "multimodal reranking failed", reason="model_failed", stage="reranking"
                ) from None
            record_unmetered_model_usage(request_count=len(selected))
            for index, value in zip(eligible, refined, strict=True):
                base = min(max(scores[index], 1e-7), 1 - 1e-7)
                media = min(max(value, 1e-7), 1 - 1e-7)
                logit = (1 - self._media_weight) * math.log(
                    base / (1 - base)
                ) + self._media_weight * math.log(media / (1 - media))
                scores[index] = 1 / (1 + math.exp(-logit))
            return tuple(scores)

    def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        query = _text(query, "query")
        if isinstance(documents, (str, bytes)):
            raise ValidationError("documents must be a sequence of text values")
        try:
            batch = tuple(documents)
        except TypeError:
            raise ValidationError("documents must be a sequence of text values") from None
        if any(not isinstance(document, str) for document in batch):
            raise ValidationError("documents must be a sequence of text values")
        with self._lock:
            if self._engine is None:
                raise ModelError("reranking backend is closed", stage="reranking")
            if not batch:
                return ()
            requests = (len(batch) + self._batch_size - 1) // self._batch_size
            mark_model_requests(requests, token_usage_expected=0)
            try:
                values = self._engine.score(
                    query, batch, batch_size=self._batch_size, max_length=self._max_length
                )
            except ModelError as error:
                raise ModelError(
                    "reranking model failed", reason=error.reason, stage="reranking"
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
            record_unmetered_model_usage(request_count=requests)
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
