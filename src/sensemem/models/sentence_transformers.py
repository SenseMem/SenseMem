"""Dependency-light Sentence Transformers embedding and cross-encoder adapters."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from importlib import import_module
from typing import Protocol, cast

from sensemem._telemetry import mark_model_requests, record_unmetered_model_usage
from sensemem._validation import commit_revision, nonempty_text, optional_text, positive_int
from sensemem.exceptions import ModelError, ValidationError
from sensemem.models._media import fetch_videos
from sensemem.models.base import EmbedTask, ModelInput
from sensemem.models.reranking import reranking_inputs, reranking_scores
from sensemem.types import EmbeddingRoute, Modality

_ATOMIC_MODALITIES = (Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)
_STANDARD_RECIPE = "sentence-transformers-standard-input-v1"
_EMBEDDINGGEMMA2_MODEL = "google/embeddinggemma-2"
_EMBEDDINGGEMMA2_DIMENSIONS = (128, 256, 512, 768)
_EMBEDDINGGEMMA2_RECIPE = "embeddinggemma2-text-prompts-pyav-v1"


class _EmbeddingMatrix(Protocol):
    def tolist(self) -> object: ...


class _SentenceEncoder(Protocol):
    def supports(self, modality: str | tuple[str, ...]) -> bool: ...

    def get_embedding_dimension(self) -> int | None: ...

    def encode_query(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _EmbeddingMatrix: ...

    def encode_document(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> _EmbeddingMatrix: ...


class _SentenceTransformerFactory(Protocol):
    def __call__(self, model_name_or_path: str, **kwargs: object) -> _SentenceEncoder: ...


class SentenceTransformersEmbedder:
    """Query/document encoder with standard ST inputs and pinned model adaptations."""

    def __init__(
        self,
        encoder: _SentenceEncoder | None = None,
        *,
        model_id: str,
        revision: str | None = None,
        dimension: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model_id = nonempty_text(model_id, "model_id")
        self._device = optional_text(device, "device")
        self._batch_size = positive_int(batch_size, "batch_size")
        if dimension is not None:
            positive_int(dimension, "dimension")
        self._revision = _resolve_revision(self._model_id, revision, encoder)
        self._lock = threading.RLock()
        self._closed = False
        recipe = _STANDARD_RECIPE
        if encoder is None:
            encoder = _load_encoder(self._model_id, revision=self._revision, device=self._device)
        self._encoder = encoder
        self._dimension, self._truncate_dim = _dimensions(encoder, dimension, self._model_id)
        encode = getattr(encoder, "encode", None)
        if callable(encode):
            object.__setattr__(encoder, "encode", encode)
        self._encode_query = encoder.encode_query
        self._encode_document = encoder.encode_document
        self._capabilities = self._discover_capabilities()
        self._message_inputs = self._supports("message")
        combinations = EmbeddingRoute.combinations(
            self._capabilities, max_assets_per_modality=None if self._message_inputs else 1
        )
        self._compound_inputs = frozenset(
            route.modalities
            for route in combinations
            if len(route.modalities) > 1
            and self._supports(tuple(sorted(value.value for value in route.modalities)))
        )
        self._embedding_routes = tuple(
            route
            for route in combinations
            if len(route.modalities) == 1
            or self._message_inputs
            or route.modalities in self._compound_inputs
        )
        if self._model_id == _EMBEDDINGGEMMA2_MODEL:
            recipe = _EMBEDDINGGEMMA2_RECIPE
        self._space_id = _recipe_space(recipe, self._model_id, self._revision, self._dimension)

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        dimension: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> SentenceTransformersEmbedder:
        """Resolve and load an immutable model revision."""
        return cls(
            model_id=model_id,
            revision=revision,
            dimension=dimension,
            device=device,
            batch_size=batch_size,
        )

    @property
    def embedding_capabilities(self) -> frozenset[Modality]:
        return self._capabilities

    @property
    def embedding_model(self) -> str:
        return self._model_id

    @property
    def embedding_space(self) -> str:
        return self._space_id

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    @property
    def embedding_routes(self) -> tuple[EmbeddingRoute, ...]:
        return self._embedding_routes

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode a batch through the retrieval-side method selected by ``task``."""
        if isinstance(inputs, (str, bytes)):
            raise ValidationError("inputs must be a sequence of ModelInput values")
        try:
            batch = tuple(inputs)
        except TypeError:
            raise ValidationError("inputs must be a sequence of ModelInput values") from None
        if any(not isinstance(value, ModelInput) for value in batch):
            raise ValidationError("inputs must be a sequence of ModelInput values")
        try:
            selected_task = EmbedTask(task)
        except (TypeError, ValueError):
            raise ValidationError("embedding task is invalid") from None

        encoding_kwargs: dict[str, object] = {}
        if self._model_id == _EMBEDDINGGEMMA2_MODEL:
            prompts = getattr(self._encoder, "prompts", None)
            name = "query" if selected_task is EmbedTask.QUERY else "document"
            if not isinstance(prompts, Mapping) or not isinstance(prefix := prompts.get(name), str):
                raise ModelError("EmbeddingGemma 2 does not declare its retrieval prompts")
            # ST applies batch prompts to media too; this model prefixes text only.
            batch = tuple(
                ModelInput(text=prefix + value.text if value.text else "", assets=value.assets)
                for value in batch
            )
            encoding_kwargs["prompt"] = ""
        prepared = [self._prepare(value) for value in batch]
        mark_model_requests(1 if batch else 0, token_usage_expected=0)
        with self._lock:
            if self._closed:
                raise ModelError("embedding backend is closed")
            if not batch:
                return ()
            encode = cast(
                Callable[..., _EmbeddingMatrix],
                self._encode_query if selected_task is EmbedTask.QUERY else self._encode_document,
            )
            try:
                matrix = encode(
                    prepared,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    truncate_dim=self._truncate_dim,
                    batch_size=self._batch_size,
                    **encoding_kwargs,
                )
            except Exception:
                raise ModelError("embedding model failed") from None
        vectors = _vectors(matrix, len(batch), self._dimension)
        record_unmetered_model_usage()
        return vectors

    def close(self) -> None:
        """Stop new calls and release an encoder-owned resource when it exposes one."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._encoder, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    raise ModelError("failed to close embedding model") from None

    def _discover_capabilities(self) -> frozenset[Modality]:
        capabilities = frozenset(
            modality for modality in _ATOMIC_MODALITIES if self._supports(modality.value)
        )
        if not capabilities:
            raise ModelError("embedding model does not advertise a supported modality")
        return capabilities

    def _prepare(self, value: ModelInput) -> object:
        parts: list[tuple[str, str]] = []
        if value.text:
            parts.append((Modality.TEXT.value, value.text))
        # ModelInput owns the resolved-asset check.
        parts.extend(
            (cast(Modality, asset.modality).value, str(asset.path)) for asset in value.assets
        )
        modalities = tuple(modality for modality, _ in parts)
        for modality in set(modalities):
            if Modality(modality) not in self._capabilities:
                raise ModelError(f"embedding model does not support {modality} input")
        if len(parts) == 1:
            modality, part = parts[0]
            return part if modality == Modality.TEXT.value else {modality: part}

        compound = tuple(sorted(set(modalities)))
        if len(compound) == len(modalities) and value.modalities in self._compound_inputs:
            return dict(parts)
        if not self._message_inputs:
            raise ModelError("embedding model cannot combine these input parts")
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": modality,
                        "text" if modality == Modality.TEXT.value else modality: part,
                    }
                    for modality, part in parts
                ],
            }
        ]

    def _supports(self, modality: str | tuple[str, ...]) -> bool:
        supported = self._encoder.supports(modality)
        if not isinstance(supported, bool):
            raise ModelError(
                "embedding capability probe must return a boolean", reason="response_invalid"
            )
        return supported


def _resolve_revision(model_id: str, revision: str | None, encoder: _SentenceEncoder | None) -> str:
    if revision is not None:
        return commit_revision(revision)
    if encoder is not None:
        return commit_revision(_config_value(encoder, "_commit_hash"))
    try:
        hub = import_module("huggingface_hub")
    except ImportError:
        raise ModelError(
            "Hugging Face Hub is unavailable; install SenseMem with the local extra"
        ) from None
    try:
        return commit_revision(hub.model_info(model_id).sha)
    except Exception:
        raise ModelError("failed to resolve an immutable embedding model revision") from None


def _load_encoder(
    model_id: str,
    *,
    revision: str,
    device: str | None,
) -> _SentenceEncoder:
    try:
        module = import_module("sentence_transformers")
        factory = cast(_SentenceTransformerFactory, module.SentenceTransformer)
    except (AttributeError, ImportError):
        raise ModelError(
            "Sentence Transformers is unavailable; install SenseMem with the local extra"
        ) from None

    try:
        # FP16 silently corrupts EmbeddingGemma 2 activations. FP32 also works on CPUs.
        model_kwargs = {"dtype": "float32"} if model_id == _EMBEDDINGGEMMA2_MODEL else {}
        encoder = factory(
            model_id,
            revision=revision,
            trust_remote_code=False,
            device=device,
            **({"model_kwargs": model_kwargs} if model_kwargs else {}),
        )
        if model_id == _EMBEDDINGGEMMA2_MODEL:
            processor = getattr(_first_module(encoder), "processor", None)
            video_processor = getattr(processor, "video_processor", None)
            if video_processor is not None:
                # Transformers defaults to torchcodec; PyAV is already in the local extra.
                loader = import_module("transformers.video_utils").load_video
                video_processor.fetch_videos = partial(fetch_videos, loader)
        return encoder
    except Exception:
        raise ModelError("failed to load the embedding model") from None


def _dimensions(
    encoder: _SentenceEncoder, requested: int | None, model_id: str
) -> tuple[int, int | None]:
    try:
        native = encoder.get_embedding_dimension()
    except Exception:
        raise ModelError("embedding model does not declare its native dimension") from None
    if isinstance(native, bool) or not isinstance(native, int) or native <= 0:
        raise ModelError("embedding model does not declare its native dimension")
    if requested is None:
        return native, None
    if requested == native:
        return requested, None
    advertised = _config_value(encoder, "matryoshka_dimensions")
    matryoshka = _config_value(encoder, "is_matryoshka")
    # The upstream checkpoint documents MRL but omits these optional config fields.
    if (
        model_id == _EMBEDDINGGEMMA2_MODEL
        and _config_value(encoder, "model_type") == "embedding_gemma2"
        and native == 768
    ):
        advertised = _EMBEDDINGGEMMA2_DIMENSIONS
        matryoshka = True
    if (
        requested > native
        or matryoshka is not True
        or not isinstance(advertised, (list, tuple))
        or any(isinstance(value, bool) or not isinstance(value, int) for value in advertised)
        or requested not in advertised
    ):
        raise ModelError(
            f"model native dimension is {native}; requested dimension {requested} is not an "
            "advertised Matryoshka dimension"
        )
    return requested, requested


def _config_value(encoder: object, name: str) -> object:
    first = _first_module(encoder)
    candidates = (
        getattr(encoder, "config", None),
        getattr(first, "config", None),
        getattr(getattr(first, "auto_model", None), "config", None),
        getattr(getattr(first, "model", None), "config", None),
    )
    for config in candidates:
        for source in (config, getattr(config, "text_config", None)):
            if source is not None and hasattr(source, name):
                return getattr(source, name)
    return None


def _first_module(encoder: object) -> object | None:
    try:
        return cast(object, encoder[0])  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return None


def _vectors(
    matrix: _EmbeddingMatrix,
    expected_count: int,
    dimension: int,
) -> tuple[tuple[float, ...], ...]:
    try:
        rows = matrix.tolist()
    except Exception:
        raise ModelError("embedding model returned an invalid response") from None
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ModelError("embedding model returned an invalid response")
    vectors = tuple(_vector(row, dimension) for row in rows)
    return vectors


def _vector(row: object, dimension: int) -> tuple[float, ...]:
    if isinstance(row, (str, bytes)):
        raise ModelError("embedding model returned an invalid response")
    if not isinstance(row, Iterable):
        raise ModelError("embedding model returned an invalid response") from None
    raw: tuple[object, ...] = tuple(cast(Iterable[object], row))
    if len(raw) != dimension or any(
        isinstance(value, bool) or not isinstance(value, (float, int)) for value in raw
    ):
        raise ModelError("embedding model returned an invalid response")
    try:
        values = tuple(float(cast(float | int, value)) for value in raw)
    except OverflowError:
        raise ModelError("embedding model returned an invalid response") from None
    if any(not math.isfinite(value) for value in values):
        raise ModelError("embedding model returned an invalid response")
    return values


def _recipe_space(recipe: str, model_id: str, revision: str, dimension: int) -> str:
    payload = json.dumps(
        {
            "dimension": dimension,
            "document": "encode_document",
            "model": model_id,
            "normalize": True,
            "query": "encode_query",
            "recipe": recipe,
            "revision": revision,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"sentence-transformers:{digest}"


class _ScoreArray(Protocol):
    def tolist(self) -> object: ...


class _CrossEncoder(Protocol):
    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        activation_fn: Callable[[object], object],
        apply_softmax: bool,
    ) -> _ScoreArray: ...


class _CrossEncoderFactory(Protocol):
    def __call__(self, model_name_or_path: str, **kwargs: object) -> _CrossEncoder: ...


class SentenceTransformersReranker:
    """Score text pairs in input order; raw logits are not answer probabilities.

    The model owns tokenization and ``max_length`` truncation. Documents retain their
    complete text at this boundary; source identity and result ordering stay in the kernel.
    """

    def __init__(
        self,
        encoder: _CrossEncoder | None = None,
        *,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 1024,
    ) -> None:
        self._model_id = nonempty_text(model_id, "model_id")
        self._device = optional_text(device, "device")
        self._batch_size = positive_int(batch_size, "batch_size")
        self._max_length = positive_int(max_length, "max_length")
        if revision is None:
            try:
                revision = import_module("huggingface_hub").model_info(self._model_id).sha
            except Exception:
                raise ModelError(
                    "failed to resolve an immutable reranking model revision"
                ) from None
        self._revision = commit_revision(revision)
        if encoder is None:
            try:
                factory = cast(
                    _CrossEncoderFactory, import_module("sentence_transformers").CrossEncoder
                )
            except (AttributeError, ImportError):
                raise ModelError(
                    "Sentence Transformers is unavailable; install SenseMem with the local extra"
                ) from None
            try:
                encoder = factory(
                    self._model_id,
                    revision=self._revision,
                    device=self._device,
                    max_length=self._max_length,
                    trust_remote_code=False,
                )
            except Exception:
                raise ModelError("failed to load the reranking model") from None
        self._encoder = encoder
        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 1024,
    ) -> SentenceTransformersReranker:
        """Resolve an immutable revision and load the optional local model."""
        return cls(
            model_id=model_id,
            revision=revision,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

    @property
    def reranking_capabilities(self) -> frozenset[Modality]:
        return frozenset({Modality.TEXT})

    @property
    def reranking_model(self) -> str:
        return f"{self._model_id}@{self._revision}"

    def rerank(self, query: ModelInput, documents: Sequence[ModelInput]) -> tuple[float, ...]:
        """Return one finite raw relevance score for each supplied document."""
        batch = reranking_inputs(query, documents, self.reranking_capabilities)
        with self._lock:
            if self._closed:
                raise ModelError("reranking backend is closed", stage="reranking")
            if not batch:
                return ()
            mark_model_requests(1, token_usage_expected=0)
            try:
                result = self._encoder.predict(
                    [(query.text, document.text) for document in batch],
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    activation_fn=lambda value: value,
                    apply_softmax=False,
                ).tolist()
            except Exception:
                raise ModelError(
                    "reranking model failed", reason="model_failed", stage="reranking"
                ) from None
        scores = reranking_scores(result, len(batch))
        record_unmetered_model_usage()
        return scores

    def close(self) -> None:
        """Stop inference and release an injected encoder resource when present."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._encoder, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    raise ModelError("failed to close reranking model", stage="reranking") from None
