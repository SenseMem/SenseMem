"""Optional, pinned Sentence Transformers query-document cross-encoder."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol, cast

from mindbridge._telemetry import mark_model_requests, record_unmetered_model_usage
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.sentence_transformers import (
    _optional_text,
    _positive_integer,
    _revision,
    _text,
)


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
        self._model_id = _text(model_id, "model_id")
        self._device = _optional_text(device, "device")
        self._batch_size = _positive_integer(batch_size, "batch_size")
        self._max_length = _positive_integer(max_length, "max_length")
        if revision is None:
            try:
                revision = import_module("huggingface_hub").model_info(self._model_id).sha
            except Exception:
                raise ModelError(
                    "failed to resolve an immutable reranking model revision"
                ) from None
        self._revision = _revision(revision)
        if encoder is None:
            try:
                factory = cast(
                    _CrossEncoderFactory, import_module("sentence_transformers").CrossEncoder
                )
            except (AttributeError, ImportError):
                raise ModelError(
                    "Sentence Transformers is unavailable; install MindBridge with the local extra"
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
    def reranking_model(self) -> str:
        return f"{self._model_id}@{self._revision}"

    def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        """Return one finite raw relevance score for each supplied document."""
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
            if self._closed:
                raise ModelError("reranking backend is closed", stage="reranking")
            if not batch:
                return ()
            mark_model_requests(1, token_usage_expected=0)
            try:
                result = self._encoder.predict(
                    [(query, document) for document in batch],
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


def reranking_scores(values: object, expected_count: int) -> tuple[float, ...]:
    """Validate the model boundary before scores can change a source's position."""
    if not isinstance(values, (list, tuple)) or len(values) != expected_count:
        raise ModelError(
            "reranking model returned the wrong number of scores",
            reason="response_invalid",
            stage="reranking",
        )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ModelError(
            "reranking model returned invalid scores", reason="response_invalid", stage="reranking"
        )
    try:
        scores = tuple(float(value) for value in values)
    except OverflowError:
        raise ModelError(
            "reranking model returned invalid scores", reason="response_invalid", stage="reranking"
        ) from None
    if any(not math.isfinite(value) for value in scores):
        raise ModelError(
            "reranking model returned non-finite scores",
            reason="response_invalid",
            stage="reranking",
        )
    return scores
