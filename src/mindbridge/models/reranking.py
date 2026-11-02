"""Pure input and output validation shared by reranking backends and the kernel."""

from __future__ import annotations

import math
from collections.abc import Sequence

from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import ModelInput
from mindbridge.types import Modality


def reranking_inputs(
    query: ModelInput, documents: Sequence[ModelInput], capabilities: frozenset[Modality]
) -> tuple[ModelInput, ...]:
    if not isinstance(query, ModelInput):
        raise ValidationError("reranking query must be a ModelInput")
    if isinstance(documents, (str, bytes)):
        raise ValidationError("documents must be a sequence of ModelInput values")
    try:
        batch = tuple(documents)
    except TypeError:
        raise ValidationError("documents must be a sequence of ModelInput values") from None
    if any(not isinstance(document, ModelInput) for document in batch):
        raise ValidationError("documents must be a sequence of ModelInput values")
    if any(value.modalities - capabilities for value in (query, *batch)):
        raise ValidationError("reranking input contains an unsupported modality")
    return batch


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
