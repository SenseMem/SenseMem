"""Span helpers shared by every kernel plane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager

from opentelemetry.trace import Span, Tracer
from opentelemetry.util.types import AttributeValue

from sensemem._telemetry import (
    IDENTITY_CACHED,
    IDENTITY_CREATED,
    IDENTITY_IDENTITIES,
    IDENTITY_MATCHED,
    IDENTITY_OBSERVATIONS,
    MODEL_MODULE,
    SPAN_KIND,
    model_span,
    operation_span,
    traced_span,
)
from sensemem.exceptions import SenseMemError
from sensemem.types import Modality

# Model span modules mapped onto the failure stage a caller sees.
_MODEL_STAGES = {
    "consolidation": "consolidate",
    "event_segmentation": "segment",
    "embedding": "embed",
    "face": "recognize",
    "formation": "form",
    "generation": "generate",
    "reranking": "reranking",
    "transcription": "transcribe",
    "vision": "describe",
}


@contextmanager
def _staged(span: AbstractContextManager[Span], stage: str) -> Iterator[Span]:
    """Name the failing stage on public errors so an agent never parses prose to find it.

    The innermost span wins: an adapter that already classified its own failure keeps its stage.
    """
    with span as opened:
        try:
            yield opened
        except SenseMemError as error:
            if error.stage is None:
                error.stage = stage
            raise


class Traced:
    """Base for every plane: one tracer, the same span helpers."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def _trace(
        self,
        name: str,
        *,
        kind: str,
        attributes: Mapping[str, AttributeValue] | None = None,
        failure_stage: str | None = None,
    ) -> AbstractContextManager[Span]:
        values = dict(attributes or {})
        values[SPAN_KIND] = kind
        if kind == "operation":
            return operation_span(self._tracer, name, attributes=values)
        return _staged(
            traced_span(self._tracer, name, attributes=values),
            failure_stage or name.removeprefix("sensemem."),
        )

    def _model_trace(
        self,
        module: str,
        operation: str,
        *,
        model: str | None,
        batch_size: int,
        modalities: Iterable[Modality],
    ) -> AbstractContextManager[Span]:
        attributes: dict[str, AttributeValue] = {
            MODEL_MODULE: module,
            "gen_ai.operation.name": operation,
            "sensemem.model.batch_size": batch_size,
            "sensemem.input.modalities": tuple(sorted({modality.value for modality in modalities})),
        }
        if model is not None:
            attributes["gen_ai.request.model"] = model
        attributes[SPAN_KIND] = "model"
        return _staged(
            model_span(self._tracer, f"sensemem.model.{module}", attributes=attributes),
            _MODEL_STAGES[module],
        )


def record_identity_yield(
    span: Span,
    resolved: Sequence[tuple[str | None, float | None]],
    *,
    cached: bool,
) -> None:
    """Record recognizer results on the speech or vision stage that produced them.

    Observation and match counts include every observation; identity and creation counts
    deduplicate IDs. Emit all five attributes even for an empty result, distinguishing an
    empty detection from an unconfigured recognizer. A missing match score marks a new ID.
    """
    identities = {identity for identity, _ in resolved if identity is not None}
    created = {identity for identity, score in resolved if identity is not None and score is None}
    span.set_attribute(IDENTITY_OBSERVATIONS, len(resolved))
    span.set_attribute(IDENTITY_IDENTITIES, len(identities))
    span.set_attribute(
        IDENTITY_MATCHED,
        sum(1 for _, score in resolved if score is not None),
    )
    span.set_attribute(IDENTITY_CREATED, len(created))
    span.set_attribute(IDENTITY_CACHED, cached)
