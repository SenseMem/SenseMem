"""Pure source-interval records and SDK delivery projection for benchmark evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from sensemem import ContextExcerpt, ContextProjection, SearchHit


@dataclass(frozen=True, slots=True)
class EvidenceInterval:
    """One retrieved benchmark memory and its original source interval."""

    memory_id: str
    source_id: str | None
    start_seconds: float | None
    end_seconds: float | None

    def __post_init__(self) -> None:
        if not self.memory_id:
            raise ValueError("evidence memory_id must not be empty")
        if self.source_id is not None and not self.source_id:
            raise ValueError("evidence source_id must not be empty")
        for value in (self.start_seconds, self.end_seconds):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise ValueError("evidence interval values must be finite and non-negative")
        if self.end_seconds is not None and (
            self.start_seconds is None or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("evidence end_seconds must be later than start_seconds")

    def json(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "source_id": self.source_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


def evidence(hit: SearchHit) -> EvidenceInterval:
    source_id = hit.metadata.get("source_id")
    start = hit.metadata.get("start_seconds")
    end = hit.metadata.get("end_seconds")
    return EvidenceInterval(
        hit.id,
        source_id if isinstance(source_id, str) else None,
        _optional_seconds(start),
        _optional_seconds(end),
    )


def limited_source_evidence(source: ContextExcerpt | ContextProjection) -> EvidenceInterval:
    """Record limited delivery without certifying original media, complete spans, or support."""
    context = source.context
    return EvidenceInterval(
        source.source_memory_id,
        None if context is None else context.source_id,
        None,
        None,
    )


def _optional_seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None
