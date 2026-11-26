"""Records values and pure operations for the public SDK."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

from sensemem._validation import encode_page_cursor, positive_int
from sensemem._value_validation import (
    _metadata,
    _require_aware,
    _require_interval,
    _text,
    _trace_number,
)
from sensemem.content import AssetRef, ContentPart, _assets
from sensemem.exceptions import ValidationError
from sensemem.semantics import RecordSemantics
from sensemem.types import AbstentionReason, MemoryCategory, Modality


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One stored memory.

    content is the caller's text followed by any text the configured models derived from the
    media. Derived sections are appended, never substituted: what the caller supplied stays
    byte-identical at the front, and each derived section is introduced by its own marker line --
    [transcript:<asset_id>], [visual description:<asset_id>], [facts:<asset_id>], or
    [speech identities:<asset_id>], or [model interpretation: not original evidence] -- so a
    reader can tell interpretation from evidence and which asset it came from when applicable.
    add derives before its first write and settle derives after
    capture already committed, so the same record shape results either way; the raw media is
    never rewritten and stays in assets under its content address.
    """

    id: str
    content: str
    created_at: datetime
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)
    assets: tuple[AssetRef, ...] = ()
    modality: Modality = Modality.TEXT
    memory_type: MemoryCategory = MemoryCategory.SEMANTIC
    context: RecordSemantics | None = None
    # The symbolic place this was captured in, as supplied on ObservationContext. Read-back
    # matters because the label is equality-matched: an application cannot notice it wrote
    # "the kitchen" where "kitchen" was meant unless it can see what was stored.
    place_id: str | None = None
    forgotten_at: datetime | None = None
    parts: tuple[ContentPart, ...] = ()

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _require_aware(self.created_at, "created_at")
        _require_interval(self.occurred_at, self.occurred_end)
        _require_aware(self.forgotten_at, "forgotten_at")
        if self.place_id is not None:
            place = self.place_id
            if not isinstance(place, str) or not place.strip() or place != place.strip():
                raise ValidationError("place_id must be non-empty and trimmed")
        if not isinstance(self.modality, Modality):
            raise ValidationError("memory modality is invalid")
        if not isinstance(self.memory_type, MemoryCategory):
            raise ValidationError("memory_type is invalid")
        assets = _assets(self.assets, self.modality)
        if not isinstance(self.content, str) or (not self.content.strip() and not assets):
            raise ValidationError("memory must contain content or media assets")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "assets", assets)
        if any(not isinstance(part, ContentPart) for part in self.parts):
            raise ValidationError("parts must contain ContentPart values")
        if self.context is not None and not isinstance(self.context, RecordSemantics):
            raise ValidationError("memory context is invalid")


@dataclass(frozen=True, slots=True)
class VisualRetrievalMatch:
    """An image-only retrieval association, not a verified identity or factual claim."""

    query_asset_id: str
    memory_id: str
    source_asset_id: str
    rank: int
    cosine_similarity: float

    def __post_init__(self) -> None:
        _text(self.query_asset_id, "query asset id")
        _text(self.memory_id, "memory id")
        _text(self.source_asset_id, "source asset id")
        positive_int(self.rank, "visual retrieval rank")
        value = self.cosine_similarity
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
            raise ValidationError("visual retrieval cosine must be between minus one and one")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked memory, flattened for direct use."""

    id: str
    content: str
    score: float
    created_at: datetime
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)
    assets: tuple[AssetRef, ...] = ()
    modality: Modality = Modality.TEXT
    memory_type: MemoryCategory = MemoryCategory.SEMANTIC
    context: RecordSemantics | None = None
    # A hit is a memory, so a place-scoped search reports which place each came from.
    place_id: str | None = None
    forgotten_at: datetime | None = None
    parts: tuple[ContentPart, ...] = ()

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _require_aware(self.created_at, "created_at")
        _require_interval(self.occurred_at, self.occurred_end)
        _require_aware(self.forgotten_at, "forgotten_at")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValidationError("score must be between zero and one")
        if not isinstance(self.modality, Modality):
            raise ValidationError("memory modality is invalid")
        if not isinstance(self.memory_type, MemoryCategory):
            raise ValidationError("memory_type is invalid")
        assets = _assets(self.assets, self.modality)
        if not isinstance(self.content, str) or (not self.content.strip() and not assets):
            raise ValidationError("memory must contain content or media assets")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "assets", assets)
        if any(not isinstance(part, ContentPart) for part in self.parts):
            raise ValidationError("parts must contain ContentPart values")
        if self.context is not None and not isinstance(self.context, RecordSemantics):
            raise ValidationError("memory context is invalid")


@dataclass(frozen=True, slots=True)
class PendingCapture:
    """One record whose deferred work is not finished, and why it is still waiting.

    Memory.pending_captures returns these. awaiting is the stage the record is stopped at:
    "enrichment" has no vectors and is invisible to search, while "formation" is already
    embedded, indexed, and searchable and owes only the formation add holds a row for between
    its commit and its model call. A memory ID that is absent from the result is not pending: it
    is either settled or unknown, and get distinguishes the two. attempts and last_error are
    the failure state a poisoned record accumulates, so an operator can see the reason without
    opening SQLite.
    """

    memory_id: str
    enqueued_at: datetime
    attempts: int = 0
    last_error: str | None = None
    awaiting: Literal["enrichment", "formation"] = "enrichment"

    @property
    def cursor(self) -> str:
        return encode_page_cursor(self.enqueued_at, self.memory_id)

    def __post_init__(self) -> None:
        _text(self.memory_id, "memory_id")
        _require_aware(self.enqueued_at, "enqueued_at")
        if self.enqueued_at is None:
            raise ValidationError("enqueued_at must include a timezone")
        if self.awaiting not in {"enrichment", "formation"}:
            raise ValidationError("awaiting must be enrichment or formation")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise ValidationError("attempts must be an integer")
        if self.attempts < 0:
            raise ValidationError("attempts must not be negative")
        if self.last_error is not None and not isinstance(self.last_error, str):
            raise ValidationError("last_error must be text")


class RetrievalRejection(str, Enum):
    """Why one bounded retrieval candidate did not become a search hit."""

    STALE_INDEX = "stale_index"
    OCCURRENCE_RANGE = "occurrence_range"
    MISSING_MEMORY = "missing_memory"
    MEMORY_TYPE = "memory_type"
    MINIMUM_RELEVANCE = "minimum_relevance"
    AMBIGUITY = "ambiguity"
    LIMIT = "limit"


@dataclass(frozen=True, slots=True)
class RetrievalCandidateTrace:
    """Effective score components and final disposition for one considered parent memory.

    Dense values can come from an unquantized ANN hit or from exact persisted-vector scoring
    for quantized candidates and lexical-only parents. index_ids are the candidates that
    admitted the parent; they neither identify the admitting route nor list every persisted
    part used by exact scoring. reranking_score is an optional raw query-document logit;
    it can change rank without changing the bounded final_score or relevance gate.
    """

    memory_id: str | None
    index_ids: tuple[str, ...]
    dense_relevance: float | None = None
    dense_confidence: float | None = None
    lexical_relevance: float | None = None
    lexical_rerank_bonus: float | None = None
    lexical_match: bool = False
    gate_relevance: float | None = None
    base_relevance: float | None = None
    reinforcement_factor: float | None = None
    temporal_factor: float | None = None
    retention_factor: float | None = None
    final_score: float | None = None
    reranking_score: float | None = None
    rank: int | None = None
    rejected_by: RetrievalRejection | None = None

    def __post_init__(self) -> None:
        if self.memory_id is not None:
            _text(self.memory_id, "trace memory_id")
        index_ids = tuple(self.index_ids)
        if not index_ids or len(set(index_ids)) != len(index_ids):
            raise ValidationError("trace index_ids must be non-empty and unique")
        for index_id in index_ids:
            _text(index_id, "trace index_id")
        for name in (
            "dense_relevance",
            "dense_confidence",
            "lexical_relevance",
            "lexical_rerank_bonus",
            "gate_relevance",
            "base_relevance",
            "reinforcement_factor",
            "temporal_factor",
            "retention_factor",
            "final_score",
        ):
            _trace_number(getattr(self, name), name)
        for name in (
            "dense_relevance",
            "dense_confidence",
            "lexical_relevance",
            "lexical_rerank_bonus",
            "gate_relevance",
            "base_relevance",
            "final_score",
        ):
            _trace_number(getattr(self, name), name, maximum=1.0)
        if self.reranking_score is not None and (
            isinstance(self.reranking_score, bool)
            or not isinstance(self.reranking_score, int | float)
            or not math.isfinite(self.reranking_score)
        ):
            raise ValidationError("trace reranking_score must be a finite number")
        if not isinstance(self.lexical_match, bool):
            raise ValidationError("trace lexical_match must be a boolean")
        if self.rank is not None and (
            isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0
        ):
            raise ValidationError("trace rank must be a positive integer")
        if self.rejected_by is not None and not isinstance(self.rejected_by, RetrievalRejection):
            raise ValidationError("trace rejected_by is invalid")
        object.__setattr__(self, "index_ids", index_ids)


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Bounded candidate trace for one explicit traced search."""

    candidates: tuple[RetrievalCandidateTrace, ...]
    candidate_limit: int
    exhaustive: bool
    ambiguous: bool = False

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, RetrievalCandidateTrace) for candidate in candidates):
            raise ValidationError("trace candidates are invalid")
        if (
            isinstance(self.candidate_limit, bool)
            or not isinstance(self.candidate_limit, int)
            or self.candidate_limit <= 0
        ):
            raise ValidationError("trace candidate_limit must be a positive integer")
        if not isinstance(self.exhaustive, bool) or not isinstance(self.ambiguous, bool):
            raise ValidationError("trace exhaustive and ambiguous must be booleans")
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Ranked search hits with an optional diagnostic trace."""

    hits: tuple[SearchHit, ...]
    trace: RetrievalTrace | None = None

    def __post_init__(self) -> None:
        hits = tuple(self.hits)
        if any(not isinstance(hit, SearchHit) for hit in hits):
            raise ValidationError("search hits are invalid")
        if self.trace is not None and not isinstance(self.trace, RetrievalTrace):
            raise ValidationError("search trace is invalid")
        object.__setattr__(self, "hits", hits)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """A grounded answer and the memories used to produce it."""

    answer: str
    hits: tuple[SearchHit, ...] = ()
    abstained: bool = False
    abstention_reason: AbstentionReason | None = None

    def __post_init__(self) -> None:
        _text(self.answer, "answer")
        try:
            hits = tuple(self.hits)
        except TypeError:
            raise ValidationError("answer hits must contain SearchHit values") from None
        if any(not isinstance(hit, SearchHit) for hit in hits):
            raise ValidationError("answer hits must contain SearchHit values")
        object.__setattr__(self, "hits", hits)
        if not isinstance(self.abstained, bool):
            raise ValidationError("abstained must be a boolean")
        if self.abstention_reason is not None and not isinstance(
            self.abstention_reason, AbstentionReason
        ):
            raise ValidationError("abstention_reason is invalid")
        if self.abstained != (self.abstention_reason is not None):
            raise ValidationError("abstained and abstention_reason must agree")


@dataclass(frozen=True, slots=True)
class AnswerChunk:
    """One streamed answer delta, or the terminal chunk carrying the grounded result.

    Exactly one field is populated. A delta carries text and no result; the single terminal
    chunk carries result and no text. "".join(chunk.text for chunk in stream) therefore
    reconstructs the answer, and a caller that only wants the grounded outcome can wait for the
    chunk whose result is set.
    """

    text: str = ""
    result: AnswerResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("answer chunk text must be a string")
        if self.result is not None and not isinstance(self.result, AnswerResult):
            raise ValidationError("answer chunk result is invalid")
        if bool(self.text) == (self.result is not None):
            raise ValidationError("an answer chunk carries either text or a result")


@dataclass(frozen=True, slots=True)
class Page:
    """One stable page of memories."""

    items: tuple[MemoryRecord, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None:
            _text(self.next_cursor, "next_cursor")
