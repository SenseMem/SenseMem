"""Replaceable retrieval scoring over authoritative, immutable candidate snapshots."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from sensemem._validation import nonempty_text, unit_interval
from sensemem.content import AssetRef, ContentPart
from sensemem.semantics import RecordSemantics
from sensemem.types import MemoryCategory, Modality, RetrievalMode

LEXICAL_FULL_COVERAGE = 0.999
LEXICAL_FULL_COVERAGE_RELEVANCE = 0.75
MAX_LEXICAL_RERANK_BONUS = 0.3


@dataclass(frozen=True, slots=True)
class RankingMemory:
    """Authoritative record metadata without opening media files or granting storage access.

    Asset references deliberately have no local path. Final selection resolves original media
    through the ordinary hydration boundary; a discarded candidate need not have readable media.
    Raw update and access history lets a policy define its own reinforcement and decay rules.
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
    place_id: str | None = None
    parts: tuple[ContentPart, ...] = ()
    updated_at: datetime | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0


@dataclass(frozen=True, slots=True)
class RankingCandidate:
    """One hydrated record and its query-local retrieval and history signals.

    History factors are available to the policy; observation confidence is applied separately
    by the kernel. The memory contains metadata only, with unresolved media references.
    """

    memory: RankingMemory
    dense_relevance: float
    lexical_relevance: float
    lexical_coverage: float
    reinforcement_factor: float
    temporal_factor: float | None
    retention_factor: float | None


@dataclass(frozen=True, slots=True)
class RankingRequest:
    """One query's complete eligible candidate set, with no storage or mutation handle."""

    query: str
    candidates: tuple[RankingCandidate, ...]
    mode: RetrievalMode
    reference_at: datetime


@dataclass(frozen=True, slots=True)
class RankingScore:
    """A proposed score and relevance for exactly one supplied memory ID.

    Scores precede observation-confidence scaling. Relevance additionally passes the kernel's
    temporal, minimum-relevance and ambiguity checks. Lexical fields explain the policy's
    contribution; they never grant authority to insert or replace a candidate.
    """

    memory_id: str
    score: float
    relevance: float
    lexical_relevance: float = 0.0
    lexical_bonus: float = 0.0

    def __post_init__(self) -> None:
        nonempty_text(self.memory_id, "ranking memory ID")
        unit_interval(self.score, "ranking score")
        unit_interval(self.relevance, "ranking relevance")
        unit_interval(self.lexical_relevance, "ranking lexical relevance")
        unit_interval(self.lexical_bonus, "ranking lexical bonus")


@runtime_checkable
class RankingPolicy(Protocol):
    """Score every supplied candidate exactly once, without performing durable effects.

    Implementations must be thread-safe. Candidate content is untrusted evidence. Memory owns
    the policy and closes each distinct resource once, including resources shared with models.
    """

    def rank(self, request: RankingRequest) -> Sequence[RankingScore]: ...

    def close(self) -> None: ...


class DefaultRankingPolicy:
    """SenseMem's bounded dense/lexical fusion, reinforcement, temporal and decay policy."""

    def rank(self, request: RankingRequest) -> tuple[RankingScore, ...]:
        proposed = []
        for candidate in request.candidates:
            lexical = (
                LEXICAL_FULL_COVERAGE_RELEVANCE * candidate.lexical_relevance
                if candidate.lexical_coverage >= LEXICAL_FULL_COVERAGE
                else 0.0
            )
            if request.mode is RetrievalMode.LEXICAL:
                relevance = base = lexical = candidate.lexical_relevance
            else:
                base = max(candidate.dense_relevance, lexical)
                relevance = bounded_scale(
                    base, 1.0 + MAX_LEXICAL_RERANK_BONUS * candidate.lexical_coverage
                )
            score = bounded_scale(relevance, candidate.reinforcement_factor)
            if candidate.temporal_factor is not None:
                score = bounded_scale(score, candidate.temporal_factor)
            if candidate.retention_factor is not None:
                score = bounded_scale(score, candidate.retention_factor)
            proposed.append(
                RankingScore(candidate.memory.id, score, relevance, lexical, relevance - base)
            )
        return tuple(proposed)

    def close(self) -> None:
        pass


def bounded_scale(score: float, factor: float) -> float:
    """Scale a bounded score while retaining order below the upper bound."""
    if factor <= 1.0:
        return score * factor
    return score + (1.0 - score) * (factor - 1.0)
