"""Replaceable candidate discovery over a bounded, request-scoped native read port."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from sensemem._validation import nonempty_text, unit_interval
from sensemem.exceptions import ValidationError
from sensemem.types import MemoryCategory, RetrievalMode

CandidateKind = Literal["dense", "lexical"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateQuery:
    """One native read; its bounds can narrow the enclosing request, never expand it.

    Dense reads use vector; lexical reads use text. The kernel supplies the embedding
    space, task, scope and route depth. exact and ef tune native dense discovery only.
    """

    kind: CandidateKind
    vector: tuple[float, ...] = ()
    text: str = ""
    memory_type: MemoryCategory | None = None
    occurred_from: datetime | None = None
    occurred_until: datetime | None = None
    exact: bool = False
    ef: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateHit:
    """An opaque local index ID and ranking hints, without memory content or authority."""

    id: str
    relevance: float
    confidence: float | None = None
    lexical_match: bool = False
    lexical_score: float | None = None
    lexical_route: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", nonempty_text(self.id, "candidate index ID"))
        unit_interval(self.relevance, "candidate relevance")
        if self.confidence is None:
            object.__setattr__(self, "confidence", self.relevance)
        else:
            unit_interval(self.confidence, "candidate confidence")
        if self.lexical_score is not None and (
            isinstance(self.lexical_score, bool)
            or not isinstance(self.lexical_score, int | float)
            or not math.isfinite(self.lexical_score)
            or self.lexical_score < 0.0
        ):
            raise ValidationError("candidate lexical score must be finite and non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryCandidate:
    """An external shortlist entry referring to an already-indexed local memory.

    Unknown, stale, forgotten or out-of-scope IDs contribute no evidence. A source supplies
    ranking hints only; SQLite supplies the record and original media in every case.
    """

    memory_id: str
    relevance: float
    kind: CandidateKind = "dense"

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", nonempty_text(self.memory_id, "candidate memory ID"))
        unit_interval(self.relevance, "candidate relevance")
        if self.kind not in {"dense", "lexical"}:
            raise ValidationError("candidate kind must be dense or lexical")


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateBatch:
    """Candidate hints from one source pass, before authoritative hydration.

    exhausted describes the discovery routes, never exhaustive coverage of a recall plan.
    The source may return at most request.limit external memory references and at most
    request.max_routes * request.limit native hits in total.
    """

    dense: tuple[CandidateHit, ...] = ()
    lexical: tuple[CandidateHit, ...] = ()
    memories: tuple[MemoryCandidate, ...] = ()
    exhausted: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateRequest:
    """Prepared query routes and immutable bounds for one candidate-window pass."""

    query_text: str
    vectors: tuple[tuple[float, ...], ...]
    lexical_queries: tuple[str, ...]
    limit: int
    max_routes: int
    mode: RetrievalMode
    memory_types: frozenset[MemoryCategory] | None
    occurred_from: datetime | None
    occurred_until: datetime | None
    preferred_range: tuple[datetime, datetime] | None


class CandidateReadPort(Protocol):
    """Execute native routes with fixed instance identity and request scope.

    A batch retains the runtime's bounded native parallelism. The port carries no write,
    lifecycle, storage, asset, flush or rebuild capability and is valid only during search.
    """

    def read(self, queries: tuple[CandidateQuery, ...]) -> tuple[tuple[CandidateHit, ...], ...]: ...


@runtime_checkable
class CandidateSource(Protocol):
    """Discover candidate references; the kernel hydrates, authorizes and ranks them.

    requires_embeddings is fixed at assembly. False skips query embedding preparation,
    allowing a text-only external source without an unused embedding-model request. Sources
    must be thread-safe; Memory owns and closes each distinct configured source once.
    """

    requires_embeddings: bool

    def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch: ...

    def close(self) -> None: ...


class DefaultCandidateSource:
    """Discover native dense and lexical routes with SenseMem's temporal score normalization."""

    requires_embeddings = True

    def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch:
        memory_types = (
            (None,)
            if request.memory_types is None
            else tuple(sorted(request.memory_types, key=lambda value: value.value))
        )
        ranges = (
            ((request.occurred_from, request.occurred_until),)
            if request.preferred_range is None
            else (request.preferred_range, (request.occurred_from, request.occurred_until))
        )
        queries: list[CandidateQuery] = []
        sizes = []
        for occurred_from, occurred_until in ranges:
            dense = tuple(
                CandidateQuery(
                    kind="dense",
                    vector=vector,
                    memory_type=memory_type,
                    occurred_from=occurred_from,
                    occurred_until=occurred_until,
                )
                for memory_type in memory_types
                for vector in request.vectors
                if request.mode is not RetrievalMode.LEXICAL
            )
            lexical = tuple(
                CandidateQuery(
                    kind="lexical",
                    text=query,
                    memory_type=memory_type,
                    occurred_from=occurred_from,
                    occurred_until=occurred_until,
                )
                for memory_type in memory_types
                for query in request.lexical_queries
                if request.mode is not RetrievalMode.DENSE
            )
            sizes.append((len(dense), len(lexical)))
            queries.extend((*dense, *lexical))
        routes = reader.read(tuple(queries))
        groups = []
        offset = 0
        for dense_count, lexical_count in sizes:
            dense_routes = routes[offset : offset + dense_count]
            offset += dense_count
            lexical_routes = routes[offset : offset + lexical_count]
            offset += lexical_count
            groups.append(
                CandidateBatch(
                    dense=merge_candidate_hits(*dense_routes),
                    lexical=(
                        ()
                        if not lexical_routes
                        else lexical_routes[0]
                        if len(lexical_routes) == 1
                        else tuple(
                            sorted(
                                merge_candidate_hits(*lexical_routes),
                                key=lambda hit: -hit.relevance,
                            )
                        )
                    ),
                    exhausted=all(
                        len(route) < request.limit for route in (*dense_routes, *lexical_routes)
                    ),
                )
            )
        if len(groups) == 1:
            return groups[0]
        preferred, fallback = groups
        return CandidateBatch(
            dense=merge_candidate_hits(preferred.dense, fallback.dense),
            lexical=_temporal_lexical(preferred.lexical, fallback.lexical),
            exhausted=preferred.exhausted and fallback.exhausted,
        )

    def close(self) -> None:
        pass


def merge_candidate_hits(*groups: tuple[CandidateHit, ...]) -> tuple[CandidateHit, ...]:
    """Merge a discovered index ID's signals once, preserving its first-seen order."""
    merged: dict[str, CandidateHit] = {}
    for group in groups:
        for hit in group:
            current = merged.get(hit.id)
            if current is None:
                merged[hit.id] = hit
                continue
            assert current.confidence is not None and hit.confidence is not None
            merged[hit.id] = CandidateHit(
                id=hit.id,
                relevance=max(current.relevance, hit.relevance),
                confidence=max(current.confidence, hit.confidence),
                lexical_match=current.lexical_match or hit.lexical_match,
                lexical_score=max(
                    (
                        value
                        for value in (current.lexical_score, hit.lexical_score)
                        if value is not None
                    ),
                    default=None,
                ),
                lexical_route=current.lexical_route or hit.lexical_route,
            )
    return tuple(merged.values())


def _temporal_lexical(
    preferred: tuple[CandidateHit, ...], fallback: tuple[CandidateHit, ...]
) -> tuple[CandidateHit, ...]:
    global_ids = {hit.id for hit in fallback}
    maxima: dict[str | None, float] = {}
    tails: dict[str | None, float] = {}
    for hit in fallback:
        route = hit.lexical_route
        tails[route] = min(tails.get(route, 1.0), hit.relevance)
        if hit.lexical_score is not None:
            maxima[route] = max(maxima.get(route, 0.0), hit.lexical_score)
    added = []
    for hit in preferred:
        if hit.id in global_ids:
            continue
        tail = tails.get(hit.lexical_route, 0.0)
        maximum = maxima.get(hit.lexical_route, 0.0)
        relevance = (
            min(tail, hit.lexical_score / maximum)
            if hit.lexical_score is not None and maximum > 0.0
            else math.nextafter(tail, 0.0) * hit.relevance
        )
        added.append(
            CandidateHit(
                id=hit.id,
                relevance=relevance,
                confidence=hit.confidence,
                lexical_match=hit.lexical_match,
                lexical_score=hit.lexical_score,
                lexical_route=hit.lexical_route,
            )
        )
    return (*fallback, *added)
