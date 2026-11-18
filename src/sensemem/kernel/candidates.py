"""Bound native candidate reads and resolve source references against SQLite."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import closing
from datetime import datetime
from functools import partial

from opentelemetry.trace import Tracer

from sensemem.candidate_sources import (
    CandidateBatch,
    CandidateHit,
    CandidateQuery,
    CandidateRequest,
    CandidateSource,
    MemoryCandidate,
    merge_candidate_hits,
)
from sensemem.exceptions import ModelError, SenseMemError, ValidationError
from sensemem.infrastructure.local.store import LocalStore
from sensemem.infrastructure.local.zvec_index import IndexHit
from sensemem.kernel.execution import Execution
from sensemem.kernel.ranking import IndexCandidates
from sensemem.kernel.runtime import Index, translate_index_errors, translate_storage_errors
from sensemem.kernel.tracing import Traced
from sensemem.types import MemoryCategory


class NativeCandidateReader:
    """A request-local read capability; no plugin receives its storage or index owner."""

    def __init__(
        self,
        *,
        index: Index,
        execution: Execution,
        request: CandidateRequest,
        space_id: str,
        task: str,
        dimension: int,
        place_id: str | None,
        identity_id: str | None,
    ) -> None:
        self._index = index
        self._execution = execution
        self._request = request
        self._space_id = space_id
        self._task = task
        self._dimension = dimension
        self._place_id = place_id
        self._identity_id = identity_id
        self.route_count = 0

    def read(  # noqa: C901 - validate one bounded native-read capability at its owning boundary
        self, queries: tuple[CandidateQuery, ...]
    ) -> tuple[tuple[CandidateHit, ...], ...]:
        if not isinstance(queries, tuple) or any(
            not isinstance(query, CandidateQuery) for query in queries
        ):
            raise ValidationError("candidate reads require a tuple of CandidateQuery values")
        if self.route_count + len(queries) > self._request.max_routes:
            raise ValidationError("candidate source exceeded the native route budget")
        self.route_count += len(queries)
        calls: list[Callable[[], tuple[IndexHit, ...]]] = []
        for query in queries:
            if query.memory_type is not None and not isinstance(query.memory_type, MemoryCategory):
                raise ValidationError("candidate memory type must be a MemoryCategory")
            if self._request.memory_types is not None and (
                query.memory_type not in self._request.memory_types
            ):
                raise ValidationError("candidate read cannot expand the requested memory types")
            if any(
                not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
                for value in (query.occurred_from, query.occurred_until)
                if value is not None
            ):
                raise ValidationError("candidate occurrence bounds must be timezone-aware")
            bounds_from = tuple(
                value
                for value in (query.occurred_from, self._request.occurred_from)
                if value is not None
            )
            bounds_until = tuple(
                value
                for value in (query.occurred_until, self._request.occurred_until)
                if value is not None
            )
            occurred_from = max(bounds_from, default=None)
            occurred_until = min(bounds_until, default=None)
            if (
                occurred_from is not None
                and occurred_until is not None
                and occurred_from > occurred_until
            ):
                calls.append(lambda: ())
                continue
            if query.kind == "dense":
                if (
                    query.text
                    or not isinstance(query.vector, tuple)
                    or len(query.vector) != self._dimension
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int | float)
                        or not math.isfinite(value)
                        for value in query.vector
                    )
                    or not isinstance(query.exact, bool)
                    or (
                        query.ef is not None
                        and (
                            isinstance(query.ef, bool)
                            or not isinstance(query.ef, int)
                            or query.ef <= 0
                        )
                    )
                ):
                    raise ValidationError(
                        "candidate dense query has invalid vector or search controls"
                    )
                call = partial(
                    self._index.search,
                    query.vector,
                    limit=self._request.limit,
                    space_id=self._space_id,
                    task=self._task,
                    memory_type=None if query.memory_type is None else query.memory_type.value,
                    occurred_from=occurred_from,
                    occurred_until=occurred_until,
                    place_id=self._place_id,
                    identity_id=self._identity_id,
                )
                if query.exact or query.ef is not None:
                    call = partial(call, exact=query.exact, ef=query.ef)
                calls.append(call)
            elif query.kind == "lexical":
                if (
                    not isinstance(query.text, str)
                    or not query.text.strip()
                    or query.vector
                    or query.exact
                    or query.ef is not None
                ):
                    raise ValidationError("candidate lexical query must contain text")
                calls.append(
                    partial(
                        self._index.lexical_search,
                        query.text,
                        limit=self._request.limit,
                        space_id=self._space_id,
                        task=self._task,
                        memory_type=None if query.memory_type is None else query.memory_type.value,
                        occurred_from=occurred_from,
                        occurred_until=occurred_until,
                        place_id=self._place_id,
                        identity_id=self._identity_id,
                    )
                )
            else:
                raise ValidationError("candidate query kind must be dense or lexical")
        with translate_index_errors("search memories"):
            routes = self._execution.index_routes(calls)
        return tuple(
            tuple(
                CandidateHit(
                    id=hit.id,
                    relevance=hit.relevance,
                    confidence=hit.confidence,
                    lexical_match=hit.lexical_match,
                    lexical_score=hit.lexical_score,
                    lexical_route=hit.lexical_route,
                )
                for hit in route
            )
            for route in routes
        )


class Candidates(Traced):
    """Discover bounded source references and resolve their identity from SQLite."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        source: CandidateSource,
        store: LocalStore,
        index: Index,
        execution: Execution,
        space_id: str,
        task: str,
        dimension: int,
    ) -> None:
        super().__init__(tracer)
        self._source = source
        self._store = store
        self._index = index
        self._execution = execution
        self._space_id = space_id
        self._task = task
        self._dimension = dimension
        self.requires_embeddings = source.requires_embeddings

    def search(  # noqa: C901 - validate source output and resolve both local reference forms once
        self,
        request: CandidateRequest,
        *,
        place_id: str | None,
        identity_id: str | None,
    ) -> IndexCandidates:
        reader = NativeCandidateReader(
            index=self._index,
            execution=self._execution,
            request=request,
            space_id=self._space_id,
            task=self._task,
            dimension=self._dimension,
            place_id=place_id,
            identity_id=identity_id,
        )
        with self._trace("sensemem.index.search", kind="stage") as span:
            try:
                batch = self._source.search(request, reader)
            except SenseMemError:
                raise
            except Exception:
                raise ModelError(
                    "candidate source failed", stage="candidates", reason="model_failed"
                ) from None
            span.set_attribute("sensemem.index.route_count", reader.route_count)
            if not isinstance(batch, CandidateBatch):
                raise ValidationError("candidate source must return a CandidateBatch")
            if (
                not isinstance(batch.dense, tuple)
                or not isinstance(batch.lexical, tuple)
                or not isinstance(batch.memories, tuple)
                or any(not isinstance(hit, CandidateHit) for hit in (*batch.dense, *batch.lexical))
                or any(not isinstance(hit, MemoryCandidate) for hit in batch.memories)
                or not isinstance(batch.exhausted, bool)
            ):
                raise ValidationError("candidate source returned invalid candidate values")
            if (
                len(batch.dense) + len(batch.lexical) > request.max_routes * request.limit
                or len(batch.memories) > request.limit
            ):
                raise ValidationError("candidate source exceeded the candidate budget")
            dense = merge_candidate_hits(batch.dense)
            lexical = merge_candidate_hits(batch.lexical)
            if batch.memories:
                by_id: dict[str, dict[str, MemoryCandidate]] = {}
                for proposal in batch.memories:
                    signals = by_id.setdefault(proposal.memory_id, {})
                    existing = signals.get(proposal.kind)
                    if existing is None or proposal.relevance > existing.relevance:
                        signals[proposal.kind] = proposal
                external_dense: list[CandidateHit] = []
                external_lexical: list[CandidateHit] = []
                with (
                    translate_storage_errors("resolve external search candidates"),
                    closing(
                        self._store.index.iter_memory_embedding_parts(
                            tuple(by_id), space_id=self._space_id, task=self._task
                        )
                    ) as embeddings,
                ):
                    # Only identity is needed here; keep stored vectors packed and leave
                    # authoritative content and asset hydration to the shared retrieval path.
                    for memory_id, embedding_id, _vector in embeddings:
                        for proposal in by_id[memory_id].values():
                            hit = CandidateHit(
                                id=embedding_id,
                                relevance=proposal.relevance,
                                lexical_match=proposal.kind == "lexical",
                            )
                            target = (
                                external_dense if proposal.kind == "dense" else external_lexical
                            )
                            target.append(hit)
                dense = merge_candidate_hits(dense, tuple(external_dense))
                lexical = merge_candidate_hits(lexical, tuple(external_lexical))
            return IndexCandidates(
                dense=tuple(_index_hit(hit) for hit in dense),
                lexical=tuple(_index_hit(hit) for hit in lexical),
                exhausted=batch.exhausted,
            )


def _index_hit(hit: CandidateHit) -> IndexHit:
    return IndexHit(
        id=hit.id,
        relevance=hit.relevance,
        confidence=hit.confidence,
        lexical_match=hit.lexical_match,
        lexical_score=hit.lexical_score,
        lexical_route=hit.lexical_route,
    )
