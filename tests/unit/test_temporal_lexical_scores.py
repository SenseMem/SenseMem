"""Default candidate discovery retains native evidence strength across temporal routes."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sensemem import ValidationError
from sensemem.candidate_sources import (
    CandidateBatch,
    CandidateHit,
    CandidateQuery,
    CandidateRequest,
    DefaultCandidateSource,
)
from sensemem.types import MemoryCategory, RetrievalMode

_PREFERRED_RANGE = (
    datetime(2026, 8, 20, tzinfo=timezone.utc),
    datetime(2026, 8, 21, tzinfo=timezone.utc),
)


class _ReadPort:
    def __init__(self, routes: tuple[tuple[CandidateHit, ...], ...]) -> None:
        self.routes = routes
        self.calls: list[tuple[CandidateQuery, ...]] = []

    def read(self, queries: tuple[CandidateQuery, ...]) -> tuple[tuple[CandidateHit, ...], ...]:
        self.calls.append(queries)
        assert len(queries) == len(self.routes)
        return self.routes


def _search(
    routes: tuple[tuple[CandidateHit, ...], ...],
    *,
    mode: RetrievalMode = RetrievalMode.LEXICAL,
    memory_types: frozenset[MemoryCategory] | None = None,
    lexical_queries: tuple[str, ...] = ("query",),
    preferred_range: tuple[datetime, datetime] | None = _PREFERRED_RANGE,
) -> CandidateBatch:
    reader = _ReadPort(routes)
    result = DefaultCandidateSource().search(
        CandidateRequest(
            query_text="query",
            vectors=((1.0, 0.0),),
            lexical_queries=lexical_queries,
            limit=10,
            max_routes=8,
            mode=mode,
            memory_types=memory_types,
            occurred_from=None,
            occurred_until=None,
            preferred_range=preferred_range,
        ),
        reader,
    )
    assert len(reader.calls) == 1
    if preferred_range is not None:
        queries = reader.calls[0]
        midpoint = len(queries) // 2
        assert all(
            (query.occurred_from, query.occurred_until) == preferred_range
            for query in queries[:midpoint]
        )
        assert all(
            query.occurred_from is None and query.occurred_until is None
            for query in queries[midpoint:]
        )
    return result


def test_temporal_merge_preserves_known_global_scores() -> None:
    global_hit = CandidateHit(id="weak", relevance=0.08, lexical_match=True)
    preferred = CandidateHit(id="weak", relevance=1.0, lexical_match=True)
    assert _search(((preferred,), (global_hit,))).lexical == (global_hit,)


def test_temporal_bm25_normalization_uses_each_global_type_route() -> None:
    episodic_preferred = CandidateHit(
        id="episodic-new",
        relevance=1.0,
        lexical_match=True,
        lexical_score=0.75,
        lexical_route="episodic",
    )
    semantic_preferred = CandidateHit(
        id="semantic-new",
        relevance=1.0,
        lexical_match=True,
        lexical_score=2.0,
        lexical_route="semantic",
    )
    episodic_global = CandidateHit(
        id="episodic-best",
        relevance=1.0,
        lexical_match=True,
        lexical_score=1.0,
        lexical_route="episodic",
    )
    semantic_global = CandidateHit(
        id="semantic-best",
        relevance=1.0,
        lexical_match=True,
        lexical_score=4.0,
        lexical_route="semantic",
    )
    result = _search(
        ((episodic_preferred,), (semantic_preferred,), (episodic_global,), (semantic_global,)),
        memory_types=frozenset({MemoryCategory.EPISODIC, MemoryCategory.SEMANTIC}),
    )
    assert {hit.id: hit.relevance for hit in result.lexical} == {
        "semantic-best": 1.0,
        "episodic-best": 1.0,
        "semantic-new": 0.5,
        "episodic-new": 0.75,
    }


def test_temporal_rank_fusion_without_global_ranks_stays_below_global_tail() -> None:
    preferred = (
        CandidateHit(id="new-first", relevance=1.0, lexical_match=True),
        CandidateHit(id="new-second", relevance=0.5, lexical_match=True),
    )
    fallback = (
        CandidateHit(id="best", relevance=1.0, lexical_match=True),
        CandidateHit(id="tail", relevance=0.2, lexical_match=True),
    )
    merged = _search((preferred, fallback)).lexical
    assert merged[:2] == fallback
    assert 0.0 < merged[3].relevance < merged[2].relevance < 0.2
    assert all(hit.relevance == 0.0 for hit in _search((preferred, ())).lexical)


def test_route_merge_preserves_native_lexical_score_metadata() -> None:
    hit = CandidateHit(
        id="shared", relevance=0.8, lexical_match=True, lexical_score=2.0, lexical_route="semantic"
    )
    merged = _search(
        ((hit,), (hit,)), lexical_queries=("query", "paraphrase"), preferred_range=None
    )
    assert merged.lexical == (hit,)


def test_temporal_candidate_merge_keeps_the_best_vector_score() -> None:
    result = _search(
        (
            (CandidateHit(id="shared", relevance=0.2),),
            (CandidateHit(id="shared", relevance=0.9),),
        ),
        mode=RetrievalMode.DENSE,
    )
    assert result.dense == (CandidateHit(id="shared", relevance=0.9),)


@pytest.mark.parametrize("score", (float("nan"), float("inf"), -1.0, True))
def test_native_lexical_score_rejects_invalid_strength(score: float) -> None:
    with pytest.raises(ValidationError, match="lexical score"):
        CandidateHit(id="bad", relevance=1.0, lexical_match=True, lexical_score=score)
