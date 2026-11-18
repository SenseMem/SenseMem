"""Lexical route variants do not replace the question delivered to a scoring policy."""

from datetime import datetime, timezone

import pytest

from sensemem import (
    DefaultRankingPolicy,
    RankingMemory,
    RankingRequest,
    RankingScore,
    RetrievalMode,
)
from sensemem.infrastructure.local.store import StoredMemory
from sensemem.kernel.ranking import rank_memories


def test_ranking_policy_receives_original_query_with_multiple_lexical_routes() -> None:
    class RecordingPolicy(DefaultRankingPolicy):
        def __init__(self) -> None:
            self.requests: list[RankingRequest] = []

        def rank(self, request: RankingRequest) -> tuple[RankingScore, ...]:
            self.requests.append(request)
            return super().rank(request)

    reference = datetime(2026, 11, 16, tzinfo=timezone.utc)
    memory = StoredMemory(
        memory_id="source",
        content="hint1 hint2",
        metadata_json="{}",
        created_at=reference,
        updated_at=reference,
    )
    snapshot = RankingMemory(memory.memory_id, memory.content, memory.created_at)
    policy = RecordingPolicy()
    ranked, _trace = rank_memories(
        (memory,),
        policy=policy,
        query="原问题",
        hydrated=(snapshot,),
        dense_relevance={memory.memory_id: 0.6},
        dense_confidence={memory.memory_id: 1.0},
        lexical_relevance_by_rank={memory.memory_id: 0.7},
        lexical_matches={memory.memory_id},
        lexical_queries=("hint1", "hint2"),
        index_ids_by_memory={},
        retrieval_mode=RetrievalMode.HYBRID,
        reference_at=reference,
        temporal_range=None,
        decay_half_life=None,
        capture_trace=False,
    )
    assert policy.requests[0].query == "原问题"
    assert len(ranked) == 1
    assert ranked[0][0] == memory
    assert ranked[0][1] == pytest.approx(0.72)
