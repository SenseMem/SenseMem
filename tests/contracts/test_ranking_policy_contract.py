"""External scoring policies share the SDK path without gaining candidate or write authority."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sensemem import (
    ContextBudget,
    EmbedTask,
    Memory,
    MemorySettings,
    Modality,
    ModelError,
    ModelInput,
    ObservationContext,
    RankingRequest,
    RankingScore,
    RetrievalScope,
    ValidationError,
)


class FixtureEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "ranking-policy-contract:v1"
    embedding_space = "ranking-policy-contract:4:v1"
    embedding_dimension = 4

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return ((1.0, 0.0, 0.0, 0.0),) * len(inputs)

    def close(self) -> None:
        pass


class PreferGarden:
    def __init__(self) -> None:
        self.requests: list[RankingRequest] = []
        self.closed = 0

    def rank(self, request: RankingRequest) -> Sequence[RankingScore]:
        self.requests.append(request)
        return tuple(
            RankingScore(
                candidate.memory.id,
                0.9 if "garden" in candidate.memory.content else 0.2,
                0.8,
            )
            for candidate in request.candidates
        )

    def close(self) -> None:
        self.closed += 1


def test_external_policy_controls_search_and_compile_without_rewriting_records(
    tmp_path: Path,
) -> None:
    policy = PreferGarden()
    with Memory(
        tmp_path,
        embedder=FixtureEmbedder(),
        ranking_policy=policy,
        settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        kitchen, garden = memory.add_many(("shared kitchen", "shared garden"))
        result = memory.search("shared", explain=True)
        assert [hit.id for hit in result.hits] == [garden.id, kitchen.id]
        assert [hit.score for hit in result.hits] == [0.9, 0.2]
        assert result.trace is not None
        assert result.trace.candidates[0].final_score in {0.9, 0.2}
        assert memory.compile("shared", budget=ContextBudget(max_items=1)).hits[0].id == garden.id
        assert policy.requests[0].query == "shared"
        assert {candidate.memory.content for candidate in policy.requests[0].candidates} == {
            "shared garden",
            "shared kitchen",
        }
        assert memory.get(kitchen.id) == kitchen
        assert memory.get(garden.id) == garden
    assert policy.closed == 1


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unknown", "invalid", "raises"])
def test_invalid_policy_proposals_fail_explicitly_without_leaking_error_bodies(
    tmp_path: Path, failure: str
) -> None:
    class Broken(PreferGarden):
        def rank(self, request: RankingRequest) -> Sequence[RankingScore]:
            scores = tuple(super().rank(request))
            if failure == "missing":
                return scores[:-1]
            if failure == "duplicate":
                return (scores[0],) * len(scores)
            if failure == "unknown":
                return (RankingScore("invented", 1.0, 1.0), *scores[1:])
            if failure == "invalid":
                return (RankingScore(scores[0].memory_id, float("nan"), 1.0), *scores[1:])
            raise RuntimeError("private provider response body")

    with Memory(tmp_path, embedder=FixtureEmbedder(), ranking_policy=Broken()) as memory:
        memory.add_many(("shared kitchen", "shared garden"))
        with pytest.raises(ModelError) as caught:
            memory.search("shared")
        assert caught.value.reason == (
            "model_failed" if failure == "raises" else "response_invalid"
        )
        assert caught.value.stage == "ranking"
        assert "private provider" not in str(caught.value)
        assert len(memory.list().items) == 2


def test_policy_only_receives_scoped_records_and_cannot_override_observation_confidence(
    tmp_path: Path,
) -> None:
    policy = PreferGarden()
    with Memory(
        tmp_path,
        embedder=FixtureEmbedder(),
        ranking_policy=policy,
        settings=MemorySettings(minimum_relevance=0.5, ambiguity_margin=0),
    ) as memory:
        kitchen = memory.add("shared kitchen", context=ObservationContext(place_id="kitchen"))
        memory.add("shared garden", context=ObservationContext(place_id="garden", confidence=0.1))
        assert [hit.id for hit in memory.search("shared").hits] == [kitchen.id]
        assert not memory.search("shared", scope=RetrievalScope(place_id="garden")).hits
        assert {candidate.memory.place_id for candidate in policy.requests[-1].candidates} == {
            "garden"
        }


def test_policy_can_rank_from_raw_update_and_access_history_without_bypassing_the_gate(
    tmp_path: Path,
) -> None:
    class AccessHistoryPolicy(PreferGarden):
        def rank(self, request: RankingRequest) -> Sequence[RankingScore]:
            self.requests.append(request)
            scores = []
            for candidate in request.candidates:
                history = candidate.memory
                anchor = history.last_accessed_at or history.updated_at or history.created_at
                age_days = max(0.0, (request.reference_at - anchor).total_seconds() / 86_400)
                score = 0.2 + 0.3 * min(history.access_count, 2) + 0.1 / (1.0 + age_days)
                scores.append(RankingScore(history.id, score, 0.8))
            return tuple(scores)

    policy = AccessHistoryPolicy()
    with Memory(
        tmp_path,
        embedder=FixtureEmbedder(),
        ranking_policy=policy,
        settings=MemorySettings(minimum_relevance=0.5, ambiguity_margin=0),
    ) as memory:
        older = memory.add(
            "shared older", occurred_at=datetime.now(timezone.utc) - timedelta(days=30)
        )
        fresh = memory.add("shared fresh")
        uncertain = memory.add("shared uncertain", context=ObservationContext(confidence=0.1))
        assert memory.reinforce((older.id, uncertain.id)) == 2
        assert memory.reinforce((uncertain.id,)) == 1
        result = memory.search("shared")
        assert [hit.id for hit in result.hits] == [older.id, fresh.id]
        candidates = {
            candidate.memory.id: candidate.memory for candidate in policy.requests[0].candidates
        }
        older_candidate = candidates[older.id]
        assert older_candidate.updated_at == older.created_at
        assert older_candidate.last_accessed_at is not None
        assert older_candidate.last_accessed_at >= older.created_at
        assert older_candidate.access_count == 1
        assert candidates[fresh.id].last_accessed_at is None
        assert candidates[fresh.id].access_count == 0
        assert candidates[uncertain.id].access_count == 2
    assert policy.closed == 1


@pytest.mark.parametrize("score", [True, -0.1, 1.1, float("inf"), float("nan")])
def test_score_contract_requires_finite_bounded_values(score: float) -> None:
    with pytest.raises(ValidationError):
        RankingScore("memory", score, 0.5)
