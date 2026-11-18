"""Computation plugins share factory resolution and lifetime with bundled model adapters."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from _feature_support import TinyEmbedder

import sensemem.configuration as configuration
from sensemem import (
    CandidateBatch,
    CandidateReadPort,
    CandidateRequest,
    CandidateSource,
    ContextSelection,
    ContextSelectionRequest,
    ContextSelector,
    DefaultCandidateSource,
    DefaultContextSelector,
    DefaultQueryPolicy,
    DefaultRankingPolicy,
    IngestionStage,
    Memory,
    ModelInput,
    QueryPolicy,
    QueryRepresentationPlan,
    QueryRequest,
    RankingPolicy,
    RankingRequest,
    RankingScore,
    StageResult,
    ValidationError,
    build_memory_composition,
)


class SharedPlugin(TinyEmbedder):
    name = "application-analysis"
    space = "application-analysis:v1"
    requires_embeddings = True

    def __init__(self) -> None:
        self.closed = 0
        self.processed = 0
        self.ranked = 0
        self.selected = 0
        self.planned = 0
        self.sourced = 0

    def plan(self, request: QueryRequest) -> QueryRepresentationPlan:
        self.planned += 1
        return DefaultQueryPolicy().plan(request)

    def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch:
        self.sourced += 1
        return DefaultCandidateSource().search(request, reader)

    def process(self, input: ModelInput) -> StageResult:
        self.processed += 1
        return StageResult("The source describes a workshop")

    def rank(self, request: RankingRequest) -> tuple[RankingScore, ...]:
        self.ranked += 1
        return DefaultRankingPolicy().rank(request)

    def select(self, request: ContextSelectionRequest) -> ContextSelection:
        self.selected += 1
        return DefaultContextSelector().select(request)

    def close(self) -> None:
        self.closed += 1


class CloseOnly:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


@pytest.mark.parametrize("registered", [False, True])
def test_bundled_and_application_factories_share_product_paths_and_resource_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    shared = SharedPlugin()
    received: list[configuration.EmbeddingProviderConfig] = []

    def build_embedding(config: configuration.EmbeddingProviderConfig) -> SharedPlugin:
        received.append(config)
        return shared

    monkeypatch.setattr(configuration, "_build_embedding", build_embedding)
    config: dict[str, object] = {
        "data_dir": str(tmp_path / "memory"),
        "ingestion_stages": [{"factory": "application"}],
        "settings": {"minimum_relevance": 0, "ambiguity_margin": 0},
    }
    if registered:
        config.update(
            {
                "embedding": {"factory": "sensemem.embedding.openai", "options": {"dimension": 4}},
                "ranking_policy": {"factory": "application"},
                "context_selector": {"factory": "application"},
                "query_policy": {"factory": "application"},
                "candidate_source": {"factory": "application"},
            }
        )
    else:
        config["embedding"] = {"provider": "openai", "dimension": 4}
        config["ranking_policy"] = {"factory": "application"}
        config["context_selector"] = {"factory": "application"}
        config["query_policy"] = {"factory": "application"}
        config["candidate_source"] = {"factory": "application"}
    with Memory.from_config(config, factories={"application": lambda _: shared}) as memory:
        record = memory.add("The key is in the workshop")
        assert memory.search("key").hits[0].id == record.id
        assert memory.compile("key").hits[0].id == record.id
        assert shared.processed == 1
        assert shared.ranked >= 2
        assert shared.selected >= 1
        assert shared.planned >= 2
        assert shared.sourced >= 2
        assert shared.closed == 0
    assert len(received) == 1
    assert isinstance(received[0], configuration.OpenAIEmbeddingConfig)
    assert received[0].dimension == 4
    assert shared.closed == 1


def test_registered_builtin_default_policies_use_the_common_catalog(tmp_path: Path) -> None:
    shared = SharedPlugin()
    composition = build_memory_composition(
        {
            "data_dir": str(tmp_path / "unopened"),
            "embedding": {"factory": "application"},
            "ranking_policy": {"factory": "sensemem.ranking.default"},
            "context_selector": {"factory": "sensemem.context.default"},
            "query_policy": {"factory": "sensemem.query.default"},
            "candidate_source": {"factory": "sensemem.candidates.default"},
        },
        factories={"application": lambda _: shared},
    )
    assert isinstance(composition.plugins.ranking_policy, DefaultRankingPolicy)
    assert isinstance(composition.plugins.context_selector, DefaultContextSelector)
    assert isinstance(composition.plugins.query_policy, DefaultQueryPolicy)
    assert isinstance(composition.plugins.candidate_source, DefaultCandidateSource)
    assert not composition.data_dir.exists()
    composition.close()
    assert shared.closed == 1


@pytest.mark.parametrize(
    "slot",
    ["ranking_policy", "context_selector", "query_policy", "candidate_source", "ingestion_stages"],
)
def test_invalid_computation_roles_close_every_resource_before_storage_opens(
    tmp_path: Path, slot: str
) -> None:
    shared, wrong = SharedPlugin(), CloseOnly()
    config: dict[str, object] = {
        "data_dir": str(tmp_path / "unopened"),
        "embedding": {"factory": "application"},
    }
    if slot == "ingestion_stages":
        config[slot] = [{"factory": "wrong"}]
    else:
        config[slot] = {"factory": "wrong"}
    with pytest.raises(ValidationError, match="incompatible capability"):
        Memory.from_config(
            config, factories={"application": lambda _: shared, "wrong": lambda _: wrong}
        )
    assert shared.closed == wrong.closed == 1
    assert not (tmp_path / "unopened").exists()


@pytest.mark.parametrize(
    "slot",
    ["ranking_policy", "context_selector", "query_policy", "candidate_source", "ingestion_stages"],
)
def test_direct_computation_roles_are_validated_before_storage_opens(
    tmp_path: Path, slot: str
) -> None:
    shared, wrong = SharedPlugin(), CloseOnly()
    data_dir = tmp_path / "unopened"
    with pytest.raises(ValidationError, match=rf"plugins\.{slot}"):
        if slot == "ranking_policy":
            Memory(data_dir, embedder=shared, ranking_policy=cast(RankingPolicy, wrong))
        elif slot == "context_selector":
            Memory(data_dir, embedder=shared, context_selector=cast(ContextSelector, wrong))
        elif slot == "query_policy":
            Memory(data_dir, embedder=shared, query_policy=cast(QueryPolicy, wrong))
        elif slot == "candidate_source":
            Memory(data_dir, embedder=shared, candidate_source=cast(CandidateSource, wrong))
        else:
            Memory(data_dir, embedder=shared, ingestion_stages=(cast(IngestionStage, wrong),))
    assert shared.closed == wrong.closed == 1
    assert not data_dir.exists()


def test_candidate_requirements_are_validated_before_opening_storage(tmp_path: Path) -> None:
    shared = SharedPlugin()
    shared.requires_embeddings = cast(bool, "yes")
    data_dir = tmp_path / "unopened"
    with pytest.raises(ValidationError, match="requires_embeddings must be a boolean"):
        Memory(data_dir, embedder=shared, candidate_source=shared)
    assert shared.closed == 1
    assert not data_dir.exists()


@pytest.mark.parametrize("invalid", ["duplicate", "name", "space"])
def test_stage_identity_validation_precedes_storage_and_deduplicates_cleanup(
    tmp_path: Path, invalid: str
) -> None:
    shared = SharedPlugin()
    if invalid != "duplicate":
        setattr(shared, invalid, " ")
    stages = (shared, shared) if invalid == "duplicate" else (shared,)
    with pytest.raises(ValidationError, match="stage"):
        Memory(tmp_path / "unopened", embedder=shared, ingestion_stages=stages)
    assert shared.closed == 1
    assert not (tmp_path / "unopened").exists()


def test_later_stage_factory_failure_closes_shared_policy_and_backend_once(tmp_path: Path) -> None:
    shared = SharedPlugin()

    def fail(_options: Mapping[str, object]) -> SharedPlugin:
        raise RuntimeError("construction failed")

    with pytest.raises(RuntimeError, match="construction failed"):
        build_memory_composition(
            {
                "data_dir": str(tmp_path / "unopened"),
                "embedding": {"factory": "application"},
                "ranking_policy": {"factory": "application"},
                "ingestion_stages": [{"factory": "application"}, {"factory": "failure"}],
            },
            factories={"application": lambda _: shared, "failure": fail},
        )
    assert shared.closed == 1
    assert not (tmp_path / "unopened").exists()


def test_application_cannot_override_bundled_factory_names(tmp_path: Path) -> None:
    shared = SharedPlugin()
    with pytest.raises(ValidationError, match="cannot replace bundled factories"):
        build_memory_composition(
            {
                "data_dir": str(tmp_path / "unopened"),
                "embedding": {"factory": "application"},
            },
            factories={
                "application": lambda _: shared,
                "sensemem.ranking.default": lambda _: shared,
            },
        )
    assert shared.closed == 0
    assert not (tmp_path / "unopened").exists()


def test_invalid_builtin_options_close_prior_application_resources(tmp_path: Path) -> None:
    shared = SharedPlugin()
    with pytest.raises(ValidationError, match="invalid bundled plugin options"):
        build_memory_composition(
            {
                "data_dir": str(tmp_path / "unopened"),
                "embedding": {"factory": "application"},
                "ranking_policy": {
                    "factory": "sensemem.ranking.default",
                    "options": {"unexpected": True},
                },
            },
            factories={"application": lambda _: shared},
        )
    assert shared.closed == 1
    assert not (tmp_path / "unopened").exists()
