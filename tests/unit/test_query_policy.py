"""Public query policies control routes while the kernel owns capabilities and execution."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from fixtures.routing import Answerer, Describer, VisualBackend

from sensemem import (
    AssetRef,
    Blob,
    EmbeddingRoute,
    EmbedTask,
    Memory,
    MemorySettings,
    Modality,
    ModelError,
    ModelInput,
    QueryRepresentationPlan,
    QueryRequest,
    RetrievalMode,
)


class QueryEmbedder:
    embedding_model = "query-policy-fixture"
    embedding_space = "query-policy-fixture:v1"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(self) -> None:
        self.query_batches: list[tuple[ModelInput, ...]] = []

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        if task is EmbedTask.QUERY:
            self.query_batches.append(tuple(inputs))
            if any(value.text == "optional failure" for value in inputs):
                raise ModelError("private provider body", reason="model_failed")
        return tuple((1.0, 0.0) if "greenhouse" in value.text else (0.0, 1.0) for value in inputs)

    def close(self) -> None:
        pass


class Policy:
    def __init__(self, plan: Callable[[QueryRequest], QueryRepresentationPlan]) -> None:
        self.plan_query = plan
        self.requests: list[QueryRequest] = []
        self.closed = 0

    def plan(self, request: QueryRequest) -> QueryRepresentationPlan:
        self.requests.append(request)
        return self.plan_query(request)

    def close(self) -> None:
        self.closed += 1


def test_query_rewrite_controls_search_ask_and_compile_without_rewriting_memory(
    tmp_path: Path,
) -> None:
    policy = Policy(
        lambda request: QueryRepresentationPlan(
            "greenhouse", embedding_inputs=(ModelInput(text="greenhouse"),)
        )
    )
    embedder = QueryEmbedder()
    with Memory(
        tmp_path,
        embedder=embedder,
        answerer=Answerer(),
        query_policy=policy,
        settings=MemorySettings(
            retrieval_mode=RetrievalMode.DENSE,
            minimum_relevance=0.8,
            ambiguity_margin=0,
        ),
    ) as memory:
        target, other = memory.add_many(("greenhouse key", "meeting archive"))
        assert memory.search("meeting").hits[0].id == target.id
        assert memory.ask("meeting").hits[0].id == target.id
        assert memory.compile("meeting").hits[0].id == target.id
        assert all(request.content.text == "meeting" for request in policy.requests)
        assert all(batch == (ModelInput(text="greenhouse"),) for batch in embedder.query_batches)
        assert memory.get(other.id) == other
    assert policy.closed == 1


@pytest.mark.parametrize("mode", [RetrievalMode.HYBRID, RetrievalMode.LEXICAL])
def test_policy_can_select_lexical_rewrites_without_query_embeddings(
    tmp_path: Path, mode: RetrievalMode
) -> None:
    policy = Policy(
        lambda request: QueryRepresentationPlan(
            "greenhouse", lexical_queries=(" greenhouse ", "", "the", "greenhouse")
        )
    )
    embedder = QueryEmbedder()
    with Memory(
        tmp_path,
        embedder=embedder,
        query_policy=policy,
        settings=MemorySettings(retrieval_mode=mode, minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        target = memory.add("greenhouse key")
        memory.add("meeting archive")
        assert memory.search("meeting").hits[0].id == target.id
        assert embedder.query_batches == []


@pytest.mark.parametrize("mode", list(RetrievalMode))
@pytest.mark.parametrize("native_image", [False, True])
def test_default_query_routes_keep_text_after_media(
    tmp_path: Path, mode: RetrievalMode, native_image: bool
) -> None:
    class ExactPartEmbedder(QueryEmbedder):
        embedding_capabilities = (
            QueryEmbedder.embedding_capabilities if native_image else frozenset({Modality.TEXT})
        )

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            if task is EmbedTask.QUERY:
                self.query_batches.append(tuple(inputs))
                return tuple(
                    (1.0, 0.0) if value.text == "greenhouse" else (0.0, 1.0) for value in inputs
                )
            return super().embed(inputs, task)

    embedder = ExactPartEmbedder()
    with Memory(
        tmp_path,
        embedder=embedder,
        vision_describer=Describer() if not native_image or mode is RetrievalMode.LEXICAL else None,
        settings=MemorySettings(retrieval_mode=mode, minimum_relevance=0),
    ) as memory:
        target = memory.add("greenhouse")
        result = memory.search(("Where is it?", Blob(b"query", "image/png"), "greenhouse"))
    assert [hit.id for hit in result.hits] == [target.id]
    if mode is not RetrievalMode.LEXICAL:
        inputs = embedder.query_batches[0]
        assert ModelInput(text="greenhouse") in inputs
        assert ModelInput(text="Where is it?") in inputs
        if native_image:
            assert any(value.assets and not value.text for value in inputs)
    else:
        assert embedder.query_batches == []


@pytest.mark.parametrize("mode", [RetrievalMode.LEXICAL, RetrievalMode.DENSE])
def test_policy_cannot_override_the_configured_mode(tmp_path: Path, mode: RetrievalMode) -> None:
    policy = Policy(
        lambda request: (
            QueryRepresentationPlan("query", embedding_inputs=(ModelInput(text="query"),))
            if mode is RetrievalMode.LEXICAL
            else QueryRepresentationPlan("query", lexical_queries=("query",))
        )
    )
    embedder = QueryEmbedder()
    with Memory(
        tmp_path,
        embedder=embedder,
        query_policy=policy,
        settings=MemorySettings(retrieval_mode=mode),
    ) as memory:
        memory.add("greenhouse key")
        with pytest.raises(ModelError, match="conflict with the retrieval mode"):
            memory.search("query")
        assert embedder.query_batches == []


@pytest.mark.parametrize("change", ["foreign", "changed_reference"])
def test_query_media_must_be_exact_references_from_the_leased_input(
    tmp_path: Path, change: str
) -> None:
    foreign: AssetRef | None = None

    def plan(request: QueryRequest) -> QueryRepresentationPlan:
        if change == "foreign":
            assert foreign is not None
            asset = foreign
        else:
            asset = replace(request.content.assets[0], name="different.png")
        return QueryRepresentationPlan("query", embedding_inputs=(ModelInput(assets=(asset,)),))

    embedder = QueryEmbedder()
    with Memory(tmp_path, embedder=embedder, query_policy=Policy(plan)) as memory:
        foreign = memory.add(Blob(b"foreign", "image/png")).assets[0]
        with pytest.raises(ModelError, match="assets must belong to the leased query"):
            memory.search(("query", Blob(b"query", "image/png")))
        assert embedder.query_batches == []


@pytest.mark.parametrize("selection", ["joint", "asset_count"])
def test_query_policy_must_obey_joint_modality_and_asset_count_limits(
    tmp_path: Path, selection: str
) -> None:
    class AtomicEmbedder(QueryEmbedder):
        embedding_routes = (
            EmbeddingRoute(frozenset({Modality.TEXT})),
            EmbeddingRoute(frozenset({Modality.IMAGE}), max_assets_per_modality=1),
        )

    def plan(request: QueryRequest) -> QueryRepresentationPlan:
        value = (
            request.content if selection == "joint" else ModelInput(assets=request.content.assets)
        )
        return QueryRepresentationPlan("query", embedding_inputs=(value,))

    embedder = AtomicEmbedder()
    with Memory(tmp_path, embedder=embedder, query_policy=Policy(plan)) as memory:
        memory.add("greenhouse key")
        with pytest.raises(ModelError, match="unsupported embedding input") as caught:
            memory.search(("query", Blob(b"one", "image/png"), Blob(b"two", "image/png")))
        assert caught.value.reason == "unsupported_modality"
        assert embedder.query_batches == []


def test_optional_embedding_failure_keeps_the_required_route(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    policy = Policy(
        lambda request: QueryRepresentationPlan(
            "greenhouse",
            embedding_inputs=(ModelInput(text="greenhouse"),),
            optional_embedding_inputs=(ModelInput(text="optional failure"),),
        )
    )
    embedder = QueryEmbedder()
    with Memory(
        tmp_path,
        embedder=embedder,
        query_policy=policy,
        settings=MemorySettings(minimum_relevance=0.8, ambiguity_margin=0),
    ) as memory:
        target = memory.add("greenhouse key")
        assert memory.search("query").hits[0].id == target.id
        assert len(embedder.query_batches) == 2
        assert "optional query embedding failed: model_failed" in caplog.text
        assert "private provider body" not in caplog.text


@pytest.mark.parametrize("failure", ["wrong_type", "invalid_value", "oversized", "raises"])
def test_invalid_query_policy_output_is_safe_and_never_reaches_embedding(
    tmp_path: Path, failure: str
) -> None:
    def plan(request: QueryRequest) -> QueryRepresentationPlan:
        if failure == "wrong_type":
            return cast(QueryRepresentationPlan, "private policy body")
        if failure == "invalid_value":
            return QueryRepresentationPlan(cast(str, None))
        if failure == "oversized":
            return QueryRepresentationPlan("x" * 65_537)
        raise RuntimeError("private policy body")

    embedder = QueryEmbedder()
    with Memory(tmp_path, embedder=embedder, query_policy=Policy(plan)) as memory:
        memory.add("greenhouse key")
        with pytest.raises(ModelError) as caught:
            memory.search("query")
        assert caught.value.stage == "query.plan"
        assert "private policy body" not in str(caught.value)
        assert embedder.query_batches == []


def test_selected_bare_image_routes_keep_query_provenance(tmp_path: Path) -> None:
    policy = Policy(
        lambda request: QueryRepresentationPlan(
            request.focused_text,
            embedding_inputs=tuple(ModelInput(assets=(asset,)) for asset in request.content.assets),
        )
    )
    backend = VisualBackend()
    with Memory(
        tmp_path,
        embedder=backend,
        answerer=backend,
        query_policy=policy,
        settings=MemorySettings(recall_planning=False),
    ) as memory:
        target = memory.add(("Original name", Blob(b"near", "image/png", "near.png")))
        memory.ask(("Which object?", Blob(b"query", "image/png", "query.png")))
        matches = backend.questions[-1].image_retrieval_matches
        assert {match.memory_id for match in matches} == {target.id}
        assert {match.query_asset_id for match in matches} == {
            policy.requests[0].content.assets[0].id
        }
