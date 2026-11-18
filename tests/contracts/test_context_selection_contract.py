"""Applications select context through public values without replacing evidence admission."""

from collections.abc import Sequence
from pathlib import Path

from sensemem import (
    AssetRef,
    Blob,
    ConsentState,
    ContextBudget,
    ContextSelection,
    ContextSelectionRequest,
    ContextSelector,
    ContextUnknownKind,
    DefaultContextSelector,
    EmbedTask,
    FaceAnalysis,
    FaceEmbedding,
    Memory,
    MemoryPlugins,
    MemorySettings,
    Modality,
    ModelInput,
    ObservationContext,
    RecordKind,
    RetrievalScope,
)


class ContextEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    embedding_model = embedding_space = "context-selection-fixture-v1"
    embedding_dimension = 2

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _value in inputs)

    def close(self) -> None:
        pass


class GoalSelector:
    def __init__(self) -> None:
        self.requests: list[ContextSelectionRequest] = []
        self.closed = 0

    def select(self, request: ContextSelectionRequest) -> ContextSelection:
        self.requests.append(request)
        return ContextSelection(
            tuple(
                candidate.hit.id
                for candidate in request.candidates
                if candidate.hit.metadata.get("topic") == request.goal
            )
        )

    def close(self) -> None:
        self.closed += 1


class PersonFace:
    face_capabilities = frozenset({Modality.IMAGE})
    face_model = "context-selection-face-fixture-v1"
    face_space = "context-selection-face-fixture:2:v1"
    face_analysis_space = "context-selection-face-analysis:v1"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        return tuple(
            FaceAnalysis((FaceEmbedding("face", (0.0, 1.0), (0.1, 0.1, 0.4, 0.5), None),))
            for _asset in assets
        )

    def close(self) -> None:
        pass


def test_public_selector_uses_goal_and_hits_and_is_owned_by_memory(tmp_path: Path) -> None:
    selector = GoalSelector()
    assert isinstance(selector, ContextSelector)
    assert isinstance(DefaultContextSelector(), ContextSelector)
    with Memory.from_plugins(
        tmp_path,
        plugins=MemoryPlugins(embedder=ContextEmbedder(), context_selector=selector),
        settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        tea = memory.add("Tea is stored in the kitchen.", metadata={"topic": "tea"})
        coffee = memory.add("Coffee is stored in the office.", metadata={"topic": "coffee"})
        for goal, expected in (("tea", tea), ("coffee", coffee)):
            bundle = memory.compile(goal, budget=ContextBudget(max_items=1))
            assert tuple(hit.id for hit in bundle.hits) == (expected.id,)
            request = selector.requests[-1]
            assert request.goal == goal
            assert {candidate.hit.id for candidate in request.candidates} == {tea.id, coffee.id}
            assert request.reserved_chars > 0
            assert all(
                candidate.hit.id in candidate.required_ids
                and candidate.closure_chars >= candidate.chars
                for candidate in request.candidates
            )
            assert bundle.chars <= bundle.budget.max_chars
    assert selector.closed == 1


def test_selector_receives_only_the_requested_place_scope(tmp_path: Path) -> None:
    selector = GoalSelector()
    with Memory(
        tmp_path,
        embedder=ContextEmbedder(),
        context_selector=selector,
        settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        visible = memory.add(
            "Tea is in the kitchen.",
            metadata={"topic": "tea"},
            context=ObservationContext(place_id="home"),
        )
        hidden = memory.add(
            "Tea is in the cupboard.",
            metadata={"topic": "tea"},
            context=ObservationContext(place_id="office"),
        )
        bundle = memory.compile("tea", scope=RetrievalScope(place_id="home"))
        assert tuple(hit.id for hit in bundle.hits) == (visible.id,)
        assert {candidate.hit.id for candidate in selector.requests[-1].candidates} == {visible.id}
        assert hidden.id not in bundle.render()


def test_selector_can_deliberately_return_no_context(tmp_path: Path) -> None:
    selector = GoalSelector()
    with Memory(
        tmp_path,
        embedder=ContextEmbedder(),
        context_selector=selector,
        settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add("The office key is blue.", metadata={"topic": "key"})
        bundle = memory.compile("tea")
        assert not bundle.hits and bundle.omitted == 1
        assert any(
            unknown.kind is ContextUnknownKind.EVIDENCE_UNAVAILABLE
            and "excluded by the context selector" in unknown.detail
            for unknown in bundle.unknowns
        )
        assert not any(
            unknown.kind is ContextUnknownKind.BUDGET_EXCLUDED for unknown in bundle.unknowns
        )


def test_consent_is_applied_before_the_selector_sees_naming_assertions(tmp_path: Path) -> None:
    class AllCandidates(GoalSelector):
        def select(self, request: ContextSelectionRequest) -> ContextSelection:
            self.requests.append(request)
            return ContextSelection(tuple(candidate.hit.id for candidate in request.candidates))

    selector = AllCandidates()
    with Memory(
        tmp_path,
        embedder=ContextEmbedder(),
        face_analyzer=PersonFace(),
        context_selector=selector,
        settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        photo = memory.add(Blob(b"Ann arrives", "image/png"))
        identity_id = memory.perceive(photo.id).faces[0].identity_id
        memory.register_identity(identity_id, "Ann")
        memory.compile("Who is Ann?")
        naming_ids = {
            candidate.hit.id
            for candidate in selector.requests[-1].candidates
            if candidate.hit.context is not None
            and candidate.hit.context.kind is RecordKind.ENTITY
            and candidate.hit.context.identity_id == identity_id
        }
        assert naming_ids
        memory.record_consent(identity_id, ConsentState.WITHDRAWN)
        bundle = memory.compile("Who is Ann?")
        assert naming_ids.isdisjoint(
            candidate.hit.id for candidate in selector.requests[-1].candidates
        )
        assert not bundle.actors
        assert any(
            unknown.kind is ContextUnknownKind.CONSENT_WITHHELD for unknown in bundle.unknowns
        )
