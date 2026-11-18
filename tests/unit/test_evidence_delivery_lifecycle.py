"""Public-SDK acceptance of native evidence and reversible source withdrawal."""

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AsyncEventStream,
    AsyncMemory,
    Blob,
    ContextBudget,
    EventChunk,
    EventObservation,
    EventSegment,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemorySettings,
    Modality,
    ModelError,
    ModelInput,
    RecordKind,
    SearchHit,
    StreamInput,
)


class TextEmbedder(TinyEmbedder):
    embedding_capabilities = frozenset({Modality.TEXT})


class ImageEmbedder(TinyEmbedder):
    embedding_capabilities = frozenset({Modality.IMAGE})


class Segmenter:
    event_model = "delivery-event-fixture"
    event_recipe = "delivery-event-fixture:v1"

    def segment_events(
        self, observations: Sequence[EventObservation], *, end_of_stream: bool = False
    ) -> tuple[EventSegment, ...]:
        return (EventSegment(len(observations), "The technician examines a pump."),)


@pytest.mark.asyncio
async def test_native_only_models_can_ignore_an_event_interpretation(tmp_path: Path) -> None:
    reader = Reader(text=False)
    reference = datetime(2026, 11, 16, tzinfo=timezone.utc)

    async def chunks() -> AsyncIterator[EventChunk]:
        yield EventChunk(
            Blob(b"event image", "image/png"), reference, reference + timedelta(seconds=1)
        )

    async with AsyncMemory(
        Memory(
            tmp_path,
            embedder=ImageEmbedder(),
            answerer=reader,
            settings=MemorySettings(minimum_relevance=0),
        )
    ) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(memory, segmenter=Segmenter()).consume(chunks())
        ]
        assert len(commits) == 1
        assert any(part.kind == "model_interpretation" for part in commits[0].record.parts)
        result = await memory.ask(Blob(b"event query", "image/png"))
        assert commits[0].record.id in {hit.id for hit in result.hits}
        assert reader.received[-1][0].assets
        assert reader.received[-1][0].content == ""


class Describer:
    vision_capabilities = frozenset({Modality.IMAGE})
    vision_model = "delivery-caption-fixture"
    vision_space = "delivery-caption-fixture:v1"

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        return ("An equipment label is visible.",) * len(inputs)

    def close(self) -> None:
        pass


class Reader:
    def __init__(self, *, text: bool = True) -> None:
        self.generation_capabilities = frozenset(
            {Modality.TEXT, Modality.IMAGE} if text else {Modality.IMAGE}
        )
        self.received: list[tuple[SearchHit, ...]] = []

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        self.received.append(tuple(hits))
        return AnswerResult(answer="Evidence received.", hits=tuple(hits))

    def close(self) -> None:
        pass


@pytest.mark.parametrize("caption_only", (False, True))
def test_a_caption_does_not_authorize_discarding_the_recorded_words(
    tmp_path: Path, caption_only: bool
) -> None:
    reader = Reader(text=False)
    with Memory(
        tmp_path,
        embedder=TextEmbedder(),
        answerer=reader,
        vision_describer=Describer(),
        settings=MemorySettings(query_vision=True, minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        image = Blob(b"equipment source", "image/png")
        content = image if caption_only else ("The technician recorded a pressure of eight.", image)
        record = memory.capture(content)
        assert memory.settle() == 1
        assert record.id in {row.id for row in memory.list().items}

    with Memory(
        tmp_path,
        embedder=TextEmbedder(),
        answerer=reader,
        vision_describer=Describer(),
        settings=MemorySettings(query_vision=True, minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        query = Blob(b"equipment query", "image/png")
        if caption_only:
            result = memory.ask(query)
            assert record.id in {hit.id for hit in result.hits}
            assert reader.received[-1][0].assets
        else:
            with pytest.raises(ModelError) as failure:
                memory.ask(query)
            assert failure.value.reason == "unsupported_modality"
            assert reader.received == []


@pytest.mark.parametrize(
    ("subject", "attribute", "value"),
    (("Nora", "lamp_code", "L-71"), ("Omar", "pump_capacity", "eight litres")),
)
@pytest.mark.parametrize("selected_proofs", (False, True))
def test_multimodal_retained_evidence_can_be_withdrawn_and_restored_after_reopening(
    tmp_path: Path, subject: str, attribute: str, value: str, selected_proofs: bool
) -> None:
    reader = Reader()
    config = MemorySettings(minimum_relevance=0, ambiguity_margin=0)
    budget = ContextBudget(selected_proofs=selected_proofs)
    with Memory(
        tmp_path,
        embedder=TextEmbedder(),
        answerer=reader,
        vision_describer=Describer(),
        settings=config,
    ) as memory:
        source = memory.capture(
            (
                f"{subject} recorded:",
                Blob(b"equipment evidence", "image/png"),
                f"{attribute}: {value}",
            )
        )
        assert memory.settle() == 1
        operation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                target_ids=(source.id,),
                proposal=FormationProposal(
                    kind=RecordKind.RELATION,
                    content=f"{subject}'s {attribute} is {value}.",
                    subject=subject,
                    predicate=attribute,
                    value=value,
                    confidence=0.8,
                ),
            )
        )
        claim_id = operation.created_ids[0]

    with Memory(
        tmp_path,
        embedder=TextEmbedder(),
        answerer=reader,
        vision_describer=Describer(),
        settings=config,
    ) as memory:
        for hits in (memory.compile(attribute, budget=budget).hits, memory.ask(attribute).hits):
            assert {claim_id, source.id} <= {hit.id for hit in hits}
            delivered = next(hit for hit in hits if hit.id == source.id)
            assert delivered.parts == memory.get(source.id).parts
            assert (
                memory.read_asset(source.id, delivered.assets[0].id).data == b"equipment evidence"
            )
        withdrawal = memory.forget((source.id,))
        assert withdrawal is not None

    with Memory(
        tmp_path,
        embedder=TextEmbedder(),
        answerer=reader,
        vision_describer=Describer(),
        settings=config,
    ) as memory:
        assert claim_id not in {hit.id for hit in memory.compile(attribute, budget=budget).hits}
        assert claim_id not in {hit.id for hit in memory.ask(attribute).hits}
        assert memory.rollback(withdrawal.operation_id)
        assert {claim_id, source.id} <= {
            hit.id for hit in memory.compile(attribute, budget=budget).hits
        }


@pytest.mark.parametrize(
    ("chain", "withdrawal"),
    [(chain, withdrawal) for chain in (False, True) for withdrawal in (None, "source", "claim")]
    + [(True, "intermediate")],
)
def test_source_replay_preserves_only_retired_evidence_with_a_live_dependent(
    tmp_path: Path, chain: bool, withdrawal: str | None
) -> None:
    config = MemorySettings(minimum_relevance=0)
    with Memory(tmp_path / "source", embedder=TinyEmbedder(), settings=config) as memory:
        original = memory.add(
            ("Iris's component:", Blob(b"source component", "image/png"), "serial C-83"),
        )
        parent = original.id
        claims = []
        for depth in range(2 if chain else 1):
            operation = memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(parent,),
                    target_ids=(parent,),
                    proposal=FormationProposal(
                        kind=RecordKind.RELATION,
                        content=f"Iris's component has serial C-83, assessment {depth}.",
                        subject="Iris",
                        predicate=f"component_record_{depth}",
                        value="C-83",
                        confidence=0.8,
                    ),
                )
            )
            parent = operation.created_ids[0]
            claims.append(parent)
        if withdrawal == "source":
            changed = memory.forget((original.id,))
        elif withdrawal in {"intermediate", "claim"}:
            changed = memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.CORRECT,
                    target_ids=(claims[0] if withdrawal == "intermediate" else parent,),
                )
            )
        else:
            changed = None

    with (
        Memory(tmp_path / "source", embedder=TinyEmbedder(), settings=config) as memory,
        Memory(tmp_path / "destination", embedder=TinyEmbedder(), settings=config) as destination,
    ):
        migrated = tuple(memory.replay_sources(destination))
        assert {record.id for record in migrated} == (
            {original.id} if withdrawal is None else set()
        )
        if migrated:
            assert migrated[0].parts == original.parts
            assert (
                destination.read_asset(original.id, original.assets[0].id).data
                == b"source component"
            )
            assert destination.operations() == ()
        else:
            assert changed is not None
            assert memory.rollback(changed.operation_id)
            assert {record.id for record in memory.replay_sources(destination)} == {original.id}


@pytest.mark.parametrize("failure", ["other_parent", "missing_conjunct"])
@pytest.mark.parametrize("capture", [False, True])
def test_source_replay_requires_a_complete_legally_retained_proof(
    tmp_path: Path, failure: str, capture: bool
) -> None:
    config = MemorySettings(minimum_relevance=0)
    content = ("Iris's component:", Blob(b"source component", "image/png"), "serial C-83")
    with Memory(tmp_path / "source", embedder=TinyEmbedder(), settings=config) as memory:
        original = next(
            memory.add_stream([StreamInput(content, interpretation="Iris inspected C-83.")])
        )
        evidence_ids: tuple[str, ...]
        if failure == "other_parent":
            memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(original.id,),
                    proposal=FormationProposal(
                        kind=RecordKind.RELATION,
                        content="Iris has another assessment of C-83.",
                        subject="Iris",
                        predicate="other_assessment",
                        value="C-83",
                        confidence=0.8,
                    ),
                )
            )
            evidence_ids = (original.id,)
        else:
            companion = memory.add("The C-83 inspection meter read 70.")
            evidence_ids = (original.id, companion.id)
        consolidated = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=evidence_ids,
                target_ids=(original.id,),
                proposal=FormationProposal(
                    kind=RecordKind.RELATION,
                    content="Iris's C-83 inspection is complete.",
                    subject="Iris",
                    predicate="inspection",
                    value="C-83",
                    confidence=0.8,
                ),
            )
        )
        withdrawal = (
            memory.apply(
                MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=consolidated.created_ids)
            )
            if failure == "other_parent"
            else memory.forget((companion.id,))
        )
        assert withdrawal is not None
        refreshed = next(
            memory.add_stream(
                [StreamInput(content, interpretation="Iris examined C-83 carefully.")],
                capture=capture,
            )
        )
        assert refreshed.id == original.id and refreshed.forgotten_at is not None
        if capture:
            assert memory.settle() == 1
        assert memory.compile("C-83").hits == ()

    with (
        Memory(tmp_path / "source", embedder=TinyEmbedder(), settings=config) as memory,
        Memory(tmp_path / "destination", embedder=TinyEmbedder()) as destination,
    ):
        assert tuple(memory.replay_sources(destination)) == ()
        assert memory.rollback(withdrawal.operation_id)
        replayed = {record.id: record for record in memory.replay_sources(destination)}
        assert original.id in replayed
        assert all(part.kind in {"text", "asset"} for part in replayed[original.id].parts)
        assert (
            destination.read_asset(original.id, original.assets[0].id).data == b"source component"
        )
