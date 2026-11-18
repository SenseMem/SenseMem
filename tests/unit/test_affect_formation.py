"""Form emotional associations over actual cues, without inventing independent evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    AffectLink,
    AffectObjectRef,
    AffectWitness,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemoryRecord,
    MemorySettings,
    Modality,
    ObservationContext,
    RecordKind,
)


class Former:
    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "affect-grounding-fixture"
    formation_space = "affect-grounding-fixture:v1"

    def __init__(self) -> None:
        self.bad_subject = False
        self.bad_reference = False
        self.bad_quote = False
        self.cue_id = ""
        self.event_id = ""

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(self._one(value) for value in inputs)

    def _one(self, source: FormationInput) -> tuple[FormationProposal, ...]:
        if source.content.text == "clarification":
            return (
                FormationProposal(
                    kind=RecordKind.RELATION,
                    content="The user denies that cancellation caused the earlier anger.",
                    subject="user",
                    predicate="affect_elicitor",
                    value="cancellation",
                    affect_link=AffectLink(
                        affect_id=self.cue_id,
                        role="elicitor",
                        object_ref=AffectObjectRef(memory_id=self.event_id),
                        polarity="denied",
                    ),
                ),
            )
        cue = FormationProposal(
            kind=RecordKind.AFFECT,
            local_id="cue",
            content="The user is angry about Bob canceling the activity.",
            subject="user",
            value="angry",
            cue_modality=Modality.TEXT,
            valence=-0.5,
            arousal=0.6,
            affect_witnesses=(
                AffectWitness(
                    source_id=source.memory_id,
                    reporter="user",
                    quote="invented" if self.bad_quote else source.content.text,
                ),
            ),
        )
        return (
            FormationProposal(kind=RecordKind.ENTITY, local_id="bob", content="Bob", subject="Bob"),
            FormationProposal(
                kind=RecordKind.EVENT,
                local_id="event",
                content="Bob canceled the activity.",
                subject="Bob",
            ),
            cue,
            *(
                FormationProposal(
                    kind=RecordKind.RELATION,
                    content=f"The user's anger has a sourced {role}.",
                    subject="someone else" if self.bad_subject else "user",
                    predicate=f"affect_{role}",
                    value="Bob" if role == "target" else "cancellation",
                    affect_link=AffectLink(
                        affect_id="proposal:missing" if self.bad_reference else "proposal:cue",
                        role=role,
                        object_ref=AffectObjectRef(
                            memory_id="proposal:bob" if role == "target" else "proposal:event"
                        ),
                    ),
                )
                for role in ("target", "elicitor")
            ),
        )

    def close(self) -> None:
        pass


def _records(memory: Memory, kind: RecordKind) -> tuple[MemoryRecord, ...]:
    return tuple(
        item
        for item in memory.list(limit=100).items
        if item.context is not None and item.context.kind is kind
    )


def test_local_references_commit_with_their_premises_and_retract_together(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=Former()) as memory:
        source = memory.add("Bob canceled the activity and I am angry.")
        (cue,) = _records(memory, RecordKind.AFFECT)
        links = _records(memory, RecordKind.RELATION)
        assert len(links) == 2
        for record in links:
            assert record.context is not None
            link = record.context.affect_link
            assert link is not None
            assert link.affect_id == cue.id
            assert not link.affect_id.startswith("proposal:")
            assert {source.id, cue.id, link.object_ref.memory_id} <= set(
                record.context.evidence_ids
            )
            if link.role == "elicitor":
                assert record.context.value == "Bob canceled the activity."
        operation = memory.apply(MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(cue.id,)))
        assert memory.get(source.id).content == source.content
        assert not any(
            hit.context is not None and hit.context.affect_link is not None
            for hit in memory.search("anger", limit=20).hits
        )
        assert memory.rollback(operation.operation_id)
        assert {item.id for item in _records(memory, RecordKind.RELATION)} == {
            item.id for item in links
        }


@pytest.mark.parametrize("failure", ("bad_subject", "bad_reference", "bad_quote"))
def test_bad_link_or_witness_does_not_cost_the_observation(tmp_path: Path, failure: str) -> None:
    former = Former()
    setattr(former, failure, True)
    with Memory(tmp_path, embedder=TinyEmbedder(), former=former) as memory:
        source = memory.add("Bob canceled the activity and I am angry.")
        assert memory.get(source.id) == source
        assert _records(memory, RecordKind.EVENT)
        assert _records(memory, RecordKind.RELATION) == ()
        assert bool(_records(memory, RecordKind.AFFECT)) is (failure != "bad_quote")


def test_later_denial_keeps_the_original_episode_time(tmp_path: Path) -> None:
    former = Former()
    first_time = datetime.now(timezone.utc) - timedelta(hours=1)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(
            formation_history_max_rows=32, formation_history_budget_chars=16_000
        ),
    ) as memory:
        memory.add("Bob canceled the activity and I am angry.", occurred_at=first_time)
        former.cue_id = _records(memory, RecordKind.AFFECT)[0].id
        former.event_id = _records(memory, RecordKind.EVENT)[0].id
        source = memory.add("clarification")
        links = _records(memory, RecordKind.RELATION)
        denied = next(
            record
            for record in links
            if record.context is not None
            and record.context.affect_link is not None
            and record.context.affect_link.polarity == "denied"
        )
        assert denied.context is not None
        assert denied.occurred_at == denied.context.valid_from == first_time
        assert source.id in denied.context.evidence_ids
        assert denied.context.recorded_at >= source.created_at
        assert len(_records(memory, RecordKind.AFFECT)) == 1


def test_unwitnessed_self_report_is_refused_at_the_kernel_boundary(tmp_path: Path) -> None:
    class Unwitnessed(Former):
        def _one(self, source: FormationInput) -> tuple[FormationProposal, ...]:
            proposals = super()._one(source)
            return (replace(proposals[2], basis=EvidenceBasis.USER_STATEMENT, affect_witnesses=()),)

    with Memory(tmp_path, embedder=TinyEmbedder(), former=Unwitnessed()) as memory:
        source = memory.add(
            "I am angry.", context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT)
        )
        assert memory.get(source.id) == source
        assert _records(memory, RecordKind.AFFECT) == ()


def test_a_model_cannot_make_a_third_party_report_authoritative(tmp_path: Path) -> None:
    quote = "Ana says Bob is sad because the launch was canceled."

    class ForgedReporter(Former):
        def _one(self, source: FormationInput) -> tuple[FormationProposal, ...]:
            return (
                FormationProposal(
                    kind=RecordKind.AFFECT,
                    content=quote,
                    subject="Bob",
                    value="sad",
                    cue_modality=Modality.TEXT,
                    basis=EvidenceBasis.USER_STATEMENT,
                    affect_witnesses=(
                        AffectWitness(source_id=source.memory_id, reporter="Bob", quote=quote),
                    ),
                ),
            )

    with Memory(tmp_path, embedder=TinyEmbedder(), former=ForgedReporter()) as memory:
        source = memory.add(quote, context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT))
        assert memory.get(source.id).content == quote
        assert _records(memory, RecordKind.AFFECT) == ()


def test_the_host_can_explicitly_assert_a_witnessed_self_report(tmp_path: Path) -> None:
    quote = "Ana says I feel sad."
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        source = memory.add(quote, context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT))
        operation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=FormationProposal(
                    kind=RecordKind.AFFECT,
                    content=quote,
                    subject="Ana",
                    value="sad",
                    cue_modality=Modality.TEXT,
                    basis=EvidenceBasis.USER_STATEMENT,
                    affect_witnesses=(
                        AffectWitness(source_id=source.id, reporter="Ana", quote=quote),
                    ),
                ),
            )
        )
        (record_id,) = operation.created_ids
        context = memory.get(record_id).context
        assert context is not None and context.basis is EvidenceBasis.USER_STATEMENT
        assert context.affect_witnesses[0].reporter == "Ana"
