"""Executable contract for the affective-memory principles in docs/affective-memory.md.

Model-free: one scripted FormationBackend turns a small companion dialogue into typed
proposals, and every assertion is about public SDK behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _feature_support import ATOMIC_MODALITIES, CueConsolidator, TinyEmbedder

from sensemem import (
    AffectLink,
    AffectObjectRef,
    AffectWitness,
    Blob,
    ContextBudget,
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
    RecordSemantics,
    RetrievalScope,
    SpatialAnchor,
    SpatialContext,
    ValidationError,
)
from sensemem.kernel.affect import resolve_affect_proposal
from sensemem.kernel.formation_identity import history_identity_refusal
from sensemem.kernel.formation_rules import formation_memory_id
from sensemem.models.base import ModelInput

MISSING_DOG = "the user's voice rose while telling me the dog Max went missing"
CORRECTION = "the user said: I wasn't angry, I was nervous about the missing dog"
SIGH = "the user sighed about the missing dog and said it was fine"
MEDICATION = "the user takes their medication lamotrigine every morning"

DOG_GOAL = "how did the user feel about the missing dog"


class CompanionFormer:
    """Scripted former: each companion turn maps to the proposals that turn licenses."""

    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "companion-test"
    formation_space = "companion-test:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(self._proposals(item.content.text, item.memory_id) for item in inputs)

    def _proposals(self, text: str, source_id: str) -> tuple[FormationProposal, ...]:
        if "went missing" in text:
            return (
                FormationProposal(kind=RecordKind.EVENT, content="the dog Max went missing"),
                FormationProposal(
                    kind=RecordKind.AFFECT,
                    content="the user seemed anxious about the missing dog",
                    subject="user",
                    value="anxious",
                    confidence=0.7,
                    cue_modality=Modality.TEXT,
                    valence=-0.6,
                    arousal=0.7,
                ),
            )
        if "wasn't angry" in text:
            return (
                FormationProposal(
                    kind=RecordKind.AFFECT,
                    content=text,
                    basis=EvidenceBasis.MODEL_INFERENCE,
                    subject="user",
                    value="nervous",
                    cue_modality=Modality.TEXT,
                    valence=-0.3,
                    arousal=0.6,
                    affect_witnesses=(
                        AffectWitness(source_id=source_id, reporter="user", quote=text),
                    ),
                ),
            )
        if "sighed" in text:
            return (
                FormationProposal(
                    kind=RecordKind.AFFECT,
                    content="the words about the missing dog read as settled",
                    subject="user",
                    value="settled",
                    confidence=0.5,
                    cue_modality=Modality.TEXT,
                    valence=0.2,
                    arousal=0.2,
                ),
                FormationProposal(
                    kind=RecordKind.AFFECT,
                    content="the voice about the missing dog sounded weary",
                    subject="user",
                    value="weary",
                    confidence=0.6,
                    cue_modality=Modality.AUDIO,
                    valence=-0.5,
                    arousal=0.3,
                ),
            )
        return ()

    def close(self) -> None:
        return None


def _memory(data_dir: Path) -> Memory:
    return Memory(
        data_dir,
        embedder=TinyEmbedder(),
        former=CompanionFormer(),
        consolidator=CueConsolidator(),
        settings=MemorySettings(minimum_relevance=0),
    )


def _records(memory: Memory, kind: RecordKind) -> tuple[MemoryRecord, ...]:
    return tuple(
        record
        for record in memory.list(limit=100).items
        if record.context is not None and record.context.kind is kind
    )


def test_a_cue_an_event_and_an_observation_stay_three_traceable_records(tmp_path: Path) -> None:
    """Affective cue, affective state, and emotional event are separate layers, each sourced."""
    with _memory(tmp_path) as memory:
        observation = memory.add(MISSING_DOG, context=ObservationContext(source_id="turn-1"))
        bundle = memory.compile(DOG_GOAL, budget=ContextBudget(max_items=8))

        (cue,) = bundle.affect
        (event,) = _records(memory, RecordKind.EVENT)
        assert cue.context is not None
        assert cue.context.evidence_ids == (observation.id,)
        assert cue.event_ids == (event.id,)
        assert event.id in {hit.id for hit in bundle.episodes}
        assert cue.context.basis is EvidenceBasis.MODEL_INFERENCE
        line = next(row for row in bundle.render().splitlines() if row.startswith(f"- [{cue.id}]"))
        assert "basis model_inference" in line
        assert "cue text" in line
        assert "valence -0.60" in line
        assert "arousal 0.70" in line


def test_a_reported_correction_coexists_with_the_inference_it_corrects(tmp_path: Path) -> None:
    """Automatic interpretation preserves the report without granting self-report authority."""
    with _memory(tmp_path) as memory:
        memory.add(MISSING_DOG, context=ObservationContext(source_id="turn-1"))
        memory.add(
            CORRECTION,
            context=ObservationContext(source_id="turn-2", basis=EvidenceBasis.USER_STATEMENT),
        )
        cues = _records(memory, RecordKind.AFFECT)
        bundle = memory.compile(DOG_GOAL, budget=ContextBudget(max_items=8))

        assert {record.context.value for record in cues if record.context} == {
            "anxious",
            "nervous",
        }
        assert all(memory.get(record.id) == record for record in cues)
        assert {hit.id for hit in bundle.affect} == {record.id for record in cues}
        # Affect is outside the conflict kinds, so two disagreeing cues are two records to read,
        # not a contradiction to reconcile.
        assert bundle.conflicts == ()
        statement = next(
            hit for hit in bundle.affect if hit.context and hit.context.value == "nervous"
        )
        assert "basis model_inference" in next(
            row for row in bundle.render().splitlines() if row.startswith(f"- [{statement.id}]")
        )
        assert statement.context is not None and statement.context.affect_witnesses


def test_two_modalities_of_one_moment_stay_two_cues(tmp_path: Path) -> None:
    """Per-modality estimates stay separate; fusion is never stored as one label."""
    with _memory(tmp_path) as memory:
        memory.add(
            (SIGH, Blob(b"sigh", "audio/wav")),
            context=ObservationContext(source_id="turn-3"),
        )
        cues = _records(memory, RecordKind.AFFECT)
        bundle = memory.compile(DOG_GOAL, budget=ContextBudget(max_items=8))

        assert {record.context.cue_modality for record in cues if record.context} == {
            Modality.TEXT,
            Modality.AUDIO,
        }
        assert {record.context.valence for record in cues if record.context} == {0.2, -0.5}
        assert {hit.id for hit in bundle.affect} == {record.id for record in cues}


def test_a_second_independent_capture_reinforces_the_same_trait_rather_than_creating_a_rival(
    tmp_path: Path,
) -> None:
    """The claim is the same, so a second capture reinforces the record instead of rivalling it."""
    with _memory(tmp_path) as memory:
        memory.add(
            (SIGH, Blob(b"sigh", "audio/wav")),
            context=ObservationContext(source_id="turn-3"),
        )
        one_capture = tuple(record.id for record in _records(memory, RecordKind.AFFECT))
        first = memory.consolidate(evidence_ids=one_capture)
        (trait_id,) = first.operations[0].created_ids

        memory.add(MISSING_DOG, context=ObservationContext(source_id="turn-1"))
        (independent,) = (
            record.id
            for record in _records(memory, RecordKind.AFFECT)
            if record.id not in one_capture
        )
        second = memory.consolidate(evidence_ids=(one_capture[0], independent))

        assert second.rejected == ()
        assert second.operations[0].changed_ids == (trait_id,)
        assert second.operations[0].created_ids == ()


def test_forgetting_a_cue_is_cognitive_and_reversible(tmp_path: Path) -> None:
    """Affect can be forgotten without erasing the audit trail, and forgetting is reversible."""
    with _memory(tmp_path) as memory:
        memory.add(MISSING_DOG, context=ObservationContext(source_id="turn-1"))
        (cue,) = _records(memory, RecordKind.AFFECT)

        operation = memory.forget((cue.id,))
        assert operation is not None
        assert memory.compile(DOG_GOAL, budget=ContextBudget(max_items=8)).affect == ()
        assert not any(hit.id == cue.id for hit in memory.search(DOG_GOAL, limit=10).hits)
        assert memory.get(cue.id).forgotten_at is not None

        assert memory.rollback(operation.operation_id) is True
        restored = memory.compile(DOG_GOAL, budget=ContextBudget(max_items=8))

        assert [hit.id for hit in restored.affect] == [cue.id]
        assert memory.get(cue.id).forgotten_at is None


def test_affect_is_not_a_retrieval_key_for_a_factual_goal(tmp_path: Path) -> None:
    """Ranking is relevance, person, time, and task; affect never buys a slot it did not earn."""
    with _memory(tmp_path) as memory:
        memory.add(MISSING_DOG, context=ObservationContext(source_id="turn-1"))
        medication = memory.add(MEDICATION, context=ObservationContext(source_id="turn-5"))
        bundle = memory.compile(
            "what medication does the user take",
            budget=ContextBudget(max_items=1),
        )

        assert [hit.id for hit in bundle.hits] == [medication.id]
        assert bundle.affect == ()


_OFFICE = SpatialContext(frame_id="home", anchor=SpatialAnchor.OBSERVER, x=0, y=0, z=0)
_KITCHEN = SpatialContext(frame_id="home", anchor=SpatialAnchor.OBSERVER, x=20, y=0, z=0)


class _Former:
    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "episode-scope-fixture"
    formation_space = "episode-scope-fixture:v1"

    def __init__(self, *, manual: bool) -> None:
        self.manual = manual
        self.affect_id = "proposal:cue"
        self.event_id = "proposal:event"

    def denial(self) -> FormationProposal:
        return FormationProposal(
            kind=RecordKind.RELATION,
            content="The user denies that cancellation caused the earlier anger.",
            subject="user",
            predicate="affect_elicitor",
            value="cancellation",
            affect_link=AffectLink(
                affect_id=self.affect_id,
                role="elicitor",
                object_ref=AffectObjectRef(memory_id=self.event_id),
                polarity="denied",
            ),
        )

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (() if self.manual else (self.denial(),))
            if source.content.text == "clarification"
            else (
                FormationProposal(
                    kind=RecordKind.AFFECT,
                    local_id="cue",
                    content="The user seems angry.",
                    subject="user",
                    value="angry",
                    cue_modality=Modality.TEXT,
                ),
                FormationProposal(
                    kind=RecordKind.EVENT,
                    local_id="event",
                    content="The activity was canceled.",
                ),
                FormationProposal(
                    kind=RecordKind.RELATION,
                    content="Cancellation may explain the user's anger.",
                    subject="user",
                    predicate="affect_elicitor",
                    value="cancellation",
                    affect_link=AffectLink(
                        affect_id="proposal:cue",
                        role="elicitor",
                        object_ref=AffectObjectRef(memory_id="proposal:event"),
                    ),
                ),
            )
            for source in inputs
        )

    def close(self) -> None:
        pass


@pytest.mark.parametrize("manual", (False, True), ids=("formation", "consolidation"))
@pytest.mark.parametrize(
    "scope",
    (RetrievalScope(place_id="office"), RetrievalScope(near=_OFFICE, radius_m=1)),
    ids=("place", "near"),
)
def test_later_attributions_keep_the_original_episode_scope(
    tmp_path: Path, manual: bool, scope: RetrievalScope
) -> None:
    former = _Former(manual=manual)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(
            minimum_relevance=0,
            formation_history_max_rows=32,
            formation_history_budget_chars=16_000,
        ),
    ) as memory:
        memory.add(
            "The activity was canceled and I am angry.",
            context=ObservationContext(place_id="office", spatial=_OFFICE),
        )
        records = memory.list(limit=20).items
        former.affect_id = next(
            record.id
            for record in records
            if record.context and record.context.kind is RecordKind.AFFECT
        )
        former.event_id = next(
            record.id
            for record in records
            if record.context and record.context.kind is RecordKind.EVENT
        )
        clarification = memory.add(
            "clarification", context=ObservationContext(place_id="kitchen", spatial=_KITCHEN)
        )
        if manual:
            for omitted in (former.affect_id, former.event_id):
                with pytest.raises(ValidationError, match="invalid_proposal"):
                    memory.apply(
                        MemoryOperation(
                            intent=MemoryIntent.CONSOLIDATE,
                            evidence_ids=tuple(
                                memory_id
                                for memory_id in (
                                    former.affect_id,
                                    former.event_id,
                                    clarification.id,
                                )
                                if memory_id != omitted
                            ),
                            proposal=former.denial(),
                        )
                    )
            memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(former.affect_id, former.event_id, clarification.id),
                    proposal=former.denial(),
                )
            )

        links = tuple(
            hit
            for hit in memory.search("anger", limit=20, scope=scope).hits
            if hit.context and hit.context.affect_link
        )
        assert {
            hit.context.affect_link.polarity
            for hit in links
            if hit.context and hit.context.affect_link
        } == {
            "affirmed",
            "denied",
        }
        assert all(hit.place_id == "office" for hit in links)
        assert all(hit.context and hit.context.spatial == _OFFICE for hit in links)

        bundle = memory.compile(
            "why was the user angry", scope=scope, budget=ContextBudget(max_items=20)
        )
        delivered = {
            hit.context.affect_link.polarity
            for hit in bundle.relationships
            if hit.context and hit.context.affect_link
        }
        assert delivered in (set(), {"affirmed", "denied"})


_NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)


def _record(
    identifier: str,
    kind: RecordKind,
    *,
    subject: str | None = None,
    identity_id: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=identifier,
        content=identifier,
        created_at=_NOW,
        context=RecordSemantics(
            kind=kind,
            basis=EvidenceBasis.MODEL_INFERENCE,
            confidence=0.8,
            valid_from=None,
            valid_until=None,
            recorded_at=_NOW,
            subject=subject,
            identity_id=identity_id,
        ),
    )


def test_a_renamed_experiencer_requires_one_cited_identity_witness() -> None:
    affect = _record("cue", RecordKind.AFFECT, subject="Ana", identity_id="person")
    event = _record("event", RecordKind.EVENT)
    naming = _record("name", RecordKind.ENTITY, subject="Alice", identity_id="person")
    source = _record("source", RecordKind.OBSERVATION)
    records = {record.id: record for record in (affect, event, naming, source)}
    proposal = FormationProposal(
        kind=RecordKind.RELATION,
        content="Alice attributes the earlier feeling to the event.",
        subject="Alice",
        predicate="affect_elicitor",
        value="event",
        evidence_ids=("source", "name"),
        affect_link=AffectLink(
            affect_id="cue", role="elicitor", object_ref=AffectObjectRef(memory_id="event")
        ),
    )
    resolved = resolve_affect_proposal(proposal, records, {})
    assert resolved is not None
    offered = FormationInput(
        memory_id="source",
        content=ModelInput(text="a clarification"),
        context=ObservationContext(),
        history=(naming,),
    )
    assert not history_identity_refusal(
        resolved, offered, tuple(records[key] for key in resolved.evidence_ids or ())
    )
    uncited = resolve_affect_proposal(replace(proposal, evidence_ids=("source",)), records, {})
    assert uncited is not None
    assert history_identity_refusal(
        uncited, offered, tuple(records[key] for key in uncited.evidence_ids or ())
    )

    ambiguous = _record("other-name", RecordKind.ENTITY, subject="Alice", identity_id="other")
    assert resolve_affect_proposal(proposal, {**records, ambiguous.id: ambiguous}, {}) is None
    assert (
        resolve_affect_proposal(proposal, {"cue": affect, "event": event, "source": source}, {})
        is None
    )


def test_display_names_do_not_fork_a_link_but_another_episode_does() -> None:
    proposal = FormationProposal(
        kind=RecordKind.RELATION,
        content="The user is upset with Bob.",
        subject="user",
        predicate="affect_target",
        value="Bob",
        affect_link=AffectLink(
            affect_id="cue", role="target", object_ref=AffectObjectRef(identity_id="bob-id")
        ),
    )
    context = RecordSemantics(
        kind=RecordKind.RELATION,
        basis=EvidenceBasis.MODEL_INFERENCE,
        confidence=0.8,
        valid_from=None,
        valid_until=None,
        recorded_at=_NOW,
        subject="user",
        predicate="affect_target",
        value="Bob",
        affect_link=proposal.affect_link,
    )
    original = formation_memory_id("source", proposal, recipe="test", context=context)
    renamed = formation_memory_id(
        "source",
        replace(proposal, value="Robert"),
        recipe="test",
        context=replace(context, value="Robert"),
    )
    assert renamed == original
    assert proposal.affect_link is not None
    later_link = replace(proposal.affect_link, affect_id="another-cue")
    later = formation_memory_id(
        "source",
        replace(proposal, affect_link=later_link),
        recipe="test",
        context=replace(context, affect_link=later_link),
    )
    assert later != original
