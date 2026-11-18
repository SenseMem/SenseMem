"""Attribution identity follows grounded people and episodes, not display names."""

from dataclasses import replace
from datetime import datetime, timezone

from sensemem import (
    AffectLink,
    AffectObjectRef,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    MemoryRecord,
    ObservationContext,
    RecordKind,
    RecordSemantics,
)
from sensemem.kernel.affect import resolve_affect_proposal
from sensemem.kernel.formation_identity import history_identity_refusal
from sensemem.kernel.formation_rules import formation_memory_id
from sensemem.models.base import ModelInput

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
