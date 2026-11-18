"""Pure grounding of emotional targets and elicitors in the evidence actually supplied."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace

from sensemem.kernel.predicates import normalize_label
from sensemem.kernel.temporal import intervals_overlap
from sensemem.records import MemoryRecord
from sensemem.semantics import AffectObjectRef, FormationProposal
from sensemem.types import EvidenceBasis, RecordKind


def resolve_affect_proposal(
    proposal: FormationProposal,
    records: Mapping[str, MemoryRecord],
    local_ids: Mapping[str, str],
) -> FormationProposal | None:
    """Resolve only a link's offered references, adding its necessary support premises.

    The focal cue and an unbound object are premises, not independent corroboration. A named
    identity is stable across ordinary naming versions, so its naming row is a resolution
    witness rather than a standing premise that a rename would withdraw.
    """
    link = proposal.affect_link
    if link is None:
        return proposal

    affect_id = local_ids.get(link.affect_id, link.affect_id)
    affect = records.get(affect_id)
    if (
        affect is None
        or affect.context is None
        or affect.context.kind is not RecordKind.AFFECT
        or not affect.context.visible
        or affect.context.retired_at is not None
        or affect.forgotten_at is not None
    ):
        return None
    subject = normalize_label(proposal.subject)
    named_identities = {
        record.context.identity_id
        for record in records.values()
        if record.context is not None
        and record.context.kind is RecordKind.ENTITY
        and record.context.identity_id is not None
        and normalize_label(record.context.subject) == subject
        and record.context.visible
        and record.context.retired_at is None
        and record.forgotten_at is None
    }
    if (
        affect.context.identity_id is not None
        and named_identities
        and named_identities != {affect.context.identity_id}
    ):
        return None
    if subject != normalize_label(affect.context.subject) and (
        affect.context.identity_id is None or not named_identities
    ):
        return None

    premises = [affect_id]
    object_ref = link.object_ref
    if object_ref.memory_id is not None:
        object_id = local_ids.get(object_ref.memory_id, object_ref.memory_id)
        target = records.get(object_id)
        if (
            target is None
            or target.context is None
            or target.context.kind not in {RecordKind.ENTITY, RecordKind.EVENT}
            or (link.role == "elicitor" and target.context.kind is not RecordKind.EVENT)
            or not target.context.visible
            or target.context.retired_at is not None
            or target.forgotten_at is not None
        ):
            return None
        if target.context.kind is RecordKind.ENTITY and target.context.identity_id is not None:
            object_ref = AffectObjectRef(identity_id=target.context.identity_id)
        else:
            object_ref = AffectObjectRef(memory_id=object_id)
            premises.append(object_id)
        value = (
            target.content
            if target.context.kind is RecordKind.EVENT
            else target.context.subject or target.content
        )
    else:
        target = next(
            (
                record
                for record in records.values()
                if record.context is not None
                and record.context.identity_id == object_ref.identity_id
                and record.context.kind is RecordKind.ENTITY
                and record.context.visible
                and record.context.retired_at is None
                and record.forgotten_at is None
            ),
            None,
        )
        if target is None:
            return None
        assert target.context is not None
        value = target.context.subject or target.content

    start = proposal.valid_from or affect.context.valid_from or affect.occurred_at
    end = proposal.valid_until or affect.context.valid_until
    if not intervals_overlap(start, end, affect.context.valid_from, affect.context.valid_until):
        return None
    evidence = tuple(dict.fromkeys((*(proposal.evidence_ids or ()), *premises)))
    return replace(
        proposal,
        value=value,
        valid_from=start,
        valid_until=end,
        affect_link=replace(link, affect_id=affect_id, object_ref=object_ref),
        evidence_ids=evidence,
    )


def affect_witness_refusal(
    proposal: FormationProposal,
    records: Mapping[str, MemoryRecord],
    *,
    host_authored: bool = False,
) -> bool:
    """Check exact original-text witnesses without treating a quote as causal proof."""
    for witness in proposal.affect_witnesses:
        source = records.get(witness.source_id)
        if (
            source is None
            or (source.context is not None and source.context.kind is not RecordKind.OBSERVATION)
            or witness.source_id not in (proposal.evidence_ids or ())
            or source.forgotten_at is not None
        ):
            return True
        originals = (
            tuple(part.value for part in source.parts if part.kind == "text")
            if source.parts
            else (source.content,)
        )
        if not any(witness.quote in text for text in originals):
            return True
    if proposal.basis is EvidenceBasis.USER_STATEMENT and (
        proposal.kind is RecordKind.AFFECT or proposal.affect_link is not None
    ):
        # An exact quotation proves provenance, not who spoke about whom. Only an explicit
        # host assertion can authorize self-report status; model-written reporter names cannot.
        return not host_authored or not any(
            normalize_label(witness.reporter) == normalize_label(proposal.subject)
            and proposal.content == witness.quote
            and re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(witness.reporter)}(?![A-Za-z0-9_])",
                witness.quote,
                re.IGNORECASE,
            )
            is not None
            and (source := records[witness.source_id]).context is not None
            and source.context.basis is EvidenceBasis.USER_STATEMENT
            for witness in proposal.affect_witnesses
        )
    return False
