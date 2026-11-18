"""Identity binding follows cited naming premises, never a global display-name lookup."""

from __future__ import annotations

import re
from collections.abc import Sequence

from sensemem.control import CONSENT_PREDICATE
from sensemem.exceptions import ValidationError
from sensemem.kernel.predicates import normalize_label
from sensemem.models.base import FormationInput
from sensemem.records import MemoryRecord
from sensemem.semantics import FormationProposal
from sensemem.types import RecordKind


def claim_identity(proposal: FormationProposal, evidence: Sequence[MemoryRecord]) -> str | None:
    """Resolve a unique, standing cited name, refusing an ambiguous conjunction."""
    subject = normalize_label(proposal.subject)
    if (
        subject is None
        or proposal.kind is RecordKind.ENTITY
        or proposal.predicate == CONSENT_PREDICATE
    ):
        return None
    identities = {
        context.identity_id
        for record in evidence
        if (context := record.context) is not None
        and context.kind is RecordKind.ENTITY
        and normalize_label(context.subject) == subject
        and context.identity_id is not None
        and record.forgotten_at is None
        and context.visible
        and context.retired_at is None
    }
    if len(identities) > 1:
        raise ValidationError("claim evidence names more than one identity")
    return next(iter(identities), None)


def history_identity_refusal(
    proposal: FormationProposal, source: FormationInput, evidence: Sequence[MemoryRecord]
) -> bool:
    """Require historical naming unless original text already grounds an unbound label.

    An original explicit label can stand unbound even when history names a stable identity.
    Binding still needs its cited naming premise; this exception grants no identity, statement
    authority, or permission to remove a declared premise.
    """
    subject = normalize_label(proposal.subject)
    if subject is None or proposal.kind is RecordKind.ENTITY:
        return False
    known = tuple(
        record
        for record in source.history
        if record.context is not None
        and record.context.kind is RecordKind.ENTITY
        and normalize_label(record.context.subject) == subject
    )
    if not known:
        return False
    witnesses = tuple(
        record
        for record in evidence
        if record.context is not None
        and record.context.kind is RecordKind.ENTITY
        and normalize_label(record.context.subject) == subject
        and record.forgotten_at is None
        and record.context.visible
        and record.context.retired_at is None
    )
    try:
        claim_identity(proposal, witnesses)
    except ValidationError:
        return True
    if witnesses:
        return False
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(subject)}(?![A-Za-z0-9_])")
    return not any(
        part.kind == "text" and pattern.search(normalize_label(part.value) or "") is not None
        for part in source.content.parts
    )
