"""Pure proposal rules shared by formation, identity governance, and memory control.

These functions build stable claim identities and contexts, qualify evidence, and construct host
assertions. They read no storage and call no model; each plane owns its I/O and transactions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime

from sensemem.control import CONSENT_PREDICATE, affect_link_payload
from sensemem.control_values import MemoryIntent, MemoryOperation
from sensemem.identity import ConsentClaim
from sensemem.infrastructure.local.store import (
    StoredMemory,
    datetime_text,
)
from sensemem.kernel.content import encode_metadata, observation_from_record
from sensemem.kernel.predicates import claim_lineage_id, normalize_label, resolve_predicate
from sensemem.models.base import FormationInput, ModelInput
from sensemem.records import MemoryRecord
from sensemem.semantics import FormationProposal, RecordSemantics
from sensemem.spatial import SpatialContext
from sensemem.types import CATEGORY_BY_KIND, EvidenceBasis, MemoryCategory, RecordKind


def formation_evidence_active(record: MemoryRecord) -> bool:
    context = record.context
    return record.forgotten_at is None and (
        context is None or (context.visible and context.retired_at is None)
    )


# Naming a person is a claim the host asserts, so its recipe names the kernel rule that produced
# it rather than a model space. That is what lets `register_identity` work with no former and no
# consolidator configured.
NAMING_RECIPE = "sensemem-identity-naming-v1"


NAMING_PREDICATE = "identity"


# Recording consent rests on the same authority as naming -- somebody said so -- so it is the
# same kind of assertion under its own recipe and its own predicate, which keeps the two in
# separate lineages: consenting never renames anybody and renaming never re-opens consent.
CONSENT_RECIPE = "sensemem-identity-consent-v1"


def formation_memory_type(kind: RecordKind) -> MemoryCategory:
    return CATEGORY_BY_KIND.get(kind, MemoryCategory.SEMANTIC)


def formation_refusal(
    proposal: FormationProposal,
    source: FormationInput,
    *,
    host_authored: bool = False,
) -> str | None:
    """Say why the kernel refuses to ground this proposal in this source, or `None` to keep it.

    A refusal costs the proposal and nothing else, on both paths that ask: formation drops it,
    and consolidation moves on to the next cited source.
    """
    if proposal.kind is RecordKind.RESPONSE_POLICY and not host_authored:
        # How the system should behave toward somebody is a grant, not an inference: one
        # observed cue must not become standing guidance. Keyed on where the proposal came
        # from, never on the basis it claims -- a formation or consolidation backend builds
        # its own `FormationProposal` and could name any basis -- so the only path that
        # authorizes one is the host's own `apply()`.
        return "response_policy formation requires explicit host authorization"
    if not host_authored and proposal.basis is not EvidenceBasis.MODEL_INFERENCE:
        # Source provenance does not authorize the model's interpretation of a subclaim.
        return "automatic formation requires model_inference basis"
    if proposal.kind is RecordKind.AFFECT and (
        proposal.cue_modality is None or proposal.cue_modality not in source.content.modalities
    ):
        return "affect formation must name a modality present in its source"
    if proposal.spatial is not None:
        observed = source.context.spatial
        if (
            observed is None
            or proposal.spatial.frame_id != observed.frame_id
            or proposal.spatial.anchor is not observed.anchor
        ):
            return "spatial formation must use the source observation frame and anchor"
    return None


def naming_proposal(
    name: str,
    relationship: str | None,
    *,
    basis: EvidenceBasis,
) -> FormationProposal:
    """Build the ENTITY proposal one naming claim asserts, whoever claimed it."""
    return FormationProposal(
        kind=RecordKind.ENTITY,
        content=(
            f"{name} is a recognized person."
            if relationship is None
            else f"{name} is a recognized person, {relationship}."
        ),
        basis=basis,
        subject=name,
        predicate=NAMING_PREDICATE,
        value=relationship,
        confidence=1.0 if basis is EvidenceBasis.USER_STATEMENT else 0.8,
    )


def consent_proposal(claim: ConsentClaim) -> FormationProposal:
    """Build the STATE proposal one consent statement asserts."""
    stated = f"Processing consent for {claim.identity_id} is {claim.state.value}."
    return FormationProposal(
        kind=RecordKind.STATE,
        content=stated if claim.note is None else f"{stated} {claim.note}",
        # The one basis a consent statement can have. Nothing was observed and no model may
        # infer it: the person said so.
        basis=EvidenceBasis.USER_STATEMENT,
        # The identity itself, not the name it currently projects: a person may be unnamed, and
        # renaming one must not fork their consent history into a second subject.
        subject=claim.identity_id,
        predicate=CONSENT_PREDICATE,
        value=claim.state.value,
        confidence=1.0,
    )


def is_consent_assertion(
    memories: Mapping[str, StoredMemory],
    memory_id: str,
) -> bool:
    """Report whether one record is a standing consent statement about a person."""
    memory = memories.get(memory_id)
    context = None if memory is None else memory.context
    return (
        context is not None
        and context.identity_id is not None
        and context.predicate == CONSENT_PREDICATE
    )


def is_bound_naming_assertion(
    memories: Mapping[str, StoredMemory],
    memory_id: str,
) -> bool:
    memory = memories.get(memory_id)
    context = None if memory is None else memory.context
    return (
        context is not None
        and context.kind is RecordKind.ENTITY
        and context.identity_id is not None
    )


def is_derived(memories: Mapping[str, StoredMemory], memory_id: str) -> bool:
    memory = memories.get(memory_id)
    context = None if memory is None else memory.context
    return context is not None and context.kind is not RecordKind.OBSERVATION


def consolidation_primary(
    proposal: FormationProposal,
    sources: Sequence[MemoryRecord],
    *,
    host_authored: bool = False,
) -> MemoryRecord | None:
    """Return the newest cited source the proposal is grounded in, or `None`.

    Formation binds one proposal to one source. A consolidation cites several, so the kernel
    requires the AFFECT cue modality and the spatial frame to match at least one of them and
    inherits validity from the newest such source, which keeps the derived ID stable when the
    same evidence set is proposed again.
    """
    ranked = sorted(
        sources,
        key=lambda source: (
            source.occurred_end or source.occurred_at or source.created_at,
            source.id,
        ),
    )
    for source in reversed(ranked):
        refusal = formation_refusal(
            proposal,
            FormationInput(
                memory_id=source.id,
                content=ModelInput(text=source.content, assets=source.assets, parts=source.parts),
                context=observation_from_record(source),
                occurred_at=source.occurred_at,
                occurred_end=source.occurred_end,
                created_at=source.created_at,
            ),
            host_authored=host_authored,
        )
        if refusal is None:
            return source
    return None


def retiring_targets(operation: MemoryOperation) -> set[str]:
    """IDs this operation would take out of ordinary recall or retire the current version of.

    A `REINFORCE` target is strengthened rather than retired, so it is not one of these; a
    `CONSOLIDATE` target is consolidation forgetting and is.
    """
    if operation.intent is MemoryIntent.REINFORCE:
        return set()
    return set(operation.target_ids)


def agreed_inheritance(
    sources: Sequence[MemoryRecord],
) -> tuple[str | None, Mapping[str, object] | None]:
    """The place and the metadata every cited source agrees on, or nothing.

    A place is a hard retrieval filter, so disagreement inherits nothing rather than a majority.
    """
    places = {source.place_id for source in sources}
    tags = {encode_metadata(source.metadata) for source in sources}
    return (
        sources[0].place_id if len(places) == 1 else None,
        sources[0].metadata if len(tags) == 1 else None,
    )


def formation_context(
    source: MemoryRecord | None,
    proposal: FormationProposal,
    *,
    model_id: str | None,
    recipe: str,
    recorded_at: datetime,
    identity_id: str | None = None,
) -> RecordSemantics:
    """Build the typed context for one proposal, optionally derived from a source memory.

    A naming assertion has no source memory and no model behind it: the household owner said
    so. It passes `source=None` and carries the identity binding instead.
    """
    source_context = None if source is None else observation_from_record(source)
    valid_from = proposal.valid_from
    valid_until = proposal.valid_until
    spatial = proposal.spatial
    if source is not None and source_context is not None:
        valid_from = valid_from or source_context.valid_from or source.occurred_at
        valid_until = valid_until or source_context.valid_until
        spatial = spatial or source_context.spatial
    if valid_from is None and proposal.kind in {RecordKind.STATE, RecordKind.TRAIT}:
        valid_from = recorded_at
    return RecordSemantics(
        kind=proposal.kind,
        # How the system should behave toward somebody is a grant: the kernel refuses every
        # model-proposed `RESPONSE_POLICY`, so one that reaches persistence was authorized by
        # the host calling `apply()` itself, and the stored record names that authorization
        # rather than the default the proposal carried. Stamped here, on the persisted context
        # alone, and never on the proposal: the log keeps the proposal it was handed and
        # `formation_memory_id` keys on that basis, so replaying a logged row mints the same
        # ID and is recognised as the duplicate it is.
        basis=(
            EvidenceBasis.RESPONSE_FEEDBACK
            if proposal.kind is RecordKind.RESPONSE_POLICY
            else proposal.basis
        ),
        confidence=proposal.confidence,
        valid_from=valid_from,
        valid_until=valid_until,
        recorded_at=recorded_at,
        lineage_id=_formation_lineage_id(proposal, spatial=spatial, identity_id=identity_id),
        source_id=None if source_context is None else source_context.source_id,
        subject=proposal.subject,
        predicate=proposal.predicate,
        value=proposal.value,
        evidence_ids=() if source is None else (source.id,),
        model_id=model_id,
        recipe=recipe,
        identity_id=identity_id,
        spatial=spatial,
        cue_modality=proposal.cue_modality,
        valence=proposal.valence,
        arousal=proposal.arousal,
        affect_link=proposal.affect_link,
        affect_witnesses=proposal.affect_witnesses,
    )


def _formation_lineage_id(
    proposal: FormationProposal,
    *,
    spatial: SpatialContext | None = None,
    identity_id: str | None = None,
) -> str:
    if proposal.affect_link is not None:
        payload = affect_link_payload(proposal.affect_link)
        assert payload is not None
        del payload["polarity"]
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"sensemem-affect-lineage-v1:{encoded}".encode()).hexdigest()
    return claim_lineage_id(
        kind=proposal.kind.value,
        predicate=proposal.predicate,
        subject=proposal.subject,
        identity_id=identity_id,
        frame_id=None if spatial is None else spatial.frame_id,
        anchor=None if spatial is None else spatial.anchor.value,
        resolution=resolve_predicate(
            proposal.predicate,
            kind=proposal.kind.value,
            basis=proposal.basis.value,
            value=proposal.value,
        ),
    )


def formation_memory_id(
    source_id: str,
    proposal: FormationProposal,
    *,
    recipe: str,
    context: RecordSemantics,
) -> str:
    episode_source = (
        source_id
        if (
            proposal.kind in {RecordKind.EVENT, RecordKind.AFFECT, RecordKind.STATE}
            or (
                proposal.kind is RecordKind.TRAIT and proposal.basis is EvidenceBasis.USER_STATEMENT
            )
        )
        else None
    )
    spatial = context.spatial
    payload = json.dumps(
        {
            "recipe": recipe,
            "kind": proposal.kind.value,
            "affect_link": affect_link_payload(proposal.affect_link),
            "affect_reporters": sorted({item.reporter for item in proposal.affect_witnesses}),
            # Only present once something is bound, so two identities that share a name stay two
            # records while every ID minted before the binding existed keeps its value.
            **({} if context.identity_id is None else {"identity_id": context.identity_id}),
            # A bound ENTITY asserts the name itself; its subject distinguishes naming claims.
            "subject": None
            if context.identity_id is not None and proposal.kind is not RecordKind.ENTITY
            else normalize_label(proposal.subject),
            "predicate": resolve_predicate(
                proposal.predicate,
                kind=proposal.kind.value,
                basis=proposal.basis.value,
                value=proposal.value,
            ).key,
            "value": None if proposal.affect_link is not None else normalize_label(proposal.value),
            "assertion_basis": (
                proposal.basis.value
                if proposal.basis is not EvidenceBasis.MODEL_INFERENCE
                else None
            ),
            "cue_modality": (
                None if proposal.cue_modality is None else proposal.cue_modality.value
            ),
            "episode_source": episode_source,
            "content": proposal.content if proposal.kind is RecordKind.EVENT else None,
            "valid_from": (
                datetime_text(context.valid_from)
                if episode_source is not None and context.valid_from is not None
                else None
            ),
            "valid_until": (
                datetime_text(context.valid_until)
                if episode_source is not None and context.valid_until is not None
                else None
            ),
            "spatial": (
                None
                if spatial is None
                else {
                    "frame_id": spatial.frame_id,
                    "anchor": spatial.anchor.value,
                    "position_m": (spatial.x, spatial.y, spatial.z),
                    "orientation_xyzw": spatial.orientation_xyzw,
                    "position_uncertainty_m": spatial.position_uncertainty_m,
                }
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"sensemem-formation-v1:{payload}".encode()).hexdigest()
