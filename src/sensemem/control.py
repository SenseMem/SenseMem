"""Pure serialization, identity, and record selection for control-plane operations.

Nothing here touches storage or models. `dump_operation` and `load_operation` round-trip one
`MemoryOperation` through the append-only operation log so a logged operation can be replayed or
rolled back. `operation_key` derives the idempotency identity the log's unique index enforces; it
deliberately excludes `rationale`, which is logged but never interpreted, so two proposals that
differ only in their prose stay one operation.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from typing import Literal, cast

from pydantic import TypeAdapter

from sensemem.control_values import (
    MemoryIntent,
    MemoryOperation,
    MemoryOperationRecord,
    QueryFailure,
)
from sensemem.exceptions import StorageError, ValidationError
from sensemem.identity import ConsentClaim, ConsentState, IdentityChange, IdentityClaim
from sensemem.semantics import AffectLink, AffectWitness, FormationProposal
from sensemem.spatial import RetrievalScope, SpatialContext
from sensemem.types import EvidenceBasis, MemoryCategory, Modality, RecordKind, SpatialAnchor

# Binding a voice to a face requires an explicit host statement. Recognizers supply separate
# observations; co-occurrence is not evidence that a visible person produced the voice.
IDENTITY_LINK_RECIPE = "sensemem-identity-host-binding-v1"
# Shared by host consent assertions and their authoritative storage projections.
CONSENT_PREDICATE = "consent"
_AFFECT_LINK = TypeAdapter(AffectLink)
_AFFECT_WITNESSES = TypeAdapter(tuple[AffectWitness, ...])


def dump_query_failure(failure: QueryFailure) -> str:
    scope = failure.scope
    return _canonical_json(
        {
            "query": failure.query,
            "operation": failure.operation,
            "memory_types": None
            if failure.memory_types is None
            else [value.value for value in failure.memory_types],
            "reference_at": _encode_optional_datetime(failure.reference_at),
            "occurred_from": _encode_optional_datetime(failure.occurred_from),
            "occurred_until": _encode_optional_datetime(failure.occurred_until),
            "scope": None
            if scope is None
            else {
                "valid_at": _encode_optional_datetime(scope.valid_at),
                "known_at": _encode_optional_datetime(scope.known_at),
                "near": None if scope.near is None else _spatial_payload(scope.near),
                "radius_m": scope.radius_m,
                "place_id": scope.place_id,
                "identity_id": scope.identity_id,
            },
        }
    )


def load_query_failure(payload: str) -> QueryFailure:
    try:
        value = json.loads(payload)
        scope = value["scope"]
        return QueryFailure(
            query=value["query"],
            operation=cast(Literal["search", "compile", "ask"], value["operation"]),
            memory_types=None
            if value["memory_types"] is None
            else tuple(MemoryCategory(item) for item in value["memory_types"]),
            reference_at=_decode_optional_datetime(value["reference_at"]),
            occurred_from=_decode_optional_datetime(value["occurred_from"]),
            occurred_until=_decode_optional_datetime(value["occurred_until"]),
            scope=None
            if scope is None
            else RetrievalScope(
                valid_at=_decode_optional_datetime(scope["valid_at"]),
                known_at=_decode_optional_datetime(scope["known_at"]),
                near=None if scope["near"] is None else _decode_spatial_context(scope["near"]),
                radius_m=scope["radius_m"],
                place_id=scope["place_id"],
                identity_id=scope["identity_id"],
            ),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise StorageError("stored query failure is invalid") from error


def query_failure_key(failure: QueryFailure) -> str:
    value = json.loads(dump_query_failure(failure))
    folded = unicodedata.normalize("NFKC", failure.query).casefold()
    value["query"] = " ".join(
        "".join(character if character.isalnum() else " " for character in folded).split()
    )
    value["reference_at"] = (
        None if failure.reference_at is None else failure.reference_at.date().isoformat()
    )
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def operation_touches(
    record: MemoryOperationRecord,
    memory_ids: frozenset[str],
    identity_ids: frozenset[str],
) -> bool:
    """Report whether one logged operation moved any of these records or any of these people."""
    operation = record.operation
    named = {
        *operation.evidence_ids,
        *operation.target_ids,
        *record.created_ids,
        *record.changed_ids,
        *record.forgotten_ids,
        *(memory_id for memory_id, _version in record.superseded),
    }
    if named & memory_ids:
        return True
    people: set[str] = set()
    if operation.identity is not None:
        people.update((operation.identity.identity_id, *operation.identity.moved_ids))
    if operation.claim is not None:
        people.add(operation.claim.identity_id)
    if operation.consent is not None:
        people.add(operation.consent.identity_id)
    return bool(people & identity_ids)


def dump_operation(operation: MemoryOperation) -> str:
    """Return the canonical JSON the operation log stores for one operation."""
    return _canonical_json(
        {**_operation_effect_payload(operation), "rationale": operation.rationale}
    )


def load_operation(payload: str) -> MemoryOperation:
    """Rebuild one operation from a logged payload, validating it like model output."""
    try:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError
        proposal = value["proposal"]
        claim = value["claim"]
        consent = value["consent"]
        identity = value["identity"]
        # Constructors own value validation; these casts do not coerce untrusted JSON.
        return MemoryOperation(
            intent=cast(MemoryIntent, value["intent"]),
            evidence_ids=tuple(value["evidence_ids"]),
            target_ids=tuple(value["target_ids"]),
            proposal=None if proposal is None else _decode_formation_proposal(proposal),
            claim=None if claim is None else _decode_identity_claim(claim),
            consent=None if consent is None else _decode_consent_claim(consent),
            identity=None if identity is None else _decode_identity_change(identity),
            rationale=cast(str | None, value["rationale"]),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
        ValidationError,
    ) as error:
        raise StorageError("a logged memory operation could not be read") from error


def operation_key(operation: MemoryOperation, *, recipe: str | None) -> str:
    """Return the idempotency identity of one operation under one reasoning recipe."""
    payload = _canonical_json({**_operation_effect_payload(operation), "recipe": recipe})
    return hashlib.sha256(f"sensemem-operation-v1:{payload}".encode()).hexdigest()


def _operation_effect_payload(operation: MemoryOperation) -> dict[str, object]:
    """Serialize effect-defining fields shared by logging and the idempotency key."""
    proposal = operation.proposal
    claim = operation.claim
    consent = operation.consent
    change = operation.identity
    return {
        "intent": operation.intent.value,
        "evidence_ids": sorted(operation.evidence_ids),
        "target_ids": sorted(operation.target_ids),
        "proposal": None if proposal is None else proposal_payload(proposal),
        # Part of the idempotency identity: renaming the same person supersedes rather than
        # replays, so two claims that differ only in the name must be two operations.
        "claim": None
        if claim is None
        else {
            "identity_id": claim.identity_id,
            "name": claim.name,
            "relationship": claim.relationship,
        },
        # Part of the idempotency identity: re-stating the state that already stands is the same
        # operation and logs nothing, while moving to a different state is a different one.
        "consent": None
        if consent is None
        else {
            "identity_id": consent.identity_id,
            "state": consent.state.value,
            "note": consent.note,
        },
        # Part of the idempotency identity: merging or splitting a different pair of people is a
        # different operation, so two merges of one pair stay one row and two pairs stay two.
        "identity": None
        if change is None
        else {"identity_id": change.identity_id, "moved_ids": sorted(change.moved_ids)},
    }


def _decode_identity_change(value: object) -> IdentityChange:
    if not isinstance(value, dict):
        raise ValueError
    return IdentityChange(
        identity_id=cast(str, value["identity_id"]),
        moved_ids=tuple(value["moved_ids"]),
    )


def _decode_consent_claim(value: object) -> ConsentClaim:
    if not isinstance(value, dict):
        raise ValueError
    return ConsentClaim(
        identity_id=cast(str, value["identity_id"]),
        state=cast(ConsentState, value["state"]),
        note=cast(str | None, value["note"]),
    )


def _decode_identity_claim(value: object) -> IdentityClaim:
    if not isinstance(value, dict):
        raise ValueError
    return IdentityClaim(
        identity_id=cast(str, value["identity_id"]),
        name=cast(str, value["name"]),
        relationship=cast(str | None, value["relationship"]),
    )


def proposal_payload(proposal: FormationProposal) -> dict[str, object]:
    """Return the JSON one formation proposal is logged as, which `_decode_formation_proposal` reads back.

    Shared with the CLI so an operation row it prints stays the shape `load_operation` accepts.
    """
    spatial = proposal.spatial
    return {
        "kind": proposal.kind.value,
        "content": proposal.content,
        "basis": proposal.basis.value,
        "subject": proposal.subject,
        "predicate": proposal.predicate,
        "value": proposal.value,
        "confidence": proposal.confidence,
        "valid_from": _encode_optional_datetime(proposal.valid_from),
        "valid_until": _encode_optional_datetime(proposal.valid_until),
        "cue_modality": (None if proposal.cue_modality is None else proposal.cue_modality.value),
        "valence": proposal.valence,
        "arousal": proposal.arousal,
        "spatial": None if spatial is None else _spatial_payload(spatial),
        "evidence_ids": (None if proposal.evidence_ids is None else list(proposal.evidence_ids)),
        "local_id": proposal.local_id,
        "affect_link": affect_link_payload(proposal.affect_link),
        "affect_witnesses": [
            {"source_id": item.source_id, "reporter": item.reporter, "quote": item.quote}
            for item in proposal.affect_witnesses
        ],
    }


def affect_link_payload(link: AffectLink | None) -> dict[str, object] | None:
    """Serialize a resolved link or formation-local references without changing their meaning."""
    if link is None:
        return None
    return {
        "affect_id": link.affect_id,
        "role": link.role,
        "object_ref": {
            "memory_id": link.object_ref.memory_id,
            "identity_id": link.object_ref.identity_id,
        },
        "polarity": link.polarity,
    }


def _spatial_payload(spatial: SpatialContext) -> dict[str, object]:
    orientation = spatial.orientation_xyzw
    return {
        "frame_id": spatial.frame_id,
        "anchor": spatial.anchor.value,
        "x": spatial.x,
        "y": spatial.y,
        "z": spatial.z,
        "orientation_xyzw": None if orientation is None else list(orientation),
        "position_uncertainty_m": spatial.position_uncertainty_m,
    }


def _decode_formation_proposal(value: object) -> FormationProposal:
    if not isinstance(value, dict):
        raise ValueError
    spatial = value["spatial"]
    evidence_ids = value["evidence_ids"]
    return FormationProposal(
        kind=cast(RecordKind, value["kind"]),
        content=cast(str, value["content"]),
        basis=cast(EvidenceBasis, value["basis"]),
        subject=cast(str | None, value["subject"]),
        predicate=cast(str | None, value["predicate"]),
        value=cast(str | None, value["value"]),
        confidence=cast(float, value["confidence"]),
        valid_from=_decode_optional_datetime(value["valid_from"]),
        valid_until=_decode_optional_datetime(value["valid_until"]),
        spatial=None if spatial is None else _decode_spatial_context(spatial),
        cue_modality=cast(Modality | None, value["cue_modality"]),
        valence=cast(float | None, value["valence"]),
        arousal=cast(float | None, value["arousal"]),
        evidence_ids=None if evidence_ids is None else tuple(evidence_ids),
        local_id=cast(str | None, value["local_id"]),
        affect_link=(
            None
            if value["affect_link"] is None
            else _AFFECT_LINK.validate_python(value["affect_link"])
        ),
        affect_witnesses=_AFFECT_WITNESSES.validate_python(value["affect_witnesses"]),
    )


def _decode_spatial_context(value: object) -> SpatialContext:
    if not isinstance(value, dict):
        raise ValueError
    orientation = value["orientation_xyzw"]
    return SpatialContext(
        frame_id=cast(str, value["frame_id"]),
        anchor=cast(SpatialAnchor, value["anchor"]),
        x=cast(float, value["x"]),
        y=cast(float, value["y"]),
        z=cast(float, value["z"]),
        orientation_xyzw=(
            None
            if orientation is None
            else cast(tuple[float, float, float, float], tuple(orientation))
        ),
        position_uncertainty_m=cast(float | None, value["position_uncertainty_m"]),
    )


def _encode_optional_datetime(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _decode_optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError
    return datetime.fromisoformat(value)


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
