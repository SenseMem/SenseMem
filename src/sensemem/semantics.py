"""Semantics values and pure operations for the public SDK."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sensemem._validation import unit_interval
from sensemem._value_validation import (
    _bounded_optional,
    _optional_text,
    _require_aware,
    _require_named_interval,
    _text,
)
from sensemem.exceptions import ValidationError
from sensemem.spatial import SpatialContext
from sensemem.types import EvidenceBasis, Modality, RecordKind


@dataclass(frozen=True, slots=True, kw_only=True)
class AffectObjectRef:
    """One affect target: a stable recognized identity or a particular entity/event record."""

    memory_id: str | None = None
    identity_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("memory_id", "identity_id"):
            object.__setattr__(
                self, name, _optional_text(getattr(self, name), f"affect object {name}")
            )
        if (self.memory_id is None) == (self.identity_id is None):
            raise ValidationError("affect object requires exactly one memory_id or identity_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class AffectLink:
    """One independently sourced assertion about the target or elicitor of an affect cue."""

    affect_id: str
    role: Literal["target", "elicitor"]
    object_ref: AffectObjectRef
    polarity: Literal["affirmed", "denied"] = "affirmed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "affect_id", _text(self.affect_id, "affect link affect_id"))
        if self.role not in {"target", "elicitor"}:
            raise ValidationError("affect link role must be target or elicitor")
        if self.polarity not in {"affirmed", "denied"}:
            raise ValidationError("affect link polarity must be affirmed or denied")
        if not isinstance(self.object_ref, AffectObjectRef):
            raise ValidationError("affect link object_ref must be AffectObjectRef")
        if self.role == "elicitor" and self.object_ref.identity_id is not None:
            raise ValidationError("affect elicitor must reference an event memory")


@dataclass(frozen=True, slots=True, kw_only=True)
class AffectWitness:
    """A report's exact words and reporter, tied to an original observation record."""

    source_id: str
    reporter: str
    quote: str

    def __post_init__(self) -> None:
        for name in ("source_id", "reporter", "quote"):
            object.__setattr__(self, name, _text(getattr(self, name), f"affect witness {name}"))


@dataclass(frozen=True, slots=True, kw_only=True)
class FormationProposal:
    """One typed, source-grounded semantic memory proposed by a model adapter.

    ``evidence_ids=None`` uses the primary input as the sole evidence member. A supplied tuple
    declares the offered records the proposal jointly depends on; the memory kernel verifies
    the primary source and the supplied batch/history window before commit. Typed affect links
    also require their focal cue and any record-valued object as premises.
    """

    kind: RecordKind
    content: str
    basis: EvidenceBasis = EvidenceBasis.MODEL_INFERENCE
    subject: str | None = None
    predicate: str | None = None
    value: str | None = None
    confidence: float = 1.0
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    spatial: SpatialContext | None = None
    cue_modality: Modality | None = None
    valence: float | None = None
    arousal: float | None = None
    evidence_ids: tuple[str, ...] | None = None
    local_id: str | None = None
    affect_link: AffectLink | None = None
    affect_witnesses: tuple[AffectWitness, ...] = ()

    def __post_init__(self) -> None:  # noqa: C901 - kind-specific boundary validation
        if not isinstance(self.kind, RecordKind):
            try:
                object.__setattr__(self, "kind", RecordKind(self.kind))
            except (TypeError, ValueError):
                raise ValidationError("formation kind is invalid") from None
        if self.kind is RecordKind.OBSERVATION:
            raise ValidationError("formation cannot propose an observation")
        if not isinstance(self.basis, EvidenceBasis):
            try:
                object.__setattr__(self, "basis", EvidenceBasis(self.basis))
            except (TypeError, ValueError):
                raise ValidationError("formation basis is invalid") from None
        if self.basis is EvidenceBasis.OBSERVATION:
            raise ValidationError("formation basis cannot be observation")
        object.__setattr__(self, "content", _text(self.content, "formation content"))
        for name in ("subject", "predicate", "value"):
            object.__setattr__(
                self,
                name,
                _optional_text(getattr(self, name), f"formation {name}"),
            )
        needs_subject = {
            RecordKind.ENTITY,
            RecordKind.STATE,
            RecordKind.RELATION,
            RecordKind.AFFECT,
            RecordKind.TRAIT,
            RecordKind.RESPONSE_POLICY,
        }
        needs_predicate = {
            RecordKind.STATE,
            RecordKind.RELATION,
            RecordKind.TRAIT,
            RecordKind.RESPONSE_POLICY,
        }
        needs_value = needs_predicate | {RecordKind.AFFECT}
        if self.kind in needs_subject and self.subject is None:
            raise ValidationError(f"{self.kind.value} formation requires a subject")
        if self.kind in needs_predicate and self.predicate is None:
            raise ValidationError(f"{self.kind.value} formation requires a predicate")
        if self.kind in needs_value and self.value is None:
            raise ValidationError(f"{self.kind.value} formation requires a value")
        object.__setattr__(
            self,
            "confidence",
            unit_interval(self.confidence, "formation confidence"),
        )
        _require_named_interval(
            self.valid_from,
            self.valid_until,
            "formation valid_from",
            "formation valid_until",
        )
        if self.evidence_ids is not None:
            if isinstance(self.evidence_ids, str | bytes):
                raise ValidationError(
                    "formation evidence_ids must be a non-empty sequence of memory IDs"
                )
            try:
                evidence_ids = tuple(self.evidence_ids)
            except TypeError:
                raise ValidationError(
                    "formation evidence_ids must be a non-empty sequence of memory IDs"
                ) from None
            if not evidence_ids or any(
                not isinstance(value, str) or not value.strip() for value in evidence_ids
            ):
                raise ValidationError(
                    "formation evidence_ids must be a non-empty sequence of memory IDs"
                )
            if len(set(evidence_ids)) != len(evidence_ids):
                raise ValidationError("formation evidence_ids must be unique")
            object.__setattr__(self, "evidence_ids", evidence_ids)
        if self.spatial is not None and not isinstance(self.spatial, SpatialContext):
            raise ValidationError("formation spatial context is invalid")
        if self.cue_modality is not None:
            try:
                cue_modality = Modality(self.cue_modality)
            except (TypeError, ValueError):
                raise ValidationError("formation cue_modality is invalid") from None
            if cue_modality is Modality.OMNI:
                raise ValidationError("formation cue_modality must be atomic")
            object.__setattr__(self, "cue_modality", cue_modality)
        object.__setattr__(self, "valence", _bounded_optional(self.valence, "valence", -1, 1))
        object.__setattr__(self, "arousal", _bounded_optional(self.arousal, "arousal", 0, 1))
        if self.kind is not RecordKind.AFFECT and any(
            value is not None for value in (self.cue_modality, self.valence, self.arousal)
        ):
            raise ValidationError("affect fields require kind='affect'")
        object.__setattr__(self, "local_id", _optional_text(self.local_id, "formation local_id"))
        if self.local_id is not None and (
            len(self.local_id) > 128 or not re.fullmatch(r"[A-Za-z0-9_-]+", self.local_id)
        ):
            raise ValidationError("formation local_id must be a short alphanumeric identifier")
        _validate_affect_link(self.kind, self.predicate, self.affect_link, self.affect_witnesses)
        object.__setattr__(self, "affect_witnesses", tuple(dict.fromkeys(self.affect_witnesses)))


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordSemantics:
    """Authoritative typed semantics attached to a stored memory."""

    kind: RecordKind
    basis: EvidenceBasis
    confidence: float
    valid_from: datetime | None
    valid_until: datetime | None
    recorded_at: datetime
    visible: bool = True
    retired_at: datetime | None = None
    lineage_id: str | None = None
    source_id: str | None = None
    subject: str | None = None
    predicate: str | None = None
    value: str | None = None
    evidence_ids: tuple[str, ...] = ()
    supersedes_id: str | None = None
    model_id: str | None = None
    recipe: str | None = None
    identity_id: str | None = None
    spatial: SpatialContext | None = None
    cue_modality: Modality | None = None
    valence: float | None = None
    arousal: float | None = None
    affect_link: AffectLink | None = None
    affect_witnesses: tuple[AffectWitness, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RecordKind) or not isinstance(self.basis, EvidenceBasis):
            raise ValidationError("memory context kind or basis is invalid")
        object.__setattr__(
            self,
            "confidence",
            unit_interval(self.confidence, "memory context confidence"),
        )
        _require_named_interval(
            self.valid_from,
            self.valid_until,
            "memory context valid_from",
            "memory context valid_until",
        )
        _require_aware(self.recorded_at, "memory context recorded_at")
        if not isinstance(self.visible, bool):
            raise ValidationError("memory context visible must be a boolean")
        _require_aware(self.retired_at, "memory context retired_at")
        if self.retired_at is not None and self.retired_at <= self.recorded_at:
            raise ValidationError("memory context retired_at must follow recorded_at")
        for name in (
            "source_id",
            "lineage_id",
            "subject",
            "predicate",
            "value",
            "supersedes_id",
            "model_id",
            "recipe",
            "identity_id",
        ):
            object.__setattr__(
                self,
                name,
                _optional_text(getattr(self, name), f"memory context {name}"),
            )
        evidence_ids = tuple(dict.fromkeys(self.evidence_ids))
        if any(not isinstance(value, str) or not value.strip() for value in evidence_ids):
            raise ValidationError("memory context evidence_ids are invalid")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        if self.spatial is not None and not isinstance(self.spatial, SpatialContext):
            raise ValidationError("memory context spatial value is invalid")
        if self.cue_modality is not None and not isinstance(self.cue_modality, Modality):
            raise ValidationError("memory context cue_modality is invalid")
        object.__setattr__(self, "valence", _bounded_optional(self.valence, "valence", -1, 1))
        object.__setattr__(self, "arousal", _bounded_optional(self.arousal, "arousal", 0, 1))
        _validate_affect_link(self.kind, self.predicate, self.affect_link, self.affect_witnesses)
        object.__setattr__(self, "affect_witnesses", tuple(dict.fromkeys(self.affect_witnesses)))
        if self.affect_link is not None and any(
            reference is not None and reference.startswith("proposal:")
            for reference in (
                self.affect_link.affect_id,
                self.affect_link.object_ref.memory_id,
            )
        ):
            raise ValidationError("stored affect links cannot contain proposal references")


def _validate_affect_link(
    kind: RecordKind,
    predicate: str | None,
    link: AffectLink | None,
    witnesses: Sequence[AffectWitness],
) -> None:
    if link is not None:
        if not isinstance(link, AffectLink) or kind is not RecordKind.RELATION:
            raise ValidationError("affect_link requires a typed relation")
        if predicate != f"affect_{link.role}":
            raise ValidationError("affect relation predicate must match its role")
    elif predicate in {"affect_target", "affect_elicitor"}:
        raise ValidationError("affect relation predicates require affect_link")
    if witnesses and kind is not RecordKind.AFFECT and link is None:
        raise ValidationError("affect witnesses require affect or a typed affect relation")
    if isinstance(witnesses, str | bytes) or any(
        not isinstance(witness, AffectWitness) for witness in witnesses
    ):
        raise ValidationError("affect_witnesses must contain AffectWitness values")
