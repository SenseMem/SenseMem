"""Identity values and pure operations for the public SDK."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

from sensemem._value_validation import _SHA256, _memory_ids, _optional_text, _text
from sensemem.exceptions import ValidationError


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityClaim:
    """Who one recognized person is, as an `IDENTIFY` operation claims it.

    The kernel, not the backend, turns this into the typed ENTITY assertion `identities.name`
    projects, so a proposal only has to name the identity and cite its evidence.
    """

    identity_id: str
    name: str
    relationship: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        for label, value in (("name", self.name), ("relationship", self.relationship)):
            if value is None and label == "relationship":
                continue
            text = _text(value, f"identity {label}")
            if len(text) > 255 or not text.isprintable():
                raise ValidationError(f"identity {label} must be at most 255 printable characters")
            object.__setattr__(self, label, text)


class ConsentState(str, Enum):
    """Whether a recognized person has agreed to be processed as a recognized person.

    Three states rather than a flag, because "nobody has asked" is not "they said no": an
    identity with no recorded consent is unstated, `WITHHELD` is a refusal that was actually
    given, and `WITHDRAWN` is a permission that was given and taken back. Only the last two
    restrain the kernel, and they restrain it identically; the distinction is audit history.
    """

    GRANTED = "granted"
    WITHHELD = "withheld"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsentClaim:
    """One person's own statement about how they may be processed.

    Carried on a `CONSENT` operation the way `IdentityClaim` is carried on an `IDENTIFY` one:
    the kernel, not a backend, turns it into the typed STATE assertion `consent()` projects.
    A model may never propose one -- the kernel refuses a proposed `CONSENT` as `unauthorized`,
    the way it refuses a proposed `MERGE` -- so the recorded basis is always `USER_STATEMENT`.
    """

    identity_id: str
    state: ConsentState
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        if not isinstance(self.state, ConsentState):
            try:
                object.__setattr__(self, "state", ConsentState(self.state))
            except (TypeError, ValueError):
                raise ValidationError("consent state is invalid") from None
        object.__setattr__(self, "note", _optional_text(self.note, "consent note"))


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityChange:
    """Which recognized people one identity-lifecycle operation moved.

    `identity_id` is the person who stands after the operation: the survivor of a cross-modal
    merge, the identity a split was taken back out of, the person an erasure destroyed.
    `moved_ids` are the other identities the operation names -- the absorbed identity of a
    `MERGE`, the identity a `CORRECT` split restored, the aliases an erasure destroyed with the
    person. An identity ID is not a memory ID, which is why it travels here rather than in
    `target_ids`.
    """

    identity_id: str
    moved_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        moved = _memory_ids(self.moved_ids, "moved_ids")
        if self.identity_id in moved:
            raise ValidationError("an identity change must not move the identity it keeps")
        object.__setattr__(self, "moved_ids", moved)


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """One timed transcript turn linked to a stable local speaker identity."""

    asset_id: str
    start_ms: int
    end_ms: int
    text: str
    speaker_id: str | None = None
    speaker_name: str | None = None
    identity_score: float | None = None

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.asset_id) is None:
            raise ValidationError("speaker segment asset_id must be a SHA-256 identifier")
        if (
            isinstance(self.start_ms, bool)
            or not isinstance(self.start_ms, int)
            or isinstance(self.end_ms, bool)
            or not isinstance(self.end_ms, int)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValidationError("speaker segment must have a positive time range")
        object.__setattr__(self, "text", _text(self.text, "speaker segment text"))
        if self.speaker_id is not None:
            object.__setattr__(
                self,
                "speaker_id",
                _text(self.speaker_id, "speaker_id"),
            )
        if self.speaker_name is not None:
            name = _text(self.speaker_name, "speaker_name")
            if len(name) > 255 or not name.isprintable():
                raise ValidationError("speaker_name must be at most 255 printable characters")
            object.__setattr__(self, "speaker_name", name)
        if self.identity_score is not None and (
            isinstance(self.identity_score, bool)
            or not isinstance(self.identity_score, int | float)
            or not math.isfinite(float(self.identity_score))
            or not 0.0 <= self.identity_score <= 1.0
        ):
            raise ValidationError("identity_score must be between zero and one")


@dataclass(frozen=True, slots=True)
class IdentityProfile:
    """What a caller has registered about one recognized person.

    `confirmed` reports a visible host-authored naming assertion. A model-inferred name may
    become visible with independent support, but remains unconfirmed; `evidence_ids` names
    its sources. An unnamed recognized person is provisional rather than an unknown ID.
    """

    identity_id: str
    name: str | None = None
    relationship: str | None = None
    confirmed: bool = False
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        object.__setattr__(self, "confirmed", bool(self.confirmed))
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(_text(value, "evidence id") for value in self.evidence_ids),
        )
        for label in ("name", "relationship"):
            value = getattr(self, label)
            if value is None:
                continue
            text = _text(value, f"identity {label}")
            if len(text) > 255 or not text.isprintable():
                raise ValidationError(f"identity {label} must be at most 255 printable characters")
            object.__setattr__(self, label, text)


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """One detected face linked to a stable local multimodal identity."""

    asset_id: str
    bounding_box: tuple[float, float, float, float]
    identity_id: str
    identity_name: str | None = None
    identity_score: float | None = None
    observed_at_ms: int | None = None

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.asset_id) is None:
            raise ValidationError("face observation asset_id must be a SHA-256 identifier")
        box = tuple(self.bounding_box)
        if len(box) != 4 or any(not math.isfinite(value) for value in box):
            raise ValidationError("face bounding_box must contain four finite values")
        x, y, width, height = box
        if (
            x < 0.0
            or y < 0.0
            or width <= 0.0
            or height <= 0.0
            or x + width > 1.0
            or y + height > 1.0
        ):
            raise ValidationError("face bounding_box must be normalized within the frame")
        object.__setattr__(self, "bounding_box", box)
        object.__setattr__(self, "identity_id", _text(self.identity_id, "identity_id"))
        if self.identity_name is not None:
            name = _text(self.identity_name, "identity_name")
            if len(name) > 255 or not name.isprintable():
                raise ValidationError("identity_name must be at most 255 printable characters")
            object.__setattr__(self, "identity_name", name)
        if self.identity_score is not None and (
            isinstance(self.identity_score, bool)
            or not isinstance(self.identity_score, int | float)
            or not math.isfinite(float(self.identity_score))
            or not 0.0 <= self.identity_score <= 1.0
        ):
            raise ValidationError("identity_score must be between zero and one")
        if self.observed_at_ms is not None and (
            isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 0
        ):
            raise ValidationError("observed_at_ms must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class IdentityErasure:
    """What one `forget_identity` call actually destroyed.

    The counts are the audit record for a privacy operation: a caller that erased a person can
    state how many biometric templates and annotations were removed instead of assuming. It lives
    here rather than in the storage layer because it is the return type of a public `Memory`
    operation, and the supported import path is `sensemem`.
    """

    identity_id: str
    alias_ids: tuple[str, ...]
    face_exemplars: int
    voice_exemplars: int
    face_observations: int
    speech_segments: int

    def __post_init__(self) -> None:
        _text(self.identity_id, "identity_id")
        for name in ("face_exemplars", "voice_exemplars", "face_observations", "speech_segments"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValidationError(f"{name} must be a non-negative integer")


PerceptionKind: TypeAlias = Literal["speech", "faces"]


PerceptionSkipReason: TypeAlias = Literal[
    "not_requested", "backend_not_configured", "unsupported_modality"
]


@dataclass(frozen=True, slots=True)
class PerceptionCoverage:
    """One asset's completed analysis, or the reason it was not analyzed."""

    kind: PerceptionKind
    asset_id: str
    reason: PerceptionSkipReason | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"speech", "faces"}:
            raise ValidationError("perception kind is invalid")
        _text(self.asset_id, "asset_id")
        if self.reason not in {
            None,
            "not_requested",
            "backend_not_configured",
            "unsupported_modality",
        }:
            raise ValidationError("perception skip reason is invalid")

    @property
    def processed(self) -> bool:
        """Analysis ran successfully or an existing analysis was reused."""
        return self.reason is None


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    """Source-local speech and face observations, retaining their distinct value types."""

    speech: tuple[SpeakerSegment, ...] = ()
    faces: tuple[FaceObservation, ...] = ()
    coverage: tuple[PerceptionCoverage, ...] = ()

    def __post_init__(self) -> None:
        speech = tuple(self.speech)
        faces = tuple(self.faces)
        if any(not isinstance(value, SpeakerSegment) for value in speech):
            raise ValidationError("perception speech is invalid")
        if any(not isinstance(value, FaceObservation) for value in faces):
            raise ValidationError("perception faces are invalid")
        object.__setattr__(self, "speech", speech)
        object.__setattr__(self, "faces", faces)
        coverage = tuple(self.coverage)
        if any(not isinstance(value, PerceptionCoverage) for value in coverage):
            raise ValidationError("perception coverage is invalid")
        keys = {(value.kind, value.asset_id) for value in coverage}
        if len(keys) != len(coverage):
            raise ValidationError("perception coverage must be unique per kind and asset")
        object.__setattr__(self, "coverage", coverage)
