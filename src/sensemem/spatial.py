"""Spatial values and pure operations for the public SDK."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from sensemem._validation import unit_interval
from sensemem._value_validation import (
    _optional_text,
    _require_aware,
    _require_named_interval,
    _text,
)
from sensemem.exceptions import ValidationError
from sensemem.types import EvidenceBasis, SpatialAnchor


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialContext:
    """One metric pose in an application-selected Cartesian coordinate frame."""

    frame_id: str
    anchor: SpatialAnchor
    x: float
    y: float
    z: float = 0.0
    orientation_xyzw: tuple[float, float, float, float] | None = None
    position_uncertainty_m: float | None = None

    def __post_init__(self) -> None:  # noqa: C901 - validates one physical value
        object.__setattr__(self, "frame_id", _text(self.frame_id, "spatial frame_id"))
        if not isinstance(self.anchor, SpatialAnchor):
            try:
                object.__setattr__(self, "anchor", SpatialAnchor(self.anchor))
            except (TypeError, ValueError):
                raise ValidationError("spatial anchor is invalid") from None
        for name in ("x", "y", "z"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
            ):
                raise ValidationError(f"spatial {name} must be a finite number")
            object.__setattr__(self, name, float(value))
        uncertainty = self.position_uncertainty_m
        if uncertainty is not None:
            if (
                isinstance(uncertainty, bool)
                or not isinstance(uncertainty, int | float)
                or not math.isfinite(float(uncertainty))
                or uncertainty < 0
            ):
                raise ValidationError(
                    "spatial position_uncertainty_m must be a non-negative finite number"
                )
            object.__setattr__(self, "position_uncertainty_m", float(uncertainty))
        supplied_orientation = self.orientation_xyzw
        if supplied_orientation is None:
            return
        try:
            values = tuple(float(value) for value in supplied_orientation)
        except (TypeError, ValueError):
            raise ValidationError("spatial orientation_xyzw must contain four numbers") from None
        if len(values) != 4 or any(not math.isfinite(value) for value in values):
            raise ValidationError("spatial orientation_xyzw must contain four finite numbers")
        x, y, z, w = values
        norm = math.hypot(*values)
        if norm == 0:
            raise ValidationError("spatial orientation_xyzw must not be zero")
        normalized: tuple[float, float, float, float] = (
            x / norm,
            y / norm,
            z / norm,
            w / norm,
        )
        sign_value = normalized[3]
        if sign_value == 0:
            sign_value = next((value for value in normalized[:3] if value != 0), 0)
        if sign_value < 0:
            normalized = (
                -normalized[0],
                -normalized[1],
                -normalized[2],
                -normalized[3],
            )
        object.__setattr__(
            self,
            "orientation_xyzw",
            tuple(0.0 if value == 0 else value for value in normalized),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationContext:
    """Typed source provenance, validity, and optional pose.

    The source's basis does not authorize a model's interpretation of extracted subclaims.
    """

    basis: EvidenceBasis = EvidenceBasis.OBSERVATION
    source_id: str | None = None
    confidence: float = 1.0
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    spatial: SpatialContext | None = None
    # The symbolic room-level place, as distinct from `spatial`'s metric pose: a robot can label
    # "kitchen" when it cannot localise, and "in the kitchen" is the query a household asks.
    # Equality-matched, so a producer must label consistently -- "kitchen" and "the kitchen" would
    # partition the store, and nothing normalises it beyond rejecting untrimmed text.
    place_id: str | None = None

    def __post_init__(self) -> None:
        if self.place_id is not None:
            place = self.place_id
            if not isinstance(place, str) or not place.strip() or place != place.strip():
                raise ValidationError("place_id must be non-empty and trimmed")
        if not isinstance(self.basis, EvidenceBasis):
            try:
                object.__setattr__(self, "basis", EvidenceBasis(self.basis))
            except (TypeError, ValueError):
                raise ValidationError("observation basis is invalid") from None
        object.__setattr__(
            self,
            "source_id",
            _optional_text(self.source_id, "observation source_id"),
        )
        object.__setattr__(
            self,
            "confidence",
            unit_interval(self.confidence, "observation confidence"),
        )
        _require_named_interval(
            self.valid_from,
            self.valid_until,
            "valid_from",
            "valid_until",
        )
        if self.spatial is not None and not isinstance(self.spatial, SpatialContext):
            raise ValidationError("observation spatial context is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalScope:
    """Optional world-time, knowledge-time, spatial, and identity retrieval scope.

    Two spatial axes, because a household asks in both. `near`/`radius_m` is metric and answers
    "within two metres of here". `place_id` is symbolic and answers "in the kitchen" — the query a
    person actually asks, and the one a robot can label when it cannot localise metrically. They
    are independent: a symbolic equality is the one spatial predicate SQLite indexes cheaply,
    while the metric radius is a filter over retrieved candidates.

    `identity_id` answers "about this person". A memory is in scope when its semantic subject is
    that identity (`memory_semantics.identity_id`, the binding a naming or claim assertion
    carries) **or** the identity was observed in it: a diarised speech segment or a face
    observation on one of the memory's media assets. A merged alias resolves to the surviving
    identity, so scoping by either ID of a merged pair returns the same memories.
    """

    valid_at: datetime | None = None
    known_at: datetime | None = None
    near: SpatialContext | None = None
    radius_m: float | None = None
    # `None` means "do not scope by place", never "memories that have no place".
    place_id: str | None = None
    # `None` means "do not scope by identity", never "memories about nobody".
    identity_id: str | None = None

    def __post_init__(self) -> None:
        _require_aware(self.valid_at, "scope valid_at")
        _require_aware(self.known_at, "scope known_at")
        if self.place_id is not None:
            place = self.place_id
            # Matches the store's own rule and its SQLite CHECK, so a value that would be
            # rejected on the write path cannot be silently accepted on the read path.
            if not isinstance(place, str) or not place.strip() or place != place.strip():
                raise ValidationError("scope place_id must be non-empty and trimmed")
        if self.identity_id is not None:
            identity = self.identity_id
            if (
                not isinstance(identity, str)
                or not identity.strip()
                or identity != identity.strip()
            ):
                raise ValidationError("scope identity_id must be non-empty and trimmed")
        if (self.near is None) != (self.radius_m is None):
            raise ValidationError("scope near and radius_m must be supplied together")
        if self.near is not None and not isinstance(self.near, SpatialContext):
            raise ValidationError("scope near must be a SpatialContext")
        if self.radius_m is not None:
            radius = self.radius_m
            if (
                isinstance(radius, bool)
                or not isinstance(radius, int | float)
                or not math.isfinite(float(radius))
                or radius < 0
            ):
                raise ValidationError("scope radius_m must be a non-negative finite number")
            object.__setattr__(self, "radius_m", float(radius))
