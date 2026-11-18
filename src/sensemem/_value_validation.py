"""value validation values and pure operations for the public SDK."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime

from sensemem.exceptions import ValidationError
from sensemem.types import Modality

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


_MEDIA_TYPE = re.compile(r"[!#$&^_.+0-9A-Za-z-]+/[!#$&^_.+0-9A-Za-z-]+\Z")


def _metadata(value: Mapping[str, object]) -> Mapping[str, object]:
    return dict(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must not be blank")
    return value.strip()


def _optional_text(value: object | None, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _media_type(value: object) -> str:
    if not isinstance(value, str):
        raise ValidationError("media_type must be an image, video, or audio media type")
    normalized = value.strip().lower()
    if _MEDIA_TYPE.fullmatch(normalized) is None or normalized.split("/", 1)[0] not in {
        "image",
        "video",
        "audio",
    }:
        raise ValidationError("media_type must be an image, video, or audio media type")
    return normalized


def _optional_media_type(value: object | None) -> str | None:
    return None if value is None else _media_type(value)


def _require_matching_media_type(modality: Modality, media_type: str) -> None:
    if media_type.split("/", 1)[0] != modality.value:
        raise ValidationError("asset modality does not match media_type")


def _memory_ids(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValidationError(f"{name} must be a sequence of memory IDs")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValidationError(f"{name} must contain non-blank memory IDs")
    return tuple(dict.fromkeys(values))


def _require_aware(value: datetime | None, name: str) -> None:
    if value is not None and (
        not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
    ):
        raise ValidationError(f"{name} must include a timezone")


def _require_interval(start: datetime | None, end: datetime | None) -> None:
    _require_aware(start, "occurred_at")
    _require_aware(end, "occurred_end")
    if end is not None and (start is None or end <= start):
        raise ValidationError("occurred_end must be later than occurred_at")


def _require_named_interval(
    start: datetime | None,
    end: datetime | None,
    start_name: str,
    end_name: str,
) -> None:
    _require_aware(start, start_name)
    _require_aware(end, end_name)
    if end is not None and (start is None or end <= start):
        raise ValidationError(f"{end_name} must be later than {start_name}")


def _bounded_optional(
    value: object | None,
    name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not minimum <= value <= maximum
    ):
        raise ValidationError(f"{name} must be between {minimum:g} and {maximum:g}")
    return float(value)


def _trace_number(value: object, name: str, *, maximum: float | None = None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise ValidationError(f"trace {name} must be a non-negative finite number")
    if maximum is not None and value > maximum:
        raise ValidationError(f"trace {name} must not exceed one")
