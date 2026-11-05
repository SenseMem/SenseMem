"""Pure scalar validation shared by public values, the kernel, and model adapters."""

import math
import re

from sensemem.exceptions import ValidationError


def unit_interval(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= value <= 1.0
    ):
        raise ValidationError(f"{name} must be between zero and one")
    return float(value)


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be non-empty text")
    return value.strip()


def optional_text(value: object | None, name: str) -> str | None:
    return None if value is None else nonempty_text(value, name)


def commit_revision(value: object) -> str:
    normalized = nonempty_text(value, "revision").lower()
    if re.fullmatch(r"[0-9a-f]{40}", normalized) is None:
        raise ValidationError("revision must be an immutable 40-character commit hash")
    return normalized
