"""Pure scalar validation shared by public values, the kernel, and model adapters."""

import base64
import json
import math
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

from sensemem.exceptions import ValidationError


def encode_page_cursor(at: datetime, identifier: str) -> str:
    payload = json.dumps(
        ["v1", at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), identifier],
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


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


def absolute_http_url(value: str) -> bool:
    """Check an absolute HTTP URL accepted by OpenAI-compatible clients."""
    if any(character.isspace() for character in value) or "\\" in value:
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and parsed.hostname is not None
        and (port is None or 1 <= port <= 65_535)
    )


def server_base_url(value: str) -> str:
    """Normalize a server root whose endpoint path will be appended by the client."""
    value = nonempty_text(value, "base_url")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValidationError("base_url must be an absolute HTTP or HTTPS URL") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or (port is not None and not 1 <= port <= 65535)
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ValidationError(
            "base_url must be an absolute HTTP or HTTPS URL without query or fragment"
        )
    return value.rstrip("/")
