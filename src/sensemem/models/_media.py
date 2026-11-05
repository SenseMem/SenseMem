"""Small shared helpers for reading local media and its metadata."""

from __future__ import annotations

import base64
import math
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import datetime
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from sensemem.exceptions import ModelError, ValidationError
from sensemem.types import AssetRef, Modality

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def image_capture_metadata(
    path: Path, *, opened_image: PILImage | None = None
) -> dict[str, object]:
    """Read bounded EXIF capture data, without decoding pixels or trusting free-form tags.

    Callers that expose this data must verify the original asset's integrity. Invalid or
    unavailable metadata is optional input and must never prevent the pixels being used.
    """
    try:
        from PIL import Image

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(path) if opened_image is None else nullcontext(opened_image) as image:
                if image.width * image.height > 64_000_000:
                    return {}
                raw = image.info.get("exif")
                if isinstance(raw, bytes) and len(raw) > 1_048_576:
                    return {}
                exif = image.getexif()
                gps = exif.get_ifd(34853) if 34853 in exif else {}
                details = exif.get_ifd(34665) if 34665 in exif else {}
                return _capture_fields(exif, details, gps)
    except Exception:
        # EXIF is untrusted and parsers reject many malformed offsets, types and rationals.
        # Integrity failures are checked by the caller outside this optional parsing boundary.
        return {}


def _capture_fields(
    exif: Mapping[int, object], details: Mapping[int, object], gps: Mapping[int, object]
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    latitude = _gps_coordinate(gps.get(2), gps.get(1), latitude=True)
    longitude = _gps_coordinate(gps.get(4), gps.get(3), latitude=False)
    if latitude is not None and longitude is not None:
        metadata["gps"] = {"latitude": latitude, "longitude": longitude}
    for date_tag, offset_tag in ((36867, 36881), (36868, 36882), (306, 36880)):
        value = details.get(date_tag, exif.get(date_tag))
        if not isinstance(value, str):
            continue
        value = value.rstrip("\x00")
        if (
            re.fullmatch(
                r"[0-9]{4}:[0-9]{2}:[0-9]{2} (?:[01][0-9]|2[0-3]):[0-9]{2}:[0-9]{2}", value
            )
            is None
        ):
            continue
        try:
            local = datetime.fromisoformat(
                f"{value[:10].replace(':', '-')}T{value[11:]}"
            ).isoformat()
        except ValueError:
            continue
        capture_time = {"local_datetime": local}
        offset = details.get(offset_tag, exif.get(offset_tag))
        if isinstance(offset, str) and _valid_capture_offset(offset.rstrip("\x00")):
            capture_time["utc_offset"] = offset.rstrip("\x00")
        metadata["capture_time"] = capture_time
        break
    return metadata


def _valid_capture_offset(value: str) -> bool:
    if re.fullmatch(r"[+-][0-9]{2}:[0-9]{2}", value) is None:
        return False
    hours, minutes = int(value[1:3]), int(value[4:])
    return minutes < 60 and (hours < 14 or (hours == 14 and minutes == 0))


def _gps_coordinate(value: object, reference: object, *, latitude: bool) -> float | None:
    if isinstance(reference, bytes):
        try:
            reference = reference.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(reference, str):
        return None
    reference = reference.rstrip("\x00")
    if reference not in (("N", "S") if latitude else ("E", "W")):
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        return None
    components = []
    for component in value:
        number = _nonnegative_finite_number(component)
        if number is None:
            return None
        components.append(number)
    degrees, minutes, seconds = components
    limit = 90 if latitude else 180
    coordinate = degrees + minutes / 60 + seconds / 3600
    if minutes >= 60 or seconds >= 60 or coordinate > limit:
        return None
    return round(-coordinate if reference in ("S", "W") else coordinate, 8)


def _nonnegative_finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def fetch_videos(
    load_video: Callable[..., object],
    videos: object,
    sample_indices_fn: Callable[..., object] | None = None,
) -> object:
    """Decode through PyAV while preserving the provider's sampling and metadata contract."""
    if isinstance(videos, list):
        decoded = (
            cast(tuple[object, object], fetch_videos(load_video, video, sample_indices_fn))
            for video in videos
        )
        return list(zip(*decoded, strict=False))
    return load_video(videos, backend="pyav", sample_indices_fn=sample_indices_fn)


class _Container(Protocol):
    duration: int | None


class _Stream(Protocol):
    duration: int | None
    time_base: Fraction | None


def media_duration_seconds(path: Path, *, stream_kind: Literal["audio", "video"]) -> float | None:
    """Return a positive stream/container duration when an installed decoder can read it."""
    try:
        import av

        with av.open(str(path)) as container:
            streams = container.streams.audio if stream_kind == "audio" else container.streams.video
            duration = container_duration_seconds(container, streams[0]) if streams else None
            if duration is not None:
                return duration
    except Exception:
        pass
    if stream_kind == "audio":
        try:
            import soundfile  # type: ignore[import-untyped]

            duration = float(soundfile.info(str(path)).duration)
            return duration if math.isfinite(duration) and duration > 0 else None
        except Exception:
            pass
    return None


def container_duration_seconds(
    container: _Container, stream: _Stream | None = None
) -> float | None:
    """Read duration from an already-open PyAV container without decoding it."""
    if stream is not None and stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
        if math.isfinite(duration) and duration > 0:
            return duration
    if container.duration is not None:
        import av

        duration = float(container.duration / av.time_base)
        if math.isfinite(duration) and duration > 0:
            return duration
    return None


def asset_url_part(
    asset: AssetRef,
    cache: dict[str, str],
) -> dict[str, object]:
    modality, _media_type, _path = resolved_asset(asset)
    kind = f"{modality.value}_url"
    url = cache.get(asset.id)
    if url is None:
        url = asset_url(asset)
        cache[asset.id] = url
    return {"type": kind, kind: {"url": url}}


def asset_url(asset: AssetRef) -> str:
    _modality, media_type, _path = resolved_asset(asset)
    return f"data:{media_type};base64,{asset_data(asset)}"


def asset_data(asset: AssetRef) -> str:
    _modality, _media_type, path = resolved_asset(asset)
    try:
        data = path.resolve(strict=True).read_bytes()
    except OSError:
        raise ModelError(
            "local media asset is unavailable",
            reason="asset_unavailable",
            subject=asset.id,
        ) from None
    if asset.size_bytes is not None and len(data) != asset.size_bytes:
        raise ModelError(
            "local media asset changed after ingestion",
            reason="asset_changed",
            subject=asset.id,
        )
    return base64.b64encode(data).decode("ascii")


def resolved_asset(asset: AssetRef) -> tuple[Modality, str, Path]:
    if not asset.is_resolved:
        raise ValidationError("asset reference must be resolved before model use")
    modality = asset.modality
    media_type = asset.media_type
    path = asset.path
    if modality is None or media_type is None or path is None:
        raise ValidationError("asset reference must be resolved before model use")
    return modality, media_type, path
