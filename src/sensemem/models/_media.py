"""Shared local media decoding and model transport helpers."""

from __future__ import annotations

import base64
import math
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from sensemem.content import AssetRef
from sensemem.exceptions import ModelError, ValidationError
from sensemem.types import Modality

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


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


def load_mono_audio(path: Path, *, sampling_rate: int) -> NDArray[np.float32]:
    """Decode local audio without librosa's deprecated audioread fallback."""
    import av
    import numpy as np

    chunks: list[NDArray[np.float32]] = []
    try:
        with av.open(str(path)) as source:
            if not source.streams.audio:
                raise ModelError("embedding audio has no audio stream")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=sampling_rate)
            for frame in source.decode(audio=0):
                chunks.extend(
                    value.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                    for value in resampler.resample(frame)
                )
            chunks.extend(
                value.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                for value in resampler.resample(None)
            )
    except (av.FFmpegError, OSError):
        raise ModelError("failed to decode embedding audio") from None
    if not chunks:
        raise ModelError("embedding audio is empty")
    return np.concatenate(chunks).astype(np.float32, copy=False)


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
    if len(data) != asset.size_bytes:
        raise ModelError(
            "local media asset changed after ingestion",
            reason="asset_changed",
            subject=asset.id,
        )
    return base64.b64encode(data).decode("ascii")


def resolved_asset(asset: AssetRef) -> tuple[Modality, str, Path]:
    if not asset.is_resolved:
        raise ValidationError("asset reference must be resolved before model use")
    return cast(Modality, asset.modality), cast(str, asset.media_type), cast(Path, asset.path)
