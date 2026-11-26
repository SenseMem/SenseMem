"""Shared local media decoding and model transport helpers."""

from __future__ import annotations

import base64
import math
from collections.abc import Callable
from contextlib import suppress
from fractions import Fraction
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from sensemem.content import AssetRef
from sensemem.exceptions import ModelError, ValidationError
from sensemem.types import Modality


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


class _SoundFileInfo(Protocol):
    duration: float


class _SoundFileModule(Protocol):
    def info(self, file: str) -> _SoundFileInfo: ...


def media_duration_seconds(path: Path, *, stream_kind: Literal["audio", "video"]) -> float | None:
    """Return a positive stream/container duration when an installed decoder can read it."""
    try:
        import av
    except ImportError:
        av_module = None
    else:
        av_module = av
    if av_module is not None:
        with (
            suppress(av_module.FFmpegError, OSError, ValueError),
            av_module.open(str(path)) as container,
        ):
            streams = container.streams.audio if stream_kind == "audio" else container.streams.video
            duration = container_duration_seconds(container, streams[0]) if streams else None
            if duration is not None:
                return duration
    if stream_kind == "audio":
        try:
            soundfile = cast(_SoundFileModule, import_module("soundfile"))
        except ImportError:
            return None
        try:
            duration = float(soundfile.info(str(path)).duration)
        except (OSError, RuntimeError, ValueError):
            return None
        return duration if math.isfinite(duration) and duration > 0 else None
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
    if (
        not asset.is_resolved
        or asset.modality is None
        or asset.media_type is None
        or asset.path is None
    ):
        raise ValidationError("asset reference must be resolved before model use")
    return asset.modality, asset.media_type, asset.path
