"""Encode bounded local media for OpenAI-compatible model requests.

This module owns local asset integrity, image/audio/video codecs, attachment labels and
inline byte accounting. It never invokes a model client or interprets a model response.
"""

from __future__ import annotations

import base64
import hashlib
import math
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sensemem.content import AssetRef
from sensemem.evidence import retain_source_complete_hits
from sensemem.exceptions import ModelError
from sensemem.infrastructure.local.media_metadata import image_capture_metadata
from sensemem.models._media import (
    asset_data,
    asset_url_part,
    container_duration_seconds,
    resolved_asset,
)
from sensemem.models.base import ModelInput
from sensemem.records import SearchHit
from sensemem.types import Modality

if TYPE_CHECKING:
    from av import VideoFrame
    from PIL.Image import Image


_MAX_INLINE_MODEL_BYTES = 64 * 1024 * 1024

_MAX_INLINE_MODEL_ITEM_BYTES = 20 * 1024 * 1024

_ASR_SAMPLE_RATE = 16_000

_WAV_TYPE = "audio/wav"

_GENERATION_MODALITY_BY_PART_TYPE = {
    "image_url": Modality.IMAGE,
    "video_url": Modality.VIDEO,
    "input_audio": Modality.AUDIO,
}


@contextmanager
def _transcription_file(
    asset: AssetRef,
    modality: Modality,
    media_type: str,
    path: Path,
) -> Generator[tuple[str, object, str] | None, None, None]:
    """Yield one asset's multipart file part, or None when a video holds no audio stream."""
    if modality is not Modality.VIDEO:
        with path.open("rb") as stream:
            yield (asset.name or path.name, stream, media_type)
        return
    track = _demuxed_audio(path)
    yield None if track is None else (f"{Path(asset.name or path.name).stem}.wav", track, _WAV_TYPE)


def _demuxed_audio(path: Path) -> BytesIO | None:
    """Return a video's complete audio track as 16 kHz mono WAV, or None when it has none.

    `/v1/audio/transcriptions` accepts flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav and webm, while
    SenseMem also ingests .mkv, .mov, .avi and .ogv, so forwarding the container would have the
    provider reject every video format it does not list, and even for .mp4 it would spend the
    request size limit on frames the endpoint discards. Only the audio stream is read, so the
    sparse-video-track interleaving limit that applies when muxing does not arise here.

    Unlike `_video_frame_urls`, a failure here cannot fall back to the provider: with the frames
    the provider still receives the media, whereas skipping this demux would drop the speech
    silently. So this raises, and one missing decoder fails the write before inference.
    """
    try:
        import av
    except ImportError as error:
        raise ModelError(
            "transcribing a video requires the av package",
            reason="unsupported_modality",
            stage="transcribe",
        ) from error
    encoded = BytesIO()
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        # 16 kHz mono s16 is the input an ASR model consumes, so this normalizes to the
        # endpoint's own rate rather than discarding detail it would keep, and pcm_s16le is a
        # built-in FFmpeg codec rather than an optional library, so it is present in every PyAV
        # build. Both flushes below are what carry the tail of the track.
        # ponytail: PCM costs 32 kB/s, so one request holds roughly 13 minutes before the
        # provider's file-size limit rejects it. Move to libopus in Ogg (~2 hours per request)
        # once a single video longer than that is measured on this route.
        resampler = av.AudioResampler(format="s16", layout="mono", rate=_ASR_SAMPLE_RATE)
        with av.open(encoded, "w", format="wav") as target:
            stream = target.add_stream("pcm_s16le", rate=_ASR_SAMPLE_RATE, layout="mono")
            source = container.decode(container.streams.audio[0])
            for frame in chain(source, (None,)):
                for chunk in resampler.resample(frame):
                    for packet in stream.encode(chunk):
                        target.mux(packet)
            for packet in stream.encode(None):
                target.mux(packet)
    encoded.seek(0)
    return encoded


def _vision_asset_parts(
    asset: AssetRef,
    cache: dict[str, str],
    *,
    video_frames: int = 4,
) -> tuple[dict[str, object], ...]:
    if asset.modality is not Modality.VIDEO:
        return (_generation_asset_part(asset, cache),)
    frames = _video_frame_urls(asset, count=video_frames)
    if frames is None:
        # Sending the file instead would be a silent policy change: a caption is derived text
        # worth one small request, not worth uploading a whole clip to a generation endpoint.
        # The decoder ships with this adapter's own extra, so reaching here means the file has
        # no readable video stream or no readable duration.
        raise ModelError(
            "cannot decode video frames for description; the asset has no readable video "
            "stream, or drop video from the vision modalities",
            reason="asset_unavailable",
            subject=asset.id,
            stage="describe",
        )
    return tuple({"type": "image_url", "image_url": {"url": frame}} for frame in frames)


def _has_video(inputs: Sequence[ModelInput]) -> bool:
    return any(asset.modality is Modality.VIDEO for value in inputs for asset in value.assets)


def _generation_modalities(
    content: str | Sequence[Mapping[str, object]],
) -> frozenset[Modality]:
    modalities = {Modality.TEXT}
    if not isinstance(content, str):
        modalities.update(
            _GENERATION_MODALITY_BY_PART_TYPE[kind]
            for part in content
            if isinstance((kind := part.get("type")), str)
            and kind in _GENERATION_MODALITY_BY_PART_TYPE
        )
    return frozenset(modalities)


def _media_labels(question: ModelInput, hits: Sequence[SearchHit]) -> dict[str, str]:
    """Number each distinct media asset in order of first appearance: `attachment 1`, `attachment 2`.

    The label is what the request text uses to say which attached media belongs to which memory,
    and which memories share one. It replaces the asset ID -- a content hash -- that used to sit
    there under the key `assets`, because a reader asked for "the image ids" took those hashes to
    be the answer: on ATM-Bench-Hard, 2 of 12 list questions whose retrieval was otherwise
    complete were answered with asset hashes instead of the identifiers in `metadata`. A hash
    identifies nothing to the caller and is not an identifier the reader should ever repeat. The
    label deliberately does not name the media kind: labelled `image 1`, the same reader answered
    "image 1, image 2" to the same question, because the words matched the ones it was asked for.
    `_answer_parts` writes the same label as a text part right before the asset's media parts,
    which is what makes the number mean something when a video arrives as several stills. Both
    consumers call this function on the same question and hits rather than sharing a mapping:
    it is deterministic, so they agree by construction, and neither signature can be handed a
    mapping built from something else.
    """
    labels: dict[str, str] = {}
    for asset in (*question.assets, *(asset for hit in hits for asset in hit.assets)):
        if asset.id not in labels:
            labels[asset.id] = f"attachment {len(labels) + 1}"
    return labels


def _transport_label(
    label: str,
    sampled: Sequence[Mapping[str, object]] | None,
    modality: Modality | None,
) -> str:
    if sampled is None:
        return label
    if modality is Modality.IMAGE:
        return f"{label} (image re-encoded or resized; very fine detail may be reduced)"
    return (
        f"{label} (video sampled as {len(sampled)} ordered stills; motion, audio and exact "
        "timing are not supplied)"
    )


def _source_capture_metadata(
    assets: Sequence[AssetRef], *, cached: Mapping[str, dict[str, object]] | None = None
) -> dict[str, dict[str, object]]:
    """Expose only whitelisted metadata from integrity-checked original image bytes."""
    capture: dict[str, dict[str, object]] = {}
    for asset in _require_consistent_assets(assets):
        if asset.modality is not Modality.IMAGE:
            continue
        _modality, _media_type, path = resolved_asset(asset)
        fields = (
            cached[asset.id]
            if cached is not None and asset.id in cached
            else image_capture_metadata(path)
        )
        if not fields:
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
        except OSError:
            raise ModelError(
                "local media asset is unavailable", reason="asset_unavailable", subject=asset.id
            ) from None
        if digest.hexdigest() != asset.sha256:
            raise ModelError(
                "local media asset changed after ingestion",
                reason="asset_changed",
                subject=asset.id,
            )
        capture[asset.id] = fields
    return capture


def _require_inline_size(assets: Sequence[AssetRef]) -> None:
    """Bound the base64 a request will carry, checking every asset is present and unchanged."""
    _require_verified_inline_size(_require_asset_integrity(assets))


def _require_verified_inline_size(
    assets: Sequence[tuple[AssetRef, int]],
) -> None:
    size = 0
    for asset, actual_size in assets:
        encoded_size = _encoded_size(actual_size)
        if _inline_item_size(asset, actual_size) > _MAX_INLINE_MODEL_ITEM_BYTES:
            raise ModelError(
                "encoded inline model media item exceeds 20 MiB; use a provider upload adapter",
                reason="payload_too_large",
            )
        size += encoded_size
    if size > _MAX_INLINE_MODEL_BYTES:
        raise ModelError(
            "encoded inline model media exceeds 64 MiB; use a provider upload adapter",
            reason="payload_too_large",
        )


def _require_asset_integrity(
    assets: Sequence[AssetRef],
) -> tuple[tuple[AssetRef, int], ...]:
    verified = []
    for asset in assets:
        _modality, _media_type, path = resolved_asset(asset)
        try:
            actual_size = path.resolve(strict=True).stat().st_size
        except OSError:
            raise ModelError(
                "local media asset is unavailable",
                reason="asset_unavailable",
                subject=asset.id,
            ) from None
        if asset.size_bytes != actual_size:
            raise ModelError(
                "local media asset changed after ingestion",
                reason="asset_changed",
                subject=asset.id,
            )
        verified.append((asset, actual_size))
    return tuple(verified)


def _image_transport_recipe(max_pixels: int) -> str:
    return f":image-rgb-jpeg85-s2-lanczos-v1:{max_pixels}:{_MAX_INLINE_MODEL_ITEM_BYTES}"


def _image_transport_parts(
    assets: Sequence[AssetRef],
    max_pixels: int | None,
    *,
    capture_metadata: dict[str, dict[str, object]] | None = None,
) -> dict[str, tuple[dict[str, object], ...]]:
    if max_pixels is None:
        return {}
    parts: dict[str, tuple[dict[str, object], ...]] = {}
    for asset in _require_consistent_assets(assets):
        if asset.modality is not Modality.IMAGE:
            continue
        url = _bounded_image_url(asset, max_pixels, capture_metadata=capture_metadata)
        if url is not None:
            parts[asset.id] = ({"type": "image_url", "image_url": {"url": url}},)
            _require_inline_media_parts(tuple(part for group in parts.values() for part in group))
    return parts


def _bounded_image_url(  # noqa: C901 - reuse source headers while bounding image transport
    asset: AssetRef,
    max_pixels: int,
    *,
    capture_metadata: dict[str, dict[str, object]] | None = None,
    max_encoded_bytes: int = _MAX_INLINE_MODEL_ITEM_BYTES,
) -> str | None:
    """Inspect headers before decoding, and transform only when a source needs transport bounds."""
    _require_asset_integrity((asset,))
    _modality, _media_type, path = resolved_asset(asset)
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise ModelError(
            "bounded image transport requires Pillow", reason="backend_not_configured"
        ) from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                if capture_metadata is not None:
                    capture_metadata[asset.id] = image_capture_metadata(path, opened_image=source)
                pixels = source.width * source.height
                if pixels > 64_000_000:
                    raise ModelError(
                        "image source exceeds the 64 million pixel decode bound",
                        reason="payload_too_large",
                        subject=asset.id,
                    )
                if (
                    pixels <= max_pixels
                    and _inline_item_size(asset, cast(int, asset.size_bytes)) <= max_encoded_bytes
                ):
                    return None
                image = (
                    ImageOps.exif_transpose(source) if source.getexif().get(274, 1) != 1 else source
                )
                bound = max_pixels
                while True:
                    if image.width * image.height > bound:
                        scale = math.sqrt(bound / (image.width * image.height))
                        width = max(1, min(image.width, bound, int(image.width * scale)))
                        height = max(1, min(image.height, bound // width))
                        image = image.resize((width, height), Image.Resampling.LANCZOS)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    encoded = BytesIO()
                    image.save(encoded, format="JPEG", quality=85, subsampling=2, optimize=False)
                    if (
                        len("data:image/jpeg;base64,") + _encoded_size(encoded.tell())
                        <= max_encoded_bytes
                    ):
                        return "data:image/jpeg;base64," + base64.b64encode(
                            encoded.getvalue()
                        ).decode("ascii")
                    if image.width * image.height == 1:
                        break
                    bound = max(1, image.width * image.height // 2)
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ModelError(
            "image cannot be decoded for bounded transport",
            reason="payload_too_large",
            subject=asset.id,
        ) from error
    raise ModelError(
        "image cannot fit the inline byte limit", reason="payload_too_large", subject=asset.id
    )


def _embedding_video_sampling(
    assets: Sequence[AssetRef],
    *,
    image_parts: Mapping[str, tuple[dict[str, object], ...]] | None = None,
) -> frozenset[str]:
    verified = _require_asset_integrity(assets)
    sampled = frozenset(
        asset.id
        for asset, size in verified
        if asset.modality is Modality.VIDEO
        and _inline_item_size(asset, size) > _MAX_INLINE_MODEL_ITEM_BYTES
    )
    # Sampling is a property of the source, never of the batch it happens to share.
    _require_verified_inline_size(
        tuple(
            value
            for value in verified
            if value[0].id not in sampled
            and (image_parts is None or value[0].id not in image_parts)
        )
    )
    return sampled


def _require_inline_media_parts(parts: Sequence[Mapping[str, object]]) -> None:
    total = 0
    for part in parts:
        kind = part.get("type")
        if kind == "input_audio":
            encoded = cast(Mapping[str, str], part[kind])["data"]
            total += len(encoded.encode())
            continue
        if kind not in {"image_url", "video_url", "audio_url"}:
            continue
        url = cast(Mapping[str, str], part[kind])["url"]
        if len(url.encode()) > _MAX_INLINE_MODEL_ITEM_BYTES:
            raise ModelError(
                "encoded inline model media item exceeds 20 MiB; use a provider upload adapter",
                reason="payload_too_large",
            )
        total += len(url.partition(",")[2].encode())
    if total > _MAX_INLINE_MODEL_BYTES:
        raise ModelError(
            "encoded inline model media exceeds 64 MiB; use a provider upload adapter",
            reason="payload_too_large",
        )


def _encoded_size(size: int) -> int:
    """Return the base64 length the request carries for a media file of ``size`` bytes."""
    return (size + 2) // 3 * 4


def _inline_item_size(asset: AssetRef, size: int) -> int:
    return len(f"data:{asset.media_type};base64,".encode()) + _encoded_size(size)


def _require_consistent_assets(assets: Sequence[AssetRef]) -> tuple[AssetRef, ...]:
    unique: dict[str, AssetRef] = {}
    for asset in assets:
        existing = unique.setdefault(asset.id, asset)
        if existing != asset:
            raise ModelError("one asset ID has conflicting media descriptors")
    return tuple(unique.values())


def _fit_grounding_media(  # noqa: C901 - price one request while retaining source references
    question: ModelInput,
    hits: Sequence[SearchHit],
    *,
    video_limit: int | None,
    sampled_parts: dict[str, tuple[dict[str, object], ...]] | None = None,
    sample_videos: bool = False,
    image_max_pixels: int | None = None,
    capture_metadata: dict[str, dict[str, object]] | None = None,
) -> tuple[SearchHit, ...]:
    sampled_parts = {} if sampled_parts is None else sampled_parts
    attempted: set[str] = set()
    failed: set[str] = set()

    def prepare(asset: AssetRef, *, required: bool = False) -> bool:
        if asset.id in failed:
            return False
        if asset.id not in attempted:
            attempted.add(asset.id)
            if image_max_pixels is not None and asset.modality is Modality.IMAGE:
                try:
                    sampled_parts.update(
                        _image_transport_parts(
                            (asset,), image_max_pixels, capture_metadata=capture_metadata
                        )
                    )
                except ModelError as error:
                    if error.reason != "payload_too_large" or required:
                        raise
                    failed.add(asset.id)
                    return False
            elif sample_videos:
                parts = _oversized_video_parts(asset)
                if parts is not None:
                    sampled_parts[asset.id] = parts
        return True

    for asset in question.assets:
        prepare(asset, required=True)
    seen = {asset.id for asset in question.assets}
    used = sum(
        _transport_size(asset, sampled_parts)
        for asset in _require_consistent_assets(question.assets)
    )
    videos = 0
    selected = []
    for hit in hits:
        assets = []
        for asset in hit.assets:
            if asset.id in seen:
                assets.append(asset)
                continue
            if (
                asset.modality is Modality.VIDEO
                and video_limit is not None
                and videos >= video_limit
            ):
                continue
            if not prepare(asset):
                continue
            size = cast(int, asset.size_bytes)
            encoded_size = _transport_size(asset, sampled_parts)
            if (
                asset.id not in sampled_parts
                and _inline_item_size(asset, size) > _MAX_INLINE_MODEL_ITEM_BYTES
            ) or used + encoded_size > _MAX_INLINE_MODEL_BYTES:
                continue
            assets.append(asset)
            seen.add(asset.id)
            used += encoded_size
            if asset.modality is Modality.VIDEO:
                videos += 1
        if assets:
            admitted = tuple(assets)
            selected.append(
                replace(
                    hit,
                    assets=admitted,
                    modality=ModelInput(assets=admitted).modality,
                )
            )
        elif not hit.assets:
            selected.append(hit)
        elif hit.content.strip():
            selected.append(replace(hit, assets=(), modality=Modality.TEXT))
    return retain_source_complete_hits(selected, hits)


def _transport_size(
    asset: AssetRef,
    sampled_parts: Mapping[str, tuple[dict[str, object], ...]] | None,
) -> int:
    if sampled_parts is not None and asset.id in sampled_parts:
        return _image_parts_size(sampled_parts[asset.id])
    return _encoded_size(cast(int, asset.size_bytes))


def _oversized_video_parts(
    asset: AssetRef,
) -> tuple[dict[str, object], ...] | None:
    if (
        asset.modality is not Modality.VIDEO
        or _inline_item_size(asset, cast(int, asset.size_bytes)) <= _MAX_INLINE_MODEL_ITEM_BYTES
    ):
        return None
    _require_asset_integrity((asset,))
    frames = _video_frame_urls(asset, count=16)
    if frames is None:
        return None
    parts: tuple[dict[str, object], ...] = tuple(
        {"type": "image_url", "image_url": {"url": frame}} for frame in frames
    )
    try:
        _require_inline_media_parts(parts)
    except ModelError:
        return None
    return parts


def _omitted_grounding_media(
    question: ModelInput,
    retrieved: Sequence[SearchHit],
    grounded: Sequence[SearchHit],
) -> dict[str, dict[str, int]]:
    """Count unique source media absent from the whole generation request, by retained hit."""
    supplied = {asset.id for asset in question.assets}
    supplied.update(asset.id for hit in grounded for asset in hit.assets)
    grounded_ids = {hit.id for hit in grounded}
    omitted: dict[str, dict[str, int]] = {}
    for hit in retrieved:
        if hit.id not in grounded_ids:
            continue
        absent = {
            asset.id: cast(Modality, asset.modality).value
            for asset in hit.assets
            if asset.id not in supplied
        }
        counts: dict[str, int] = {}
        for modality in absent.values():
            counts[modality] = counts.get(modality, 0) + 1
        if counts:
            omitted[hit.id] = dict(sorted(counts.items()))
    return omitted


def _input_parts(
    value: ModelInput,
    cache: dict[str, str],
    *,
    sample_video: bool = False,
    sampled_assets: frozenset[str] = frozenset(),
    image_parts: Mapping[str, tuple[dict[str, object], ...]] | None = None,
) -> tuple[list[dict[str, object]], bool]:
    """Build one input's request parts, reporting whether any video was replaced by stills.

    A video whose duration cannot be read is sent whole even on the sampling retry, so presence
    of video is not evidence that sampling happened; the flag is what keeps the recorded count
    from claiming a degradation that did not occur.
    """
    parts: list[dict[str, object]] = []
    sampled = False
    for item in value.ordered_content:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
            continue
        asset = item
        if image_parts is not None and asset.id in image_parts:
            parts.extend(image_parts[asset.id])
            continue
        frames = (
            _video_frame_urls(asset)
            if asset.modality is Modality.VIDEO and (sample_video or asset.id in sampled_assets)
            else None
        )
        if frames is None:
            if asset.id in sampled_assets:
                raise ModelError(
                    "oversized video cannot be decoded within the inline media limits",
                    reason="payload_too_large",
                    subject=asset.id,
                    stage="embed",
                )
            parts.append(asset_url_part(asset, cache))
            continue
        sampled = True
        parts.extend({"type": "image_url", "image_url": {"url": frame}} for frame in frames)
    return parts, sampled


def _generation_asset_part(
    asset: AssetRef,
    cache: dict[str, str],
    *,
    audio_data_url: bool = False,
) -> dict[str, object]:
    modality, media_type, path = resolved_asset(asset)
    if modality is Modality.VIDEO and audio_data_url:
        url = cache.get(asset.id)
        if url is None:
            url = _qwen_video_url(asset)
            cache[asset.id] = url
        return {"type": "video_url", "video_url": {"url": url}}
    if modality is not Modality.AUDIO:
        return asset_url_part(asset, cache)
    encoded = cache.get(asset.id)
    if encoded is None:
        encoded = asset_data(asset)
        cache[asset.id] = encoded
    return {
        "type": "input_audio",
        "input_audio": {
            # DashScope requires a data URL here; OpenAI requires raw Base64.
            "data": f"data:;base64,{encoded}" if audio_data_url else encoded,
            "format": _audio_format(asset.name or path.name, media_type),
        },
    }


def _qwen_video_url(asset: AssetRef) -> str:  # noqa: C901 - keep packet ownership and flushes together
    """Replace unsupported Opus audio on the wire; copy every video packet unchanged.

    Browser MP4 recordings can contain Opus, which Qwen Omni rejects even when its H264
    video is supported. This request-local conversion preserves the original durable asset,
    frame timestamps and audio offsets; a silent video or an already compatible track keeps
    its original container bytes.
    """
    import av

    _modality, media_type, _path = resolved_asset(asset)
    original = asset_data(asset)
    try:
        with av.open(BytesIO(base64.b64decode(original)), "r") as source:
            if not any(stream.codec_context.name == "opus" for stream in source.streams.audio):
                return f"data:{media_type};base64,{original}"
            encoded = BytesIO()
            with av.open(
                encoded, "w", format="mp4", options={"movie_timescale": "1000000"}
            ) as target:
                streams: dict[int, av.stream.Stream] = {}
                audio: dict[int, tuple[av.audio.stream.AudioStream, av.AudioResampler]] = {}
                for stream in source.streams:
                    if stream.type == "audio" and stream.codec_context.name == "opus":
                        original_audio = cast(av.audio.stream.AudioStream, stream)
                        converted = target.add_stream("aac", rate=original_audio.rate)
                        converted.layout = original_audio.layout
                        converted.bit_rate = 192_000
                        streams[stream.index] = converted
                        audio[stream.index] = (
                            converted,
                            av.AudioResampler(
                                format="fltp", layout=converted.layout, rate=converted.rate
                            ),
                        )
                    elif stream.type in {"audio", "video"}:
                        streams[stream.index] = target.add_stream_from_template(stream)
                for packet in source.demux():
                    index = packet.stream.index
                    if index in audio:
                        converted, resampler = audio[index]
                        for frame in packet.decode():
                            for chunk in resampler.resample(cast(av.AudioFrame, frame)):
                                target.mux(converted.encode(chunk))
                    elif index in streams and packet.dts is not None:
                        packet.stream = streams[index]
                        target.mux(packet)
                for converted, resampler in audio.values():
                    for chunk in resampler.resample(None):
                        target.mux(converted.encode(chunk))
                    target.mux(converted.encode(None))
            return f"data:video/mp4;base64,{base64.b64encode(encoded.getvalue()).decode('ascii')}"
    except (av.FFmpegError, OSError, ValueError):
        raise ModelError(
            "cannot encode video audio for the model provider",
            reason="asset_unavailable",
            subject=asset.id,
            stage="generate",
        ) from None


def _generation_asset_parts(
    asset: AssetRef,
    cache: dict[str, str],
    minimum_video_seconds: float | None,
    maximum_encoded_bytes: int,
    *,
    audio_data_url: bool = False,
) -> tuple[dict[str, object], ...]:
    if asset.modality is Modality.VIDEO and minimum_video_seconds is not None:
        frames = _video_frame_urls(asset, below_seconds=minimum_video_seconds)
        if (
            frames is not None
            and all(len(frame.encode()) <= _MAX_INLINE_MODEL_ITEM_BYTES for frame in frames)
            and sum(len(frame.partition(",")[2].encode()) for frame in frames)
            <= maximum_encoded_bytes
        ):
            # Ordered stills retain visual evidence, but cannot retain native video timing.
            return tuple({"type": "image_url", "image_url": {"url": frame}} for frame in frames)
    return (_generation_asset_part(asset, cache, audio_data_url=audio_data_url),)


def _image_parts_size(parts: Sequence[Mapping[str, object]]) -> int:
    return sum(
        len(cast(Mapping[str, str], part["image_url"])["url"].partition(",")[2].encode())
        for part in parts
    )


def _video_frame_urls(  # noqa: C901 - bounded ordered sampling in one decode pass
    asset: AssetRef,
    *,
    below_seconds: float | None = None,
    count: int = 4,
    max_frame_bytes: int | None = None,
) -> tuple[str, ...] | None:
    """Decode ordered JPEG stills, optionally only for a video under a provider limit."""
    try:
        import av

        _modality, _media_type, path = resolved_asset(asset)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = container_duration_seconds(container, stream)
            if duration is None:
                return None
            if below_seconds is not None and duration >= below_seconds:
                return None
            start = (
                float(stream.start_time * stream.time_base)
                if stream.start_time is not None and stream.time_base is not None
                else 0.0
            )
            targets = tuple(start + duration * index / (count - 1) for index in range(count - 1))
            sampled: list[str] = []
            last: VideoFrame | None = None
            last_url: str | None = None
            encoded_bytes = 0
            for index, frame in enumerate(container.decode(stream)):
                last = frame
                last_url = None
                timestamp = frame.time
                if timestamp is None and stream.average_rate:
                    timestamp = start + index / float(stream.average_rate)
                while (
                    timestamp is not None
                    and len(sampled) < len(targets)
                    and timestamp >= targets[len(sampled)]
                ):
                    if last_url is None:
                        last_url = (
                            _jpeg_frame_url(frame)
                            if max_frame_bytes is None
                            else _jpeg_frame_url(frame, max_encoded_bytes=max_frame_bytes)
                        )
                    sampled.append(last_url)
                    encoded_bytes += len(last_url.partition(",")[2])
                    if (
                        len(last_url) > _MAX_INLINE_MODEL_ITEM_BYTES
                        or encoded_bytes > _MAX_INLINE_MODEL_BYTES
                    ):
                        return None
            if last is None:
                return None
            last_url = last_url or (
                _jpeg_frame_url(last)
                if max_frame_bytes is None
                else _jpeg_frame_url(last, max_encoded_bytes=max_frame_bytes)
            )
            remaining = count - len(sampled)
            if (
                len(last_url) > _MAX_INLINE_MODEL_ITEM_BYTES
                or encoded_bytes + remaining * len(last_url.partition(",")[2])
                > _MAX_INLINE_MODEL_BYTES
            ):
                return None
            sampled.extend(last_url for _ in range(remaining))
            return tuple(sampled)
    except Exception:
        # PyAV/Pillow are optional; decode failures preserve the existing provider fallback path.
        return None


def _jpeg_frame_url(frame: VideoFrame, *, max_encoded_bytes: int | None = None) -> str:
    if max_encoded_bytes is not None:
        from PIL import Image as PILImage

        image = cast("Callable[[], Image]", frame.to_image)().convert("RGB")
        while True:
            encoded = BytesIO()
            image.save(encoded, format="JPEG", quality=85)
            url = "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
            if len(url) <= max_encoded_bytes:
                return url
            if image.width * image.height == 1:
                raise ValueError("video frame cannot fit the request byte budget")
            bound = max(1, image.width * image.height // 2)
            scale = math.sqrt(bound / (image.width * image.height))
            width = max(1, min(image.width, bound, int(image.width * scale)))
            height = max(1, min(image.height, bound // width))
            image = image.resize((width, height), PILImage.Resampling.LANCZOS)
    encoded = BytesIO()
    cast("Callable[[], Image]", frame.to_image)().convert("RGB").save(
        encoded, format="JPEG", quality=85
    )
    return f"data:image/jpeg;base64,{base64.b64encode(encoded.getvalue()).decode('ascii')}"


def _audio_format(name: str, media_type: str) -> str:
    suffix = Path(name).suffix.removeprefix(".").lower()
    if suffix:
        return suffix
    subtype = media_type.split("/", 1)[1].removeprefix("x-")
    return "mp3" if subtype == "mpeg" else subtype
