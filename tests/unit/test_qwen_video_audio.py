"""Qwen-compatible video transport preserves source video and complete audio timing."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import cast

import av
import httpx2 as httpx
import numpy as np
import pytest
from numpy.typing import NDArray
from openai import OpenAI

from sensemem import Modality, ModelInput
from sensemem.content import AssetRef
from sensemem.models import _openai_media as media
from sensemem.models.base import EventObservation
from sensemem.models.openai_completion import OpenAICompletion


def _clip(tmp_path: Path, *, audio: str | None = "libopus") -> AssetRef:
    path = tmp_path / "camera.mp4"
    with av.open(str(path), "w", format="mp4") as target:
        video = target.add_stream("libx264", rate=10)
        video.width = video.height = 32
        video.pix_fmt = "yuv420p"
        sound = (
            cast(av.audio.stream.AudioStream, target.add_stream(audio, rate=48_000))
            if audio
            else None
        )
        if sound is not None:
            sound.layout = "mono"
        for index in range(20):
            image = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), index * 10, dtype=np.uint8), format="rgb24"
            )
            image.pts = index
            image.time_base = Fraction(1, 10)
            target.mux(video.encode(image))
            if sound is not None:
                time = np.arange(4800, dtype=np.float32) + index * 4800
                wave = (np.sin(time * (2 * np.pi * 440 / 48_000)) * 0.2).reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(wave, format="flt", layout="mono")
                frame.sample_rate = 48_000
                frame.time_base = Fraction(1, 48_000)
                # A real offset catches conversions that independently zero the tracks.
                frame.pts = index * 4800 + 12_000
                target.mux(sound.encode(frame))
        target.mux(video.encode(None))
        if sound is not None:
            target.mux(sound.encode(None))
    data = path.read_bytes()
    return AssetRef(
        id="camera",
        modality=Modality.VIDEO,
        media_type="video/mp4",
        name=path.name,
        path=path,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _wire(asset: AssetRef, cache: dict[str, str] | None = None, *, qwen: bool = True) -> bytes:
    part = media._generation_asset_part(
        asset, cache if cache is not None else {}, audio_data_url=qwen
    )
    assert part["type"] == "video_url"
    url = cast(dict[str, str], part["video_url"])["url"]
    assert url.startswith("data:video/mp4;base64,")
    return base64.b64decode(url.partition(",")[2])


def _video_packets(data: bytes) -> list[tuple[bytes, Fraction, Fraction, Fraction]]:
    with av.open(BytesIO(data), "r") as source:
        packets = []
        for packet in source.demux(video=0):
            if packet.dts is not None:
                assert (
                    packet.pts is not None
                    and packet.time_base is not None
                    and packet.duration is not None
                )
                packets.append(
                    (
                        bytes(packet),
                        packet.pts * packet.time_base,
                        packet.dts * packet.time_base,
                        packet.duration * packet.time_base,
                    )
                )
        return packets


def _sound(data: bytes) -> tuple[str, int, Fraction, Fraction, NDArray[np.float32]]:
    with av.open(BytesIO(data), "r") as source:
        stream = source.streams.audio[0]
        frames = list(source.decode(audio=0))
        first, last = frames[0], frames[-1]
        assert first.pts is not None and first.time_base is not None
        assert last.pts is not None and last.time_base is not None
        return (
            stream.codec_context.name,
            sum(frame.samples for frame in frames),
            first.pts * first.time_base,
            last.pts * last.time_base + Fraction(last.samples, last.sample_rate),
            np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames]),
        )


def test_qwen_opus_video_copies_packets_preserves_audio_offset_and_tail(tmp_path: Path) -> None:
    asset = _clip(tmp_path)
    assert asset.path is not None
    original = asset.path.read_bytes()
    wire = _wire(asset)

    assert wire != original
    assert asset.path.read_bytes() == original
    assert hashlib.sha256(original).hexdigest() == asset.sha256
    assert _video_packets(wire) == _video_packets(original)
    source_codec, source_samples, source_start, source_end, source_wave = _sound(original)
    wire_codec, wire_samples, wire_start, wire_end, wire_wave = _sound(wire)
    assert source_codec == "opus" and wire_codec == "aac"
    # AAC keeps one frame of encoder priming before a nonzero audio start. Neither that
    # frame nor final padding may shift or truncate the original, complete sample sequence.
    priming = round((source_start - wire_start) * 48_000)
    assert 0 <= priming <= 1024
    assert 0 <= wire_samples - priming - source_samples < 1024
    assert 0 <= wire_end - source_end < Fraction(1024, 48_000)
    assert np.corrcoef(source_wave, wire_wave[priming : priming + source_samples])[0, 1] > 0.98


@pytest.mark.parametrize("codec", (None, "aac"))
def test_qwen_compatible_video_keeps_original_container(tmp_path: Path, codec: str | None) -> None:
    asset = _clip(tmp_path, audio=codec)
    assert asset.path is not None
    assert _wire(asset) == asset.path.read_bytes()


def test_other_models_keep_opus_and_qwen_reuses_wire_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _clip(tmp_path)
    assert asset.path is not None
    assert _wire(asset, qwen=False) == asset.path.read_bytes()
    cache: dict[str, str] = {}
    first = _wire(asset, cache)

    def unexpected_read(_asset: AssetRef) -> str:
        pytest.fail("a reused attachment must use its already encoded wire value")

    monkeypatch.setattr(media, "asset_data", unexpected_read)
    assert _wire(asset, cache) == first


@pytest.mark.parametrize("model_name", ("qwen3.8-omni-flash", "qwen/qwen3.8-omni-flash", "other"))
def test_event_segmentation_normalizes_only_qwen_wire_audio(
    tmp_path: Path, model_name: str
) -> None:
    asset = _clip(tmp_path)
    assert asset.path is not None
    original = asset.path.read_bytes()
    started = datetime(2026, 11, 17, tzinfo=timezone.utc)
    observation = EventObservation(
        ModelInput(assets=(asset,)), started, started + timedelta(seconds=3)
    )

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == model_name
        parts = body["messages"][1]["content"]
        assert [part["type"] for part in parts] == ["text", "text", "video_url"]
        assert json.loads(parts[1]["text"])["occurred_at"] == started.isoformat()
        data = base64.b64decode(parts[2]["video_url"]["url"].partition(",")[2])
        assert _video_packets(data) == _video_packets(original)
        assert _sound(data)[0] == ("opus" if model_name == "other" else "aac")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"events":[]}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        model = OpenAICompletion(
            OpenAI(api_key="local-fixture", http_client=transport, max_retries=0),
            generation_model=model_name,
            generation_capabilities=frozenset({Modality.VIDEO}),
        )
        assert model.segment_events((observation,), end_of_stream=True) == ()
    assert asset.path.read_bytes() == original
