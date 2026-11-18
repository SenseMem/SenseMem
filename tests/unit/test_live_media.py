"""The continuous browser container retains every original encoded A/V packet."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from io import BytesIO
from itertools import cycle, pairwise
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING

import av
import numpy as np
import pytest

from sensemem import EventChunk, ValidationError

if TYPE_CHECKING:
    from examples.live.media import MediaReceiver


@pytest.fixture(autouse=True)
def example_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))


def _recording(*, video: bool = True, frame_count: int = 24) -> bytes:
    encoded = BytesIO()
    with av.open(
        encoded,
        "w",
        format="mp4",
        options={"movflags": "empty_moov+default_base_moof+frag_keyframe"},
    ) as target:
        visual = target.add_stream("libvpx-vp9", rate=10) if video else None
        if visual is not None:
            visual.width = visual.height = 16
            visual.pix_fmt = "yuv420p"
            visual.codec_context.gop_size = 7
            visual.codec_context.options = {"deadline": "realtime", "cpu-used": "8"}
        audio = target.add_stream("libopus", rate=48_000)
        audio.layout = "mono"
        for index in range(frame_count):
            if visual is not None:
                frame = av.VideoFrame.from_ndarray(
                    np.full((16, 16, 3), index % 256, dtype=np.uint8), format="rgb24"
                )
                frame.pts = index
                frame.time_base = Fraction(1, 10)
                for packet in visual.encode(frame):
                    target.mux(packet)
            sound = av.AudioFrame.from_ndarray(np.zeros((1, 4800), dtype=np.float32), format="flt")
            sound.sample_rate = 48_000
            sound.pts = index * 4800
            sound.time_base = Fraction(1, 48_000)
            for packet in audio.encode(sound):
                target.mux(packet)
        if visual is not None:
            for packet in visual.encode():
                target.mux(packet)
        for packet in audio.encode():
            target.mux(packet)
    return encoded.getvalue()


def _receiver(
    started_at: datetime, *, progress: asyncio.Event | None = None
) -> tuple[MediaReceiver, list[EventChunk]]:
    from examples.live.capture import EventCapture
    from examples.live.media import MediaReceiver

    class RecordingCapture(EventCapture):
        def __init__(self) -> None:
            self.chunks: list[EventChunk] = []

        async def push(self, chunk: EventChunk) -> None:
            self.chunks.append(chunk)
            if progress is not None:
                progress.set()

    capture = RecordingCapture()
    return MediaReceiver(capture, started_at), capture.chunks


def _fragments(data: bytes) -> Iterator[bytes]:
    position = 0
    for size in cycle((1, 17, 257, 3, 4096, 31)):
        if position >= len(data):
            return
        yield data[position : position + size]
        position += size


def _packets(data: bytes, kind: str) -> list[bytes]:
    with av.open(BytesIO(data), mode="r") as source:
        return [
            bytes(packet)
            for packet in source.demux()
            if packet.dts is not None and packet.stream.type == kind
        ]


def _packet_times(data: bytes, kind: str) -> list[float]:
    with av.open(BytesIO(data), mode="r") as source:
        return [
            float(packet.pts * packet.time_base)
            for packet in source.demux()
            if packet.dts is not None
            and packet.pts is not None
            and packet.time_base is not None
            and packet.stream.type == kind
        ]


@pytest.mark.asyncio
async def test_arbitrary_transport_flushes_preserve_codec_groups_and_every_original_packet() -> (
    None
):
    recording = _recording(frame_count=60)
    started = datetime(2026, 11, 16, 10, tzinfo=timezone.utc)
    with av.open(BytesIO(recording), mode="r") as source:
        keyframes = [
            Fraction(packet.pts) * packet.time_base
            for packet in source.demux(video=0)
            if packet.dts is not None and packet.is_keyframe and packet.pts is not None
        ]
    assert len(keyframes) == 9
    assert [float(b - a) for a, b in pairwise(keyframes)] == pytest.approx([0.7] * 8, abs=0.01)
    receiver, chunks = _receiver(started)
    for fragment in _fragments(recording):
        await receiver.feed(fragment)
    await asyncio.wait_for(receiver.finish(), 5)
    assert len(chunks) == 4
    assert chunks[0].occurred_at == started
    assert [chunk.occurred_end for chunk in chunks[:-1]] == [
        started + timedelta(seconds=float(keyframes[index])) for index in (2, 4, 6)
    ]
    assert all(left.occurred_end == right.occurred_at for left, right in pairwise(chunks))
    frames_per_group = []
    for chunk in chunks:
        assert chunk.occurred_end > chunk.occurred_at
        assert (chunk.occurred_end - chunk.occurred_at).total_seconds() >= 1
        blob = chunk.media[0]
        assert blob.media_type == "video/mp4" and blob.name == "camera.mp4"
        assert blob.data[4:12] == b"ftypisom"
        with av.open(BytesIO(blob.data), mode="r") as source:
            assert len(source.streams.video) == len(source.streams.audio) == 1
            assert source.streams.video[0].codec_context.name == "vp9"
            assert source.streams.audio[0].codec_context.name == "opus"
            frames = list(source.decode(video=0))
            assert frames[0].key_frame
            frames_per_group.append(len(frames))
        with av.open(BytesIO(blob.data), mode="r") as source:
            assert list(source.decode(audio=0))
    # The four-frame EOF tail stays with its preceding eligible group.
    assert frames_per_group == [14, 14, 14, 18]
    for kind in ("audio", "video"):
        assert [
            packet for chunk in chunks for packet in _packets(chunk.media[0].data, kind)
        ] == _packets(recording, kind)
        assert [
            timestamp + (chunk.occurred_at - started).total_seconds()
            for chunk in chunks
            for timestamp in _packet_times(chunk.media[0].data, kind)
        ] == pytest.approx(_packet_times(recording, kind), abs=1 / 48_000)


@pytest.mark.asyncio
async def test_complete_groups_reach_native_capture_before_browser_eof() -> None:
    progress = asyncio.Event()
    receiver, chunks = _receiver(datetime.now(timezone.utc), progress=progress)
    try:
        await receiver.feed(_recording(frame_count=80))
        await asyncio.wait_for(progress.wait(), 3)
        assert chunks and not receiver.input.ended and not receiver.task.done()
    finally:
        await receiver.finish()


@pytest.mark.asyncio
async def test_whole_call_shorter_than_transport_target_preserves_originals_without_padding() -> (
    None
):
    recording = _recording(frame_count=5)
    started = datetime.now(timezone.utc)
    receiver, chunks = _receiver(started)
    await receiver.feed(recording)
    await asyncio.wait_for(receiver.finish(), 5)
    assert len(chunks) == 1
    assert 0 < (chunks[0].occurred_end - started).total_seconds() < 1
    data = chunks[0].media[0].data
    for kind in ("audio", "video"):
        assert _packets(data, kind) == _packets(recording, kind)
        assert _packet_times(data, kind) == pytest.approx(
            _packet_times(recording, kind), abs=1 / 48_000
        )
    with av.open(BytesIO(data), mode="r") as source:
        assert len(list(source.decode(video=0))) == 5


@pytest.mark.asyncio
async def test_pending_predecessor_and_current_group_share_one_buffer_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.live import media

    recording = _recording(frame_count=28)
    reference, groups = _receiver(datetime.now(timezone.utc))
    await reference.feed(recording)
    await reference.finish()
    assert len(groups) == 2
    limit = max(len(group.media[0].data) for group in groups) + 1
    monkeypatch.setattr(media, "MAX_BUFFER_BYTES", limit)
    receiver, chunks = _receiver(datetime.now(timezone.utc))
    try:
        with suppress(ValidationError):
            for position in range(0, len(recording), 512):
                await receiver.feed(recording[position : position + 512])
        with pytest.raises(ValidationError, match="codec group exceeds") as failure:
            await receiver.finish()
        assert failure.value.reason == "capacity_exceeded"
        assert chunks == []
    finally:
        receiver.input.finish()
        with suppress(ValidationError):
            await receiver.finish()


def test_input_reader_enforces_eight_mib_and_backpressures_until_bytes_are_consumed() -> None:
    from examples.live.media import MAX_BUFFER_BYTES, _BufferedMediaInput

    assert MAX_BUFFER_BYTES == 8 * 1024 * 1024
    reader = _BufferedMediaInput()
    with pytest.raises(ValidationError, match="budget"):
        reader.feed(b"x" * (MAX_BUFFER_BYTES + 1))
    reader.feed(b"a" * MAX_BUFFER_BYTES)
    entered, complete = Event(), Event()

    def feed_one_more() -> None:
        entered.set()
        reader.feed(b"b")
        complete.set()

    producer = Thread(target=feed_one_more)
    producer.start()
    try:
        assert entered.wait(5)
        assert not complete.wait(0.02)
        assert reader.size == MAX_BUFFER_BYTES
        assert reader.read(1) == b"a"
        assert complete.wait(5)
        assert reader.size == MAX_BUFFER_BYTES
        reader.finish()
        assert reader.read(MAX_BUFFER_BYTES) == b"a" * (MAX_BUFFER_BYTES - 1)
        assert reader.read() == b"b"
        assert reader.read() == b""
        with pytest.raises(ValidationError, match="closed"):
            reader.feed(b"c")
    finally:
        reader.finish()
        producer.join(5)
        assert not producer.is_alive()
        reader.close()


@pytest.mark.asyncio
async def test_garbage_recording_fails_without_creating_a_memory_chunk() -> None:
    receiver, chunks = _receiver(datetime.now(timezone.utc))
    try:
        await receiver.feed(b"This is not an MP4 container.")
        with pytest.raises(av.error.InvalidDataError):
            await asyncio.wait_for(receiver.finish(), 5)
        assert chunks == []
    finally:
        receiver.input.finish()


@pytest.mark.asyncio
async def test_audio_only_recording_is_rejected_without_creating_a_memory_chunk() -> None:
    receiver, chunks = _receiver(datetime.now(timezone.utc))
    try:
        await receiver.feed(_recording(video=False))
        with pytest.raises(ValidationError, match="video track"):
            await asyncio.wait_for(receiver.finish(), 5)
        assert chunks == []
    finally:
        receiver.input.finish()


@pytest.mark.asyncio
async def test_completed_mp4_group_enforces_container_byte_budget() -> None:
    from examples.live.media import MAX_BUFFER_BYTES

    receiver, chunks = _receiver(datetime.now(timezone.utc))
    try:
        with pytest.raises(ValidationError, match="codec group") as failure:
            await asyncio.wait_for(
                asyncio.to_thread(
                    receiver._append,
                    b"x" * (MAX_BUFFER_BYTES + 1),
                    Fraction(0),
                    Fraction(1),
                ),
                3,
            )
        assert failure.value.reason == "capacity_exceeded"
        assert not chunks
    finally:
        receiver.input.finish()
        with suppress(av.error.InvalidDataError):
            await receiver.finish()
