"""Demux one continuous browser recording into independently decodable codec groups."""

from __future__ import annotations

import asyncio
import io
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from threading import Condition

import av

from examples.live.capture import EventCapture
from sensemem import Blob, EventChunk, ValidationError

MAX_BUFFER_BYTES = 8 * 1024 * 1024
# A conservative transport target; native models alone decide semantic event boundaries.
MIN_VIDEO_SPAN = Fraction(1)
_MP4_OPTIONS = {
    "movflags": "empty_moov+delay_moov+default_base_moof+frag_keyframe",
    "movie_timescale": "1000000",
    "brand": "isom",
}


@dataclass(frozen=True, slots=True)
class _CodecGroup:
    data: bytes
    start: Fraction
    end: Fraction


class _BufferedMediaInput(io.RawIOBase):
    """A bounded blocking reader for FFmpeg, fed by the asynchronous WebSocket."""

    def __init__(self) -> None:
        super().__init__()
        self.condition = Condition()
        self.buffers: deque[bytes] = deque()
        self.size = 0
        self.ended = False

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        with self.condition:
            self.condition.wait_for(lambda: bool(self.buffers) or self.ended)
            if not self.buffers:
                return b""
            data = self.buffers.popleft()
            if 0 < size < len(data):
                self.buffers.appendleft(data[size:])
                data = data[:size]
            self.size -= len(data)
            self.condition.notify_all()
            return data

    def feed(self, data: bytes) -> None:
        if len(data) > MAX_BUFFER_BYTES:
            raise ValidationError(
                "media packet exceeds the input budget", reason="capacity_exceeded"
            )
        with self.condition:
            self.condition.wait_for(lambda: self.ended or self.size + len(data) <= MAX_BUFFER_BYTES)
            if self.ended:
                raise ValidationError("the media source is closed")
            self.buffers.append(data)
            self.size += len(data)
            self.condition.notify_all()

    def finish(self) -> None:
        with self.condition:
            self.ended = True
            self.condition.notify_all()


class MediaReceiver:
    """Copy original packets, retaining a predecessor for a short final codec tail.

    Transport flushes may split any MP4 box. They never become EventChunk boundaries.
    The source is demuxed once; joining closed groups never decodes or re-encodes packets.
    """

    def __init__(self, capture: EventCapture, started_at: datetime) -> None:
        self.capture = capture
        self.started_at = started_at
        self.input = _BufferedMediaInput()
        self.loop = asyncio.get_running_loop()
        self.task = asyncio.create_task(asyncio.to_thread(self._consume))

    async def feed(self, data: bytes) -> None:
        if self.task.done():
            await self.task
        await asyncio.to_thread(self.input.feed, data)

    async def finish(self) -> None:
        self.input.finish()
        await self.task

    def _consume(self) -> None:
        try:
            self._demux()
        finally:
            self.input.finish()

    def _demux(self) -> None:  # noqa: C901 - keep codec packet ownership and cleanup together
        with av.open(
            self.input,
            mode="r",
            format="mp4",
            options={"probesize": "32768", "analyzeduration": "0"},
        ) as source:
            if not source.streams.video:
                raise ValidationError("the recording requires a video track")
            target: av.container.OutputContainer | None = None
            output = io.BytesIO()
            streams: dict[int, av.stream.Stream] = {}
            start = Fraction(0)
            end = Fraction(0)
            video_start: Fraction | None = None
            video_end = Fraction(0)
            packet_bytes = 0
            pending: _CodecGroup | None = None
            try:
                for packet in source.demux():
                    if packet.dts is None or packet.stream.type not in {"audio", "video"}:
                        continue
                    assert packet.time_base is not None
                    timestamp = (
                        Fraction(packet.pts if packet.pts is not None else packet.dts)
                        * packet.time_base
                    )
                    if (
                        packet.stream.type == "video"
                        and packet.is_keyframe
                        and video_start is not None
                        and video_end - video_start >= MIN_VIDEO_SPAN
                    ):
                        assert target is not None
                        target.close()
                        target = None
                        group = _CodecGroup(output.getvalue(), start, timestamp)
                        if (
                            len(group.data) + (len(pending.data) if pending else 0)
                            > MAX_BUFFER_BYTES
                        ):
                            raise ValidationError(
                                "codec group exceeds the input budget", reason="capacity_exceeded"
                            )
                        if pending is not None:
                            self._append(pending.data, pending.start, pending.end)
                        pending = group
                        start = timestamp
                        output = io.BytesIO()
                        video_start = None
                        video_end = Fraction(0)
                        packet_bytes = 0
                    if target is None:
                        target = av.open(
                            output,
                            "w",
                            format="mp4",
                            # isom matches native browser MP4; edit lists preserve AV offsets.
                            options=_MP4_OPTIONS,
                        )
                        streams = {
                            stream.index: target.add_stream_from_template(stream)
                            for stream in source.streams
                            if stream.type in {"audio", "video"}
                        }
                    end = max(end, timestamp + (packet.duration or 0) * packet.time_base)
                    if packet.stream.type == "video":
                        if video_start is None:
                            video_start = timestamp
                        video_end = max(
                            video_end, timestamp + (packet.duration or 0) * packet.time_base
                        )
                    offset = round(start / packet.time_base)
                    if packet.pts is not None:
                        packet.pts -= offset
                    packet.dts -= offset
                    packet.stream = streams[packet.stream.index]
                    packet_bytes += packet.size
                    held_bytes = len(pending.data) if pending else 0
                    if packet_bytes + held_bytes > MAX_BUFFER_BYTES:
                        raise ValidationError(
                            "codec group exceeds the input budget", reason="capacity_exceeded"
                        )
                    target.mux(packet)
                    if output.tell() + held_bytes > MAX_BUFFER_BYTES:
                        raise ValidationError(
                            "codec group exceeds the input budget", reason="capacity_exceeded"
                        )
                if target is not None:
                    target.close()
                    target = None
                    if video_start is not None and end > start:
                        group = _CodecGroup(output.getvalue(), start, end)
                        if (
                            len(group.data) + (len(pending.data) if pending else 0)
                            > MAX_BUFFER_BYTES
                        ):
                            raise ValidationError(
                                "codec group exceeds the input budget", reason="capacity_exceeded"
                            )
                        if pending is not None and video_end - video_start < MIN_VIDEO_SPAN:
                            group = self._merge(pending, group)
                            pending = None
                        if pending is not None:
                            self._append(pending.data, pending.start, pending.end)
                        self._append(group.data, group.start, group.end)
            finally:
                if target is not None:
                    target.close()

    def _merge(self, first: _CodecGroup, second: _CodecGroup) -> _CodecGroup:
        output = io.BytesIO()
        with av.open(output, "w", format="mp4", options=_MP4_OPTIONS) as target:
            streams: dict[int, av.stream.Stream] | None = None
            for group in (first, second):
                with av.open(io.BytesIO(group.data), "r", format="mp4") as source:
                    if streams is None:
                        streams = {
                            stream.index: target.add_stream_from_template(stream)
                            for stream in source.streams
                            if stream.type in {"audio", "video"}
                        }
                    for packet in source.demux():
                        if packet.dts is None or packet.stream.type not in {"audio", "video"}:
                            continue
                        assert packet.time_base is not None
                        offset = round((group.start - first.start) / packet.time_base)
                        if packet.pts is not None:
                            packet.pts += offset
                        packet.dts += offset
                        packet.stream = streams[packet.stream.index]
                        target.mux(packet)
                        if output.tell() > MAX_BUFFER_BYTES:
                            raise ValidationError(
                                "codec group exceeds the input budget", reason="capacity_exceeded"
                            )
        return _CodecGroup(output.getvalue(), first.start, second.end)

    def _append(self, data: bytes, start: Fraction, end: Fraction) -> None:
        if len(data) > MAX_BUFFER_BYTES:
            raise ValidationError(
                "codec group exceeds the input budget", reason="capacity_exceeded"
            )
        chunk = EventChunk(
            Blob(data, "video/mp4", "camera.mp4"),
            self.started_at + timedelta(seconds=float(start)),
            self.started_at + timedelta(seconds=float(end)),
        )
        # append() waits for the native checkpoint, never for a model response.
        asyncio.run_coroutine_threadsafe(self.capture.push(chunk), self.loop).result()
