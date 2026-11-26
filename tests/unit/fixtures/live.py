"""Shared live-demo test fakes and media helpers."""

import asyncio
from collections.abc import Sequence
from datetime import datetime
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import av
import numpy as np

from sensemem import (
    AsyncMemory,
    ContentInput,
    ContextBudget,
    ContextBundle,
    ContextSupplement,
    EventObservation,
    EventSegment,
    Memory,
    MemorySpec,
    ModelError,
    RetrievalScope,
)


class Segmenter:
    event_model = "experience-fixture"
    event_recipe = "experience-fixture:v1"

    def segment_events(
        self, observations: Sequence[EventObservation], *, end_of_stream: bool = False
    ) -> tuple[EventSegment, ...]:
        _ = end_of_stream
        return tuple(
            EventSegment(index + 1, "A complete observed activity")
            for index in range(len(observations))
        )


def live_config(path: Path, *, generation: bool = False, lifecycle: bool = False) -> MemorySpec:
    return MemorySpec.model_validate(
        {
            "data_dir": path,
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai", "model": "fixture"} if generation else None,
            "formation": {"provider": "openai", "model": "fixture"} if lifecycle else None,
            "consolidation": {"provider": "openai", "model": "fixture"} if lifecycle else None,
        }
    )


def make_recording(*, video: bool = True, frame_count: int = 24) -> bytes:
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


class ControlledMemory(AsyncMemory):
    def __init__(self, memory: Memory) -> None:
        super().__init__(memory)
        self.calls: list[ContentInput] = []
        self.started: asyncio.Queue[ContentInput] = asyncio.Queue()
        self.completed: asyncio.Queue[ContextBundle] = asyncio.Queue()
        self.release = asyncio.Event()
        self.block = False
        self.fail: str | None = None

    async def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        allow_partial_sources: bool = False,
        supplements: Sequence[ContextSupplement] = (),
    ) -> ContextBundle:
        self.calls.append(goal)
        self.started.put_nowait(goal)
        if self.block and len(self.calls) == 1:
            await self.release.wait()
        if goal == self.fail:
            raise ModelError("private provider failure", reason="timeout")
        result = await super().compile(
            goal,
            budget=budget,
            reference_at=reference_at,
            scope=scope,
            allow_partial_sources=allow_partial_sources,
            supplements=supplements,
        )
        self.completed.put_nowait(result)
        return result
