"""Persist observed visual windows without waiting for an ongoing activity to end."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from examples.live.perception import PerceptionFrame, PerceptionResult
from sensemem import (
    AsyncCaptureStream,
    AsyncMemory,
    Blob,
    MemoryCategory,
    ObservationContext,
    SenseMemError,
    StreamCommit,
    StreamEvent,
    StreamInput,
    StreamPhase,
)


@dataclass(frozen=True, slots=True)
class _Window:
    result: PerceptionResult
    frames: tuple[PerceptionFrame, ...]
    perception_id: str


class VisualMemoryCapture:
    """Capture original images; keep the model's reading as a separate interpretation.

    One native write runs in the background. At most two reviewed windows wait behind it:
    the first accepted window and the latest meaningful change. The independent continuous
    audiovisual recording remains the full source when intermediate reviews are coalesced.
    """

    def __init__(
        self,
        memory: AsyncMemory,
        source_id: str,
        *,
        committed: Callable[[StreamCommit], None],
        changed: Callable[[], None] | None = None,
        model_id: str | None = None,
    ) -> None:
        self.stream = AsyncCaptureStream(memory, capture=True)
        self.source_id = source_id
        self.committed = committed
        self.changed = changed
        self.model_id = model_id
        self.pending: deque[_Window] = deque()
        self.task: asyncio.Task[None] | None = None
        self.last_epoch = -1
        self.last_sequence = 0
        self.failure: str | None = None
        self.processing = False
        self.closed = False

    def submit(
        self,
        result: PerceptionResult,
        frames: tuple[PerceptionFrame, ...],
        *,
        perception_id: str,
    ) -> None:
        """Accept an initial or changed review immediately, without waiting for storage."""
        if self.closed or not frames:
            return
        if result.epoch == self.last_epoch and (
            frames[-1].sequence <= self.last_sequence or not result.changed
        ):
            return
        self.last_epoch = result.epoch
        self.last_sequence = frames[-1].sequence
        window = _Window(result, frames, perception_id)
        if len(self.pending) == 2:
            self.pending[-1] = window
        else:
            self.pending.append(window)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._consume())

    async def _events(self) -> AsyncIterator[StreamEvent]:
        while self.pending:
            window = self.pending.popleft()
            result, frames = window.result, window.frames
            yield StreamEvent(
                StreamPhase.FINAL,
                StreamInput(
                    tuple(
                        Blob(frame.jpeg, "image/jpeg", f"frame-{frame.sequence}.jpg")
                        for frame in frames
                    ),
                    occurred_at=frames[0].captured_at,
                    occurred_end=frames[-1].captured_at
                    if frames[-1].captured_at > frames[0].captured_at
                    else None,
                    memory_type=MemoryCategory.EPISODIC,
                    context=ObservationContext(source_id=self.source_id),
                    interpretation=result.summary,
                    metadata={
                        "visual_observation": {
                            "perception_id": window.perception_id,
                            "boundary": "observed_window",
                            "frame_sequences": [frame.sequence for frame in frames],
                            "frames": [
                                {
                                    "sequence": frame.sequence,
                                    "captured_at": frame.captured_at.isoformat(),
                                    "timestamp_ms": frame.timestamp_ms,
                                }
                                for frame in frames
                            ],
                            "changed": result.changed,
                            "model_id": self.model_id,
                            "confidence": result.confidence,
                            "evidence": [item.model_dump() for item in result.evidence],
                        }
                    },
                ),
                window.perception_id,
            )

    async def _consume(self) -> None:
        self.processing = True
        if self.changed is not None:
            self.changed()
        try:
            while self.pending:
                try:
                    async for commit in self.stream.consume(self._events()):
                        self.failure = None
                        self.committed(commit)
                except SenseMemError:
                    self.failure = "visual_capture_failed"
                    if self.changed is not None:
                        self.changed()
        finally:
            self.processing = False
            if self.changed is not None:
                self.changed()

    async def close(self) -> None:
        """Drain accepted observations without completing the ongoing audiovisual activity."""
        self.closed = True
        if self.task is not None:
            await self.task
