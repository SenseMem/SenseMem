"""Keep durable stream input independent of semantic model execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from sensemem import AsyncEventStream, EventChunk, StreamCommit, ValidationError


class EventCapture:
    """One native stream, with a single background processor and a durable inbox."""

    def __init__(
        self,
        stream: AsyncEventStream,
        work: asyncio.Event,
        observe: Callable[[StreamCommit], None] | None = None,
        changed: Callable[[], None] | None = None,
    ) -> None:
        self.stream = stream
        self.work = work
        self.observe = observe
        self.state_changed = changed
        self.lock = asyncio.Lock()
        self.changed = asyncio.Event()
        self.closed = False
        self.processing = False
        self.failure: Exception | None = None
        self.task: asyncio.Task[None] | None = None

    async def _consume(self) -> None:
        while True:
            await self.changed.wait()
            self.changed.clear()
            ending = self.closed
            self.processing = True
            if self.state_changed is not None:
                self.state_changed()
            try:
                async for commit in self.stream.process(end_of_stream=ending):
                    self.work.set()
                    if self.observe is not None:
                        self.observe(commit)
            except Exception as error:
                # Already acknowledged input remains in the native checkpoint for retry.
                # Expose a safe failure flag instead of provider response bodies.
                self.failure = error
                return
            else:
                self.failure = None
            finally:
                self.processing = False
                if self.state_changed is not None:
                    self.state_changed()
            if ending:
                return

    def _start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._consume())

    async def push(self, chunk: EventChunk) -> None:
        async with self.lock:
            if self.closed:
                raise ValidationError("the capture source is closed")
            await self.stream.append(chunk)
            self.changed.set()
            self._start()

    async def finish(self) -> None:
        """Close input promptly; background processing and unfinished originals survive."""
        async with self.lock:
            self.closed = True
            self.changed.set()
            self._start()

    async def wait(self) -> None:
        if self.task is not None:
            await self.task
        if self.failure is not None:
            raise self.failure
