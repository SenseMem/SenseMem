"""Shared evaluation admission limits and asynchronous memory boundary."""

from __future__ import annotations

import asyncio
from collections.abc import (
    Awaitable,
    Callable,
)
from pathlib import Path
from typing import Protocol

from sensemem import (
    AsyncMemory,
)
from sensemem.benchmarks.eval_results import (
    Retried,
)

EVAL_RUNNER_VERSION = "sensemem_eval_official_v28"


SDK_STREAM_LIMIT = 8


SDK_OPERATION_LIMIT = 28


class MemoryContext(Protocol):
    async def __aenter__(self) -> AsyncMemory: ...

    async def __aexit__(self, *_error: object) -> None: ...


MemoryFactory = Callable[[Path], MemoryContext]


SearchReplay = Callable[[], Awaitable[None]]


async def model_operation(
    semaphore: asyncio.Semaphore | None, operation: Callable[[], Awaitable[Retried]]
) -> Retried:
    if semaphore is None:
        return await operation()
    async with semaphore:
        pending = asyncio.ensure_future(operation())
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # Cancelling the waiter cannot stop synchronous SDK work or release its slot early.
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not pending.cancelled():
                pending.exception()
            raise
