"""Request-local delivery from a caller-owned async context supplier."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

from sensemem._content import snapshot_content
from sensemem.content import ContentInput
from sensemem.context_values import ContextSupplement
from sensemem.exceptions import ValidationError


class ContextSupplementProvider(Protocol):
    def __call__(self, query: ContentInput) -> Awaitable[Sequence[ContextSupplement]]: ...


@dataclass(frozen=True, slots=True)
class ContextSourceResult:
    """Temporary context tied to the input revision returned by submit."""

    revision: int
    supplements: tuple[ContextSupplement, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise ValidationError("context source revision must be a positive integer")
        try:
            supplements = tuple(self.supplements)
        except TypeError:
            raise ValidationError("context source must return ContextSupplement values") from None
        if any(not isinstance(item, ContextSupplement) for item in supplements):
            raise ValidationError("context source must return ContextSupplement values")
        object.__setattr__(self, "supplements", supplements)


class AsyncContextSource:
    """Coalesce snapshots for an async supplier without writing to memory or starting an agent."""

    def __init__(
        self,
        provider: ContextSupplementProvider,
    ) -> None:
        if not callable(provider):
            raise ValidationError("context provider must be an async callable")
        self._provider = provider
        self._revision = 0
        self._submitted: tuple[int, ContentInput] | None = None
        self._pending: tuple[int, ContentInput] | None = None
        self._latest: ContextSourceResult | None = None
        self._failure: tuple[int, Exception] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._discard_results = False

    @property
    def latest(self) -> ContextSourceResult | None:
        """Read the last matching completion without waiting; inspect its revision before use."""
        return self._latest

    def submit(self, query: ContentInput) -> int:
        """Submit an immutable input snapshot and return its request-local revision."""
        if self._closed:
            raise ValidationError("context source is closed")
        loop = asyncio.get_running_loop()
        snapshot = snapshot_content(query)
        if (
            self._submitted is not None
            and self._submitted[1] == snapshot
            and (self._failure is None or self._failure[0] != self._submitted[0])
        ):
            return self._submitted[0]
        self._revision += 1
        self._submitted = (self._revision, snapshot)
        self._pending = self._submitted
        self._failure = None
        if self._worker is None:
            self._worker = loop.create_task(self._run())
        return self._revision

    async def finalize(self, query: ContentInput | None = None) -> ContextSourceResult:
        """Return only the final snapshot's result; supplier failures remain explicit."""
        if self._closed:
            raise ValidationError("context source is closed")
        if query is not None:
            self.submit(query)
        if self._submitted is None:
            raise ValidationError("context source has no submitted query")
        target = self._submitted[0]
        self._closed = True
        worker = self._worker
        if worker is not None:
            try:
                await worker
            except asyncio.CancelledError:
                self._pending = None
                self._discard_results = True
                raise
        if self._failure is not None and self._failure[0] == target:
            raise self._failure[1]
        if self._latest is not None and self._latest.revision == target:
            return self._latest
        raise ValidationError("context source was closed before the final result arrived")

    async def close(self) -> None:
        """Discard queued input and cancel and drain the supplier already running."""
        self._closed = True
        self._discard_results = True
        self._pending = None
        worker = self._worker
        if worker is not None:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    async def _run(self) -> None:
        try:
            while self._pending is not None:
                revision, query = self._pending
                self._pending = None
                try:
                    result = ContextSourceResult(revision, tuple(await self._provider(query)))
                except Exception as error:
                    if self._submitted is not None and revision == self._submitted[0]:
                        self._failure = (revision, error)
                    continue
                if (
                    not self._discard_results
                    and self._submitted is not None
                    and revision == self._submitted[0]
                ):
                    self._latest = result
        finally:
            self._worker = None
