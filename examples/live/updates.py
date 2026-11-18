"""Push native final events and revision-bound context beside continuous capture."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from examples.live.memories import EventSummary, memory_cards
from examples.live.recall import LiveRecall
from sensemem import AsyncMemory, AsyncOmniPrefetch, ContextBudget, SenseMemError, StreamCommit

Trigger = Literal["visual", "audio", "memory"]
EventModality = Literal["audio", "image", "video"]


@dataclass(frozen=True, slots=True)
class LiveEvent:
    """A committed activity or observed window with immutable original evidence IDs."""

    event_id: str
    source_id: str | None
    evidence_ids: tuple[str, ...]
    summary: str
    occurred_at: datetime
    occurred_end: datetime
    modalities: tuple[EventModality, ...]


Assessor = Callable[[LiveEvent, int, Trigger], Awaitable[dict[str, object] | None]]


class _NativeEvent(EventSummary):
    supplied_modalities: tuple[EventModality, ...]


class _VisualWindow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    perception_id: str = Field(strict=True, min_length=1)
    boundary: Literal["observed_window"]


@dataclass(slots=True)
class _Parts:
    boundary: _NativeEvent
    source_id: str | None
    evidence: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Work:
    event: LiveEvent
    revision: int
    trigger: Trigger


class LiveUpdates:
    """Coalesce background recall and assessment without delaying native media ingestion.

    Native event and visual-window capture supply original evidence. Dialogue writes and
    model formation invalidate context without creating a fresh observed window.
    """

    def __init__(
        self, memory: AsyncMemory, recall: LiveRecall, assessor: Assessor | None = None
    ) -> None:
        self.memory = memory
        self.recall = recall
        self.assessor = assessor
        self.memory_revision = 0
        self.latest: LiveEvent | None = None
        self.parts: dict[str, _Parts] = {}
        self.changed = asyncio.Event()
        self.notices = asyncio.Event()
        self.announced = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.notifier: asyncio.Task[None] | None = None
        self.closed = False
        self._work: _Work | None = None
        self._published_event: str | None = None
        self._pending_revision: int | None = None
        self._pending_event: LiveEvent | None = None

    def observe(self, commit: StreamCommit) -> None:
        """Accept only durable native event commits; perform no model work here."""
        if self.closed:
            return
        try:
            boundary = _NativeEvent.model_validate(commit.record.metadata.get("event_segmentation"))
        except PydanticValidationError:
            self._observe_visual(commit)
            return
        if self.latest is not None and boundary.event_id == self.latest.event_id:
            return
        context = commit.record.context
        group = self.parts.setdefault(
            boundary.event_id,
            _Parts(boundary, None if context is None else context.source_id),
        )
        group.evidence[boundary.part_index] = commit.record.id
        if len(group.evidence) != boundary.part_count:
            return
        self.parts.pop(boundary.event_id)
        self._enqueue(
            LiveEvent(
                event_id=boundary.event_id,
                source_id=group.source_id,
                evidence_ids=tuple(group.evidence[index] for index in range(boundary.part_count)),
                summary=boundary.summary,
                occurred_at=boundary.occurred_at,
                occurred_end=boundary.occurred_end,
                modalities=boundary.supplied_modalities,
            )
        )

    def _observe_visual(self, commit: StreamCommit) -> None:
        record = commit.record
        try:
            window = _VisualWindow.model_validate(record.metadata.get("visual_observation"))
        except PydanticValidationError:
            return
        summary = "\n".join(
            part.value for part in record.parts if part.kind == "model_interpretation"
        )
        if not summary or not record.assets or record.occurred_at is None:
            return
        self._enqueue(
            LiveEvent(
                event_id=window.perception_id,
                source_id=None if record.context is None else record.context.source_id,
                evidence_ids=(record.id,),
                summary=summary,
                occurred_at=record.occurred_at,
                occurred_end=record.occurred_end or record.occurred_at,
                modalities=("image",),
            )
        )

    def _enqueue(self, event: LiveEvent) -> None:
        if (
            self.latest is not None
            and event.source_id == self.latest.source_id
            and event.occurred_end < self.latest.occurred_end
        ):
            return
        self.latest = event
        trigger: Trigger = (
            "visual" if {"image", "video"}.intersection(self.latest.modalities) else "audio"
        )
        self._work = _Work(self.latest, self.memory_revision, trigger)
        self._pending_event = self.latest
        self._start()

    def memory_changed(self, revision: int, *, trigger: Trigger = "memory") -> None:
        """Invalidate active results synchronously and reevaluate only the latest event."""
        if self.closed:
            return
        self.memory_revision = revision
        self._pending_revision = revision
        if self.latest is not None:
            self._work = _Work(self.latest, revision, trigger)
        self._start()

    def _start(self) -> None:
        # Withdrawal must reach the browser while an older native model call drains.
        self.notices.set()
        self.changed.set()
        if self.notifier is None or self.notifier.done():
            self.notifier = asyncio.create_task(self._notify())
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())

    def _current(self, work: _Work) -> bool:
        return not self.closed and self._work is work and work.revision == self.memory_revision

    async def _notify(self) -> None:
        while not self.closed:
            await self.notices.wait()
            self.notices.clear()
            if self.closed:
                return
            if (revision := self._pending_revision) is not None:
                self._pending_revision = None
                await self.recall.publish({"type": "invalidated", "memory_revision": revision})
                await self.recall.publish({"type": "memory_changed", "memory_revision": revision})
            if (event := self._pending_event) is not None:
                self._pending_event = None
                await self.recall.publish(
                    {
                        "type": "memory_event",
                        "event_id": event.event_id,
                        "source_id": event.source_id,
                        "evidence_ids": list(event.evidence_ids),
                        "summary": event.summary,
                        "summary_verified": False,
                        "occurred_at": event.occurred_at.isoformat(),
                        "occurred_end": event.occurred_end.isoformat(),
                        "memory_revision": self.memory_revision,
                    }
                )
                self._published_event = event.event_id
                self.announced.set()

    async def _run(self) -> None:
        while not self.closed:
            await self.changed.wait()
            self.changed.clear()
            if self.closed:
                return
            if (work := self._work) is None:
                continue
            while self._current(work) and self._published_event != work.event.event_id:
                self.announced.clear()
                await self.announced.wait()
            if not self._current(work):
                continue
            if self.assessor is None:
                await self._context(work)
            else:
                await asyncio.gather(self._context(work), self._assess(work))

    async def _context(self, work: _Work) -> None:
        helper = AsyncOmniPrefetch(
            self.memory,
            budget=ContextBudget(max_chars=64_000, max_items=4, max_media_items=8),
        )
        helper.submit(work.event.summary)
        try:
            result = await helper.finalize()
            if not self._current(work):
                return
            assert result.bundle is not None
            hits = await memory_cards(self.memory, result.hits)
            if not self._current(work):
                return
            await self.recall.publish(
                {
                    "type": "context",
                    "event_id": work.event.event_id,
                    "evidence_ids": list(work.event.evidence_ids),
                    "context": result.bundle.render(),
                    "hits": hits,
                    "memory_revision": work.revision,
                }
            )
        except SenseMemError:
            # Provider bodies stay private; a later native commit or settlement retries work.
            if self._current(work):
                await self.recall.publish(
                    {"type": "error", "code": "context_failed", "memory_revision": work.revision}
                )
        finally:
            await helper.close()

    async def _assess(self, work: _Work) -> None:
        assert self.assessor is not None
        try:
            candidate = await self.assessor(work.event, work.revision, work.trigger)
            # The assessor revalidates actual evidence after model work. Later observations
            # do not withdraw this completed past event or relabel its original evidence.
            if (
                candidate is not None
                and not self.closed
                and candidate.get("memory_revision") == self.memory_revision
            ):
                await self.recall.publish(
                    {
                        **candidate,
                        "type": "proactive",
                        "event_id": work.event.event_id,
                        "source_id": work.event.source_id,
                        "evidence_ids": list(work.event.evidence_ids),
                        "trigger": work.trigger,
                    }
                )
        except SenseMemError:
            if self._current(work):
                await self.recall.publish(
                    {"type": "error", "code": "context_failed", "memory_revision": work.revision}
                )

    async def close(self) -> None:
        """Discard queued work and drain the current native operation before memory closes."""
        self.closed = True
        self.changed.set()
        self.notices.set()
        self.announced.set()
        await asyncio.gather(*(task for task in (self.task, self.notifier) if task is not None))
        self.parts.clear()
