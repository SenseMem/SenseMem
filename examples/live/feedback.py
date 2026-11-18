"""Project current native recall evidence into bounded, attributed historical facts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, TypedDict
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from examples.live.memories import MemoryCard, memory_cards
from sensemem import AsyncMemory, MemoryNotFoundError, MemoryRecord, ModelError, SenseMemError
from sensemem.prompts import demo as demo_prompts

Publisher = Callable[[dict[str, object]], Awaitable[None]]


class _Fact(TypedDict):
    memory_id: str
    text: str
    occurred_at: str | None
    verification: Literal["source_review", "record_text"]
    asset_ids: list[str]


class _Input(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["sensory_recall", "context"]
    source_id: str | None = Field(default=None, strict=True)
    memory_revision: int = Field(strict=True, ge=0)
    query: str = Field(default="与当前音视频相关的历史记录", strict=True, max_length=2000)
    hits: tuple[MemoryCard, ...] = Field(max_length=16)
    expires_at: AwareDatetime | None = None


class _Dialogue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    origin: str | None = Field(default=None, strict=True)
    speaker: str | None = Field(default=None, strict=True)


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    speaker: str | None = Field(default=None, strict=True)
    dialogue: _Dialogue | None = None


@dataclass(frozen=True, slots=True)
class _Work:
    query: str
    revision: int
    ids: tuple[str, ...]
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class _Evidence:
    record: MemoryRecord
    role: str


class MemoryFeedback:
    """Drain one review while coalescing the latest pending recall; never block input."""

    def __init__(
        self,
        memory: AsyncMemory,
        publish: Publisher,
        source_id: str,
        *,
        revision: int = 0,
    ) -> None:
        self.memory = memory
        self.publish = publish
        self.source_id = source_id
        self.revision = revision
        self.closed = False
        self.task: asyncio.Task[None] | None = None
        self._pending: _Work | None = None

    def observe(self, message: dict[str, object]) -> None:
        """Observe known native result events synchronously, without model or store calls."""
        kind = message.get("type")
        if (
            self.closed
            or kind not in {"sensory_recall", "context"}
            or (kind == "sensory_recall" and message.get("status") != "ready")
        ):
            return
        try:
            supplied = _Input.model_validate(message)
        except PydanticValidationError:
            return
        if (
            supplied.memory_revision != self.revision
            or supplied.source_id not in {None, self.source_id}
            or (kind == "sensory_recall" and supplied.source_id != self.source_id)
        ):
            return
        now = datetime.now(timezone.utc)
        expires_at = min(
            supplied.expires_at or now + timedelta(seconds=20), now + timedelta(seconds=20)
        )
        if expires_at <= now:
            return
        self._pending = _Work(
            supplied.query,
            supplied.memory_revision,
            tuple(dict.fromkeys(hit.id for hit in supplied.hits)),
            expires_at,
        )
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())

    def invalidate(self, revision: int) -> None:
        """Withdraw all work from earlier native memory revisions immediately."""
        if revision < self.revision:
            return
        self.revision = revision
        if self._pending is not None and self._pending.revision != revision:
            self._pending = None

    def _current(self, work: _Work) -> bool:
        return not self.closed and work.revision == self.revision

    async def _run(self) -> None:
        while not self.closed and (work := self._pending) is not None:
            self._pending = None
            if not self._current(work) or work.expires_at <= datetime.now(timezone.utc):
                continue
            try:
                facts, records, abstained = await self._project(work)
                hits = await memory_cards(self.memory, records)
            except SenseMemError:
                if self._current(work):
                    await self.publish(
                        self._message(work, status="error", code="memory_feedback_failed")
                    )
                continue
            if self._current(work):
                await self.publish(self._message(work, facts=facts, hits=hits, abstained=abstained))

    def _message(self, work: _Work, **values: object) -> dict[str, object]:
        return {
            "type": "memory_feedback",
            "status": "ready",
            "feedback_id": uuid4().hex,
            "source_id": self.source_id,
            "memory_revision": work.revision,
            # The input deadline admits work; historical source review can outlast a
            # camera snapshot without making the retained original cease to exist.
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=20)).isoformat(),
            "query": work.query,
            "facts": [],
            "hits": [],
            "abstained": False,
            **values,
        }

    async def _load(self, work: _Work) -> list[_Evidence]:
        evidence: list[_Evidence] = []
        for memory_id in work.ids:
            try:
                record = await self.memory.get(memory_id)
            except MemoryNotFoundError:
                continue
            context = record.context
            if record.forgotten_at is not None or (
                context is not None and (not context.visible or context.retired_at is not None)
            ):
                continue
            try:
                metadata = _Metadata.model_validate(record.metadata)
            except PydanticValidationError:
                continue
            dialogue = metadata.dialogue
            agent = metadata.speaker == "assistant" or (
                dialogue is not None
                and (
                    dialogue.speaker == "assistant"
                    or dialogue.origin in {"realtime", "proactive", "perception"}
                )
            )
            if agent and context is not None and context.source_id == self.source_id:
                continue
            role = "Agent记录" if agent else "对话记录" if dialogue is not None else "历史记录"
            evidence.append(_Evidence(record, role))
            if len(evidence) == 2:
                break
        if not evidence:
            return []
        pending = {
            item.memory_id
            for item in await self.memory.pending_captures(
                limit=2, memory_ids=tuple(item.record.id for item in evidence)
            )
        }
        return [item for item in evidence if item.record.id not in pending]

    async def _fact(self, work: _Work, item: _Evidence, budget: int) -> _Fact | None:
        record = item.record
        asset_ids: list[str] = []
        verification: Literal["source_review", "record_text"]
        if record.assets:
            asset = record.assets[0]
            blob = await self.memory.read_asset(record.id, asset.id)
            if not self._current(work):
                return None
            prefix = f"{item.role}原片复核(模型): "
            question = demo_prompts.feedback_question(
                work.query,
                memory_id=record.id,
                occurred_at=None if record.occurred_at is None else record.occurred_at.isoformat(),
                role=item.role,
                max_chars=budget - len(prefix),
            )
            reviewed = await self.memory.ask((question, blob), limit=1, answer_policy="strict")
            if reviewed.abstained:
                return None
            text = prefix + reviewed.answer.strip()
            if len(text) > budget:
                raise ModelError("The source review exceeded the feedback text budget.")
            verification = "source_review"
            asset_ids = [asset.id]
        else:
            prefix = f"{item.role}文本/未核实: "
            excerpt = record.content.strip()
            available = budget - len(prefix)
            text = prefix + (
                excerpt if len(excerpt) <= available else excerpt[: available - 1] + "…"
            )
            verification = "record_text"
        return {
            "memory_id": record.id,
            "text": text,
            "occurred_at": None if record.occurred_at is None else record.occurred_at.isoformat(),
            "verification": verification,
            "asset_ids": asset_ids,
        }

    async def _project(self, work: _Work) -> tuple[list[_Fact], tuple[MemoryRecord, ...], bool]:
        evidence = await self._load(work)
        if not evidence or not self._current(work):
            return [], (), False
        facts: list[_Fact] = []
        abstained = False
        remaining = 160
        for index, item in enumerate(evidence):
            budget = remaining // (len(evidence) - index)
            fact = await self._fact(work, item, budget)
            if fact is None:
                abstained = True
                facts.clear()
                break
            remaining -= len(fact["text"])
            facts.append(fact)
            if not self._current(work):
                return [], (), False
        # Model work yields to correction/forgetting. Rehydrate before publishing, even
        # when an external SDK mutation did not pass through the host's revision hook.
        stable: list[MemoryRecord] = []
        for item in evidence:
            try:
                latest = await self.memory.get(item.record.id)
            except MemoryNotFoundError:
                continue
            if latest == item.record:
                stable.append(latest)
        pending = (
            {
                item.memory_id
                for item in await self.memory.pending_captures(
                    limit=2, memory_ids=tuple(record.id for record in stable)
                )
            }
            if stable
            else set()
        )
        records = tuple(record for record in stable if record.id not in pending)
        retained = {record.id for record in records}
        return [fact for fact in facts if fact["memory_id"] in retained], records, abstained

    async def close(self) -> None:
        """Discard pending feedback and drain the current public SDK operation."""
        self.closed = True
        self._pending = None
        if self.task is not None:
            await self.task
