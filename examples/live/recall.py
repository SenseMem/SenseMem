"""Temporary, revision-bound recall while the speech service is still transcribing."""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from sensemem import (
    AsyncMemory,
    AsyncOmniPrefetch,
    Blob,
    ContentInput,
    ContextBudget,
    PrefetchResult,
    SenseMemError,
    ValidationError,
)

Presenter = Callable[[PrefetchResult], Awaitable[dict[str, object]]]
Sender = Callable[[dict[str, object]], Awaitable[None]]


class RecallInput(BaseModel):
    """A complete current hypothesis; final replaces all earlier ASR hypotheses."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["update", "final"]
    turn_id: str = Field(strict=True, min_length=1, max_length=128)
    text: str = Field(strict=True, max_length=2000)
    image: str | None = Field(default=None, strict=True, max_length=350_000)

    def snapshot(self) -> ContentInput:
        text = self.text.strip()
        if self.image is None:
            if not text:
                raise ValidationError("recall needs text or an image")
            return text
        header, separator, encoded = self.image.partition(",")
        if not separator or header not in {"data:image/jpeg;base64", "data:image/png;base64"}:
            raise ValidationError("recall image must be an encoded JPEG or PNG")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise ValidationError("recall image encoding is invalid") from None
        if not data or len(data) > 256 * 1024:
            raise ValidationError("recall image exceeds the demo size limit")
        image = Blob(data, header[5:-7])
        return (text, image) if text else (image,)


@dataclass(slots=True)
class _Turn:
    prefetch: AsyncOmniPrefetch
    watcher: asyncio.Task[None]
    offset: int = 0
    revision: int = 0
    final_text: str | None = None
    text: str = ""
    snapshot: ContentInput = ""
    finalization: asyncio.Task[PrefetchResult] | None = None
    completion: asyncio.Task[None] | None = None
    delivered: bool = False


@dataclass(frozen=True, slots=True)
class _Refresh:
    turn_id: str
    text: str
    snapshot: ContentInput
    final_text: str | None
    finalized: bool


class LiveRecall:
    """One call's native speculative helpers, never a durable observation writer."""

    def __init__(self, memory: AsyncMemory) -> None:
        self.memory = memory
        self.turns: dict[str, _Turn] = {}
        self.current_turn: str | None = None
        self.sender: Sender | None = None
        self.presenter: Presenter | None = None
        self.observer: Callable[[dict[str, object]], None] | None = None
        self.send_lock = asyncio.Lock()
        self.retiring: set[asyncio.Task[None]] = set()
        self.highest_revision = 0
        self.closed = False
        self.refreshing: asyncio.Task[None] | None = None
        self._refresh: _Refresh | None = None
        self._refresh_wait: asyncio.Task[None] | None = None

    @property
    def latest_text(self) -> str:
        """Keep real input available while obsolete retrieval drains after a memory write."""
        current = self.turns.get(self.current_turn or "")
        if current is not None:
            return current.text
        return "" if self._refresh is None else self._refresh.text

    async def _send(
        self, message: dict[str, object], *, owner: tuple[str, _Turn] | None = None
    ) -> None:
        async with self.send_lock:
            if owner is not None and (
                self.turns.get(owner[0]) is not owner[1] or owner[0] != self.current_turn
            ):
                return
            if self.sender is not None:
                try:
                    await self.sender(message)
                except WebSocketDisconnect:
                    self.sender = None

    async def publish(self, message: dict[str, object]) -> None:
        """Serialize native event pushes with ordinary speculative recall delivery."""
        if not self.closed:
            if self.observer is not None:
                self.observer(message)
            await self._send(message)

    async def serve(self, socket: WebSocket, present: Presenter) -> None:
        """Bind the call's single recall socket; inference never blocks its receive loop."""
        await socket.accept()
        if self.closed or self.sender is not None:
            await socket.close(code=1008)
            return
        self.sender = socket.send_json
        self.presenter = present
        try:
            while True:
                supplied = RecallInput.model_validate_json(await socket.receive_text())
                await self.update(supplied)
        except WebSocketDisconnect:
            pass
        except (PydanticValidationError, ValidationError):
            await self._send({"type": "error", "code": "invalid_recall_input"})
            self.sender = None
            await socket.close(code=1008)
        finally:
            self.sender = None

    async def update(self, supplied: RecallInput) -> None:
        """Submit real text/image input and publish its revision before it can complete."""
        if self.closed:
            raise ValidationError("live recall is closed")
        if not supplied.text.strip() and supplied.image is None:
            # An empty authoritative transcript retracts, rather than finalizes, a partial.
            if supplied.type == "final":
                await self._retract(supplied.turn_id)
            return
        snapshot = supplied.snapshot()
        self._refresh = None
        turn = self.turns.get(supplied.turn_id)
        if turn is not None and turn.finalization is not None:
            if supplied.type != "final" or snapshot == turn.snapshot:
                return
            # A corrected authoritative final withdraws its older completion immediately.
            self.turns.pop(supplied.turn_id)
            self._retire_later(turn)
            turn = None
        if turn is None:
            # Only a short, request-local tail is needed for late tool calls from prior turns.
            if len(self.turns) == 8:
                oldest = self.turns.pop(next(iter(self.turns)))
                self._retire_later(oldest)
            turn = self._new_turn(supplied.turn_id)
        revision = turn.prefetch.submit(snapshot)
        changed = revision != turn.revision
        turn.revision = revision
        self.highest_revision = max(self.highest_revision, turn.offset + revision)
        turn.snapshot = snapshot
        turn.text = supplied.text
        turn.final_text = supplied.text if supplied.type == "final" else turn.final_text
        if changed:
            await self._send(
                {
                    "type": "pending",
                    "turn_id": supplied.turn_id,
                    "revision": turn.offset + revision,
                    "query": supplied.text,
                    "input_final": supplied.type == "final",
                    "final": False,
                },
                owner=(supplied.turn_id, turn),
            )
        if supplied.type == "final" and turn.completion is None:
            # Exact final recall runs beside speech generation; no tool call is needed.
            turn.completion = asyncio.create_task(self._complete(supplied.turn_id, turn))

    async def _retract(self, turn_id: str) -> None:
        pending = self._refresh is not None and self._refresh.turn_id == turn_id
        if pending:
            self._refresh = None
        retracted = self.turns.pop(turn_id, None)
        if retracted is not None:
            self._retire_later(retracted)
        if turn_id == self.current_turn:
            self.current_turn = None
        if pending or retracted is not None:
            await self._send({"type": "cleared", "turn_id": turn_id})

    def _new_turn(self, turn_id: str) -> _Turn:
        helper = AsyncOmniPrefetch(
            self.memory,
            budget=ContextBudget(max_chars=64_000, max_items=4, max_media_items=8),
        )
        turn = _Turn(
            helper,
            asyncio.create_task(self._watch(turn_id, helper)),
            offset=self.highest_revision,
        )
        self.turns[turn_id] = turn
        self.current_turn = turn_id
        return turn

    async def _complete(self, turn_id: str, turn: _Turn) -> None:
        if self.turns.get(turn_id) is not turn:
            return
        with suppress(SenseMemError):
            await self.finalize(turn_id, turn.text)

    async def _watch(self, turn_id: str, prefetch: AsyncOmniPrefetch) -> None:
        async for update in prefetch.updates():
            turn = self.turns.get(turn_id)
            if (
                turn is None
                or turn.prefetch is not prefetch
                or turn_id != self.current_turn
                or turn.finalization is not None
                or self.presenter is None
            ):
                continue
            try:
                rendered = await self.presenter(update.result)
            except SenseMemError:
                await self._send(
                    {"type": "error", "turn_id": turn_id, "code": "recall_failed", "final": False},
                    owner=(turn_id, turn),
                )
                continue
            # Hydration can yield to a newer ASR hypothesis or a newer user turn.
            if (
                turn.revision != update.revision
                or turn_id != self.current_turn
                or turn.finalization is not None
            ):
                continue
            await self._send(
                {
                    **rendered,
                    "type": "result",
                    "turn_id": turn_id,
                    "revision": turn.offset + update.revision,
                    "query": turn.text,
                    "input_final": turn.final_text is not None,
                    "final": False,
                },
                owner=(turn_id, turn),
            )

    async def finalize(
        self,
        turn_id: str,
        fallback_query: str,
        *,
        transcript: str | None = None,
    ) -> PrefetchResult | None:
        """Deliver exact authoritative input, reusing a matching compiled completion."""
        turn = self.turns.get(turn_id)
        if turn is None or self.closed:
            return None
        if transcript is not None:
            if turn.finalization is not None and transcript != turn.text:
                self.turns.pop(turn_id)
                self._retire_later(turn)
                await self._send({"type": "cleared", "turn_id": turn_id})
                return None
            turn.final_text = transcript
        if turn.finalization is None:
            text = turn.final_text if turn.final_text is not None else fallback_query
            if not text.strip():
                raise ValidationError("the final transcript is empty")
            # A visual snapshot remains visual at finality; ordinary speech queries stay text.
            query: ContentInput = (
                (text, *turn.snapshot[-1:]) if isinstance(turn.snapshot, tuple) else text
            )
            turn.text = text
            if query != turn.snapshot:
                turn.revision = turn.prefetch.submit(query)
                turn.snapshot = query
            self.highest_revision = max(self.highest_revision, turn.offset + turn.revision)
            turn.finalization = asyncio.create_task(turn.prefetch.finalize())
        try:
            result = await asyncio.shield(turn.finalization)
        except SenseMemError:
            if self.turns.get(turn_id) is not turn:
                return None
            if not turn.delivered:
                turn.delivered = True
                await self._send(
                    {"type": "error", "turn_id": turn_id, "code": "recall_failed", "final": True},
                    owner=(turn_id, turn),
                )
            raise
        await self._deliver_final(turn_id, turn, result)
        return result if self.turns.get(turn_id) is turn else None

    async def _deliver_final(self, turn_id: str, turn: _Turn, result: PrefetchResult) -> None:
        """Publish one exact completion even when HTTP and background callers share it."""
        if (
            self.turns.get(turn_id) is turn
            and turn_id == self.current_turn
            and self.presenter is not None
            and not turn.delivered
        ):
            rendered = await self.presenter(result)
            if not turn.delivered:
                turn.delivered = True
                await self._send(
                    {
                        **rendered,
                        "type": "result",
                        "turn_id": turn_id,
                        "revision": turn.offset + turn.revision,
                        "query": turn.text,
                        "input_final": turn.final_text is not None,
                        "final": True,
                    },
                    owner=(turn_id, turn),
                )

    def invalidate(self, *, refresh: bool = False) -> None:
        """Withdraw context, optionally resubmitting the latest real input without waiting."""
        turn_id = self.current_turn
        latest = self.turns.get(turn_id) if turn_id is not None else None
        for previous_id, turn in self.turns.items():
            retired = self._retire_later(turn)
            if previous_id == turn_id:
                self._refresh_wait = retired
        self.turns.clear()
        self.current_turn = None
        if not refresh or self.closed:
            self._refresh = None
            return
        if turn_id is not None and latest is not None:
            self._refresh = _Refresh(
                turn_id,
                latest.text,
                latest.snapshot,
                latest.final_text,
                latest.finalization is not None,
            )
        if self.refreshing is None or self.refreshing.done():
            self.refreshing = asyncio.create_task(self._refresh_latest())

    async def _refresh_latest(self) -> None:
        # Repeated memory changes replace one pending snapshot while its old call drains.
        while (supplied := self._refresh) is not None and not self.closed:
            if self._refresh_wait is not None:
                await asyncio.shield(self._refresh_wait)
            if self._refresh is not supplied:
                continue
            self._refresh = None
            fresh = self._new_turn(supplied.turn_id)
            fresh.text = supplied.text
            fresh.snapshot = supplied.snapshot
            fresh.final_text = supplied.final_text
            fresh.revision = fresh.prefetch.submit(fresh.snapshot)
            self.highest_revision = fresh.offset + fresh.revision
            if supplied.finalized or supplied.final_text is not None:
                fresh.completion = asyncio.create_task(self._complete(supplied.turn_id, fresh))

    def _retire_later(self, turn: _Turn) -> asyncio.Task[None]:
        task = asyncio.create_task(self._retire(turn))
        self.retiring.add(task)
        task.add_done_callback(self.retiring.discard)
        return task

    async def _retire(self, turn: _Turn) -> None:
        await turn.prefetch.close()
        if turn.finalization is not None:
            with suppress(SenseMemError):
                await turn.finalization
        if turn.completion is not None:
            await turn.completion
        await turn.watcher

    async def close(self) -> None:
        """Detach delivery, discard queued hypotheses, and drain native SDK operations."""
        self.closed = True
        self.sender = None
        self.invalidate()
        if self.refreshing is not None:
            await self.refreshing
        await asyncio.gather(*self.retiring)
        self.retiring.clear()
