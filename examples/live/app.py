"""Local continuous audio/video companion using SenseMem and speech-to-speech."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Self, cast
from uuid import uuid4

import av
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic import ValidationError as PydanticValidationError
from starlette.types import Scope

from examples.live import prompts as demo_prompts
from examples.live.capture import EventCapture
from examples.live.dialogue import DialogueInput, capture_dialogue
from examples.live.events import open_event_memory
from examples.live.feedback import MemoryFeedback
from examples.live.media import MediaReceiver
from examples.live.memories import memory_cards, memory_router
from examples.live.realtime import DEFAULT_REALTIME_URL, MAX_MESSAGE_BYTES, RealtimeRelay
from examples.live.recall import LiveRecall
from examples.live.updates import LiveEvent, LiveUpdates
from examples.live.user_audio import UserAudioCapture
from examples.live.vision import LiveVision
from examples.live.visual_memory import VisualMemoryCapture
from sensemem import (
    AsyncEventStream,
    AsyncMemory,
    ContextBudget,
    EventSegmentationBackend,
    GenerationBackend,
    MemoryNotFoundError,
    MemoryRecord,
    MemorySpec,
    ObservationContext,
    PluginConfig,
    PrefetchResult,
    SearchHit,
    SenseMemError,
    StreamCommit,
    ValidationError,
)

MAX_DIALOGUE_BYTES = 20 * 1024 * 1024
SSE_KEEPALIVE_SECONDS = 15
GRACEFUL_SHUTDOWN_SECONDS = 5
DEFAULT_LIVE_HOST = "127.0.0.1"
DEFAULT_LIVE_PORT = 8787
MemoryFactory = Callable[[MemorySpec], tuple[AsyncMemory, EventSegmentationBackend]]


class DemoStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        # Unversioned module imports must revalidate together after local edits.
        response.headers["Cache-Control"] = "no-cache"
        return response


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(strict=True, min_length=1, max_length=2000)
    turn_id: str | None = Field(default=None, strict=True, min_length=1, max_length=128)
    transcript: str | None = Field(default=None, strict=True, max_length=2000)


class MediaStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["start"]
    started_at: AwareDatetime


class MediaEnd(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["end"]


class ProactiveDecision(BaseModel):
    """Validate an optional spoken proposal before it leaves the memory side path."""

    model_config = ConfigDict(extra="forbid")
    speak: bool = Field(strict=True)
    utterance: str = Field(strict=True, max_length=400)
    reason: Literal["follow_up", "memory_connection", "none"]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.speak != bool(self.utterance.strip()) or self.speak == (self.reason == "none"):
            raise ValueError("a spoken decision needs an utterance and a reason")
        return self


class LiveRuntime:
    """Own the live example's memory, capture lifecycle and shared provider clients."""

    def __init__(
        self,
        config: MemorySpec,
        memory_factory: MemoryFactory,
        realtime_url: str,
    ) -> None:
        self.config = config
        self.memory_factory = memory_factory
        self.realtime_url = realtime_url
        self.memory: AsyncMemory | None = None
        self.realtime: RealtimeRelay | None = None
        self.user_audio: UserAudioCapture | None = None
        self.lock = asyncio.Lock()
        self.connection_lock = asyncio.Lock()
        self.work = asyncio.Event()
        self.closing = False
        self.worker: asyncio.Task[None] | None = None
        self.segmenter: EventSegmentationBackend | None = None
        self.capture: EventCapture | None = None
        self.source_id: str | None = None
        self.finishing: set[EventCapture] = set()
        self.capture_failed = False
        self.capture_lock = asyncio.Lock()
        self.maintaining = False
        self.maintenance_failed = False
        self.recall: LiveRecall | None = None
        self.feedback: MemoryFeedback | None = None
        self.vision: LiveVision | None = None
        self.recall_closing: set[asyncio.Task[None]] = set()
        self.memory_revision = 0
        self.updates: LiveUpdates | None = None
        self.initiatives: dict[str, dict[str, object]] = {}
        self.visual_initiatives: dict[str, dict[str, object]] = {}
        self.completed_initiatives: set[str] = set()
        self.latest_dialogue: dict[str, object] | None = None
        self.changes = asyncio.Condition()
        self.notifying: asyncio.Task[None] | None = None
        self.view_revision = 0

    def state_changed(self) -> None:
        self.view_revision += 1
        if self.notifying is None or self.notifying.done():
            self.notifying = asyncio.create_task(self._notify_changes())

    def memory_changed(self, *, schedule: bool = True) -> None:
        self.memory_revision += 1
        if schedule:
            self.work.set()
        if self.recall is not None:
            self.recall.invalidate(refresh=True)
        if self.updates is not None:
            self.updates.memory_changed(self.memory_revision)
        if self.vision is not None:
            self.vision.invalidate(self.memory_revision)
        if self.feedback is not None:
            self.feedback.invalidate(self.memory_revision)
        self.state_changed()

    async def _notify_changes(self) -> None:
        async with self.changes:
            self.changes.notify_all()

    def observe(self, commit: StreamCommit) -> None:
        self.memory_changed(schedule=False)
        if (
            self.updates is not None
            and commit.record.context is not None
            and commit.record.context.source_id == self.source_id
        ):
            self.updates.observe(commit)

    def observe_visual(self, commit: StreamCommit) -> None:
        self.memory_changed()
        context = commit.record.context
        if context is None or context.source_id != self.source_id:
            return
        if self.updates is not None:
            self.updates.observe(commit)
        if self.recall is not None:
            task = asyncio.create_task(
                self.recall.publish(
                    {
                        "type": "visual_memory",
                        "source_id": self.source_id,
                        "perception_id": commit.stream_id,
                        "status": "saved",
                        "memory_ids": [commit.record.id],
                    }
                )
            )
            self.recall_closing.add(task)
            task.add_done_callback(self.recall_closing.discard)

    async def assess_event(
        self, event: LiveEvent, revision: int, trigger: Literal["visual", "audio", "memory"]
    ) -> dict[str, object] | None:
        """Review original media asynchronously; this never gates a speech-service turn."""
        memory = await self.open_memory()
        recall = self.recall
        if event.source_id != self.source_id or recall is None:
            return None
        observations = [await memory.get(memory_id) for memory_id in event.evidence_ids]
        if any(
            record.forgotten_at is not None
            or (record.context is not None and record.context.retired_at is not None)
            for record in observations
        ):
            return None
        assets = [
            await memory.read_asset(record.id, asset.id)
            for record in observations
            for asset in record.assets
        ]
        if not assets or self.recall is not recall:
            return None
        latest_text = recall.latest_text
        latest_dialogue = self.latest_dialogue
        data = json.dumps(
            {
                "event_id": event.event_id,
                "summary": event.summary,
                "summary_verified": False,
                "occurred_at": event.occurred_at.isoformat(),
                "occurred_end": event.occurred_end.isoformat(),
                "assessed_at": datetime.now(timezone.utc).isoformat(),
                "evidence_ids": event.evidence_ids,
                "observation_sources": [
                    {
                        "memory_id": record.id,
                        "asset_order": [asset.id for asset in record.assets],
                        "native_event": record.metadata.get("event_segmentation"),
                    }
                    for record in observations
                ],
                "latest_user_text": latest_text,
                "latest_completed_dialogue": latest_dialogue,
                "input_memory_revision": revision,
                "previous_suggestions": [
                    item["utterance"]
                    for candidate_id, item in self.initiatives.items()
                    if candidate_id in self.completed_initiatives
                ],
                "trigger": trigger,
            },
            ensure_ascii=False,
        )
        reviewed = await memory.ask(
            (
                demo_prompts.proactive_question(
                    data, f"{event.summary[:512]}\n{latest_text[:512]}"
                ),
                *assets,
            ),
            limit=4,
            answer_policy="best_effort",
        )
        if (
            reviewed.abstained
            or event.source_id != self.source_id
            or self.recall is not recall
            or latest_text != recall.latest_text
            or latest_dialogue is not self.latest_dialogue
        ):
            return None
        try:
            decision = ProactiveDecision.model_validate_json(reviewed.answer)
        except PydanticValidationError:
            return None
        if not decision.speak:
            return None
        verified_revision = await self._revalidate_review((*observations, *reviewed.hits))
        if verified_revision is None or self.recall is not recall:
            return None
        now = datetime.now(timezone.utc)
        candidate_id = uuid4().hex
        proposal: dict[str, object] = {
            "candidate_id": candidate_id,
            "event_id": event.event_id,
            "source_id": event.source_id,
            "evidence_ids": list(event.evidence_ids),
            "reviewed_memory_ids": [hit.id for hit in reviewed.hits],
            "utterance": decision.utterance.strip(),
            "reason": decision.reason,
            "trigger": trigger,
            "memory_revision": verified_revision,
            "expires_at": (now + timedelta(seconds=30)).isoformat(),
            "context": json.dumps(
                {
                    "event_id": event.event_id,
                    "evidence_ids": event.evidence_ids,
                    "reviewed_memory_ids": [hit.id for hit in reviewed.hits],
                    "reviewed_suggestion": decision.utterance.strip(),
                },
                ensure_ascii=False,
            ),
        }
        if len(self.initiatives) == 16:
            retired = next(iter(self.initiatives))
            self.initiatives.pop(retired)
            self.completed_initiatives.discard(retired)
        self.initiatives[candidate_id] = proposal
        return proposal

    async def _revalidate_review(self, sources: Sequence[MemoryRecord | SearchHit]) -> int | None:
        """Allow unrelated writes while rejecting changed or withdrawn actual evidence."""
        memory = await self.open_memory()
        revision = self.memory_revision
        for source in sources:
            try:
                current = await memory.get(source.id)
            except MemoryNotFoundError:
                return None
            if (
                current.forgotten_at is not None
                or (
                    current.context is not None
                    and (not current.context.visible or current.context.retired_at is not None)
                )
                or not current.content.startswith(source.content)
                or any(asset not in current.assets for asset in source.assets)
            ):
                return None
        return revision if revision == self.memory_revision else None

    async def open_memory(self) -> AsyncMemory:
        """Open the shared memory and its deferred-work loop once."""
        async with self.lock:
            if self.memory is None:
                memory, self.segmenter = await asyncio.to_thread(self.memory_factory, self.config)
                self.memory = memory
                self.worker = asyncio.create_task(self._process_pending_memories(memory))
                self.work.set()
            return self.memory

    async def start_capture(self) -> str:
        """Start native event capture and return the issued source identity."""
        async with self.capture_lock:
            memory = await self.open_memory()
            if self.recall is None:
                self.recall = LiveRecall(memory)
                self.initiatives.clear()
                self.visual_initiatives.clear()
                self.completed_initiatives.clear()
                self.latest_dialogue = None
                self.updates = LiveUpdates(memory, self.recall, self.assess_event)
                self.updates.memory_changed(self.memory_revision)
            assert self.segmenter is not None
            directory = self.config.data_dir / "live-events"
            if self.capture is None:
                capture_id = uuid4().hex
                self.source_id = f"live:{capture_id}"
                self.capture = EventCapture(
                    AsyncEventStream(
                        memory,
                        segmenter=self.segmenter,
                        max_chunks=4,
                        capture=True,
                        context=ObservationContext(source_id=self.source_id),
                        checkpoint_path=directory / f"{capture_id}.json",
                    ),
                    self.work,
                    observe=self.observe,
                    changed=self.state_changed,
                )
            assert self.source_id is not None
            if self.feedback is None:
                assert self.recall is not None
                self.feedback = MemoryFeedback(
                    memory, self.recall.publish, self.source_id, revision=self.memory_revision
                )
                self.recall.observer = self.feedback.observe
            return self.source_id

    def _capture_done(self, capture: EventCapture) -> None:
        self.capture_failed |= capture.failure is not None
        self.finishing.discard(capture)
        self.state_changed()

    async def finish_capture(self, capture: EventCapture | None = None) -> None:
        """Finish the specified capture while its deferred processing drains separately."""
        async with self.capture_lock:
            capture = self.capture if capture is None else capture
            if capture is not None and not capture.closed:
                if capture is self.capture:
                    self.capture = None
                self.finishing.add(capture)
                await capture.finish()
                self.state_changed()
                assert capture.task is not None
                capture.task.add_done_callback(lambda _task: self._capture_done(capture))

    async def _process_pending_memories(self, memory: AsyncMemory) -> None:
        """Settle captures and run configured consolidation when the host schedules work."""
        while True:
            await self.work.wait()
            self.work.clear()
            self.maintaining = True
            self.state_changed()
            try:
                # SenseMem owns the durable queue and retry policy; the host only schedules work.
                while True:
                    try:
                        settled = await memory.settle(limit=16)
                    except SenseMemError:
                        # A batch can commit its successful records before reporting one failure.
                        self.memory_changed(schedule=False)
                        break
                    if not settled:
                        break
                    self.memory_changed(schedule=False)
                if self.config.consolidation is not None and not self.closing:
                    try:
                        report = await memory.consolidate_pending(
                            limit=16, max_rounds=1, idle=self.capture is None
                        )
                        if report.operations:
                            self.memory_changed(schedule=False)
                        self.maintenance_failed = False
                    except SenseMemError:
                        self.maintenance_failed = True
            finally:
                self.maintaining = False
                self.state_changed()
            if self.closing:
                return

    async def disconnect(self) -> None:
        async with self.connection_lock:
            if self.user_audio is not None:
                user_audio, self.user_audio = self.user_audio, None
                task = asyncio.create_task(self._close_user_audio(user_audio))
                self.recall_closing.add(task)
                task.add_done_callback(self.recall_closing.discard)
            if self.feedback is not None:
                feedback, self.feedback = self.feedback, None
                if self.recall is not None:
                    self.recall.observer = None
                task = asyncio.create_task(feedback.close())
                self.recall_closing.add(task)
                task.add_done_callback(self.recall_closing.discard)
            if self.vision is not None:
                vision, self.vision = self.vision, None
                task = asyncio.create_task(self._close_vision(vision))
                self.recall_closing.add(task)
                task.add_done_callback(self.recall_closing.discard)
            if self.updates is not None:
                updates, self.updates = self.updates, None
                task = asyncio.create_task(updates.close())
                self.recall_closing.add(task)
                task.add_done_callback(self.recall_closing.discard)
            if self.recall is not None:
                recall, self.recall = self.recall, None
                task = asyncio.create_task(self._close_recall(recall))
                self.recall_closing.add(task)
                task.add_done_callback(self.recall_closing.discard)
            if self.realtime is not None:
                realtime, self.realtime = self.realtime, None
                await realtime.close()

    async def _close_recall(self, recall: LiveRecall) -> None:
        # Native provider work drains in the background, independently of call teardown.
        with suppress(SenseMemError):
            await recall.close()

    async def _close_user_audio(self, capture: UserAudioCapture) -> None:
        await capture.close()
        self.capture_failed |= capture.failure is not None
        self.state_changed()

    async def _close_vision(self, vision: LiveVision) -> None:
        await vision.close()
        self.capture_failed |= (
            vision.memory_capture is not None and vision.memory_capture.failure is not None
        )
        self.state_changed()

    async def close(self) -> None:
        with suppress(OpenAIError):
            await self.disconnect()
        with suppress(SenseMemError, OpenAIError):
            await self.finish_capture()
        for capture in tuple(self.finishing):
            with suppress(SenseMemError, OpenAIError):
                await capture.wait()
        for task in tuple(self.recall_closing):
            await task
        if self.worker is not None:
            self.closing = True
            self.work.set()
            await self.worker
        if self.notifying is not None:
            await self.notifying
        if self.memory is not None:
            await self.memory.close()


async def _request_json_body(request: Request, limit: int) -> bytes:
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise HTTPException(413, "This upload exceeds the demo size limit.")
        data.extend(chunk)
    if not data:
        raise HTTPException(422, "The request body is empty.")
    return bytes(data)


def _create_realtime_session() -> dict[str, object]:
    return {
        "type": "realtime",
        "instructions": demo_prompts.REALTIME_INSTRUCTIONS,
        "output_modalities": ["audio"],
        "extensions": ["speech_to_speech.input_audio_transcription.snapshot"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"language": "auto"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.35,
                    "silence_duration_ms": 600,
                    "create_response": False,
                    "interrupt_response": True,
                },
            },
            "output": {"format": {"type": "audio/pcm", "rate": 24000}},
        },
        "tools": [
            {
                "type": "function",
                "name": "search_memory",
                "description": demo_prompts.SEARCH_MEMORY_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 2000}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": "auto",
        "reasoning": {"effort": "none"},
    }


def create_app(  # noqa: C901 - one literal example route registry
    config: MemorySpec,
    *,
    data_dir: Path | None = None,
    realtime_url: str | None = None,
    memory_factory: MemoryFactory | None = None,
) -> FastAPI:
    """Build the local example and validate its memory runtime during startup."""
    configured: dict[str, object] = {}
    for slot, backend in (
        ("generation", config.generation),
        ("formation", config.formation),
        ("consolidation", config.consolidation),
    ):
        if backend is None or isinstance(backend, PluginConfig):
            continue
        configured[slot] = backend.model_copy(
            update={
                "api_key": backend.api_key
                or SecretStr(os.environ.get("SENSEMEM_GENERATION_API_KEY", "")),
                "base_url": os.environ.get("SENSEMEM_GENERATION_BASE_URL") or backend.base_url,
            }
        )
    config = config.model_copy(update=configured)
    runtime = LiveRuntime(
        config if data_dir is None else config.model_copy(update={"data_dir": data_dir}),
        memory_factory or open_event_memory,
        realtime_url or os.environ.get("SENSEMEM_REALTIME_URL") or DEFAULT_REALTIME_URL,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            await runtime.open_memory()
            yield
        finally:
            await runtime.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(memory_router(runtime.open_memory, runtime.memory_changed))

    async def present(result: PrefetchResult) -> dict[str, object]:
        assert result.bundle is not None
        return {
            "context": result.bundle.render(),
            "hits": await memory_cards(await runtime.open_memory(), result.hits),
            "query": result.bundle.goal,
            "source_id": runtime.source_id,
            "memory_revision": runtime.memory_revision,
        }

    @app.exception_handler(SenseMemError)
    async def memory_error(_request: Request, error: SenseMemError) -> JSONResponse:
        status = 404 if isinstance(error, MemoryNotFoundError) else 503
        if isinstance(error, ValidationError):
            status = 422
        return JSONResponse({"detail": "The memory operation could not be completed."}, status)

    @app.exception_handler(OpenAIError)
    async def provider_error(_request: Request, _error: OpenAIError) -> JSONResponse:
        return JSONResponse({"detail": "The model operation could not be completed."}, 502)

    @app.get("/api/config")
    async def public_config() -> dict[str, object]:
        realtime_model = os.environ.get("SENSEMEM_REALTIME_MODEL") or (
            None
            if runtime.config.generation is None
            or isinstance(runtime.config.generation, PluginConfig)
            else runtime.config.generation.model
        )
        return {
            "session": _create_realtime_session(),
            "realtime_model": realtime_model,
            "speech_service": "speech-to-speech",
            "capabilities": {
                "input_audio": True,
                "input_video": False,
                "output_audio": True,
                "output_text": True,
                "input_transcription": True,
                "function_calling": True,
                "dynamic_context": True,
            },
            "memory_context_instructions": demo_prompts.MEMORY_CONTEXT_INSTRUCTIONS,
            "recall_unavailable_instructions": demo_prompts.RECALL_UNAVAILABLE_INSTRUCTIONS,
            "context_unavailable_instructions": demo_prompts.CONTEXT_UNAVAILABLE_INSTRUCTIONS,
            "visual_unavailable_instructions": demo_prompts.VISUAL_UNAVAILABLE_INSTRUCTIONS,
            "proactive_instructions": demo_prompts.PROACTIVE_INSTRUCTIONS,
            "perception_instructions": demo_prompts.PERCEPTION_RESPONSE_INSTRUCTIONS,
            "perception_ready": isinstance(runtime.segmenter, GenerationBackend),
            "realtime_ready": bool(runtime.realtime_url),
            "formation_enabled": runtime.config.formation is not None,
            "maintenance_enabled": runtime.config.consolidation is not None,
        }

    @app.post("/api/session")
    async def start() -> dict[str, object]:
        source_id = await runtime.start_capture()
        runtime.work.set()
        return {"source_id": source_id, "memory_revision": runtime.memory_revision}

    @app.post("/api/dialogue", status_code=202)
    async def dialogue(request: Request) -> dict[str, str]:  # noqa: C901 - one provenance-checked capture
        try:
            supplied = DialogueInput.model_validate_json(
                await _request_json_body(request, MAX_DIALOGUE_BYTES)
            )
        except PydanticValidationError:
            raise HTTPException(422, "Invalid completed dialogue.") from None
        memory = await runtime.open_memory()
        reviewed_memory_ids: list[str] | None = None
        if supplied.origin == "proactive":
            issued = runtime.initiatives.get(supplied.candidate_id or "")
            if issued is None or (
                issued["event_id"] != supplied.event_id
                or issued["source_id"] != supplied.source_id
                or issued["evidence_ids"] != list(supplied.evidence_ids)
            ):
                raise HTTPException(422, "The proactive reply needs its issued native evidence.")
            reviewed_memory_ids = cast(list[str], issued["reviewed_memory_ids"])
        elif supplied.origin == "perception":
            issued = runtime.visual_initiatives.get(supplied.candidate_id or "")
            assert supplied.perception is not None
            if issued is None or (
                issued["perception_id"] != supplied.event_id
                or issued["source_id"] != supplied.source_id
                or issued["frame_sequences"] != list(supplied.perception.frame_sequences)
                or issued["occurred_at"] != supplied.perception.occurred_at.isoformat()
                or issued["occurred_end"] != supplied.perception.occurred_end.isoformat()
            ):
                raise HTTPException(
                    422, "The visual reply needs its issued timed frame provenance."
                )
        record = await capture_dialogue(memory, supplied, reviewed_memory_ids=reviewed_memory_ids)
        if supplied.origin == "proactive":
            assert supplied.candidate_id is not None
            runtime.completed_initiatives.add(supplied.candidate_id)
        if supplied.origin == "realtime" and (
            supplied.playback is None or supplied.playback.status == "completed"
        ):
            for receipt in supplied.feedback_receipts:
                issued = runtime.initiatives.get(receipt.candidate_id or "")
                if issued is not None and (
                    issued["source_id"] == supplied.source_id
                    and issued["memory_revision"] == receipt.memory_revision
                ):
                    assert receipt.candidate_id is not None
                    runtime.completed_initiatives.add(receipt.candidate_id)
        if supplied.source_id == runtime.source_id and (
            supplied.playback is None or supplied.playback.status == "completed"
        ):
            runtime.latest_dialogue = {
                "origin": supplied.origin,
                "user": None if supplied.user is None else supplied.user.text,
                "assistant": supplied.assistant.text,
            }
        runtime.memory_changed()
        return {"id": record.id}

    @app.websocket("/api/realtime")
    async def realtime(socket: WebSocket) -> None:
        await socket.accept()
        async with runtime.connection_lock:
            if runtime.realtime is not None or runtime.capture is None:
                await socket.close(code=1008)
                return
            source_id = runtime.source_id
            assert source_id is not None

            def observe_audio(data: bytes, sequence: int, timestamp_ms: float) -> None:
                vision = runtime.vision
                if vision is not None and vision.source_id == source_id:
                    vision.append_audio(data, sequence, timestamp_ms)

            user_audio = runtime.user_audio = UserAudioCapture(
                await runtime.open_memory(),
                source_id,
                committed=lambda _commit: runtime.memory_changed(),
                changed=runtime.state_changed,
            )
            relay = runtime.realtime = RealtimeRelay(socket, observe_audio, user_audio=user_audio)
        try:
            await relay.serve(runtime.realtime_url)
        finally:
            if runtime.realtime is relay:
                runtime.realtime = None
            if runtime.user_audio is user_audio:
                runtime.user_audio = None
                task = asyncio.create_task(runtime._close_user_audio(user_audio))
                runtime.recall_closing.add(task)
                task.add_done_callback(runtime.recall_closing.discard)

    @app.websocket("/api/perception")
    async def perception(socket: WebSocket) -> None:
        async with runtime.connection_lock:
            if (
                runtime.vision is not None
                or runtime.recall is None
                or runtime.source_id is None
                or not isinstance(runtime.segmenter, GenerationBackend)
            ):
                await socket.accept()
                await socket.close(code=1008)
                return
            vision = runtime.vision = LiveVision(
                await runtime.open_memory(),
                runtime.segmenter,
                runtime.recall,
                runtime.source_id,
                runtime.visual_initiatives,
                memory_capture=VisualMemoryCapture(
                    await runtime.open_memory(),
                    runtime.source_id,
                    committed=runtime.observe_visual,
                    changed=runtime.state_changed,
                    model_id=runtime.segmenter.event_model,
                ),
            )
            vision.invalidate(runtime.memory_revision)
        try:
            await vision.serve(socket)
        finally:
            if runtime.vision is vision:
                runtime.vision = None
            # Draining the current model call must not delay microphone teardown.
            task = asyncio.create_task(runtime._close_vision(vision))
            runtime.recall_closing.add(task)
            task.add_done_callback(runtime.recall_closing.discard)

    @app.websocket("/api/media")
    async def media(socket: WebSocket) -> None:  # noqa: C901 - one WebSocket lifecycle
        await socket.accept()
        capture = runtime.capture
        if capture is None:
            await socket.close(code=1008)
            return
        source: MediaReceiver | None = None
        try:
            started = MediaStart.model_validate_json(await socket.receive_text())
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (data := message.get("bytes")) is not None:
                    if source is None:
                        source = MediaReceiver(capture, started.started_at)
                    await source.feed(data)
                elif (text := message.get("text")) is not None:
                    MediaEnd.model_validate_json(text)
                    if source is not None:
                        await source.finish()
                    await socket.send_json({"type": "accepted"})
                    await socket.close()
                    break
        except WebSocketDisconnect:
            pass
        except (
            SenseMemError,
            PydanticValidationError,
            av.FFmpegError,
            ValueError,
            OSError,
        ) as error:
            code = (
                "capacity_exceeded"
                if isinstance(error, SenseMemError) and error.reason == "capacity_exceeded"
                else "recording_failed"
            )
            await socket.send_json({"type": "error", "code": code})
            await socket.close(code=1011)
        finally:
            if source is not None:
                with suppress(SenseMemError, av.FFmpegError, ValueError, OSError):
                    await source.finish()
            await runtime.finish_capture(capture)

    @app.websocket("/api/recall")
    async def recall(socket: WebSocket) -> None:
        source = runtime.recall
        if source is None:
            await socket.accept()
            await socket.close(code=1008)
            return
        await source.serve(socket, present)

    @app.get("/api/memories")
    async def memories() -> dict[str, object]:
        memory = await runtime.open_memory()
        return {
            "items": await memory_cards(memory, (await memory.list(limit=100)).items),
            "capture_failed": runtime.capture_failed
            or (runtime.capture is not None and runtime.capture.failure is not None)
            or (runtime.user_audio is not None and runtime.user_audio.failure is not None)
            or (
                runtime.vision is not None
                and runtime.vision.memory_capture is not None
                and runtime.vision.memory_capture.failure is not None
            ),
            "processing": bool(runtime.finishing)
            or (runtime.capture is not None and runtime.capture.processing)
            or (runtime.user_audio is not None and runtime.user_audio.processing)
            or (
                runtime.vision is not None
                and runtime.vision.memory_capture is not None
                and runtime.vision.memory_capture.processing
            )
            or runtime.maintaining,
            "maintenance_failed": runtime.maintenance_failed,
        }

    @app.get("/api/events")
    async def changes() -> StreamingResponse:
        async def updates() -> AsyncIterator[str]:
            revision = runtime.view_revision
            while not runtime.closing:
                yield (
                    f"id: {revision}\nevent: memory_changed\n"
                    f"data: {json.dumps({'revision': revision})}\n\n"
                )
                while revision == runtime.view_revision and not runtime.closing:

                    def changed(observed: int = revision) -> bool:
                        return observed != runtime.view_revision or runtime.closing

                    try:
                        async with runtime.changes:
                            await asyncio.wait_for(
                                runtime.changes.wait_for(changed),
                                timeout=SSE_KEEPALIVE_SECONDS,
                            )
                    except TimeoutError:
                        yield ": keepalive\n\n"
                revision = runtime.view_revision

        return StreamingResponse(
            updates(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/search")
    async def search(request: Request) -> dict[str, object]:
        try:
            supplied = SearchInput.model_validate_json(
                await _request_json_body(request, MAX_MESSAGE_BYTES)
            )
        except PydanticValidationError:
            raise HTTPException(422, "Invalid memory query.") from None
        memory = await runtime.open_memory()
        revision = runtime.memory_revision
        result = None
        if runtime.recall is not None and supplied.turn_id is not None:
            result = await runtime.recall.finalize(
                supplied.turn_id, supplied.query, transcript=supplied.transcript
            )
        if revision != runtime.memory_revision:
            result = None
        if result is None:
            revision = runtime.memory_revision
            query = supplied.transcript if supplied.transcript is not None else supplied.query
            if not query.strip():
                raise HTTPException(422, "The authoritative transcript is empty.")
            bundle = await memory.compile(
                query, budget=ContextBudget(max_chars=64_000, max_items=4, max_media_items=8)
            )
            result = PrefetchResult(bundle.hits, bundle)
        evidence_revision = revision
        sources = result.hits
        rendered = await present(result)
        if runtime.config.generation is not None and any(hit.assets for hit in result.hits):
            assert result.bundle is not None
            # Use the SDK's grounding and native media transport, rather than treating a
            # segmented synopsis as if it had rechecked the original audio and video.
            reviewed = await memory.ask(result.bundle.goal, limit=4)
            sources = (*sources, *reviewed.hits)
            rendered["hits"] = await memory_cards(memory, reviewed.hits)
            rendered["media_review"] = {
                "answer": reviewed.answer,
                "abstained": reviewed.abstained,
                "source_ids": [hit.id for hit in reviewed.hits if hit.assets],
            }
        # Independent new utterances do not invalidate unchanged historical originals.
        # Corrections and forgetting must still withdraw the actual reviewed sources.
        current_revision = await runtime._revalidate_review(sources)
        if current_revision is None or (
            not sources and evidence_revision != runtime.memory_revision
        ):
            raise HTTPException(409, "Memory changed during retrieval; please ask again.")
        rendered["memory_revision"] = current_revision
        return rendered

    @app.get("/api/memories/{memory_id}/assets/{asset_id}")
    async def asset(memory_id: str, asset_id: str) -> Response:
        original = await (await runtime.open_memory()).read_asset(memory_id, asset_id)
        return Response(original.data, media_type=original.media_type)

    @app.delete("/api/session")
    async def stop() -> Response:
        try:
            await runtime.disconnect()
        finally:
            await runtime.finish_capture()
        return Response(status_code=204)

    app.mount(
        "/", DemoStaticFiles(directory=Path(__file__).parent / "static", html=True, check_dir=False)
    )
    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="MemorySpec JSON file")
    parser.add_argument("--data-dir", type=Path, default=Path(".sensemem/live-demo"))
    parser.add_argument("--host", default=DEFAULT_LIVE_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_LIVE_PORT)
    parser.add_argument("--realtime-url", help="Local speech-to-speech Realtime WebSocket URL")
    args = parser.parse_args(argv)
    try:
        config = MemorySpec.model_validate_json(args.config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        parser.error("provide a readable, valid MemorySpec JSON file")
    uvicorn.run(
        create_app(config, data_dir=args.data_dir, realtime_url=args.realtime_url),
        host=args.host,
        port=args.port,
        access_log=False,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )
    return 0
