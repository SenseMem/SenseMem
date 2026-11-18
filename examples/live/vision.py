"""Live perception and recall alongside asynchronous original-window memory capture."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import math
import wave
from array import array
from collections import deque
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, cast
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from examples.live.memories import memory_cards
from examples.live.perception import (
    LivePerception,
    PerceptionContext,
    PerceptionFrame,
    PerceptionResult,
)
from examples.live.recall import LiveRecall
from examples.live.visual_memory import VisualMemoryCapture
from sensemem import (
    AsyncMemory,
    AsyncOmniPrefetch,
    Blob,
    ContextBudget,
    GenerationBackend,
    SenseMemError,
    ValidationError,
)


class VisionStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["start"]
    started_at: AwareDatetime
    audio_started_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class VisionFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["frame"]
    sequence: int = Field(strict=True, ge=1)
    captured_at: AwareDatetime
    timestamp_ms: float = Field(ge=0, allow_inf_nan=False)
    image: str = Field(strict=True, max_length=262_144)

    def frame(self, started_at: datetime) -> PerceptionFrame:
        if abs((self.captured_at - started_at).total_seconds() * 1000 - self.timestamp_ms) > 50:
            raise ValidationError("the frame does not use the call clock")
        if self.captured_at > datetime.now(timezone.utc) + timedelta(seconds=2):
            raise ValidationError("the frame capture time is in the future")
        prefix = "data:image/jpeg;base64,"
        if not self.image.startswith(prefix):
            raise ValidationError("perception requires an encoded JPEG")
        try:
            data = base64.b64decode(self.image[len(prefix) :], validate=True)
        except (binascii.Error, ValueError):
            raise ValidationError("invalid perception frame encoding") from None
        return PerceptionFrame(
            sequence=self.sequence,
            captured_at=self.captured_at,
            timestamp_ms=self.timestamp_ms,
            jpeg=data,
        )


class VisionState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["state"]
    user_speaking: bool = Field(strict=True)
    generating: bool = Field(strict=True)
    playing: bool = Field(strict=True)
    response_id: str | None = Field(default=None, strict=True, max_length=128)
    assistant_text: str = Field(default="", strict=True, max_length=8000)
    user_text: str = Field(default="", strict=True, max_length=4000)


class VisionCamera(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["camera", "recall"]
    enabled: bool = Field(strict=True)


_INPUT = TypeAdapter(
    Annotated[VisionFrame | VisionState | VisionCamera, Field(discriminator="type")]
)


class LiveVision:
    """One call's bounded perception and one-in-flight/latest-pending native recall.

    New packets never cancel a compiling snapshot: every completed query keeps its real
    time range, so a fast camera cannot starve retrieval. Raw media is persisted separately
    by EventCapture/AsyncEventStream. A separate native capture stream retains reviewed
    original frame windows with the interpretation attributed separately.
    """

    def __init__(
        self,
        memory: AsyncMemory,
        generation: GenerationBackend,
        recall: LiveRecall,
        source_id: str,
        issued: dict[str, dict[str, object]] | None = None,
        memory_capture: VisualMemoryCapture | None = None,
    ) -> None:
        self.memory, self.generation, self.recall, self.source_id = (
            memory,
            generation,
            recall,
            source_id,
        )
        self.state = VisionState(type="state", user_speaking=False, generating=False, playing=False)
        self.memory_revision = 0
        self.memory_context = ""
        self.observation: dict[str, object] | None = None
        self.camera_enabled = True
        self.camera_epoch = 0
        self.latest: PerceptionFrame | None = None
        self.last_sequence = 0
        self.started_at: datetime | None = None
        self.audio: deque[bytes] = deque()
        self.audio_bytes = self.audio_samples = self.queried_audio_samples = 0
        self.audio_end_ms = 0.0
        self.audio_sequence = 0
        self.query_revision = 0
        self.changed = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.closed = False
        self.issued = {} if issued is None else issued
        self.memory_capture = memory_capture
        self.perception = LivePerception(
            source_id, generation, self._observed, self.context, on_error=self._failed
        )

    def context(self) -> PerceptionContext:
        return PerceptionContext(
            latest_user_text=self.state.user_text,
            assistant_text=self.state.assistant_text,
            response_id=self.state.response_id,
            user_speaking=self.state.user_speaking,
            generating=self.state.generating,
            playing=self.state.playing,
            memory_context=self.memory_context,
            memory_revision=self.memory_revision,
        )

    def invalidate(self, revision: int) -> None:
        self.memory_revision = revision
        self.memory_context = ""
        if self.latest is not None or self.audio_bytes:
            self._wake()

    def _wake(self) -> None:
        self.changed.set()
        if self.task is None:
            self.task = asyncio.create_task(self._retrieve())

    def append_audio(self, data: bytes, sequence: int, timestamp_ms: float) -> None:
        """Receive the 16 kHz PCM16 side path after native audio has been forwarded."""
        if self.closed or self.started_at is None or not data or len(data) % 2:
            return
        if sequence <= self.audio_sequence or timestamp_ms < self.audio_end_ms:
            return
        self.audio_sequence, self.audio_end_ms = sequence, timestamp_ms
        self.audio.append(data)
        self.audio_bytes += len(data)
        self.audio_samples += len(data) // 2
        while self.audio_bytes > 64_000 and self.audio:
            self.audio_bytes -= len(self.audio.popleft())
        samples = array("h", data)
        energy = math.sqrt(sum(value * value for value in samples) / len(samples)) / 32768
        # This limits retrieval work; it does not define the native transport or
        # recording boundaries. Raw A/V continues through its independent event stream.
        if energy >= 0.008 and self.audio_samples - self.queried_audio_samples >= 4000:
            self._wake()

    async def serve(self, socket: WebSocket) -> None:
        await socket.accept()
        try:
            start = VisionStart.model_validate_json(await socket.receive_text())
            self.started_at = start.started_at
            await socket.send_json(
                {"type": "ready", "max_fps": 6, "max_frame_bytes": 262_144, "max_in_flight": 2}
            )
            while True:
                data = await socket.receive_text()
                if len(data.encode()) > 262_144:
                    raise ValidationError("the perception packet exceeds the byte budget")
                supplied = _INPUT.validate_json(data)
                if isinstance(supplied, VisionFrame):
                    frame = supplied.frame(start.started_at)
                    if frame.sequence <= self.last_sequence:
                        raise ValidationError("perception frame sequences must increase")
                    self.last_sequence = frame.sequence
                    if self.camera_enabled:
                        self.perception.submit(frame)
                        self.latest = frame
                        self._wake()
                    await socket.send_json({"type": "ack", "sequence": frame.sequence})
                elif isinstance(supplied, VisionState):
                    self.state = supplied
                elif supplied.type == "camera":
                    self.camera_enabled = supplied.enabled
                    self.camera_epoch += 1
                    self.latest = None
                    self.observation = None
                    self.perception.reset()
                    await self.recall.publish(
                        {
                            "type": "perception_reset",
                            "source_id": self.source_id,
                            "enabled": supplied.enabled,
                        }
                    )
                # The recall control chooses which provisional evidence the UI shows;
                # background multimodal retrieval does not depend on that toggle.
        except WebSocketDisconnect:
            pass
        except (PydanticValidationError, ValidationError):
            await socket.send_json({"type": "error", "code": "invalid_perception_input"})
            await socket.close(code=1008)

    async def _failed(self, code: str) -> None:
        if not self.closed:
            await self.recall.publish(
                {"type": "perception_error", "source_id": self.source_id, "code": code}
            )

    async def _observed(self, result: PerceptionResult) -> None:
        if self.closed or not self.camera_enabled or result.epoch != self.perception.epoch:
            return
        perception_id = uuid4().hex
        observed: dict[str, object] = {
            "source_id": self.source_id,
            "perception_id": perception_id,
            "frame_sequences": list(result.frame_sequences),
            "occurred_at": result.occurred_at.isoformat(),
            "occurred_end": result.occurred_end.isoformat(),
            "summary": result.summary,
            "current": result.current,
            "expires_at": result.expires_at.isoformat(),
            "confidence": result.confidence,
        }
        self.observation = observed
        await self.recall.publish({**observed, "type": "perception"})
        if self.memory_capture is not None:
            self.memory_capture.submit(result, result.frames, perception_id=perception_id)
        if self.closed or result.epoch != self.perception.epoch:
            return
        await self.recall.publish(
            {
                **observed,
                "type": "visual_context",
                "memory_revision": self.memory_revision,
                "context": self._context_text(),
                "hits": [],
            }
        )
        if (
            result.action == "quiet"
            or result.expires_at <= datetime.now(timezone.utc)
            or result.context.memory_revision != self.memory_revision
            or result.context.latest_user_text != self.state.user_text
        ):
            return
        candidate_id = uuid4().hex
        candidate = {
            **observed,
            "type": "visual_response",
            "candidate_id": candidate_id,
            "action": result.action,
            "utterance": result.utterance,
            "reason": result.reason,
            "target_response_id": result.target_response_id,
            "memory_revision": self.memory_revision,
            "context": self._context_text(),
        }
        self.issued[candidate_id] = candidate
        if len(self.issued) > 16:
            self.issued.pop(next(iter(self.issued)))
        await self.recall.publish(candidate)

    def _context_text(self) -> str:
        observation = self.observation
        if observation is not None and datetime.fromisoformat(
            cast(str, observation["expires_at"])
        ) <= datetime.now(timezone.utc):
            observation = None
        return json.dumps(
            {
                "temporary_visual_observation": observation,
                "historical_memory": self.memory_context,
            },
            ensure_ascii=False,
        )

    async def _retrieve(self) -> None:
        while not self.closed:
            await self.changed.wait()
            self.changed.clear()
            if self.closed:
                return
            frame, camera_epoch = self.latest, self.camera_epoch
            revision = self.memory_revision
            parts: list[str | Blob] = [
                self.state.user_text or "与刚刚听到或看到的内容有关的过去经历"
            ]
            times: list[datetime] = []
            if frame is not None:
                parts.append(Blob(frame.jpeg, "image/jpeg"))
                times.append(frame.captured_at)
            if self.audio and self.started_at is not None:
                encoded = io.BytesIO()
                with wave.open(encoded, "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16_000)
                    audio.writeframes(b"".join(self.audio))
                parts.append(Blob(encoded.getvalue(), "audio/wav"))
                end = self.started_at + timedelta(milliseconds=self.audio_end_ms)
                times.extend((end - timedelta(seconds=self.audio_bytes / 32_000), end))
            self.queried_audio_samples = self.audio_samples
            if not times:
                continue
            occurred_at, occurred_end = min(times), max(times)
            self.query_revision += 1
            event: dict[str, object] = {
                "type": "sensory_recall",
                "source_id": self.source_id,
                "snapshot_id": uuid4().hex,
                "revision": self.query_revision,
                "memory_revision": revision,
                "frame_sequences": [] if frame is None else [frame.sequence],
                "occurred_at": occurred_at.isoformat(),
                "occurred_end": occurred_end.isoformat(),
                "expires_at": (occurred_end + timedelta(seconds=8)).isoformat(),
                "query": parts[0],
                "status": "searching",
                "hits": [],
            }
            await self.recall.publish(event)
            helper = AsyncOmniPrefetch(
                self.memory, budget=ContextBudget(max_chars=64_000, max_items=4, max_media_items=8)
            )
            try:
                helper.submit(tuple(parts))
                result = await helper.finalize()
                if (
                    self.closed
                    or revision != self.memory_revision
                    or camera_epoch != self.camera_epoch
                ):
                    continue
                assert result.bundle is not None
                hits = await memory_cards(self.memory, result.hits)
                if (
                    self.closed
                    or revision != self.memory_revision
                    or camera_epoch != self.camera_epoch
                ):
                    continue
                self.memory_context = json.dumps(
                    {
                        "query": parts[0],
                        "input_occurred_at": occurred_at.isoformat(),
                        "input_occurred_end": occurred_end.isoformat(),
                        "historical_recall": result.bundle.render(),
                    },
                    ensure_ascii=False,
                )
                await self.recall.publish(
                    {
                        **event,
                        "status": "ready",
                        "expires_at": (
                            datetime.now(timezone.utc) + timedelta(seconds=20)
                        ).isoformat(),
                        "hits": hits,
                        "context": self._context_text(),
                    }
                )
            except SenseMemError:
                if not self.closed:
                    await self.recall.publish({**event, "status": "error"})
            finally:
                await helper.close()

    async def close(self) -> None:
        self.closed = True
        self.changed.set()
        self.latest = None
        self.audio.clear()
        if self.task is not None:
            with suppress(SenseMemError):
                await self.task
        await self.perception.close()
        if self.memory_capture is not None:
            await self.memory_capture.close()
