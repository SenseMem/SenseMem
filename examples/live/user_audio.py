"""Capture completed native user speech independently of visual event completion."""

from __future__ import annotations

import asyncio
import io
import json
import wave
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

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

MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_OPEN_TURNS = 8


class _InputEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    item_id: str = Field(min_length=1, max_length=128)


class _Started(_InputEvent):
    type: Literal["input_audio_buffer.speech_started"]
    audio_start_ms: int = Field(ge=0)


class _Stopped(_InputEvent):
    type: Literal["input_audio_buffer.speech_stopped"]
    audio_end_ms: int = Field(ge=0)


class _Transcribed(_InputEvent):
    type: Literal["conversation.item.input_audio_transcription.completed"]
    transcript: str = Field(max_length=16_384)


_INPUT = TypeAdapter(Annotated[_Started | _Stopped | _Transcribed, Field(discriminator="type")])
_EVENT_TYPES = frozenset(
    {
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio_transcription.completed",
    }
)


@dataclass(slots=True)
class _Turn:
    start_ms: int | None = None
    end_ms: int | None = None
    transcript: str | None = None


@dataclass(frozen=True, slots=True)
class _Utterance:
    item_id: str
    start_ms: int
    end_ms: int
    transcript: str
    pcm: bytes


class UserAudioCapture:
    """Accept a PCM copy synchronously; durable writes run in one background SDK stream.

    VAD offsets use the native service's cumulative input-audio clock, not ASR arrival time.
    Only a stopped item with its authoritative final ASR and all original samples commits.
    The continuous audiovisual event stream remains an independent source of evidence.
    """

    def __init__(
        self,
        memory: AsyncMemory,
        source_id: str,
        *,
        committed: Callable[[StreamCommit], None],
        changed: Callable[[], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.stream = AsyncCaptureStream(memory, capture=True)
        self.source_id = source_id
        self.committed = committed
        self.changed = changed
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sample_rate = 0
        self.samples = 0
        self.origin: datetime | None = None
        self.audio: deque[tuple[int, bytes]] = deque()
        self.audio_bytes = 0
        self.turns: dict[str, _Turn] = {}
        self.completed: set[str] = set()
        self.pending: deque[_Utterance] = deque()
        self.pending_bytes = 0
        self.task: asyncio.Task[None] | None = None
        self.failure: str | None = None
        self.processing = False
        self.closed = False

    def _fail(self, code: str) -> None:
        self.failure = code
        if self.changed is not None:
            self.changed()

    def append_audio(self, data: bytes, sample_rate: int) -> None:
        """Retain actual upstream PCM16 after forwarding it to the speech service."""
        if self.closed:
            return
        if self.sample_rate and self.sample_rate != sample_rate:
            self._fail("audio_format_changed")
            return
        if not self.sample_rate:
            self.sample_rate = sample_rate
            self.origin = self.clock() - timedelta(seconds=len(data) / (2 * sample_rate))
        self.audio.append((self.samples, data))
        self.samples += len(data) // 2
        self.audio_bytes += len(data)
        while self.audio_bytes > MAX_AUDIO_BYTES:
            self.audio_bytes -= len(self.audio.popleft()[1])
        # The receive pump can observe finality before the send pump resumes its PCM copy.
        for item_id in tuple(self.turns):
            self._ready(item_id)

    def observe_event(self, text: str) -> None:
        """Read server VAD/final ASR only; partials and Agent output never become memories."""
        if self.closed:
            return
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict) or event.get("type") not in _EVENT_TYPES:
            return
        try:
            supplied = _INPUT.validate_python(event)
        except PydanticValidationError:
            self._fail("invalid_speech_event")
            return
        item_id = supplied.item_id
        if item_id in self.completed:
            return
        if item_id not in self.turns:
            if len(self.turns) >= MAX_OPEN_TURNS:
                self.turns.pop(next(iter(self.turns)))
                self._fail("audio_capture_capacity")
            self.turns[item_id] = _Turn()
        turn = self.turns[item_id]
        if isinstance(supplied, _Started):
            turn.start_ms = supplied.audio_start_ms
        elif isinstance(supplied, _Stopped):
            turn.end_ms = supplied.audio_end_ms
        else:
            turn.transcript = supplied.transcript.strip()
        self._ready(item_id)

    def _ready(self, item_id: str) -> None:
        turn = self.turns[item_id]
        if turn.start_ms is None or turn.end_ms is None or turn.transcript is None:
            return
        if not turn.transcript:
            self.turns.pop(item_id)
            self.completed.add(item_id)
            return
        if turn.end_ms <= turn.start_ms:
            self.turns.pop(item_id)
            self.completed.add(item_id)
            self._fail("invalid_speech_interval")
            return
        if not self.sample_rate:
            return
        start = turn.start_ms * self.sample_rate // 1000
        end = turn.end_ms * self.sample_rate // 1000
        if end > self.samples:
            return
        self.turns.pop(item_id)
        self.completed.add(item_id)
        if not self.audio or start < self.audio[0][0]:
            self._fail("audio_capture_capacity")
            return
        pcm = b"".join(
            data[max(0, start - position) * 2 : min(len(data), (end - position) * 2)]
            for position, data in self.audio
            if position < end and position + len(data) // 2 > start
        )
        if self.pending_bytes + len(pcm) > MAX_AUDIO_BYTES:
            self._fail("audio_capture_capacity")
            return
        self.pending.append(_Utterance(item_id, turn.start_ms, turn.end_ms, turn.transcript, pcm))
        self.pending_bytes += len(pcm)
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._consume())

    async def _events(self) -> AsyncIterator[StreamEvent]:
        while self.pending:
            utterance = self.pending.popleft()
            self.pending_bytes -= len(utterance.pcm)
            assert self.origin is not None
            encoded = io.BytesIO()
            with wave.open(encoded, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(self.sample_rate)
                audio.writeframes(utterance.pcm)
            yield StreamEvent(
                StreamPhase.FINAL,
                StreamInput(
                    Blob(encoded.getvalue(), "audio/wav", "user-speech.wav"),
                    occurred_at=self.origin + timedelta(milliseconds=utterance.start_ms),
                    occurred_end=self.origin + timedelta(milliseconds=utterance.end_ms),
                    memory_type=MemoryCategory.EPISODIC,
                    context=ObservationContext(source_id=self.source_id),
                    transcript=utterance.transcript,
                    metadata={
                        "speaker": "user",
                        "speech": {
                            "item_id": utterance.item_id,
                            "boundary": "server_vad",
                            "transcript_basis": "server_final_asr",
                            "transcript_alignment": "unverified",
                            "audio_start_ms": utterance.start_ms,
                            "audio_end_ms": utterance.end_ms,
                            "sample_rate_hz": self.sample_rate,
                            "clock": "native_input_pcm",
                            "wall_clock_origin": "first_pcm_received_minus_packet_duration",
                        },
                    },
                ),
                utterance.item_id,
            )

    async def _consume(self) -> None:
        self.processing = True
        if self.changed is not None:
            self.changed()
        try:
            async for commit in self.stream.consume(self._events()):
                self.committed(commit)
        except SenseMemError:
            self._fail("audio_capture_failed")
        finally:
            self.processing = False
            if self.changed is not None:
                self.changed()

    async def close(self) -> None:
        """Drain already completed utterances; disconnect does not finalize open hypotheses."""
        self.closed = True
        self.turns.clear()
        self.audio.clear()
        self.audio_bytes = 0
        if self.task is not None:
            await self.task
