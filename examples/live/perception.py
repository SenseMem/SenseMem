"""Bounded, temporary visual observations beside speech and durable memory work."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from sensemem import (
    AssetRef,
    GenerationBackend,
    Modality,
    ModelInput,
    SenseMemError,
    ValidationError,
)
from sensemem.prompts import demo as demo_prompts

MAX_FRAME_BYTES = 256 * 1024
PerceptionAction = Literal["quiet", "context", "wait", "interrupt"]
PerceptionFailure = Literal["model_failed", "invalid_output"]


@dataclass(frozen=True, slots=True)
class PerceptionContext:
    """Actual interaction state captured once when a visual review starts."""

    latest_user_text: str = ""
    assistant_text: str = ""
    response_id: str | None = None
    user_speaking: bool = False
    generating: bool = False
    playing: bool = False
    memory_context: str = ""
    memory_revision: int = 0


class PerceptionFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence: int = Field(ge=1)
    captured_at: AwareDatetime
    timestamp_ms: float = Field(ge=0, allow_inf_nan=False)
    jpeg: bytes = Field(min_length=4, max_length=MAX_FRAME_BYTES, repr=False, exclude=True)

    @model_validator(mode="after")
    def jpeg_markers(self) -> Self:
        if not self.jpeg.startswith(b"\xff\xd8") or not self.jpeg.endswith(b"\xff\xd9"):
            raise ValueError("a perception frame must contain JPEG bytes")
        return self


class FrameEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)

    sequence: int = Field(ge=1)
    observation: str = Field(min_length=1, max_length=400)


class _Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=1200)
    changed: bool
    action: PerceptionAction
    utterance: str | None = Field(min_length=1, max_length=400)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evidence: tuple[FrameEvidence, ...] = Field(max_length=8)
    reason: str = Field(min_length=1, max_length=400)
    contradicted_claim: str | None = Field(min_length=1, max_length=400)
    correction: str | None = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        sequences = {item.sequence for item in self.evidence}
        if len(sequences) != len(self.evidence):
            raise ValueError("frame evidence must have unique sequences")
        if self.changed and len(sequences) < 2:
            raise ValueError("a temporal change needs at least two observed frames")
        if self.action != "quiet" and not sequences:
            raise ValueError("an actionable observation needs visible frame evidence")
        if self.action in {"quiet", "context"} and self.utterance is not None:
            raise ValueError("quiet or context observations cannot request speech")
        if (self.contradicted_claim is None) != (self.correction is None):
            raise ValueError("a correction needs the concrete contradicted claim")
        if self.action == "interrupt" and (
            self.confidence < 0.9
            or len(sequences) < 2
            or not self.utterance
            or self.correction is None
        ):
            raise ValueError("interruptions need a confident, evidenced concrete correction")
        return self


class PerceptionResult(_Decision):
    """An explicitly timed past window; only the owner may decide whether to act on it."""

    source_id: str
    frame_sequences: tuple[int, ...]
    occurred_at: AwareDatetime
    occurred_end: AwareDatetime
    timestamp_ms: float
    end_timestamp_ms: float
    context: PerceptionContext
    target_response_id: str | None
    current: bool
    latest_sequence: int
    expires_at: AwareDatetime
    epoch: int
    frames: tuple[PerceptionFrame, ...] = Field(default=(), exclude=True, repr=False)


class LivePerception:
    """One model call and one coalesced pending window, without durable writes.

    The caller owns the shared generation backend. A result can describe a valid past window
    even when newer frames have arrived; it is published with current=False in that case.
    """

    def __init__(
        self,
        source_id: str,
        generation: GenerationBackend,
        publish: Callable[[PerceptionResult], Awaitable[None]],
        context: Callable[[], PerceptionContext],
        *,
        max_frames: int = 8,
        max_bytes: int = 2 * 1024 * 1024,
        window_seconds: float = 6,
        max_age_seconds: float = 8,
        clock: Callable[[], datetime] | None = None,
        on_error: Callable[[PerceptionFailure], Awaitable[None]] | None = None,
    ) -> None:
        if not {Modality.TEXT, Modality.IMAGE} <= generation.generation_capabilities:
            raise ValidationError("visual perception requires native text and image generation")
        if not source_id.strip() or not 2 <= max_frames <= 8 or max_bytes < 1:
            raise ValidationError("perception needs a source and a bounded frame window")
        if not 0 < window_seconds <= 6 or not 0 < max_age_seconds <= 8:
            raise ValidationError("perception time windows exceed their live bounds")
        self.source_id = source_id
        self.generation = generation
        self.publish = publish
        self.context = context
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.window_ms = window_seconds * 1000
        self.max_age = timedelta(seconds=max_age_seconds)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.on_error = on_error
        self.pending: list[PerceptionFrame] = []
        self.anchor: tuple[PerceptionFrame, ...] = ()
        self.latest: PerceptionFrame | None = None
        self.changed = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.closed = False
        self.failure: PerceptionFailure | None = None
        self.epoch = 0
        self.last_digest: bytes | None = None
        self.last_context: PerceptionContext | None = None

    def submit(self, frame: PerceptionFrame) -> None:
        """Accept a validated frame synchronously, without waiting for the model."""
        if self.closed:
            raise ValidationError("the perception session is closed")
        if len(frame.jpeg) > self.max_bytes:
            raise ValidationError("the frame exceeds the perception window capacity")
        if self.latest is not None and (
            frame.sequence <= self.latest.sequence
            or frame.timestamp_ms <= self.latest.timestamp_ms
            or frame.captured_at < self.latest.captured_at
        ):
            raise ValidationError("perception frames must have increasing sequence and time")
        self.latest = frame
        self.pending = self._bounded([*self.pending, frame])
        self.changed.set()
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    def _bounded(self, frames: list[PerceptionFrame]) -> list[PerceptionFrame]:
        cutoff = frames[-1].timestamp_ms - self.window_ms
        frames = [frame for frame in frames if frame.timestamp_ms >= cutoff]
        size = sum(len(frame.jpeg) for frame in frames)
        while len(frames) > self.max_frames or size > self.max_bytes:
            # Retain temporal endpoints; remove the densest interior sample first.
            index = (
                min(
                    range(1, len(frames) - 1),
                    key=lambda item: frames[item + 1].timestamp_ms - frames[item - 1].timestamp_ms,
                )
                if len(frames) > 2
                else 0
            )
            size -= len(frames.pop(index).jpeg)
        return frames

    async def _run(self) -> None:
        while not self.closed:
            await self.changed.wait()
            self.changed.clear()
            if self.closed or not self.pending:
                continue
            pending = tuple(self.pending)
            self.pending.clear()
            context = self.context()
            if context == self.last_context and all(
                hashlib.sha256(frame.jpeg).digest() == self.last_digest for frame in pending
            ):
                # Reuse only byte-identical imagery under the same interaction snapshot.
                # Keep the earlier observation's real times; do not mint a fresh spoken cue.
                self.anchor = (*self.anchor, *pending)[-2:]
                continue
            frames = tuple(self._bounded([*self.anchor, *pending]))
            epoch = self.epoch
            try:
                decision = await asyncio.to_thread(self._analyze, frames, context)
            except (PydanticValidationError, ValueError):
                failure: PerceptionFailure | None = "invalid_output"
            except SenseMemError:
                failure = "model_failed"
            else:
                failure = None
                if not self.closed and epoch == self.epoch:
                    assert self.latest is not None
                    now = self.clock()
                    expires_at = frames[-1].captured_at + self.max_age
                    result = PerceptionResult(
                        **decision.model_dump(),
                        source_id=self.source_id,
                        frame_sequences=tuple(frame.sequence for frame in frames),
                        occurred_at=frames[0].captured_at,
                        occurred_end=frames[-1].captured_at,
                        timestamp_ms=frames[0].timestamp_ms,
                        end_timestamp_ms=frames[-1].timestamp_ms,
                        context=context,
                        target_response_id=context.response_id,
                        current=self.latest.sequence == frames[-1].sequence
                        and frames[-1].captured_at <= now <= expires_at,
                        latest_sequence=self.latest.sequence,
                        expires_at=expires_at,
                        epoch=epoch,
                        frames=frames,
                    )
                    self.last_digest = hashlib.sha256(frames[-1].jpeg).digest()
                    self.last_context = context
                    await self.publish(result)
            if self.closed or epoch != self.epoch:
                continue
            self.failure = failure
            self.anchor = frames[-2:]
            if self.failure is not None and self.on_error is not None and not self.closed:
                await self.on_error(self.failure)

    def _analyze(
        self, frames: tuple[PerceptionFrame, ...], context: PerceptionContext
    ) -> _Decision:
        with TemporaryDirectory(prefix="sensemem-live-perception-") as directory:
            assets: list[AssetRef] = []
            manifest: list[dict[str, object]] = []
            for frame in frames:
                path = Path(directory) / f"frame-{frame.sequence}.jpg"
                path.write_bytes(frame.jpeg)
                assets.append(
                    AssetRef(
                        f"frame:{frame.sequence}",
                        modality=Modality.IMAGE,
                        media_type="image/jpeg",
                        size_bytes=len(frame.jpeg),
                        sha256=hashlib.sha256(frame.jpeg).hexdigest(),
                        path=path,
                    )
                )
                manifest.append(
                    {
                        "sequence": frame.sequence,
                        "captured_at": frame.captured_at.isoformat(),
                        "timestamp_ms": frame.timestamp_ms,
                    }
                )
            question = ModelInput(
                text=demo_prompts.VISUAL_PERCEPTION_INSTRUCTIONS
                + "\nUntrusted interaction snapshot and ordered frame manifest:\n"
                + json.dumps({"context": asdict(context), "frames": manifest}, ensure_ascii=False),
                assets=tuple(assets),
            )
            # There are no durable memory hits: use the original temporary images directly.
            answer = self.generation.answer(question, (), answer_policy="best_effort")
            decision = _Decision.model_validate_json(answer.answer)
            supplied = {frame.sequence for frame in frames}
            evidence = tuple(item.sequence for item in decision.evidence)
            if not set(evidence) <= supplied or evidence != tuple(sorted(evidence)):
                raise ValueError("perception evidence is outside the observed frame window")
            if decision.contradicted_claim is not None and (
                decision.contradicted_claim not in context.assistant_text
            ):
                raise ValueError("the claimed contradiction is absent from the active response")
            if decision.action == "interrupt" and (
                context.response_id is None
                or not (context.generating or context.playing)
                or context.user_speaking
            ):
                raise ValueError("the snapshot does not permit an interruption suggestion")
            return decision

    def reset(self) -> None:
        """Withdraw one camera epoch without starting a second model worker."""
        self.epoch += 1
        self.pending.clear()
        self.anchor = ()
        self.latest = None
        self.last_digest = None
        self.last_context = None
        self.failure = None
        self.changed.clear()

    async def close(self) -> None:
        """Drain this worker, suppress late results, and leave the shared backend open."""
        self.closed = True
        self.changed.set()
        if self.task is not None:
            await self.task
        self.reset()
