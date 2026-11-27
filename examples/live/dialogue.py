"""Validated dialogue and rendered Agent audio as attributed native observations."""

from __future__ import annotations

import base64
import binascii
import io
import wave
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from sensemem import (
    AsyncMemory,
    Blob,
    EvidenceBasis,
    MemoryCategory,
    MemoryRecord,
    ObservationContext,
)

DialogueItemId = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"^\S+$")]


class DialogueUser(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: DialogueItemId
    text: str = Field(strict=True, min_length=1, max_length=4000, pattern=r"\S")


class DialogueAssistant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_ids: tuple[DialogueItemId, ...] = Field(min_length=1, max_length=8)
    text: str = Field(strict=True, max_length=8000)


class DialoguePlayback(BaseModel):
    """PCM actually rendered by the client's Web Audio clock, not proof of hearing."""

    model_config = ConfigDict(extra="forbid")
    sample_rate: int = Field(strict=True, ge=24000, le=24000)
    pcm_s16le: str = Field(strict=True, min_length=4, max_length=19_200_000)
    rendered_samples: int = Field(strict=True, ge=1, le=7_200_000)
    status: Literal["completed", "interrupted", "stopped"]
    _pcm: bytes = PrivateAttr(default=b"")

    @model_validator(mode="after")
    def decode_pcm(self) -> Self:
        try:
            self._pcm = base64.b64decode(self.pcm_s16le, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("playback PCM must be valid base64") from None
        if len(self._pcm) != self.rendered_samples * 2:
            raise ValueError(
                "playback PCM must contain exactly rendered_samples mono PCM16 samples"
            )
        return self

    def wav(self) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(self.sample_rate)
            audio.writeframes(self._pcm)
        return output.getvalue()


class PerceptionSource(BaseModel):
    """Timed transient frames, distinct from committed native memory evidence IDs."""

    model_config = ConfigDict(extra="forbid")
    frame_sequences: tuple[Annotated[int, Field(strict=True, ge=1)], ...] = Field(
        min_length=1, max_length=8
    )
    occurred_at: AwareDatetime
    occurred_end: AwareDatetime

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if (
            self.occurred_end < self.occurred_at
            or any(value < 1 for value in self.frame_sequences)
            or tuple(sorted(set(self.frame_sequences))) != self.frame_sequences
        ):
            raise ValueError("perception provenance must retain ordered frames and times")
        return self


class FeedbackReceipt(BaseModel):
    """Client-reported native input acknowledgment, not proof of semantic adoption."""

    model_config = ConfigDict(extra="forbid")
    input_id: DialogueItemId
    feedback_id: DialogueItemId | None = None
    memory_revision: int = Field(strict=True, ge=0)
    memory_ids: tuple[DialogueItemId, ...] = Field(max_length=2)
    candidate_id: DialogueItemId | None = None
    action: Literal["quiet", "respond"]
    event_id: DialogueItemId
    operation: Literal["snapshot", "replace", "withdraw"]


class DialogueInput(BaseModel):
    """An attributed reply with optional rendered audio and a precise playback status."""

    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(strict=True, pattern=r"^live:[a-f0-9]{32}$")
    response_id: DialogueItemId
    started_at: AwareDatetime
    completed_at: AwareDatetime
    origin: Literal["user", "proactive", "perception", "realtime"] = "user"
    user: DialogueUser | None = None
    assistant: DialogueAssistant
    candidate_id: DialogueItemId | None = None
    event_id: DialogueItemId | None = None
    evidence_ids: tuple[DialogueItemId, ...] = Field(default=(), max_length=32)
    perception: PerceptionSource | None = None
    playback: DialoguePlayback | None = None
    feedback_receipts: tuple[FeedbackReceipt, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def valid_turn(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("dialogue completion precedes its input")
        if len(set(self.assistant.item_ids)) != len(self.assistant.item_ids):
            raise ValueError("assistant item identifiers must be unique")
        if self.user is not None and self.user.item_id in self.assistant.item_ids:
            raise ValueError("user and assistant items must be distinct")
        if self.origin != "realtime" and (
            not self.assistant.text.strip() or self.playback is not None or self.feedback_receipts
        ):
            raise ValueError("a non-realtime dialogue needs generated text and no native playback")
        if self.origin == "realtime" and not self.assistant.text.strip() and self.playback is None:
            raise ValueError("a native realtime reply needs generated text or rendered audio")
        if self.origin == "user" and (
            self.user is None
            or self.candidate_id is not None
            or self.event_id is not None
            or self.evidence_ids
            or self.perception is not None
        ):
            raise ValueError("a user dialogue needs only its authoritative user input")
        if self.origin == "realtime" and (
            self.candidate_id is not None
            or self.event_id is not None
            or self.evidence_ids
            or self.perception is not None
        ):
            raise ValueError("a realtime reply keeps trigger provenance in its feedback receipts")
        if self.origin == "proactive" and (
            self.user is not None
            or not self.candidate_id
            or not self.event_id
            or not self.evidence_ids
            or self.perception is not None
        ):
            raise ValueError(
                "a proactive dialogue needs native event evidence and no invented user"
            )
        if self.origin == "perception" and (
            self.user is not None
            or not self.candidate_id
            or not self.event_id
            or self.evidence_ids
            or self.perception is None
        ):
            raise ValueError(
                "a visual reply needs issued transient frame provenance, not memory IDs"
            )
        return self


async def capture_dialogue(
    memory: AsyncMemory,
    supplied: DialogueInput,
    *,
    reviewed_memory_ids: list[str] | None = None,
) -> MemoryRecord:
    """Capture a validated reply after the caller checks any issued trigger evidence."""
    reviewed_evidence: dict[str, object] = {}
    playback = supplied.playback
    content = (
        (f"User:\n{supplied.user.text}\n\n" if supplied.user is not None else "")
        + f"SenseMem:\n{supplied.assistant.text if supplied.assistant.text.strip() else 'Voice reply'}"
    )
    if supplied.origin == "proactive":
        assert reviewed_memory_ids is not None
        reviewed_evidence = {"reviewed_memory_ids": reviewed_memory_ids}
    elif supplied.origin == "realtime":
        reviewed_evidence = {
            "speaker": "assistant",
            "playback_clock": "web_audio_render_clock",
            "actual_hearing_verified": False,
            "generated_text_alignment": "unverified",
            "input_transcript_basis": (
                "client_reported_final_asr" if supplied.user is not None else "absent"
            ),
            "feedback_receipt_basis": "client_reported_native_input_acknowledgment",
            "feedback_semantic_adoption_verified": False,
        }
    captured: str | tuple[str, Blob] = content
    if playback is not None:
        captured = (content, Blob(playback.wav(), "audio/wav", "agent-playback.wav"))
    return await memory.capture(
        captured,
        occurred_at=supplied.started_at,
        occurred_end=supplied.completed_at,
        memory_type=MemoryCategory.EPISODIC,
        # This observes who said what; the Agent's claims are not user statements.
        context=ObservationContext(source_id=supplied.source_id, basis=EvidenceBasis.OBSERVATION),
        metadata={
            "dialogue": {
                **supplied.model_dump(mode="json", exclude={"playback": {"pcm_s16le"}}),
                "completion": (
                    "generated"
                    if supplied.origin != "realtime"
                    else "text_only"
                    if playback is None
                    else playback.status
                ),
                **reviewed_evidence,
            }
        },
    )
