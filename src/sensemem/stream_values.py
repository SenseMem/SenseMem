"""Stream values values and pure operations for the public SDK."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypeAlias

from sensemem._validation import positive_int
from sensemem._value_validation import _optional_text, _require_aware, _require_interval, _text
from sensemem.content import AssetRef, Blob, ContentInput, _stream_content
from sensemem.context_bundle import ContextBundle
from sensemem.exceptions import ValidationError
from sensemem.records import MemoryRecord, SearchHit
from sensemem.spatial import ObservationContext
from sensemem.types import MemoryCategory


@dataclass(frozen=True, slots=True)
class StreamInput:
    """One durable stream observation, with model interpretation separate from original input."""

    content: ContentInput
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    metadata: Mapping[str, object] | None = field(default=None, hash=False)
    memory_type: MemoryCategory = MemoryCategory.SEMANTIC
    context: ObservationContext | None = None
    transcript: str | None = None
    description: str | None = None
    interpretation: str | None = None

    def __post_init__(self) -> None:
        content = _stream_content(self.content)
        _require_interval(self.occurred_at, self.occurred_end)
        if not isinstance(self.memory_type, MemoryCategory):
            raise ValidationError("memory_type is invalid")
        metadata = self.metadata
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise ValidationError("metadata must be a mapping")
            metadata = dict(metadata)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "metadata", metadata)
        if self.context is not None and not isinstance(self.context, ObservationContext):
            raise ValidationError("stream context must be an ObservationContext")
        object.__setattr__(
            self,
            "transcript",
            _optional_text(self.transcript, "stream transcript"),
        )
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, "stream description"),
        )
        object.__setattr__(
            self,
            "interpretation",
            _optional_text(self.interpretation, "stream interpretation"),
        )


class AudioBoundary(str, Enum):
    """A canonical acoustic segment boundary."""

    START = "start"
    END = "end"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class PCMChunk:
    """One immutable chunk of interleaved WAV-compatible linear PCM."""

    data: bytes = field(repr=False)
    sample_rate_hz: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValidationError("PCM chunk data must be non-empty bytes")
        for name in ("sample_rate_hz", "channels", "sample_width_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"PCM {name} must be a positive integer")
        if self.sample_width_bytes > 4:
            raise ValidationError("PCM sample_width_bytes must be between one and four")
        frame_bytes = self.channels * self.sample_width_bytes
        if frame_bytes > 0xFFFF or self.sample_rate_hz * frame_bytes > 0xFFFFFFFF:
            raise ValidationError("PCM format exceeds WAV limits")
        if len(self.data) % frame_bytes:
            raise ValidationError("PCM chunk data must contain complete sample frames")
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "PCM occurred_at")


@dataclass(frozen=True, slots=True)
class VADPacket:
    """One canonical voice-activity state transition."""

    active: bool
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.active, bool):
            raise ValidationError("VAD active must be a boolean")
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "VAD occurred_at")


@dataclass(frozen=True, slots=True)
class ASRPartial:
    """The complete current ASR hypothesis for one audio stream."""

    text: str
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("ASR partial text must be text")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "ASR occurred_at")


@dataclass(frozen=True, slots=True)
class AcousticBoundary:
    """Start, finish, or cancel one canonical acoustic segment."""

    boundary: AudioBoundary
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, AudioBoundary):
            try:
                object.__setattr__(self, "boundary", AudioBoundary(self.boundary))
            except (TypeError, ValueError):
                raise ValidationError("audio boundary is invalid") from None
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "audio boundary occurred_at")


AudioStreamPacket: TypeAlias = PCMChunk | VADPacket | ASRPartial | AcousticBoundary


class VisionBoundary(str, Enum):
    """A canonical visual scene boundary."""

    START = "start"
    END = "end"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class VisionFrame:
    """One immutable encoded image frame from a visual stream."""

    image: Blob
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, Blob) or not self.image.media_type.startswith("image/"):
            raise ValidationError("vision frame must contain an image Blob")
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "vision frame occurred_at")


@dataclass(frozen=True, slots=True)
class VisionPartial:
    """The complete current caption, OCR, or detector description for one scene."""

    text: str
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("vision partial text must be text")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "vision partial occurred_at")


@dataclass(frozen=True, slots=True)
class SceneBoundary:
    """Start, finish, or cancel one canonical visual scene."""

    boundary: VisionBoundary
    stream_id: str = "default"
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, VisionBoundary):
            try:
                object.__setattr__(self, "boundary", VisionBoundary(self.boundary))
            except (TypeError, ValueError):
                raise ValidationError("vision boundary is invalid") from None
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        _require_aware(self.occurred_at, "vision boundary occurred_at")


VisionStreamPacket: TypeAlias = VisionFrame | VisionPartial | SceneBoundary


class StreamPhase(str, Enum):
    """Lifecycle boundary emitted by a modality-specific capture adapter."""

    UPDATE = "update"
    FINAL = "final"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """A complete current snapshot, final durable observation, or cancellation."""

    phase: StreamPhase
    item: ContentInput | StreamInput | None = None
    stream_id: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.phase, StreamPhase):
            try:
                object.__setattr__(self, "phase", StreamPhase(self.phase))
            except (TypeError, ValueError):
                raise ValidationError("stream phase is invalid") from None
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        if self.phase is StreamPhase.CANCEL:
            if self.item is not None:
                raise ValidationError("cancel stream events must not contain an item")
            return
        if self.item is None:
            raise ValidationError("update and final stream events require an item")
        if isinstance(self.item, StreamInput):
            if self.phase is StreamPhase.UPDATE:
                raise ValidationError("update stream events require a content snapshot")
            content = self.item.content
        else:
            content = _stream_content(self.item)
            object.__setattr__(self, "item", content)
        atoms = (content,) if isinstance(content, (str, Path, Blob, AssetRef)) else tuple(content)
        if self.phase is StreamPhase.UPDATE and any(isinstance(atom, Path) for atom in atoms):
            raise ValidationError("update stream paths are mutable; use Blob or AssetRef")


@dataclass(frozen=True, slots=True)
class PrefetchResult:
    """The newest completed speculative search for one streaming turn."""

    hits: tuple[SearchHit, ...]
    bundle: ContextBundle | None = None

    def __post_init__(self) -> None:
        hits = tuple(self.hits)
        if any(not isinstance(hit, SearchHit) for hit in hits):
            raise ValidationError("prefetch hits are invalid")
        if self.bundle is not None and (
            not isinstance(self.bundle, ContextBundle) or self.bundle.hits != hits
        ):
            raise ValidationError("prefetch bundle must contain the returned hits")
        object.__setattr__(self, "hits", hits)


@dataclass(frozen=True, slots=True)
class PrefetchUpdate:
    """A completed speculative result tied to the revision returned by ``submit``."""

    revision: int
    result: PrefetchResult

    def __post_init__(self) -> None:
        positive_int(self.revision, "prefetch revision")
        if not isinstance(self.result, PrefetchResult):
            raise ValidationError("prefetch update result is invalid")


@dataclass(frozen=True, slots=True)
class StreamCommit:
    """One durable final observation, with retrieval skipped when awaiting settlement.

    `pending_settlement` is true when the reducer committed through `capture` rather than `add`:
    the record is durable and readable but has no vectors yet, so it stays invisible to `search`
    until the host calls `settle`. Both `prefetch` and `retrieval_error` may then be `None`.
    """

    record: MemoryRecord
    prefetch: PrefetchResult | None
    retrieval_error: str | None = None
    stream_id: str = "default"
    pending_settlement: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise ValidationError("stream commit values are invalid")
        if not isinstance(self.pending_settlement, bool):
            raise ValidationError("stream commit pending_settlement must be a boolean")
        if self.prefetch is not None and not isinstance(self.prefetch, PrefetchResult):
            raise ValidationError("stream prefetch result is invalid")
        error = _optional_text(self.retrieval_error, "stream retrieval_error")
        if (self.prefetch is not None and error is not None) or (
            self.prefetch is None and error is None and not self.pending_settlement
        ):
            raise ValidationError("stream commit requires either prefetch or retrieval_error")
        object.__setattr__(self, "retrieval_error", error)
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
