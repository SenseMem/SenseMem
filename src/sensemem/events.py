"""Immutable media windows and proposed semantic event boundaries."""

from dataclasses import dataclass, field
from datetime import datetime

from sensemem._validation import nonempty_text, positive_int
from sensemem._value_validation import _require_interval
from sensemem.content import Blob
from sensemem.exceptions import ValidationError
from sensemem.spatial import ObservationContext


@dataclass(frozen=True, slots=True)
class EventChunk:
    """One timed capture window, optionally pairing sound with an image or video.

    Windows are transport units, not semantic events. Each encoded blob must be independently
    decodable; paired blobs describe the same interval. Use one stream ID for synchronized media.
    """

    content: Blob | tuple[Blob, ...]
    occurred_at: datetime
    occurred_end: datetime
    stream_id: str = "default"
    context: ObservationContext | None = None
    media: tuple[Blob, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        media = (self.content,) if isinstance(self.content, Blob) else self.content
        if (
            not isinstance(media, tuple)
            or not 1 <= len(media) <= 2
            or any(not isinstance(blob, Blob) for blob in media)
        ):
            raise ValidationError("event chunk must contain one or two media Blobs")
        modalities = tuple(blob.media_type.split("/", 1)[0] for blob in media)
        if any(value not in {"image", "audio", "video"} for value in modalities) or (
            len(media) == 2 and ("audio" not in modalities or len(set(modalities)) != 2)
        ):
            raise ValidationError(
                "event chunk must contain audio, visual media, or a paired capture"
            )
        if self.occurred_at is None or self.occurred_end is None:
            raise ValidationError("event chunk requires both capture timestamps")
        _require_interval(self.occurred_at, self.occurred_end)
        if self.context is not None and not isinstance(self.context, ObservationContext):
            raise ValidationError("event chunk context must be an ObservationContext")
        object.__setattr__(self, "stream_id", nonempty_text(self.stream_id, "stream_id"))
        object.__setattr__(self, "media", media)


@dataclass(frozen=True, slots=True)
class EventSegment:
    """One event assessment ending at an exclusive observation offset.

    The first event starts at zero; subsequent events start at their predecessor's end.
    Only ``complete=True`` establishes finality. A final ``complete=False`` item summarizes
    the entire unfinished suffix for the next assessment; its text remains model interpretation.
    """

    end_chunk: int
    summary: str
    complete: bool = True

    def __post_init__(self) -> None:
        positive_int(self.end_chunk, "event end_chunk")
        summary = nonempty_text(self.summary, "event summary")
        if len(summary) > 16_384:
            raise ValidationError("event summary must not exceed 16384 characters")
        if not isinstance(self.complete, bool):
            raise ValidationError("event complete must be a boolean")
        object.__setattr__(self, "summary", summary)
