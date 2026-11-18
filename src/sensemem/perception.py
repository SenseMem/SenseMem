"""Speech and visual analysis values, with source-local naming proposals."""

import math
import re
from dataclasses import dataclass

from sensemem.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class SpeakerName:
    speaker_label: str
    name: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.speaker_label, str)
            or re.fullmatch(r"speaker_[1-9][0-9]{0,2}", self.speaker_label) is None
        ):
            raise ValidationError("speaker name must refer to a local speaker_N label")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("speaker name must not be blank")
        object.__setattr__(self, "name", self.name.strip())


@dataclass(frozen=True, slots=True)
class VisualDescription:
    text: str = ""
    facts: tuple[str, ...] = ()
    speaker_names: tuple[SpeakerName, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("visual description text must be text")
        facts = tuple(self.facts)
        names = tuple(self.speaker_names)
        if any(not isinstance(fact, str) or not fact.strip() for fact in facts):
            raise ValidationError("visual facts must be non-empty text")
        if any(not isinstance(name, SpeakerName) for name in names):
            raise ValidationError("visual names must contain SpeakerName values")
        if not self.text.strip() and not facts:
            raise ValidationError("visual description must contain text or facts")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "facts", tuple(fact.strip() for fact in facts))
        object.__setattr__(self, "speaker_names", names)

    @property
    def content(self) -> str:
        return "\n".join((self.text, *self.facts)).strip()


@dataclass(frozen=True, slots=True)
class SpeechTurn:
    """One timed local-speaker turn returned by a speech backend."""

    start_ms: int
    end_ms: int
    text: str
    speaker_label: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_ms, bool)
            or not isinstance(self.start_ms, int)
            or isinstance(self.end_ms, bool)
            or not isinstance(self.end_ms, int)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValidationError("speech turn must have a positive time range")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValidationError("speech turn text must not be blank")
        object.__setattr__(self, "text", self.text.strip())
        if self.speaker_label is not None:
            if not isinstance(self.speaker_label, str) or not self.speaker_label.strip():
                raise ValidationError("speech turn speaker_label must not be blank")
            object.__setattr__(self, "speaker_label", self.speaker_label.strip())


@dataclass(frozen=True, slots=True)
class SpeakerEmbedding:
    """One backend-local speaker label and its voice identity exemplar."""

    speaker_label: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_label, str) or not self.speaker_label.strip():
            raise ValidationError("speaker embedding label must not be blank")
        values = tuple(self.values)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValidationError("speaker embedding must contain finite values")
        object.__setattr__(self, "speaker_label", self.speaker_label.strip())
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class SpeechAnalysis:
    """Timed transcript and voice identity exemplars for one media asset."""

    turns: tuple[SpeechTurn, ...]
    speakers: tuple[SpeakerEmbedding, ...]

    def __post_init__(self) -> None:
        turns, speakers = tuple(self.turns), tuple(self.speakers)
        if any(not isinstance(turn, SpeechTurn) for turn in turns) or any(
            not isinstance(speaker, SpeakerEmbedding) for speaker in speakers
        ):
            raise ValidationError("speech analysis contains invalid values")
        object.__setattr__(self, "turns", turns)
        object.__setattr__(self, "speakers", speakers)


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    """One detected face and its backend-local identity exemplar."""

    face_label: str
    values: tuple[float, ...]
    bounding_box: tuple[float, float, float, float]
    observed_at_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.face_label, str) or not self.face_label.strip():
            raise ValidationError("face embedding label must not be blank")
        values = tuple(self.values)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValidationError("face embedding must contain finite values")
        box = tuple(self.bounding_box)
        if len(box) != 4 or any(not math.isfinite(value) for value in box):
            raise ValidationError("face bounding_box must contain four finite values")
        x, y, width, height = box
        if (
            x < 0.0
            or y < 0.0
            or width <= 0.0
            or height <= 0.0
            or x + width > 1.0
            or y + height > 1.0
        ):
            raise ValidationError("face bounding_box must be normalized within the frame")
        if self.observed_at_ms is not None and (
            isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 0
        ):
            raise ValidationError("face observed_at_ms must be a non-negative integer")
        object.__setattr__(self, "face_label", self.face_label.strip())
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "bounding_box", box)


@dataclass(frozen=True, slots=True)
class FaceAnalysis:
    """Detected face exemplars for one image or sampled video."""

    faces: tuple[FaceEmbedding, ...]

    def __post_init__(self) -> None:
        faces = tuple(self.faces)
        if any(not isinstance(face, FaceEmbedding) for face in faces):
            raise ValidationError("face analysis contains invalid values")
        labels = {face.face_label for face in faces}
        if len(labels) != len(faces):
            raise ValidationError("face analysis labels must be unique")
        object.__setattr__(self, "faces", faces)
