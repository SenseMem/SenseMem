"""Typed visual observations and explicit, source-local naming proposals."""

import re
from dataclasses import dataclass

from mindbridge.exceptions import ValidationError


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
