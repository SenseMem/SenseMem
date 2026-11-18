"""Typed computations that derive interpretation without changing original evidence."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sensemem.exceptions import ValidationError
from sensemem.models.base import ModelInput


@dataclass(frozen=True, slots=True)
class StageResult:
    """An untrusted interpretation; empty text means the stage derived nothing."""

    interpretation: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.interpretation, str):
            raise ValidationError("stage interpretation must be text")
        object.__setattr__(self, "interpretation", self.interpretation.strip())


@runtime_checkable
class IngestionStage(Protocol):
    """Compute derived text from leased input, with an explicit durable recipe identity.

    Stages run in the configured order before built-in perception and embedding. Each sees
    interpretations from preceding stages. Results cannot replace source text, assets, IDs,
    metadata, or semantic authority. The owning Memory closes each distinct stage once.
    """

    @property
    def name(self) -> str: ...

    @property
    def space(self) -> str: ...

    def process(self, input: ModelInput) -> StageResult: ...

    def close(self) -> None: ...
