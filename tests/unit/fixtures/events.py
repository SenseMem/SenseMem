"""Shared event-stream, capture, and vision-describer fakes for unit tests."""

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta, timezone
from typing import cast

from _feature_support import ATOMIC_MODALITIES, TinyEmbedder

from sensemem import (
    Blob,
    EmbedTask,
    EventChunk,
    EventObservation,
    EventSegment,
    FormationInput,
    FormationProposal,
    Modality,
    ModelInput,
    RecordKind,
)

REFERENCE_AT = datetime(2026, 11, 14, 8, tzinfo=timezone.utc)


class ScriptedSegmenter:
    event_model = "event-test-snapshot"
    event_recipe = "event-test-snapshot:complete-prefix-v1"

    def __init__(self, replies: Sequence[object]) -> None:
        self._replies = iter(replies)
        self.calls: list[tuple[tuple[EventObservation, ...], bool]] = []

    def segment_events(
        self,
        observations: Sequence[EventObservation],
        *,
        end_of_stream: bool = False,
    ) -> tuple[EventSegment, ...]:
        self.calls.append((tuple(observations), end_of_stream))
        reply = next(self._replies)
        if isinstance(reply, Exception):
            raise reply
        # Illegal replies deliberately exercise the public model boundary's validation.
        return cast(tuple[EventSegment, ...], reply)


class CountingEmbedder(TinyEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        return super().embed(inputs, task)


def make_chunk(
    offset: int,
    *,
    stream_id: str = "default",
    paired: bool = False,
    data: bytes | None = None,
) -> EventChunk:
    audio = Blob(data or f"{stream_id}-sound-{offset}".encode(), "audio/wav")
    content = (
        (audio, Blob(f"{stream_id}-frame-{offset}".encode(), "image/png")) if paired else audio
    )
    return EventChunk(
        content,
        REFERENCE_AT + timedelta(seconds=offset),
        REFERENCE_AT + timedelta(seconds=offset + 1),
        stream_id,
    )


async def stream_chunks(*values: EventChunk) -> AsyncIterator[EventChunk]:
    for value in values:
        yield value


class CaptureCountingEmbedder:
    """TinyEmbedder that counts its calls and can refuse one poisoned input."""

    embedding_model = TinyEmbedder.embedding_model
    embedding_space = TinyEmbedder.embedding_space
    embedding_dimension = TinyEmbedder.embedding_dimension

    def __init__(self, *, capabilities: frozenset[Modality] = ATOMIC_MODALITIES) -> None:
        self.embedding_capabilities = capabilities
        self.calls = 0
        self.poison: str | None = None
        self._inner = TinyEmbedder()

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        if self.poison is not None and any(self.poison in value.text for value in inputs):
            raise RuntimeError("simulated embedding failure")
        return self._inner.embed(inputs, task)

    def close(self) -> None:
        self._inner.close()


class CountingFormer:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "capture-test-former"
    formation_space = "capture-test-former:v1"

    def __init__(self) -> None:
        self.calls = 0

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.calls += 1
        return tuple(
            (
                FormationProposal(
                    kind=RecordKind.ENTITY,
                    content="The user is an entity",
                    subject="user",
                    confidence=0.9,
                ),
            )
            for _value in inputs
        )

    def close(self) -> None:
        pass


class RecordingVisionDescriber:
    vision_capabilities = frozenset({Modality.IMAGE})
    vision_model = "test-vision"
    vision_space = "test-vision:caption-v1"

    def __init__(self) -> None:
        self.inputs: list[ModelInput] = []
        self.closed = False

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        self.inputs.extend(inputs)
        return tuple("automatically described red toolbox" for _value in inputs)

    def close(self) -> None:
        self.closed = True


class CountingDescriber:
    """A describer that never repeats itself, so a reused caption is distinguishable from a call."""

    vision_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    vision_model = "fake-describer"

    def __init__(self, prefix: str = "caption") -> None:
        self.vision_space = "fake-describer:caption-v1"
        self.calls = 0
        self.described: list[str] = []
        self.captions: list[str] = []
        self._prefix = prefix

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        batch = tuple(inputs)
        self.calls += 1
        self.described.extend(str(value.assets[0].sha256) for value in batch)
        fresh = tuple(
            f"{self._prefix} bicycle number {len(self.captions) + index}"
            for index in range(len(batch))
        )
        self.captions.extend(fresh)
        return fresh

    def close(self) -> None:
        return None
