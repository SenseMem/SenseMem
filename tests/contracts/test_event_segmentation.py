"""Public event windows and model proposals have strict immutable value contracts."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from inspect import Parameter, isasyncgenfunction, iscoroutinefunction, signature
from typing import cast

import pytest

from sensemem import (
    AsyncEventStream,
    Blob,
    EventChunk,
    EventObservation,
    EventSegment,
    ModelInput,
    ObservationContext,
    ValidationError,
)

REFERENCE_AT = datetime(2026, 11, 14, 8, tzinfo=timezone.utc)
END = REFERENCE_AT + timedelta(seconds=1)
AUDIO = Blob(b"audio", "audio/wav")
IMAGE = Blob(b"frame", "image/png")
VIDEO = Blob(b"clip", "video/mp4")


def test_durable_event_stream_entry_points_have_explicit_async_contracts() -> None:
    assert iscoroutinefunction(AsyncEventStream.append)
    assert tuple(signature(AsyncEventStream.append).parameters) == ("self", "chunk")
    assert isasyncgenfunction(AsyncEventStream.process)
    processing = signature(AsyncEventStream.process).parameters
    assert tuple(processing) == ("self", "end_of_stream")
    assert processing["end_of_stream"].kind is Parameter.KEYWORD_ONLY
    assert processing["end_of_stream"].default is False
    pending_budget = signature(AsyncEventStream).parameters["max_pending_bytes"]
    assert pending_budget.kind is Parameter.KEYWORD_ONLY
    assert pending_budget.default == 64 * 1024 * 1024


@pytest.mark.parametrize("content", [AUDIO, IMAGE, VIDEO, (AUDIO, IMAGE), (AUDIO, VIDEO)])
def test_event_chunk_preserves_native_media_and_capture_interval(
    content: Blob | tuple[Blob, ...],
) -> None:
    chunk = EventChunk(content, REFERENCE_AT, END, " camera ")
    assert chunk.media == ((content,) if isinstance(content, Blob) else content)
    assert chunk.occurred_at == REFERENCE_AT
    assert chunk.occurred_end == END
    assert chunk.stream_id == "camera"
    with pytest.raises(FrozenInstanceError):
        chunk.stream_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "content",
    [
        (),
        [AUDIO],
        "transcript alone",
        (AUDIO, AUDIO),
        (IMAGE, VIDEO),
        (AUDIO, IMAGE, VIDEO),
        (AUDIO, "invalid"),
    ],
)
def test_event_chunk_rejects_missing_or_ambiguous_media(content: object) -> None:
    with pytest.raises(ValidationError):
        EventChunk(cast(Blob | tuple[Blob, ...], content), REFERENCE_AT, END)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (None, END),
        (REFERENCE_AT, None),
        (REFERENCE_AT.replace(tzinfo=None), END),
        (REFERENCE_AT, END.replace(tzinfo=None)),
        (REFERENCE_AT, REFERENCE_AT),
        (END, REFERENCE_AT),
    ],
)
def test_event_chunk_requires_a_positive_aware_capture_interval(
    start: datetime | None, end: datetime | None
) -> None:
    with pytest.raises(ValidationError):
        EventChunk(AUDIO, cast(datetime, start), cast(datetime, end))


@pytest.mark.parametrize("stream_id", ["", "   ", None, 4])
def test_event_chunk_rejects_an_invalid_stream_id(stream_id: object) -> None:
    with pytest.raises(ValidationError):
        EventChunk(AUDIO, REFERENCE_AT, END, cast(str, stream_id))


@pytest.mark.parametrize("end_chunk", [0, -1, True, 1.5, "1", None])
def test_event_segment_requires_a_positive_integer_boundary(end_chunk: object) -> None:
    with pytest.raises(ValidationError):
        EventSegment(cast(int, end_chunk), "An event finished")


@pytest.mark.parametrize("summary", ["", "   ", None, 4, "x" * 16_385])
def test_event_segment_requires_a_bounded_nonempty_summary(summary: object) -> None:
    with pytest.raises(ValidationError):
        EventSegment(1, cast(str, summary))


def test_event_segment_normalizes_summary_and_is_immutable() -> None:
    proposal = EventSegment(2, "  Tea is served  ")
    assert proposal.end_chunk == 2
    assert proposal.summary == "Tea is served"
    with pytest.raises(FrozenInstanceError):
        proposal.end_chunk = 3  # type: ignore[misc]


def test_event_chunk_keeps_context_and_incomplete_summary_does_not_establish_finality() -> None:
    context = ObservationContext(place_id="kitchen")
    chunk = EventChunk(AUDIO, REFERENCE_AT, END, context=context)
    assert chunk.context is context
    proposal = EventSegment(1, "The repair remains unfinished", complete=False)
    assert proposal.complete is False
    with pytest.raises(ValidationError, match="boolean"):
        EventSegment(1, "Still working", complete=cast(bool, 1))
    with pytest.raises(ValidationError, match="context"):
        EventChunk(AUDIO, REFERENCE_AT, END, context=cast(ObservationContext, "kitchen"))


def test_event_observation_keeps_the_source_window_and_native_model_input() -> None:
    content = ModelInput(text="Source-derived description")
    observation = EventObservation(content, REFERENCE_AT, END)
    assert observation.content is content
    assert (observation.occurred_at, observation.occurred_end) == (REFERENCE_AT, END)


@pytest.mark.parametrize(
    ("content", "start", "end"),
    [
        ("not model input", REFERENCE_AT, END),
        (ModelInput(text="evidence"), None, END),
        (ModelInput(text="evidence"), REFERENCE_AT, None),
        (ModelInput(text="evidence"), REFERENCE_AT.replace(tzinfo=None), END),
        (ModelInput(text="evidence"), REFERENCE_AT, REFERENCE_AT),
    ],
)
def test_event_observation_rejects_invalid_model_input_or_capture_interval(
    content: object, start: datetime | None, end: datetime | None
) -> None:
    with pytest.raises(ValidationError):
        EventObservation(cast(ModelInput, content), cast(datetime, start), cast(datetime, end))
