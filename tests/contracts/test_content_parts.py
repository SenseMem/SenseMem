"""Ordered content has a closed public shape independent of local asset paths."""

import pytest
from pydantic import TypeAdapter

from sensemem import ContentPart, StreamInput, ValidationError


def test_ordered_parts_round_trip_without_flattening_their_positions() -> None:
    adapter = TypeAdapter(tuple[ContentPart, ...])
    parts = (
        ContentPart("text", "before"),
        ContentPart("asset", "image-id"),
        ContentPart("model_interpretation", "A lossy model summary"),
    )
    assert adapter.validate_json(adapter.dump_json(parts)) == parts


def test_content_parts_reject_an_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        ContentPart("video", "image-id")  # type: ignore[arg-type]


def test_stream_interpretation_has_an_explicit_validated_contract() -> None:
    adapter = TypeAdapter(StreamInput)
    item = StreamInput("Original words", interpretation="A model's interpretation")
    assert adapter.validate_json(adapter.dump_json(item)) == item
    with pytest.raises(ValidationError, match="stream interpretation"):
        StreamInput("Original words", interpretation=" ")
