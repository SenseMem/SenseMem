"""Streaming content parts, commit acknowledgements, and prefetch revision contracts."""

from datetime import datetime, timezone
from typing import cast

import pytest
from pydantic import TypeAdapter

from sensemem import (
    AsyncMemory,
    AsyncOmniPrefetch,
    ContentPart,
    ContextBudget,
    MemoryCategory,
    MemoryRecord,
    PrefetchResult,
    PrefetchUpdate,
    StreamCommit,
    StreamInput,
    ValidationError,
)


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


@pytest.mark.parametrize("pending_settlement", [False, True])
@pytest.mark.parametrize("has_prefetch", [False, True])
@pytest.mark.parametrize("retrieval_error", [None, "retrieval_failed"])
def test_stream_commit_retrieval_contract(
    pending_settlement: bool, has_prefetch: bool, retrieval_error: str | None
) -> None:
    record = MemoryRecord(
        id="captured", content="observation", created_at=datetime(2026, 10, 8, tzinfo=timezone.utc)
    )
    prefetch = PrefetchResult(hits=()) if has_prefetch else None
    valid = has_prefetch != (retrieval_error is not None) or (
        pending_settlement and not has_prefetch and retrieval_error is None
    )
    if valid:
        commit = StreamCommit(
            record, prefetch, retrieval_error, pending_settlement=pending_settlement
        )
        assert commit.record == record
        assert commit.prefetch == prefetch
        assert commit.retrieval_error == retrieval_error
        assert commit.pending_settlement is pending_settlement
    else:
        with pytest.raises(ValidationError, match="prefetch or retrieval_error"):
            StreamCommit(record, prefetch, retrieval_error, pending_settlement=pending_settlement)


def test_prefetch_updates_carry_valid_request_local_revisions() -> None:
    result = PrefetchResult(hits=())
    assert result.bundle is None
    assert PrefetchUpdate(1, result).result is result
    for invalid in (0, -1, True):
        with pytest.raises(ValidationError):
            PrefetchUpdate(invalid, result)


def test_compiled_prefetch_does_not_silently_ignore_search_filters() -> None:
    with pytest.raises(ValidationError, match="budget filters"):
        AsyncOmniPrefetch(
            cast(AsyncMemory, object()),
            budget=ContextBudget(),
            memory_type=MemoryCategory.EPISODIC,
        )
