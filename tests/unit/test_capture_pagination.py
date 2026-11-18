"""Failed captures cannot hide later backlog entries, including after a cursor row disappears."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from sensemem import EmbedTask, Memory, ModelError, ModelInput


class _FailedEmbedder(TinyEmbedder):
    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        raise ModelError("fixture model failure", reason="model_failed")


def test_backlog_pages_advance_past_failed_records_and_a_deleted_cursor(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_FailedEmbedder()) as memory:
        originals = {memory.capture(f"observation {index}").id for index in range(105)}
        with pytest.raises(ModelError):
            memory.settle(limit=100, max_attempts=1)
        first = memory.pending_captures()
        assert len(first) == 100
        assert all(record.attempts == 1 for record in first)
        cursor = first[-1].cursor
        memory.delete(first[-1].memory_id)
        last = memory.pending_captures(cursor=cursor)
        assert len(last) == 5
        assert {record.memory_id for record in (*first, *last)} == originals
        assert memory.pending_captures(cursor=last[-1].cursor) == ()
