"""Connection-level admission to the durable deferred-capture settlement queue."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from sensemem.infrastructure.local.store._codec import datetime_text
from sensemem.infrastructure.local.store.errors import LocalStoreCapacityError

MAX_PENDING_CAPTURES = 10_000


def enqueue_capture(connection: sqlite3.Connection, memory_id: str, enqueued_at: datetime) -> None:
    """Queue a captured memory for settlement; one already waiting keeps its earlier slot."""
    if (
        connection.execute(
            "SELECT 1 FROM capture_queue WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        is not None
    ):
        return
    if (
        connection.execute("SELECT count(*) FROM capture_queue").fetchone()[0]
        >= MAX_PENDING_CAPTURES
    ):
        raise LocalStoreCapacityError(
            "pending capture capacity exceeded; settle existing captures before retrying"
        )
    connection.execute(
        """
        INSERT INTO capture_queue (memory_id, enqueued_at) VALUES (?, ?)
        ON CONFLICT (memory_id) DO NOTHING
        """,
        (memory_id, datetime_text(enqueued_at)),
    )
