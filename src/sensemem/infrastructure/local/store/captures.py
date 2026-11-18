"""The deferred-capture queue."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

from sensemem.infrastructure.local.store._capture_queue import enqueue_capture
from sensemem.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    datetime_text,
    parse_datetime,
    prepare_write_batch,
    row_text,
)
from sensemem.infrastructure.local.store._connections import Connections
from sensemem.infrastructure.local.store._embeddings import write_embeddings
from sensemem.infrastructure.local.store.media import commit_speech_plans
from sensemem.infrastructure.local.store.records import write_memory
from sensemem.infrastructure.local.store.rows import (
    SpeechPlan,
    StoredEmbedding,
    StoredMemory,
    require_aware,
    require_identifier,
)
from sensemem.records import PendingCapture


class CaptureQueue:
    """Rows that owe a later `settle()`."""

    def __init__(
        self,
        *,
        connections: Connections,
    ) -> None:
        self._connections = connections

    def write_captures(
        self,
        memories: Iterable[StoredMemory],
        *,
        enqueued_at: datetime,
        replace_existing: bool = False,
    ) -> tuple[str, ...]:
        """Commit new captured records, their media, their context, and their queue rows.

        Returns the IDs this call enqueued. Ordinary duplicate captures remain unchanged.
        An explicit derivation refresh replaces content, withdraws old vectors, and re-enqueues
        settlement in the same transaction.
        """
        supplied_memories, _embeddings, supplied_by_memory = prepare_write_batch(memories, ())
        if not supplied_memories:
            return ()
        require_aware(enqueued_at, "enqueued_at")
        enqueued: list[str] = []
        with self._connections.transaction() as connection:
            transaction_memory_ids: set[str] = set()
            for memory in supplied_memories:
                if (
                    connection.execute(
                        "SELECT 1 FROM memory_records WHERE memory_id = ?",
                        (memory.memory_id,),
                    ).fetchone()
                    is not None
                ):
                    if not replace_existing:
                        continue
                    connection.execute(
                        "DELETE FROM embeddings WHERE memory_id = ?", (memory.memory_id,)
                    )
                    connection.execute(
                        "DELETE FROM capture_queue WHERE memory_id = ?", (memory.memory_id,)
                    )
                write_memory(
                    connection,
                    memory,
                    supplied_embedding_ids=supplied_by_memory[memory.memory_id],
                    transaction_memory_ids=transaction_memory_ids,
                )
                transaction_memory_ids.add(memory.memory_id)
                enqueue_capture(connection, memory.memory_id, enqueued_at)
                enqueued.append(memory.memory_id)
        return tuple(enqueued)

    def settle_capture(
        self,
        memory: StoredMemory,
        embeddings: Iterable[StoredEmbedding] = (),
        *,
        enqueued_at: datetime,
        speech_plans: Sequence[SpeechPlan] = (),
    ) -> bool:
        """Store one captured record's derived content and vectors while it stays queued.

        Returns false when this queue version was removed or refreshed, so an older settlement
        cannot overwrite newer captured content. The queue row is removed once every deferred stage,
        formation included, has succeeded; a retry after a later failure re-runs this write, which
        is an idempotent upsert of the same derived content and vectors.
        """
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            (memory,),
            embeddings,
        )
        require_aware(enqueued_at, "enqueued_at")
        with self._connections.transaction() as connection:
            queued = connection.execute(
                "SELECT 1 FROM capture_queue WHERE memory_id = ? AND enqueued_at = ?",
                (memory.memory_id, datetime_text(enqueued_at)),
            ).fetchone()
            if queued is None:
                return False
            write_memory(
                connection,
                supplied_memories[0],
                supplied_embedding_ids=supplied_by_memory[memory.memory_id],
            )
            commit_speech_plans(connection, speech_plans)
            write_embeddings(connection, supplied_embeddings)
        return True

    def complete_captures(self, memory_ids: Sequence[str]) -> int:
        """Remove captures from the queue after every deferred stage succeeded.

        Retention also uses it, to abandon captures whose repeated failures have aged out: the
        queue row is the promise to keep retrying, and that is all it drops.
        """
        selected = tuple(dict.fromkeys(memory_ids))
        for memory_id in selected:
            require_identifier(memory_id, "memory_id")
        if not selected:
            return 0
        removed = 0
        with self._connections.transaction() as connection:
            for offset in range(0, len(selected), SQLITE_PARAMETER_BATCH):
                batch = selected[offset : offset + SQLITE_PARAMETER_BATCH]
                removed += connection.execute(
                    f"DELETE FROM capture_queue WHERE memory_id IN "
                    f"({', '.join('?' for _value in batch)})",
                    batch,
                ).rowcount
        return removed

    def complete_settlements(
        self,
        captures: Mapping[str, datetime],
        *,
        through: bool = False,
    ) -> tuple[str, ...]:
        """Complete only the queue versions the caller processed, returning their IDs.

        A blocking add uses its record's update time as an inclusive boundary: it may finish
        an older formation debt, but must leave a later capture refresh queued.
        """
        for memory_id, enqueued_at in captures.items():
            require_identifier(memory_id, "memory_id")
            require_aware(enqueued_at, "enqueued_at")
        if not captures:
            return ()
        completed: list[str] = []
        comparison = "<=" if through else "="
        with self._connections.transaction() as connection:
            for memory_id, enqueued_at in captures.items():
                if connection.execute(
                    f"DELETE FROM capture_queue WHERE memory_id = ? AND enqueued_at {comparison} ?",
                    (memory_id, datetime_text(enqueued_at)),
                ).rowcount:
                    completed.append(memory_id)
        return tuple(completed)

    def record_capture_failure(
        self,
        memory_id: str,
        error: str,
        *,
        enqueued_at: datetime | None = None,
    ) -> bool:
        """Count one failed settlement and store its reason, leaving the row queued."""
        require_identifier(memory_id, "memory_id")
        if enqueued_at is not None:
            require_aware(enqueued_at, "enqueued_at")
        version = "" if enqueued_at is None else "AND enqueued_at = ?"
        with self._connections.transaction() as connection:
            updated = connection.execute(
                f"""
                UPDATE capture_queue
                SET attempts = attempts + 1, last_error = ?
                WHERE memory_id = ? {version}
                """,
                (
                    error.strip() or None,
                    memory_id,
                    *(() if enqueued_at is None else (datetime_text(enqueued_at),)),
                ),
            ).rowcount
        return bool(updated)

    def failed_capture_ids(
        self,
        *,
        enqueued_before: datetime,
        limit: int = 100,
    ) -> tuple[str, ...]:
        """Return aged failed captures, filtering before paging so fresh work cannot starve them."""
        require_aware(enqueued_before, "enqueued_before")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._connections.connection() as connection:
            rows = connection.execute(
                """
                SELECT memory_id FROM capture_queue
                WHERE attempts > 0 AND enqueued_at < ?
                ORDER BY enqueued_at, memory_id
                LIMIT ?
                """,
                (datetime_text(enqueued_before), limit),
            ).fetchall()
        return tuple(row_text(row, "memory_id") for row in rows)

    def pending_captures(
        self,
        *,
        limit: int = 100,
        memory_ids: Sequence[str] | None = None,
        max_attempts: int | None = None,
        after: tuple[datetime, str] | None = None,
    ) -> tuple[PendingCapture, ...]:
        """Return queued captures in enqueue order, optionally restricted or attempt-capped.

        `max_attempts` excludes rows that already failed that many times. They stay queued and
        stay visible, so one poisoned record cannot block every record enqueued after it.

        `awaiting` is read from the vectors themselves: a row whose part-0 embedding exists is
        already searchable and owes formation only, which is the same test settlement applies
        before deciding whether to re-run the model stages.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if max_attempts is not None and (
            isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if memory_ids is not None:
            if not memory_ids:
                return ()
            for memory_id in memory_ids:
                require_identifier(memory_id, "memory_id")
        selected = None if memory_ids is None else tuple(dict.fromkeys(memory_ids))
        batches: list[tuple[str, ...] | None] = (
            [None]
            if selected is None
            else [
                selected[offset : offset + SQLITE_PARAMETER_BATCH]
                for offset in range(0, len(selected), SQLITE_PARAMETER_BATCH)
            ]
        )
        queued: list[PendingCapture] = []
        with self._connections.read_transaction() as connection:
            for batch in batches:
                clauses = (
                    []
                    if batch is None
                    else [f"memory_id IN ({', '.join('?' for _value in batch)})"]
                )
                if max_attempts is not None:
                    clauses.append("attempts < ?")
                if after is not None:
                    clauses.append("(enqueued_at > ? OR (enqueued_at = ? AND memory_id > ?))")
                restriction = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                queued.extend(
                    PendingCapture(
                        memory_id=row_text(row, "memory_id"),
                        enqueued_at=parse_datetime(row_text(row, "enqueued_at")),
                        attempts=int(row["attempts"]),
                        last_error=row["last_error"],
                        awaiting="formation" if row["embedded"] else "enrichment",
                    )
                    for row in connection.execute(
                        f"""
                        SELECT memory_id, enqueued_at, attempts, last_error,
                               EXISTS (
                                   SELECT 1 FROM embeddings
                                   WHERE embeddings.memory_id = capture_queue.memory_id
                                     AND embeddings.object_part = 0
                               ) AS embedded
                        FROM capture_queue
                        {restriction}
                        ORDER BY enqueued_at, memory_id
                        LIMIT ?
                        """,
                        (
                            *(batch or ()),
                            *(() if max_attempts is None else (max_attempts,)),
                            *(
                                ()
                                if after is None
                                else (datetime_text(after[0]), datetime_text(after[0]), after[1])
                            ),
                            limit,
                        ),
                    ).fetchall()
                )
        # Each chunk is only oldest-first within itself, so the chunks are merged rather than
        # concatenated: stopping at the first `limit` rows would starve ids past the 900th
        # forever, however long they had been queued.
        queued.sort(key=lambda row: (row.enqueued_at, row.memory_id))
        return tuple(queued[:limit])
