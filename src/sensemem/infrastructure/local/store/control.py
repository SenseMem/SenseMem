"""The memory control plane's durable side: candidates, deliberations, operations, rollback."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from itertools import zip_longest

from sensemem.infrastructure.local.store._candidates import (
    consumed_at_by_memory,
    contradiction_candidates,
    deliberated_at_by_memory,
    evidence_candidates,
    feedback_candidates,
    idle_candidates,
    merge_weighed,
    pressure_candidates,
)
from sensemem.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    datetime_text,
    parse_datetime,
    row_text,
)
from sensemem.infrastructure.local.store._connections import Connections
from sensemem.infrastructure.local.store._corroboration import _load_nodes
from sensemem.infrastructure.local.store._identity import reproject_named_identities
from sensemem.infrastructure.local.store._lineage import (
    add_evidence_clause,
    asserted_confidence,
    evidence_clause_changes_are_current,
    require_active_memories,
    require_every,
    require_unretired_memories,
    require_visible_memories,
    response_policy_dependencies,
    restore_memory_versions,
    retire_memory_evidence,
    retire_memory_versions,
    reverse_evidence_clause_changes,
    set_forgotten,
    version_retired,
    withdraw_corrected_evidence,
)
from sensemem.infrastructure.local.store._naming_projection import (
    NamingProjection,
    naming_snapshot,
    refresh_naming_documents,
)
from sensemem.infrastructure.local.store._operations import (
    active_operation_id,
    insert_operation,
    later_operation_depends_on,
    operation_from_row,
    retained_evidence_edges,
)
from sensemem.infrastructure.local.store.errors import ProtectedMemoryError, StaleOperationError
from sensemem.infrastructure.local.store.identities import (
    identity_link_plan,
    merge_identity_rows,
    split_identity_rows,
)
from sensemem.infrastructure.local.store.records import delete_memory, read_memories
from sensemem.infrastructure.local.store.rows import (
    StoredAsset,
    StoredCandidate,
    StoredEvidenceClauseChange,
    StoredOperation,
    StoredQueryFailure,
    require_aware,
    require_identifier,
)


def _replayable_sources(
    connection: sqlite3.Connection, memory_id: str, source_ids: set[str]
) -> set[str]:
    """Keep roots that participate in a complete current proof of this live dependent."""
    nodes, _truncated = _load_nodes(connection, memory_id)
    retained = retained_evidence_edges(connection, tuple(nodes))
    retained_ids = frozenset(source for _parent, source in retained)
    eligible = {
        memory.memory_id
        for memory in read_memories(
            connection, tuple(nodes), active_only=True, retained_ids=retained_ids
        )
    }
    grounded = {
        node_id
        for node_id, node in nodes.items()
        if node_id in eligible
        and node.root_group is not None
        and not node.assessments
        and node.confidence > 0
    }
    origins = {node_id: {node_id} & source_ids for node_id in grounded}
    changed = True
    while changed:
        changed = False
        for parent, node in nodes.items():
            if parent not in eligible:
                continue
            for assessment in node.assessments:
                if not assessment.sources or not all(
                    source in grounded
                    and (source not in retained_ids or (parent, source) in retained)
                    for source in assessment.sources
                ):
                    continue
                inherited = {origin for source in assessment.sources for origin in origins[source]}
                previous = origins.setdefault(parent, set())
                if parent not in grounded or not inherited <= previous:
                    grounded.add(parent)
                    previous.update(inherited)
                    changed = True
    # The bounded loader may omit branches. Only complete proofs found above can contribute;
    # omitted or cyclic-only support therefore cannot authorize replay.
    return origins.get(memory_id, set())


class OperationLog:
    """Candidates, deliberations, the operation log, and its reversal."""

    def __init__(
        self,
        *,
        connections: Connections,
    ) -> None:
        self._connections = connections

    def read_consolidation_candidates(
        self,
        *,
        limit: int,
        idle: bool = False,
        record_budget: int | None = None,
    ) -> tuple[StoredCandidate, ...]:
        """Derive due deliberation work from evidence, lineage, and feedback already recorded.

        There is no queue and no timer behind this: every row is a fact the store already holds,
        read back as a question. Rows are interleaved across triggers so a busy trigger cannot
        starve the others out of the window.

        `record_budget` is the configured ceiling on active records; exceeding it derives
        `PRESSURE` rows. `idle` is the operator declaring an approved window, never the store
        guessing from a clock, and derives `IDLE` rows for lineages nothing has ever weighed.
        """
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if record_budget is not None and record_budget <= 0:
            raise ValueError("record_budget must be positive")
        with self._connections.read_transaction() as connection:
            deliberated = deliberated_at_by_memory(connection)
            weighed = merge_weighed(consumed_at_by_memory(connection), deliberated)
            groups = [
                evidence_candidates(connection, weighed, limit),
                # Deliberations only, not the whole weighed map: the two consolidations that
                # created a pair of disagreeing claims weighed their own sources, not the
                # disagreement between the results, so a fresh contradiction is due even though
                # an operation touched both records a moment ago.
                contradiction_candidates(connection, deliberated, limit),
                feedback_candidates(connection, weighed, limit),
                pressure_candidates(connection, weighed, limit, record_budget),
            ]
            if idle:
                groups.append(idle_candidates(connection, weighed, limit))
        interleaved = [
            candidate for row in zip_longest(*groups) for candidate in row if candidate is not None
        ]
        return tuple(interleaved[:limit])

    def retained_evidence_edges(self, memory_ids: Sequence[str]) -> frozenset[tuple[str, str]]:
        """Sources hidden by consolidation may prove only that consolidation's outputs."""
        with self._connections.read_transaction() as connection:
            return retained_evidence_edges(connection, memory_ids)

    def retained_evidence_sources(self, source_ids: Sequence[str]) -> frozenset[str]:
        with self._connections.read_transaction() as connection:
            return frozenset(
                source
                for _memory, source in retained_evidence_edges(connection, source_ids=source_ids)
            )

    def replayable_evidence_sources(self, source_ids: Sequence[str]) -> frozenset[str]:
        """Retired originals with a live dependent through current positive support clauses.

        Consolidation permission alone does not establish a surviving dependent. Traverse the
        existing clause graph, including legally retained intermediate summaries, and stop at
        current visible knowledge. Explicit withdrawal cannot be crossed by this traversal.
        """
        accepted: set[str] = set()
        pending = {source_id: {source_id} for source_id in source_ids}
        visited: dict[str, set[str]] = {}
        with self._connections.read_transaction() as connection:
            while pending:
                current, pending = pending, {}
                memories = read_memories(connection, tuple(current))
                retained_edges = retained_evidence_edges(connection, source_ids=tuple(current))
                withdrawn = {
                    memory.memory_id for memory in memories if memory.forgotten_at is not None
                }
                retained = {source for _parent, source in retained_edges}
                eligible = tuple(
                    memory.memory_id
                    for memory in memories
                    if memory.forgotten_at is None or memory.memory_id in retained
                )
                active = {
                    memory.memory_id
                    for memory in read_memories(connection, eligible, active_only=True)
                }
                for memory_id in active:
                    accepted.update(_replayable_sources(connection, memory_id, current[memory_id]))
                for offset in range(0, len(eligible), SQLITE_PARAMETER_BATCH):
                    batch = eligible[offset : offset + SQLITE_PARAMETER_BATCH]
                    placeholders = ", ".join("?" for _memory_id in batch)
                    rows = connection.execute(
                        f"""
                        SELECT DISTINCT m.source_memory_id, m.memory_id
                        FROM memory_evidence_clause_members AS m
                        JOIN memory_evidence_clauses AS c
                          ON c.memory_id = m.memory_id AND c.clause_id = m.clause_id
                        WHERE m.source_memory_id IN ({placeholders})
                          AND c.retired_at IS NULL AND c.confidence > 0
                        """,
                        batch,
                    ).fetchall()
                    for row in rows:
                        source, parent = (
                            row_text(row, "source_memory_id"),
                            row_text(row, "memory_id"),
                        )
                        if source in withdrawn and (parent, source) not in retained_edges:
                            continue
                        seen = visited.setdefault(parent, set())
                        origins = current[source] - accepted - seen
                        if origins:
                            seen.update(origins)
                            pending.setdefault(parent, set()).update(origins)
        return frozenset(accepted)

    def record_query_failure(
        self,
        query: str,
        normalized: str,
        *,
        failed_at: datetime,
        keep: int,
        context_json: str,
    ) -> None:
        """Append one empty-recall signal, keeping at most `keep` of them.

        Bounded on write rather than by a sweeper: the table is a signal buffer, so the oldest
        rows falling out is the retention policy, not a loss.
        """
        if not query.strip() or not normalized:
            raise ValueError("a query failure must carry its query text")
        require_aware(failed_at, "failed_at")
        if keep <= 0:
            raise ValueError("keep must be positive")
        with self._connections.transaction() as connection:
            connection.execute(
                "INSERT INTO query_failures (query, normalized, failed_at, context_json) VALUES (?, ?, ?, ?)",
                (query, normalized, datetime_text(failed_at), context_json),
            )
            connection.execute(
                """
                DELETE FROM query_failures WHERE failure_id NOT IN (
                    SELECT failure_id FROM query_failures ORDER BY failure_id DESC LIMIT ?
                )
                """,
                (keep,),
            )

    def read_repeated_query_failures(
        self,
        *,
        limit: int,
        since: datetime,
        minimum: int,
    ) -> tuple[StoredQueryFailure, ...]:
        """Return near-equal queries that failed at least `minimum` times since `since`."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        require_aware(since, "since")
        if minimum < 2:
            raise ValueError("a repeated failure needs at least two failures")
        with self._connections.read_transaction() as connection:
            rows = connection.execute(
                """
                WITH repeated AS (
                    SELECT normalized, COUNT(*) AS failures, MAX(failure_id) AS newest
                    FROM query_failures WHERE failed_at >= ?
                    GROUP BY normalized HAVING failures >= ?
                )
                SELECT f.query, f.normalized, f.failed_at, f.context_json, repeated.failures
                FROM repeated JOIN query_failures AS f ON f.failure_id = repeated.newest
                WHERE NOT EXISTS (
                    SELECT 1 FROM memory_deliberations AS d
                    WHERE d.query_failure_key = f.normalized AND d.weighed_at >= f.failed_at
                )
                ORDER BY f.failed_at DESC, f.normalized
                LIMIT ?
                """,
                (datetime_text(since), minimum, limit),
            ).fetchall()
        return tuple(
            StoredQueryFailure(
                query=row_text(row, "query"),
                normalized=row_text(row, "normalized"),
                failures=int(row["failures"]),
                failed_at=parse_datetime(row_text(row, "failed_at")),
                context_json=row_text(row, "context_json"),
            )
            for row in rows
        )

    def record_deliberation(
        self,
        trigger: str,
        memory_ids: Sequence[str],
        *,
        weighed_at: datetime,
        proposed: int,
        applied: int,
        rejected: int,
        query_failure_key: str | None = None,
    ) -> int:
        """Mark one evidence set weighed, whatever the pass yielded.

        A zero-yield pass records the same row as a productive one. That is the whole point: the
        candidate that produced nothing must stop coming back until its own signal moves.
        """
        require_identifier(trigger, "trigger")
        for memory_id in memory_ids:
            require_identifier(memory_id, "memory_id")
        require_aware(weighed_at, "weighed_at")
        if min(proposed, applied, rejected) < 0:
            raise ValueError("deliberation counts must not be negative")
        with self._connections.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memory_deliberations (
                    trigger, weighed_at, proposed, applied, rejected, query_failure_key
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trigger,
                    datetime_text(weighed_at),
                    proposed,
                    applied,
                    rejected,
                    query_failure_key,
                ),
            )
            deliberation_id = int(cursor.lastrowid or 0)
            connection.executemany(
                """
                INSERT OR IGNORE INTO memory_deliberation_memories (deliberation_id, memory_id)
                VALUES (?, ?)
                """,
                tuple((deliberation_id, memory_id) for memory_id in dict.fromkeys(memory_ids)),
            )
        return deliberation_id

    def record_operation_outcome(
        self,
        operation_id: int,
        *,
        outcome: str,
        note: str | None,
    ) -> bool:
        """Record what later evidence said about one logged operation.

        Reports `False` for an unknown operation. The kernel never reads this back into a
        decision: it exists so consolidation precision, false retirement, and contradiction
        recovery are derivable from the log.
        """
        require_identifier(outcome, "outcome")
        if note is not None and not note.strip():
            raise ValueError("an outcome note must not be blank")
        with self._connections.transaction() as connection:
            cursor = connection.execute(
                "UPDATE memory_operations SET outcome = ?, outcome_note = ? WHERE operation_id = ?",
                (outcome, note, operation_id),
            )
        return cursor.rowcount > 0

    def apply_control_operation(
        self,
        operation: StoredOperation,
        *,
        reinforce: Sequence[tuple[str, str]] = (),
        correct_ids: Sequence[str] = (),
        forget_ids: Sequence[str] = (),
        require_active: Sequence[str] = (),
        require_unretired: Sequence[str] = (),
        require_visible: Sequence[str] = (),
        withdraw_retained: Sequence[str] = (),
        preserve_response_policies: bool = False,
        record_budget: int | None = None,
        naming_projection: NamingProjection | None = None,
    ) -> StoredOperation | None:
        """Apply one already-validated operation and its log row in one transaction.

        The caller owns policy; this only executes the supplied effects. It returns `None` when
        the operation key is already applied and not rolled back.

        `require_active` names memories the caller validated and that must still exist and still
        be un-forgotten here. That check and the all-or-nothing effect check below are what make
        the gap between validation and this transaction safe: a target or source that moved in
        between raises `StaleOperationError` with nothing written, so the log never claims an
        effect that did not happen. `require_unretired` is the stricter half of that check, for
        the callers that need the current version of a record to still stand -- a `REINFORCE`
        target must not have been corrected in between -- which `require_active` must not check
        globally, because the host's own `forget()` may forget an already-corrected record.
        `require_visible` checks cited premises, not reinforcement targets: their latest
        recorded version must remain visible and unretired, and the record un-forgotten.
        """
        for memory_id, source_memory_id in reinforce:
            require_identifier(memory_id, "memory_id")
            require_identifier(source_memory_id, "source_memory_id")
        for memory_id in (*correct_ids, *forget_ids):
            require_identifier(memory_id, "memory_id")
        with self._connections.transaction() as connection:
            if active_operation_id(connection, operation.operation_key) is not None:
                return None
            require_active_memories(connection, require_active)
            retained = {
                source
                for _memory, source in retained_evidence_edges(
                    connection, source_ids=withdraw_retained
                )
            }
            require_every(tuple(retained), withdraw_retained, "withdraw retained evidence")
            require_unretired_memories(connection, require_unretired)
            require_visible_memories(connection, require_visible)
            if preserve_response_policies:
                blocked = response_policy_dependencies(connection).intersection(
                    (*correct_ids, *forget_ids, *(memory_id for memory_id, _source in reinforce))
                )
                if blocked:
                    raise ProtectedMemoryError(blocked)
            if record_budget is not None:
                active = int(
                    connection.execute(
                        "SELECT COUNT(*) AS records FROM memory_records WHERE forgotten_at IS NULL"
                    ).fetchone()["records"]
                )
                if active - len(forget_ids) < record_budget:
                    raise StaleOperationError("forgetting would exceed the host's record budget")
            naming_before = naming_snapshot(
                connection,
                (*(memory_id for memory_id, _source in reinforce), *correct_ids, *forget_ids),
            )
            changed: list[str] = []
            linked: list[tuple[str, str]] = []
            clause_changes: list[StoredEvidenceClauseChange] = []
            for memory_id, source_memory_id in reinforce:
                change = add_evidence_clause(
                    connection,
                    memory_id,
                    (source_memory_id,),
                    confidence=asserted_confidence(connection, memory_id),
                    recorded_at=operation.applied_at,
                )
                if change is None:
                    raise StaleOperationError(f"{source_memory_id} already supports {memory_id}")
                changed.append(memory_id)
                linked.append((memory_id, source_memory_id))
                clause_changes.append(change)
            retired = retire_memory_versions(
                connection, correct_ids, retired_at=operation.applied_at
            )
            require_every(retired, correct_ids, "retire")
            changed.extend(retired)
            clause_changes.extend(
                withdraw_corrected_evidence(
                    connection, correct_ids, changed_at=operation.applied_at
                )
            )
            newly_forgotten = tuple(
                memory_id for memory_id in forget_ids if memory_id not in retained
            )
            forgotten = set_forgotten(
                connection, newly_forgotten, forgotten_at=operation.applied_at
            )
            require_every(forgotten, newly_forgotten, "forget")
            changed.extend(forget_ids)
            reproject_named_identities(
                connection,
                (*changed, *(change.memory_id for change in clause_changes)),
            )
            applied = replace(
                operation,
                changed_ids=tuple(dict.fromkeys(changed)),
                forgotten_ids=forgotten,
                linked=tuple(linked),
                clause_changes=tuple(clause_changes),
            )
            logged = replace(applied, operation_id=insert_operation(connection, applied))
            refresh_naming_documents(connection, naming_before, naming_projection)
            return logged

    def rollback_operation(
        self,
        operation_id: int,
        *,
        rolled_back_at: datetime,
        retire_evidence: Sequence[tuple[str, str]] = (),
        reverse_clause_changes: Sequence[StoredEvidenceClauseChange] = (),
        restore_versions: Sequence[str | tuple[str, int]] = (),
        retire_versions: Sequence[str] = (),
        clear_forgotten: Sequence[str] = (),
        delete_memory_ids: Sequence[str] = (),
        require_in_force: Sequence[str] = (),
        split_identity: str | None = None,
        merge_identities: tuple[str, str] | None = None,
        projection_identity_ids: Sequence[str] = (),
        require_no_later_dependencies: Sequence[str] = (),
        naming_projection: NamingProjection | None = None,
    ) -> tuple[bool, tuple[StoredAsset, ...]]:
        """Apply the caller's reversal and mark one operation rolled back, atomically.

        Returns `(False, ())` when the operation is unknown, already rolled back, or no longer
        reversible, in which case no reversal is applied. Otherwise the second value lists the
        assets that the deleted records were the last to reference; index cleanup follows through
        the durable outbox.

        `retire_versions` names records whose current version this reversal retires without
        deleting the record, which is what retracting a naming assertion is: the record and its
        log row stay readable while `restore_versions` brings back what it displaced.

        `require_in_force` names the records this operation put in force. If a later standing
        operation has since superseded one of them, reversing this one would restore a version
        beside a newer one and delete a record that newer one supersedes, so the reversal is
        refused instead: operations on one lineage reverse newest first.

        `split_identity` reverses a logged cross-modal merge by restoring that alias as its own
        identity, and `merge_identities` reverses a logged split by re-merging `(survivor,
        restored)` back under the survivor. Both are refused -- `(False, ())`, nothing written --
        when the identity graph no longer admits the reversal, which is how identity operations
        on one person also reverse newest first: a later split has already removed the alias a
        merge would need, and an erasure has removed both.

        `require_no_later_dependencies` names deterministic outputs this operation introduced or
        restated. A later standing log row that cites or changes one makes this operation no
        longer independently reversible.
        """
        require_aware(rolled_back_at, "rolled_back_at")
        for memory_id in delete_memory_ids:
            require_identifier(memory_id, "memory_id")
        unreferenced: list[StoredAsset] = []
        with self._connections.transaction() as connection:
            row = connection.execute(
                """
                SELECT applied_at FROM memory_operations
                WHERE operation_id = ? AND rolled_back_at IS NULL
                """,
                (operation_id,),
            ).fetchone()
            if row is None:
                return False, ()
            if later_operation_depends_on(
                connection,
                operation_id,
                require_no_later_dependencies,
            ):
                return False, ()
            if any(version_retired(connection, memory_id) for memory_id in require_in_force):
                return False, ()
            if not evidence_clause_changes_are_current(connection, reverse_clause_changes):
                return False, ()
            naming_before = naming_snapshot(
                connection,
                (
                    *delete_memory_ids,
                    *(memory_id for memory_id, _source in retire_evidence),
                    *(change.memory_id for change in reverse_clause_changes),
                    *retire_versions,
                    *(entry if isinstance(entry, str) else entry[0] for entry in restore_versions),
                    *clear_forgotten,
                ),
                identity_ids=projection_identity_ids,
                force=bool(projection_identity_ids),
            )
            # Both identity reversals refuse before they write, so a refusal here leaves the
            # transaction with nothing to undo. The merge plan is checked rather than trusted:
            # re-merging under the wrong survivor would silently rename a person.
            if merge_identities is not None:
                survivor, restored = merge_identities
                plan = identity_link_plan(
                    connection,
                    survivor,
                    restored,
                    allow_rejected_pair=True,
                    preferred_identity=survivor,
                )
                if plan is None or plan.target_id != survivor:
                    return False, ()
                merge_identity_rows(connection, plan)
            restored_id = (
                None if split_identity is None else split_identity_rows(connection, split_identity)
            )
            if split_identity is not None and restored_id is None:
                return False, ()
            reverted_at = max(rolled_back_at, parse_datetime(row_text(row, "applied_at")))
            for memory_id in dict.fromkeys(delete_memory_ids):
                _deleted, orphaned = delete_memory(connection, memory_id)
                unreferenced.extend(orphaned)
            retire_memory_evidence(connection, retire_evidence, retired_at=reverted_at)
            reverse_evidence_clause_changes(
                connection,
                reverse_clause_changes,
                reversed_at=reverted_at,
            )
            # Retired before the restore, so an assertion and the one it displaced are never both
            # in force: a naming assertion is retracted rather than deleted, because the log row
            # that recorded it stays readable and the audit trail must show both names.
            retire_memory_versions(connection, retire_versions, retired_at=reverted_at)
            withdrawn = withdraw_corrected_evidence(
                connection, retire_versions, changed_at=reverted_at
            )
            restore_memory_versions(connection, restore_versions, recorded_at=reverted_at)
            set_forgotten(connection, clear_forgotten, forgotten_at=None)
            reproject_named_identities(
                connection,
                (
                    *(memory_id for memory_id, _source in retire_evidence),
                    *(change.memory_id for change in reverse_clause_changes),
                    *(change.memory_id for change in withdrawn),
                    *retire_versions,
                    *(entry if isinstance(entry, str) else entry[0] for entry in restore_versions),
                    *clear_forgotten,
                ),
            )
            connection.execute(
                "UPDATE memory_operations SET rolled_back_at = ? WHERE operation_id = ?",
                (datetime_text(reverted_at), operation_id),
            )
            refresh_naming_documents(
                connection,
                naming_before,
                naming_projection,
                force_ids=projection_identity_ids,
            )
        return True, tuple({asset.asset_id: asset for asset in unreferenced}.values())

    def read_operations(
        self,
        *,
        limit: int = 100,
        operation_id: int | None = None,
        operation_key: str | None = None,
        before_operation_id: int | None = None,
    ) -> tuple[StoredOperation, ...]:
        """Return logged operations newest first, or the one matching an id or active key.

        `before_operation_id` continues a newest-first scan, which is what lets a caller that
        must see the whole log -- a data-subject export -- page it instead of truncating it.
        """
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        where = ""
        parameters: tuple[object, ...] = ()
        if operation_id is not None:
            where = "WHERE operation_id = ?"
            parameters = (operation_id,)
        elif operation_key is not None:
            require_identifier(operation_key, "operation_key")
            where = "WHERE operation_key = ? AND rolled_back_at IS NULL"
            parameters = (operation_key,)
        elif before_operation_id is not None:
            where = "WHERE operation_id < ?"
            parameters = (before_operation_id,)
        with self._connections.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT operation_id, operation_key, intent, trigger, model_id, recipe,
                       operation_json, effects_json, applied_at, rolled_back_at,
                       outcome, outcome_note
                FROM memory_operations
                {where}
                ORDER BY operation_id DESC
                LIMIT ?
                """,
                (*parameters, limit),
            ).fetchall()
        return tuple(operation_from_row(row) for row in rows)

    def naming_operation_key(self, operation_key: str, memory_id: str) -> str | None:
        """Return a log key for a naming attempt, or None for a standing duplicate.

        A superseded deterministic assertion still has an active historical operation with the
        base key. Restating that retired assertion is a new auditable operation, keyed by the
        version it is about to create; repeating a currently standing assertion remains a no-op.
        """
        require_identifier(operation_key, "operation_key")
        require_identifier(memory_id, "memory_id")
        with self._connections.connection() as connection:
            if active_operation_id(connection, operation_key) is None:
                return operation_key
            if not version_retired(connection, memory_id):
                return None
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM memory_versions WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("failed to read naming assertion version")
        payload = f"sensemem-operation-restatement-v1:{operation_key}:{int(row['version']) + 1}"
        return hashlib.sha256(payload.encode()).hexdigest()
