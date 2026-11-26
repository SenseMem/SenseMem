"""The append-only memory-operation log."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from typing import TypeVar

from sensemem.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    datetime_text,
    optional_datetime_from_row,
    optional_row_text,
    parse_datetime,
    row_text,
)
from sensemem.infrastructure.local.store.errors import StoredDataError
from sensemem.infrastructure.local.store.rows import (
    StoredEvidenceClauseChange,
    StoredOperation,
    require_identifier,
)

_EffectValue = TypeVar("_EffectValue", str, int, bool)


def _effect_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("logged operation effect IDs must encode an array")
    return tuple(_effect_value(item, str) for item in value)


def active_operation_id(connection: sqlite3.Connection, operation_key: str) -> int | None:
    row = connection.execute(
        """
        SELECT operation_id FROM memory_operations
        WHERE operation_key = ? AND rolled_back_at IS NULL
        """,
        (operation_key,),
    ).fetchone()
    return None if row is None else int(row["operation_id"])


def retained_evidence_edges(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str] = (),
    *,
    source_ids: Sequence[str] = (),
) -> frozenset[tuple[str, str]]:
    """Return consolidation-only source permissions that no explicit forgetting withdrew."""
    if not memory_ids and not source_ids:
        return frozenset()
    axis = "member.memory_id" if memory_ids else "member.source_memory_id"
    requested = tuple(dict.fromkeys(memory_ids or source_ids))
    edges: set[tuple[str, str]] = set()
    for offset in range(0, len(requested), SQLITE_PARAMETER_BATCH):
        batch = requested[offset : offset + SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        # Fix the join order and timestamp index: starting at operation history makes every
        # evidence hop scan unrelated logs and repeatedly evaluate the withdrawal subquery.
        rows = connection.execute(
            f"""
            SELECT DISTINCT member.memory_id, member.source_memory_id
            FROM memory_evidence_clause_members AS member
            CROSS JOIN memory_records AS r
              ON r.memory_id = member.source_memory_id
            CROSS JOIN memory_operations AS o INDEXED BY memory_operations_retained_sources_idx
              ON o.applied_at = r.forgotten_at
             AND o.intent = 'consolidate' AND o.rolled_back_at IS NULL
            WHERE {axis} IN ({placeholders})
              AND EXISTS (
                SELECT 1 FROM json_each(o.effects_json, '$.forgotten_ids') AS source
                WHERE source.value = member.source_memory_id
              )
              AND EXISTS (
                SELECT 1 FROM json_tree(json_array(
                    json_extract(o.effects_json, '$.created_ids'),
                    json_extract(o.effects_json, '$.changed_ids')
                )) AS derived
                WHERE derived.type = 'text' AND derived.value = member.memory_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM memory_operations AS withdrawal
                JOIN json_each(withdrawal.operation_json, '$.target_ids') AS target
                  ON target.value = member.source_memory_id
                WHERE withdrawal.intent = 'forget' AND withdrawal.rolled_back_at IS NULL
                  AND withdrawal.operation_id > o.operation_id
              )
            """,
            batch,
        ).fetchall()
        edges.update(
            (row_text(row, "memory_id"), row_text(row, "source_memory_id")) for row in rows
        )
    return frozenset(edges)


def later_operation_depends_on(
    connection: sqlite3.Connection,
    operation_id: int,
    memory_ids: Sequence[str],
) -> bool:
    """Return whether a later standing operation names one of these outputs."""
    ids = tuple(dict.fromkeys(memory_ids))
    for memory_id in ids:
        require_identifier(memory_id, "memory_id")
    for offset in range(0, len(ids), SQLITE_PARAMETER_BATCH - 1):
        batch = ids[offset : offset + SQLITE_PARAMETER_BATCH - 1]
        placeholders = ", ".join("?" for _memory_id in batch)
        if (
            connection.execute(
                f"""
            SELECT 1
            FROM memory_operations AS o
            JOIN json_tree(json_array(
                json_extract(o.operation_json, '$.evidence_ids'),
                json_extract(o.operation_json, '$.target_ids'),
                json_extract(o.effects_json, '$.created_ids'),
                json_extract(o.effects_json, '$.changed_ids'),
                json_extract(o.effects_json, '$.activated_ids'),
                json_extract(o.effects_json, '$.linked'),
                json_extract(o.effects_json, '$.superseded')
            )) AS j
            WHERE o.operation_id > ? AND o.rolled_back_at IS NULL
              AND j.type = 'text' AND j.value IN ({placeholders})
            LIMIT 1
            """,
                (operation_id, *batch),
            ).fetchone()
            is not None
        ):
            return True
    return False


def insert_operation(connection: sqlite3.Connection, operation: StoredOperation) -> int:
    cursor = connection.execute(
        """
        INSERT INTO memory_operations (
            operation_key, intent, trigger, model_id, recipe,
            operation_json, effects_json, applied_at, rolled_back_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            operation.operation_key,
            operation.intent,
            operation.trigger,
            operation.model_id,
            operation.recipe,
            operation.operation_json,
            json.dumps(
                {
                    "created_ids": list(operation.created_ids),
                    "changed_ids": list(operation.changed_ids),
                    "activated_ids": list(operation.activated_ids),
                    "forgotten_ids": list(operation.forgotten_ids),
                    "linked": [list(pair) for pair in operation.linked],
                    "clause_changes": [
                        {
                            "memory_id": change.memory_id,
                            "clause_id": change.clause_id,
                            "previous_active": change.previous_active,
                            "previous_confidence": change.previous_confidence,
                            "applied_confidence": change.applied_confidence,
                            "applied_recorded_at": datetime_text(change.applied_recorded_at),
                            "applied_version": change.applied_version,
                        }
                        for change in operation.clause_changes
                    ],
                    "superseded": [list(pair) for pair in operation.superseded],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            datetime_text(operation.applied_at),
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("failed to log a memory operation")
    return int(cursor.lastrowid)


def operation_from_row(row: sqlite3.Row) -> StoredOperation:
    try:
        effects = json.loads(row_text(row, "effects_json"))
        if not isinstance(effects, dict):
            raise ValueError("logged operation effects must encode an object")
        return StoredOperation(
            operation_id=int(row["operation_id"]),
            operation_key=row_text(row, "operation_key"),
            intent=row_text(row, "intent"),
            trigger=row_text(row, "trigger"),
            model_id=optional_row_text(row, "model_id"),
            recipe=optional_row_text(row, "recipe"),
            operation_json=row_text(row, "operation_json"),
            created_ids=_effect_ids(effects["created_ids"]),
            changed_ids=_effect_ids(effects["changed_ids"]),
            activated_ids=_effect_ids(effects["activated_ids"]),
            forgotten_ids=_effect_ids(effects["forgotten_ids"]),
            linked=_effect_pairs(effects["linked"], str),
            clause_changes=tuple(
                StoredEvidenceClauseChange(
                    memory_id=change["memory_id"],
                    clause_id=change["clause_id"],
                    previous_active=_effect_value(change["previous_active"], bool),
                    previous_confidence=change["previous_confidence"],
                    applied_confidence=change["applied_confidence"],
                    applied_recorded_at=parse_datetime(change["applied_recorded_at"]),
                    applied_version=_effect_value(change["applied_version"], int),
                )
                for change in effects["clause_changes"]
            ),
            superseded=_effect_pairs(effects["superseded"], int),
            applied_at=parse_datetime(row_text(row, "applied_at")),
            rolled_back_at=optional_datetime_from_row(row, "rolled_back_at"),
            outcome=optional_row_text(row, "outcome"),
            outcome_note=optional_row_text(row, "outcome_note"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise StoredDataError("stored memory operation is invalid") from error


def _effect_pairs(
    value: object, value_type: type[_EffectValue]
) -> tuple[tuple[str, _EffectValue], ...]:
    if not isinstance(value, list):
        raise ValueError("logged operation effect pairs must encode an array")
    pairs = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("logged operation effect pair must encode a two-element array")
        pairs.append((_effect_value(pair[0], str), _effect_value(pair[1], value_type)))
    return tuple(pairs)


def _effect_value(value: object, value_type: type[_EffectValue]) -> _EffectValue:
    if type(value) is not value_type:
        raise ValueError(f"logged operation effects must encode {value_type.__name__} values")
    return value
