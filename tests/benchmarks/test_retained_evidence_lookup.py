"""SQLite microbenchmark: one proof lookup must not traverse unrelated operation history."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from sensemem.infrastructure.local.store._operations import retained_evidence_edges
from sensemem.infrastructure.local.store._schema import create_schema


def test_retained_proof_lookup_has_a_bounded_sqlite_work_budget(tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "proofs.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        create_schema(connection, frozenset())
        count = 10_000
        stamps = [f"2026-11-14T00:00:00.{index:06d}+00:00" for index in range(count)]
        connection.executemany(
            "INSERT INTO memory_records "
            "(memory_id, content, modality, metadata_json, created_at, updated_at, forgotten_at) "
            "VALUES (?, 'original', 'text', '{}', ?, ?, ?)",
            ((f"source{index}", stamp, stamp, stamp) for index, stamp in enumerate(stamps)),
        )
        connection.execute(
            "INSERT INTO memory_records "
            "(memory_id, content, modality, metadata_json, created_at, updated_at) "
            "VALUES ('summary0', 'summary', 'text', '{}', ?, ?)",
            (stamps[0], stamps[0]),
        )
        connection.execute(
            "INSERT INTO memory_semantics (memory_id, lineage_id, kind, basis) "
            "VALUES ('summary0', 'lineage', 'event', 'model_inference')"
        )
        clause_id = "a" * 64
        connection.execute(
            "INSERT INTO memory_evidence_clauses "
            "(memory_id, clause_id, member_count, confidence, recorded_at) VALUES (?, ?, 1, 1, ?)",
            ("summary0", clause_id, stamps[0]),
        )
        connection.execute(
            "INSERT INTO memory_evidence_clause_members "
            "(memory_id, clause_id, source_memory_id, position) VALUES (?, ?, ?, 0)",
            ("summary0", clause_id, "source0"),
        )
        connection.executemany(
            "INSERT INTO memory_operations "
            "(operation_key, intent, trigger, operation_json, effects_json, applied_at) "
            "VALUES (?, 'consolidate', 'manual', '{}', ?, ?)",
            (
                (
                    f"operation{index}",
                    json.dumps(
                        {
                            "created_ids": [f"summary{index}"],
                            "changed_ids": [],
                            "forgotten_ids": [f"source{index}"],
                        }
                    ),
                    stamp,
                )
                for index, stamp in enumerate(stamps)
            ),
        )
        steps = 0

        def budget_exceeded() -> int:
            nonlocal steps
            steps += 1000
            return int(steps > 5000)

        connection.set_progress_handler(budget_exceeded, 1000)
        try:
            expected = frozenset({("summary0", "source0")})
            assert retained_evidence_edges(connection, ("summary0",)) == expected
            assert retained_evidence_edges(connection, source_ids=("source0",)) == expected
        finally:
            connection.set_progress_handler(None, 0)
