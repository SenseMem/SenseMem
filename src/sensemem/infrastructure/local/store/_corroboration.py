"""Bounded SQLite ancestry reads for the independent evidence projection."""

from __future__ import annotations

import logging
import sqlite3
from collections import deque
from dataclasses import replace

from sensemem.infrastructure.local.store._codec import optional_row_text, row_text
from sensemem.kernel.corroboration import (
    Assessment,
    EvidenceNode,
    SupportSummary,
    independent_support,
)

_MAX_NODES = 256
_MAX_MEMBER_ROWS = 4096
LOGGER = logging.getLogger(__name__)


def independent_evidence_enabled(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT value FROM store_metadata WHERE key = 'evidence.projection_recipe'"
    ).fetchone()
    return row is not None and row_text(row, "value") == "independent-v2"


def evidence_support(
    connection: sqlite3.Connection,
    memory_id: str,
    *,
    pending_assessment: Assessment | None = None,
) -> SupportSummary:
    """Load at most 256 records and 4096 clause members, then compute a support lower bound.

    Missing and omitted nodes have no proofs. A partial AND clause is discarded in its entirety.
    Forgotten records remain evidence: consolidation intentionally soft-forgets its inputs while
    retaining their provenance. Withdrawal instead retires clauses or deletes the records.
    A pending assessment replaces the same source set, just like durable clause insertion.
    """
    nodes, truncated = _load_nodes(connection, memory_id, pending_assessment=pending_assessment)
    support = independent_support(nodes, memory_id)
    result = replace(support, truncated=support.truncated or truncated)
    if result.truncated:
        LOGGER.warning(
            f"independent evidence projection for {memory_id} reached its resource limit; "
            "using a support lower bound"
        )
    return result


def _load_nodes(
    connection: sqlite3.Connection,
    memory_id: str,
    *,
    pending_assessment: Assessment | None = None,
) -> tuple[dict[str, EvidenceNode], bool]:
    nodes: dict[str, EvidenceNode] = {}
    pending = deque([memory_id])
    seen = {memory_id}
    member_rows = 0
    visited = 0
    truncated = False
    while pending and visited < _MAX_NODES:
        current = pending.popleft()
        visited += 1
        node, used, clipped = _read_node(connection, current, _MAX_MEMBER_ROWS - member_rows)
        member_rows += used
        truncated = truncated or clipped
        if node is None:
            continue
        if current == memory_id and pending_assessment is not None:
            # Formation must compare grounded support before retiring another state, although
            # its explicit clause is not inserted until the semantic row has been written.
            members = len(pending_assessment.sources)
            pending_sources = frozenset(pending_assessment.sources)
            node = replace(
                node,
                root_group=None,
                assessments=tuple(
                    assessment
                    for assessment in node.assessments
                    if frozenset(assessment.sources) != pending_sources
                ),
            )
            if member_rows + members > _MAX_MEMBER_ROWS:
                truncated = True
            else:
                member_rows += members
                node = replace(
                    node,
                    root_group=None,
                    assessments=(*node.assessments, pending_assessment),
                )
        nodes[current] = node
        sources = sorted(
            {source for assessment in node.assessments for source in assessment.sources}
        )
        for source in sources:
            if source not in seen:
                seen.add(source)
                pending.append(source)
    return nodes, truncated or bool(pending)


def _read_node(
    connection: sqlite3.Connection, memory_id: str, remaining: int
) -> tuple[EvidenceNode | None, int, bool]:
    row = connection.execute(
        """
            SELECT s.kind, s.basis, s.source_id, s.predicate, s.identity_id,
                   COALESCE(
                       (SELECT MIN(e.source_group_id) FROM memory_evidence AS e
                        WHERE e.memory_id = r.memory_id AND e.retired_at IS NULL
                        HAVING MIN(e.source_group_id) = MAX(e.source_group_id)),
                       CASE WHEN s.kind = 'observation' THEN s.source_id END,
                       r.memory_id
                   ) AS source_group_id,
                   COALESCE((SELECT confidence FROM memory_versions AS v
                             WHERE v.memory_id = r.memory_id ORDER BY version LIMIT 1), 1.0)
                       AS confidence,
                   EXISTS(SELECT 1 FROM memory_evidence_clauses AS c
                          WHERE c.memory_id = r.memory_id) AS has_ancestry
            FROM memory_records AS r
            LEFT JOIN memory_semantics AS s ON s.memory_id = r.memory_id
            WHERE r.memory_id = ?
            """,
        (memory_id,),
    ).fetchone()
    if row is None:
        return None, 0, False
    kind = optional_row_text(row, "kind")
    basis = optional_row_text(row, "basis")
    node = EvidenceNode(
        root_group=None,
        kind=kind,
        basis=basis,
        predicate=optional_row_text(row, "predicate"),
        identity_id=optional_row_text(row, "identity_id"),
        source_group_id=row_text(row, "source_group_id"),
    )
    if not row["has_ancestry"] and (
        kind is None or kind == "observation" or basis in {"user_statement", "response_feedback"}
    ):
        return (
            replace(
                node,
                root_group=optional_row_text(row, "source_id") or memory_id,
                confidence=float(row["confidence"]),
            ),
            0,
            False,
        )
    if remaining <= 0:
        return None, 0, True
    rows = connection.execute(
        """
            SELECT c.clause_id, c.confidence, c.member_count, m.source_memory_id
            FROM memory_evidence_clauses AS c
            JOIN memory_evidence_clause_members AS m
              ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
            WHERE c.memory_id = ? AND c.retired_at IS NULL AND c.confidence > 0
            ORDER BY c.clause_id, m.position
            LIMIT ?
            """,
        (memory_id, remaining + 1),
    ).fetchall()
    selected = rows[:remaining]
    return (
        replace(node, assessments=_complete_assessments(selected)),
        len(selected),
        len(rows) > remaining,
    )


def _complete_assessments(rows: list[sqlite3.Row]) -> tuple[Assessment, ...]:
    members: dict[str, list[str]] = {}
    scores: dict[str, tuple[int, float]] = {}
    for member in rows:
        clause = row_text(member, "clause_id")
        members.setdefault(clause, []).append(row_text(member, "source_memory_id"))
        scores[clause] = (int(member["member_count"]), float(member["confidence"]))
    return tuple(
        Assessment(sources=tuple(sources), confidence=scores[clause][1])
        for clause, sources in members.items()
        if len(sources) == scores[clause][0]
    )
