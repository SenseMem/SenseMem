"""Persist and validate exact predicate resolution for the current schema."""

from __future__ import annotations

import sqlite3

from sensemem.infrastructure.local.store.errors import UnsupportedSchemaError
from sensemem.kernel.predicates import (
    PREDICATE_REGISTRY_VERSION,
    claim_lineage_id,
    resolve_predicate,
)


def predicate_binding(
    lineage_id: str,
    *,
    kind: str,
    basis: str,
    predicate: str | None,
    value: str | None,
    subject: str | None,
    identity_id: str | None,
    frame_id: str | None,
    anchor: str | None,
) -> tuple[str | None, str]:
    resolution = resolve_predicate(predicate, kind=kind, basis=basis, value=value)
    canonical = claim_lineage_id(
        kind=kind,
        predicate=predicate,
        subject=subject,
        identity_id=identity_id,
        frame_id=frame_id,
        anchor=anchor,
        resolution=resolution,
    )
    if lineage_id != canonical:
        # A caller's custom lineage is authority too; an apparent alias is no permission to
        # reinterpret it. Leave the resolved key empty and identify the custom binding explicitly.
        return None, "custom"
    return resolution.key, resolution.match


def validate_predicate_provenance(connection: sqlite3.Connection) -> None:
    if connection.execute(
        "SELECT 1 FROM memory_semantics WHERE predicate_registry IS NULL OR predicate_registry <> ? "
        "OR predicate_match IS NULL OR predicate_match NOT IN "
        "('canonical', 'alias', 'literal', 'ineligible', 'custom') LIMIT 1",
        (PREDICATE_REGISTRY_VERSION,),
    ).fetchone():
        raise UnsupportedSchemaError("local schema has unsupported predicate provenance")
