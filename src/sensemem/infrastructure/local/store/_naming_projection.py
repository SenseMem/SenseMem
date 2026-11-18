"""Atomic speech-document updates for candidate evidence naming changes."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from sensemem.identity import SpeakerSegment
from sensemem.infrastructure.local.store._codec import (
    memory_from_row,
    optional_row_text,
    prepare_write_batch,
    row_text,
)
from sensemem.infrastructure.local.store._corroboration import independent_evidence_enabled
from sensemem.infrastructure.local.store._identity import dependent_naming_identities
from sensemem.infrastructure.local.store.errors import StaleOperationError
from sensemem.infrastructure.local.store.media import _read_speech
from sensemem.infrastructure.local.store.records import (
    _read_memory_assets,
    read_memories,
    replace_memory_embeddings,
)
from sensemem.infrastructure.local.store.rows import StoredEmbedding, StoredMemory


@dataclass(frozen=True, slots=True)
class NamingDocuments:
    """Exact candidate names, documents and cached speech needed for a naming update."""

    names: tuple[tuple[str, str | None], ...]
    memories: tuple[StoredMemory, ...]
    speech: tuple[tuple[str, tuple[SpeakerSegment, ...]], ...]


@dataclass(frozen=True, slots=True)
class NamingProjection:
    """Model-prepared text and vectors, conditional on the same candidate mutation."""

    source: NamingDocuments
    memories: tuple[StoredMemory, ...]
    embeddings: tuple[StoredEmbedding, ...]


class NamingPreparationRequired(Exception):
    """Roll back a candidate mutation so the kernel can prepare its model-dependent documents."""

    def __init__(self, documents: NamingDocuments) -> None:
        super().__init__("naming changes require prepared speech documents")
        self.documents = documents


def validate_speech_projection(
    connection: sqlite3.Connection, identity_id: str, expected: Sequence[StoredMemory]
) -> None:
    """Reject a prepared reindex if an affected speech record arrived, changed, or disappeared."""
    identifiers = tuple(
        row_text(row, "memory_id")
        for row in connection.execute(
            """
        SELECT DISTINCT m.memory_id FROM speech_segments AS s
        JOIN memory_assets AS ma ON ma.asset_id = s.asset_id
        JOIN memory_records AS m ON m.memory_id = ma.memory_id
        WHERE s.speaker_id = ? AND instr(m.content, '[speech identities:') > 0
        """,
            (identity_id,),
        ).fetchall()
    )
    previous = {memory.memory_id: memory for memory in expected}
    if set(identifiers) != previous.keys():
        raise StaleOperationError("speech projection sources changed during preparation")
    for memory in read_memories(connection, identifiers):
        before = previous[memory.memory_id]
        if (
            replace(
                memory,
                access_count=before.access_count,
                last_accessed_at=before.last_accessed_at,
                updated_at=before.updated_at,
            )
            != before
        ):
            raise StaleOperationError("speech projection content changed during preparation")


def naming_snapshot(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    identity_ids: Sequence[str] = (),
    force: bool = False,
) -> dict[str, str | None]:
    """Remember candidate names before links or versions can remove dependencies."""
    if not force and not independent_evidence_enabled(connection):
        return {}
    affected = {*dependent_naming_identities(connection, memory_ids), *identity_ids}
    before = _names(connection, sorted(affected))
    if force:
        for identity_id in identity_ids:
            # An absorbed identity has no registry row until the candidate split restores it.
            before.setdefault(identity_id, None)
    return before


def _names(connection: sqlite3.Connection, identity_ids: Sequence[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for identity_id in identity_ids:
        row = connection.execute(
            "SELECT name FROM identities WHERE identity_id = ?", (identity_id,)
        ).fetchone()
        if row is not None:
            result[identity_id] = optional_row_text(row, "name")
    return result


def refresh_naming_documents(
    connection: sqlite3.Connection,
    before: Mapping[str, str | None],
    projection: NamingProjection | None,
    *,
    force_ids: Sequence[str] = (),
) -> None:
    """Apply prepared vectors, or return preparation inputs by rolling the candidate back.

    No callback or model runs in this transaction. The kernel handles the preparation request
    outside SQLite, then repeats the mutation with a projection checked against its candidate
    names, source documents and cached speech. Both passes use the same authoritative write.
    """
    changed = {
        identity_id: name
        for identity_id, name in _names(connection, tuple(before)).items()
        if name != before[identity_id] or identity_id in force_ids
    }
    memory_ids = sorted(
        {
            row_text(row, "memory_id")
            for identity_id in changed
            for row in connection.execute(
                """
                SELECT DISTINCT ma.memory_id FROM speech_segments AS s
                JOIN memory_assets AS ma ON ma.asset_id = s.asset_id
                JOIN memory_records AS m ON m.memory_id = ma.memory_id
                WHERE s.speaker_id = ? AND instr(m.content, '[speech identities:') > 0
                """,
                (identity_id,),
            ).fetchall()
        }
    )
    assets = _read_memory_assets(connection, memory_ids)
    current = {
        memory_id: memory_from_row(
            connection.execute(
                "SELECT * FROM memory_records WHERE memory_id = ?", (memory_id,)
            ).fetchone(),
            assets=assets[memory_id],
            context=None,
        )
        for memory_id in memory_ids
    }
    speech = {
        asset.asset_id: _read_speech(connection, asset.asset_id)
        for memory in current.values()
        for asset in memory.assets
        if asset.modality in {"audio", "video"}
        and connection.execute(
            "SELECT 1 FROM speech_analyses WHERE asset_id = ?", (asset.asset_id,)
        ).fetchone()
        is not None
    }
    source = NamingDocuments(
        names=tuple(sorted(changed.items())),
        memories=tuple(current.values()),
        speech=tuple(sorted(speech.items())),
    )
    if projection is not None and source != projection.source:
        raise StaleOperationError("naming projection inputs changed during preparation")
    if not current:
        return
    if projection is None:
        raise NamingPreparationRequired(source)
    memories, embeddings = projection.memories, projection.embeddings
    # Only text and vectors belong to the prepared projection. The transaction may already
    # have retired, forgotten, or narrowed a row: never reapply old semantics.
    updated = tuple(
        replace(current[memory.memory_id], content=memory.content, updated_at=memory.updated_at)
        for memory in memories
    )
    prepared, vectors, by_memory = prepare_write_batch(updated, embeddings)
    replace_memory_embeddings(connection, prepared, vectors, by_memory)
