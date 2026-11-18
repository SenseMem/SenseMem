"""Connection-level writes for authoritative embeddings and their exact text selectors."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence

from sensemem.infrastructure.local.store._codec import (
    datetime_text,
    pack_vector,
    require_storable_vector,
    row_text,
)
from sensemem.infrastructure.local.store.rows import StoredEmbedding


def write_embedding(
    connection: sqlite3.Connection,
    embedding: StoredEmbedding,
) -> None:
    _write_embedding(connection, embedding, parent_content=None)


def write_embeddings(
    connection: sqlite3.Connection,
    embeddings: Sequence[StoredEmbedding],
) -> None:
    """Reuse one parent's selector source across its consecutive keys in this transaction."""
    parent_id: str | None = None
    parent_content: tuple[str, str] | None = None
    for embedding in embeddings:
        if embedding.memory_id != parent_id:
            parent_content = None
        parent_content = _write_embedding(connection, embedding, parent_content=parent_content)
        parent_id = embedding.memory_id


def _write_embedding(
    connection: sqlite3.Connection,
    embedding: StoredEmbedding,
    *,
    parent_content: tuple[str, str] | None,
) -> tuple[str, str] | None:
    # The only place a caller-supplied vector reaches the authoritative table, and so the only
    # place its content is checked. See StoredEmbedding for why hydration does not repeat this.
    require_storable_vector(embedding.values, normalized=embedding.normalized)
    connection.execute(
        """
        INSERT INTO embeddings (
            embedding_id, memory_id, object_part, model_id, space_id, task,
            dimension, normalized, vector, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (embedding_id) DO UPDATE SET
            memory_id = excluded.memory_id,
            object_part = excluded.object_part,
            model_id = excluded.model_id,
            space_id = excluded.space_id,
            task = excluded.task,
            dimension = excluded.dimension,
            normalized = excluded.normalized,
            vector = excluded.vector,
            created_at = excluded.created_at
        """,
        (
            embedding.embedding_id,
            embedding.memory_id,
            embedding.object_part,
            embedding.model_id,
            embedding.space_id,
            embedding.task,
            len(embedding.values),
            int(embedding.normalized),
            pack_vector(embedding.values),
            datetime_text(embedding.created_at),
        ),
    )
    connection.execute(
        "DELETE FROM embedding_text_selectors WHERE embedding_id = ?",
        (embedding.embedding_id,),
    )
    if not embedding.text_selectors:
        return parent_content
    if parent_content is None:
        row = connection.execute(
            "SELECT content FROM memory_records WHERE memory_id = ?",
            (embedding.memory_id,),
        ).fetchone()
        if row is None:
            raise ValueError("text selectors require an existing parent memory")
        content = row_text(row, "content")
        parent_content = (content, hashlib.sha256(content.encode("utf-8")).hexdigest())
    content, parent_digest = parent_content
    for selector_position, selector in enumerate(embedding.text_selectors):
        if selector.parent_content_sha256 != parent_digest:
            raise ValueError("text selector parent digest does not match stored content")
        source_pieces = tuple(
            content[piece.start_codepoint : piece.end_codepoint] for piece in selector.pieces
        )
        for piece, source_text in zip(selector.pieces, source_pieces, strict=True):
            if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != piece.piece_sha256:
                raise ValueError("text selector piece does not match stored content")
        embedding_input = (
            source_pieces[0]
            if len(source_pieces) == 1
            else f"{source_pieces[0]}\n\n{source_pieces[1]}"
        )
        if (
            hashlib.sha256(embedding_input.encode("utf-8")).hexdigest()
            != selector.embedding_input_sha256
        ):
            raise ValueError("text selector does not reconstruct its embedding input")
        connection.execute(
            """
            INSERT INTO embedding_text_selectors (
                embedding_id, selector_position, parent_content_sha256,
                embedding_input_sha256, recipe_version
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                embedding.embedding_id,
                selector_position,
                selector.parent_content_sha256,
                selector.embedding_input_sha256,
                selector.recipe_version,
            ),
        )
        for piece_position, piece in enumerate(selector.pieces):
            connection.execute(
                """
                INSERT INTO embedding_text_span_pieces (
                    embedding_id, selector_position, piece_position, role,
                    start_codepoint, end_codepoint, piece_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    embedding.embedding_id,
                    selector_position,
                    piece_position,
                    piece.role,
                    piece.start_codepoint,
                    piece.end_codepoint,
                    piece.piece_sha256,
                ),
            )
    return parent_content
