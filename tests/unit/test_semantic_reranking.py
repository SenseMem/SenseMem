"""Public SDK tests for semantic ordering after authoritative candidate filtering."""

from __future__ import annotations

import math
import shutil
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sensemem import (
    Blob,
    Memory,
    MemoryConfig,
    Modality,
    ObservationContext,
    RetrievalMode,
    RetrievalScope,
)
from sensemem._telemetry import MODEL_REQUEST_COUNT
from sensemem.exceptions import ModelError, ValidationError
from sensemem.models.base import EmbedTask, ModelInput
from sensemem.types import ContentInput, MemoryType, RetrievalCandidateTrace


class Embedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})
    embedding_model = "reranking-fixture"
    embedding_space = "reranking-fixture:64"
    embedding_dimension = 64

    def __init__(self) -> None:
        self.document_calls = 0

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        if task is EmbedTask.DOCUMENT:
            self.document_calls += 1
        return tuple(
            (
                (1.0, 0.0)
                if task is EmbedTask.QUERY
                else (0.6, 0.8)
                if "cancelled" in value.text
                else (0.95, math.sqrt(1 - 0.95**2))
                if value.assets and not value.text
                else (0.9, math.sqrt(0.19))
            )
            + (0.0,) * 62
            for value in inputs
        )

    def close(self) -> None:
        return None


class Reranker:
    reranking_model = "query-document-fixture@fixed"
    reranking_capabilities = frozenset({Modality.TEXT})

    def __init__(self, *, scores: tuple[float, ...] | None = None, fail: bool = False) -> None:
        self.calls: list[tuple[ModelInput, tuple[ModelInput, ...]]] = []
        self.scores = scores
        self.fail = fail
        self.closed = 0

    def rerank(self, query: ModelInput, documents: Sequence[ModelInput]) -> tuple[float, ...]:
        self.calls.append((query, tuple(documents)))
        if self.fail:
            raise RuntimeError("private provider body")
        if self.scores is not None:
            return self.scores
        return tuple(8.0 if "cancelled" in document.text else -3.0 for document in documents)

    def close(self) -> None:
        self.closed += 1


def test_complete_eligible_pool_is_reranked_before_limit_without_rewriting_scores(
    tmp_path: Path,
) -> None:
    reranker = Reranker()
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with Memory(
        tmp_path,
        embedder=Embedder(),
        reranker=reranker,
        tracer=provider.get_tracer("test"),
        config=MemoryConfig(
            retrieval_mode=RetrievalMode.DENSE, minimum_relevance=0, ambiguity_margin=0
        ),
    ) as memory:
        decoys = [memory.add(f"confirmed reservation {index}") for index in range(12)]
        target = memory.add("cancelled reservation, refunded GBP 55")
        result = memory.search("Which reservation was cancelled?", limit=1, explain=True)
        assert result.trace is not None
        assert result.hits[0].id == target.id
        assert result.hits[0].content == target.content
        assert result.hits[0].score == pytest.approx(0.6)
        assert len(reranker.calls[0][1]) == 13
        by_id = {row.memory_id: row for row in result.trace.candidates}
        assert by_id[target.id].reranking_score == 8.0
        assert all(by_id[row.id].reranking_score == -3.0 for row in decoys)
        complete = memory.search("Which reservation was cancelled?", limit=20).hits
        assert {row.id for row in complete} == {target.id, *(row.id for row in decoys)}
        spans = [
            span
            for span in exporter.get_finished_spans()
            if span.name == "sensemem.model.reranking"
        ]
        assert len(spans) == 2
        for span in spans:
            assert span.attributes is not None
            assert span.attributes[MODEL_REQUEST_COUNT] == 1
            assert span.attributes["sensemem.reranking.scored_candidates"] == 13
    assert reranker.closed == 1


def test_reranker_never_receives_stale_out_of_range_or_wrong_type_sources(tmp_path: Path) -> None:
    reranker = Reranker()
    embedder = Embedder()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    scope = RetrievalScope(place_id="visible")
    with Memory(
        tmp_path,
        embedder=embedder,
        reranker=reranker,
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        live = memory.add(
            "cancelled live",
            memory_type=MemoryType.EPISODIC,
            occurred_at=now,
            context=ObservationContext(place_id="visible"),
        )
        stale = memory.add(
            "stale source",
            memory_type=MemoryType.EPISODIC,
            occurred_at=now,
            context=ObservationContext(place_id="visible"),
        )
        memory.add(
            "wrong type",
            memory_type=MemoryType.PROCEDURAL,
            occurred_at=now,
            context=ObservationContext(place_id="visible"),
        )
        memory.add(
            "outside requested time",
            memory_type=MemoryType.EPISODIC,
            occurred_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
            context=ObservationContext(place_id="visible"),
        )
        memory.add(
            "outside requested place",
            memory_type=MemoryType.EPISODIC,
            occurred_at=now,
            context=ObservationContext(place_id="other"),
        )
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection, connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("DELETE FROM memory_records WHERE memory_id=?", (stale.id,))
        result = memory.search(
            "live",
            memory_type=MemoryType.EPISODIC,
            occurred_from=now,
            scope=scope,
            limit=5,
            explain=True,
        )
        assert result.trace is not None
        assert [row.id for row in result.hits] == [live.id]
        assert {
            row.memory_id for row in result.trace.candidates if row.reranking_score is not None
        } == {live.id}
        assert len(reranker.calls[-1][1]) == 1
        assert "cancelled live" in reranker.calls[-1][1][0].text
    shutil.rmtree(tmp_path / "zvec")
    document_calls = embedder.document_calls
    with Memory(
        tmp_path,
        embedder=embedder,
        reranker=Reranker(),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as rebuilt:
        assert embedder.document_calls == document_calls
        assert (
            rebuilt.search("live", memory_type=MemoryType.EPISODIC, occurred_from=now, scope=scope)
            .hits[0]
            .id
            == live.id
        )
        assert embedder.document_calls == document_calls


def test_empty_text_media_keep_positions_and_media_only_queries_skip_text_reranking(
    tmp_path: Path,
) -> None:
    reranker = Reranker()
    with Memory(
        tmp_path,
        embedder=Embedder(),
        reranker=reranker,
        config=MemoryConfig(
            retrieval_mode=RetrievalMode.DENSE, minimum_relevance=0, ambiguity_margin=0
        ),
    ) as memory:
        media = memory.add(Blob(b"original", "image/png"))
        confirmed = memory.add("confirmed reservation")
        cancelled = memory.add("cancelled reservation")
        hits = memory.search("cancelled reservation", limit=3).hits
        assert [row.id for row in hits] == [media.id, cancelled.id, confirmed.id]
        assert len(reranker.calls[-1][1]) == 2
        captioned = memory.add(("cancelled photo caption", Blob(b"captioned", "image/png")))
        mixed = memory.search("cancelled reservation", limit=4).hits
        assert mixed[0].id == media.id
        assert captioned.id in {row.id for row in mixed}
        assert any("cancelled photo caption" in document.text for document in reranker.calls[-1][1])
        assert len(reranker.calls[-1][1]) == 3
        calls = len(reranker.calls)
        memory.search(Blob(b"query", "image/png"), limit=3)
        assert len(reranker.calls) == calls


@pytest.mark.parametrize("scores", [(1.0,), (float("nan"), 1.0)])
def test_invalid_reranking_output_is_explicit_instead_of_silent_fallback(
    tmp_path: Path, scores: tuple[float, ...]
) -> None:
    with Memory(
        tmp_path,
        embedder=Embedder(),
        reranker=Reranker(scores=scores),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add("first")
        memory.add("second")
        with pytest.raises(ModelError) as raised:
            memory.search("query")
        assert raised.value.reason == "response_invalid"
        assert raised.value.stage == "reranking"


def test_reranker_failure_is_sanitized_and_existing_ambiguity_policy_still_applies(
    tmp_path: Path,
) -> None:
    reranker = Reranker(fail=True)
    with Memory(
        tmp_path,
        embedder=Embedder(),
        reranker=reranker,
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0.1),
    ) as memory:
        memory.add("first")
        memory.add("second")
        assert memory.search("query", limit=1).hits == ()
        assert reranker.calls == []
        with pytest.raises(ModelError) as raised:
            memory.search("query", limit=2)
        assert str(raised.value) == "reranking model failed"
        assert raised.value.reason == "model_failed"
        assert raised.value.stage == "reranking"


def test_trace_reranking_logits_allow_negative_values_without_relaxing_existing_scores() -> None:
    assert (
        RetrievalCandidateTrace("source", ("index",), reranking_score=-8.0).reranking_score == -8.0
    )
    with pytest.raises(ValidationError, match="reranking_score"):
        RetrievalCandidateTrace("source", ("index",), reranking_score=float("inf"))
    with pytest.raises(ValidationError, match="must not exceed one"):
        RetrievalCandidateTrace("source", ("index",), final_score=8.0)


@pytest.mark.parametrize("query_kind", ["text", "image", "video", "mixed"])
def test_native_reranking_automatically_includes_original_query_and_candidate_media(
    tmp_path: Path,
    query_kind: str,
) -> None:
    class MultimodalReranker(Reranker):
        reranking_capabilities = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})

        def rerank(self, query: ModelInput, documents: Sequence[ModelInput]) -> tuple[float, ...]:
            self.calls.append((query, tuple(documents)))
            for value in (query, *documents):
                for asset in value.assets:
                    assert asset.path is not None and asset.path.read_bytes()
            return tuple(0.9 if value.assets else 0.1 for value in documents)

    reranker = MultimodalReranker()
    queries: dict[str, ContentInput] = {
        "text": "Find this shape",
        "image": Blob(b"query-image", "image/png"),
        "video": Blob(b"query-video", "video/mp4"),
        "mixed": (
            "Find this shape",
            Blob(b"query-image", "image/png"),
            Blob(b"query-video", "video/mp4"),
        ),
    }
    with Memory(
        tmp_path,
        embedder=Embedder(),
        reranker=reranker,
        config=MemoryConfig(
            retrieval_mode=RetrievalMode.DENSE, minimum_relevance=0, ambiguity_margin=0
        ),
    ) as memory:
        text = memory.add("A text candidate")
        captioned = memory.add(("A caption", Blob(b"original-image", "image/png")))
        video = memory.add(Blob(b"original-video", "video/mp4"))
        outcome = memory.search(queries[query_kind], limit=3, explain=True)
        assert outcome.trace is not None
        assert {hit.id for hit in outcome.hits[:2]} == {captioned.id, video.id}
        assert outcome.hits[-1].id == text.id
        query, documents = reranker.calls[-1]
        assert (
            query.modalities
            == {
                "text": frozenset({Modality.TEXT}),
                "image": frozenset({Modality.IMAGE}),
                "video": frozenset({Modality.VIDEO}),
                "mixed": frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
            }[query_kind]
        )
        assert len(documents) == 3
        assert any(
            value.text == "A caption" and value.assets[0].sha256 == captioned.assets[0].sha256
            for value in documents
            if value.assets
        )
        assert any(
            not value.text and value.modalities == frozenset({Modality.VIDEO})
            for value in documents
        )
        assert all(row.reranking_score is not None for row in outcome.trace.candidates)


def test_candidate_media_stay_readable_during_reranking_and_are_released_on_failure(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []

    class DeletingReranker(Reranker):
        reranking_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

        def rerank(self, query: ModelInput, documents: Sequence[ModelInput]) -> tuple[float, ...]:
            path = documents[0].assets[0].path
            assert path is not None
            paths.append(path)
            assert memory.delete(record.id)
            assert path.read_bytes() == b"original-image"
            raise RuntimeError("private failure after reading media")

    with Memory(
        tmp_path,
        embedder=Embedder(),
        reranker=DeletingReranker(),
        config=MemoryConfig(
            retrieval_mode=RetrievalMode.DENSE, minimum_relevance=0, ambiguity_margin=0
        ),
    ) as memory:
        record = memory.add(Blob(b"original-image", "image/png"))
        with pytest.raises(ModelError, match="reranking model failed"):
            memory.search("find image")
        assert len(paths) == 1 and not paths[0].exists()
        assert memory.search("find image").hits == ()
