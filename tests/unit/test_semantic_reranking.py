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

from mindbridge import Blob, Memory, Modality, ObservationContext, RetrievalMode, RetrievalScope
from mindbridge._telemetry import MODEL_REQUEST_COUNT
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import EmbedTask, ModelInput
from mindbridge.types import MemoryType, RetrievalCandidateTrace


class Embedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
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

    def __init__(self, *, scores: tuple[float, ...] | None = None, fail: bool = False) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.scores = scores
        self.fail = fail
        self.closed = 0

    def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        self.calls.append((query, tuple(documents)))
        if self.fail:
            raise RuntimeError("private provider body")
        if self.scores is not None:
            return self.scores
        return tuple(8.0 if "cancelled" in document else -3.0 for document in documents)

    def close(self) -> None:
        self.closed += 1


def test_public_search_supplies_original_image_video_and_stable_source_binding(
    tmp_path: Path,
) -> None:
    class MediaReranker(Reranker):
        reranking_capabilities = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})

        def rerank_media(
            self, query: ModelInput, documents: Sequence[ModelInput]
        ) -> tuple[float, ...]:
            assert query.text == "find evidence"
            kinds = {asset.modality for document in documents for asset in document.assets}
            assert kinds == {Modality.IMAGE, Modality.VIDEO}
            for document in documents:
                for asset in document.assets:
                    assert asset.path is not None and asset.path.is_file()
                    assert asset.path.read_bytes() in (b"original image", b"original video")
            return tuple(
                2.0 if any(asset.modality is Modality.VIDEO for asset in document.assets) else 0.1
                for document in documents
            )

    class MediaEmbedder(Embedder):
        embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})

    with Memory(
        tmp_path,
        embedder=MediaEmbedder(),
        reranker=MediaReranker(),
        minimum_relevance=0,
        ambiguity_margin=0,
    ) as memory:
        image = memory.add(("photo", Blob(b"original image", "image/png")))
        video = memory.add(("clip", Blob(b"original video", "video/mp4")))
        hits = memory.search("find evidence", limit=2)
        assert [hit.id for hit in hits] == [video.id, image.id]
        assert hits[0].assets[0].modality is Modality.VIDEO


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
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0,
        ambiguity_margin=0,
    ) as memory:
        decoys = [memory.add(f"confirmed reservation {index}") for index in range(12)]
        target = memory.add("cancelled reservation, refunded GBP 55")
        result = memory.search_with_trace("Which reservation was cancelled?", limit=1)
        assert result.hits[0].id == target.id
        assert result.hits[0].content == target.content
        assert result.hits[0].score == pytest.approx(0.6)
        assert len(reranker.calls[0][1]) == 13
        by_id = {row.memory_id: row for row in result.trace.candidates}
        assert by_id[target.id].reranking_score == 8.0
        assert all(by_id[row.id].reranking_score == -3.0 for row in decoys)
        complete = memory.search("Which reservation was cancelled?", limit=20)
        assert {row.id for row in complete} == {target.id, *(row.id for row in decoys)}
        spans = [
            span
            for span in exporter.get_finished_spans()
            if span.name == "mindbridge.model.reranking"
        ]
        assert len(spans) == 2
        for span in spans:
            assert span.attributes is not None
            assert span.attributes[MODEL_REQUEST_COUNT] == 1
            assert span.attributes["mindbridge.reranking.scored_candidates"] == 13
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
        minimum_relevance=0,
        ambiguity_margin=0,
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
        result = memory.search_with_trace(
            "live", memory_type=MemoryType.EPISODIC, occurred_from=now, scope=scope, limit=5
        )
        assert [row.id for row in result.hits] == [live.id]
        assert {
            row.memory_id for row in result.trace.candidates if row.reranking_score is not None
        } == {live.id}
        assert len(reranker.calls[-1][1]) == 1
        assert "cancelled live" in reranker.calls[-1][1][0]
    shutil.rmtree(tmp_path / "zvec")
    document_calls = embedder.document_calls
    with Memory(
        tmp_path,
        embedder=embedder,
        reranker=Reranker(),
        minimum_relevance=0,
        ambiguity_margin=0,
    ) as rebuilt:
        assert embedder.document_calls == document_calls
        assert (
            rebuilt.search("live", memory_type=MemoryType.EPISODIC, occurred_from=now, scope=scope)[
                0
            ].id
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
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0,
        ambiguity_margin=0,
    ) as memory:
        media = memory.add(Blob(b"original", "image/png"))
        confirmed = memory.add("confirmed reservation")
        cancelled = memory.add("cancelled reservation")
        hits = memory.search("cancelled reservation", limit=3)
        assert [row.id for row in hits] == [media.id, cancelled.id, confirmed.id]
        assert len(reranker.calls[-1][1]) == 2
        captioned = memory.add(("cancelled photo caption", Blob(b"captioned", "image/png")))
        mixed = memory.search("cancelled reservation", limit=4)
        assert mixed[0].id == media.id
        assert captioned.id in {row.id for row in mixed}
        assert any("cancelled photo caption" in document for document in reranker.calls[-1][1])
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
        minimum_relevance=0,
        ambiguity_margin=0,
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
        minimum_relevance=0,
        ambiguity_margin=0.1,
    ) as memory:
        memory.add("first")
        memory.add("second")
        assert memory.search("query", limit=1) == ()
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
