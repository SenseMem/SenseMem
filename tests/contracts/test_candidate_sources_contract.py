"""Public candidate sources choose native routes or shortlist local IDs from another system."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sensemem import (
    CandidateBatch,
    CandidateHit,
    CandidateQuery,
    CandidateReadPort,
    CandidateRequest,
    DefaultCandidateSource,
    EmbedTask,
    Memory,
    MemoryCandidate,
    MemoryCategory,
    MemoryPlugins,
    MemorySettings,
    Modality,
    ModelError,
    ModelInput,
    ObservationContext,
    RetrievalMode,
    RetrievalScope,
    ValidationError,
)

REFERENCE = datetime(2026, 11, 16, tzinfo=timezone.utc)


class CandidateEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = embedding_space = "candidate-source-fixture-v1"
    embedding_dimension = 2

    def __init__(self) -> None:
        self.query_calls = 0

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        self.query_calls += int(task is EmbedTask.QUERY)
        return tuple((1.0, 0.0) for _value in inputs)

    def close(self) -> None:
        pass


class RewrittenLexicalSource:
    requires_embeddings = False

    def __init__(self) -> None:
        self.requests: list[CandidateRequest] = []

    def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch:
        self.requests.append(request)
        (hits,) = reader.read((CandidateQuery(kind="lexical", text="sapphire"),))
        return CandidateBatch(lexical=hits, exhausted=len(hits) < request.limit)

    def close(self) -> None:
        pass


class ExternalShortlist:
    requires_embeddings = False

    def __init__(self) -> None:
        self.matches: tuple[MemoryCandidate, ...] = ()
        self.closed = 0

    def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch:
        return CandidateBatch(memories=self.matches, exhausted=True)

    def close(self) -> None:
        self.closed += 1


def test_custom_source_changes_native_lexical_routes_without_query_embedding(
    tmp_path: Path,
) -> None:
    embedder, source = CandidateEmbedder(), RewrittenLexicalSource()
    with Memory(
        tmp_path,
        embedder=embedder,
        candidate_source=source,
        settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        target = memory.add("The access token is sapphire.")
        memory.add("The access token is amber.")
        result = memory.search("Find the remembered access token")
        assert tuple(hit.id for hit in result.hits) == (target.id,)
        assert source.requests[-1].vectors == ()
        assert source.requests[-1].mode is RetrievalMode.HYBRID
        assert embedder.query_calls == 0


def test_external_ids_are_hydrated_from_sqlite_with_all_request_filters(tmp_path: Path) -> None:
    source, embedder = ExternalShortlist(), CandidateEmbedder()
    with Memory.from_plugins(
        tmp_path,
        plugins=MemoryPlugins(embedder=embedder, candidate_source=source),
        settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        target = memory.add(
            "SQLite holds the sapphire token.",
            occurred_at=REFERENCE,
            context=ObservationContext(place_id="home"),
        )
        wrong_place = memory.add(
            "Office token.", occurred_at=REFERENCE, context=ObservationContext(place_id="office")
        )
        wrong_type = memory.add(
            "A token episode.",
            memory_type=MemoryCategory.EPISODIC,
            occurred_at=REFERENCE,
            context=ObservationContext(place_id="home"),
        )
        old = memory.add(
            "Old token.",
            occurred_at=REFERENCE - timedelta(days=30),
            context=ObservationContext(place_id="home"),
        )
        deleted = memory.add("Deleted token.", occurred_at=REFERENCE)
        assert memory.delete(deleted.id)
        source.matches = tuple(
            MemoryCandidate(memory_id=memory_id, relevance=0.95)
            for memory_id in (
                target.id,
                wrong_place.id,
                wrong_type.id,
                old.id,
                deleted.id,
                "unknown",
            )
        )
        result = memory.search(
            "query known only to the external source",
            memory_type=MemoryCategory.SEMANTIC,
            occurred_from=REFERENCE - timedelta(days=1),
            scope=RetrievalScope(place_id="home"),
        )
        assert tuple(hit.id for hit in result.hits) == (target.id,)
        assert result.hits[0].content == target.content
        assert embedder.query_calls == 0
    assert source.closed == 1


def test_custom_native_source_cannot_expand_the_memory_type_filter(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=CandidateEmbedder(),
        candidate_source=RewrittenLexicalSource(),
    ) as memory:
        memory.add("A sapphire token.")
        with pytest.raises(ValidationError, match="cannot expand the requested memory types"):
            memory.search("token", memory_type=MemoryCategory.SEMANTIC)


def test_external_source_can_compose_with_the_default_routes(tmp_path: Path) -> None:
    class LocalAndExternal(ExternalShortlist):
        requires_embeddings = True

        def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch:
            local = DefaultCandidateSource().search(request, reader)
            return CandidateBatch(
                dense=local.dense,
                lexical=local.lexical,
                memories=self.matches,
                exhausted=local.exhausted,
            )

    source = LocalAndExternal()
    with Memory(
        tmp_path,
        embedder=CandidateEmbedder(),
        candidate_source=source,
        settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        record = memory.add("The token is sapphire.")
        source.matches = (MemoryCandidate(memory_id=record.id, relevance=0.9),)
        assert tuple(hit.id for hit in memory.search("token").hits) == (record.id,)


def test_stale_native_references_do_not_invent_records(tmp_path: Path) -> None:
    class StaleNativeSource(ExternalShortlist):
        def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch:
            return CandidateBatch(
                dense=(CandidateHit(id="unknown-index-entry", relevance=0.95),), exhausted=True
            )

    with Memory(
        tmp_path, embedder=CandidateEmbedder(), candidate_source=StaleNativeSource()
    ) as memory:
        memory.add("The token is sapphire.")
        assert not memory.search("token").hits


def test_candidate_source_cannot_exceed_the_external_reference_budget(tmp_path: Path) -> None:
    class OversizedSource(ExternalShortlist):
        def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch:
            return CandidateBatch(
                memories=tuple(
                    MemoryCandidate(memory_id=f"unknown-{index}", relevance=0.95)
                    for index in range(request.limit + 1)
                ),
            )

    with (
        Memory(
            tmp_path, embedder=CandidateEmbedder(), candidate_source=OversizedSource()
        ) as memory,
        pytest.raises(ValidationError, match="exceeded the candidate budget"),
    ):
        memory.search("token")


def test_candidate_source_failure_keeps_provider_bodies_out_of_public_errors(
    tmp_path: Path,
) -> None:
    class FailingSource(ExternalShortlist):
        def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch:
            raise RuntimeError("provider response body contains private content")

    with Memory(tmp_path, embedder=CandidateEmbedder(), candidate_source=FailingSource()) as memory:
        with pytest.raises(ModelError) as captured:
            memory.search("token")
        assert captured.value.stage == "candidates"
        assert captured.value.reason == "model_failed"
        assert str(captured.value) == "candidate source failed"


@pytest.mark.parametrize("reverse", (False, True))
def test_external_duplicate_hints_keep_both_routes_and_the_strongest_score(
    tmp_path: Path, reverse: bool
) -> None:
    source = ExternalShortlist()
    with Memory(
        tmp_path,
        embedder=CandidateEmbedder(),
        candidate_source=source,
        settings=MemorySettings(minimum_relevance=0.5, ambiguity_margin=0),
    ) as memory:
        record = memory.add("The token is sapphire.")
        matches = (
            MemoryCandidate(memory_id=record.id, relevance=0.95),
            MemoryCandidate(memory_id=record.id, relevance=0.1),
            MemoryCandidate(memory_id=record.id, relevance=0.8, kind="lexical"),
        )
        source.matches = tuple(reversed(matches)) if reverse else matches
        result = memory.search("sapphire", explain=True)
        assert tuple(hit.id for hit in result.hits) == (record.id,)
        assert result.trace is not None
        trace = next(item for item in result.trace.candidates if item.memory_id == record.id)
        assert trace.dense_relevance == 0.95 and trace.lexical_match


@pytest.mark.parametrize("reverse", (False, True))
def test_native_duplicate_hints_keep_the_strongest_score(tmp_path: Path, reverse: bool) -> None:
    class DuplicateNativeSource(ExternalShortlist):
        def search(self, request: CandidateRequest, reader: CandidateReadPort) -> CandidateBatch:
            (hits,) = reader.read((CandidateQuery(kind="lexical", text="sapphire"),))
            assert hits
            proposals = (
                CandidateHit(id=hits[0].id, relevance=0.95),
                CandidateHit(id=hits[0].id, relevance=0.1),
            )
            return CandidateBatch(
                dense=tuple(reversed(proposals)) if reverse else proposals, exhausted=True
            )

    with Memory(
        tmp_path,
        embedder=CandidateEmbedder(),
        candidate_source=DuplicateNativeSource(),
        settings=MemorySettings(minimum_relevance=0.5, ambiguity_margin=0),
    ) as memory:
        record = memory.add("The token is sapphire.")
        result = memory.search("sapphire", explain=True)
        assert tuple(hit.id for hit in result.hits) == (record.id,)
        assert result.trace is not None
        trace = next(item for item in result.trace.candidates if item.memory_id == record.id)
        assert trace.dense_relevance == 0.95
