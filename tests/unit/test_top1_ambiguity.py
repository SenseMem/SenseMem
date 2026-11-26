"""Lexical route membership must not authorize an unresolved singleton."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sensemem import (
    EmbedTask,
    Memory,
    MemorySettings,
    Modality,
    ModelInput,
    ObservationContext,
    RetrievalMode,
    RetrievalRejection,
    RetrievalScope,
)


class _EqualEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "top1-ambiguity-regression-v1"
    embedding_space = "top1-ambiguity-regression-v1:2"
    embedding_dimension = 2

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _value in inputs)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("mode", (RetrievalMode.HYBRID, RetrievalMode.LEXICAL))
@pytest.mark.parametrize("query", ("Alice", "Alice bakery", "爱丽丝面包店"))
@pytest.mark.parametrize("reverse", (False, True))
def test_lexical_ties_are_ambiguous_even_with_full_query_coverage(
    tmp_path: Path, mode: RetrievalMode, query: str, reverse: bool
) -> None:
    # Both records fully cover the question but disagree. The real index must discover both;
    # replacing only the embedder leaves route generation, hydration, fusion and gating intact.
    with Memory(
        tmp_path,
        embedder=_EqualEmbedder(),
        settings=MemorySettings(retrieval_mode=mode, ambiguity_margin=0.05),
    ) as memory:
        documents = (
            "Alice bakery is open. 爱丽丝面包店营业。",
            "Alice bakery not open. 爱丽丝面包店停业。",
        )
        first, second = memory.add_many(tuple(reversed(documents)) if reverse else documents)
        multiple = memory.search(query, limit=2, explain=True)
        assert multiple.trace is not None
        assert {hit.id for hit in multiple.hits} == {first.id, second.id}
        assert not multiple.trace.ambiguous
        assert all(candidate.lexical_match for candidate in multiple.trace.candidates)

        single = memory.search(query, limit=1, explain=True)
        assert single.trace is not None

    assert single.hits == ()
    assert single.trace.ambiguous
    assert {candidate.rejected_by for candidate in single.trace.candidates} == {
        RetrievalRejection.AMBIGUITY
    }


@pytest.mark.parametrize("reverse", (False, True))
def test_validity_scope_preserves_current_and_historical_preferences(
    tmp_path: Path, reverse: bool
) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    change = datetime(2026, 6, 1, tzinfo=timezone.utc)
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    with Memory(tmp_path, embedder=_EqualEmbedder()) as memory:
        documents = (
            ("Alice prefers tea", ObservationContext(valid_from=start, valid_until=change)),
            ("Alice prefers coffee", ObservationContext(valid_from=change)),
        )
        records = {
            text: memory.add(text, context=context)
            for text, context in (reversed(documents) if reverse else documents)
        }
        previous, current = (records[text] for text, _ in documents)
        assert [
            hit.id for hit in memory.search("Alice prefers", limit=1, reference_at=now).hits
        ] == [current.id]
        assert [
            hit.id
            for hit in memory.search(
                "Alice prefers", limit=1, reference_at=now, scope=RetrievalScope(valid_at=now)
            ).hits
        ] == [current.id]
        assert [
            hit.id
            for hit in memory.search(
                "Alice prefers", limit=1, reference_at=now, scope=RetrievalScope(valid_at=start)
            ).hits
        ] == [previous.id]


@pytest.mark.parametrize("reverse", (False, True))
def test_question_echo_does_not_authorize_an_ambiguous_answer(
    tmp_path: Path, reverse: bool
) -> None:
    documents = (
        "Unanswered question: Alice bakery location?",
        "Her shop is at 17 Elm Street.",
    )
    with Memory(tmp_path, embedder=_EqualEmbedder()) as memory:
        memory.add_many(tuple(reversed(documents)) if reverse else documents)
        result = memory.search("Alice bakery location", limit=1, explain=True)
    assert result.trace is not None
    assert result.hits == ()
    assert result.trace.ambiguous


def test_zero_margin_retains_explicit_opt_out(tmp_path: Path) -> None:
    with Memory(
        tmp_path, embedder=_EqualEmbedder(), settings=MemorySettings(ambiguity_margin=0)
    ) as memory:
        first = memory.add("Alice bakery is open")
        second = memory.add("Alice bakery is closed")
        result = memory.search("Alice bakery", limit=1, explain=True)
        assert result.trace is not None
    assert len(result.hits) == 1
    assert result.hits[0].id in {first.id, second.id}
    assert not result.trace.ambiguous
