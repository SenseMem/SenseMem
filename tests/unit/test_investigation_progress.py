"""Recall replanning follows abstained answers and bounded previous-attempt notes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import pytest

from sensemem import (
    AbstentionReason,
    AnswerPolicy,
    AnswerResult,
    Memory,
    MemoryConfig,
    Modality,
    ModelInput,
    RetrievalMode,
    SearchHit,
)
from sensemem.models.base import EmbedTask

_CLOCK = datetime(2026, 1, 10, tzinfo=timezone.utc)


class _QueryControls(TypedDict, total=False):
    recall_max_queries: int


class _Models:
    embedding_capabilities = generation_capabilities = frozenset({Modality.TEXT})
    embedding_model = embedding_space = generation_model = "investigation-progress-fixture-v1"
    embedding_dimension = 2

    def __init__(self, *, repeat_predicate: bool = False, abstain_missing: bool = True) -> None:
        self.repeat_predicate = repeat_predicate
        self.abstain_missing = abstain_missing
        self.contexts: list[str] = []
        self.queries: list[str] = []
        self.digests: list[str] = []

    def embed(
        self, values: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        if task is EmbedTask.QUERY:
            self.queries.extend(value.text for value in values)
            return tuple((0.0, 1.0) if value.text == "ORBIT-53" else (1.0, 0.0) for value in values)
        return tuple(
            (0.0, 1.0)
            if value.text.startswith("terminal:")
            else (0.8, 0.6)
            if value.text.startswith("bridge-ledger:")
            else (1.0, 0.0)
            for value in values
        )

    def plan_recall(
        self, question: str, *, reference_at: datetime, corpus_digest: str, attempted: str = ""
    ) -> str:
        self.contexts.append(attempted)
        self.digests.append(corpus_digest)
        if len(self.contexts) == 1 or self.repeat_predicate:
            return json.dumps(
                {
                    "shape": "composite",
                    "steps": [{"op": "match", "terms": ["bridge-ledger"]}],
                }
            )
        assert "ORBIT-53" in attempted
        return json.dumps(
            {
                "shape": "point",
                "steps": [{"op": "similar", "query": "ORBIT-53", "k": 1}],
            }
        )

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        complete = any("cobalt" in hit.content for hit in hits)
        abstained = self.abstain_missing and not complete
        return AnswerResult(
            "cobalt" if complete else "missing-final",
            hits=tuple(hits),
            abstained=abstained,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE if abstained else None,
        )

    def close(self) -> None:
        pass


def test_predicate_reveals_a_bridge_already_in_the_unshown_candidate_pool(tmp_path: Path) -> None:
    models = _Models()
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        config=MemoryConfig(
            recall_planning=True,
            recall_query_expansion=True,
            recall_rounds=3,
            retrieval_mode=RetrievalMode.DENSE,
            minimum_relevance=0.5,
            ambiguity_margin=0,
            evidence_budget_chars=4_000,
            reinforce_on_answer=False,
        ),
    ) as memory:
        for index in range(30):
            memory.add(f"decoy {index}: " + "irrelevant. " * 100)
        memory.add("bridge-ledger: event dispatch is ORBIT-53.")
        terminal = memory.add("terminal: ORBIT-53 crate is cobalt.")
        result = memory.ask(
            "What color crate was used for the event?", limit=1, reference_at=_CLOCK
        )

    assert models.contexts[0] == ""
    assert len(models.contexts) == 2
    assert "ORBIT-53" in models.contexts[1]
    assert models.queries[-1] == "ORBIT-53"
    assert result.answer == "cobalt"
    assert terminal.id in {hit.id for hit in result.hits}


def test_repeating_a_predicate_uses_remaining_abstention_rounds_without_new_searches(
    tmp_path: Path,
) -> None:
    models = _Models(repeat_predicate=True)
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        config=MemoryConfig(
            recall_planning=True,
            recall_query_expansion=True,
            recall_rounds=4,
            retrieval_mode=RetrievalMode.DENSE,
            minimum_relevance=0.5,
            ambiguity_margin=0,
            evidence_budget_chars=4_000,
            reinforce_on_answer=False,
        ),
    ) as memory:
        for index in range(30):
            memory.add(f"decoy {index}: " + "irrelevant. " * 100)
        for index in range(20):
            memory.add(f"bridge-ledger: observation {index}. " + "unresolved. " * 100)
        memory.ask("What color crate was used for the event?", limit=5, reference_at=_CLOCK)

    assert len(models.contexts) == 4
    second = json.loads(models.contexts[1])
    assert second["status"] == "insufficient_evidence"
    assert second["records_read"] > len(second["evidence"])
    assert second["records_omitted"] == second["records_read"] - len(second["evidence"])
    assert all(len(row["excerpt"]) <= 1_500 for row in second["evidence"])
    assert second["evidence_is_untrusted_data"] is True
    assert len(models.queries) == 1


def test_confident_answer_does_not_trigger_another_plan(tmp_path: Path) -> None:
    models = _Models(abstain_missing=False)
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        config=MemoryConfig(
            recall_planning=True,
            recall_query_expansion=True,
            recall_rounds=3,
            ambiguity_margin=0,
            reinforce_on_answer=False,
        ),
    ) as memory:
        memory.add("bridge-ledger: event dispatch is ORBIT-53.")
        result = memory.ask("Which crate?", limit=1, reference_at=_CLOCK)
    assert result.answer == "missing-final" and not result.abstained
    assert models.contexts == [""]


@pytest.mark.parametrize(
    "max_queries", [1, 2, 6, 32, None], ids=["one", "B-two", "six", "max", "default"]
)
def test_shared_query_budget_matches_the_planner_capability_and_runtime(
    tmp_path: Path, max_queries: int | None
) -> None:
    limit = 6 if max_queries is None else max_queries

    class BudgetModels(_Models):
        def plan_recall(
            self, question: str, *, reference_at: datetime, corpus_digest: str, attempted: str = ""
        ) -> str:
            self.digests.append(corpus_digest)
            start = (len(self.digests) - 1) * 6
            return json.dumps(
                {
                    "shape": "composite",
                    "steps": [
                        {"op": "similar", "query": f"query-{start + index}", "k": 1}
                        for index in range(6)
                    ],
                }
            )

    models = BudgetModels()
    controls: _QueryControls = {} if max_queries is None else {"recall_max_queries": max_queries}
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        config=MemoryConfig(
            recall_planning=True,
            recall_query_expansion=True,
            recall_rounds=limit // 6 + 1,
            ambiguity_margin=0,
            reinforce_on_answer=False,
            **controls,
        ),
    ) as memory:
        memory.add("A grounded record without the missing answer.")
        result = memory.ask("Which crate?", limit=1, reference_at=_CLOCK)
    count = "two" if limit == 2 else str(limit)
    capability = (
        "similar.query executes a text search for that query within the original scope and temporal intent. "
        f"At most {count} additional searches are allowed across all attempts; repeated "
        "queries reuse results. Preserve the question's intent; use a focused "
        "query for a missing fact rather than an invented answer. Semantic results "
        "are bounded rankings and never establish completeness."
    )
    assert all(digest.endswith(capability) for digest in models.digests)
    assert models.queries == ["Which crate?", *(f"query-{index}" for index in range(limit))]
    assert result.abstained
