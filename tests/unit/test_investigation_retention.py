"""Investigation retention cannot change admitted evidence or its stated time order."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    Memory,
    Modality,
    ModelInput,
    RetrievalMode,
    SearchHit,
)
from mindbridge.models.base import EmbedTask

_CLOCK = datetime(2026, 1, 10, tzinfo=timezone.utc)


class _Models:
    embedding_capabilities = generation_capabilities = frozenset({Modality.TEXT})
    embedding_model = embedding_space = generation_model = "investigation-retention-fixture-v1"
    embedding_dimension = 4

    def __init__(self) -> None:
        self.delivered: tuple[SearchHit, ...] = ()
        self.question = ModelInput("Not yet asked.")

    def embed(
        self, values: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0, 0.0, 0.0) for _ in values)

    def plan_recall(
        self, question: str, *, reference_at: datetime, corpus_digest: str, attempted: str = ""
    ) -> str:
        return json.dumps({"shape": "point", "steps": [], "keep": ["R1"]})

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        self.question, self.delivered = question, tuple(hits)
        return AnswerResult(answer="Available records.", hits=tuple(hits))

    def close(self) -> None:
        pass


def test_keeping_a_floor_record_does_not_apply_predicate_row_caps_to_ranked_tail(
    tmp_path: Path,
) -> None:
    models = _Models()
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        recall_planning=True,
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0,
        ambiguity_margin=0,
        evidence_budget_chars=10_000,
        recall_set_budget_chars=10_000,
        recall_set_max_rows=3,
        reinforce_on_answer=False,
    ) as memory:
        records = [memory.add(f"record {index}") for index in range(8)]
        result = memory.ask("List available records.", limit=1, reference_at=_CLOCK)
    assert {hit.id for hit in result.hits} == {record.id for record in records}
    assert len(models.delivered) == 8


@pytest.mark.parametrize("retain_predicate", [False, True], ids=["no-keep", "keep-predicate"])
def test_cross_round_predicate_union_is_chronological_before_claiming_time_order(
    tmp_path: Path, retain_predicate: bool
) -> None:
    class ChronologicalModels(_Models):
        def __init__(self) -> None:
            super().__init__()
            self.plan_calls = 0

        def embed(
            self, values: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            return tuple(
                (1.0, 0.0, 0.0, 0.0)
                if task is EmbedTask.QUERY or value.text == "anchor"
                else (0.0, 1.0, 0.0, 0.0)
                for value in values
            )

        def plan_recall(
            self, question: str, *, reference_at: datetime, corpus_digest: str, attempted: str = ""
        ) -> str:
            self.plan_calls += 1
            return json.dumps(
                {
                    "shape": "set",
                    "steps": [
                        {"op": "match", "terms": ["later" if self.plan_calls == 1 else "earlier"]}
                    ],
                    "unresolved": ["Read the other event."] if self.plan_calls == 1 else [],
                    "keep": ["R2"] if retain_predicate and self.plan_calls == 2 else [],
                }
            )

    models = ChronologicalModels()
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        recall_planning=True,
        recall_rounds=2,
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0.5,
        ambiguity_margin=0,
        evidence_budget_chars=10_000,
        reinforce_on_answer=False,
    ) as memory:
        memory.add("anchor", occurred_at=_CLOCK.replace(day=5))
        later = memory.add("later event", occurred_at=_CLOCK.replace(day=9))
        earlier = memory.add("earlier event", occurred_at=_CLOCK.replace(day=2))
        result = memory.ask("List the recorded events.", limit=1, reference_at=_CLOCK)
    assert models.plan_calls == 2
    assert [hit.id for hit in result.hits if hit.id in {later.id, earlier.id}] == [
        earlier.id,
        later.id,
    ]
    assert "below, in corpus time order" in models.question.text
