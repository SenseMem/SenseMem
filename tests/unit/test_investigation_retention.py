"""Recall attempts preserve the ranked floor and execute the current predicate plan."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

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
        return json.dumps({"shape": "point", "steps": [{"op": "similar", "k": 1}]})

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


def test_point_plan_does_not_apply_predicate_row_caps_to_ranked_tail(
    tmp_path: Path,
) -> None:
    models = _Models()
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        config=MemoryConfig(
            recall_planning=True,
            retrieval_mode=RetrievalMode.DENSE,
            minimum_relevance=0,
            ambiguity_margin=0,
            evidence_budget_chars=10_000,
            recall_set_budget_chars=10_000,
            recall_set_max_rows=3,
            reinforce_on_answer=False,
        ),
    ) as memory:
        records = [memory.add(f"record {index}") for index in range(8)]
        result = memory.ask("List available records.", limit=1, reference_at=_CLOCK)
    assert {hit.id for hit in result.hits} == {record.id for record in records}
    assert len(models.delivered) == 8


def test_replanning_uses_the_current_predicate_without_accumulating_previous_predicate_rows(
    tmp_path: Path,
) -> None:
    class ChronologicalModels(_Models):
        def __init__(self) -> None:
            super().__init__()
            self.plan_calls = 0
            self.reads: list[tuple[SearchHit, ...]] = []

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
            self.question, self.delivered = question, tuple(hits)
            self.reads.append(tuple(hits))
            abstained = self.plan_calls == 1
            return AnswerResult(
                "Missing earlier event." if abstained else "Earlier event.",
                hits=tuple(hits),
                abstained=abstained,
                abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE if abstained else None,
            )

    models = ChronologicalModels()
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        config=MemoryConfig(
            recall_planning=True,
            recall_query_expansion=True,
            recall_rounds=2,
            retrieval_mode=RetrievalMode.DENSE,
            minimum_relevance=0.5,
            ambiguity_margin=0,
            evidence_budget_chars=10_000,
            reinforce_on_answer=False,
        ),
    ) as memory:
        memory.add("anchor", occurred_at=_CLOCK.replace(day=5))
        later = memory.add("later event", occurred_at=_CLOCK.replace(day=9))
        earlier = memory.add("earlier event", occurred_at=_CLOCK.replace(day=2))
        result = memory.ask("List the recorded events.", limit=1, reference_at=_CLOCK)
    assert models.plan_calls == 2
    assert later.id in {hit.id for hit in models.reads[0]}
    assert earlier.id in {hit.id for hit in result.hits}
    assert later.id not in {hit.id for hit in result.hits}
    assert "Recall program (set): records containing earlier (1)." in models.question.text
    assert "records containing later" not in models.question.text
