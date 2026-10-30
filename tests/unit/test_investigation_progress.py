"""Investigation progress follows observations rather than candidate-pool growth."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

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
    embedding_model = embedding_space = generation_model = "investigation-progress-fixture-v1"
    embedding_dimension = 2

    def __init__(self, *, repeat_predicate: bool = False) -> None:
        self.repeat_predicate = repeat_predicate
        self.contexts: list[str] = []
        self.queries: list[str] = []

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
        if len(self.contexts) == 1 or self.repeat_predicate:
            return json.dumps(
                {
                    "shape": "composite",
                    "steps": [{"op": "match", "terms": ["bridge-ledger"]}],
                    "unresolved": ["Read the dispatch named in bridge-ledger."],
                }
            )
        assert "ORBIT-53" in attempted
        return json.dumps(
            {
                "shape": "point",
                "steps": [{"op": "similar", "query": "ORBIT-53", "k": 1}],
                "unresolved": [],
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
        return AnswerResult("cobalt" if complete else "missing-final", hits=tuple(hits))

    def close(self) -> None:
        pass


def test_predicate_reveals_a_bridge_already_in_the_unshown_candidate_pool(tmp_path: Path) -> None:
    models = _Models()
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        recall_planning=True,
        recall_query_expansion=True,
        recall_rounds=3,
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0.5,
        ambiguity_margin=0,
        evidence_budget_chars=4_000,
        reinforce_on_answer=False,
    ) as memory:
        for index in range(30):
            memory.add(f"decoy {index}: " + "irrelevant. " * 100)
        memory.add("bridge-ledger: event dispatch is ORBIT-53.")
        terminal = memory.add("terminal: ORBIT-53 crate is cobalt.")
        result = memory.ask(
            "What color crate was used for the event?", limit=1, reference_at=_CLOCK
        )

    first = json.loads(models.contexts[0])
    assert first["records_available"] == 31
    assert first["records_omitted"] > 0
    assert "ORBIT-53" not in models.contexts[0]
    assert len(models.contexts) == 2
    assert "ORBIT-53" in models.contexts[1]
    assert models.queries[-1] == "ORBIT-53"
    assert result.answer == "cobalt"
    assert terminal.id in {hit.id for hit in result.hits}


def test_repeating_a_predicate_stops_when_its_rows_still_exceed_the_observation_budget(
    tmp_path: Path,
) -> None:
    models = _Models(repeat_predicate=True)
    with Memory(
        tmp_path,
        embedder=models,
        answerer=models,
        recall_planning=True,
        recall_query_expansion=True,
        recall_rounds=4,
        retrieval_mode=RetrievalMode.DENSE,
        minimum_relevance=0.5,
        ambiguity_margin=0,
        evidence_budget_chars=4_000,
        reinforce_on_answer=False,
    ) as memory:
        for index in range(30):
            memory.add(f"decoy {index}: " + "irrelevant. " * 100)
        for index in range(20):
            memory.add(f"bridge-ledger: observation {index}. " + "unresolved. " * 100)
        memory.ask("What color crate was used for the event?", limit=5, reference_at=_CLOCK)

    assert len(models.contexts) == 2
    second = json.loads(models.contexts[1])
    assert second["previous_reads"][0]["rows"] == 20
    assert sum("bridge-ledger" in json.dumps(row) for row in second["evidence"]) < 20
    assert len(models.queries) == 1
