"""Public SDK recall ordering, shared query budgets, and per-request caches."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest

from sensemem import (
    AbstentionReason,
    AnswerPolicy,
    AnswerResult,
    EmbedTask,
    Memory,
    MemorySettings,
    Modality,
    ModelInput,
    RetrievalMode,
    SearchHit,
)


class _InvestigatingModels:
    embedding_capabilities = frozenset({Modality.TEXT})
    generation_capabilities = frozenset({Modality.TEXT})
    embedding_model = "independent-investigation-contract"
    embedding_space = "independent-investigation-contract:2:v1"
    embedding_dimension = 2
    generation_model = "confident-reader-contract"

    def __init__(self, mode: str = "investigate") -> None:
        self.mode = mode
        self.order: list[str] = []
        self.contexts: list[str] = []
        self.queries: list[str] = []

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        if task is EmbedTask.QUERY:
            self.queries.extend(item.text for item in inputs)
        return tuple(
            (0.0, 1.0)
            if (task is EmbedTask.QUERY and item.text == "VELA-72")
            or (task is EmbedTask.DOCUMENT and item.text.startswith("terminal:"))
            else (1.0, 0.0)
            for item in inputs
        )

    def plan_recall(
        self,
        question: str,
        *,
        reference_at: datetime,
        corpus_digest: str,
        attempted: str = "",
    ) -> str:
        del reference_at, corpus_digest
        self.order.append("plan")
        self.contexts.append(attempted)
        if attempted:
            context = json.loads(attempted)
            assert context["evidence_is_untrusted_data"] is True
            assert context["evidence"] and context["evidence"][0]["excerpt"]
            assert "VELA-72" in attempted
        if self.mode == "no_progress":
            steps: list[dict[str, object]] = [{"op": "match", "terms": ["absent-proof"]}]
        elif self.mode == "ready" or not attempted:
            steps = [{"op": "similar", "query": question, "k": 1}]
        else:
            steps = [{"op": "similar", "query": "VELA-72", "k": 1}]
        return json.dumps(
            {
                "shape": "composite",
                "steps": steps,
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
        del question, answer_policy, exhaustive
        self.order.append("answer")
        complete = any("cobalt" in hit.content for hit in hits)
        abstained = self.mode == "repeat" or (not complete and self.mode == "investigate")
        return AnswerResult(
            "cobalt" if complete else "confident_wrong",
            hits=tuple(hits),
            abstained=abstained,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE if abstained else None,
        )

    def close(self) -> None:
        pass


def _memory(path: Path, models: _InvestigatingModels, *, queries: int = 3) -> Memory:
    memory = Memory(
        path,
        embedder=models,
        answerer=models,
        settings=MemorySettings(
            retrieval_mode=RetrievalMode.DENSE,
            minimum_relevance=0.5,
            recall_planning=True,
            recall_query_expansion=True,
            recall_rounds=4,
            recall_max_queries=queries,
            evidence_budget_chars=64000,
            reinforce_on_answer=False,
        ),
    )
    memory.add("anchor: the outing dispatch code is VELA-72.")
    memory.add("terminal: dispatch VELA-72 has the cobalt crate.")
    return memory


def test_planner_observes_bridge_only_after_an_abstained_answer(
    tmp_path: Path,
) -> None:
    models = _InvestigatingModels()
    with _memory(tmp_path, models) as memory:
        result = memory.ask("Which color crate was used for the outing?", limit=1)
    assert result.answer == "cobalt" and not result.abstained
    assert models.order == ["plan", "answer", "plan", "answer"]
    assert models.contexts[0] == ""
    second = json.loads(models.contexts[1])
    assert second["status"] == "insufficient_evidence"
    assert "VELA-72" in models.contexts[1] and "cobalt" not in models.contexts[1]


@pytest.mark.parametrize("mode", ["ready", "no_progress"])
def test_confident_answer_stops_without_spending_every_round(tmp_path: Path, mode: str) -> None:
    models = _InvestigatingModels(mode)
    with _memory(tmp_path, models) as memory:
        result = memory.ask("Which color crate was used for the outing?", limit=1)
    assert result.answer == "confident_wrong" and not result.abstained
    assert models.order == ["plan", "answer"]


def test_repeated_queries_share_one_budget_and_each_ask_starts_a_new_cache(tmp_path: Path) -> None:
    models = _InvestigatingModels("repeat")
    with _memory(tmp_path, models, queries=1) as memory:
        for _ in range(2):
            assert (
                memory.ask("Which color crate was used for the outing?", limit=1).answer == "cobalt"
            )
    assert models.order == ["plan", "answer"] * 8
    assert [index for index, note in enumerate(models.contexts) if not note] == [0, 4]
    assert models.queries.count("VELA-72") == 2
