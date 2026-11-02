"""Public SDK refusal-triggered recall contracts and independent quality grading."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType

import httpx2 as httpx
import pytest

from mindbridge import (
    AbstentionReason,
    AnswerPolicy,
    AnswerResult,
    EmbedTask,
    Memory,
    MemoryConfig,
    Modality,
    ModelInput,
    RetrievalMode,
    SearchHit,
)


@pytest.fixture
def quality(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).parents[2] / "benchmarks/evidence_investigation_quality.py"
    spec = importlib.util.spec_from_file_location("evidence_investigation_quality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


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
        config=MemoryConfig(
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


def test_quality_grades_reject_extra_ids_wrong_granularity_and_unsupported_amounts(
    quality: ModuleType,
) -> None:
    cases = {case.case_id: case for case in quality.CASES}
    assert quality.grade(cases["source_id_set"], AnswerResult("survey-photo-a,survey-photo-b"))
    assert not quality.grade(
        cases["source_id_set"], AnswerResult("survey-photo-a,survey-photo-b,invented-id")
    )
    assert not quality.grade(cases["date_granularity"], AnswerResult("2042-06-14T09:35:00Z"))
    assert quality.grade(cases["datetime_granularity"], AnswerResult("2042-07-08T16:25"))
    assert quality.grade(cases["refund_net"], AnswerResult("137.45 GBP"))
    assert not quality.grade(cases["refund_net"], AnswerResult("GBP 195.80"))
    assert not quality.grade(cases["missing_private_fact"], AnswerResult("Maybe 1234"))
    assert len([case for case in quality.CASES if case.split == "development"]) == 8
    assert len([case for case in quality.CASES if case.split == "holdout"]) == 4
    assert not any(
        re.fullmatch(r"\d{8}_\d{6}", doc.source_id)
        for case in quality.CASES
        for doc in case.documents
    )


def test_quality_wire_diagnostics_do_not_classify_sse_as_invalid_model_json(
    quality: ModuleType,
) -> None:
    response = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content="data: [DONE]\n\n"
    )
    assert quality._response_shape(response) == {
        "status_code": 200,
        "wire_format": "sse",
        "json_content_checked": False,
    }
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"draft":{"kind":"text","answer":"private"}}'},
                }
            ]
        },
    )
    shape = quality._response_shape(response)
    assert shape["kind"] == "text" and shape["finish_reason"] == "stop"
    assert "private" not in json.dumps(shape)
