"""AMA-Bench release, evaluator and public leaderboard protocol fixtures."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI

from sensemem.benchmarks import ama_bench, eval_judge
from sensemem.benchmarks._official import ama_scoring
from sensemem.benchmarks.eval_adapters import load_task
from sensemem.benchmarks.eval_config import _Arguments, _JudgeConfig
from sensemem.benchmarks.eval_metrics import _metrics
from sensemem.benchmarks.eval_results import SampleResult
from sensemem.benchmarks.official_scorers import (
    judge_model_is_official,
    judge_plan,
    local_scores,
    parse_judge_response,
    task_primary_metric,
)
from sensemem.benchmarks.task_catalog import TASKS, expand


def _episode(episode_id: int = 0, **overrides: object) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "task": "Find a blue key",
        "task_type": "fixture-task",
        "domain": "Game",
        "success": True,
        "num_turns": 2,
        "total_tokens": 10,
        "trajectory": [
            {"turn_idx": 0, "action": None, "observation": "The key is blue."},
            {"turn_idx": 1, "action": "take key", "observation": None},
        ],
        "qa_pairs": [
            {
                "question_uuid": f"q-{episode_id}",
                "question": "Which key? " + "Use the recorded trajectory. " * 20,
                "answer": "private-reference",
                "type": "C",
            }
        ],
        **overrides,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sample(
    *,
    unit_id: str = "0",
    question_id: str = "q-0",
    dataset_sha256: str = ama_bench.AMA_DATA_SHA256,
    score: float = 1.0,
    metrics: Mapping[str, float] | None = None,
    error_code: str | None = None,
) -> SampleResult:
    return SampleResult(
        task="ama-bench",
        benchmark="AMA-Bench",
        dataset_sha256=dataset_sha256,
        evaluation_sha256="evaluation",
        unit_id=unit_id,
        question_id=question_id,
        prediction="Answer[1]: A blue key.",
        parsed_choice=None,
        score=score,
        exact_match=1.0,
        latency_ms=1.0,
        confidence=1.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=error_code,
        metadata={
            "domain": "Game",
            "task_type": "fixture-task",
            "qa_type": "C",
            "episode_id": "0",
            "task_description": "Find a blue key",
        },
        metrics=metrics if metrics is not None else {"llm_judge_score": 1.0, "accuracy": 1.0},
        judge_model="Qwen/Qwen3-32B",
        scorer_protocol=ama_scoring.SCORER_VERSION,
    )


def test_catalog_loader_preserves_trajectory_and_keeps_gold_private(tmp_path: Path) -> None:
    spec = TASKS["ama-bench"]
    dataset = spec.dataset_path(tmp_path)
    _write(dataset, [_episode(), _episode(1)])
    task = load_task(spec, root=tmp_path, limit=1, offset=1, verify_digest=False)
    assert spec.revision == ama_bench.AMA_DATA_REVISION
    assert spec.digest == ama_bench.AMA_DATA_SHA256
    assert spec.adapter_version == ama_bench.AMA_ADAPTER_VERSION
    assert expand(("ama-bench",)) == ("ama-bench",)
    assert len(task.units) == 1
    unit = task.units[0]
    assert unit.unit_id == "1"
    assert tuple(m.content for m in unit.memories) == (
        ("Task: Find a blue key",),
        ("Step 0:\nAction: None\nObservation: The key is blue.",),
        ("Step 1:\nAction: take key\nObservation: None",),
    )
    question = unit.questions[0]
    assert question.question_id == "q-1"
    assert question.references == ("private-reference",)
    assert question.source_question is not None and len(question.source_question) > 255
    assert question.metadata["domain"] == "Game"
    assert question.metadata["qa_type"] == "C"
    assert str(question.content[0]).endswith("Answer[1]: [your answer here]")
    visible = repr([m.content for m in unit.memories]) + repr(question.content)
    assert "private-reference" not in visible
    assert "success" not in visible
    assert "qa_type" not in visible


def test_large_observation_is_split_without_evidence_loss(tmp_path: Path) -> None:
    spec = TASKS["ama-bench"]
    _write(
        spec.dataset_path(tmp_path),
        [_episode(trajectory=[{"turn_idx": 0, "action": None, "observation": "x" * 100_000}])],
    )
    task = load_task(spec, root=tmp_path, verify_digest=False)
    blocks = [m for m in task.units[0].memories if m.source_id.startswith("step-0")]
    assert len(blocks) > 1
    assert sum(str(m.content[0]).count("x") for m in blocks) == 100_000


@pytest.mark.parametrize("failure", ["episode", "uuid", "capability", "extra", "digest"])
def test_invalid_release_fails(tmp_path: Path, failure: str) -> None:
    spec = TASKS["ama-bench"]
    rows = [_episode(), _episode(1)]
    if failure == "episode":
        rows[1] = _episode()
    elif failure == "uuid":
        rows[1]["qa_pairs"] = rows[0]["qa_pairs"]
    elif failure == "capability":
        rows[0]["qa_pairs"] = [{"question": "q", "answer": "a", "type": "E", "question_uuid": "q"}]
    elif failure == "extra":
        rows[0]["gold_context"] = "must not enter memory"
    _write(spec.dataset_path(tmp_path), rows)
    with pytest.raises(ValueError):
        load_task(spec, root=tmp_path, verify_digest=failure == "digest")


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("<think>no</think>\nAnswer[1]: first\nsecond", "first\nsecond"),
        ("reason\n###Answer: blue", "blue"),
        ("<THINK>keep this</THINK> blue", "<THINK>keep this</THINK> blue"),
        ("plain answer", "plain answer"),
        ("Answer[1]: ", ""),
    ],
)
def test_answer_extraction_matches_upstream_markers(prediction: str, expected: str) -> None:
    assert ama_scoring.extract_answer(prediction) == expected


@pytest.mark.parametrize(
    ("prediction", "reference", "em", "f1"),
    [
        ("The, BLUE-key!", "a blue key", 1.0, 1.0),
        ("red red blue", "red blue blue", 0.0, 2 / 3),
        ("the", "", 1.0, 1.0),
        ("", "blue", 0.0, 0.0),
        ("café_1", "CAFÉ_1", 1.0, 1.0),
    ],
)
def test_normalization_and_counter_f1(
    prediction: str, reference: str, em: float, f1: float
) -> None:
    result = local_scores(
        "ama-bench",
        score_kind="text",
        prediction=prediction,
        parsed_choice=None,
        expected_choice=None,
        references=(reference,),
        question="q",
        metadata={},
        evidence_source_ids=(),
    )
    assert result == {"exact_match": em, "token_f1": pytest.approx(f1)}


@pytest.mark.parametrize(
    ("response", "score"),
    [("<think>no</think>yes", 1.0), ("YES ... no", 0.0), ("no, YES", 1.0), ("yesterday", 2 / 3)],
)
def test_judge_last_verdict_and_fractional_fallback(response: str, score: float) -> None:
    plan = judge_plan(
        "ama-bench",
        question="Which key?",
        references=("blue key",),
        prediction="Answer[1]: blue",
        metadata=_sample().metadata,
    )
    assert plan is not None
    assert plan.max_tokens == 2048
    assert plan.extra_body is None
    assert len(plan.calls) == len(plan.calls[0]) == 1
    assert plan.calls[0][0].role == "user"
    assert (
        "Task Type: fixture-task\nEpisode ID: 0\nTask Context: Find a blue key"
        in plan.calls[0][0].content
    )
    assert "Predicted Answer: blue\n" in plan.calls[0][0].content
    assert plan.calls[0][0].content.endswith("Answer:<think></think>")
    assert parse_judge_response(plan, response) == {
        "llm_judge_score": pytest.approx(score),
        "accuracy": float(score == 1.0),
    }
    assert task_primary_metric("ama-bench") == "accuracy"
    assert judge_model_is_official("ama-bench", "Qwen/Qwen3-32B")
    assert not judge_model_is_official("ama-bench", "Qwen/Qwen3-30B-A3B")


async def test_judge_transport_keeps_official_qwen_sampling_and_thinking() -> None:
    requests: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "Qwen/Qwen3-32B",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "yes"}}],
            },
        )

    plan = judge_plan(
        "ama-bench",
        question="q",
        references=("blue",),
        prediction="blue",
        metadata=_sample().metadata,
    )
    assert plan is not None
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply))
    ) as client:
        scores, _, cached = await eval_judge._judge_call(
            client,
            plan.calls[0],
            sample=_sample(),
            plan=plan,
            call_index=0,
            cache=None,
            semaphore=asyncio.Semaphore(1),
            config=_JudgeConfig("Qwen/Qwen3-32B", "http://judge.invalid/v1"),
        )
    assert not cached
    assert scores["accuracy"] == 1.0
    assert requests[0]["max_tokens"] == 2048
    assert requests[0]["temperature"] == 0.0
    assert "chat_template_kwargs" not in requests[0]
    assert "enable_thinking" not in requests[0]
    assert requests[0]["messages"] == [{"role": "user", "content": plan.calls[0][0].content}]


def test_evaluator_summary_distinguishes_fallback_average_from_accuracy() -> None:
    samples = (
        _sample(),
        _sample(question_id="q-1", score=0.0, metrics={"accuracy": 0.0, "llm_judge_score": 0.5}),
        _sample(question_id="q-2", error_code="JudgeError", metrics={"llm_judge_score": 0.0}),
    )
    result = ama_bench.result_summary(samples, full_dataset_selected=False)
    evaluation = cast(dict[str, object], result["official_evaluation"])
    assert evaluation["overall"] == {"total_questions": 2, "avg_score": 0.75, "accuracy": 0.5}
    assert evaluation["by_domain"] == {"Game": {"count": 2, "avg_score": 0.75, "accuracy": 0.5}}
    board = cast(dict[str, object], result["leaderboard"])
    assert board["complete"] is False
    assert board["verified"] is False
    assert board["scored_questions"] == 2


def test_unified_metrics_include_official_summary_and_judge_provenance(tmp_path: Path) -> None:
    spec = TASKS["ama-bench"]
    _write(spec.dataset_path(tmp_path), [_episode()])
    task = load_task(spec, root=tmp_path, verify_digest=False)
    arguments = cast(
        _Arguments,
        SimpleNamespace(
            seed=7, bootstrap_samples=20, recall_limit=20, blind=False, offset=0, limit=None
        ),
    )
    result = _metrics(task, (_sample(),), arguments)
    assert result["primary_metric"] == "accuracy"
    assert result["official_metric"] is True
    assert result["judge_model_official"] is True
    assert cast(dict[str, object], result["official_evaluation"])["overall"] == {
        "total_questions": 1,
        "avg_score": 1.0,
        "accuracy": 1.0,
    }
    assert cast(dict[str, object], result["leaderboard"])["complete"] is False
    proxy = _metrics(task, (replace(_sample(), judge_model="another-judge"),), arguments)
    assert proxy["official_metric"] is False


def test_public_board_retains_four_decimal_cells_and_published_weights() -> None:
    # One of three GAME/C answers is correct. The evaluator mean is 1/3;
    # the board first rounds the cell to .3333, then uses the published weights.
    scores = ama_bench.leaderboard_scores(
        [("Game", "C", 1.0), ("Game", "C", 0.0), ("Game", "C", 0.5)]
    )
    assert cast(dict[str, object], scores["Score"])["GAME"] == [
        {"A": 0.0},
        {"B": 0.0},
        {"C": 0.3333},
        {"D": 0.0},
    ]
    assert cast(dict[str, float], scores["domain_accuracy"])["GAME"] == 0.3333 * 0.25
    expected_c = (
        0.3333 * 0.036058 / (0.053686 + 0.029247 + 0.037260 + 0.036058 + 0.060096 + 0.042868)
    )
    assert cast(dict[str, float], scores["capability_accuracy"])["C"] == pytest.approx(expected_c)
    assert scores["avg_f1"] == scores["avg_accuracy"]
    assert scores["domain_f1"] == scores["domain_accuracy"]
    assert ama_bench.leaderboard_scores([])["avg_accuracy"] == 0.0


def test_export_requires_pinned_complete_successful_judged_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, samples, output = (
        tmp_path / name for name in ("dataset.jsonl", "samples.jsonl", "submission.jsonl")
    )
    _write(dataset, [_episode(), _episode(1)])
    with pytest.raises(ValueError, match="pinned official dataset"):
        ama_bench.export_submission(dataset, samples, output)
    # The small fixture replaces only the release digest; coverage still comes
    # from every question in that release, rather than from the sample count.
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    monkeypatch.setattr(ama_bench, "AMA_DATA_SHA256", digest)
    rows = [
        _sample(dataset_sha256=digest).json(),
        _sample(
            unit_id="1", question_id="q-1", dataset_sha256=digest, metrics={"llm_judge_score": 0.5}
        ).json(),
    ]
    _write(samples, rows)
    assert (
        ama_bench.main(
            [
                "export",
                "--dataset",
                str(dataset),
                "--samples",
                str(samples),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["verified"] is False
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {
            "episode_id": "0",
            "question_uuid_list": ["q-0"],
            "answer_list": ["A blue key."],
            "llm_as_judge_score_list": [True],
        },
        {
            "episode_id": "1",
            "question_uuid_list": ["q-1"],
            "answer_list": ["A blue key."],
            "llm_as_judge_score_list": [False],
        },
    ]
    for bad in (
        rows[:1],
        rows + rows[:1],
        [{**rows[0], "error_code": "JudgeError"}, rows[1]],
        [{**rows[0], "ingest_failure_count": 1}, rows[1]],
        [{**rows[0], "metrics": {}}, rows[1]],
        [{**rows[0], "dataset_sha256": "different"}, rows[1]],
        [{**rows[0], "judge_model": "proxy-judge"}, rows[1]],
        [{**rows[0], "scorer_protocol": "wrong-protocol"}, rows[1]],
    ):
        _write(samples, bad)
        with pytest.raises(ValueError):
            ama_bench.export_submission(dataset, samples, tmp_path / "bad.jsonl")
