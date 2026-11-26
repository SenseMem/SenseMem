"""PersonaMem-v3 task protocols pinned to upstream ad80a3b1b322."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from openai import AsyncOpenAI
from opentelemetry import trace

from sensemem import AsyncMemory
from sensemem.benchmarks import eval_answers, eval_judge, eval_metrics
from sensemem.benchmarks._official import personamem_v3_tasks as tasks
from sensemem.benchmarks.artifact_io import encode_jsonl_bytes
from sensemem.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask, load_task
from sensemem.benchmarks.eval_arms import Arm
from sensemem.benchmarks.eval_config import Arguments, JudgeConfig
from sensemem.benchmarks.eval_options import EvaluationOptions
from sensemem.benchmarks.eval_results import AnswerOutcome, SampleResult
from sensemem.benchmarks.eval_runner import _run_unit
from sensemem.benchmarks.official_scorers import (
    JudgePlan,
    combine_judge_scores,
    judge_plan,
    local_scores,
    parse_judge_response,
)
from sensemem.benchmarks.personamem_v3_results import complete_lifecycle, mistake_summary
from sensemem.benchmarks.task_catalog import TASKS

SUBSTANTIVE = (
    "Try an outdoor painting workshop with friends this weekend, exploring unfamiliar colors, "
    "materials, scenery, techniques, and creative approaches in a relaxed setting."
)


def _scores(prediction: str, task_type: str, evidence: Mapping[str, object]) -> dict[str, float]:
    return local_scores(
        "personamem-v3",
        score_kind="text",
        prediction=prediction,
        parsed_choice=None,
        expected_choice=None,
        references=("gold",),
        question="question",
        metadata={"task_type": task_type, "judge_evidence": evidence},
        evidence_source_ids=(),
    )


def _sample(
    question_id: str, task_type: str, evidence: Mapping[str, object], metrics: Mapping[str, float]
) -> SampleResult:
    return SampleResult(
        task="personamem-v3",
        benchmark="PersonaMem-v3",
        dataset_sha256="d" * 64,
        evaluation_sha256="e" * 64,
        unit_id="persona",
        question_id=question_id,
        prediction=SUBSTANTIVE,
        parsed_choice=None,
        score=metrics.get("personamem_score"),
        exact_match=None,
        latency_ms=1.0,
        confidence=1.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={"task_type": task_type, "task_family": "fixture", "judge_evidence": evidence},
        metrics=metrics,
    )


def test_released_unprompted_and_cluster_rows_keep_labels_out_of_prompts(tmp_path: Path) -> None:
    persona = tmp_path / "personamem-v3" / "backend" / "1"
    persona.mkdir(parents=True)
    (persona / "instagram.json").write_text(
        json.dumps([{"source_object_id": "event", "source_timestamp": 1}])
    )
    # Ground-truth profiles are never loaded as memory.
    (persona / "profile.json").write_text("not a memory input")
    instances = [
        (
            "new_suggestions_chatbot",
            "",
            {
                "trigger_kind": "post_fatigue",
                "fatigued_pref": "SCORER-SECRET",
                "gold_topic": "SCORER-SECRET",
            },
        ),
        (
            "active_mistake_prevention",
            "",
            {
                "polarity": "warn",
                "expected_warning_frame": {"must_mention": ["SCORER-SECRET"]},
            },
        ),
        (
            "short_vs_long_term_lifecycle",
            "lifecycle ranking",
            {
                "phase": "post",
                "canonical_id": "SCORER-SECRET",
                "stop_condition_description": "SCORER-SECRET",
                "matching_indices": [0],
                "candidates": [{"title": "Public option", "hashtags": ["travel"]}],
                "geo_context": {"city_at_t_test": "Denver"},
                "calendar_context": [{"title": "A public meeting", "start_ts": 3}],
            },
        ),
        (
            "over_personalization_repetition_chatbot",
            "A normal question",
            {
                "cluster_id": "cluster",
                "target_pref": "SCORER-SECRET",
                "queries": [
                    {
                        "ts": 2,
                        "user_query": "A normal question",
                        "anchor_index": 0,
                        "natural_anchor": "SCORER-SECRET",
                        "is_target": True,
                    }
                ],
            },
        ),
    ]
    (persona / "test.json").write_text(
        json.dumps(
            [
                {
                    "query_id": str(index),
                    "task_family": "fixture",
                    "task_type": kind,
                    "ts": index + 2,
                    "user_query": query,
                    "instance_full": evidence,
                }
                for index, (kind, query, evidence) in enumerate(instances)
            ]
        )
    )
    task = load_task(TASKS["personamem-v3"], root=tmp_path, verify_digest=False)
    questions = task.units[0].questions
    assert len(questions) == 4
    assert all(
        "SCORER-SECRET" not in "".join(str(part) for part in question.content)
        for question in questions
    )
    assert "Denver" in str(questions[2].content)
    assert "A public meeting" in str(questions[2].content)
    assert questions[3].metadata["cluster_queries"] == (
        {
            "ts": 2,
            "user_query": "A normal question",
            "is_target": True,
        },
    )
    encode_jsonl_bytes(dict(question.metadata) for question in questions)


@pytest.mark.parametrize(
    ("tags", "failed"),
    [
        (["tired", "fresh"], True),
        (["recent1", "recent2", "fresh", "new"], True),
        (["recent1", "fresh", "new"], False),
        ([], False),
    ],
)
def test_novel_suggestion_reuses_the_official_overlap_threshold(
    tags: list[str], failed: bool
) -> None:
    scores = _scores(
        json.dumps({"recommendation": SUBSTANTIVE, "hashtags": tags}),
        "new_suggestions_chatbot",
        {"fatigued_hashtags": ["#Tired"], "leak_set_hashtags": ["recent1", "recent2"]},
    )
    assert bool(scores["hard_fail"]) is failed
    assert ("personamem_score" in scores) is failed
    if failed:
        assert scores["personamem_score"] == 0.0


@pytest.mark.parametrize(
    ("alignment", "hard_fail", "expected"),
    [(3, False, 1), (2, False, 1), (1, False, 0), (3, True, 0)],
)
def test_novel_suggestion_requires_alignment_and_no_judged_hard_failure(
    alignment: int, hard_fail: bool, expected: int
) -> None:
    plan = judge_plan(
        "personamem-v3",
        question="Suggest something",
        references=("gold",),
        prediction=json.dumps({"recommendation": SUBSTANTIVE, "hashtags": ["fresh"]}),
        metadata={
            "task_type": "new_suggestions_chatbot",
            "judge_evidence": {
                "gold_topic": "art",
                "gold_hashtags": ["painting"],
                "fatigued_hashtags": ["running"],
                "leak_set_hashtags": [],
            },
        },
    )
    assert plan is not None
    scores = parse_judge_response(
        plan, json.dumps({"alignment_score": alignment, "hard_fail": hard_fail})
    )
    assert scores["personamem_score"] == expected
    assert "gold_topic: 'art'" in plan.calls[0][0].content


@pytest.mark.parametrize(
    ("response", "polarity", "expected"),
    [
        ("Heads up: your airport is Haneda, not Narita.", "warn", 1),
        ("Heads up: enjoy your flight.", "warn", 0),
        (SUBSTANTIVE, "foil", 1),
        ("Fine.", "foil", 0),
        ("Heads up: " + SUBSTANTIVE, "foil", 0),
    ],
)
def test_mistake_prevention_keeps_warn_coverage_and_substantive_foil_gate(
    response: str, polarity: str, expected: int
) -> None:
    scores = _scores(
        json.dumps({"response": response, "warned": False}),
        "active_mistake_prevention",
        {
            "polarity": polarity,
            "expected_warning_frame": {
                "must_mention": ["Haneda", "Narita"],
                "must_not_mention": ["private-secret"],
            },
        },
    )
    assert scores["correct"] == scores["personamem_score"] == expected


def test_lifecycle_pairs_are_isolated_by_persona_and_map_signed_scores() -> None:
    def row(name: str, phase: str, rate: float) -> SampleResult:
        return _sample(
            name,
            "short_vs_long_term_lifecycle",
            {
                "canonical_id": "trip",
                "phase": phase,
            },
            {"match_rate_at_3": rate},
        )

    before, after = row("pre", "pre", 1.0), row("post", "post", 0.0)
    other = replace(after, unit_id="other-persona")
    assert all(sample.score is None for sample in complete_lifecycle((before, other)))
    scored = complete_lifecycle((before, after))
    assert [sample.score for sample in scored] == [1.0, 1.0]
    assert scored[0].metrics["lifecycle_score"] == 1.0
    neutral = complete_lifecycle((before, row("post", "post", 1.0)))
    assert neutral[0].score == 0.5
    backwards = complete_lifecycle((row("pre", "pre", 0.0), row("post", "post", 1.0)))
    assert backwards[0].score == 0.0


def test_warning_pairs_have_separate_micro_and_pair_denominators() -> None:
    rows = tuple(
        _sample(
            str(index),
            "active_mistake_prevention",
            {"pair_id": pair, "polarity": polarity},
            {
                "correct_warn": float(correct and polarity == "warn"),
                "correct_foil": float(correct and polarity == "foil"),
                "personamem_score": float(correct),
            },
        )
        for index, (pair, polarity, correct) in enumerate(
            [
                ("one", "warn", True),
                ("one", "foil", False),
                ("two", "warn", True),
            ]
        )
    )
    summary = mistake_summary(rows)
    assert summary["n_pairs"] == 2
    assert summary["complete_pairs"] == summary["incomplete_pairs"] == 1
    assert summary["paired_correct"] == 0.0
    assert summary["warn_recall"] == 1.0
    assert summary["foil_precision"] == summary["paired_f1"] == 0.0


def _cluster_question(name: str = "cluster-row") -> EvalQuestion:
    return EvalQuestion(
        name,
        ("question",),
        ("gold",),
        cutoff_seconds=1.0,
        metadata={
            "task_type": "over_personalization_repetition_chatbot",
            "cluster_id": "cluster",
            "n_allowed_repetitions": 0,
            "cluster_queries": tuple(
                {"ts": index + 1, "user_query": f"query-{index}", "is_target": index != 1}
                for index in range(4)
            ),
            "judge_evidence": {"target_pref": "SCORER-SECRET"},
        },
        source_question="question",
    )


async def test_cluster_threads_filler_answers_without_disclosing_hidden_evidence() -> None:
    prompts = []

    async def request(question: EvalQuestion) -> AnswerOutcome:
        prompts.append("\n".join(str(part) for part in question.content))
        return AnswerOutcome(
            json.dumps({"response": f"reply-{len(prompts)} {SUBSTANTIVE}"}), 3.0, 1.0, (), ()
        )

    outcome = await eval_answers._answer_personamem_cluster(_cluster_question(), request)
    responses = tasks.cluster_responses(outcome.prediction)
    assert len(prompts) == len(responses) == 4
    assert "reply-1" not in prompts[0]
    assert "reply-1" in prompts[1]
    assert "reply-2" in prompts[2]
    assert not any("SCORER-SECRET" in prompt for prompt in prompts)
    assert [row["is_target"] for row in responses] == [True, False, True, True]
    assert outcome.latency_ms == 12.0


def test_cluster_ignores_fillers_and_preserves_failed_judge_positions() -> None:
    prediction = json.dumps(
        {
            "responses": [
                tasks.cluster_response(
                    SUBSTANTIVE,
                    user_query=f"query-{index}",
                    timestamp=index,
                    is_target=index != 1,
                )
                for index in range(5)
            ]
        }
    )
    plan = judge_plan(
        "personamem-v3",
        question="question",
        references=("gold",),
        prediction=prediction,
        metadata=_cluster_question().metadata,
    )
    assert plan is not None
    assert len(plan.calls) == 4
    scores = combine_judge_scores(
        plan, ({"restraint_score": 0.0}, {}, {"restraint_score": 8.0}, {"restraint_score": 10.0})
    )
    assert scores["n_warmup"] == scores["n_judge_failed"] == 1
    assert scores["n_monitored"] == 2
    assert scores["query_score_0_10"] == 9.0
    assert scores["personamem_score"] == 0.9


async def test_failed_repetition_judgment_is_reported_without_shifting_the_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = _cluster_question()
    prediction = json.dumps(
        {
            "responses": [
                tasks.cluster_response(
                    SUBSTANTIVE,
                    user_query="question",
                    timestamp=index,
                    is_target=True,
                )
                for index in range(3)
            ]
        }
    )
    sample = replace(
        _sample("one", "over_personalization_repetition_chatbot", {}, {}),
        prediction=prediction,
        metadata=question.metadata,
    )
    plan = judge_plan(
        "personamem-v3",
        question="question",
        references=("gold",),
        prediction=prediction,
        metadata=question.metadata,
    )
    assert plan is not None

    async def judge_call(*_args: object, **kwargs: object) -> tuple[Mapping[str, float], str, bool]:
        index = cast(int, kwargs["call_index"])
        if index == 0:
            raise ValueError("unparseable judgment")
        score = 8.0 if index == 1 else 10.0
        return {"restraint_score": score}, json.dumps({"restraint_score": score}), False

    monkeypatch.setattr(eval_judge, "_traced_judge_call", judge_call)
    result = await eval_judge._judge_sample(
        sample,
        plan,
        client=cast(AsyncOpenAI, object()),
        cache=None,
        semaphore=asyncio.Semaphore(1),
        config=JudgeConfig("gpt-5.5", "http://judge.test/v1", api_key="test"),
        log_samples=False,
        tracer=trace.get_tracer(__name__),
    )
    assert result.score == 0.9
    assert result.metrics["n_judge_failed"] == 1
    assert result.metrics["n_monitored"] == 2
    assert result.error_code == "JudgeError"
    assert result.scorer_error == "PersonaMem-v3 repetition judgments are incomplete"


async def test_cluster_generation_is_shared_across_cutoffs_but_isolated_per_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def guarded(
        _memory: AsyncMemory, _question: EvalQuestion, **kwargs: object
    ) -> AnswerOutcome:
        calls.append(cast(Arm, kwargs["arm"]).name)
        await asyncio.sleep(0)
        return AnswerOutcome('{"responses": []}', 1.0, 1.0, (), ())

    class MemoryContext:
        async def __aenter__(self) -> AsyncMemory:
            return cast(AsyncMemory, object())

        async def __aexit__(self, *_error: object) -> None:
            return None

    def memory_factory(_path: Path) -> MemoryContext:
        return MemoryContext()

    monkeypatch.setattr(eval_answers, "_guarded_answer", guarded)
    questions = (_cluster_question("one"), replace(_cluster_question("two"), cutoff_seconds=2.0))
    unit = EvalUnit("persona", (), questions)
    task = LoadedTask(TASKS["personamem-v3"], tmp_path, "d" * 64, (unit,))
    samples = await _run_unit(
        task,
        unit,
        tmp_path / "store",
        memory_factory=memory_factory,
        options=EvaluationOptions(predict_only=True),
        request_semaphore=asyncio.Semaphore(4),
        response_cache=None,
        arms=(Arm("blind"), Arm("full-context")),
    )
    assert calls == ["blind", "full-context"]
    assert len(samples) == 4
    assert samples[0].prediction == samples[1].prediction
    encode_jsonl_bytes(sample.json() for sample in samples)


async def test_cluster_judging_is_shared_across_published_sibling_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    question = _cluster_question()
    prediction = json.dumps(
        {
            "responses": [
                tasks.cluster_response(
                    SUBSTANTIVE,
                    user_query="question",
                    timestamp=index,
                    is_target=True,
                )
                for index in range(3)
            ]
        }
    )
    samples = tuple(
        replace(
            _sample(name, "over_personalization_repetition_chatbot", {}, {}),
            prediction=prediction,
            metadata=question.metadata,
        )
        for name in ("one", "two")
    )
    questions = tuple(replace(question, question_id=sample.question_id) for sample in samples)
    task = LoadedTask(
        TASKS["personamem-v3"], tmp_path, "d" * 64, (EvalUnit("persona", (), questions),)
    )

    async def judge(
        sample: SampleResult, plan: JudgePlan | None, **_kwargs: object
    ) -> SampleResult:
        if plan is None:
            return sample
        calls.append(sample.sample_id)
        return replace(sample, score=0.8, metrics={"personamem_score": 0.8}, judge_model="gpt-5.5")

    monkeypatch.setattr(eval_judge, "_judge_sample", judge)
    arguments = cast(Arguments, SimpleNamespace(quiet=True, use_cache=None, log_samples=False))
    results = await eval_judge.apply_judges(
        (task,),
        samples,
        arguments=arguments,
        config=JudgeConfig("gpt-5.5", "http://judge.test/v1", api_key="test"),
    )
    assert len(calls) == 1
    assert [sample.score for sample in results] == [0.8, 0.8]


def test_complete_micro_counts_lifecycle_pairs_once_and_requires_full_coverage(
    tmp_path: Path,
) -> None:
    rows: Sequence[SampleResult] = (
        _sample(
            "pre",
            "short_vs_long_term_lifecycle",
            {"canonical_id": "trip", "phase": "pre"},
            {"match_rate_at_3": 1.0},
        ),
        _sample(
            "post",
            "short_vs_long_term_lifecycle",
            {"canonical_id": "trip", "phase": "post"},
            {"match_rate_at_3": 0.0},
        ),
        _sample(
            "other",
            "active_mistake_prevention",
            {"pair_id": "warn", "polarity": "warn"},
            {"personamem_score": 0.0},
        ),
    )
    unit = EvalUnit("persona", (), (EvalQuestion("q", ("question",), ("gold",)),))
    task = LoadedTask(TASKS["personamem-v3"], tmp_path, "d" * 64, (unit,))
    arguments = cast(
        Arguments,
        SimpleNamespace(
            seed=7, bootstrap_samples=32, recall_limit=20, blind=False, offset=0, limit=None
        ),
    )
    result = eval_metrics.compute_task_metrics(task, rows, arguments)
    assert cast(Mapping[str, object], result["score"])["mean"] == 50.0
    assert cast(Mapping[str, object], result["score_coverage"])["complete"] is True
    missing = eval_metrics.compute_task_metrics(task, rows[1:], arguments)
    assert cast(Mapping[str, object], missing["score"])["mean"] == 0.0
    coverage = cast(Mapping[str, object], missing["score_coverage"])
    assert coverage["complete"] is True
    assert coverage["excluded_question_count"] == 1
