"""Conversion of observed and cached outcomes into scored evaluation samples."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import cast

from sensemem import (
    AnswerPolicy,
)
from sensemem.benchmarks.eval_adapters import (
    EvalQuestion,
    EvalUnit,
    LoadedTask,
)
from sensemem.benchmarks.eval_arms import (
    PRODUCT_ARM,
    Arm,
    answers_through_ask,
    requests_answer_policy,
)
from sensemem.benchmarks.eval_cache import CachedAnswer, ResponseCache
from sensemem.benchmarks.eval_evidence import EvidenceInterval
from sensemem.benchmarks.eval_inputs import cache_task, memory_end
from sensemem.benchmarks.eval_results import (
    AnswerOutcome,
    FailureDetail,
    SampleResult,
    failure_detail,
)
from sensemem.benchmarks.eval_statistics import parse_choice
from sensemem.benchmarks.official_scorers import (
    finalize_scores,
    local_scores,
    retrieval_gold_ids,
    sample_primary_metric,
    scorer_protocol,
)
from sensemem.benchmarks.prompts import task_answer_policy


def cached_results(
    cache: ResponseCache | None,
    task: LoadedTask,
    unit: EvalUnit,
    *,
    arm: Arm = PRODUCT_ARM,
    predict_only: bool,
    log_samples: bool,
    answer_policy: AnswerPolicy | None = None,
) -> dict[tuple[str, str], SampleResult]:
    if cache is None:
        return {}
    results = {}
    for question in unit.questions:
        answer = cache.get(cache_task(task, arm), unit.unit_id, question.question_id)
        if answer is not None:
            outcome = AnswerOutcome(
                answer.prediction,
                0.0,
                answer.confidence,
                answer.memory_ids,
                answer.evidence,
                abstained=answer.abstained,
                abstention_reason=answer.abstention_reason,
                cached=True,
                ranked_source_ids=answer.ranked_source_ids or (),
                ranked_source_ids_complete=answer.ranked_source_ids is not None,
                generation_evidence=answer.generation_evidence,
                trace_id=answer.trace_id,
                answer_span_id=answer.answer_span_id,
                excerpt_evidence=answer.excerpt_evidence,
                excerpt_source_ids=tuple(item.memory_id for item in answer.excerpt_evidence),
                projection_evidence=answer.projection_evidence,
                projection_source_ids=tuple(item.memory_id for item in answer.projection_evidence),
            )
            results[(arm.name, question.question_id)] = build_sample_result(
                task,
                unit,
                question,
                outcome,
                ingest_failures=0,
                predict_only=predict_only,
                log_samples=log_samples,
                arm=arm,
                answer_policy=answer_policy,
            )
    return results


def cache_outcome(
    cache: ResponseCache | None,
    task: LoadedTask,
    unit: EvalUnit,
    question: EvalQuestion,
    outcome: AnswerOutcome | BaseException,
    ingest_failures: int,
    *,
    arm: Arm = PRODUCT_ARM,
) -> None:
    if (
        cache is None
        or ingest_failures
        or not isinstance(outcome, AnswerOutcome)
        or not outcome.prediction.strip()
        or outcome.retrieval_diagnostic_error is not None
        or (
            answers_through_ask(arm, task.spec.name)
            and retrieval_gold_ids(question.metadata)
            and not outcome.ranked_source_ids_complete
        )
    ):
        return
    cache.put(
        cache_task(task, arm),
        unit.unit_id,
        question.question_id,
        CachedAnswer(
            outcome.prediction,
            outcome.confidence,
            outcome.memory_ids,
            outcome.evidence,
            outcome.abstained,
            outcome.abstention_reason,
            outcome.ranked_source_ids if outcome.ranked_source_ids_complete else None,
            outcome.generation_evidence,
            outcome.trace_id,
            outcome.answer_span_id,
            outcome.excerpt_evidence,
            outcome.projection_evidence,
        ),
    )


def _candidate_count(unit: EvalUnit, question: EvalQuestion) -> int:
    """Count distinct sources a ranker could return for one question at its cutoff."""
    boundary = math.inf if question.cutoff_seconds is None else question.cutoff_seconds
    return bisect_right(unit._source_ends, boundary)


def build_sample_result(
    task: LoadedTask,
    unit: EvalUnit,
    question: EvalQuestion,
    outcome: AnswerOutcome | BaseException,
    *,
    ingest_failures: int,
    ingest_failure_details: tuple[FailureDetail, ...] = (),
    predict_only: bool,
    log_samples: bool,
    arm: Arm = PRODUCT_ARM,
    answer_policy: AnswerPolicy | None = None,
) -> SampleResult:
    memory_ids: tuple[str, ...]
    evidence: tuple[EvidenceInterval, ...]
    excerpt_source_ids: tuple[str, ...]
    excerpt_evidence: tuple[EvidenceInterval, ...]
    projection_source_ids: tuple[str, ...]
    projection_evidence: tuple[EvidenceInterval, ...]
    error_code: str | None
    error_detail: FailureDetail | None
    retrieval_diagnostic_error: FailureDetail | None = None
    ranked_source_ids: tuple[str, ...] = ()
    ranked_source_ids_complete = False
    if isinstance(outcome, BaseException):
        prediction, latency_ms, confidence, memory_ids, evidence, cached = (
            "",
            0.0,
            0.0,
            (),
            (),
            False,
        )
        excerpt_source_ids, excerpt_evidence = (), ()
        projection_source_ids, projection_evidence = (), ()
        error_detail = failure_detail(outcome)
        error_code = error_detail.code
        abstained = False
        abstention_reason = None
    else:
        prediction = outcome.prediction
        latency_ms = outcome.latency_ms
        confidence = outcome.confidence
        memory_ids = outcome.memory_ids
        evidence = outcome.evidence
        excerpt_source_ids = outcome.excerpt_source_ids
        excerpt_evidence = outcome.excerpt_evidence
        projection_source_ids = outcome.projection_source_ids
        projection_evidence = outcome.projection_evidence
        cached = outcome.cached
        ranked_source_ids = outcome.ranked_source_ids
        ranked_source_ids_complete = outcome.ranked_source_ids_complete
        error_detail = None if outcome.error is None else failure_detail(outcome.error)
        retrieval_diagnostic_error = (
            None
            if outcome.retrieval_diagnostic_error is None
            else failure_detail(outcome.retrieval_diagnostic_error)
        )
        error_code = None if error_detail is None else error_detail.code
        abstained = outcome.abstained
        abstention_reason = outcome.abstention_reason
    choices = tuple(
        str(value) for value in cast(Sequence[object], question.metadata.get("choices", ()))
    )
    parsed = parse_choice(prediction, choices)
    metrics, score, matched = _score(
        task.spec.name,
        question,
        prediction,
        parsed,
        ranked_source_ids,
        predict_only=predict_only,
        arm=arm,
        # An answer surface that exposes no ranked list carries no retrieval score. Reporting
        # its empty list as recall zero would invent one. The task's retrieval block excludes
        # the same samples so both surfaces report over one denominator.
        retrieval_available=(
            arm.retrieves and ranked_source_ids_complete and retrieval_diagnostic_error is None
        ),
        # A provider failure produced no answer; _score counts it as a wrong one. Transient
        # failures are waited out before they get here, so what remains is the system's loss.
        answer_failed=error_code is not None,
    )
    if (
        not isinstance(outcome, BaseException)
        and outcome.compiled_chars is not None
        and outcome.compiled_items is not None
    ):
        # "Useful evidence per token": the compile arm's own bundle size, reported per question
        # next to its other metrics so it never needs a second pass to reconstruct.
        metrics = {
            **metrics,
            "compile_bundle_chars": float(outcome.compiled_chars),
            "compile_bundle_items": float(outcome.compiled_items),
            "compile_bundle_excerpts": float(len(outcome.excerpt_source_ids)),
            "compile_bundle_projections": float(len(outcome.projection_source_ids)),
        }
    labelled_sources = set(retrieval_gold_ids(question.metadata))
    return SampleResult(
        task=task.spec.name,
        benchmark=task.spec.benchmark,
        dataset_sha256=task.dataset_sha256,
        evaluation_sha256=task.evaluation_sha256,
        unit_id=unit.unit_id,
        question_id=question.question_id,
        prediction=prediction,
        parsed_choice=parsed,
        score=score,
        exact_match=matched,
        latency_ms=latency_ms,
        confidence=confidence,
        memory_ids=memory_ids,
        candidate_count=_candidate_count(unit, question),
        ingest_failure_count=ingest_failures,
        ingest_failures=ingest_failure_details,
        error_code=error_code,
        error_reason=None if error_detail is None else error_detail.reason,
        error_stage=None if error_detail is None else error_detail.stage,
        error_cause_type=None if error_detail is None else error_detail.cause_type,
        error_message=None if error_detail is None else error_detail.message,
        retrieval_diagnostic_error=retrieval_diagnostic_error,
        abstained=abstained,
        abstention_reason=abstention_reason,
        cached=cached,
        metadata=question.metadata,
        prompt=tuple(str(part) for part in question.content) if log_samples else None,
        source_question=question.source_question if log_samples else None,
        references=question.references if log_samples else None,
        evidence=evidence,
        excerpt_source_ids=excerpt_source_ids,
        excerpt_evidence=excerpt_evidence,
        projection_source_ids=projection_source_ids,
        projection_evidence=projection_evidence,
        ref_at_300=(
            _reference_grounding(task, unit, question, evidence) if arm.generates else None
        ),
        metrics=metrics,
        scorer_protocol=scorer_protocol(task.spec.name),
        arm=arm.name,
        # Host-assistant compile tasks and the random retrieval control request no policy.
        answer_policy=(
            task_answer_policy(task.spec.name, answer_policy)
            if requests_answer_policy(arm, task.spec.name)
            else None
        ),
        retrieval_candidates=len(ranked_source_ids),
        ranked_source_ids=tuple(ranked_source_ids),
        ranked_source_ids_complete=ranked_source_ids_complete,
        generation_evidence=(
            None if isinstance(outcome, BaseException) else outcome.generation_evidence
        ),
        input_source_ids=tuple(
            dict.fromkeys(
                item.source_id
                for item in unit.memories
                if item.source_id in labelled_sources
                and (question.cutoff_seconds is None or memory_end(item) <= question.cutoff_seconds)
            )
        ),
        trace_id=None if isinstance(outcome, BaseException) else outcome.trace_id,
        answer_span_id=None if isinstance(outcome, BaseException) else outcome.answer_span_id,
    )


def _score(
    task_name: str,
    question: EvalQuestion,
    prediction: str,
    parsed_choice: str | None,
    ranked_source_ids: Sequence[str],
    *,
    predict_only: bool,
    arm: Arm = PRODUCT_ARM,
    retrieval_available: bool = True,
    answer_failed: bool = False,
) -> tuple[Mapping[str, float], float | None, float | None]:
    if predict_only:
        return {}, None, None
    metrics = local_scores(
        task_name,
        score_kind=question.score_kind,
        prediction=prediction,
        parsed_choice=parsed_choice,
        expected_choice=question.expected_choice,
        references=question.references,
        question=question.source_question,
        metadata=question.metadata,
        evidence_source_ids=tuple(ranked_source_ids),
    )
    metrics = _arm_metrics(metrics, arm, retrieval_available=retrieval_available)
    if answer_failed:
        # A question the system did not answer is a question it got wrong: it scores zero under
        # the task's primary metric, and the separately measured retrieval diagnostic stays.
        # Judge-scored families carry no local answer metric, so the zero is written explicitly
        # rather than read off the empty prediction, and the derived joint_* metrics follow.
        # The earlier rule left it unscored so one failure did not deflate the arms unevenly,
        # at the price of a single residual failure invalidating a task's whole score; the
        # failure is still counted and its provider message kept for diagnosis.
        failed = finalize_scores(
            task_name,
            {
                **metrics,
                sample_primary_metric(task_name): 0.0,
                **({"exact_match": 0.0} if "exact_match" in metrics else {}),
            },
        )
        return failed, 0.0, failed.get("exact_match")
    score = metrics.get(sample_primary_metric(task_name), metrics.get("token_f1"))
    return metrics, score, metrics.get("exact_match")


def _arm_metrics(
    metrics: Mapping[str, float], arm: Arm, *, retrieval_available: bool
) -> dict[str, float]:
    """Keep only the metrics an arm can honestly carry.

    An arm that never retrieved has no retrieval score -- reporting zero would read as a
    retrieval failure -- and an arm that never generated has no answer score.
    """
    diagnostic = ("retrieval_", "joint_")
    return {
        name: value
        for name, value in metrics.items()
        if (retrieval_available or not name.startswith(diagnostic))
        and (arm.generates or name.startswith(diagnostic))
    }


def _reference_grounding(
    task: LoadedTask,
    unit: EvalUnit,
    question: EvalQuestion,
    evidence: Sequence[EvidenceInterval],
) -> float | None:
    if task.spec.benchmark != "MM-Lifelong":
        return None
    reference = question.metadata.get("reference_intervals")
    ends = tuple(item.end_seconds for item in unit.memories if item.end_seconds is not None)
    if not isinstance(reference, Sequence) or isinstance(reference, str | bytes) or not ends:
        return None
    try:
        expected = tuple((float(interval[0]), float(interval[1])) for interval in reference)
    except (IndexError, TypeError, ValueError):
        return None
    predicted = tuple(
        (item.start_seconds, item.end_seconds)
        for item in evidence
        if item.start_seconds is not None and item.end_seconds is not None
    )
    return _ref_at_n(predicted, expected, total_seconds=max(ends), bucket_size=300.0)


def _ref_at_n(
    predicted: Sequence[tuple[float, float]],
    expected: Sequence[tuple[float, float]],
    *,
    total_seconds: float,
    bucket_size: float,
) -> float:
    """Official MM-Lifelong quantized temporal IoU."""
    if not math.isfinite(total_seconds) or total_seconds <= 0:
        raise ValueError("total_seconds must be positive and finite")
    if not math.isfinite(bucket_size) or bucket_size <= 0:
        raise ValueError("bucket_size must be positive and finite")

    def buckets(intervals: Sequence[tuple[float, float]]) -> set[int]:
        values: set[int] = set()
        for start, end in intervals:
            start = max(0.0, start)
            end = min(total_seconds, end)
            if start >= end:
                continue
            first = int(start // bucket_size)
            last = int((end - 1e-9) // bucket_size)
            values.update(range(first, last + 1))
        return values

    predicted_buckets = buckets(predicted)
    expected_buckets = buckets(expected)
    if not predicted_buckets and not expected_buckets:
        return 0.0
    return len(predicted_buckets & expected_buckets) / len(predicted_buckets | expected_buckets)
