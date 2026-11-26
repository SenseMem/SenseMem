"""Question admission, product and baseline execution, and answer timing."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import (
    Awaitable,
    Callable,
    Iterator,
    Sequence,
)
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from opentelemetry.trace import Span, StatusCode, Tracer

from sensemem import (
    AbstentionReason,
    AnswerResult,
    AsyncMemory,
    ContextBundle,
    SearchHit,
)
from sensemem._telemetry import (
    OPERATION_TTFT,
    SPAN_KIND,
    _observe_generation_inputs,
    _observe_retrieval_results,
    traced_span,
)
from sensemem.benchmarks._official import personamem_v3_prompts as pm3_prompts
from sensemem.benchmarks._official import personamem_v3_tasks as pm3_tasks
from sensemem.benchmarks.baseline_prompts import (
    COMPILE_SURFACE_SYSTEM_PROMPT,
)
from sensemem.benchmarks.eval_adapters import (
    EvalQuestion,
)
from sensemem.benchmarks.eval_arms import (
    PRODUCT_ARM,
    Arm,
    answers_through_ask,
)
from sensemem.benchmarks.eval_config import (
    RETRIEVAL_CANDIDATE_LIMIT,
)
from sensemem.benchmarks.eval_evidence import EvidenceInterval, evidence, limited_source_evidence
from sensemem.benchmarks.eval_inputs import normalize_content
from sensemem.benchmarks.eval_options import EvaluationOptions
from sensemem.benchmarks.eval_results import (
    DEFAULT_ARM,
    AnswerOutcome,
    Retried,
    SystemicEmbeddingFailure,
    retry_transient,
    systemic_embedding_stage_failure,
)
from sensemem.benchmarks.eval_runtime import (
    SDK_OPERATION_LIMIT,
    SDK_STREAM_LIMIT,
    model_operation,
)
from sensemem.benchmarks.eval_telemetry import (
    BENCHMARK_ANSWER_SPAN,
    BENCHMARK_ARM,
    BENCHMARK_ARM_SPAN,
    BENCHMARK_COMPILE_CHARS,
    BENCHMARK_COMPILE_ITEMS,
    BENCHMARK_COMPILE_MEDIA_ITEMS,
    BENCHMARK_COMPILE_SPAN,
    BENCHMARK_PURPOSE,
    BENCHMARK_SAMPLE,
    BENCHMARK_TASK,
    PRODUCT_PURPOSE,
)
from sensemem.benchmarks.official_scorers import (
    retrieval_gold_ids,
)
from sensemem.benchmarks.prompts import task_answer_policy
from sensemem.models._openai_output import _answer_result


def _declined(answer: str, question: EvalQuestion) -> bool:
    """Report a refusal a task worded itself, which the product cannot recognise."""
    return question.refusal is not None and answer.strip().rstrip(".") == (
        question.refusal.strip().rstrip(".")
    )


def _require_active_embedding_service(systemic_stop: asyncio.Event) -> None:
    if systemic_stop.is_set():
        raise SystemicEmbeddingFailure("systemic query embedding failure; queued question aborted")


async def _guarded_answer(
    memory: AsyncMemory,
    question: EvalQuestion,
    *,
    semaphore: asyncio.Semaphore,
    instance_semaphore: asyncio.Semaphore,
    model_semaphore: asyncio.Semaphore,
    systemic_stop: asyncio.Event,
    arm: Arm,
    task_name: str,
    unit_id: str,
    options: EvaluationOptions,
    context: str,
    tracer: Tracer | None,
    on_answer: Callable[[EvalQuestion, AnswerOutcome], None] | None,
    on_complete: Callable[[], None] | None,
) -> AnswerOutcome:
    identity = f"{task_name}/{unit_id}/{question.question_id}"
    sample_id = identity if arm.name == DEFAULT_ARM else f"{arm.name}:{identity}"

    async def answer() -> AnswerOutcome:
        _require_active_embedding_service(systemic_stop)

        async def request(current: EvalQuestion) -> AnswerOutcome:
            return await _arm_answer(
                memory,
                current,
                arm=arm,
                task_name=task_name,
                options=options,
                context=context,
                tracer=tracer,
                sample_id=sample_id,
                started=started if arm.generates else time.perf_counter(),
                answer_span=answer_span,
                model_semaphore=model_semaphore,
                systemic_stop=systemic_stop,
            )

        if task_name == "personamem-v3" and question.metadata.get("cluster_id"):
            return await _answer_personamem_cluster(question, request)
        return await request(question)

    try:
        _require_active_embedding_service(systemic_stop)
        # Start the caller span before admission so latency and TTFT retain queueing time.
        span = (
            _answer_span(tracer, task_name, sample_id, arm.name)
            if arm.generates
            else nullcontext((None, 0.0))
        )
        with span as (answer_span, started):
            async with instance_semaphore, semaphore:
                outcome = await answer()
                if outcome.error is not None and answer_span is not None:
                    answer_span.set_status(StatusCode.ERROR)
                span_context = None if answer_span is None else answer_span.get_span_context()
                if span_context is not None and span_context.is_valid:
                    outcome = replace(
                        outcome,
                        trace_id=f"{span_context.trace_id:032x}",
                        answer_span_id=f"{span_context.span_id:016x}",
                    )
        if arm.generates and on_answer is not None:
            on_answer(question, outcome)
        return outcome
    except SystemicEmbeddingFailure:
        systemic_stop.set()
        raise
    finally:
        if on_complete is not None:
            on_complete()


async def answer_many(  # noqa: C901 - shared sequences and admission use one request lifetime
    memory: AsyncMemory,
    questions: Sequence[EvalQuestion],
    *,
    options: EvaluationOptions,
    request_semaphore: asyncio.Semaphore | None = None,
    on_answer: Callable[[EvalQuestion, AnswerOutcome], None] | None = None,
    on_complete: Callable[[], None] | None = None,
    arm: Arm = PRODUCT_ARM,
    task_name: str = "",
    unit_id: str = "",
    context: str = "",
    tracer: Tracer | None = None,
    model_semaphore: asyncio.Semaphore | None = None,
    cluster_answers: dict[tuple[str, str], AnswerOutcome] | None = None,
) -> tuple[AnswerOutcome | BaseException, ...]:
    semaphore = request_semaphore or asyncio.Semaphore(options.request_concurrency)
    model_semaphore = model_semaphore or asyncio.Semaphore(SDK_OPERATION_LIMIT)
    # The SDK's documented limits are per physical instance, while the request semaphore is
    # shared across units. Queue excess callers here before they can occupy global slots or
    # become immediate admission errors. Each unit owns its store and runs its arms in order.
    instance_limit = SDK_OPERATION_LIMIT if arm.reads_memory else options.request_concurrency
    if options.stream_answers and answers_through_ask(arm, task_name):
        instance_limit = SDK_STREAM_LIMIT
    instance_semaphore = asyncio.Semaphore(min(options.request_concurrency, instance_limit))
    systemic_stop = asyncio.Event()
    cluster_answers = {} if cluster_answers is None else cluster_answers
    pending_clusters: dict[tuple[str, str], asyncio.Task[AnswerOutcome]] = {}

    async def answer(question: EvalQuestion) -> AnswerOutcome:
        cluster_id = question.metadata.get("cluster_id") if task_name == "personamem-v3" else None
        key = (arm.name, str(cluster_id))
        clustered = bool(cluster_id)

        async def guarded() -> AnswerOutcome:
            return await _guarded_answer(
                memory,
                question,
                semaphore=semaphore,
                instance_semaphore=instance_semaphore,
                model_semaphore=model_semaphore,
                systemic_stop=systemic_stop,
                arm=arm,
                task_name=task_name,
                unit_id=unit_id,
                options=options,
                context=context,
                tracer=tracer,
                on_answer=None if clustered else on_answer,
                on_complete=None if clustered else on_complete,
            )

        if not clustered:
            return await guarded()
        try:
            previous = cluster_answers.get(key)
            if previous is not None:
                outcome = replace(previous, latency_ms=0.0, cached=True)
            else:
                if key not in pending_clusters:
                    pending_clusters[key] = asyncio.create_task(guarded())
                outcome = await pending_clusters[key]
                if outcome.error is None:
                    cluster_answers[key] = outcome
            if arm.generates and on_answer is not None:
                on_answer(question, outcome)
            return outcome
        finally:
            if on_complete is not None:
                on_complete()

    with _arm_run_span(tracer, task_name, arm.name):
        answered = tuple(
            await asyncio.gather(
                *(answer(question) for question in questions),
                return_exceptions=True,
            )
        )

    diagnosed = tuple(
        replace(
            outcome,
            retrieval_diagnostic_error=RuntimeError(
                "the ranked retrieval list from the product answer was not observed"
            ),
        )
        if isinstance(outcome, AnswerOutcome)
        # Memory.compile exposes no ranked list, so a product answer on a compile task has
        # none to observe; only the ask and random paths owe one.
        and (arm.name == "random" or answers_through_ask(arm, task_name))
        and retrieval_gold_ids(question.metadata)
        and not outcome.ranked_source_ids_complete
        else outcome
        for question, outcome in zip(questions, answered, strict=True)
    )
    if not arm.generates and on_answer is not None:
        for question, outcome in zip(questions, diagnosed, strict=True):
            if isinstance(outcome, AnswerOutcome):
                on_answer(question, outcome)
    return diagnosed


async def _answer_personamem_cluster(
    question: EvalQuestion,
    request: Callable[[EvalQuestion], Awaitable[AnswerOutcome]],
) -> AnswerOutcome:
    """Thread all released turns once; labels and fatigue thresholds stay outside prompts."""
    turns = cast(tuple[pm3_prompts.ClusterTurn, ...], question.metadata["cluster_queries"])
    responses: list[pm3_prompts.ClusterResponse] = []
    elapsed = 0.0
    for turn in turns:
        if question.metadata["task_type"] == "over_personalization_repetition_recsys":
            prompt = pm3_prompts.over_personalization_repetition_recsys_prompt(
                "", "", turn["user_query"], [], [], [], responses, 0
            )
        else:
            prompt = pm3_prompts.over_personalization_repetition_chatbot_prompt(
                turn["user_query"], "", "", responses, 0
            )
        outcome = await request(
            replace(
                question,
                content=(prompt,),
                source_question=turn["user_query"],
                reference_at=datetime.fromtimestamp(turn["ts"], timezone.utc),
            )
        )
        if outcome.error is not None:
            return outcome
        elapsed += outcome.latency_ms
        responses.append(
            pm3_tasks.cluster_response(
                outcome.prediction,
                user_query=turn["user_query"],
                timestamp=turn["ts"],
                is_target=turn["is_target"],
            )
        )
    return replace(outcome, prediction=json.dumps({"responses": responses}), latency_ms=elapsed)


@contextmanager
def _compile_span(tracer: Tracer | None, bundle: ContextBundle) -> Iterator[None]:
    """Tag one compiled bundle's size so eval_telemetry can report it next to its latency.

    compile() itself carries no size attribute -- it is a benchmark-only measurement, so the
    harness tags it on a harness-owned span rather than asking product code to carry it.
    """
    if tracer is None:
        yield
        return
    with traced_span(
        tracer,
        BENCHMARK_COMPILE_SPAN,
        attributes={
            SPAN_KIND: "stage",
            BENCHMARK_COMPILE_CHARS: bundle.chars,
            BENCHMARK_COMPILE_ITEMS: (
                len(bundle.hits)
                + len(bundle.excerpts)
                + len(bundle.projections)
                + len(bundle.supplements)
            ),
            # Grounded parts, not memories carrying them, because that is the quantity
            # ContextBudget.max_media_items bounds: an omni memory with a still and a clip is
            # two parts against the budget and has to be two here, or a multi-asset bundle
            # reports as thrifty as a single-asset one.
            BENCHMARK_COMPILE_MEDIA_ITEMS: sum(len(hit.assets) for hit in bundle.hits),
        },
    ):
        yield


@contextmanager
def _arm_run_span(tracer: Tracer | None, task_name: str, arm: str) -> Iterator[None]:
    if tracer is None:
        yield
        return
    with traced_span(
        tracer,
        BENCHMARK_ARM_SPAN,
        attributes={
            BENCHMARK_TASK: task_name,
            BENCHMARK_ARM: arm,
            BENCHMARK_PURPOSE: PRODUCT_PURPOSE,
            SPAN_KIND: "benchmark",
        },
    ):
        yield


@contextmanager
def _answer_span(
    tracer: Tracer | None,
    task_name: str,
    sample_id: str,
    arm: str | None = None,
) -> Iterator[tuple[Span | None, float]]:
    """Scope one answer's model spans so grounding loss can be attributed to its sample."""
    if tracer is None:
        yield None, time.perf_counter()
        return
    selected_arm = (
        sample_id.split(":", 1)[0]
        if arm is None and ":" in sample_id
        else DEFAULT_ARM
        if arm is None
        else arm
    )
    with traced_span(
        tracer,
        BENCHMARK_ANSWER_SPAN,
        attributes={
            BENCHMARK_TASK: task_name,
            BENCHMARK_SAMPLE: sample_id,
            BENCHMARK_ARM: selected_arm,
            BENCHMARK_PURPOSE: PRODUCT_PURPOSE,
            SPAN_KIND: "benchmark",
        },
    ) as span:
        yield span, time.perf_counter()


async def _arm_answer(  # noqa: C901 - baseline and streamed product paths share one clock
    memory: AsyncMemory,
    question: EvalQuestion,
    *,
    arm: Arm,
    task_name: str,
    options: EvaluationOptions,
    context: str,
    sample_id: str,
    started: float,
    answer_span: Span | None,
    tracer: Tracer | None = None,
    model_semaphore: asyncio.Semaphore | None = None,
    systemic_stop: asyncio.Event | None = None,
) -> AnswerOutcome:
    latency_started = time.perf_counter()
    content = normalize_content(question.content)
    ranked: tuple[SearchHit, ...] = ()
    ranked_complete = False
    generation_evidence: tuple[EvidenceInterval, ...] | None = None
    excerpt_evidence: tuple[EvidenceInterval, ...] = ()
    projection_evidence: tuple[EvidenceInterval, ...] = ()
    requested_policy = task_answer_policy(task_name, options.answer_policy)

    async def attempt(operation: Callable[[], Awaitable[Retried]]) -> Retried:
        # Only the attempt that answered is timed: a retry's backoff is outage time, not latency.
        async def timed() -> Retried:
            async def admitted() -> Retried:
                nonlocal \
                    latency_started, \
                    ranked, \
                    ranked_complete, \
                    generation_evidence, \
                    excerpt_evidence, \
                    projection_evidence
                if systemic_stop is not None:
                    _require_active_embedding_service(systemic_stop)
                latency_started = time.perf_counter()
                ranked, ranked_complete, generation_evidence = (), False, None
                excerpt_evidence = ()
                projection_evidence = ()
                return await operation()

            return await model_operation(model_semaphore, admitted)

        return await retry_transient(timed)

    def observe_retrieval(value: object) -> None:
        nonlocal ranked, ranked_complete
        if isinstance(value, tuple) and all(isinstance(hit, SearchHit) for hit in value):
            ranked = value
            ranked_complete = True

    def observe_generation(value: object) -> None:
        nonlocal generation_evidence
        if isinstance(value, tuple) and all(isinstance(hit, SearchHit) for hit in value):
            generation_evidence = tuple(evidence(hit) for hit in value)

    try:
        if arm.name == "random":
            ranked = (
                await attempt(
                    lambda: memory.search(
                        content,
                        limit=(
                            RETRIEVAL_CANDIDATE_LIMIT
                            if retrieval_gold_ids(question.metadata)
                            else options.recall_limit
                        ),
                        reference_at=question.reference_at,
                    )
                )
            ).hits
            order = list(ranked)
            random.Random(f"{arm.seed}:{sample_id}").shuffle(order)
            return AnswerOutcome(
                "",
                (time.perf_counter() - latency_started) * 1_000,
                0.0,
                tuple(hit.id for hit in order),
                tuple(evidence(hit) for hit in order),
                ranked_source_ids=_source_ids(order),
                ranked_source_ids_complete=True,
            )
        # The product arm answers through the surface the task calls for: Memory.ask for a
        # question with an answer in memory, Memory.compile plus the host generator for a task
        # that wants an assistant's response. task_answer_surface owns the mapping and its
        # measurements.
        compile_surface = arm.name == DEFAULT_ARM and not answers_through_ask(arm, task_name)
        if arm.name == "compile" or compile_surface:
            bundle = await attempt(
                lambda: memory.compile(
                    content,
                    budget=options.compile_budget,
                    reference_at=question.reference_at,
                    allow_partial_sources=arm.allow_partial_sources,
                )
            )
            with _compile_span(tracer, bundle):
                rendered = bundle.render()
            compile_generator = arm.generator
            if compile_generator is None:
                raise RuntimeError(
                    f"answering through Memory.compile requires a generator ({arm.name} arm, "
                    f"task {task_name})"
                )

            async def answer_compiled() -> str:
                nonlocal generation_evidence, excerpt_evidence, projection_evidence
                generation_evidence = tuple(evidence(hit) for hit in bundle.hits)
                excerpt_evidence = tuple(limited_source_evidence(item) for item in bundle.excerpts)
                projection_evidence = tuple(
                    limited_source_evidence(item) for item in bundle.projections
                )
                return await compile_generator.answer(
                    _question_text(question),
                    rendered,
                    question_assets=tuple(
                        atom for atom in question.content if isinstance(atom, Path)
                    ),
                    evidence_hits=bundle.hits,
                    system_prompt=COMPILE_SURFACE_SYSTEM_PROMPT if compile_surface else None,
                    answer_policy=requested_policy,
                )

            prediction = await attempt(answer_compiled)
            generated = (
                AnswerResult(prediction, ())
                if compile_surface
                else _answer_result(prediction, (), answer_policy=requested_policy)
            )
            prediction = generated.answer
            declined_here = generated.abstained or _declined(prediction, question)
            return AnswerOutcome(
                prediction,
                (time.perf_counter() - latency_started) * 1_000,
                0.0,
                tuple(hit.id for hit in bundle.hits),
                tuple(evidence(hit) for hit in bundle.hits),
                abstained=declined_here,
                abstention_reason=(
                    generated.abstention_reason.value
                    if generated.abstention_reason is not None
                    else AbstentionReason.INSUFFICIENT_EVIDENCE.value
                    if declined_here
                    else None
                ),
                ranked_source_ids=_source_ids(ranked),
                compiled_chars=bundle.chars,
                compiled_items=(
                    len(bundle.hits)
                    + len(bundle.excerpts)
                    + len(bundle.projections)
                    + len(bundle.supplements)
                ),
                excerpt_source_ids=tuple(excerpt.source_memory_id for excerpt in bundle.excerpts),
                excerpt_evidence=tuple(limited_source_evidence(item) for item in bundle.excerpts),
                projection_source_ids=tuple(item.source_memory_id for item in bundle.projections),
                projection_evidence=tuple(
                    limited_source_evidence(item) for item in bundle.projections
                ),
                generation_evidence=tuple(evidence(hit) for hit in bundle.hits),
            )
        generator = arm.generator
        if generator is not None and arm.name != DEFAULT_ARM:
            prediction = await attempt(
                lambda: generator.answer(
                    _question_text(question),
                    context if arm.name == "full-context" else None,
                    question_assets=tuple(
                        atom for atom in question.content if isinstance(atom, Path)
                    ),
                    answer_policy=requested_policy,
                )
            )
            generated = _answer_result(prediction, (), answer_policy=requested_policy)
            declined_here = generated.abstained or _declined(generated.answer, question)
            return AnswerOutcome(
                generated.answer,
                (time.perf_counter() - latency_started) * 1_000,
                0.0,
                (),
                (),
                abstained=declined_here,
                abstention_reason=(
                    generated.abstention_reason.value
                    if generated.abstention_reason is not None
                    else AbstentionReason.INSUFFICIENT_EVIDENCE.value
                    if declined_here
                    else None
                ),
            )

        # Protocol alignment, not a scorer change: one task's official evaluation gives no credit
        # for "unknown", so the request asks for a committed answer there. task_answer_policy
        # owns the mapping -- {m3-bench-robot} -- and its rationale.
        async def answered() -> AnswerResult:
            if not options.stream_answers:
                return await memory.ask(
                    content,
                    limit=options.recall_limit,
                    reference_at=question.reference_at,
                    answer_policy=requested_policy,
                )
            first_token_seen = False
            streamed: AnswerResult | None = None
            async for chunk in memory.ask_stream(
                content,
                limit=options.recall_limit,
                reference_at=question.reference_at,
                answer_policy=requested_policy,
            ):
                if chunk.text.strip() and not first_token_seen:
                    first_token_seen = True
                    if answer_span is not None:
                        answer_span.set_attribute(
                            OPERATION_TTFT,
                            (time.perf_counter() - started) * 1_000,
                        )
                if chunk.result is not None:
                    streamed = chunk.result
            if streamed is None:
                raise RuntimeError("answer stream ended without a terminal result")
            return streamed

        with (
            _observe_retrieval_results(observe_retrieval),
            _observe_generation_inputs(observe_generation),
        ):
            result = await attempt(answered)
    except Exception as error:
        if systemic_embedding_stage_failure(error):
            raise SystemicEmbeddingFailure(
                "systemic query embedding provider failure; answer matrix aborted"
            ) from error
        return AnswerOutcome(
            "",
            (time.perf_counter() - latency_started) * 1_000,
            max((hit.score for hit in ranked), default=0.0),
            tuple(hit.id for hit in ranked),
            tuple(evidence(hit) for hit in ranked),
            error=error,
            ranked_source_ids=_source_ids(ranked),
            ranked_source_ids_complete=ranked_complete,
            generation_evidence=generation_evidence,
            excerpt_evidence=excerpt_evidence,
            excerpt_source_ids=tuple(item.memory_id for item in excerpt_evidence),
            projection_evidence=projection_evidence,
            projection_source_ids=tuple(item.memory_id for item in projection_evidence),
        )
    # Count task-worded refusals on every generating arm, alongside the reader's own signal.
    declined = _declined(result.answer, question)
    return AnswerOutcome(
        result.answer,
        (time.perf_counter() - latency_started) * 1_000,
        max((hit.score for hit in result.hits), default=0.0),
        tuple(hit.id for hit in result.hits),
        tuple(evidence(hit) for hit in result.hits),
        abstained=result.abstained or declined,
        abstention_reason=(
            result.abstention_reason.value
            if result.abstention_reason is not None
            else (AbstentionReason.INSUFFICIENT_EVIDENCE.value if declined else None)
        ),
        ranked_source_ids=_source_ids(ranked),
        ranked_source_ids_complete=ranked_complete,
        generation_evidence=generation_evidence,
    )


def _source_ids(hits: Sequence[SearchHit]) -> tuple[str, ...]:
    return tuple(evidence(hit).source_id or "" for hit in hits)


def _question_text(question: EvalQuestion) -> str:
    return "\n".join(str(part) for part in question.content if not isinstance(part, Path))
