"""Isolated unit lifecycles, causal replay and bounded task workers."""

from __future__ import annotations

import asyncio
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from functools import partial
from pathlib import Path

from opentelemetry.trace import StatusCode, Tracer

from sensemem import (
    AsyncMemory,
)
from sensemem._telemetry import (
    SPAN_KIND,
    traced_span,
)
from sensemem.benchmarks.eval_adapters import (
    EvalQuestion,
    EvalUnit,
    LoadedTask,
    MemoryItem,
)
from sensemem.benchmarks.eval_answers import answer_many
from sensemem.benchmarks.eval_arms import (
    PRODUCT_ARM,
    Arm,
)
from sensemem.benchmarks.eval_cache import ResponseCache
from sensemem.benchmarks.eval_console import (
    ignore_progress,
)
from sensemem.benchmarks.eval_ingest import (
    IngestCheckpoint,
    IngestRunState,
    deliberate_after_ingest,
    ingest,
    ingest_observer,
)
from sensemem.benchmarks.eval_inputs import (
    full_context,
    has_question_cutoffs,
    memory_end,
    normalize_content,
    planned_ingest_count,
    prefix_end,
)
from sensemem.benchmarks.eval_options import EvaluationOptions
from sensemem.benchmarks.eval_results import (
    DEFAULT_ARM,
    AnswerOutcome,
    FailureDetail,
    SampleResult,
    SystemicEmbeddingFailure,
)
from sensemem.benchmarks.eval_runtime import (
    SDK_OPERATION_LIMIT,
    MemoryFactory,
    SearchReplay,
    model_operation,
)
from sensemem.benchmarks.eval_samples import build_sample_result, cache_outcome, cached_results
from sensemem.benchmarks.eval_telemetry import (
    BENCHMARK_ARM,
    BENCHMARK_DIAGNOSTIC_SPAN,
    BENCHMARK_PURPOSE,
    BENCHMARK_SAMPLE,
    BENCHMARK_TASK,
    DIAGNOSTIC_PURPOSE,
    SHARED_BENCHMARK_ARM,
)
from sensemem.benchmarks.isolation import BenchmarkRun
from sensemem.benchmarks.personamem_v3_results import complete_lifecycle

_BENCHMARK_SEARCH_REPLAY_SETUP_SPAN = "sensemem.benchmark.search_replay_setup"


async def run_loaded_task(  # noqa: C901 - bounded workers also own one isolated replay barrier
    task: LoadedTask,
    *,
    run: BenchmarkRun,
    memory_factory: MemoryFactory,
    options: EvaluationOptions,
    response_cache: ResponseCache | None = None,
    arms: Sequence[Arm] = (PRODUCT_ARM,),
    ingest_digest: str | None = None,
    tracer: Tracer | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_ingest_progress: Callable[[int, int], None] | None = None,
    on_activity: Callable[[str], None] | None = None,
    on_search_replay_ready: Callable[[SearchReplay], None] | None = None,
    prefetch_speech: Callable[[Sequence[MemoryItem]], None] | None = None,
    run_state: IngestRunState | None = None,
) -> tuple[SampleResult, ...]:
    """Run normalized units with bounded workers while preserving release order."""
    run_state = run_state or IngestRunState()
    if options.ingest_concurrency > 1 and has_question_cutoffs((task,)):
        raise ValueError("concurrent ingest requires static units without question cutoffs")
    slots: list[tuple[SampleResult, ...] | None] = [None] * len(task.units)
    unit_paths: list[Path | None] = [None] * len(task.units)
    stores_ready = [False] * len(task.units)
    queue: asyncio.Queue[tuple[int, EvalUnit]] = asyncio.Queue()
    worker_count = min(options.unit_concurrency, 16, len(task.units))
    request_semaphore = asyncio.Semaphore(options.request_concurrency)
    # Ingest and answers borrow the same process budget only while doing work.
    model_semaphore = asyncio.Semaphore(SDK_OPERATION_LIMIT)
    completed = 0
    total = sum(len(unit.questions) for unit in task.units) * len(arms)
    notify_progress = on_progress or ignore_progress
    notify_ingest = on_ingest_progress or ignore_progress
    notify_activity = on_activity or _ignore_activity
    unit_activities: list[str | None] = [None] * len(task.units)
    # One bar spans the task rather than one per unit: concurrent workers would otherwise fight
    # over the terminal, and what a reader wants is how much of the corpus is written, not which
    # worker wrote it. An arm that reads no memory ingests nothing and draws no bar.
    reads_memory = any(arm.reads_memory for arm in arms)
    planned_ingest = tuple(planned_ingest_count(unit) if reads_memory else 0 for unit in task.units)
    ingest_total = sum(planned_ingest)
    ingested_total = 0
    notify_ingest(0, ingest_total)
    for index, unit in enumerate(task.units):
        queue.put_nowait((index, unit))

    async def worker() -> None:
        while not queue.empty():
            try:
                index, unit = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            data_dir = run.unit_dir(unit.unit_id)
            unit_paths[index] = data_dir
            checkpoint = (
                None
                if ingest_digest is None
                else IngestCheckpoint(run, unit.unit_id, ingest_digest)
            )
            reported = 0
            reported_ingest = 0

            def ingest_advanced(count: int, planned: int = planned_ingest[index]) -> None:
                # A checkpoint resumes mid-corpus and a cached unit skips the corpus outright,
                # so the count arrives as an absolute position and may not arrive at all. The
                # clamp is not load-bearing -- _IngestCheckpoint.start already drops a note that
                # runs past the first pending cutoff -- it only keeps a display value honest.
                nonlocal ingested_total, reported_ingest
                bounded = min(count, planned)
                if bounded <= reported_ingest:
                    return
                ingested_total += bounded - reported_ingest
                reported_ingest = bounded
                notify_ingest(ingested_total, ingest_total)

            def activity(
                phase: str, unit_index: int = index, unit_total: int = len(task.units)
            ) -> None:
                unit_activities[unit_index] = phase
                notify_activity(_activity_summary(unit_activities, unit_total))

            def sample_completed() -> None:
                nonlocal completed, reported
                completed += 1
                reported += 1
                notify_progress(completed, total)

            def store_ready(unit_index: int = index) -> None:
                stores_ready[unit_index] = True

            activity("preparing")
            samples = await _run_unit(
                task,
                unit,
                data_dir,
                memory_factory=memory_factory,
                options=options,
                request_semaphore=request_semaphore,
                model_semaphore=model_semaphore,
                response_cache=response_cache,
                arms=arms,
                checkpoint=checkpoint,
                tracer=tracer,
                on_sample_completed=sample_completed,
                on_ingested=ingest_advanced,
                on_activity=activity,
                on_store_ready=store_ready,
                prefetch_speech=prefetch_speech,
                run_state=run_state,
            )
            slots[index] = samples
            for _ in range(len(samples) - reported):
                sample_completed()
            # A unit that answered from the cache, or whose cutoffs stop short of its tail, wrote
            # fewer items than planned. Settling its share here is what lets the bar finish.
            ingest_advanced(planned_ingest[index])
            unit_activities[index] = None
            notify_activity(_activity_summary(unit_activities, len(task.units)))
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        for pending_worker in workers:
            pending_worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    if any(group is None for group in slots):
        raise RuntimeError("evaluation worker exited before every unit completed")

    async def replay_searches() -> None:
        await _measure_standalone_searches(
            task,
            slots=slots,
            unit_paths=unit_paths,
            stores_ready=stores_ready,
            memory_factory=memory_factory,
            unit_concurrency=options.unit_concurrency,
            request_semaphore=asyncio.Semaphore(
                min(options.request_concurrency, SDK_OPERATION_LIMIT)
            ),
            recall_limit=options.recall_limit,
            arms=arms,
            tracer=tracer,
        )

    if on_search_replay_ready is None:
        await replay_searches()
    else:
        on_search_replay_ready(replay_searches)
    return tuple(sample for group in slots if group is not None for sample in group)


async def _measure_standalone_searches(  # noqa: C901 - setup failures need per-plan accounting
    task: LoadedTask,
    *,
    slots: Sequence[tuple[SampleResult, ...] | None],
    unit_paths: Sequence[Path | None],
    stores_ready: Sequence[bool],
    memory_factory: MemoryFactory,
    unit_concurrency: int,
    request_semaphore: asyncio.Semaphore,
    recall_limit: int,
    arms: Sequence[Arm],
    tracer: Tracer | None,
) -> None:
    """Replay public searches only after every formal answer has finished.

    Store opening is setup, not caller latency. Each caller span starts immediately before the
    shared request semaphore so the measured latency includes benchmark admission, while the
    nested sensemem.search span remains the public SDK boundary. The diagnostic purpose
    keeps replay embedding work out of product answer nodes and token totals.
    """
    if tracer is None or not any(arm.name == DEFAULT_ARM for arm in arms):
        return
    selected_tracer = tracer
    store_semaphore = asyncio.Semaphore(min(unit_concurrency, 16))

    def span_attributes(sample: SampleResult) -> dict[str, str]:
        return {
            BENCHMARK_TASK: task.spec.name,
            BENCHMARK_SAMPLE: sample.sample_id,
            BENCHMARK_ARM: DEFAULT_ARM,
            BENCHMARK_PURPOSE: DIAGNOSTIC_PURPOSE,
            SPAN_KIND: "benchmark",
        }

    async def measure_one(
        memory: AsyncMemory,
        question: EvalQuestion,
        sample: SampleResult,
        instance_semaphore: asyncio.Semaphore,
    ) -> None:
        try:
            with traced_span(
                selected_tracer,
                BENCHMARK_DIAGNOSTIC_SPAN,
                attributes=span_attributes(sample),
            ):
                async with instance_semaphore, request_semaphore:
                    await memory.search(
                        normalize_content(question.content),
                        limit=recall_limit,
                        reference_at=question.reference_at,
                    )
        except Exception:
            # The span records an error attempt. Search replay is diagnostic and must not replace
            # or invalidate the already-completed product answer.
            return

    async def measure_unit(index: int, unit: EvalUnit) -> None:
        path = unit_paths[index]
        samples = slots[index]
        if path is None or samples is None or not stores_ready[index]:
            return
        product_samples = tuple(sample for sample in samples if sample.arm == DEFAULT_ARM)
        # A partial ingest is not a valid warm store. Cache-only units likewise have no fresh
        # product question to replay and never reach the memory factory here.
        if any(sample.ingest_failure_count for sample in product_samples):
            return
        pending = {sample.question_id: sample for sample in product_samples if not sample.cached}
        questions = tuple(
            (question, pending[question.question_id])
            for question in unit.questions
            if question.question_id in pending
        )
        if not questions:
            return
        async with store_semaphore:
            opened = False
            try:
                with traced_span(
                    selected_tracer,
                    _BENCHMARK_SEARCH_REPLAY_SETUP_SPAN,
                    attributes={
                        BENCHMARK_TASK: task.spec.name,
                        BENCHMARK_ARM: DEFAULT_ARM,
                        BENCHMARK_PURPOSE: DIAGNOSTIC_PURPOSE,
                        SPAN_KIND: "benchmark",
                    },
                ):
                    async with memory_factory(path) as memory:
                        opened = True
                        instance_semaphore = asyncio.Semaphore(SDK_OPERATION_LIMIT)
                        await asyncio.gather(
                            *(
                                measure_one(memory, question, sample, instance_semaphore)
                                for question, sample in questions
                            )
                        )
            except Exception:
                if not opened:
                    # Store-open time is setup rather than caller latency. Still emit one
                    # zero-work error attempt per planned question so missing measurements cannot
                    # masquerade as a complete successful distribution.
                    for _question, sample in questions:
                        with traced_span(
                            selected_tracer,
                            BENCHMARK_DIAGNOSTIC_SPAN,
                            attributes=span_attributes(sample),
                        ) as span:
                            span.set_status(StatusCode.ERROR)
                return

    await asyncio.gather(*(measure_unit(index, unit) for index, unit in enumerate(task.units)))


async def _run_unit(  # noqa: C901 - causal ingest and store-readiness share one lifecycle
    task: LoadedTask,
    unit: EvalUnit,
    data_dir: Path,
    *,
    memory_factory: MemoryFactory,
    options: EvaluationOptions,
    request_semaphore: asyncio.Semaphore,
    response_cache: ResponseCache | None,
    arms: Sequence[Arm] = (PRODUCT_ARM,),
    checkpoint: IngestCheckpoint | None = None,
    tracer: Tracer | None = None,
    on_sample_completed: Callable[[], None] | None = None,
    on_ingested: Callable[[int], None] | None = None,
    on_activity: Callable[[str], None] | None = None,
    on_store_ready: Callable[[], None] | None = None,
    prefetch_speech: Callable[[Sequence[MemoryItem]], None] | None = None,
    model_semaphore: asyncio.Semaphore | None = None,
    run_state: IngestRunState | None = None,
) -> tuple[SampleResult, ...]:
    run_state = run_state or IngestRunState()
    model_semaphore = model_semaphore or asyncio.Semaphore(SDK_OPERATION_LIMIT)
    ordered = tuple((arm, question) for arm in arms for question in unit.questions)
    notify_store_ready = on_store_ready or _ignore_store_ready
    notify_ingested = on_ingested or _ignore_ingested
    notify_activity = on_activity or _ignore_activity
    results: dict[tuple[str, str], SampleResult] = {}
    for arm in arms:
        results.update(
            cached_results(
                response_cache,
                task,
                unit,
                arm=arm,
                predict_only=options.predict_only,
                log_samples=options.log_samples,
                answer_policy=options.answer_policy,
            )
        )
    _report_completions(on_sample_completed, len(results))
    cluster_answers = {
        (sample.arm, str(sample.metadata["cluster_id"])): AnswerOutcome(
            sample.prediction,
            0.0,
            sample.confidence,
            sample.memory_ids,
            sample.evidence,
            cached=True,
        )
        for sample in results.values()
        if task.spec.name == "personamem-v3" and sample.metadata.get("cluster_id")
    }
    if len(results) == len(ordered):
        restored = tuple(results[(arm.name, question.question_id)] for arm, question in ordered)
        return restored if options.predict_only else complete_lifecycle(restored)
    pending_questions = {
        arm.name: _pending_questions(
            unit,
            {name for (arm_name, name) in results if arm_name == arm.name},
        )
        for arm in arms
    }
    cutoffs = _ordered_cutoffs(pending_questions.values())
    memories = tuple(
        sorted(
            unit.memories,
            # Python's sort is stable, so equal causal boundaries retain the adapter's
            # release order.  Source IDs identify evidence; they are not a chronology and
            # using them as a tie-breaker silently reordered dialogue turns and media parts.
            key=memory_end,
        )
    )
    stuffs_context = any(arm.name == "full-context" and pending_questions[arm.name] for arm in arms)
    reading_arms = tuple(
        arm.name for arm in arms if arm.reads_memory and pending_questions[arm.name]
    )
    reads_memory = bool(reading_arms)
    # Ingestion is product setup whenever the product arm is present. This must not change when a
    # caller merely reorders the same arms; a random-only run still owns its required setup.
    memory_arm = (
        DEFAULT_ARM
        if DEFAULT_ARM in reading_arms
        else reading_arms[0]
        if len(reading_arms) == 1
        else SHARED_BENCHMARK_ARM
    )
    ingest_failure_details: list[FailureDetail] = []
    ingested = 0
    ingest_failures = 0
    # pending is the causal cursor every arm reads from; ingested is how much of it a previous
    # run already wrote. They differ only while a resumed store runs ahead of the current cutoff.
    pending = 0
    try:
        # An arm that reads no memory never writes one either, so it neither trusts nor rebuilds
        # a store another run left behind. Resolving the checkpoint inside this block reports a
        # directory owned by another run as the store failure it is.
        if checkpoint is not None and reads_memory:
            ingested, ingest_failure_details = await checkpoint.start(
                memories, cutoffs, memory_factory=memory_factory, data_dir=data_dir
            )
            ingest_failures = len(ingest_failure_details)
            # A resumed store starts the bar where the killed run left it, not at zero.
            notify_ingested(ingested)
        async with memory_factory(data_dir) as memory:
            for cutoff in cutoffs:
                end = prefix_end(memories, cutoff, pending)
                if reads_memory and end > ingested:
                    notify_activity("ingesting")
                    ingest_failures += await ingest(
                        memory,
                        memories[ingested:end],
                        batch_size=options.batch_size,
                        ingest_concurrency=options.ingest_concurrency,
                        on_failure=ingest_failure_details.append,
                        tracer=tracer,
                        mode=options.ingest_mode,
                        arm=memory_arm,
                        on_chunk=ingest_observer(
                            checkpoint, ingested, ingest_failure_details, notify_ingested
                        ),
                        prefetch=prefetch_speech,
                        model_semaphore=model_semaphore,
                        run_state=run_state,
                    )
                    ingested = end
                    if options.deliberate:
                        notify_activity("deliberating")
                        run_state.operations_applied += await model_operation(
                            model_semaphore, lambda: deliberate_after_ingest(memory)
                        )
                pending = end
                context = (
                    full_context(memories[:pending], options.full_context_chars)
                    if stuffs_context
                    else ""
                )
                notify_activity("answering")
                cutoff_results = await _answer_arms(
                    memory,
                    task,
                    unit,
                    arms=arms,
                    questions_by_arm={
                        arm.name: pending_questions[arm.name].get(cutoff, []) for arm in arms
                    },
                    context=context,
                    ingest_failures=ingest_failures,
                    ingest_failure_details=tuple(ingest_failure_details),
                    options=options,
                    request_semaphore=request_semaphore,
                    model_semaphore=model_semaphore,
                    response_cache=response_cache,
                    tracer=tracer,
                    on_sample_completed=on_sample_completed,
                    cluster_answers=cluster_answers,
                )
                results.update(cutoff_results)
                if any(
                    sample.error_code == "_SystemicEmbeddingFailure"
                    for sample in cutoff_results.values()
                ):
                    raise SystemicEmbeddingFailure(
                        "systemic query embedding failure; remaining questions aborted"
                    )
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        if len(results) == len(ordered) and not isinstance(error, SystemicEmbeddingFailure):
            raise
        for arm, question in ordered:
            identity = (arm.name, question.question_id)
            if identity not in results:
                results[identity] = build_sample_result(
                    task,
                    unit,
                    question,
                    error,
                    ingest_failures=ingest_failures,
                    ingest_failure_details=tuple(ingest_failure_details),
                    predict_only=options.predict_only,
                    log_samples=options.log_samples,
                    arm=arm,
                    answer_policy=options.answer_policy,
                )
    else:
        notify_store_ready()
    samples = tuple(results[(arm.name, question.question_id)] for arm, question in ordered)
    return samples if options.predict_only else complete_lifecycle(samples)


async def _answer_arms(
    memory: AsyncMemory,
    task: LoadedTask,
    unit: EvalUnit,
    *,
    arms: Sequence[Arm],
    questions_by_arm: Mapping[str, Sequence[EvalQuestion]],
    context: str,
    ingest_failures: int,
    ingest_failure_details: tuple[FailureDetail, ...],
    options: EvaluationOptions,
    request_semaphore: asyncio.Semaphore,
    response_cache: ResponseCache | None,
    tracer: Tracer | None = None,
    on_sample_completed: Callable[[], None] | None = None,
    model_semaphore: asyncio.Semaphore | None = None,
    cluster_answers: dict[tuple[str, str], AnswerOutcome] | None = None,
) -> dict[tuple[str, str], SampleResult]:
    """Answer one cutoff's pending questions once per arm, against one ingested store."""
    results: dict[tuple[str, str], SampleResult] = {}
    for arm in arms:
        questions = questions_by_arm.get(arm.name, ())
        if not questions:
            continue

        answered = await answer_many(
            memory,
            questions,
            options=options,
            request_semaphore=request_semaphore,
            model_semaphore=model_semaphore,
            on_answer=partial(
                cache_outcome,
                response_cache,
                task,
                unit,
                ingest_failures=ingest_failures,
                arm=arm,
            ),
            arm=arm,
            task_name=task.spec.name,
            unit_id=unit.unit_id,
            context=context,
            tracer=tracer,
            on_complete=on_sample_completed,
            cluster_answers=cluster_answers,
        )
        for question, outcome in zip(questions, answered, strict=True):
            results[(arm.name, question.question_id)] = build_sample_result(
                task,
                unit,
                question,
                outcome,
                ingest_failures=ingest_failures,
                ingest_failure_details=ingest_failure_details,
                predict_only=options.predict_only,
                log_samples=options.log_samples,
                arm=arm,
                answer_policy=options.answer_policy,
            )
    return results


def _pending_questions(
    unit: EvalUnit, completed: Sequence[str] | set[str]
) -> dict[float | None, list[EvalQuestion]]:
    groups: dict[float | None, list[EvalQuestion]] = {}
    for question in unit.questions:
        if question.question_id not in completed:
            groups.setdefault(question.cutoff_seconds, []).append(question)
    return groups


def _ordered_cutoffs(
    groups: Iterable[Mapping[float | None, Sequence[EvalQuestion]]],
) -> tuple[float | None, ...]:
    """Ingest in causal order, with the uncut questions last, across every arm's pending set."""
    values = {cutoff for group in groups for cutoff in group}
    cutoffs: list[float | None] = [*sorted(value for value in values if value is not None)]
    if None in values:
        cutoffs.append(None)
    return tuple(cutoffs)


def _report_completions(callback: Callable[[], None] | None, count: int) -> None:
    if callback is not None:
        for _ in range(count):
            callback()


def _ignore_store_ready() -> None:
    pass


def _ignore_activity(_activity: str) -> None:
    pass


def _ignore_ingested(_written: int) -> None:
    pass


def _activity_summary(activities: Sequence[str | None], total: int) -> str:
    active = tuple((index, phase) for index, phase in enumerate(activities) if phase is not None)
    if not active:
        return "finishing"
    if len(active) == 1:
        index, phase = active[0]
        return f"{phase} unit {index + 1}/{total}"
    counts: dict[str, int] = {}
    for _index, phase in active:
        counts[phase] = counts.get(phase, 0) + 1
    phases = ", ".join(f"{phase} {count}" for phase, count in counts.items())
    return f"{len(active)} active units: {phases}"
