"""Run pinned SenseMem benchmarks through the public SDK."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import os
import re
import sys
import time
from collections.abc import (
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from types import ModuleType
from typing import cast

from opentelemetry.trace import Tracer

from sensemem import (
    ContextBudget,
    MemorySpec,
    Modality,
)
from sensemem._telemetry import (
    SPAN_KIND,
    traced_span,
)
from sensemem.benchmarks.artifact_io import encode_jsonl_bytes, replace_artifact_files
from sensemem.benchmarks.download import acquire_inputs
from sensemem.benchmarks.eval_adapters import (
    LoadedTask,
    MediaManifest,
    MemoryItem,
    load_media_manifest,
    load_task,
    parse_media_manifest,
)
from sensemem.benchmarks.eval_arms import (
    Arm,
    BaselineGenerator,
    generator_arms,
)
from sensemem.benchmarks.eval_artifacts import config_artifact, merged_manifest, write_artifacts
from sensemem.benchmarks.eval_backends import BackendPool
from sensemem.benchmarks.eval_cache import ResponseCache
from sensemem.benchmarks.eval_config import (
    MEDIA_MANIFEST_FILE,
    MODALITY_BY_SUFFIX,
    PARTIAL_SAMPLES_FILE,
    Arguments,
    JudgeConfig,
    build_parser,
    description_cache_path,
    evaluation_config,
    evaluation_memory_config,
    load_memory_config,
    model_config,
    picked,
    require_output,
    resolve_arguments,
    resolve_batch_size,
    resolve_judge_config,
)
from sensemem.benchmarks.eval_console import (
    announce,
    configure_logging,
    deferred_progress,
    progress_reporter,
    render_summary_table,
    uninterpretable_tasks,
)
from sensemem.benchmarks.eval_environment import (
    TORCH,
    nvidia_smi_rows,
    unavailable_server_resources,
)
from sensemem.benchmarks.eval_ingest import IngestRunState
from sensemem.benchmarks.eval_inputs import (
    are_all_cached,
    cache_namespace,
    has_question_cutoffs,
    ingest_digest,
)
from sensemem.benchmarks.eval_judge import apply_judges
from sensemem.benchmarks.eval_metrics import (
    build_comparisons,
)
from sensemem.benchmarks.eval_options import EvaluationOptions
from sensemem.benchmarks.eval_regression import (
    load_result,
    performance_comparisons,
)
from sensemem.benchmarks.eval_report import (
    build_results_document,
    build_task_rows,
    has_regression,
    incomplete_search_replay,
    task_performance,
    with_grounding_loss,
)
from sensemem.benchmarks.eval_results import (
    DEFAULT_ARM,
    SampleResult,
)
from sensemem.benchmarks.eval_runner import run_loaded_task
from sensemem.benchmarks.eval_runtime import MemoryContext, MemoryFactory, SearchReplay
from sensemem.benchmarks.eval_server_metrics import (
    MetricsSnapshot,
    capture_metrics,
    metrics_window,
)
from sensemem.benchmarks.eval_telemetry import (
    BENCHMARK_ARM,
    BENCHMARK_PURPOSE,
    BENCHMARK_TASK,
    BENCHMARK_TASK_SPAN,
    SHARED_BENCHMARK_ARM,
    EvaluationTelemetry,
    ResourceSampler,
)
from sensemem.benchmarks.isolation import BenchmarkRun
from sensemem.benchmarks.model_config import (
    DownloadSettings,
    ModelConfig,
    ServerMetricsOverrides,
)
from sensemem.benchmarks.prepare_media import prepare_task_media
from sensemem.benchmarks.task_catalog import (
    TASKS,
    listing,
)
from sensemem.configuration import (
    FunASRSpeechConfig,
    OpenAIEmbeddingConfig,
    SentenceTransformersEmbeddingConfig,
)
from sensemem.infrastructure.local._lock import DataDirectoryInUseError, DataDirectoryLock

_LEADERBOARD_MODULE: ModuleType | None
try:
    from sensemem.benchmarks import eval_leaderboard as _eval_leaderboard_module
except ImportError:
    _LEADERBOARD_MODULE = None
else:
    _LEADERBOARD_MODULE = _eval_leaderboard_module

# Called once per task, the moment that task has answered every question, and returns the samples
# to keep. It is what lets a run persist and report a finished task without waiting for the rest.
_TaskCompletion = Callable[
    ["LoadedTask", tuple["SampleResult", ...]], Awaitable[tuple["SampleResult", ...]]
]


def main(  # noqa: C901 - offline gates and evaluation share one CLI entry point
    argv: Sequence[str] | None = None, *, prog: str | None = None
) -> int:
    """Parse one reproducible evaluation sweep and write its artifacts."""
    arguments_cli = list(sys.argv[1:] if argv is None else argv)
    if arguments_cli[:1] == ["serve"]:
        if _LEADERBOARD_MODULE is None:
            raise RuntimeError("serving requires uvicorn and sensemem[benchmarks,server]")
        return int(
            _LEADERBOARD_MODULE.main(
                arguments_cli[1:], prog=f"{prog or 'sensemem-bench eval'} serve"
            )
        )
    parser = build_parser(prog)
    parsed = parser.parse_args(arguments_cli)
    # The configuration file can name the corpus root, so it is read before the listing and
    # before argument resolution derives the default output directory from that root.
    config_path = (
        None if parsed.memory_config is None else parsed.memory_config.expanduser().resolve()
    )
    try:
        memory_config, overrides = load_memory_config(config_path)
    except ValueError as error:
        parser.error(str(error))
    configure_logging(picked(parsed.verbosity, overrides.run.verbosity, "INFO"))
    if parsed.deliberate and (memory_config is None or memory_config.consolidation is None):
        # Refused rather than silently ignored: without a consolidation backend every round
        # would raise, and a run that quietly skipped the loop would report a score for it.
        parser.error("--deliberate requires a --config declaring a `consolidation` section")
    download = DownloadSettings.resolve(
        overrides.download,
        benchmarks_root=parsed.benchmarks_root,
        data_root=parsed.data_root,
    )
    download.apply_environment()
    list_mode = _list_mode(parsed)
    if list_mode is not None:
        print(listing(download.benchmarks_root, list_mode))
        return 0
    arguments = resolve_arguments(parser, parsed, download, overrides=overrides)
    if arguments.ingest_concurrency > 1 and (
        memory_config is None
        or memory_config.formation is not None
        or memory_config.speech is not None
        or memory_config.face is not None
    ):
        parser.error(
            "--ingest-concurrency above 1 requires an explicit --config without formation, "
            "speech, or face backends"
        )
    if parsed.check_integrity:
        manifest, manifest_directory = load_media_manifest(arguments.media_manifest)
        loaded = _load_tasks(arguments, manifest, manifest_directory)
        if arguments.ingest_concurrency > 1 and has_question_cutoffs(loaded):
            parser.error(
                "--ingest-concurrency above 1 requires static units without question cutoffs"
            )
        print(
            encode_jsonl_bytes(
                (
                    {
                        "status": "ok",
                        "tasks": [
                            {
                                "task": task.spec.name,
                                "dataset_sha256": task.dataset_sha256,
                                "evaluation_sha256": task.evaluation_sha256,
                                "unit_count": len(task.units),
                                "question_count": sum(len(unit.questions) for unit in task.units),
                            }
                            for task in loaded
                        ],
                    },
                )
            ).decode(),
            end="",
        )
        return 0
    if memory_config is None and DEFAULT_ARM in arguments.arms:
        parser.error("the sensemem arm requires --config with an explicit embedding section")
    try:
        base_config = model_config(
            arguments.model_args,
            memory_config=memory_config,
        )
        judge_config = resolve_judge_config(base_config, arguments, overrides=overrides)
    except ValueError as error:
        parser.error(str(error))
    require_output(arguments.output_path, overwrite=arguments.overwrite, resume=arguments.resume)
    manifest, manifest_directory = load_media_manifest(arguments.media_manifest)
    if arguments.download:
        for name in arguments.tasks:
            acquire_inputs(
                TASKS[name],
                arguments.benchmarks_root,
                include_dataset=name not in arguments.dataset_overrides,
            )
    generated_manifest: dict[str, object] = {}
    for name in arguments.tasks:
        with deferred_progress(
            f"preparing {name} media", "source", enabled=not arguments.quiet
        ) as progress:
            prepared = prepare_task_media(
                TASKS[name],
                root=arguments.benchmarks_root,
                dataset_path=arguments.dataset_overrides.get(
                    name, TASKS[name].dataset_path(arguments.benchmarks_root)
                ),
                media_root=arguments.media_overrides.get(name),
                manifest=manifest,
                limit=arguments.limit,
                offset=arguments.offset,
                download=arguments.download,
                announce=None if arguments.quiet else announce,
                on_progress=None if arguments.quiet else progress,
            )
        if prepared is not None:
            generated_manifest[name] = prepared
    if generated_manifest:
        merged = merged_manifest(manifest, manifest_directory, generated_manifest)
        manifest = parse_media_manifest(merged)
        effective_manifest = arguments.output_path / MEDIA_MANIFEST_FILE
        effective_manifest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        replace_artifact_files(((effective_manifest, encode_jsonl_bytes((merged,))),))
        arguments = replace(arguments, media_manifest=effective_manifest)
        manifest_directory = effective_manifest.parent
    loaded = _load_tasks(arguments, manifest, manifest_directory)
    if arguments.ingest_concurrency > 1 and has_question_cutoffs(loaded):
        parser.error("--ingest-concurrency above 1 requires static units without question cutoffs")
    config = evaluation_config(base_config, loaded, memory_config=memory_config)
    memory_config = evaluation_memory_config(
        memory_config,
        config,
        gen_kwargs=arguments.gen_kwargs,
        seed=arguments.seed,
        device=arguments.device,
    )
    batch_sizes = {task.spec.name: resolve_batch_size(arguments, task) for task in loaded}
    run_state = IngestRunState()
    samples, duration, performance, resources, embedding_warmup_count = _execute(
        loaded,
        arguments,
        config,
        judge_config,
        batch_sizes,
        memory_config=memory_config,
        server_metrics=overrides.server_metrics,
        run_state=run_state,
    )
    results = build_results_document(
        arguments,
        config,
        judge_config,
        loaded,
        samples,
        duration,
        batch_sizes,
        performance,
        memory_config=memory_config,
        operations_applied=run_state.operations_applied,
        resources=resources,
        embedding_warmup_count=embedding_warmup_count,
    )
    comparisons = build_comparisons(arguments, loaded, samples)
    if comparisons:
        results["comparisons"] = comparisons
    performance_rows = (
        []
        if not arguments.performance_budgets
        else performance_comparisons(
            results,
            load_result(cast(Path, arguments.compare)),
            arguments.performance_budgets,
        )
    )
    if performance_rows:
        results["performance_comparisons"] = performance_rows
    write_artifacts(
        arguments,
        samples,
        results,
        config_bytes=config_artifact(
            arguments,
            config,
            judge_config,
            memory_config,
            download,
            overrides.server_metrics,
        ),
    )
    for reason in uninterpretable_tasks(results):
        announce(f"UNINTERPRETABLE: {reason}")
    if not arguments.quiet:
        print(render_summary_table(results))
    has_errors = _execution_has_errors(samples, results)
    regressed = arguments.fail_on_regression and has_regression(
        comparisons, threshold=arguments.regression_threshold
    )
    performance_regressed = arguments.fail_on_regression and any(
        row["regressed"] is True for row in performance_rows
    )
    return int(has_errors or regressed or performance_regressed)


def _execution_has_errors(samples: Sequence[SampleResult], results: Mapping[str, object]) -> bool:
    sample_errors = any(
        sample.error_code is not None
        or sample.ingest_failure_count
        or sample.retrieval_diagnostic_error is not None
        for sample in samples
    )
    rows = cast(Sequence[Mapping[str, object]], results["tasks"])
    return sample_errors or (results["search_replay"] is True and incomplete_search_replay(rows))


def _load_tasks(
    arguments: Arguments,
    manifest: MediaManifest | None,
    manifest_directory: Path | None,
) -> tuple[LoadedTask, ...]:
    return tuple(
        _with_fallback_reference(
            load_task(
                TASKS[name],
                root=arguments.benchmarks_root,
                dataset_path=arguments.dataset_overrides.get(name),
                media_root=arguments.media_overrides.get(name),
                media_manifest=manifest,
                manifest_directory=manifest_directory,
                limit=arguments.limit,
                offset=arguments.offset,
                verify_digest=not (
                    arguments.allow_unverified_data and name in arguments.dataset_overrides
                ),
            ),
            arguments.fallback_reference_at,
        )
        for name in arguments.tasks
    )


def _with_fallback_reference(
    task: LoadedTask,
    fallback: datetime | None,
) -> LoadedTask:
    """Fill only missing question clocks after dataset and corpus clocks are resolved."""
    if fallback is None:
        return task
    count = sum(question.reference_at is None for unit in task.units for question in unit.questions)
    if not count:
        return task
    return replace(
        task,
        units=tuple(
            replace(
                unit,
                questions=tuple(
                    question
                    if question.reference_at is not None
                    else replace(question, reference_at=fallback)
                    for question in unit.questions
                ),
            )
            for unit in task.units
        ),
        fallback_reference_at=fallback,
        fallback_reference_question_count=count,
    )


@contextmanager
def _benchmark_device_lock(
    device: str | None,
    *,
    enabled: bool,
    quiet: bool,
    lock_root: Path | None = None,
) -> Iterator[None]:
    normalized = (device or "auto").strip().lower()
    if not enabled or normalized == "cpu":
        yield
        return
    index = _cuda_logical_index(normalized)
    identity = _physical_cuda_identity(normalized)
    if identity is None:
        yield
        return
    if lock_root is None:
        owner = hashlib.sha256(str(Path.home()).encode()).hexdigest()[:12]
        lock_root = Path(os.environ.get("XDG_RUNTIME_DIR", gettempdir())) / (
            f"sensemem-benchmark-{owner}"
        )
    device_key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    # ponytail: one local model process per CUDA device; use a VRAM budget only when
    # concurrent model pools demonstrate a safe, repeatable throughput gain.
    announced = False
    while True:
        try:
            lock = DataDirectoryLock(lock_root / f"cuda-{device_key}")
            break
        except DataDirectoryInUseError:
            if not quiet and not announced:
                announce(f"waiting for local CUDA device {index}")
                announced = True
            time.sleep(0.1)
    try:
        yield
    finally:
        lock.close()


def _cuda_logical_index(device: str) -> int:
    match = re.fullmatch(r"cuda(?::(\d+))?", device)
    return int(match.group(1)) if match is not None and match.group(1) is not None else 0


def _physical_cuda_identity(device: str) -> str | None:
    logical_index = _cuda_logical_index(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = str(logical_index)
    else:
        exposed = tuple(value.strip() for value in visible.split(",") if value.strip())
        if logical_index >= len(exposed) or exposed[logical_index] == "-1":
            return None
        selected = exposed[logical_index]
    uuids = _nvidia_device_uuids()
    if selected.isdecimal():
        physical_index = int(selected)
        return uuids.get(physical_index, f"index:{physical_index}").casefold()
    normalized = selected.casefold()
    return next(
        (
            uuid.casefold()
            for uuid in uuids.values()
            if uuid.casefold().startswith(normalized) or normalized.startswith(uuid.casefold())
        ),
        normalized,
    )


def _nvidia_device_uuids() -> dict[int, str]:
    return {
        int(row[0]): row[1]
        for row in nvidia_smi_rows("index,uuid")
        if len(row) >= 2 and row[0].isdecimal() and row[1]
    }


def _release_device_memory(device: str | None) -> None:
    if (device or "auto").strip().lower() == "cpu":
        return
    gc.collect()
    if TORCH is not None and TORCH.cuda.is_available():
        TORCH.cuda.empty_cache()


def _server_metrics_endpoints(settings: ServerMetricsOverrides) -> dict[str, str]:
    return {
        name: url
        for name, url in (
            ("generation", settings.generation_url),
            ("embedding", settings.embedding_url),
        )
        if url is not None
    }


def _server_metric_results(
    settings: ServerMetricsOverrides,
    starts: Mapping[str, MetricsSnapshot],
    *,
    config: ModelConfig,
    memory_config: MemorySpec | None,
    cache_only: bool,
    excluded: Mapping[str, Sequence[tuple[MetricsSnapshot, MetricsSnapshot]]] | None = None,
) -> dict[str, object]:
    endpoints = _server_metrics_endpoints(settings)
    result: dict[str, object] = {}
    for name, url in endpoints.items():
        if cache_only:
            result[name] = {
                "scope": "server_process_global",
                "exclusive_attribution": False,
                "metrics_url": url,
                "status": "skipped",
                "reason": "the response cache issued no product model requests",
            }
        else:
            result[name] = {
                **metrics_window(
                    url,
                    starts[name],
                    timeout_seconds=settings.timeout_seconds,
                    excluded=() if excluded is None else excluded.get(name, ()),
                ),
                "phase": "product_execution_including_post_answer_search_replay",
                "judge_traffic_excluded": True,
            }
    if "generation" not in result:
        result["generation"] = unavailable_server_resources(base_url=config.generation_base_url)
    if "embedding" not in result and memory_config is not None:
        embedding = memory_config.embedding
        if isinstance(embedding, OpenAIEmbeddingConfig) and embedding.base_url is not None:
            result["embedding"] = unavailable_server_resources(base_url=embedding.base_url)
    return result


def _execute(
    loaded: Sequence[LoadedTask],
    arguments: Arguments,
    config: ModelConfig,
    judge_config: JudgeConfig,
    batch_sizes: Mapping[str, int],
    *,
    memory_config: MemorySpec | None,
    server_metrics: ServerMetricsOverrides | None = None,
    run_state: IngestRunState | None = None,
) -> tuple[
    tuple[SampleResult, ...],
    float,
    Mapping[str, Mapping[str, Mapping[str, object]]],
    Mapping[str, object],
    int,
]:
    run_state = run_state or IngestRunState()
    needs_speech = any(
        isinstance(atom, Path)
        and MODALITY_BY_SUFFIX.get(atom.suffix.casefold()) in {Modality.AUDIO, Modality.VIDEO}
        for task in loaded
        for unit in task.units
        for item in unit.memories
        for atom in item.content
    )
    started = time.perf_counter()
    telemetry = EvaluationTelemetry()
    sampler = ResourceSampler(
        storage_roots=tuple(
            BenchmarkRun.path_for(arguments.data_root, task.spec.name, arguments.run_id)
            for task in loaded
        )
    )
    metrics_settings = ServerMetricsOverrides() if server_metrics is None else server_metrics
    metric_endpoints = _server_metrics_endpoints(metrics_settings)
    metric_exclusions: dict[str, list[tuple[MetricsSnapshot, MetricsSnapshot]]] = {
        name: [] for name in metric_endpoints
    }
    embedding_warmup_count = 0
    streamed: set[str] = set()

    @contextmanager
    def exclude_task_tail_measurement() -> Iterator[None]:
        # Immediate judging and interim reporting are not product execution. Keep them between
        # tasks for prompt feedback, but split both client and server accounting around the whole
        # tail so a shared endpoint does not acquire judge traffic and client measurements do not
        # acquire bootstrap aggregation or console-I/O time.
        with sampler.exclude():
            before = {
                name: capture_metrics(
                    url,
                    timeout_seconds=metrics_settings.timeout_seconds,
                )
                for name, url in metric_endpoints.items()
            }
            try:
                yield
            finally:
                for name, url in metric_endpoints.items():
                    metric_exclusions[name].append(
                        (
                            before[name],
                            capture_metrics(
                                url,
                                timeout_seconds=metrics_settings.timeout_seconds,
                            ),
                        )
                    )

    try:
        task_completed = _task_completion(
            arguments,
            judge_config=judge_config,
            batch_sizes=batch_sizes,
            telemetry=telemetry,
            memory_config=memory_config,
            streamed=streamed,
            exclude_task_tail_measurement=exclude_task_tail_measurement,
        )
        response_cache = (
            None
            if arguments.use_cache is None
            else ResponseCache(
                arguments.use_cache,
                arguments.run_id,
                cache_namespace(
                    arguments,
                    config,
                    batch_sizes,
                    memory_config=memory_config,
                ),
            )
        )
        pool: BackendPool | None = None
        memory_factory: MemoryFactory
        arm_specs = tuple(Arm(name) for name in arguments.arms)
        all_cached = response_cache is not None and are_all_cached(
            response_cache, loaded, arm_specs
        )
        devices = _evaluation_devices(arguments.device, memory_config, needs_speech=needs_speech)
        with ExitStack() as device_locks:
            for device in devices:
                device_locks.enter_context(
                    _benchmark_device_lock(
                        device,
                        enabled=arguments.device_lock and not all_cached,
                        quiet=arguments.quiet,
                    )
                )
            try:
                prefetch_speech: Callable[[Sequence[MemoryItem]], None] | None = None
                if all_cached:
                    memory_factory = _cache_only_memory
                else:
                    pool = BackendPool(
                        config,
                        memory_config=memory_config,
                        tracer=telemetry.tracer,
                        description_cache=description_cache_path(arguments, memory_config),
                    )
                    memory_factory = pool.memory
                    prefetch_speech = pool.prefetch_speech
                    embedding_warmup_count = pool.embedding_warmup_count
                metric_starts = (
                    {}
                    if all_cached
                    else {
                        name: capture_metrics(url, timeout_seconds=metrics_settings.timeout_seconds)
                        for name, url in _server_metrics_endpoints(metrics_settings).items()
                    }
                )
                with sampler:
                    samples = asyncio.run(
                        _run_all(
                            loaded,
                            arguments,
                            batch_sizes=batch_sizes,
                            memory_factory=memory_factory,
                            response_cache=response_cache,
                            tracer=telemetry.tracer,
                            config=config,
                            memory_config=memory_config,
                            on_task_complete=task_completed,
                            prefetch_speech=prefetch_speech,
                            run_state=run_state,
                        )
                    )
                model_servers = _server_metric_results(
                    metrics_settings,
                    metric_starts,
                    config=config,
                    memory_config=memory_config,
                    cache_only=all_cached,
                    excluded=metric_exclusions,
                )
            finally:
                if pool is not None:
                    closing = pool
                    pool = None
                    memory_factory = _cache_only_memory
                    closing.close()
                    del closing
                    for device in devices:
                        _release_device_memory(device)
                if response_cache is not None:
                    response_cache.close()
        if not arguments.predict_only:
            samples = asyncio.run(
                apply_judges(
                    loaded,
                    samples,
                    arguments=arguments,
                    config=judge_config,
                    tracer=telemetry.tracer,
                    already_judged=streamed,
                )
            )
        samples = with_grounding_loss(samples, telemetry)
        performance = task_performance(telemetry, loaded, samples, arguments.arms)
        duration = time.perf_counter() - started
        resources = sampler.json(wall_seconds=duration)
        resources["model_servers"] = model_servers
        return samples, duration, performance, resources, embedding_warmup_count
    finally:
        telemetry.close()


def _task_completion(
    arguments: Arguments,
    *,
    judge_config: JudgeConfig,
    batch_sizes: Mapping[str, int],
    telemetry: EvaluationTelemetry,
    memory_config: MemorySpec | None,
    streamed: set[str],
    exclude_task_tail_measurement: Callable[[], AbstractContextManager[None]] | None = None,
) -> _TaskCompletion:
    """Build the per-task tail: persist, then normally score and print immediately.

    Answering every task before anything is written or scored made a multi-task run silent until
    the last task finished, and made an outage during the last task discard every earlier task's
    answers. The crash copy costs nothing and is always written. Immediate judging is the default;
    --no-stream-results retains the former run-global scoring pass when explicitly requested.
    """
    partial_path = arguments.output_path / PARTIAL_SAMPLES_FILE
    partial_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial_path.unlink(missing_ok=True)

    async def completed(
        task: LoadedTask, task_samples: tuple[SampleResult, ...]
    ) -> tuple[SampleResult, ...]:
        task_samples = with_grounding_loss(task_samples, telemetry)
        # Persist before anything that can fail. Judging raises on a missing extra or an
        # unreachable judge, and it must not be able to take the answers down with it: those cost
        # a full ingest to produce, the scores cost one more model call. Writing here also makes
        # the copy identical whether or not the run streams -- predictions, not scores.
        # Same 0o600 as the real artifacts, and fsynced: a copy that survives only a clean exit
        # would not survive the failures it exists for.
        descriptor = os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "wb") as partial:
            partial.write(encode_jsonl_bytes(sample.json() for sample in task_samples))
            partial.flush()
            os.fsync(partial.fileno())
        if arguments.stream_results:
            measurement_exclusion = (
                exclude_task_tail_measurement
                if exclude_task_tail_measurement is not None
                and (not arguments.predict_only or not arguments.quiet)
                else nullcontext
            )
            with measurement_exclusion():
                if not arguments.predict_only:
                    task_samples = await apply_judges(
                        (task,),
                        task_samples,
                        arguments=arguments,
                        config=judge_config,
                        tracer=telemetry.tracer,
                    )
                streamed.update(sample.sample_id for sample in task_samples)
                if not arguments.quiet:
                    announce(f"interim results for {task.spec.name} (later tasks still running)")
                    print(
                        render_summary_table(
                            {
                                "tasks": build_task_rows(
                                    arguments,
                                    (task,),
                                    task_samples,
                                    batch_sizes,
                                    task_performance(
                                        telemetry, (task,), task_samples, arguments.arms
                                    ),
                                    memory_config=memory_config,
                                )
                            }
                        )
                    )
        return task_samples

    return completed


async def _run_all(
    tasks: Sequence[LoadedTask],
    arguments: Arguments,
    *,
    batch_sizes: Mapping[str, int],
    memory_factory: MemoryFactory,
    response_cache: ResponseCache | None,
    tracer: Tracer,
    config: ModelConfig | None = None,
    memory_config: MemorySpec | None = None,
    on_task_complete: _TaskCompletion | None = None,
    prefetch_speech: Callable[[Sequence[MemoryItem]], None] | None = None,
    run_state: IngestRunState | None = None,
) -> tuple[SampleResult, ...]:
    run_state = run_state or IngestRunState()
    generated_arms = generator_arms(arguments.arms, tuple(task.spec.name for task in tasks))
    generator = (
        None
        if config is None or not generated_arms
        else BaselineGenerator(
            config,
            seed=arguments.seed,
            gen_kwargs=arguments.gen_kwargs,
            generation=memory_config,
            tracer=tracer,
        )
    )
    arms = tuple(
        Arm(
            name,
            generator=generator if name in generated_arms else None,
            seed=arguments.seed,
            allow_partial_sources=(
                arguments.compile_allow_partial_sources if name == "compile" else False
            ),
        )
        for name in arguments.arms
    )
    try:
        return await _run_arms(
            tasks,
            arguments,
            arms=arms,
            batch_sizes=batch_sizes,
            memory_factory=memory_factory,
            response_cache=response_cache,
            tracer=tracer,
            on_task_complete=on_task_complete,
            memory_config=memory_config,
            prefetch_speech=prefetch_speech,
            run_state=run_state,
        )
    finally:
        if generator is not None:
            await generator.close()


async def _run_arms(
    tasks: Sequence[LoadedTask],
    arguments: Arguments,
    *,
    arms: Sequence[Arm],
    batch_sizes: Mapping[str, int],
    memory_factory: MemoryFactory,
    response_cache: ResponseCache | None,
    tracer: Tracer,
    on_task_complete: _TaskCompletion | None = None,
    memory_config: MemorySpec | None = None,
    prefetch_speech: Callable[[Sequence[MemoryItem]], None] | None = None,
    run_state: IngestRunState | None = None,
) -> tuple[SampleResult, ...]:
    compile_budget = ContextBudget(
        max_items=arguments.compile_max_items,
        max_chars=arguments.compile_max_chars,
    )
    results: list[SampleResult] = []
    deferred_searches: list[SearchReplay] = []
    for task in tasks:
        sample_count = sum(len(unit.questions) for unit in task.units) * len(arms)
        if not arguments.quiet:
            announce(f"running {task.spec.name} ({len(task.units)} units, {sample_count} samples)")
        with (
            progress_reporter(
                f"running {task.spec.name}",
                "sample",
                total=sample_count,
                enabled=not arguments.quiet,
            ) as progress,
            # A unit writes its whole corpus before it answers one question, so the sample bar
            # can sit at zero for an hour on a media task. The ingest bar is what proves that
            # hour is progress. Its total is fixed for the task and known only to the runner,
            # which is why the bar is deferred until the runner reports it; a second total would
            # be rejected rather than resize the bar.
            deferred_progress(
                f"ingesting {task.spec.name}", "item", enabled=not arguments.quiet
            ) as ingest_progress,
            traced_span(
                tracer,
                BENCHMARK_TASK_SPAN,
                attributes={
                    BENCHMARK_TASK: task.spec.name,
                    BENCHMARK_ARM: SHARED_BENCHMARK_ARM,
                    BENCHMARK_PURPOSE: "orchestration",
                    SPAN_KIND: "benchmark",
                },
            ),
        ):
            run = BenchmarkRun(
                arguments.data_root,
                task.spec.name,
                arguments.run_id,
                resume=arguments.resume,
            )
            task_samples = await run_loaded_task(
                task,
                run=run,
                memory_factory=memory_factory,
                prefetch_speech=prefetch_speech,
                options=EvaluationOptions(
                    batch_size=batch_sizes[task.spec.name],
                    unit_concurrency=arguments.unit_concurrency,
                    ingest_concurrency=arguments.ingest_concurrency,
                    stream_answers=arguments.stream_answers,
                    request_concurrency=arguments.request_concurrency,
                    recall_limit=arguments.recall_limit,
                    predict_only=arguments.predict_only,
                    log_samples=arguments.log_samples,
                    full_context_chars=arguments.full_context_chars,
                    compile_budget=compile_budget,
                    answer_policy=arguments.answer_policy,
                    ingest_mode=arguments.ingest,
                    deliberate=arguments.deliberate,
                ),
                response_cache=response_cache,
                arms=arms,
                ingest_digest=ingest_digest(task, arguments, memory_config),
                tracer=tracer,
                on_progress=progress,
                on_ingest_progress=ingest_progress,
                on_activity=progress.set_activity,
                on_search_replay_ready=deferred_searches.append,
                run_state=run_state,
            )
        if on_task_complete is not None:
            task_samples = await on_task_complete(task, task_samples)
        results.extend(task_samples)
    # This is deliberately a run-global second pass. Replaying one task while a later task is
    # still answering changes shared model-service load and contaminates both latency families.
    if arguments.search_replay:
        for search_replay in deferred_searches:
            await search_replay()
    return tuple(results)


def _evaluation_devices(
    explicit: str | None,
    config: MemorySpec | None,
    *,
    needs_speech: bool = True,
) -> tuple[str | None, ...]:
    if config is None:
        return (explicit,)
    configured = []
    if isinstance(config.embedding, SentenceTransformersEmbeddingConfig):
        configured.append(explicit or config.embedding.device or "auto")
    # The speech backend is only constructed when a unit carries audio or video, so a text-only
    # run must not hold the GPU lock for a model it never loads: that lock serialized every
    # text task behind one video run on a shared card.
    if needs_speech and isinstance(config.speech, FunASRSpeechConfig):
        configured.append(explicit or config.speech.device)
    devices: dict[str, str | None] = {}
    for device in sorted(set(configured)):
        normalized = device.strip().lower()
        identity = None if normalized == "cpu" else _physical_cuda_identity(normalized)
        devices.setdefault(identity or normalized, None if normalized == "auto" else device)
    return tuple(devices[identity] for identity in sorted(devices))


def _cache_only_memory(_path: Path) -> MemoryContext:
    raise RuntimeError("response cache was incomplete after the cache-only preflight")


def _list_mode(parsed: argparse.Namespace) -> str | None:
    if parsed.list_tasks:
        return "all"
    return {
        "list": "all",
        "list_groups": "groups",
        "list_subtasks": "tasks",
        "list_tags": "tags",
    }.get(parsed.tasks)


if __name__ == "__main__":
    raise SystemExit(main())
