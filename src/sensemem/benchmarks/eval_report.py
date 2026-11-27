"""Evaluation metrics, provenance and final report assembly."""

from __future__ import annotations

import platform
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import cast

from sensemem import (
    DEFAULT_FUNASR_MODEL_ID,
    MemorySpec,
    Modality,
)
from sensemem.benchmarks.baseline_prompts import (
    BLIND_PROMPT_VERSION,
    COMPILE_SURFACE_PROMPT_VERSION,
    FULL_CONTEXT_PROMPT_VERSION,
)
from sensemem.benchmarks.eval_adapters import (
    LoadedTask,
)
from sensemem.benchmarks.eval_arms import (
    answers_through_ask,
    requests_answer_policy,
)
from sensemem.benchmarks.eval_config import (
    MODALITY_BY_SUFFIX,
    RETRIEVAL_CANDIDATE_LIMIT,
    Arguments,
    JudgeConfig,
    memory_config_payload,
    resolve_generation_temperature,
)
from sensemem.benchmarks.eval_environment import (
    acceleration_runtime_metadata,
    hardware_metadata,
    source_metadata,
)
from sensemem.benchmarks.eval_judge import JUDGE_REQUEST_VERSION
from sensemem.benchmarks.eval_metrics import (
    abstentions,
    answer_retrieval_candidate_limit,
    blind_baseline_rows,
    compute_task_metrics,
    in_run_blind_rows,
    resolve_evidence_budget_chars,
)
from sensemem.benchmarks.eval_results import (
    DEFAULT_ARM,
    EVAL_SCHEMA_VERSION,
    SampleResult,
)
from sensemem.benchmarks.eval_runtime import (
    EVAL_RUNNER_VERSION,
    SDK_OPERATION_LIMIT,
    SDK_STREAM_LIMIT,
)
from sensemem.benchmarks.eval_telemetry import (
    EvaluationTelemetry,
)
from sensemem.benchmarks.model_config import (
    ModelConfig,
)
from sensemem.benchmarks.official_scorers import (
    retrieval_gold_ids,
)
from sensemem.benchmarks.prompts import task_answer_policy, task_answer_surface
from sensemem.benchmarks.task_catalog import (
    TASKS,
)
from sensemem.configuration import (
    FunASRSpeechConfig,
    OpenAIGenerationConfig,
    SentenceTransformersEmbeddingConfig,
)
from sensemem.models._openai_protocol import reasoning_model
from sensemem.models.base import (
    EmbedTask,
)


def task_performance(
    telemetry: EvaluationTelemetry,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    arms: Sequence[str],
) -> dict[str, dict[str, Mapping[str, object]]]:
    """Read each (task, arm)'s aggregated spans.

    EvaluationTelemetry.result is a non-destructive read, so a task that has just finished can
    be summarised mid-run and summarised again in the final document. A mid-run read is missing
    the run-global standalone search replay, which by design runs only once every task has
    answered; the final read is the complete one.
    """
    return {
        task.spec.name: {
            arm: telemetry.result(
                task.spec.name,
                arm=arm,
                question_count=sum(
                    sample.task == task.spec.name and sample.arm == arm and not sample.cached
                    for sample in samples
                ),
            )
            for arm in arms
        }
        for task in tasks
    }


def with_grounding_loss(
    samples: Sequence[SampleResult],
    telemetry: EvaluationTelemetry,
) -> tuple[SampleResult, ...]:
    """Attach each answer's inline-budget loss, so it reads apart from retrieval loss.

    The recall plan's shape rides along: it is the same per-sample join, and a per-question
    shape is what tells a reader which question classes the planner actually reshaped.
    """
    return tuple(
        sample
        if sample.cached or (grounding := telemetry.sample_grounding(sample.sample_id)) is None
        else replace(
            sample,
            dropped_hits=grounding.dropped_hits,
            recall_shape=grounding.recall_shape,
            grounding_request=grounding.request,
            trace_id=sample.trace_id or grounding.trace_id,
            answer_span_id=sample.answer_span_id or grounding.answer_span_id,
        )
        for sample in samples
    )


def incomplete_search_replay(rows: Sequence[Mapping[str, object]]) -> bool:
    for row in rows:
        if row.get("arm") != DEFAULT_ARM:
            continue
        performance = row.get("performance")
        search = performance.get("search_e2e") if isinstance(performance, Mapping) else None
        if isinstance(search, Mapping) and search.get("complete") is False:
            return True
    return False


def build_task_rows(
    arguments: Arguments,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    batch_sizes: Mapping[str, int],
    performance: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    memory_config: MemorySpec | None,
) -> list[dict[str, object]]:
    """Score one row per (task, arm).

    Separate from _results so a task that has finished answering can be scored and printed on
    its own, with exactly the arithmetic the final document uses, while later tasks still run.
    """
    # A blind arm run here is the same control as an external --blind document, measured on the
    # same inputs, so it satisfies the blind control too. An explicitly supplied document still
    # wins: the caller named it.
    blind_rows = {
        **in_run_blind_rows(arguments, tasks, samples),
        **blind_baseline_rows(
            arguments.blind_baseline, tasks, answer_policy=arguments.answer_policy
        ),
    }
    product_candidate_limit = answer_retrieval_candidate_limit(
        arguments.recall_limit,
        memory_config,
    )
    task_rows = []
    for task, arm in ((task, arm) for task in tasks for arm in arguments.arms):
        selected = tuple(
            sample for sample in samples if sample.task == task.spec.name and sample.arm == arm
        )
        metrics = compute_task_metrics(
            task,
            selected,
            arguments,
            blind_rows.get(task.spec.name),
            arm=arm,
            retrieval_candidate_limit=(
                product_candidate_limit
                if arm == DEFAULT_ARM
                else RETRIEVAL_CANDIDATE_LIMIT
                if (
                    arm == "random"
                    and any(retrieval_gold_ids(sample.metadata) for sample in selected)
                )
                else arguments.recall_limit
            ),
        )
        task_rows.append(
            {
                "arm": arm,
                "task": task.spec.name,
                # The request policy, so a run that asked for a committed answer is not
                # byte-indistinguishable from every earlier run of the same task.
                "answer_policy": (
                    task_answer_policy(task.spec.name, arguments.answer_policy)
                    if requests_answer_policy(arm, task.spec.name)
                    else None
                ),
                "benchmark": task.spec.benchmark,
                "variant": task.spec.variant,
                "adapter_version": task.spec.adapter_version,
                "source_repository": task.spec.repository,
                "source_revision": task.spec.revision,
                "media_source": (
                    None
                    if task.spec.media_source is None
                    else {
                        "release": task.spec.media_source.release,
                        "repository": task.spec.media_source.repository,
                        "revision": task.spec.media_source.revision,
                        "patterns": task.spec.media_source.patterns,
                        "acquirer": task.spec.media_source.acquirer,
                    }
                ),
                "dataset_path": str(task.dataset_path),
                "dataset_sha256": task.dataset_sha256,
                "input_sha256": dict(task.input_sha256),
                "evaluation_sha256": task.evaluation_sha256,
                "fallback_reference_at": (
                    None
                    if task.fallback_reference_at is None
                    else task.fallback_reference_at.isoformat()
                ),
                "fallback_reference_question_count": task.fallback_reference_question_count,
                "batch_size": batch_sizes[task.spec.name],
                "input_modalities": _task_modalities(task),
                "performance": dict(performance.get(task.spec.name, {}).get(arm, {})),
                **metrics,
            }
        )

    return task_rows


def build_results_document(
    arguments: Arguments,
    config: ModelConfig,
    judge_config: JudgeConfig,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
    duration_seconds: float,
    batch_sizes: Mapping[str, int],
    performance: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    memory_config: MemorySpec | None = None,
    resources: Mapping[str, object] | None = None,
    embedding_warmup_count: int | None = None,
    operations_applied: int = 0,
) -> dict[str, object]:
    fallback_reference_at = arguments.fallback_reference_at
    task_rows = build_task_rows(
        arguments,
        tasks,
        samples,
        batch_sizes,
        performance,
        memory_config=memory_config,
    )
    media_roots = {}
    for name in arguments.tasks:
        path = arguments.media_overrides.get(name) or TASKS[name].media_root(
            arguments.benchmarks_root
        )
        media_roots[name] = None if path is None else str(path)
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "runner_version": EVAL_RUNNER_VERSION,
        "run_id": arguments.run_id,
        "status": (
            "completed_with_errors"
            if any(
                sample.error_code is not None
                or sample.ingest_failure_count
                or sample.retrieval_diagnostic_error is not None
                for sample in samples
            )
            or (arguments.search_replay and incomplete_search_replay(task_rows))
            else "completed"
        ),
        "duration_seconds": duration_seconds,
        "seed": arguments.seed,
        "seeds": arguments.seeds,
        "bootstrap_samples": arguments.bootstrap_samples,
        "predict_only": arguments.predict_only,
        "search_replay": arguments.search_replay,
        "log_samples": arguments.log_samples,
        "response_cache": None if arguments.use_cache is None else str(arguments.use_cache),
        "resume": arguments.resume,
        "cached_response_count": sum(sample.cached for sample in samples),
        "cached_judge_count": sum(sample.judge_cached for sample in samples),
        "abstentions": abstentions(samples),
        "blind": arguments.blind,
        "blind_baseline": (
            None if arguments.blind_baseline is None else str(arguments.blind_baseline)
        ),
        "controls_complete": all(
            cast(Mapping[str, object], row["controls"])["interpretable"] for row in task_rows
        ),
        "allow_unverified_data": arguments.allow_unverified_data,
        "limit": arguments.limit,
        "offset": arguments.offset,
        "fallback_reference_at": (
            None if fallback_reference_at is None else fallback_reference_at.isoformat()
        ),
        "fallback_reference_question_count": sum(
            task.fallback_reference_question_count for task in tasks
        ),
        "data_root": str(arguments.data_root),
        "media_manifest_path": (
            None if arguments.media_manifest is None else str(arguments.media_manifest.resolve())
        ),
        "media_roots": media_roots,
        "unit_concurrency": arguments.unit_concurrency,
        "ingest_concurrency": arguments.ingest_concurrency,
        "effective_ingest_concurrency": min(arguments.ingest_concurrency, SDK_OPERATION_LIMIT),
        "request_concurrency": arguments.request_concurrency,
        "stream_answers": arguments.stream_answers,
        "recall_limit": arguments.recall_limit,
        "answer_policy_override": arguments.answer_policy,
        "deliberation": {
            "enabled": arguments.deliberate,
            "operations_applied": operations_applied,
        },
        "arms": _arm_provenance(arguments, memory_config),
        "measurement_protocol": _measurement_protocol(
            arguments,
            samples,
            embedding_warmup_count=embedding_warmup_count,
        ),
        "model": _model_result(
            arguments,
            config,
            memory_config,
            embedding_warmup_count=embedding_warmup_count,
        ),
        "judge": {
            "request_version": JUDGE_REQUEST_VERSION,
            "model": judge_config.model,
            "base_url": judge_config.base_url,
            "sampling": "benchmark_protocol",
            "concurrency": judge_config.concurrency,
            "timeout_seconds": judge_config.timeout_seconds,
        },
        "environment": {
            "sensemem_version": _version("sensemem"),
            "zvec_version": _version("zvec"),
            "runtime_versions": {
                name: _version(name)
                for name in ("torch", "transformers", "opentelemetry-sdk", "openai", "funasr")
            },
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hardware": hardware_metadata(),
            "acceleration_runtime": acceleration_runtime_metadata(),
            "source": source_metadata(),
        },
        "resources": None if resources is None else dict(resources),
        "tasks": task_rows,
    }


def _measurement_protocol(
    arguments: Arguments,
    samples: Sequence[SampleResult],
    *,
    embedding_warmup_count: int | None,
) -> dict[str, object]:
    cached = sum(sample.cached for sample in samples)
    state = (
        "fresh_store"
        if arguments.use_cache is None
        else "response_cache_only"
        if cached == len(samples)
        else "mixed_fresh_store_and_response_cache"
        if cached
        else "fresh_store_with_response_cache_enabled"
    )
    return {
        "state": state,
        "store": (
            "one reused-or-created physical data directory per benchmark unit"
            if arguments.resume
            else "one newly-created physical data directory per benchmark unit"
        ),
        "repeat_index": arguments.repeat_index,
        "repeat_execution": "independent_eval_invocation",
        "measured_response_count": len(samples) - cached,
        "cached_responses_excluded_from_performance_denominators": True,
        "embedding_warmup": {
            "count": embedding_warmup_count,
            "task": EmbedTask.QUERY.value,
            "included_in_product_measurement": False,
        },
        "generation_warmup": {"count": 0},
        "vision_description_cache": "shared_within_run_only; cold_again_for_each_repeat",
        "remote_server_state": "uncontrolled",
        "answer_e2e_includes_request_admission": True,
        "post_answer_search_replay_in_client_resource_window": arguments.search_replay,
        "judge_in_client_resource_window": False,
        "judge_measurement_exclusion": (
            "interleaved sub-windows removed from client and model-server counters"
            if arguments.stream_results and not arguments.predict_only
            else "judging runs after the product measurement window"
        ),
    }


def _arm_provenance(
    arguments: Arguments,
    memory_config: MemorySpec | None = None,
) -> dict[str, object]:
    """Describe every arm precisely enough that a reader can attribute each number to one."""
    answer_candidate_limit = answer_retrieval_candidate_limit(
        arguments.recall_limit,
        memory_config,
    )
    evidence_budget_chars = resolve_evidence_budget_chars(memory_config)
    definitions: dict[str, object] = {
        DEFAULT_ARM: {
            "answers_from": (
                "retrieved memories through Memory.ask, or the generator over Memory.compile's "
                "rendered bundle on the tasks answer_surface maps to compile"
            ),
            "answer_surface": {task: task_answer_surface(task) for task in sorted(arguments.tasks)},
            "compile_surface_prompt": COMPILE_SURFACE_PROMPT_VERSION,
            "retrieval": (
                "Memory.ask in-answer ranked list; no second scoring search. None on a compile "
                "task: Memory.compile exposes no ranked list"
            ),
            "retrieval_candidate_limit": answer_candidate_limit,
            "retrieval_candidate_limit_basis": (
                "full rerank pool because evidence_budget_chars is configured"
                if evidence_budget_chars is not None
                else "min(100, recall_limit * 3)"
            ),
            "official_metrics": True,
        },
        "blind": {
            "answers_from": "the generator's prior, with no evidence",
            "retrieval": None,
            "prompt": BLIND_PROMPT_VERSION,
            "media_in_question": "native query assets when the generator supports them",
            "knowledge_source": "model prior; unsupported user facts must not be invented",
            "official_metrics": False,
        },
        "full-context": {
            "answers_from": "the corpus stuffed into the prompt, oldest first",
            "retrieval": None,
            "prompt": FULL_CONTEXT_PROMPT_VERSION,
            "budget_chars": arguments.full_context_chars,
            "budget_scope": "rendered evidence text including inter-memory separators",
            "media_in_corpus": "dropped: only text atoms are stuffed",
            "official_metrics": False,
        },
        "random": {
            "answers_from": "nothing; retrieval metrics only",
            "retrieval": (
                f"uniform shuffle of the top {RETRIEVAL_CANDIDATE_LIMIT} ranked candidates for "
                "gold-labelled questions; recall_limit otherwise"
            ),
            "retrieval_candidate_limit": {
                "gold_labelled_questions": RETRIEVAL_CANDIDATE_LIMIT,
                "questions_without_gold_labels": arguments.recall_limit,
            },
            "seed": arguments.seed,
            "official_metrics": False,
        },
        "compile": {
            "answers_from": "Memory.compile's rendered bundle",
            "retrieval": "Memory.compile",
            "prompt": FULL_CONTEXT_PROMPT_VERSION,
            "budget_max_items": arguments.compile_max_items,
            "budget_max_chars": arguments.compile_max_chars,
            "budget_scope": "ContextBudget bundle; provider prompt and native media are separate",
            "allow_partial_sources": arguments.compile_allow_partial_sources,
            "official_metrics": False,
        },
    }
    for name in arguments.arms:
        definition = cast(dict[str, object], definitions[name])
        if name == DEFAULT_ARM:
            definition["budget_chars"] = evidence_budget_chars
            definition["budget_scope"] = (
                "ask evidence selection; excludes prompt and media; compile caps are per task"
            )
            definition["budget_by_task"] = {
                task: (
                    {
                        "surface": "ask",
                        "evidence_budget_chars": evidence_budget_chars,
                    }
                    if answers_through_ask(name, task)
                    else {
                        "surface": "compile",
                        "max_chars": arguments.compile_max_chars,
                        "max_items": arguments.compile_max_items,
                    }
                )
                for task in sorted(arguments.tasks)
            }
        definition["answer_policy"] = {
            task: (
                task_answer_policy(task, arguments.answer_policy)
                if requests_answer_policy(name, task)
                else None
            )
            for task in sorted(arguments.tasks)
        }
    return {
        "selected": list(arguments.arms),
        "comparison_scope": "end_to_end_system",
        "memory_only_attribution": False,
        "retrieval_candidate_limit": answer_candidate_limit,
        "retrieval_candidate_limit_arm": DEFAULT_ARM,
        "retrieval_candidate_source": (
            "Memory.ask in-answer ranked list; none on a task answered through Memory.compile"
        ),
        "search_e2e_limit": arguments.recall_limit,
        "answer_delivery": "streamed" if arguments.stream_answers else "buffered",
        "effective_ask_concurrency": min(
            arguments.request_concurrency,
            SDK_STREAM_LIMIT if arguments.stream_answers else SDK_OPERATION_LIMIT,
        ),
        "search_e2e_source": "post-answer public Memory.search replay",
        "evidence_budget_chars": evidence_budget_chars,
        "ingest": arguments.ingest,
        "ingest_concurrency": arguments.ingest_concurrency,
        "definitions": {name: definitions[name] for name in arguments.arms},
    }


def _model_result(
    arguments: Arguments,
    config: ModelConfig,
    memory_config: MemorySpec | None,
    *,
    embedding_warmup_count: int | None = None,
) -> dict[str, object]:
    generation_seed = None if reasoning_model(config.generation_model) else arguments.seed
    if memory_config is not None:
        generation = memory_config.generation
        generation_seed = (
            generation.seed if isinstance(generation, OpenAIGenerationConfig) else None
        )
        generation_temperature = (
            generation.temperature if isinstance(generation, OpenAIGenerationConfig) else None
        )
    else:
        options = dict(
            item.split("=", 1) for item in arguments.gen_kwargs.split(",") if "=" in item
        )
        generation_temperature = resolve_generation_temperature(config.generation_model, options)
    result: dict[str, object] = {
        "adapter": DEFAULT_ARM,
        "embedding_model": None,
        "embedding_revision": None,
        "embedding_dimension": None,
        "embedding_warmup": {
            "count": embedding_warmup_count,
            "task": EmbedTask.QUERY.value,
            "included_in_product_measurement": False,
        },
        "device": arguments.device or "auto",
        "generation_model": config.generation_model,
        "generation_base_url": config.generation_base_url,
        "generation_modalities": sorted(
            modality.value for modality in config.generation_capabilities
        ),
        "generation_seed": generation_seed,
        "generation_temperature": generation_temperature,
        "generation_answer_style": (
            memory_config.generation.answer_style
            if memory_config is not None
            and isinstance(memory_config.generation, OpenAIGenerationConfig)
            else "default"
        ),
        "generation_kwargs": arguments.gen_kwargs,
        "generation_min_video_seconds": config.generation_min_video_seconds,
        "transcription_model": None,
        "timeout_seconds": config.timeout_seconds,
        "memory_config": None,
    }
    if memory_config is None:
        return result
    embedding = memory_config.embedding.model_dump(mode="json")
    speech = None if memory_config.speech is None else memory_config.speech.model_dump(mode="json")
    result.update(
        embedding_model=embedding.get("model"),
        embedding_revision=embedding.get("revision"),
        embedding_dimension=embedding.get("dimension"),
        device=_configured_device_label(arguments.device, memory_config),
        transcription_model=(
            None
            if speech is None
            else DEFAULT_FUNASR_MODEL_ID
            if speech["provider"] == "funasr"
            else speech["model"]
        ),
        memory_config=memory_config_payload(memory_config),
    )
    return result


def _task_modalities(task: LoadedTask) -> list[str]:
    """List the atomic input modalities this task actually routes through the SDK."""
    modalities = (
        {Modality.TEXT.value}
        if any(
            isinstance(atom, str)
            for unit in task.units
            for item in unit.memories
            for atom in item.content
        )
        else set()
    )
    for unit in task.units:
        for item in unit.memories:
            for atom in item.content:
                if isinstance(atom, Path):
                    modality = MODALITY_BY_SUFFIX.get(atom.suffix.casefold())
                    if modality is not None:
                        modalities.add(modality.value)
    return sorted(modalities)


def _configured_device_label(explicit: str | None, config: MemorySpec) -> str:
    if explicit is not None:
        return explicit
    devices = []
    if isinstance(config.embedding, SentenceTransformersEmbeddingConfig):
        devices.append(config.embedding.device or "auto")
    if isinstance(config.speech, FunASRSpeechConfig):
        devices.append(config.speech.device)
    return ",".join(dict.fromkeys(devices)) or "remote"


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "source"


def has_regression(comparisons: Sequence[Mapping[str, object]], *, threshold: float) -> bool:
    for comparison in comparisons:
        interval = comparison.get("confidence_interval_95")
        score_scale = cast(float, comparison["score_scale"])
        if (
            isinstance(interval, list)
            and len(interval) == 2
            and float(interval[1]) < -threshold * score_scale
        ):
            return True
    return False
