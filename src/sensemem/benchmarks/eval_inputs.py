"""Causal corpus boundaries and identities for stored and cached evaluation inputs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path

from sensemem import (
    MemorySpec,
)
from sensemem.benchmarks.artifact_io import encode_json_bytes
from sensemem.benchmarks.baseline_prompts import (
    BLIND_PROMPT_VERSION,
    COMPILE_SURFACE_PROMPT_VERSION,
    FULL_CONTEXT_PROMPT_VERSION,
)
from sensemem.benchmarks.eval_adapters import (
    EvalUnit,
    LoadedTask,
    MemoryItem,
)
from sensemem.benchmarks.eval_arms import (
    PRODUCT_ARM,
    Arm,
)
from sensemem.benchmarks.eval_cache import ResponseCache
from sensemem.benchmarks.eval_config import (
    Arguments,
    memory_config_payload,
)
from sensemem.benchmarks.eval_results import (
    DEFAULT_ARM,
    EVAL_SCHEMA_VERSION,
)
from sensemem.benchmarks.eval_runtime import EVAL_RUNNER_VERSION
from sensemem.benchmarks.model_config import (
    ModelConfig,
)
from sensemem.benchmarks.prompts import task_answer_policy, task_answer_surface


def has_question_cutoffs(tasks: Sequence[LoadedTask]) -> bool:
    return any(
        question.cutoff_seconds is not None
        for task in tasks
        for unit in task.units
        for question in unit.questions
    )


def full_context(items: Sequence[MemoryItem], budget_chars: int) -> str:
    """Stuff the corpus a retrieval arm would have searched, oldest first, under one budget.

    Media atoms have no text to stuff, so a media corpus reduces this arm to its prompt, and a
    corpus whose items all exceed the budget reduces it to the same thing. Both are lower bounds,
    recorded as such in _arm_provenance; neither is distinguishable from blind in the score.
    """
    parts: list[str] = []
    used = 0
    for item in items:
        text = "\n".join(str(atom) for atom in item.content if not isinstance(atom, Path))
        if not text:
            continue
        size = len(text) + (2 if parts else 0)
        if used + size > budget_chars:
            # Skipped, not break: breaking on the first item that does not fit discarded every
            # later memory, and when that item was the first one the arm answered from an empty
            # context under the full-context prompt -- a blind arm wearing the wrong label.
            continue
        used += size
        parts.append(text)
    return "\n\n".join(parts)


def normalize_content(parts: tuple[str | Path, ...]) -> str | Path | tuple[str | Path, ...]:
    return parts[0] if len(parts) == 1 else parts


def memory_content(item: MemoryItem) -> str | tuple[str | Path, ...]:
    first, *rest = item.content
    if not isinstance(first, str):
        # A media item's source id is already metadata, which is where _evidence reads it. As
        # content it was a third retrieval key and made the aggregate key differ from the clip's
        # own key even for a clip with no speech, so every clip went to the embedder twice; a
        # clip with a transcript still does, because its aggregate key carries the transcript.
        return item.content
    labelled = f"[source_id: {item.source_id}]\n{first}"
    return labelled if not rest else (labelled, *rest)


def memory_metadata(item: MemoryItem) -> dict[str, object]:
    values: dict[str, object] = {
        "source_id": item.source_id,
        "start_seconds": item.start_seconds,
    }
    if item.end_seconds is not None:
        values["end_seconds"] = item.end_seconds
    return values


def memory_end(item: MemoryItem) -> float:
    return math.inf if item.end_seconds is None else item.end_seconds


def planned_ingest_count(unit: EvalUnit) -> int:
    """Count the memories a unit will write, so an ingest bar has a total before it starts.

    The causal cursor stops at the latest cutoff any question asks for, so a unit whose questions
    all carry one never writes its tail. Counting from every question rather than from the pending
    set holds the total still when a resumed run answers only some of them.
    """
    boundary = -math.inf
    for question in unit.questions:
        cutoff = math.inf if question.cutoff_seconds is None else question.cutoff_seconds
        boundary = max(boundary, cutoff)
    return sum(1 for item in unit.memories if memory_end(item) <= boundary)


def prefix_end(memories: Sequence[MemoryItem], cutoff: float | None, start: int = 0) -> int:
    """Return how many ordered memories a question at cutoff is allowed to have seen."""
    boundary = math.inf if cutoff is None else cutoff
    end = start
    while end < len(memories) and memory_end(memories[end]) <= boundary:
        end += 1
    return end


def ingest_digest(
    task: LoadedTask,
    arguments: Arguments,
    memory_config: MemorySpec | None,
) -> str:
    """Identify everything that decides what one unit's store ends up holding.

    A resumed run trusts an existing store only when this matches, because a store written by
    another embedder, ingest mode, or dataset revision is not the store this run would write.
    The generator is deliberately absent: it answers questions and never writes.
    """
    payload: dict[str, object] = {
        "runner": EVAL_RUNNER_VERSION,
        "task": cache_task(task),
        "device": arguments.device or "auto",
        "ingest": arguments.ingest,
        "ingest_concurrency": arguments.ingest_concurrency,
        # --deliberate applies consolidation operations to the store between chunks, so a store
        # built with it holds different memories than one built without it.
        "deliberate": arguments.deliberate,
        # Every unoverridden dataset and media path is resolved under this root, and two roots
        # can hold differently prepared copies of the same pinned corpus.
        "benchmarks_root": str(arguments.benchmarks_root),
        "media_manifest": (
            None if arguments.media_manifest is None else str(arguments.media_manifest)
        ),
        "media_root": str(arguments.media_overrides.get(task.spec.name, "")),
    }
    if memory_config is not None:
        payload["memory_config"] = memory_config_payload(
            memory_config, exclude=_ANSWER_ONLY_SECTIONS
        )
    return hashlib.sha256(encode_json_bytes(payload)).hexdigest()


_ANSWER_ONLY_SECTIONS = frozenset({"generation"})


def cache_namespace(
    arguments: Arguments,
    config: ModelConfig,
    batch_sizes: Mapping[str, int],
    *,
    memory_config: MemorySpec | None = None,
) -> str:
    payload = {
        "runner": EVAL_RUNNER_VERSION,
        "schema": EVAL_SCHEMA_VERSION,
        "model": config.generation_model,
        "base_url": config.generation_base_url,
        "device": arguments.device or "auto",
        "seed": arguments.seed,
        "gen_kwargs": arguments.gen_kwargs,
        "generation_min_video_seconds": config.generation_min_video_seconds,
        "recall_limit": arguments.recall_limit,
        # The override and the resolved per-task policy, because either one changes what the
        # request asked for. Without them a best_effort arm replayed a cached strict refusal
        # and labelled it best_effort, and widening BEST_EFFORT_TASKS replayed a task's older
        # strict answers under its new default.
        "answer_policy_override": arguments.answer_policy,
        "answer_policy": {
            task: task_answer_policy(task, arguments.answer_policy)
            for task in sorted(arguments.tasks)
        },
        # The surface for the same reason: widening COMPILE_SURFACE_TASKS must not replay a
        # task's cached ask answers as answers from the compiled bundle.
        "answer_surface": {task: task_answer_surface(task) for task in sorted(arguments.tasks)},
        "compile_surface_prompt": COMPILE_SURFACE_PROMPT_VERSION,
        "blind_prompt": BLIND_PROMPT_VERSION,
        "full_context_prompt": FULL_CONTEXT_PROMPT_VERSION,
        "full_context_chars": arguments.full_context_chars,
        "blind": arguments.blind,
        "batch_sizes": dict(sorted(batch_sizes.items())),
        "ingest": arguments.ingest,
        "ingest_concurrency": arguments.ingest_concurrency,
        "stream_answers": arguments.stream_answers,
        "deliberate": arguments.deliberate,
        "compile_max_items": arguments.compile_max_items,
        "compile_max_chars": arguments.compile_max_chars,
        "compile_allow_partial_sources": arguments.compile_allow_partial_sources,
    }
    fallback_reference_at = arguments.fallback_reference_at
    if fallback_reference_at is not None:
        payload["fallback_reference_at"] = fallback_reference_at.isoformat()
    if memory_config is not None:
        payload["memory_config"] = memory_config_payload(memory_config)
    return hashlib.sha256(encode_json_bytes(payload)).hexdigest()


def cache_task(task: LoadedTask, arm: Arm = PRODUCT_ARM) -> str:
    identity = f"{task.spec.name}:{task.spec.adapter_version}:{task.evaluation_sha256}"
    return identity if arm.name == DEFAULT_ARM else f"{arm.name}:{identity}"


def are_all_cached(
    cache: ResponseCache,
    tasks: Sequence[LoadedTask],
    arms: Sequence[Arm] = (PRODUCT_ARM,),
) -> bool:
    return all(
        cache.get(cache_task(task, arm), unit.unit_id, question.question_id) is not None
        for task in tasks
        for arm in arms
        for unit in task.units
        for question in unit.questions
    )
