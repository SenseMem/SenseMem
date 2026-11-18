"""Share the configured native Omni model with SenseMem's semantic event stream."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import replace

from sensemem import (
    AsyncMemory,
    DefaultQueryPolicy,
    EventSegmentationBackend,
    MemorySpec,
    Modality,
    ModelInput,
    QueryPolicy,
    QueryRepresentationPlan,
    QueryRequest,
    RetrievalMode,
    ValidationError,
    build_memory_composition,
)
from sensemem.prompts import demo as demo_prompts


class LiveReviewQueryPolicy:
    """Keep proactive review instructions and original media out of query embedding."""

    def __init__(self, ordinary: QueryPolicy) -> None:
        self.ordinary = ordinary

    def plan(self, request: QueryRequest) -> QueryRepresentationPlan:
        first_line = request.content.text.partition("\n")[0]
        if first_line.startswith(demo_prompts.PROACTIVE_RECALL_PREFIX):
            try:
                query = json.loads(first_line.removeprefix(demo_prompts.PROACTIVE_RECALL_PREFIX))
            except json.JSONDecodeError:
                return self.ordinary.plan(request)
            if isinstance(query, str) and query.strip() and len(query) <= 2048:
                return QueryRepresentationPlan(
                    query,
                    embedding_inputs=()
                    if request.mode is RetrievalMode.LEXICAL
                    else (ModelInput(text=query),),
                    lexical_queries=() if request.mode is RetrievalMode.DENSE else (query,),
                )
        return self.ordinary.plan(request)

    def close(self) -> None:
        self.ordinary.close()


def open_event_memory(config: MemorySpec) -> tuple[AsyncMemory, EventSegmentationBackend]:
    """Open one SDK composition and pass original media directly to its Omni backend."""
    with ExitStack() as cleanup:
        composition = build_memory_composition(config)
        cleanup.callback(composition.close)
        generation = composition.plugins.answerer
        if not isinstance(generation, EventSegmentationBackend):
            raise ValidationError(
                "the live example requires a generation backend with event segmentation"
            )
        required = frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO})
        if not required <= generation.generation_capabilities:
            raise ValidationError(
                "the live example requires native text, image, audio, and video generation"
            )
        memory = AsyncMemory.from_plugins(
            composition.data_dir,
            plugins=replace(
                composition.plugins,
                query_policy=LiveReviewQueryPolicy(
                    composition.plugins.query_policy or DefaultQueryPolicy()
                ),
            ),
            settings=composition.settings,
        )
        cleanup.pop_all()
        return memory, generation
