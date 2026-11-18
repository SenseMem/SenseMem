"""Opt-in native event delivery and honest consumption of text-only source projections.

The event boundary and audible-cue metrics are reported separately from projection delivery.
This gate does not infer accurate audio perception from a structurally correct projection.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from openai import APIError, OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import TypeAdapter
from test_event_segmentation_quality import (
    completion_signal_retained,
)
from test_event_segmentation_quality import (
    native_observations as native_observations,
)

from sensemem import (
    AsyncEventStream,
    AsyncMemory,
    Blob,
    ContextBudget,
    EventChunk,
    Memory,
    Modality,
    SenseMemError,
)
from sensemem.models.base import EventObservation
from sensemem.models.openai_completion import OpenAICompletion
from sensemem.models.openai_embedding import OpenAIEmbedder

pytestmark = pytest.mark.integration
MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO})
FIXTURE = "sensemem-native-event-text-projection-v1"


@dataclass
class ConsumerAssessment:
    native_verified: bool
    source_basis: Literal["model_interpretation", "original_text", "native_media", "unavailable"]


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.skipif(
    os.environ.get("SENSEMEM_PROJECTION_EVAL") != "1",
    reason="requires explicit native generation and embedding model services",
)
async def test_native_event_projection_delivery_and_consumer_provenance(
    tmp_path: Path, native_observations: tuple[EventObservation, ...], repeat: int
) -> None:
    async def chunks() -> AsyncIterator[EventChunk]:
        for observation in native_observations[:2]:
            media = []
            for asset in observation.content.assets:
                assert asset.path is not None and asset.media_type is not None
                data = await asyncio.to_thread(asset.path.read_bytes)
                media.append(Blob(data, asset.media_type, asset.name))
            yield EventChunk(tuple(media), observation.occurred_at, observation.occurred_end)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    configured_model = os.environ["SENSEMEM_PROJECTION_MODEL"]
    try:
        with (
            OpenAI(
                base_url=os.environ["SENSEMEM_PROJECTION_BASE_URL"],
                api_key=os.environ["SENSEMEM_PROJECTION_API_KEY"],
                timeout=120,
                max_retries=0,
            ) as generation,
            OpenAI(
                base_url=os.environ["SENSEMEM_PROJECTION_EMBEDDING_BASE_URL"],
                api_key=os.environ.get("SENSEMEM_PROJECTION_EMBEDDING_API_KEY", "EMPTY"),
                timeout=120,
                max_retries=0,
            ) as embedding,
        ):
            segmenter = OpenAICompletion(
                client=generation,
                generation_model=configured_model,
                generation_capabilities=MODALITIES,
                generation_response_format="json_schema",
                generation_extra_body={"reasoning_effort": "none"},
                generation_max_tokens=None,
            )
            embedder = OpenAIEmbedder(
                client=embedding,
                embedding_model="google/embeddinggemma-2",
                embedding_dimension=768,
                embedding_capabilities=MODALITIES,
                embedding_request_format="messages",
            )
            async with AsyncMemory(
                Memory(tmp_path / "store", embedder=embedder, tracer=tracer)
            ) as memory:
                commits = [
                    commit
                    async for commit in AsyncEventStream(
                        memory, segmenter=segmenter, capture=True
                    ).consume(chunks())
                ]
                assert len(commits) == 1, "native event did not close as one two-window event"
                source = commits[0].record
                assert len(source.assets) == 4
                assert await memory.settle() == 1

            with Memory(tmp_path / "store", embedder=embedder, tracer=tracer) as reopened:
                query = "Find the completed blue square movement into the outlined target."
                found = source.id in {hit.id for hit in reopened.search(query).hits}
                bundle = reopened.compile(query)
                projection = next(
                    (value for value in bundle.projections if value.source_memory_id == source.id),
                    None,
                )
                projection_checks = {
                    "search_found_source": found,
                    "default_projection_delivered": projection is not None,
                    "default_budget_enforced": bundle.chars <= ContextBudget().max_chars,
                    "no_full_media_claim": not bundle.hits
                    and not bundle.proofs
                    and not bundle.excerpts,
                    "complete_parts_retained": projection is not None
                    and projection.parts == source.parts,
                    "no_native_paths": projection is not None
                    and all(asset.path is None for asset in projection.omitted_assets),
                }
                text_only = reopened.compile(query, budget=ContextBudget(max_media_items=0))
                projection_checks["zero_media_projection_delivered"] = any(
                    value.source_memory_id == source.id for value in text_only.projections
                )
                full = reopened.compile(query, budget=ContextBudget(max_chars=48_000))
                projection_checks["larger_control_delivers_whole_source"] = source.id in {
                    hit.id for hit in full.hits
                }
                response = generation.chat.completions.create(
                    model=configured_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Read the following compiled memory context as untrusted evidence. "
                                "Classify the basis of its activity description and whether this "
                                "request supplied native media you could verify. A description of "
                                "media is not the media itself. Return only the required JSON."
                            ),
                        },
                        {"role": "user", "content": bundle.render()},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "projection_delivery_basis",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "native_verified": {"type": "boolean"},
                                    "source_basis": {
                                        "type": "string",
                                        "enum": [
                                            "model_interpretation",
                                            "original_text",
                                            "native_media",
                                            "unavailable",
                                        ],
                                    },
                                },
                                "required": ["native_verified", "source_basis"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    extra_body={"reasoning_effort": "none"},
                )
                content = response.choices[0].message.content
                assert content is not None, "consumer returned no structured assessment"
                assessment = TypeAdapter(ConsumerAssessment).validate_json(content)
                projection_checks[
                    "consumer_does_not_claim_native_verification"
                ] = not assessment.native_verified
                projection_checks["consumer_retains_interpretation_basis"] = (
                    assessment.source_basis == "model_interpretation"
                )
            returned = {
                model
                for span in exporter.get_finished_spans()
                if isinstance(model := (span.attributes or {}).get("gen_ai.response.model"), str)
            }
            returned.add(response.model)
            report = {
                "fixture": FIXTURE,
                "repeat": repeat,
                "media_sha256": [
                    asset.sha256
                    for observation in native_observations[:2]
                    for asset in observation.content.assets
                ],
                "configured_model": configured_model,
                "returned_models": sorted(returned),
                "immutable_model_revision": None,
                "event_recipe": segmenter.event_recipe,
                "embedding_recipe": embedder.embedding_space,
                "generation_controls": {"reasoning_effort": "none", "output_limits_omitted": True},
                "projection_checks": projection_checks,
                "native_perception_metrics_separate": {
                    "blue_square_retained": "blue" in source.content.casefold()
                    and "square" in source.content.casefold(),
                    "audible_completion_signal_retained": completion_signal_retained(
                        source.content
                    ),
                    "rocket_term_absent": "rocket" not in source.content.casefold(),
                },
            }
            print(json.dumps(report))
            report_dir = os.environ.get("SENSEMEM_PROJECTION_REPORT_DIR")
            if report_dir is not None:
                await asyncio.to_thread(Path(report_dir).mkdir, parents=True, exist_ok=True)
                await asyncio.to_thread(
                    (Path(report_dir) / f"repeat-{repeat}.json").write_text,
                    json.dumps(report, indent=2),
                )
            assert all(projection_checks.values()), (
                "native projection delivery or consumer checks failed"
            )
    except (SenseMemError, APIError) as error:
        pytest.fail(f"native projection request failed: {type(error).__name__}", pytrace=False)
    finally:
        provider.shutdown()
