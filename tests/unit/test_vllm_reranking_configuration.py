"""Declarative remote reranking uses the public SDK and owns its transport."""

import json
from functools import partial
from pathlib import Path

import httpx2 as httpx
import pytest
from _feature_support import TinyEmbedder
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError as PydanticValidationError

import sensemem.configuration as configuration
from sensemem import Memory, Modality, SenseMemConfig, VLLMReranker
from sensemem._telemetry import (
    GEN_AI_RESPONSE_MODEL,
    MODEL_REQUEST_COUNT,
    MODEL_REQUEST_IDS,
)


def test_configured_server_reranks_search_and_records_provider_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        assert str(request.url) == "https://fixture.test/rerank"
        assert request.headers["authorization"] == "Bearer fixture-key"
        return httpx.Response(
            200,
            headers={"x-request-id": "header-fixture"},
            json={
                "id": "rerank-fixture",
                "model": "served-checkpoint",
                "usage": {"prompt_tokens": 12, "total_tokens": 12},
                "results": sorted(
                    [
                        {"index": index, "relevance_score": 0.9 if "cancelled" in text else 0.1}
                        for index, text in enumerate(payload["documents"])
                    ],
                    key=lambda row: row["relevance_score"],
                    reverse=True,
                ),
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(configuration, "_build_embedding", lambda _: TinyEmbedder())
    monkeypatch.setattr(configuration, "VLLMReranker", partial(VLLMReranker, client))
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    config = SenseMemConfig.model_validate(
        {
            "data_dir": tmp_path,
            "embedding": {"provider": "openai"},
            "reranking": {
                "provider": "vllm",
                "model": "org/model",
                "base_url": "https://fixture.test/",
                "api_key": "fixture-key",
                "modalities": ["text"],
                "batch_size": 2,
                "timeout": 7.0,
            },
            "settings": {
                "retrieval_mode": "lexical",
                "minimum_relevance": 0,
                "ambiguity_margin": 0,
            },
        }
    )
    assert isinstance(config.reranking, configuration.VLLMRerankingConfig)
    assert config.reranking.modalities == frozenset({Modality.TEXT})
    assert "fixture-key" not in repr(config) and "fixture-key" not in config.model_dump_json()
    with Memory.from_config(config, tracer=provider.get_tracer("test")) as memory:
        memory.add("reservation confirmed")
        target = memory.add("reservation cancelled")
        traced = memory.search("reservation", limit=1, explain=True)
        assert traced.trace is not None
        assert traced.hits[0].id == target.id
    assert client.is_closed
    assert len(sent) == 1 and sent[0]["model"] == "org/model"
    assert len(sent[0]["documents"]) == 2  # type: ignore[arg-type]
    spans = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get(GEN_AI_RESPONSE_MODEL) == "served-checkpoint"
    ]
    assert len(spans) == 1
    attributes = spans[0].attributes or {}
    assert attributes[MODEL_REQUEST_COUNT] == 1
    assert attributes["gen_ai.usage.input_tokens"] == 12
    assert set(attributes[MODEL_REQUEST_IDS]) == {"header-fixture", "rerank-fixture"}  # type: ignore[arg-type]
    provider.shutdown()


@pytest.mark.parametrize(
    "controls",
    [
        {"base_url": "fixture.test"},
        {"base_url": "https://fixture.test#fragment"},
        {"timeout": 0},
        {"timeout": True},
        {"timeout": "60"},
        {"batch_size": 0},
        {"media_candidates": 0},
        {"media_weight": 0},
        {"media_weight": 1.1},
        {"media_weight": True},
        {"modalities": []},
        {"modalities": ["audio"]},
        {"modalities": ["omni"]},
        {"revision": "a" * 40},
        {"device": "cuda"},
    ],
)
def test_server_configuration_rejects_invalid_or_server_owned_controls(
    controls: dict[str, object],
) -> None:
    with pytest.raises(PydanticValidationError):
        SenseMemConfig.model_validate(
            {
                "embedding": {"provider": "openai"},
                "reranking": {
                    "provider": "vllm",
                    "model": "org/model",
                    "base_url": "https://fixture.test",
                    **controls,
                },
            }
        )
