"""Public composition controls for evidence investigation and answer validation."""

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

import httpx2 as httpx
import pytest
from _feature_support import TinyEmbedder
from openai import OpenAI
from pydantic import ValidationError as PydanticValidationError

import mindbridge.configuration as configuration
import mindbridge.recipes as recipes
from mindbridge import (
    Memory,
    MindBridgeConfig,
    Modality,
    ModelInput,
    OpenAIModels,
    SearchHit,
    ValidationError,
)
from mindbridge.models.base import GenerationBackend


@pytest.mark.parametrize("value", [0, 33, True, "6"])
def test_investigation_query_limit_is_strict_and_bounded(value: object) -> None:
    with pytest.raises(PydanticValidationError):
        MindBridgeConfig.model_validate(
            {"embedding": {"provider": "openai"}, "settings": {"recall_max_queries": value}}
        )


def test_structured_answer_controls_map_to_the_generation_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    marker = cast(GenerationBackend, object())

    def build(**values: object) -> GenerationBackend:
        captured.update(values)
        return marker

    monkeypatch.setattr(recipes, "_owned_openai_models", build)
    config = configuration.OpenAIGenerationConfig(
        provider="openai", answer_format="structured", source_metadata_key="document_id"
    )
    assert configuration._build_generation(config) is marker
    assert captured["generation_answer_format"] == "structured"
    assert captured["generation_source_metadata_key"] == "document_id"


@pytest.mark.parametrize("stream", [False, True])
def test_declarative_structured_adapter_constructs_and_binds_custom_source_ids(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    sent: list[dict[str, object]] = []
    content = json.dumps({"draft": {"kind": "source_ids", "evidence_labels": ["E1"]}})

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if payload.get("stream"):
            chunks = [
                {"choices": [{"index": 0, "delta": {"content": content}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            return httpx.Response(
                200,
                text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                + "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="fixture", base_url="https://fixture.test/v1", http_client=transport
        )
        monkeypatch.setattr(recipes, "_openai_client", lambda **_values: client)
        adapter = configuration._build_generation(
            configuration.OpenAIGenerationConfig(
                provider="openai",
                model="fixture-reader",
                answer_format="structured",
                source_metadata_key="document_id",
            )
        )
        assert isinstance(adapter, OpenAIModels)
        hit = SearchHit(
            id="internal-record",
            content="A grounded source.",
            metadata={"document_id": "source-alpha"},
            score=1.0,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        try:
            if stream:
                assert (
                    "".join(adapter.stream_answer(ModelInput(text="Which source?"), (hit,)))
                    == "source-alpha"
                )
            else:
                result = adapter.answer(ModelInput(text="Which source?"), (hit,))
                assert result.answer == "source-alpha" and result.hits == (hit,)
        finally:
            adapter.close()
    assert len(sent) == 1
    assert sent[0]["response_format"] is not None


@pytest.mark.parametrize(
    ("provider", "adapter_name", "default_batch_size", "default_max_length"),
    [
        ("sentence-transformers", "SentenceTransformersReranker", 32, 1024),
        ("qwen3-vl", "Qwen3VLReranker", 32, 8192),
    ],
)
@pytest.mark.parametrize("explicit_controls", [False, True])
def test_reranker_is_an_owned_optional_declarative_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    adapter_name: str,
    default_batch_size: int,
    default_max_length: int,
    explicit_controls: bool,
) -> None:
    built: list[dict[str, object]] = []
    closed: list[str] = []

    class Reranker:
        reranking_model = "fixture@" + "a" * 40
        reranking_capabilities = frozenset({Modality.TEXT})

        def __init__(self, **values: object) -> None:
            built.append(values)

        def rerank(self, query: ModelInput, documents: Sequence[ModelInput]) -> tuple[float, ...]:
            raise AssertionError("composition must not score evidence")

        def close(self) -> None:
            closed.append("reranker")

    monkeypatch.setattr(configuration, "_build_embedding", lambda _config: TinyEmbedder())
    monkeypatch.setattr(configuration, adapter_name, Reranker)
    reranking: dict[str, object] = {
        "provider": provider,
        "model": "org/reranker",
        "revision": "a" * 40,
    }
    if explicit_controls:
        reranking.update(device="cuda", batch_size=4, max_length=512)
        if provider == "qwen3-vl":
            reranking["gpu_memory_utilization"] = 0.7
            reranking.update(image_limit=3, video_limit=1, image_max_pixels=131072, video_frames=4)
    config = MindBridgeConfig.model_validate(
        {
            "data_dir": tmp_path,
            "embedding": {"provider": "openai"},
            "reranking": reranking,
            "settings": {"recall_max_queries": 4},
        }
    )
    composition = configuration.resolve_memory_config(config)
    assert isinstance(composition.plugins.reranker, Reranker)
    assert composition.settings.recall_max_queries == 4
    composition.close()
    assert closed == ["reranker"]
    expected: dict[str, object] = {
        "model_id": "org/reranker",
        "revision": "a" * 40,
        "device": "cuda" if explicit_controls else None,
        "batch_size": 4 if explicit_controls else default_batch_size,
        "max_length": 512 if explicit_controls else default_max_length,
    }
    if provider == "qwen3-vl":
        expected["gpu_memory_utilization"] = 0.7 if explicit_controls else 0.9
        expected.update(
            image_limit=3 if explicit_controls else 8,
            video_limit=1 if explicit_controls else 2,
            image_max_pixels=131072 if explicit_controls else 262144,
            video_frames=4 if explicit_controls else 8,
        )
    assert built == [expected]


@pytest.mark.parametrize(
    "controls",
    [
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": 1.1},
        {"gpu_memory_utilization": "0.9"},
        {"device": "cpu"},
        {"device": "cuda:1"},
        {"image_limit": 0},
        {"video_limit": 0},
        {"image_max_pixels": 0},
        {"video_frames": 1},
        {"video_frames": 65},
    ],
)
def test_qwen3_vl_runtime_controls_are_strict(controls: dict[str, object]) -> None:
    with pytest.raises(PydanticValidationError):
        MindBridgeConfig.model_validate(
            {
                "embedding": {"provider": "openai"},
                "reranking": {"provider": "qwen3-vl", "model": "org/reranker", **controls},
            }
        )


@pytest.mark.parametrize("provider", ["sentence-transformers", "qwen3-vl"])
def test_reranking_validation_reports_the_public_field_path(provider: str) -> None:
    with pytest.raises(ValidationError, match=r"config\.reranking\.batch_size:"):
        configuration.resolve_memory_config(
            {
                "embedding": {"provider": "openai"},
                "reranking": {"provider": provider, "model": "org/reranker", "batch_size": 0},
            }
        )


@pytest.mark.parametrize("provider", ["sentence-transformers", "qwen3-vl"])
def test_native_benchmark_overrides_preserve_the_reranking_provider(
    provider: Literal["sentence-transformers", "qwen3-vl"],
) -> None:
    from mindbridge.benchmarks.eval_config import _evaluation_memory_config
    from mindbridge.benchmarks.model_config import ModelConfig

    config = MindBridgeConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai"},
            "reranking": {"provider": provider, "model": "org/reranker", "revision": "a" * 40},
        }
    )
    effective = _evaluation_memory_config(
        config, ModelConfig(), gen_kwargs="max_tokens=2048", seed=0, device=None
    )
    assert effective is not None
    assert effective.reranking == config.reranking
    assert effective.reranking is not None and effective.reranking.provider == provider


def test_invalid_reranker_releases_storage_ownership(tmp_path: Path) -> None:
    data_dir = tmp_path / "unopened"
    with pytest.raises(ValidationError, match="reranker"):
        Memory(data_dir, embedder=TinyEmbedder(), reranker=object())  # type: ignore[arg-type]
    assert not (data_dir / "zvec").exists()
    with Memory(data_dir, embedder=TinyEmbedder()):
        pass
