"""Public composition controls for evidence investigation and answer validation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import pytest
from _feature_support import TinyEmbedder
from pydantic import ValidationError as PydanticValidationError

import mindbridge.configuration as configuration
import mindbridge.recipes as recipes
from mindbridge import Memory, MindBridgeConfig, ValidationError
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


@pytest.mark.parametrize(
    ("provider", "adapter_name", "default_batch_size", "default_max_length"),
    [
        ("sentence-transformers", "SentenceTransformersReranker", 32, 1024),
        ("qwen3-vl", "Qwen3VLReranker", 1, 2048),
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

        def __init__(self, **values: object) -> None:
            built.append(values)

        def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
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
        reranking.update(device="cpu", batch_size=4, max_length=512)
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
    assert built == [
        {
            "model_id": "org/reranker",
            "revision": "a" * 40,
            "device": "cpu" if explicit_controls else None,
            "batch_size": 4 if explicit_controls else default_batch_size,
            "max_length": 512 if explicit_controls else default_max_length,
        }
    ]


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
