"""Native media policies survive the public configuration and composition boundary."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

import mindbridge
import mindbridge.configuration as configuration
from mindbridge import EmbedTask, MindBridgeConfig, Modality, ModelInput, resolve_memory_config


class Embedder:
    embedding_model = "contract"
    embedding_space = "contract:2"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return ((1.0, 0.0),) * len(inputs)

    def close(self) -> None:
        pass


def test_public_native_media_configuration_is_forwarded_without_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    class Adapter:
        reranking_model = "multimodal-contract@fixed"

        def __init__(self, **options: object) -> None:
            received.update(options)

        def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
            return (0.0,) * len(documents)

        def close(self) -> None:
            pass

    monkeypatch.setattr(configuration, "_build_embedding", lambda _config: Embedder())
    monkeypatch.setattr(configuration, "Qwen3VLReranker", Adapter)
    policy = {
        "provider": "qwen3-vl",
        "model": "org/model",
        "multimodal": True,
        "media_candidates": 24,
        "media_weight": 0.75,
        "media_max_length": 6144,
        "image_max_pixels": 524288,
        "video_max_pixels": 1048576,
        "video_max_frames": 12,
        "video_fps": 2.0,
    }
    config = MindBridgeConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "reranking": policy,
        }
    )
    assert json.loads(config.model_dump_json())["reranking"].items() >= policy.items()
    composition = resolve_memory_config(config)
    try:
        assert isinstance(composition.plugins.reranker, Adapter)
        for key, value in policy.items():
            if key not in ("provider", "model"):
                assert received[key] == value
        assert "MultimodalRerankingBackend" in mindbridge.__all__
    finally:
        composition.close()
