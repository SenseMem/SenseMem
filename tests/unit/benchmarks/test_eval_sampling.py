"""The product and baseline arms send the same resolved sampling controls."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import openai
import pytest

import sensemem.configuration as configuration_module
from sensemem import EmbedTask, Modality, ModelInput, SearchHit, SenseMemConfig
from sensemem.benchmarks import eval as eval_module
from sensemem.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask, MemoryItem
from sensemem.benchmarks.eval_arms import _BaselineGenerator
from sensemem.benchmarks.eval_config import (
    _Arguments,
    _evaluation_config,
    _evaluation_memory_config,
    _generation_kwargs,
    _generation_temperature,
)
from sensemem.benchmarks.model_config import ModelConfig
from sensemem.benchmarks.task_catalog import TASKS


def test_configuration_identity_is_stable_across_process_hash_seeds() -> None:
    script = """
import json
from sensemem import SenseMemConfig
from sensemem.benchmarks.eval_config import _memory_config_payload
config = SenseMemConfig.model_validate({
    "embedding": {"provider": "openai", "modalities": ["text", "image", "video"]},
    "generation": {"provider": "openai", "modalities": ["text", "image"], "routes": [
        {"provider": "openai", "modalities": ["text", "image", "audio", "video"]}
    ]},
    "vision": {"provider": "openai", "modalities": ["image", "video"]},
})
print(json.dumps(_memory_config_payload(config), sort_keys=True))
"""
    results = [
        subprocess.check_output(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for seed in ("1", "2", "3", "4")
    ]
    assert len(set(results)) == 1


@pytest.mark.parametrize("explicit_modalities", [True, False])
def test_media_inputs_preserve_declared_generation_routes(explicit_modalities: bool) -> None:
    generation: dict[str, object] = {
        "provider": "openai",
        "model": "image-reader",
        "routes": [
            {"provider": "openai", "model": "audio-reader", "modalities": ["text", "audio"]}
        ],
    }
    if explicit_modalities:
        generation["modalities"] = ["text", "image"]
    declared = SenseMemConfig.model_validate(
        {
            "embedding": {"provider": "openai"},
            "generation": generation,
        }
    )
    assert declared.generation is not None
    model = ModelConfig(
        generation_model="image-reader", generation_capabilities=declared.generation.modalities
    )
    task = LoadedTask(
        TASKS["mosaic"],
        Path("annotations"),
        "digest",
        (
            EvalUnit(
                "persona",
                (MemoryItem("audio", (Path("evidence.mp3"),)),),
                (EvalQuestion("q", ("Question?",), ("Answer",)),),
            ),
        ),
    )
    effective = _evaluation_config(model, (task,), memory_config=declared)
    assert effective.generation_capabilities == declared.generation.modalities
    resolved = _evaluation_memory_config(declared, effective, gen_kwargs="", seed=0, device=None)
    assert resolved is not None and resolved.generation is not None
    assert resolved.generation.modalities == declared.generation.modalities
    assert resolved.generation.routes[0].modalities == frozenset({Modality.TEXT, Modality.AUDIO})


def test_media_inputs_expand_undeclared_generation_capabilities() -> None:
    task = LoadedTask(
        TASKS["mosaic"],
        Path("annotations"),
        "digest",
        (
            EvalUnit(
                "persona",
                (MemoryItem("audio", (Path("evidence.mp3"),)),),
                (EvalQuestion("q", ("Question?",), ("Answer",)),),
            ),
        ),
    )
    effective = _evaluation_config(ModelConfig(), (task,))
    assert effective.generation_capabilities == frozenset({Modality.TEXT, Modality.AUDIO})


@pytest.mark.parametrize("configured_style", ["default", "concise"], ids=["configured", "concise"])
@pytest.mark.parametrize("temperature", [0.0, 0.7], ids=["default", "sampled"])
def test_every_generation_arm_emits_the_resolved_temperature(  # noqa: C901 - sync and async clients
    monkeypatch: pytest.MonkeyPatch, configured_style: str | None, temperature: float
) -> None:
    requests: list[dict[str, object]] = []

    def complete(request: dict[str, object]) -> object:
        requests.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(content="blue", refusal=None),
                )
            ],
            usage=None,
        )

    class Completions:
        def create(self, **request: object) -> object:
            return complete(request)

    class AsyncCompletions:
        async def create(self, **request: object) -> object:
            return complete(request)

    class Client:
        def __init__(self, **_options: object) -> None:
            self.chat = SimpleNamespace(completions=Completions())

        def close(self) -> None: ...

    class AsyncClient:
        def __init__(self, **_options: object) -> None:
            self.chat = SimpleNamespace(completions=AsyncCompletions())

        async def close(self) -> None: ...

    class Embedder:
        embedding_model = "sampling-fixture-v1"
        embedding_space = "sampling-fixture-v1:2"
        embedding_dimension = 2
        embedding_capabilities = frozenset({Modality.TEXT})

        def __init__(self, **_options: object) -> None: ...

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask
        ) -> tuple[tuple[float, ...], ...]:
            del task
            return tuple((1.0, 0.0) for _value in inputs)

        def close(self) -> None: ...

    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setattr(openai, "AsyncOpenAI", AsyncClient)
    monkeypatch.setattr(configuration_module, "SentenceTransformersEmbedder", Embedder)
    options = _generation_kwargs(f"temperature={temperature}", 7)
    arguments = cast(_Arguments, SimpleNamespace(gen_kwargs=options, seed=7, device=None))
    model = ModelConfig(generation_model="qwen3.8-27b")
    declared = (
        SenseMemConfig.model_validate(
            {
                "embedding": {"provider": "sentence-transformers", "model": "fixture"},
                "generation": {
                    "provider": "openai",
                    "model": model.generation_model,
                    "answer_style": configured_style,
                },
            }
        )
        if configured_style is not None
        else None
    )
    effective = _evaluation_memory_config(
        declared,
        model,
        gen_kwargs=arguments.gen_kwargs,
        seed=arguments.seed,
        device=arguments.device,
    )
    pool = eval_module._BackendPool(model, memory_config=effective)
    baseline = _BaselineGenerator(model, seed=7, gen_kwargs=options, generation=effective)
    try:
        assert pool._answerer is not None
        hit = SearchHit(
            id="toolbox",
            content="The toolbox is blue",
            score=1.0,
            created_at=datetime(2026, 10, 28, tzinfo=timezone.utc),
        )
        assert pool._answerer.answer(ModelInput(text="What colour?"), (hit,)).answer == "blue"
        assert asyncio.run(baseline.answer("What colour?", hit.content)) == "blue"
        recorded = eval_module._model_result(arguments, model, effective)
    finally:
        pool.close()
        asyncio.run(baseline.close())

    assert len(requests) == 2
    assert all(request["temperature"] == temperature for request in requests)
    assert all(request["seed"] == 7 for request in requests)
    assert recorded["generation_temperature"] == temperature
    assert recorded["generation_kwargs"] == options
    assert recorded["generation_answer_style"] == (configured_style or "default")
    for request in requests:
        messages = cast(list[dict[str, str]], request["messages"])
        assert ("Give only the answer to the question" in messages[0]["content"]) == (
            configured_style == "concise"
        )


def test_explicit_temperature_is_not_silently_dropped_for_an_unsupported_model() -> None:
    with pytest.raises(ValueError, match="does not support"):
        _generation_temperature("gpt-5-mini", {"temperature": "0.7"})
    assert (
        _generation_temperature("gpt-5.4", {"temperature": "0.7"}, {"reasoning_effort": "none"})
        == 0.7
    )
