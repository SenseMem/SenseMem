"""Borrowed benchmark plugins retain the SDK's declared contracts and shared ownership."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest

from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AssetRef,
    Blob,
    FormationInput,
    FormationProposal,
    GenerationRouter,
    MemoryComposition,
    MemoryOperation,
    MemoryPlugins,
    MemoryRecord,
    MemorySettings,
    MemorySpec,
    MemoryTrigger,
    Modality,
    ModelInput,
    QueryFailure,
    SearchHit,
)
from sensemem.benchmarks import eval_backends
from sensemem.benchmarks.model_config import ModelConfig
from sensemem.benchmarks.quality_support import _FixtureEmbedder
from sensemem.models.base import GenerationRoutingBackend


@pytest.mark.parametrize("slot", ("former", "consolidator", "transcriber", "vision_describer"))
async def test_model_slots_execute_through_the_sdk_without_closing_the_pool_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    slot: Literal["former", "consolidator", "transcriber", "vision_describer"],
) -> None:
    calls: list[str] = []
    closed: list[bool] = []

    class Backend:
        formation_capabilities = frozenset({Modality.TEXT})
        formation_model = formation_space = "formation-fixture-v1"
        consolidation_model = consolidation_recipe = "consolidation-fixture-v1"
        transcription_capabilities = frozenset({Modality.AUDIO})
        transcription_model = transcription_space = "transcription-fixture-v1"
        vision_capabilities = frozenset({Modality.IMAGE})
        vision_model = vision_space = "vision-fixture-v1"

        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            calls.append("former")
            return tuple(() for _input in inputs)

        def consolidate(
            self,
            evidence: Sequence[MemoryRecord],
            *,
            trigger: MemoryTrigger,
            query_failure: QueryFailure | None = None,
        ) -> tuple[MemoryOperation, ...]:
            calls.append("consolidator")
            return ()

        def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
            calls.append("transcriber")
            return tuple("Spoken fixture." for _asset in assets)

        def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
            calls.append("vision_describer")
            return tuple("A blue toolbox." for _input in inputs)

        def close(self) -> None:
            closed.append(True)

    backend = Backend()
    plugins = MemoryPlugins(
        embedder=_FixtureEmbedder(),
        former=backend if slot == "former" else None,
        consolidator=backend if slot == "consolidator" else None,
        transcriber=backend if slot == "transcriber" else None,
        vision_describer=backend if slot == "vision_describer" else None,
    )
    resolved = MemoryComposition(tmp_path, plugins, MemorySettings(minimum_relevance=0))
    monkeypatch.setattr(eval_backends, "build_memory_composition", lambda _config: resolved)
    pool = eval_backends._BackendPool(
        ModelConfig(),
        memory_config=MemorySpec.model_validate({"embedding": {"provider": "openai"}}),
    )
    try:
        async with pool.memory(tmp_path / "unit") as memory:
            content = (
                Blob(b"audio fixture", "audio/wav")
                if slot == "transcriber"
                else Blob(b"image fixture", "image/png")
                if slot == "vision_describer"
                else "The toolbox is blue."
            )
            record = await memory.add(content)
            if slot == "consolidator":
                await memory.consolidate(evidence_ids=(record.id,))
        assert slot in calls
        assert closed == []
    finally:
        pool.close()
    assert closed == [True]


async def test_borrowed_generation_router_preserves_routes_and_nested_model_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answered: list[str] = []
    closed: list[str] = []

    class Answerer:
        def __init__(self, name: str, capabilities: frozenset[Modality]) -> None:
            self.name = name
            self.generation_capabilities = capabilities

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: AnswerPolicy = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            answered.append(self.name)
            return AnswerResult("blue", tuple(hits))

        def close(self) -> None:
            closed.append(self.name)

    text = Answerer("text", frozenset({Modality.TEXT}))
    image = Answerer("image", frozenset({Modality.TEXT, Modality.IMAGE}))
    assert not isinstance(eval_backends._BorrowedGenerationBackend(text), GenerationRoutingBackend)
    router = GenerationRouter((text, image))
    resolved = MemoryComposition(
        tmp_path,
        MemoryPlugins(embedder=_FixtureEmbedder(), answerer=router),
        MemorySettings(minimum_relevance=0),
    )
    monkeypatch.setattr(eval_backends, "build_memory_composition", lambda _config: resolved)
    pool = eval_backends._BackendPool(
        ModelConfig(),
        memory_config=MemorySpec.model_validate({"embedding": {"provider": "openai"}}),
    )
    assert isinstance(pool._plugins.answerer, GenerationRoutingBackend)
    try:
        async with pool.memory(tmp_path / "unit") as memory:
            assert memory.capabilities.generation_routes == router.generation_routes
            await memory.add("The toolbox is blue.")
            assert (await memory.ask("What color is the toolbox?")).answer == "blue"
        assert answered == ["text"]
        assert closed == []
    finally:
        pool.close()
    assert sorted(closed) == ["image", "text"]
