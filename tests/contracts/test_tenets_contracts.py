"""Executable combinations, observable perception and readable composition boundaries."""

import ast
from collections.abc import Sequence
from dataclasses import asdict
from inspect import signature
from pathlib import Path
from typing import cast

import pytest

import mindbridge.models
from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    AssetRef,
    Blob,
    EmbedTask,
    EvidenceGeneration,
    FaceAnalysis,
    GenerationRouter,
    Memory,
    MemoryCapabilities,
    MemoryConfig,
    Modality,
    ModelError,
    ModelInput,
    SearchHit,
    ValidationError,
)
from mindbridge.api.mcp import PerceptionResult as WirePerception

TEXT = frozenset({Modality.TEXT})
ALL = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})


class Embedder:
    embedding_capabilities = ALL
    embedding_model = "contract-fixture"
    embedding_space = "contract-fixture:2"
    embedding_dimension = 2

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return ((3.0, 4.0),) * len(inputs)

    def close(self) -> None:
        pass


class Answerer:
    generation_capabilities = TEXT

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        return AnswerResult("fixture", hits=tuple(hits))

    def close(self) -> None:
        pass


@pytest.mark.parametrize("wrapping", ("plain", "outer", "inner"))
def test_declared_native_combinations_match_execution(tmp_path: Path, wrapping: str) -> None:
    class Image(Answerer):
        generation_capabilities = TEXT | {Modality.IMAGE}

    class Video(Answerer):
        generation_capabilities = TEXT | {Modality.VIDEO}

    routes = (Image(), Video())
    router = GenerationRouter(
        tuple(EvidenceGeneration(route) for route in routes) if wrapping == "inner" else routes
    )
    answerer = EvidenceGeneration(router) if wrapping == "outer" else router
    with Memory(tmp_path, embedder=Embedder(), answerer=answerer) as memory:
        capabilities = memory.capabilities
        assert capabilities.generation == TEXT | {Modality.IMAGE, Modality.VIDEO}
        assert capabilities.supports_generation(TEXT | {Modality.IMAGE})
        assert capabilities.supports_generation(TEXT | {Modality.VIDEO})
        assert not capabilities.supports_generation(TEXT | {Modality.IMAGE, Modality.VIDEO})
        assert capabilities.document()["generation_routes"] == [
            ["image", "text"],
            ["text", "video"],
        ]
        with pytest.raises(ModelError) as failure:
            memory.ask(("question", Blob(b"image", "image/png"), Blob(b"clip", "video/mp4")))
        assert failure.value.reason == "unsupported_modality"


def test_perception_distinguishes_missing_backend_from_empty_detection(tmp_path: Path) -> None:
    class EmptyFaces:
        face_capabilities = frozenset({Modality.IMAGE})
        face_model = "contract-faces"
        face_space = "contract-faces:2"
        face_analysis_space = "contract-faces:analysis"

        def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
            return tuple(FaceAnalysis(()) for _ in assets)

        def close(self) -> None:
            pass

    directory = tmp_path / "store"
    with Memory(directory, embedder=Embedder()) as memory:
        record = memory.capture((Blob(b"image", "image/png"), Blob(b"clip", "video/mp4")))
        skipped = memory.perceive(record.id)
        assert not skipped.speech and not skipped.faces
        assert not any(entry.processed for entry in skipped.coverage)
        assert {entry.reason for entry in skipped.coverage if entry.kind == "faces"} == {
            "backend_not_configured"
        }
    with Memory(directory, embedder=Embedder(), face_analyzer=EmptyFaces()) as memory:
        analyzed = memory.perceive(record.id)
        assert not analyzed.speech and not analyzed.faces
        faces = [entry for entry in analyzed.coverage if entry.kind == "faces"]
        assert sum(entry.processed for entry in faces) == 1
        assert {entry.reason for entry in faces} == {None, "unsupported_modality"}
        assert WirePerception.model_validate(analyzed).model_dump(mode="json")["coverage"] == [
            {**asdict(entry)} for entry in analyzed.coverage
        ]
        image = memory.capture(Blob(b"image", "image/png"))
        selected = memory.perceive(image.id, kinds=frozenset({"faces"}))
        assert {entry.reason for entry in selected.coverage if entry.kind == "speech"} == {
            "not_requested"
        }
        assert all(entry.processed for entry in selected.coverage if entry.kind == "faces")


@pytest.mark.parametrize("budget", (1, 4))
def test_evidence_consolidation_has_no_candidate_budget(tmp_path: Path, budget: int) -> None:
    with (
        Memory(tmp_path, embedder=Embedder()) as memory,
        pytest.raises(TypeError, match="max_rounds"),
    ):
        memory.consolidate(evidence_ids=("source",), max_rounds=budget)  # type: ignore[call-arg]


def test_expert_policy_is_one_configuration_value(tmp_path: Path) -> None:
    assert set(signature(Memory).parameters) == {
        "data_dir",
        "embedder",
        "answerer",
        "transcriber",
        "vision_describer",
        "face_analyzer",
        "former",
        "consolidator",
        "reranker",
        "config",
        "tracer",
    }
    config = MemoryConfig(minimum_relevance=0, index_speech=False)
    with Memory(tmp_path, embedder=Embedder(), config=config) as memory:
        record = memory.add("source")
        assert memory.search("source").hits[0].id == record.id


def test_model_clients_do_not_import_kernel_implementation() -> None:
    directory = Path(mindbridge.models.__file__).parent
    for path in directory.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("mindbridge.kernel"), path
            elif isinstance(node, ast.Import):
                assert not any(
                    alias.name.startswith("mindbridge.kernel") for alias in node.names
                ), path


@pytest.mark.parametrize("route", ("text", frozenset({"text"}), frozenset({Modality.IMAGE})))
def test_route_declarations_validate_the_actual_combination_values(route: object) -> None:
    with pytest.raises(ValidationError, match="generation_routes"):
        MemoryCapabilities(
            embedding=TEXT,
            embedding_model="fixture",
            embedding_space="fixture:2",
            embedding_dimension=2,
            generation=TEXT,
            generation_routes=(cast(frozenset[Modality], route),),
        )


def test_empty_router_coverage_does_not_become_a_union_route(tmp_path: Path) -> None:
    router = GenerationRouter((Answerer(),))
    router.generation_routes = ()
    with pytest.raises(ValidationError, match="generation routes"):
        Memory(tmp_path, embedder=Embedder(), answerer=router)
