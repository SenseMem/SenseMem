"""Input ablations preserve source identities and never modify questions or labels."""

from __future__ import annotations

import asyncio
import runpy
import sqlite3
import struct
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import pytest
from PIL import Image

from mindbridge import EmbedTask, Memory, MindBridgeConfig, Modality, ModelInput
from mindbridge.benchmarks import eval as runner
from mindbridge.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask, MemoryItem
from mindbridge.benchmarks.eval_telemetry import EvaluationTelemetry
from mindbridge.benchmarks.task_catalog import TASKS


class _ExecutionResult(Protocol):
    def execution_result(self) -> dict[str, object]: ...


def test_raw_attachment_preserves_questions_text_and_timestamps(tmp_path: Path) -> None:
    image = tmp_path / "20300101_120000.jpg"
    image.write_bytes(b"source bytes including original metadata")
    question = EvalQuestion("q1", ("Which building?",), ("Reference kept outside product",))
    original = MemoryItem(image.stem, ("Released description plus detailed view",))
    task = LoadedTask(
        TASKS["atm-bench-main-sgm"],
        tmp_path / "qa.json",
        "dataset-hash",
        (EvalUnit("main_sgm", (original,), (question,)),),
        {"memory": "text-memory-hash", "dataset": "dataset-hash"},
    )
    raw = replace(
        task,
        units=(replace(task.units[0], memories=(replace(original, content=(image,)),)),),
        input_sha256={"memory": "raw-memory-hash"},
    )
    attach = cast(
        Callable[[LoadedTask, LoadedTask], LoadedTask],
        runpy.run_path(str(Path(__file__).parents[3] / "benchmarks/atm_structural_ablation.py"))[
            "attach_raw_media"
        ],
    )
    mixed = attach(task, raw)
    assert mixed.units[0].questions is task.units[0].questions
    assert mixed.dataset_sha256 == task.dataset_sha256
    assert mixed.units[0].memories[0] == replace(original, content=(*original.content, image))
    assert mixed.input_sha256["memory"] != task.input_sha256["memory"]
    assert mixed.input_sha256["raw_memory"] == "raw-memory-hash"
    assert image.read_bytes() == b"source bytes including original metadata"
    with pytest.raises(ValueError, match="identities differ"):
        attach(task, replace(raw, units=(replace(raw.units[0], memories=()),)))


def test_store_reuse_verifies_authoritative_content_and_media_without_reembedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Embedder:
        embedding_model = "fixture-embedding"
        embedding_space = "fixture-embedding-v1"
        embedding_dimension = 2
        embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
        calls = 0

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            self.calls += 1
            return tuple((1.0, 0.0) for _ in inputs)

        def close(self) -> None:
            pass

    image = tmp_path / "20310101_080000.jpg"
    Image.new("RGB", (4, 4), "red").save(image)
    item = MemoryItem(image.stem, ("Original description", image))
    store = tmp_path / "store"
    embedder = Embedder()
    with Memory(store, embedder=embedder) as memory:
        memory.add(
            (f"[source_id: {item.source_id}]\nOriginal description", image),
            metadata={"source_id": item.source_id},
        )
    calls_after_ingestion = embedder.calls
    task = LoadedTask(
        TASKS["atm-bench-main-sgm"],
        tmp_path / "qa.json",
        "dataset-hash",
        (EvalUnit("main_sgm", (item,), (EvalQuestion("q1", ("Question",), ("Gold",)),)),),
        {"memory": "memory-hash"},
    )
    functions = runpy.run_path(
        str(Path(__file__).parents[3] / "benchmarks/atm_structural_ablation.py")
    )
    verify = cast(
        Callable[[Path, MindBridgeConfig, LoadedTask], dict[str, object]],
        functions["verify_store"],
    )
    monkeypatch.setitem(
        verify.__globals__,
        "Memory",
        SimpleNamespace(from_config=lambda _config: Memory(store, embedder=embedder)),
    )
    config = MindBridgeConfig.model_validate({"embedding": {"provider": "openai"}})
    assert verify(store, config, task)["asset_count"] == 1
    assert embedder.calls == calls_after_ingestion
    changed = replace(item, content=("Changed description", image))
    with pytest.raises(ValueError, match="text differs"):
        verify(store, config, replace(task, units=(replace(task.units[0], memories=(changed,)),)))
    fingerprint = cast(Callable[[Path], dict[str, object]], functions["embedding_fingerprint"])
    original = fingerprint(store)
    assert original == fingerprint(store)
    with Memory(store, embedder=embedder) as memory:
        with pytest.raises(RuntimeError, match="already in use"):
            fingerprint(store)
        memory.reindex()
    assert original == fingerprint(store)
    with closing(sqlite3.connect(store / "state.sqlite3")) as connection:
        connection.execute("UPDATE embeddings SET vector = ?", (struct.pack("<2f", 0.0, 1.0),))
        connection.commit()
    assert original != fingerprint(store)


def test_execution_audit_retains_only_counts_and_allowlisted_attributes() -> None:
    functions = runpy.run_path(
        str(Path(__file__).parents[3] / "benchmarks/atm_structural_ablation.py")
    )
    collector = cast(Callable[[], EvaluationTelemetry], functions["ExecutionTelemetry"])()
    try:
        with collector.tracer.start_as_current_span("mindbridge.reranking") as span:
            span.set_attribute("mindbridge.reranking.scored_candidates", 12)
            span.set_attribute("mindbridge.grounding.gazetteer.index_sha256", "a" * 64)
            span.set_attribute("mindbridge.grounding.gazetteer.added_chars", 200)
            span.set_attribute("private.query", "DO_NOT_RETAIN_QUERY")
            span.set_attribute("provider.body", "DO_NOT_RETAIN_PROVIDER_BODY")
        result = cast(_ExecutionResult, collector).execution_result()
        assert result["span_counts"] == {"mindbridge.reranking": 1}
        assert result["attribute_distributions"] == {
            "mindbridge.reranking.scored_candidates": {"12": 1},
            "mindbridge.grounding.gazetteer.index_sha256": {"a" * 64: 1},
            "mindbridge.grounding.gazetteer.added_chars": {"200": 1},
        }
        assert "DO_NOT_RETAIN" not in str(result)
    finally:
        collector.close()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fails", [True, False])
def test_search_replay_switch_preserves_default_and_restores_after_failure(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, fails: bool
) -> None:
    calls: list[str] = []

    async def native(*_args: object, **_kwargs: object) -> None:
        calls.append("diagnostic replay")

    monkeypatch.setattr(runner, "_measure_standalone_searches", native)
    functions = runpy.run_path(
        str(Path(__file__).parents[3] / "benchmarks/atm_structural_ablation.py")
    )
    configure = cast(Callable[[bool], AbstractContextManager[None]], functions["_search_replay"])
    try:
        with configure(enabled):
            asyncio.run(runner._measure_standalone_searches())  # type: ignore[call-arg]
            if fails:
                raise RuntimeError("native pipeline failed")
    except RuntimeError as error:
        assert fails and str(error) == "native pipeline failed"
    assert calls == (["diagnostic replay"] if enabled else [])
    assert runner._measure_standalone_searches is native
