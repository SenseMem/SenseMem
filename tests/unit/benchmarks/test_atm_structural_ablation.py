"""Input ablations preserve source identities and never modify questions or labels."""

from __future__ import annotations

import asyncio
import hashlib
import json
import runpy
import socket
import sqlite3
import struct
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, closing, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import pytest
import yaml
from PIL import Image

from sensemem import EmbedTask, Memory, Modality, ModelInput, SenseMemConfig
from sensemem.benchmarks import eval as runner
from sensemem.benchmarks.eval_adapters import EvalQuestion, EvalUnit, LoadedTask, MemoryItem
from sensemem.benchmarks.eval_config import (
    _Arguments,
    _judge_config,
    _load_memory_config,
    _model_config,
)
from sensemem.benchmarks.eval_telemetry import EvaluationTelemetry
from sensemem.benchmarks.isolation import BenchmarkRun
from sensemem.benchmarks.task_catalog import TASKS, TaskSpec


class _ExecutionResult(Protocol):
    def execution_result(self) -> dict[str, object]: ...


@pytest.mark.parametrize("concurrency", [16, 32])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("stage", ["full", "reranking-probe", "reader-probe", "native-entry"])
@pytest.mark.parametrize("answer_format", ["text", "structured"])
@pytest.mark.parametrize("gen_kwargs, temperature", [("", 0.0), ("temperature=0.3", 0.3)])
def test_portable_retrieval_recipe_resolves_controls_without_models_or_storage(  # noqa: C901 - one fixture covers preflight and native-entry configuration
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    concurrency: int,
    selected: bool,
    stage: str,
    answer_format: str,
    gen_kwargs: str,
    temperature: float,
) -> None:
    repository = Path(__file__).parents[3]
    recipe_path = Path("benchmarks/atm_ablation_recipe.json")
    source_config = repository / "docs/examples/atm-sgm-max-thinking-vllm.yaml"
    source_bytes = source_config.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    recipe: dict[str, object] = {
        "variant": "native-image-video-pooled",
        "raw_media": True,
        "measure_search_replay": False,
    }
    config_path = Path("fixtures/retrieval.yaml")
    destination = tmp_path / config_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source_bytes)
    recipe["base_config_path"] = str(config_path)
    recipe["artifact_root"] = ".benchmarks/fixture-runs"
    recipe["source_store"] = ".benchmarks/fixture-source"
    recipe["raw_media_root"] = ".benchmarks/fixture-raw"
    recipe["selection_path"] = None
    recipe["preflight"] = stage != "native-entry"
    recipe["stage"] = "full" if stage == "native-entry" else stage
    payload = yaml.safe_load(destination.read_text())
    payload["embedding"].pop("api_key", None)
    payload["generation"].pop("api_key", None)
    payload["generation"]["answer_format"] = answer_format
    payload["benchmark"]["judge"].pop("api_key", None)
    payload["benchmark"]["download"]["benchmarks_root"] = ".benchmarks/atm-inputs"
    payload["settings"].update(
        evidence_budget_chars=64_000,
        recall_rounds=2,
        recall_max_queries=2,
        recall_set_budget_chars=64_000,
        recall_set_max_rows=100,
    )
    payload["benchmark"]["run"]["recall_limit"] = 20
    for name in (
        "request_concurrency",
        "unit_concurrency",
        "ingest_concurrency",
        "judge_concurrency",
    ):
        payload["benchmark"]["run"][name] = concurrency
    destination.write_text(yaml.safe_dump(payload))
    recipe["base_config_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    selection_path = tmp_path / "selection.json"
    selected_ids = tuple(str(index * 7) for index in range(12))
    if selected:
        selection_path.write_text(
            json.dumps(
                {
                    "source_samples_sha256": "a" * 64,
                    "cases": [
                        {"question_id": question_id, "qtype": "fixture"}
                        for question_id in reversed(selected_ids)
                    ],
                }
            )
        )
        recipe["selection_path"] = str(selection_path)
    (tmp_path / recipe_path).parent.mkdir(parents=True)
    (tmp_path / recipe_path).write_text(json.dumps(recipe))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-embedding-key")
    monkeypatch.setenv("SENSEMEM_GENERATION_API_KEY", "fixture-generation-key")
    monkeypatch.setenv("SENSEMEM_JUDGE_API_KEY", "fixture-judge-key")

    def denied(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("portable example preflight must not open models, network or storage")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(sqlite3, "connect", denied)
    monkeypatch.setattr(Memory, "from_config", denied)
    monkeypatch.setattr(runner, "_BackendPool", denied)
    functions = runpy.run_path(str(repository / "benchmarks/atm_structural_ablation.py"))
    main = cast(Callable[[list[str]], int], functions["main"])
    prepare = cast(
        Callable[[Path, int, str], tuple[_Arguments, SenseMemConfig, LoadedTask]],
        functions["_prepare"],
    )
    prepared: list[tuple[_Arguments, SenseMemConfig, LoadedTask]] = []

    def observe_prepare(
        path: Path, expected_count: int = 1013, gen_kwargs: str = ""
    ) -> tuple[_Arguments, SenseMemConfig, LoadedTask]:
        result = prepare(path, expected_count, gen_kwargs)
        prepared.append(result)
        return result

    image = tmp_path / "original.jpg"
    image.write_bytes(b"original fixture media with metadata")
    item = MemoryItem("20300101_120000", ("Already fused source description",))
    memories = (
        (
            item,
            *(MemoryItem(f"source-{index}", ("Archive source",)) for index in range(11033)),
        )
        if selected
        else (item,)
    )
    questions = tuple(
        EvalQuestion(str(index), ("Question",), ("Reference",)) for index in range(1013)
    )
    task = LoadedTask(
        TASKS["atm-bench-main-sgm"],
        tmp_path / "qa.json",
        "fixture-dataset-hash",
        (EvalUnit("main_sgm", memories, questions),),
        {"memory": "fixture-memory-hash", "dataset": "fixture-dataset-hash"},
    )
    raw = replace(
        task,
        spec=TASKS["atm-bench-main"],
        units=(
            replace(
                task.units[0],
                memories=tuple(replace(memory, content=(image,)) for memory in memories),
            ),
        ),
        input_sha256={"memory": "fixture-raw-memory-hash"},
    )
    loads: list[str] = []

    def load_fixture(spec: TaskSpec, *, root: Path, **kwargs: object) -> LoadedTask:
        assert root.resolve() == tmp_path / ".benchmarks/atm-inputs"
        loads.append(spec.name)
        if spec.name == "atm-bench-main":
            assert kwargs["media_root"] == tmp_path / ".benchmarks/fixture-raw"
            return raw
        assert not selected or (kwargs["limit"] is None and kwargs["offset"] == 0)
        return task

    monkeypatch.setattr(runner, "load_task", load_fixture)
    monkeypatch.setitem(main.__globals__, "load_task", load_fixture)
    monkeypatch.setitem(main.__globals__, "_prepare", observe_prepare)
    if stage == "native-entry":

        class NativeConfigurationObserved(Exception):
            pass

        async def ingest_fixture(
            _arguments: _Arguments,
            _config: SenseMemConfig,
            _task: LoadedTask,
            run: BenchmarkRun,
        ) -> Path:
            store = run.unit_dir("main_sgm")
            store.mkdir(parents=True, exist_ok=True)
            return store

        def observe_native_configuration(
            *_args: object, memory_config: SenseMemConfig, **_kwargs: object
        ) -> None:
            assert memory_config.generation is not None
            assert memory_config.generation.temperature == temperature
            assert memory_config.generation.seed == 0
            assert memory_config.generation.max_tokens is None
            raise NativeConfigurationObserved

        monkeypatch.setitem(main.__globals__, "_ingest", ingest_fixture)
        monkeypatch.setitem(main.__globals__, "verify_store", lambda *_args: {})
        monkeypatch.setitem(main.__globals__, "embedding_fingerprint", lambda *_args: {})
        monkeypatch.setattr(
            Memory,
            "from_config",
            lambda _config: nullcontext(SimpleNamespace(reindex=lambda: len(memories))),
        )
        monkeypatch.setattr(runner, "prepare_task_media", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(runner, "_execute", observe_native_configuration)
        recipe["source_store"] = None
        (tmp_path / recipe_path).write_text(json.dumps(recipe))
        with pytest.raises(NativeConfigurationObserved):
            main(["--recipe", str(recipe_path), "--gen-kwargs", gen_kwargs])
        assert vars(runner)["load_task"] is load_fixture
        return
    assert main(["--recipe", str(recipe_path), "--gen-kwargs", gen_kwargs]) == 0
    arguments, config, loaded = prepared[0]
    if selected:
        assert tuple(question.question_id for question in loaded.units[0].questions) == selected_ids
        assert all(
            question is questions[int(question.question_id)]
            for question in loaded.units[0].questions
        )
        assert len(loaded.units[0].memories) == 11034
        assert tuple(memory.source_id for memory in loaded.units[0].memories) == tuple(
            memory.source_id for memory in memories
        )
        assert arguments.limit == 12 and arguments.offset == 0
        assert loaded.dataset_sha256 == task.dataset_sha256
        assert (
            loaded.input_sha256["question_selection"]
            == hashlib.sha256(selection_path.read_bytes()).hexdigest()
        )
    else:
        assert loaded.units[0].questions is questions
        assert arguments.limit == -1
    assert loaded.units[0].memories[0].content == (*item.content, image)
    assert loads == ["atm-bench-main-sgm", "atm-bench-main"]
    assert config.embedding.provider == "openai"
    assert config.embedding.api_key is None
    assert config.generation is not None and config.generation.provider == "openai"
    assert config.generation.api_key is not None
    assert config.generation.api_key.get_secret_value() == "fixture-generation-key"
    assert config.generation.answer_format == answer_format
    assert config.generation.temperature == temperature and config.generation.seed == 0
    assert config.generation.request_max_bytes == 33_554_432
    assert config.vision is None and config.generation.offline_gazetteer is None
    fixture_config, _ = _load_memory_config(tmp_path / config_path)
    assert fixture_config is not None and fixture_config.reranking is not None
    assert config.reranking == fixture_config.reranking
    assert config.settings.evidence_budget_chars == 64_000
    assert (
        config.settings.recall_rounds,
        config.settings.recall_max_queries,
        config.settings.recall_set_budget_chars,
        config.settings.recall_set_max_rows,
    ) == (2, 2, 64_000, 100)
    assert arguments.answer_policy == "strict" and arguments.recall_limit == 20
    assert (
        arguments.request_concurrency,
        arguments.unit_concurrency,
        arguments.ingest_concurrency,
        arguments.judge_concurrency,
    ) == (concurrency, concurrency, concurrency, concurrency)
    directory = tmp_path / ".benchmarks/fixture-runs/native-image-video-pooled"
    manifest = json.loads((directory / "preflight.json").read_text())
    assert manifest["source_store"] == str(tmp_path / ".benchmarks/fixture-source")
    assert manifest["raw_media"] is True
    assert manifest["measure_search_replay"] is False
    assert manifest["concurrency"] == concurrency
    assert manifest["generation_kwargs"] == arguments.gen_kwargs
    assert manifest["question_count"] == (12 if selected else 1013)
    assert manifest["complete_main"] is not selected
    assert manifest["full_qa_sha256"] == task.dataset_sha256
    assert manifest["selection_sha256"] == (
        hashlib.sha256(selection_path.read_bytes()).hexdigest() if selected else None
    )
    assert not (directory / "stores").exists()
    local, overrides = _load_memory_config(directory / "input.yaml")
    assert local is not None
    judge = _judge_config(_model_config("", memory_config=local), arguments, overrides=overrides)
    assert judge.model == "gpt-5-mini" and judge.api_key == "fixture-judge-key"
    assert hashlib.sha256(source_config.read_bytes()).hexdigest() == source_sha256


@pytest.mark.parametrize("expected_count", [1013, 13])
def test_prepare_rejects_question_count_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expected_count: int
) -> None:
    repository = Path(__file__).parents[3]
    functions = runpy.run_path(str(repository / "benchmarks/atm_structural_ablation.py"))
    prepare = cast(
        Callable[[Path, int], tuple[_Arguments, SenseMemConfig, LoadedTask]],
        functions["_prepare"],
    )
    task = LoadedTask(
        TASKS["atm-bench-main-sgm"],
        tmp_path / "qa.json",
        "fixture-dataset-hash",
        (
            EvalUnit(
                "main_sgm",
                (MemoryItem("source", ("Archive text",)),),
                tuple(
                    EvalQuestion(str(index), ("Question",), ("Reference",)) for index in range(12)
                ),
            ),
        ),
    )

    def load_fixture(*_args: object, **_kwargs: object) -> tuple[LoadedTask, ...]:
        return (task,)

    def denied(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("count drift must fail without model or storage initialization")

    monkeypatch.setattr(runner, "_load_tasks", load_fixture)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(sqlite3, "connect", denied)
    monkeypatch.setattr(Memory, "from_config", denied)
    monkeypatch.setattr(runner, "_BackendPool", denied)
    config_path = tmp_path / "config.yaml"
    payload = yaml.safe_load(
        (repository / "docs/examples/atm-sgm-detail-qwen3-vl.yaml").read_text()
    )
    payload["benchmark"]["run"]["run_id"] = "count-guard"
    config_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="requires exactly"):
        prepare(config_path, expected_count)


def test_portable_recipe_rejects_yaml_drift_before_loading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[3]
    config_bytes = b"settings: {}\n"
    changed = tmp_path / "changed.yaml"
    changed.write_bytes(config_bytes + b"\n# changed after pinning\n")
    recipe = {
        "base_config_path": str(changed),
        "base_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "artifact_root": str(tmp_path / "outputs"),
        "variant": "changed-config",
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    functions = runpy.run_path(str(repository / "benchmarks/atm_structural_ablation.py"))
    main = cast(Callable[[list[str]], int], functions["main"])

    def denied(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("changed config must fail before preparing inputs")

    monkeypatch.setitem(main.__globals__, "_prepare", denied)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        main(["--recipe", str(path)])
    assert not (tmp_path / "outputs").exists()


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
        Callable[[Path, SenseMemConfig, LoadedTask], dict[str, object]],
        functions["verify_store"],
    )
    monkeypatch.setitem(
        verify.__globals__,
        "Memory",
        SimpleNamespace(from_config=lambda _config: Memory(store, embedder=embedder)),
    )
    config = SenseMemConfig.model_validate({"embedding": {"provider": "openai"}})
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
        with collector.tracer.start_as_current_span("sensemem.reranking") as span:
            span.set_attribute("sensemem.reranking.scored_candidates", 12)
            span.set_attribute("sensemem.grounding.gazetteer.index_sha256", "a" * 64)
            span.set_attribute("sensemem.grounding.gazetteer.added_chars", 200)
            span.set_attribute("sensemem.grounding.request_bytes", 1024)
            span.set_attribute("sensemem.grounding.request_max_bytes", "DO_NOT_RETAIN_BODY")
            span.set_attribute("private.query", "DO_NOT_RETAIN_QUERY")
            span.set_attribute("provider.body", "DO_NOT_RETAIN_PROVIDER_BODY")
        result = cast(_ExecutionResult, collector).execution_result()
        assert result["span_counts"] == {"sensemem.reranking": 1}
        assert result["attribute_distributions"] == {
            "sensemem.reranking.scored_candidates": {"12": 1},
            "sensemem.grounding.gazetteer.index_sha256": {"a" * 64: 1},
            "sensemem.grounding.gazetteer.added_chars": {"200": 1},
            "sensemem.grounding.request_bytes": {"1024": 1},
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
