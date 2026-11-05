"""Run an isolated ATM input ablation through the native public-SDK benchmark pipeline."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import shutil
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Literal, cast

import yaml
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel, ConfigDict, Field

from sensemem import Memory, SenseMemConfig
from sensemem.benchmarks import eval as runner
from sensemem.benchmarks.eval_adapters import LoadedTask, _memory_digest, load_task
from sensemem.benchmarks.eval_config import (
    _Arguments,
    _arguments,
    _batch_size,
    _build_parser,
    _evaluation_config,
    _evaluation_memory_config,
    _load_memory_config,
    _model_config,
)
from sensemem.benchmarks.eval_results import FailureDetail
from sensemem.benchmarks.eval_telemetry import EvaluationTelemetry
from sensemem.benchmarks.isolation import BenchmarkRun
from sensemem.benchmarks.model_config import DownloadSettings
from sensemem.benchmarks.task_catalog import TASKS
from sensemem.infrastructure.local._lock import DataDirectoryLock


class ExecutionTelemetry(EvaluationTelemetry):
    """Retain counts for fixed numeric/SHA attributes, never span bodies or arbitrary fields."""

    def __init__(self) -> None:
        super().__init__()
        self._execution_lock = Lock()
        self._span_counts: Counter[str] = Counter()
        self._attribute_counts: dict[str, Counter[str]] = {}

    def on_end(self, span: ReadableSpan) -> None:
        super().on_end(span)
        attributes = span.attributes or {}
        with self._execution_lock:
            self._span_counts[span.name] += 1
            for name in (
                "sensemem.grounding.gazetteer.index_sha256",
                "sensemem.grounding.gazetteer.added_chars",
                "sensemem.grounding.gazetteer.available_candidates",
                "sensemem.grounding.gazetteer.omitted_candidates",
                "sensemem.reranking.scored_candidates",
                "sensemem.grounding.request_bytes",
                "sensemem.grounding.request_max_bytes",
                "sensemem.grounding.budget_omitted_assets",
                "sensemem.grounding.budget_transcoded_images",
                "sensemem.grounding.budget_sampled_videos",
            ):
                value = attributes.get(name)
                if name.endswith(".index_sha256"):
                    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                        continue
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                ):
                    continue
                self._attribute_counts.setdefault(name, Counter())[str(value)] += 1

    def execution_result(self) -> dict[str, object]:
        with self._execution_lock:
            return {
                "span_counts": dict(sorted(self._span_counts.items())),
                "attribute_distributions": {
                    name: dict(sorted(counts.items()))
                    for name, counts in sorted(self._attribute_counts.items())
                },
                "contains_content_or_queries": False,
            }


class Recipe(BaseModel):
    """Local paths and one declared intervention; model policy belongs in the base YAML."""

    model_config = ConfigDict(extra="forbid", strict=True)

    base_config_path: Path
    base_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_root: Path
    variant: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")
    raw_media: bool = False
    raw_media_root: Path | None = None
    source_store: Path | None = None
    selection_path: Path | None = None
    stage: Literal["full", "ingest", "reranking-probe", "reader-probe"] = "full"
    preflight: bool = False
    measure_search_replay: bool = True


class _SelectedQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    question_id: str = Field(min_length=1)


class _QuestionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[_SelectedQuestion, ...] = Field(min_length=1, max_length=1012)


@contextmanager
def _search_replay(enabled: bool) -> Iterator[None]:
    """Optionally omit only the native post-answer performance replay."""
    original = runner._measure_standalone_searches

    async def skip(*_args: object, **_kwargs: object) -> None:
        return None

    if not enabled:
        vars(runner)["_measure_standalone_searches"] = skip
    try:
        yield
    finally:
        vars(runner)["_measure_standalone_searches"] = original


def attach_raw_media(task: LoadedTask, raw: LoadedTask) -> LoadedTask:
    """Join native media by source identity, retaining every question and original text."""
    if len(task.units) != 1 or len(raw.units) != 1:
        raise ValueError("ATM ablations require one corpus unit")
    sources = {
        item.source_id: tuple(part for part in item.content if isinstance(part, Path))
        for item in raw.units[0].memories
    }
    originals = task.units[0].memories
    if set(sources) != {item.source_id for item in originals}:
        raise ValueError("SGM and raw source identities differ")
    memories = tuple(
        replace(item, content=(*item.content, *sources[item.source_id])) for item in originals
    )
    units = (replace(task.units[0], memories=memories),)
    return replace(
        task,
        units=units,
        input_sha256={
            **task.input_sha256,
            "memory": _memory_digest(units),
            "raw_memory": raw.input_sha256["memory"],
        },
    )


def verify_store(  # noqa: C901 - one contiguous preservation check before copying a whole corpus
    store: Path, config: SenseMemConfig, task: LoadedTask
) -> dict[str, object]:
    """Verify source text, timestamps and asset bytes through the public SDK before reuse."""
    if any(getattr(config, slot) is not None for slot in ("formation", "speech", "face", "vision")):
        raise ValueError(
            "reusable explicit-description corpora require formation/speech/face/vision null"
        )
    expected = {item.source_id: item for item in task.units[0].memories}
    seen: set[str] = set()
    asset_count = 0
    # Answer-only plugins are unnecessary for the authoritative read-back.
    updates: dict[str, object] = {"data_dir": store, "generation": None}
    if "reranking" in type(config).model_fields:
        updates["reranking"] = None
    with Memory.from_config(config.model_copy(update=updates)) as memory:
        cursor = None
        while True:
            page = memory.list(limit=100, cursor=cursor)
            for record in page.items:
                source = record.metadata.get("source_id")
                if not isinstance(source, str) or source not in expected or source in seen:
                    raise ValueError("store has an unknown or duplicated source identity")
                item = expected[source]
                supplied = runner._memory_content(item)
                text = (
                    supplied
                    if isinstance(supplied, str)
                    else "\n".join(part for part in supplied if isinstance(part, str))
                )
                if record.content != unicodedata.normalize("NFC", text.strip()):
                    raise ValueError(f"store text differs for source {source}")
                if (record.occurred_at, record.occurred_end) != (
                    item.occurred_at,
                    item.occurred_end,
                ):
                    raise ValueError(f"store timestamps differ for source {source}")
                paths = tuple(part for part in item.content if isinstance(part, Path))
                digests = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
                if tuple(asset.sha256 for asset in record.assets) != digests:
                    raise ValueError(f"store assets differ for source {source}")
                for asset in record.assets:
                    if (
                        asset.path is None
                        or hashlib.sha256(asset.path.read_bytes()).hexdigest() != asset.sha256
                    ):
                        raise ValueError(f"stored asset bytes differ for source {source}")
                asset_count += len(record.assets)
                seen.add(source)
            cursor = page.next_cursor
            if cursor is None:
                break
    if seen != set(expected):
        raise ValueError("store does not contain the entire corpus")
    return {
        "source_records": len(seen),
        "asset_count": asset_count,
        "source_ids_sha256": hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest(),
        "memory_sha256": task.input_sha256["memory"],
        "verification": "public SDK text/timestamps/asset identities and asset bytes",
    }


def embedding_contract(config: SenseMemConfig) -> str:
    """Exclude credentials and answer policy from the reusable embedding contract."""
    payload = {
        "embedding": config.embedding.model_dump(mode="json", exclude={"api_key"}),
        "index_quantization": config.settings.index_quantization.value,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def embedding_fingerprint(  # noqa: C901 - one SQLite snapshot includes each typed field and table
    store: Path,
) -> dict[str, object]:
    """Hash authoritative vectors and text selectors in one read-only, closed-store snapshot."""
    database = store.resolve() / "state.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    tables: dict[str, object] = {}
    with (
        closing(DataDirectoryLock(database.parent)),
        closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection,
    ):
        connection.execute("BEGIN")
        for table in ("embeddings", "embedding_text_selectors", "embedding_text_span_pieces"):
            columns = tuple(connection.execute(f"PRAGMA table_info({table})"))
            if not columns:
                raise ValueError(f"missing authoritative table: {table}")
            keys = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
            digest = hashlib.sha256()
            digest.update(
                json.dumps([(row[1], row[2]) for row in columns], separators=(",", ":")).encode()
            )
            count = 0
            for row in connection.execute(f"SELECT * FROM {table} ORDER BY {','.join(keys)}"):
                digest.update(b"ROW\x00")
                for value in row:
                    match value:
                        case bytes():
                            kind, encoded = b"B", value
                        case None:
                            kind, encoded = b"N", b""
                        case int():
                            kind, encoded = b"I", str(value).encode()
                        case float():
                            kind, encoded = b"F", value.hex().encode()
                        case str():
                            kind, encoded = b"S", value.encode()
                        case _:
                            raise TypeError(
                                f"unsupported SQLite value type: {type(value).__name__}"
                            )
                    digest.update(kind + len(encoded).to_bytes(8, "big") + encoded)
                count += 1
            tables[table] = {"rows": count, "sha256": digest.hexdigest()}
        connection.rollback()
    return {
        "tables": tables,
        "sha256": hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest(),
        "sqlite_snapshot_read_only": True,
    }


def _prepare(
    path: Path, expected_count: int = 1013, gen_kwargs: str = ""
) -> tuple[_Arguments, SenseMemConfig, LoadedTask]:
    parser = _build_parser(None)
    parsed = parser.parse_args(["--config", str(path), "--resume", "--gen-kwargs", gen_kwargs])
    config, overrides = _load_memory_config(path)
    if config is None:
        raise ValueError("an explicit memory configuration is required")
    download = DownloadSettings.resolve(
        overrides.download, benchmarks_root=parsed.benchmarks_root, data_root=parsed.data_root
    )
    arguments = _arguments(parser, parsed, download, overrides=overrides)
    task = runner._load_tasks(arguments, None, None)[0]
    if len(task.units) != 1 or len(task.units[0].questions) != expected_count:
        if expected_count == 1013:
            raise ValueError("full ATM main coverage requires exactly 1,013 questions")
        raise ValueError(f"targeted ATM coverage requires exactly {expected_count} questions")
    model = _evaluation_config(_model_config(arguments.model_args, memory_config=config), (task,))
    resolved = _evaluation_memory_config(
        config, model, gen_kwargs=arguments.gen_kwargs, seed=arguments.seed, device=arguments.device
    )
    assert resolved is not None
    return arguments, resolved, task


async def _ingest(
    arguments: _Arguments, config: SenseMemConfig, task: LoadedTask, run: BenchmarkRun
) -> Path:
    model = _evaluation_config(_model_config(arguments.model_args, memory_config=config), (task,))
    pool = runner._BackendPool(model, memory_config=config)
    store = run.unit_dir(task.units[0].unit_id)
    checkpoint = runner._IngestCheckpoint(
        run, task.units[0].unit_id, runner._ingest_digest(task, arguments, config)
    )
    failures: list[FailureDetail] = []
    last_reported = 0

    def committed(done: int) -> None:
        nonlocal last_reported
        checkpoint.record(done, failures)
        if done - last_reported >= 500 or done == len(task.units[0].memories):
            print(
                json.dumps(
                    {"phase": "ingest", "committed": done, "total": len(task.units[0].memories)}
                ),
                flush=True,
            )
            last_reported = done

    try:
        async with pool.memory(store) as memory:
            count = await runner._ingest(
                memory,
                task.units[0].memories,
                batch_size=_batch_size(arguments, task),
                ingest_concurrency=arguments.ingest_concurrency,
                on_failure=failures.append,
                on_chunk=committed,
            )
        if count:
            raise RuntimeError(f"seed ingest failed for {count} source records")
    finally:
        pool.close()
    return store


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - one native run lifecycle owns its cleanup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--gen-kwargs", default="", help="generation controls for scoring and reader qualification"
    )
    cli = parser.parse_args(argv)
    recipe = Recipe.model_validate_json(cli.recipe.read_text())
    recipe = recipe.model_copy(
        update={
            name: value.expanduser().resolve()
            for name in (
                "base_config_path",
                "artifact_root",
                "source_store",
                "raw_media_root",
                "selection_path",
            )
            if (value := getattr(recipe, name)) is not None
        }
    )
    if (
        hashlib.sha256(recipe.base_config_path.read_bytes()).hexdigest()
        != recipe.base_config_sha256
    ):
        raise ValueError("base configuration SHA-256 differs from the frozen recipe")
    selection = None
    selection_sha256 = None
    selected_ids: frozenset[str] | None = None
    if recipe.selection_path is not None:
        selection_bytes = recipe.selection_path.read_bytes()
        selection = _QuestionSelection.model_validate_json(selection_bytes)
        selected_ids = frozenset(case.question_id for case in selection.cases)
        if len(selected_ids) != len(selection.cases):
            raise ValueError("question selection must not repeat IDs")
        selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    question_count = 1013 if selected_ids is None else len(selected_ids)
    directory = recipe.artifact_root / recipe.variant
    directory.mkdir(parents=True, exist_ok=True)
    if recipe.stage == "reranking-probe" and not recipe.preflight:
        from multimodal_reranking_probe import run_probe

        return run_probe(recipe.base_config_path, directory)
    if recipe.stage == "reader-probe" and not recipe.preflight:
        from atm_reader_role_probe import run_probe as run_reader_probe

        return run_reader_probe(recipe.base_config_path, directory, gen_kwargs=cli.gen_kwargs)
    config_path = directory / "input.yaml"
    payload = yaml.safe_load(recipe.base_config_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("benchmark"), dict):
        raise ValueError("base YAML must contain a benchmark section")
    benchmark = payload["benchmark"]
    download = benchmark.setdefault("download", {})
    run_settings = benchmark.setdefault("run", {})
    if not isinstance(download, dict) or not isinstance(run_settings, dict):
        raise ValueError("benchmark download/run sections must be mappings")
    for name in (
        "request_concurrency",
        "unit_concurrency",
        "ingest_concurrency",
        "judge_concurrency",
    ):
        run_settings.setdefault(name, 16)
    download["data_root"] = str(directory / "stores")
    run_settings.update(
        {
            "tasks": "atm-bench-main-sgm",
            "arms": "sensemem",
            "limit": "all" if selected_ids is None else question_count,
            "offset": 0,
            "run_id": recipe.variant,
            "output_path": str(directory / "results"),
            "use_cache": None,
            "predict_only": False,
            "log_samples": True,
            "stream_results": True,
            "stream_answers": False,
            "download": False,
        }
    )
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    original_loader = cast(Callable[..., LoadedTask], vars(runner)["load_task"])
    original_telemetry = vars(runner)["EvaluationTelemetry"]
    collected: list[ExecutionTelemetry] = []

    class RecordedTelemetry(ExecutionTelemetry):
        def __init__(self) -> None:
            super().__init__()
            collected.append(self)

    def ablation_loader(*args: object, **kwargs: object) -> LoadedTask:
        # Native result metadata retains the subset limit, but source assembly always loads
        # the pinned full QA and complete archive before selecting questions alone.
        loading = {**kwargs, "limit": None, "offset": 0} if selected_ids is not None else kwargs
        task = original_loader(*args, **loading)
        if recipe.raw_media:
            root = cast(Path, kwargs["root"])
            raw = load_task(TASKS["atm-bench-main"], root=root, media_root=recipe.raw_media_root)
            task = attach_raw_media(task, raw)
        if selected_ids is not None:
            if (
                len(task.units) != 1
                or len(task.units[0].questions) != 1013
                or len(task.units[0].memories) != 11034
            ):
                raise ValueError("question selection requires full 1,013 QA and 11,034 memories")
            unit = task.units[0]
            questions = tuple(
                question for question in unit.questions if question.question_id in selected_ids
            )
            if len(questions) != question_count:
                raise ValueError("question selection contains IDs absent from the full QA")
            task = replace(
                task,
                units=(replace(unit, questions=questions),),
                input_sha256={
                    **task.input_sha256,
                    "question_selection": cast(str, selection_sha256),
                },
            )
        return task

    vars(runner)["load_task"] = ablation_loader
    vars(runner)["EvaluationTelemetry"] = RecordedTelemetry
    try:
        arguments, config, task = _prepare(
            config_path, expected_count=question_count, gen_kwargs=cli.gen_kwargs
        )
        manifest: dict[str, object] = {
            "variant": recipe.variant,
            "stage": recipe.stage,
            "raw_media": recipe.raw_media,
            "dataset_sha256": task.dataset_sha256,
            "evaluation_sha256": task.evaluation_sha256,
            "input_sha256": dict(task.input_sha256),
            "implementation_sha256": runner._implementation_identity(),
            "source_store": None if recipe.source_store is None else str(recipe.source_store),
            "question_count": question_count,
            "complete_main": selected_ids is None,
            "full_qa_sha256": task.dataset_sha256,
            "selection_path": (
                None if recipe.selection_path is None else str(recipe.selection_path)
            ),
            "selection_sha256": selection_sha256,
            "selection_source_samples_sha256": (
                None if selection is None else selection.source_samples_sha256
            ),
            "concurrency": arguments.request_concurrency,
            "generation_kwargs": arguments.gen_kwargs,
            "all_predictions_fresh": True,
            "all_judges_fresh": True,
            "recipe_sha256": hashlib.sha256(cli.recipe.read_bytes()).hexdigest(),
            "base_config_sha256": hashlib.sha256(recipe.base_config_path.read_bytes()).hexdigest(),
            "embedding_contract_sha256": embedding_contract(config),
            "measure_search_replay": recipe.measure_search_replay,
        }
        if recipe.preflight:
            (directory / "preflight.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "variant": recipe.variant,
                        "questions": question_count,
                        "complete_main": selected_ids is None,
                    }
                ),
                flush=True,
            )
            return 0
        run = BenchmarkRun(arguments.data_root, task.spec.name, arguments.run_id, resume=True)
        store = run.unit_path(task.units[0].unit_id)
        if recipe.source_store is not None and not store.exists():
            sidecar = recipe.source_store.with_name(f"{recipe.source_store.name}.ablation.json")
            if sidecar.exists():
                source_manifest = json.loads(sidecar.read_text())
                if source_manifest["embedding_contract_sha256"] != embedding_contract(config):
                    raise ValueError("source store embedding contract differs from this variant")
                manifest["source_embedding_contract_verified"] = True
            else:
                manifest["source_embedding_contract_verified"] = False
            manifest["reused_authoritative_state"] = verify_store(recipe.source_store, config, task)
            manifest["source_embedding_fingerprint_before_copy"] = embedding_fingerprint(
                recipe.source_store
            )
            shutil.copytree(recipe.source_store, store)
            manifest["checkpoint_digest_rebound_after_public_verification"] = True
        if not store.exists():
            store = asyncio.run(_ingest(arguments, config, task, run))
        manifest["verified_store"] = verify_store(store, config, task)
        manifest["store"] = str(store)
        before_reindex = embedding_fingerprint(store)
        manifest["embedding_fingerprint_before_reindex"] = before_reindex
        if "source_embedding_fingerprint_before_copy" in manifest:
            assert manifest["source_embedding_fingerprint_before_copy"] == before_reindex, (
                "store copy changed authoritative embedding state"
            )
        with Memory.from_config(config.model_copy(update={"data_dir": store})) as memory:
            indexed = memory.reindex()
        manifest["reindexed_rows"] = indexed
        after_reindex = embedding_fingerprint(store)
        manifest["embedding_fingerprint_after_reindex"] = after_reindex
        assert before_reindex == after_reindex, "reindex changed authoritative embedding state"
        runner._IngestCheckpoint(
            run, task.units[0].unit_id, runner._ingest_digest(task, arguments, config)
        ).record(len(task.units[0].memories), ())
        store.with_name(f"{store.name}.ablation.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        if recipe.stage == "ingest":
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                json.dumps(
                    {"status": "ingest_complete", "variant": recipe.variant, "store": str(store)}
                ),
                flush=True,
            )
            return 0
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with _search_replay(recipe.measure_search_replay):
            status = runner.main(
                ["--config", str(config_path), "--resume", "--gen-kwargs", cli.gen_kwargs]
            )
        result = json.loads((directory / "results/results.jsonl").read_text())
        row = result["tasks"][0]
        summary = {
            "variant": recipe.variant,
            "complete_main": selected_ids is None,
            "score": row["score"]["mean"],
            "questions": row["question_count"],
            "score_valid": row["score_valid"],
            "generation_errors": row["error_count"],
            "samples_sha256": result["samples_sha256"],
            "native_status": result["status"],
            "store": str(store),
            "measure_search_replay": recipe.measure_search_replay,
        }
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        return status
    finally:
        vars(runner)["load_task"] = original_loader
        vars(runner)["EvaluationTelemetry"] = original_telemetry
        if collected:
            (directory / "execution.json").write_text(
                json.dumps(
                    {
                        "measure_search_replay": recipe.measure_search_replay,
                        "collections": [item.execution_result() for item in collected],
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    raise SystemExit(main())
