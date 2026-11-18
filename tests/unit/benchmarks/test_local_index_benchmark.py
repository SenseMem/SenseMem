"""Small checks for the standalone local index benchmark."""

from __future__ import annotations

import json
import math
import random
import struct
from pathlib import Path
from typing import cast

import pytest

import sensemem.benchmarks.local_index_benchmark as benchmark
from sensemem.benchmarks.local_index_benchmark import (
    _normalized_fp32,
    main,
    run_benchmark,
)
from sensemem.infrastructure.local import LocalStore
from sensemem.infrastructure.local.zvec_index import IndexHit, ZvecIndex
from sensemem.types import IndexQuantization


def test_fp32_vectors_are_deterministic_and_normalized() -> None:
    fp32 = struct.Struct("<8f")
    first = _normalized_fp32(random.Random(7), fp32)
    second = _normalized_fp32(random.Random(7), fp32)

    assert first == second
    assert fp32.pack(*first) == fp32.pack(*second)
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0, abs=1e-6)


def test_small_benchmark_reports_valid_metrics(tmp_path: Path) -> None:
    pytest.importorskip("zvec", reason="Zvec 0.8 native wheel is not installed")
    result = run_benchmark(tmp_path, rows=32, dimension=8, queries=4, k=3, seed=7)

    assert set(result) == {
        "scope",
        "excludes",
        "rows",
        "dimension",
        "queries",
        "k",
        "seed",
        "quantization",
        "environment",
        "configuration",
        "ingest_seconds",
        "optimize_seconds",
        "recall_at_k",
        "recall_truth",
        "query_latency_ms",
        "query_qps",
        "disk_bytes",
    }
    assert result["rows"] == 32
    assert result["dimension"] == 8
    assert result["queries"] == 4
    assert result["k"] == 3
    assert result["seed"] == 7
    assert result["quantization"] == "none"
    # The microbenchmark must label itself so its numbers cannot be read as product ingest
    # or search latency; the SDK-path figures come from ``sensemem-bench eval``.
    assert result["scope"] == "storage_microbenchmark"
    assert "the public Memory API" in cast(tuple[str, ...], result["excludes"])
    assert cast(float, result["ingest_seconds"]) > 0.0
    assert cast(float, result["optimize_seconds"]) >= 0.0
    assert 0.0 <= cast(float, result["recall_at_k"]) <= 1.0
    assert cast(float, result["query_qps"]) > 0.0

    latency = cast(dict[str, float], result["query_latency_ms"])
    assert set(latency) == {"p50", "p95", "p99"}
    assert 0.0 <= latency["p50"] <= latency["p95"] <= latency["p99"]
    disk = cast(dict[str, int], result["disk_bytes"])
    assert set(disk) == {"sqlite", "zvec", "total"}
    assert disk["sqlite"] > 0
    assert disk["zvec"] > 0
    assert disk["total"] == disk["sqlite"] + disk["zvec"]
    json.dumps(result)

    with LocalStore(tmp_path) as store:
        assert store.index.pending_index_operations() == ()


@pytest.mark.parametrize(
    ("stored_vectors", "query", "returned_rows", "expected"),
    (
        (((1.0, 0.0), (0.0, 1.0)), (0.0, 1.0), (0,), 0.0),
        (((1.0,), (1.0,), (1.0,)), (1.0,), (1, 2), 1.0),
    ),
    ids=("quantized_linear_agreement_is_not_truth", "kth_boundary_ties_are_correct"),
)
def test_recall_uses_sqlite_fp32_truth_and_accepts_boundary_ties(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored_vectors: tuple[tuple[float, ...], ...],
    query: tuple[float, ...],
    returned_rows: tuple[int, ...],
    expected: float,
) -> None:
    pytest.importorskip("zvec", reason="Zvec 0.8 native wheel is not installed")
    vectors = iter((*stored_vectors, query))
    monkeypatch.setattr(benchmark, "_BATCH_SIZE", 2)
    monkeypatch.setattr(benchmark, "_normalized_fp32", lambda *_args: next(vectors))
    # ANN and a linear query can agree on the wrong quantized neighbor. Neither can supply
    # independent truth; equal FP32 neighbors at the kth boundary must also remain correct.
    monkeypatch.setattr(
        ZvecIndex,
        "search",
        lambda *_args, **_kwargs: tuple(
            IndexHit(id=f"embedding-{row:09d}", relevance=1.0) for row in returned_rows
        ),
    )

    result = run_benchmark(
        tmp_path,
        rows=len(stored_vectors),
        dimension=len(query),
        queries=1,
        k=len(returned_rows),
        quantization=IndexQuantization.FP16,
    )

    assert result["recall_at_k"] == expected


def test_cli_prints_one_json_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("zvec", reason="Zvec 0.8 native wheel is not installed")
    data_dir = tmp_path / "cli"

    assert (
        main(
            [
                "--rows",
                "16",
                "--dimension",
                "4",
                "--queries",
                "2",
                "--k",
                "2",
                "--seed",
                "3",
                "--data-dir",
                str(data_dir),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["rows"] == 16
    assert report["quantization"] == "none"
    assert 0.0 <= report["recall_at_k"] <= 1.0


def test_benchmark_rejects_invalid_or_dirty_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="k must not exceed rows"):
        run_benchmark(tmp_path / "invalid", rows=1, dimension=2, queries=1, k=2)

    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        run_benchmark(dirty, rows=1, dimension=2, queries=1, k=1)
