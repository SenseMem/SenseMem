"""Standalone SQLite + Zvec storage microbenchmark.

This is the narrow storage microbenchmark AGENTS.md allows to touch the local adapters: its
purpose is to measure SQLite and Zvec directly on synthetic vectors. Its numbers are not
product ingest or search numbers and must not be reported as such - it never embeds, never
routes a modality, never prepares media, and never grounds an answer. The end-to-end figures
the design doc asks for come from ``sensemem-bench eval``, which drives the public SDK.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import struct
from collections.abc import Sequence
from datetime import datetime, timezone
from heapq import nlargest
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from sensemem.benchmarks.eval_statistics import percentile
from sensemem.infrastructure.local import (
    IndexDocument,
    LocalStore,
    StoredEmbedding,
    StoredMemory,
)
from sensemem.infrastructure.local.zvec_index import ZvecIndex
from sensemem.kernel.metadata import index_recipe_for
from sensemem.types import IndexQuantization

_BATCH_SIZE = 512
_MODEL_ID = "synthetic-fp32"
_SPACE_ID = "synthetic-cosine-v1"
_TASK = "retrieval.document"
_NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


def run_benchmark(
    data_dir: str | Path,
    *,
    rows: int = 1_000,
    dimension: int = 128,
    queries: int = 20,
    k: int = 10,
    seed: int = 42,
    quantization: IndexQuantization = IndexQuantization.NONE,
) -> dict[str, object]:
    """Run one clean, deterministic local-index benchmark and return JSON-ready metrics."""
    _validate_shape(rows=rows, dimension=dimension, queries=queries, k=k)
    root = Path(data_dir).resolve()
    _require_empty_directory(root)
    rng = random.Random(seed)
    fp32 = struct.Struct(f"<{dimension}f")

    with (
        LocalStore(root) as store,
        ZvecIndex(
            root / "index",
            dimension,
            quantization=quantization,
        ) as index,
    ):
        ingest_started = perf_counter()
        for offset in range(0, rows, _BATCH_SIZE):
            batch_rows = min(_BATCH_SIZE, rows - offset)
            memories, embeddings, documents = _synthetic_batch(
                rng,
                fp32,
                offset=offset,
                rows=batch_rows,
            )
            store.records.write_memories(memories, embeddings)
            _drain_batch(store, index, documents)
        ingest_seconds = perf_counter() - ingest_started

        optimize_started = perf_counter()
        index.optimize()
        index.flush()
        optimize_seconds = perf_counter() - optimize_started

        query_vectors = tuple(_normalized_fp32(rng, fp32) for _query in range(queries))
        recall, latencies = _measure_queries(store, index, query_vectors, rows=rows, k=k)

    query_seconds = sum(latencies)
    sqlite_bytes = sum(
        path.stat().st_size for path in root.glob("state.sqlite3*") if path.is_file()
    )
    zvec_bytes = _tree_bytes(root / "index")
    return {
        "scope": "storage_microbenchmark",
        "excludes": (
            "embedding",
            "modality routing",
            "media preparation",
            "generation",
            "the public Memory API",
        ),
        "rows": rows,
        "dimension": dimension,
        "queries": queries,
        "k": k,
        "seed": seed,
        "quantization": quantization.value,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "sensemem": version("sensemem"),
            "zvec": version("zvec"),
            "numpy": np.__version__,
        },
        "configuration": {
            "index_recipe": index_recipe_for(quantization),
            "ef_search": index.ef_search,
        },
        "ingest_seconds": ingest_seconds,
        "optimize_seconds": optimize_seconds,
        "recall_at_k": recall,
        "recall_truth": (
            "exhaustive cosine over SQLite FP32 vectors with FP64 accumulation; "
            "ties at the kth score count as correct"
        ),
        "query_latency_ms": {
            "p50": _milliseconds(latencies, 0.50),
            "p95": _milliseconds(latencies, 0.95),
            "p99": _milliseconds(latencies, 0.99),
        },
        "query_qps": queries / query_seconds,
        "disk_bytes": {
            "sqlite": sqlite_bytes,
            "zvec": zvec_bytes,
            "total": sqlite_bytes + zvec_bytes,
        },
    }


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    """Parse CLI arguments, run the benchmark, and print one JSON document."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--rows", type=_positive_int, default=1_000)
    parser.add_argument("--dimension", type=_positive_int, default=128)
    parser.add_argument("--queries", type=_positive_int, default=20)
    parser.add_argument("--k", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quantization",
        choices=tuple(mode.value for mode in IndexQuantization),
        default=IndexQuantization.NONE.value,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = run_benchmark(
        arguments.data_dir,
        rows=arguments.rows,
        dimension=arguments.dimension,
        queries=arguments.queries,
        k=arguments.k,
        seed=arguments.seed,
        quantization=IndexQuantization(arguments.quantization),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _synthetic_batch(
    rng: random.Random,
    fp32: struct.Struct,
    *,
    offset: int,
    rows: int,
) -> tuple[list[StoredMemory], list[StoredEmbedding], list[IndexDocument]]:
    memories = []
    embeddings = []
    documents = []
    for row in range(offset, offset + rows):
        memory = StoredMemory(
            memory_id=f"memory-{row:09d}",
            content=f"Synthetic memory row {row}",
            metadata_json=f'{{"row":{row}}}',
            created_at=_NOW,
            updated_at=_NOW,
        )
        embedding = StoredEmbedding(
            embedding_id=f"embedding-{row:09d}",
            memory_id=memory.memory_id,
            values=_normalized_fp32(rng, fp32),
            model_id=_MODEL_ID,
            space_id=_SPACE_ID,
            task=_TASK,
            created_at=_NOW,
            normalized=True,
        )
        memories.append(memory)
        embeddings.append(embedding)
        documents.append(
            IndexDocument(
                embedding=embedding,
                content=memory.content,
                metadata_json=memory.metadata_json,
            )
        )
    return memories, embeddings, documents


def _drain_batch(
    store: LocalStore,
    index: ZvecIndex,
    documents: Sequence[IndexDocument],
) -> None:
    operations = store.index.pending_index_operations(limit=len(documents))
    expected_ids = tuple(document.embedding.embedding_id for document in documents)
    if tuple(operation.embedding_id for operation in operations) != expected_ids or any(
        operation.action != "upsert" for operation in operations
    ):
        raise RuntimeError("local index outbox does not match the synthetic write batch")
    index.upsert(documents)
    index.flush()
    if store.index.acknowledge_index_operations(operations) != len(operations):
        raise RuntimeError("local index outbox acknowledgement was incomplete")


def _measure_queries(
    store: LocalStore,
    index: ZvecIndex,
    query_vectors: Sequence[tuple[float, ...]],
    *,
    rows: int,
    k: int,
) -> tuple[float, tuple[float, ...]]:
    index.search(
        query_vectors[0],
        limit=k,
        space_id=_SPACE_ID,
        task=_TASK,
    )
    approximate_ids = []
    latencies = []
    for vector in query_vectors:
        started = perf_counter()
        approximate = index.search(
            vector,
            limit=k,
            space_id=_SPACE_ID,
            task=_TASK,
        )
        latencies.append(perf_counter() - started)
        approximate_ids.append({hit.id for hit in approximate})

    # A linear query against a quantized index shares its quantization error. Read SQLite's
    # original FP32 buffers instead, retaining only one vector batch and each query's top k.
    query_matrix = np.asarray(query_vectors, dtype=np.float64)
    query_matrix /= np.linalg.norm(query_matrix, axis=1, keepdims=True)
    closest_scores: list[list[float]] = [[] for _vector in query_vectors]
    found_scores: list[list[float]] = [[] for _vector in query_vectors]
    for offset in range(0, rows, _BATCH_SIZE):
        parts = tuple(
            store.index.iter_memory_embedding_parts(
                tuple(
                    f"memory-{row:09d}" for row in range(offset, min(rows, offset + _BATCH_SIZE))
                ),
                space_id=_SPACE_ID,
                task=_TASK,
            )
        )
        document_matrix = np.stack(
            [np.frombuffer(packed, dtype="<f4") for _memory_id, _embedding_id, packed in parts]
        ).astype(np.float64)
        document_matrix /= np.linalg.norm(document_matrix, axis=1, keepdims=True)
        scores = query_matrix @ document_matrix.T
        for query_number, query_scores in enumerate(scores):
            closest_scores[query_number] = nlargest(
                k, (*closest_scores[query_number], *(float(score) for score in query_scores))
            )
            found_scores[query_number].extend(
                float(score)
                for score, (_memory_id, embedding_id, _packed) in zip(
                    query_scores, parts, strict=True
                )
                if embedding_id in approximate_ids[query_number]
            )
    # Every member of a tie at the kth boundary is a correct neighbor, including in dimension
    # one where many normalized vectors are identical. Keep k as the denominator for all ties.
    recalls = [
        sum(score >= closest[-1] for score in found) / k
        for closest, found in zip(closest_scores, found_scores, strict=True)
    ]
    return sum(recalls) / len(recalls), tuple(latencies)


def _normalized_fp32(rng: random.Random, fp32: struct.Struct) -> tuple[float, ...]:
    dimension = fp32.size // 4
    while True:
        values = [rng.random() * 2.0 - 1.0 for _value in range(dimension)]
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            return fp32.unpack(fp32.pack(*(value / norm for value in values)))


def _milliseconds(values: Sequence[float], quantile: float) -> float:
    seconds = percentile(values, quantile)
    if seconds is None:
        raise ValueError("query latency percentiles need at least one measurement")
    return seconds * 1_000.0


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _require_empty_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"benchmark data path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"benchmark data directory is not empty: {path}")
        return
    path.mkdir(mode=0o700, parents=True)


def _validate_shape(*, rows: int, dimension: int, queries: int, k: int) -> None:
    for name, value in (("rows", rows), ("dimension", dimension), ("queries", queries), ("k", k)):
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if k > rows:
        raise ValueError("k must not exceed rows")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
