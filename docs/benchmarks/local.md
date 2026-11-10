# Local benchmark measurements

Size a local deployment and measure maintenance behavior with focused benchmarks for the index,
SDK concurrency, and consolidation. Each utility reports the boundary it measures.
Use the [evaluation runner](../benchmarking.md) for dataset answer quality.

## Measure the local index

```bash
uv run --frozen sensemem-bench local-index \
  --data-dir .benchmarks/local-index/trial-001 \
  --rows 1000 --dimension 128 --queries 20 --k 10 \
  --seed 42 --quantization none
```

The directory must be absent or empty. Output is one JSON object with ingestion/optimization
seconds, exact-search `recall_at_k`, p50/p95/p99 query latency, QPS, and SQLite/Zvec/total disk
bytes. Use a fresh directory for each quantization mode.

This microbenchmark isolates SQLite and Zvec with synthetic vectors. Output records
`scope: storage_microbenchmark`; use an `eval` run's `ingest` block to measure product
ingestion including embedding and media preparation.

## Measure local SDK concurrency

The credential-free load test calls public `Memory` methods with synthetic embeddings:

```bash
uv run --frozen python -m sensemem.benchmarks.sdk_concurrency \
  --workloads get search mixed \
  --concurrency 1 4 8 16 28 32 64 128 \
  --requests 512 --rows 1000 --dimension 128 --repeats 3 \
  --output .benchmarks/sdk-concurrency.jsonl
```

Each workload/concurrency/repetition gets a fresh temporary store, the same seeded corpus,
index optimization, and read warmup. Workers start together and issue the next request after
the previous one finishes, without retries. `mixed` attempts 70% search, 20% ID reads, and 10%
single-record writes; `add` measures writes alone. The output path must be new.

JSONL reports successful throughput, successful and rejected p50/p95/p99 latency, rejection
counts, CPU time, and final record counts. Every search validates its top hit; every acknowledged
write is read back after contention. Unexpected failures stop the run.

| Option | Use |
| --- | --- |
| `--rows 8000` | Measure a corpus larger than local read caches, separately from small repeated working sets |
| `--quantization int8` | Compare quantization with the default unquantized vectors |
| `--retrieval-mode dense` | Compare dense search with default hybrid retrieval |
| `--model-latency-ms 20` | Simulate embedding wait without a model service |
| `--profile-dir .benchmarks/sdk-profiles` | Profile one caller per case; affects timing and excludes child index-thread work |

Compare identical shapes across repeats, including rejection rates. The
[runtime admission limits](../operations.md#concurrency-and-overload) can reject high producer
concurrency; rejections do not count as successful throughput. Synthetic load does not establish
real model-service capacity or retrieval quality.

## Measure consolidation quality

Use a configuration with explicit embedding and consolidation backends and a fresh directory:

```bash
uv run --frozen sensemem-bench control-plane \
  --config eval.yaml --data-dir .benchmarks/control-plane/trial-001
```

This utility ignores the `benchmark` mapping. Its seeded fixture writes standing facts,
paraphrases, and changed preferences, then creates two empty recalls as failure signals.
Preference claims are fixture-created `TRAIT` records, so the measurement covers consolidation
rather than formation quality.

After `consolidate_pending()`, the runner scores applied operations and attempts rollback newest
first. The deliberation window covers the whole fixture.

| Metric | Definition |
| --- | --- |
| `consolidation_precision` | Consolidations citing exactly one injected duplicate group / all applied consolidations |
| `contradiction_recovery` | Preference flips with the new claim active and old claim retired/forgotten / injected flips |
| `false_retirement` | Retired standing gold records / all retired records |
| `rollback_success` | Reversed operations / applied operations; `state_restored` separately checks final records |
| `deliberation.model_calls` | Backend round trips; a request-count proxy, not token or currency cost |

Undefined rates are `null`. The fixture sets `minimum_relevance=0` and disables answer
reinforcement. Report the consolidator revision and scenario seed. Use representative application
evidence for deployment-specific quality evaluation.

To measure the slow loop's effect on dataset answers, use `eval --deliberate` with a
`consolidation` configuration. It runs after each causal cutoff's ingestion, before its questions,
and records enabled state and applied operation count.
