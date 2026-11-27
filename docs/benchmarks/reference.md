# Evaluation reference

Control an evaluation, compare memory strategies, diagnose answer quality, serve the evaluation Add/Search adapter, and measure performance. This reference covers the options and result fields behind [Benchmarking](../benchmarking.md).

Use [tasks](tasks.md) for dataset protocols and [local measurements](local.md) for storage, concurrency, and consolidation utilities.

## Configuration and execution

`--config` accepts the product configuration plus a harness-owned `benchmark` mapping. Keep model endpoints, modalities, and provider controls in product sections; put judge, download, server-metrics, performance-budget, and run settings under `benchmark`. See the [annotated example](../examples/eval.example.yaml) for file fields and `uv run --frozen sensemem-bench eval --help` for CLI options.

| Environment variable | Purpose or default |
| --- | --- |
| `SENSEMEM_GENERATION_API_KEY` | Reader key; falls back to `OPENAI_API_KEY` |
| `SENSEMEM_GENERATION_BASE_URL` | Reader URL; falls back to `OPENAI_BASE_URL` |
| `SENSEMEM_GENERATION_MODEL` | Reader model; defaults to `gpt-5-mini-2025-08-07` |
| `SENSEMEM_GENERATION_MODALITIES` | Comma-separated modalities or `omni` |
| `SENSEMEM_TIMEOUT_SECONDS` | Reader timeout (`300` seconds) |
| `SENSEMEM_JUDGE_MODEL`, `SENSEMEM_JUDGE_BASE_URL`, `SENSEMEM_JUDGE_API_KEY` | Separate judge settings; otherwise reuse reader endpoint |
| `SENSEMEM_JUDGE_TIMEOUT_SECONDS` | Judge timeout |
| `HF_HOME`, `HF_ENDPOINT`, `SENSEMEM_BENCH_YOUTUBE_SLEEP_SECONDS` | Dataset cache, endpoint, and YouTube download pacing |

CLI flags override file values, then environment, then defaults. `--model-args` overrides reader `generation_model`, `base_url`, `timeout_seconds`, and `generation_min_video_seconds`; `--judge-model-args` accepts `model`, `base_url`, `api_key`, and `timeout_seconds`. The `sensemem` arm requires an explicit `embedding` section.

Select sampling with `--gen-kwargs temperature=0.7,max_tokens=512` and `--seed` (temperature defaults to `0`). Evaluation files reject `generation.temperature` and `generation.seed`, and `extra_body` cannot override sampling or token limits. `generation.answer_style: concise` selects the concise answer style for product and generating baseline arms.

Search latency replay is enabled by default. Pass `--no-search-replay` or set `benchmark.run.search_replay: false` to skip post-answer standalone search replay while preserving answer-time retrieval and scoring (`--search-replay` overrides a disabled file setting).

Invocation-only flags: `--blind`, `--blind-baseline`, `--resume`, `--deliberate`, `--fallback-reference-at`, `--list-tasks`, and `--check-integrity`.

### Ingestion, concurrency, and clocks

- `--ingest add` (default) uses `add`/`add_many`; `--ingest capture` measures `capture` followed by `settle`.
- `--ingest-concurrency` defaults to `1`; higher values require `add` mode, tasks without causal cutoffs, no deliberation, and no formation, speech, or face backends.
- `--deliberate` runs `consolidate_pending()` after each causal cutoff's ingestion (requires a `consolidation` section).
- `--request-concurrency` bounds answer requests across units within the runtime's 16-store and 28-operation budgets (8 concurrent streams per store).
- `--no-stream-answers` selects buffered `AsyncMemory.ask()` instead of `ask_stream()`; `--no-stream-results` defers judging until all answers finish; `--quiet` suppresses progress tables.
- `--fallback-reference-at 2026-09-08T00:00:00Z` supplies a timezone-aware question clock only when none exists in the dataset.

## Data and isolation

`--benchmarks-root` defaults to `<repository>/.benchmarks`, and `--data-root` defaults to `<benchmarks-root>/data`. Independent units receive separate physical directories. CUDA runs acquire a device lock unless disabled via `--no-device-lock`.

Validate existing inputs without model calls or store creation:

```bash
uv run --frozen sensemem-bench eval \
  --tasks locomo-refined --check-integrity --no-download
```

Use repeatable `--task-data TASK=PATH` and `--media-root TASK=PATH` to override inputs (`--allow-unverified-data` permits digest mismatches on custom overrides, recorded in output metadata).

Timed media prepares into `.benchmarks/.prepared/`. `--media-manifest FILE` supplies prepared observations (`tasks` → `units` → observations with `path` and/or `text`, optional source IDs, and intervals); inspect a generated `OUTPUT/media-manifest.jsonl` for the format.

## Baseline arms

Select comma-separated arms with `--arms` (default `sensemem`):

| Arm | Input and output |
| --- | --- |
| `sensemem` | Product retrieval and grounded answers via `Memory.ask` (PersonaMem-v3 uses `Memory.compile`) |
| `blind` | Reader with no evidence; answer metrics only |
| `full-context` | Oldest-first corpus text under `--full-context-chars`; answer metrics only |
| `random` | Seeded shuffle of an independently retrieved top-100 pool for gold-labelled questions (`recall_limit` otherwise); retrieval metrics only |
| `compile` | Rendered `Memory.compile` bundle under `--compile-max-items` / `--compile-max-chars` passed to the reader |

Baseline arms use versioned harness prompts and are marked non-official. Reuse a separate blind run by passing its output directory to `--blind-baseline` with matching reader, judge, task selection, seed, and answer policy.

## Controls and diagnostics

The `controls` block reports the blind score, random-ranker expectation, and R@1 beside R@20 (computed from ranked candidates observed inside the answer before grounding). Missing applicable controls populate `missing` and set `interpretable: false`.

| Gold-source coverage | Tasks |
| --- | --- |
| Exact source IDs | LoCoMo-Refined, ATM-Bench, Mem-Gallery |
| Session groups | MemLens |
| Clip IDs (may be incomplete) | ICM-Bench |
| Temporal/spatial labels | VideoZeroBench, MM-Lifelong, SuperMemory-VQA |
| No compatible retrieval labels | AMA-Bench, LongMemEval-V2, M3-Bench, PersonaMem-v3, EgoTempo |

Most tasks default to `strict` answering; M3 robot, ICM, and MM-Lifelong default to `best_effort` (override with `--answer-policy`).

### Diagnose a failed answer

Samples join `quality_diagnostics` across `input`, `retrieval`, `organization`, `reader_input`, and `answer` layers. Text-only media projections appear separately in `projection_source_ids` and `projection_evidence` rather than full source coverage.

Export a stratified human-review sample without model calls:

```bash
uv run --frozen sensemem-bench review-samples \
  --samples .benchmarks/results/locomo-baseline/samples.jsonl \
  --output .benchmarks/reviews/locomo.jsonl --per-stratum 10 --seed 0
```

Score independently annotated claim-entailment fixtures with:

```bash
uv run --frozen python -m sensemem.benchmarks.semantic_support \
  --input benchmarks/semantic_support.example.json \
  --output .benchmarks/semantic-support.json
```

## Results and failures

`results.jsonl` holds one run record with task/arm aggregates; `samples.jsonl` holds per-question predictions; resolved `config.yaml` omits credentials. `--log-samples` adds prompts, references, and raw judge responses. An in-progress `samples.partial.jsonl` crash copy is removed once final artifacts commit.

Transient transport, rate-limit, and HTTP 5xx errors retry with exponential backoff for up to ten minutes. Exhausted generation errors score zero and stay in denominators and `error_count`; judge failures zero judge-dependent metrics while preserving deterministic scores. Cluster-bootstrap 95% confidence intervals treat each isolated store as one independent cluster (`null` below two clusters).

## Performance measurements

Run at least three fresh repeats without response caching or resume:

```bash
for repeat in 0 1 2; do
  uv run --frozen sensemem-bench eval \
    --config eval.yaml --tasks locomo-refined --limit all --seed 42 \
    --repeat-index "$repeat" --run-id "perf-r$repeat" \
    --output-path ".benchmarks/results/perf-r$repeat"
done
```

Each task/arm `performance` block reports `ingest`, `answer`, `ask_retrieval_core`, `search_e2e` (when replay is enabled), `capture`/`time_to_searchable_ms`/`formation` (capture mode), `asr`, `compile`, `recall`, `nodes`, `token_usage`, and client `resources` (CPU, RSS high-water mark, storage growth, local GPU readings, plus optional `benchmark.server_metrics` vLLM deltas). Token totals are never estimated.

Enforce relative performance budgets on a compatible `--compare` run (or under `benchmark.performance_budgets`):

```bash
uv run --frozen sensemem-bench eval \
  --config eval.yaml --tasks locomo-refined --limit all --seed 42 \
  --compare .benchmarks/results/perf-r0 \
  --performance-budget answer_e2e_ttft_p95=0.10 \
  --performance-budget answer_e2e_latency_p95=0.10 \
  --performance-budget retrieval_e2e_latency_p95=0.15 \
  --performance-budget tokens_per_call=0.05 \
  --performance-budget answer_throughput=0.10 \
  --fail-on-regression \
  --run-id perf-candidate --output-path .benchmarks/results/perf-candidate
```

## Cache and resume

`--use-cache .benchmarks/response-cache` caches completed answers and judge verdicts as each question finishes. `--resume` reuses committed ingestion checkpoints from the same explicit `--run-id`. Existing final outputs are protected unless `--overwrite` is passed. Cached and resumed runs cannot serve as performance baselines.

## Serve the evaluation Add/Search adapter

`sensemem-bench eval serve` exposes participant-hosted `/add`, `/search`, and `/health` HTTP endpoints over one isolated physical store per `user_id` (`users/<sha256(user_id)>`) while sharing loaded model weights across requests (requires `benchmarks` and `server` extras plus an explicit `embedding` section):

```bash
export SENSEMEM_AML_API_KEY="replace-with-your-key"
uv run --frozen sensemem-bench eval serve \
  --config .benchmarks/configs/eval.yaml \
  --data-root .benchmarks/eval-serve \
  --host 127.0.0.1 --port 8000
```

Use `--api-key-env NAME` to change the key variable, or `--public-smoke` for local unauthenticated smoke checks. Run one server process per `--data-root` behind HTTPS for remote evaluators.

## Export predictions

Export a completed `ama-bench` run (`--limit all` with `Qwen/Qwen3-32B` judge):

```bash
uv run --frozen sensemem-bench ama-bench export \
  --dataset .benchmarks/ama-bench/test/open_end_qa_set.jsonl \
  --samples .benchmarks/results/ama-bench/samples.jsonl \
  --output .benchmarks/results/ama-bench/submission.jsonl
```

Export raw `locomo-refined` predictions for an external evaluator:

```bash
uv run --frozen sensemem-bench locomo-refined \
  --config eval.yaml \
  --dataset .benchmarks/locomo-refined/data/raw/locomo_refined.json \
  --output .benchmarks/results/locomo-raw/predictions.jsonl \
  --data-root .benchmarks/data --run-id locomo-raw-001 --limit 1
```
