# Evaluation reference

For the normal workflow, start with [Benchmarking](../benchmarking.md). This page covers run
controls, diagnostics, performance, and recovery. See [tasks](tasks.md) for dataset protocols and
[local measurements](local.md) for storage, concurrency, and consolidation utilities.

## Configuration and execution

`--config` accepts the product configuration plus a harness-owned `benchmark` mapping. Keep model
endpoints, modalities, and provider controls in their product sections; put judge, download,
server-metrics, performance-budget, and run settings under `benchmark`. The
[annotated example](../examples/eval.example.yaml) lists every file field. Use
`uv run --frozen sensemem-bench eval --help` for CLI options.

| Environment variable | Purpose or default |
| --- | --- |
| `SENSEMEM_GENERATION_API_KEY` | Reader key; falls back to `OPENAI_API_KEY` |
| `SENSEMEM_GENERATION_BASE_URL` | Reader URL; falls back to `OPENAI_BASE_URL` |
| `SENSEMEM_GENERATION_MODEL` | Reader model; defaults to `gpt-5-mini-2025-08-07` |
| `SENSEMEM_GENERATION_MODALITIES` | Comma-separated modalities or `omni` |
| `SENSEMEM_TIMEOUT_SECONDS` | Reader timeout; 300 seconds |
| `SENSEMEM_JUDGE_MODEL`, `SENSEMEM_JUDGE_BASE_URL`, `SENSEMEM_JUDGE_API_KEY` | Separate judge settings; otherwise reuse the reader endpoint |
| `SENSEMEM_JUDGE_TIMEOUT_SECONDS` | Judge timeout |
| `HF_HOME`, `HF_ENDPOINT`, `SENSEMEM_BENCH_YOUTUBE_SLEEP_SECONDS` | Dataset cache, endpoint, and YouTube download pacing |

CLI flags override file values, then environment, then defaults. `--model-args` overrides reader
`generation_model`, `base_url`, `timeout_seconds`, and `generation_min_video_seconds`;
`--judge-model-args` accepts `model`, `base_url`, `api_key`, and `timeout_seconds`. Prefer environment
variables for keys. The SenseMem arm needs explicit embedding configuration; listing, integrity
checks, and baseline-only runs do not.

Select sampling with `--gen-kwargs temperature=0.7,max_tokens=512,enable_thinking=false` and
`--seed`. Temperature defaults to zero; `do_sample` must agree with it. Evaluation files reject
`generation.temperature` and `generation.seed`, and `extra_body` cannot override sampling or token
limits. Unsupported automatic controls are omitted for reasoning models; explicit unsupported
controls fail validation. Remote seeds do not guarantee deterministic output.

`generation.answer_style: concise` selects the optional answer-only style for product and
answer-generating baseline arms. Provider thinking settings belong in `generation.extra_body`.
Both choices enter cache identities; evaluate their quality with the selected model. The
[ATM](../examples/atm-sgm-max-thinking-vllm.yaml) and
[Mem-Gallery](../examples/mem-gallery-qwen-wemm.yaml) examples show task-specific configurations.

Search latency replay is enabled by default. `--no-search-replay` or
`benchmark.run.search_replay: false` skips the extra public searches after all answers, while
preserving answer-time retrieval and scoring. `--search-replay` overrides a disabled file setting.
Results record `search_replay` and
`measurement_protocol.post_answer_search_replay_in_client_resource_window`. Disabled replay does
not make an otherwise successful run fail; standalone search-latency measurements are unavailable.

These options are invocation-only: `--blind`, `--blind-baseline`, `--resume`, `--deliberate`,
`--fallback-reference-at`, `--list-tasks`, and `--check-integrity`. Evaluation is zero-shot;
`--model`, `--memory-config`, few-shot settings, and underscore option aliases are unsupported.

### Ingestion, concurrency, and clocks

- `--ingest add` is the default: SDK `add`/`add_many` calls return searchable records.
  `--ingest capture` uses `capture` followed by `settle`; all answers still see settled stores.
- `--ingest-concurrency` defaults to 1. Higher values require add mode, questions without causal
  cutoffs, no deliberation, and explicit configuration without formation, speech, or face backends.
  Independent time groups may commit concurrently; unknown or overlapping times serialize work.
- `--deliberate` runs `consolidate_pending()` after each cutoff's ingestion, before its questions.
  It requires a `consolidation` section and is off by default.
- `--request-concurrency` bounds answer requests across units. The harness opens at most 16
  stores and shares a 28-operation process budget. Streaming answers also respect eight per store;
  requested concurrency is a ceiling. Admission waiting counts in caller latency and TTFT.
- `--no-stream-answers` selects buffered `AsyncMemory.ask()` instead of `ask_stream()`, allowing
  up to 28 operations per store without caller-observed TTFT. `stream_results` controls reporting;
  `generation.stream` controls provider transport. These are separate settings.
- `--fallback-reference-at 2026-09-08T00:00:00Z` supplies a question clock only when none exists.
  It must include a timezone and writes no event timestamps. Structured question dates take
  precedence over original-question dates, then corpus dates; execution time is not substituted.

By default, each task is judged and reported before the next task starts. `--no-stream-results`
defers all judging until inference finishes. `--quiet` suppresses progress and result tables;
dependency logs are limited to warnings unless verbosity is `DEBUG`.

## Data and isolation

`--benchmarks-root` defaults to the checkout's `.benchmarks`, independently of the current
directory. `--data-root` defaults to `<benchmarks-root>/data`. Run and unit labels allocate
separate physical directories; labels are never product scope fields. CUDA runs take a device
lock unless an external scheduler owns admission.

Validate existing inputs without opening models, contacting providers, or creating run stores:

```bash
uv run --frozen sensemem-bench eval \
  --tasks locomo-refined --check-integrity --no-download
```

The response reports selected `unit_count`, `question_count`, `dataset_sha256`, and
`evaluation_sha256`. A scored task aggregate reports questions, not this input `unit_count`.

Use repeatable `--task-data TASK=PATH` and `--media-root TASK=PATH` to override inputs.
`--allow-unverified-data` is required when an annotation override fails the pinned digest; its
identity is recorded. `--no-download` prevents acquisition.

Timed media is prepared into bounded clips under `.benchmarks/.prepared/`. Invalid video-cache
entries are rebuilt; genuine one-frame inputs remain one-frame inputs. `--media-manifest FILE`
supplies prepared observations. Version 1 contains `tasks` → `units` → observations with `path`
and/or `text`, plus optional source IDs and start/end seconds. Relative paths resolve beside the
manifest. Causal tasks require intervals and ingest only observations ending by the cutoff.
Use a generated `OUTPUT/media-manifest.jsonl` as the format reference.

Custom behavior benchmarks must use the public SDK and close each isolated store before archiving.
The [local-index microbenchmark](local.md#measure-the-local-index) is the direct-adapter exception.
Retain dataset terms, restrict raw artifacts, and remove them when no longer needed. Private
research reports and measurements belong in ignored `docs/research/` and must stay local.

## Baseline arms

Select comma-separated arms with `--arms`; the default is `sensemem`. Arms share ingestion and
the selected questions. Samples and aggregates carry `arm`, and the run records each definition.

| Arm | Input and output |
| --- | --- |
| `sensemem` | Product retrieval and grounded answers through `Memory.ask`; PersonaMem-v3 uses the compile/assistant surface |
| `blind` | Reader with no evidence; answer metrics only |
| `full-context` | Oldest-first corpus text under `--full-context-chars`; answer metrics only |
| `random` | Seeded shuffle of an independently retrieved top-100 pool for gold-labelled questions, `recall_limit` otherwise; retrieval metrics only |
| `compile` | Rendered `Memory.compile` bundle under `--compile-max-items`/`--compile-max-chars`, passed to the reader |

For example, add `--arms sensemem,blind,full-context,random,compile` to an evaluation command.
The generating baselines call the model directly with versioned harness prompts. They share
resolved sampling settings, answer style, and task refusal policy, but their scores are non-official.
`full-context` stuffs text only; native query assets may still reach the reader.

Full-context text caps, compile bundle caps, and product `evidence_budget_chars` constrain different
surfaces; none is a provider-token limit. Set and report comparable evidence caps for a controlled
comparison, and measure actual tokens separately. Compile samples record bundle characters/items.
Random is a control at its recorded retrieval depth, not a permutation of identical product hits.

These are end-to-end system comparisons. Differences in evidence, prompts, budgets, planning,
and model calls prevent attributing the entire score delta to memory alone. Report answerable and
abstention subsets alongside the unchanged overall metric.

To reuse a separate blind run, run with `--blind`, then pass its output directory through
`--blind-baseline`. Both commands must use the same reader, judge, task selection, seed, and
answer policy. The loader rejects incompatible inputs and blind prompt versions.

## Controls and diagnostics

The `controls` block reports the blind score, random-ranker expectation, and R@1 beside R@20.
Absent applicable controls populate `missing`, set `interpretable: false`, and make run-level
`controls_complete` false. Non-retrieving arms do not require retrieval controls.

Recall uses the ranked candidates observed inside the answer, before grounding, at the same
causal store state; it issues no extra scoring search. Returned `evidence`, observed
`generation_evidence`, and prepared reader input are separate observations. Record candidate depth
and returned count. Unranked labelled questions and unresolved gold IDs have explicit counters.
The random-ranker expectation is `min(1, k / pool_size)` for individual labels; grouped labels
use the matching hypergeometric expectation.

| Gold-source coverage | Tasks |
| --- | --- |
| Exact source IDs | LoCoMo-Refined, ATM-Bench, Mem-Gallery |
| Session groups | MemLens; ES-MemEval at published session retrieval granularity |
| Clip IDs, possibly incomplete | ICM-Bench |
| Exact images; unresolved memory-point labels | WorldMemArena |
| Temporal/spatial labels, not joined source-record IDs | VideoZeroBench, MM-Lifelong; SuperMemory-VQA retains video IDs without gold clip groups |
| No compatible source labels, or a degenerate single-clip pool | AMA-Bench, LongMemEval-V2, M3-Bench, CL-Bench, BEAM, PersonaMem-v3, EgoTempo, OpenEQA, Video-MME-v2 |

Missing source labels do not make an answer score unavailable, but prevent a retrieval-quality
claim. Task templates and stored `[source_id: ...]` markers also affect retrieval; original bare
questions are retained for scoring, not routed separately through the SDK.

Most tasks use `strict` answering. M3 robot, ICM, and MM-Lifelong use `best_effort`; override with
`--answer-policy`. The resolved policy is recorded and enters comparisons and caches.
`abstentions` counts recognized markers and task-mandated refusal wording, so it is a lower bound.

### Diagnose a failed answer

Samples join `quality_diagnostics` by sample, trace, and answer-span IDs:

| Layer | Observed boundary |
| --- | --- |
| `input` | Gold source presence in eligible corpus and ingest failures |
| `retrieval` | Ranked candidates before planning and generation |
| `organization` | Last evidence list passed to the answerer |
| `reader_input` | Last prepared request's record/asset IDs, text bytes, fallback, and span success |
| `answer` | Score, abstention, product error, and scorer error |

Source-group coverage is unknown when labels or observations are missing or unresolved. Partial
excerpts are not credited as full sources. Even complete source coverage does not prove the
required fact reached the reader: `fact_span_coverage` and `causal_attribution` remain `null`.
Prepared-request success does not establish provider perception or correct reasoning. Cached
answers retain original trace IDs but have no fresh prepared-request telemetry.

Export a deterministic, stratified human-review sample without model calls:

```bash
uv run --frozen sensemem-bench review-samples \
  --samples .benchmarks/results/locomo-baseline/samples.jsonl \
  --output .benchmarks/reviews/locomo.jsonl --per-stratum 10 --seed 0
```

Annotations start pending. Review pinned original sources, record supporting/contradicting spans
with source IDs and units, and obtain independent review and adjudication. Preserve unknowns;
an unweighted review-sample mean is not a population metric. Causal diagnosis requires paired
interventions with the same questions, reader, and scorer, keeping gold inputs outside normal runs.

For independently annotated claim entailment, localization, admission, and withdrawal outcomes:

```bash
uv run --frozen python -m sensemem.benchmarks.semantic_support \
  --input benchmarks/semantic_support.example.json \
  --output /tmp/sensemem-semantic-support.json
```

This provider-free fixture documents the schema and metric calculations. Keep fixture, producer,
and reviewer revisions with real annotations; the scripted example is not a model-quality estimate.

## Results and failures

Artifacts use schema `23` and runner `sensemem_eval_official_v25`. `results.jsonl` contains one
run record with task/arm rows; `samples.jsonl` contains predictions; resolved `config.yaml` omits
credentials and unconstrained provider fields. `--log-samples` adds original questions, prompts,
references, and raw judge responses. An in-progress `samples.partial.jsonl` is a task-level crash
copy with no result digest, removed when final artifacts land.

Transport, rate-limit, and HTTP 5xx failures retry with exponential backoff for up to ten minutes.
Only the successful attempt is timed. After retry exhaustion, generation failures score zero;
judge failures zero judge-dependent metrics while retaining deterministic scores. Both remain in
the mean and `error_count`. Shared embedding-service failures stop dependent queued work, retaining
planned rows as structured errors. Input-specific failures isolate affected items. Local parsing,
cache, or accounting bugs stop evaluation rather than becoming wrong-answer scores.

For reproducibility retain dataset/input/evaluation hashes, source and scorer revisions, immutable
model identities, full memory settings, seed, selected slice, hardware, and measurement protocol.
Questions sharing a store are one independent cluster; cluster-bootstrap intervals are `null`
below two clusters. `noise_floor.minimum_meaningful_difference` combines the configured floor
with cluster uncertainty. It is a reporting guard, not proof of reproducibility.

Quality comparisons require compatible dataset, scorer, judge, and sample identities. Deltas,
intervals, and noise floors carry `score_scale` (1 or 100); regression thresholds are fractions.
Performance comparison applies stricter requirements below.

## Performance measurements

Use at least three independent runs with identical configuration, services, and hardware:

```bash
for repeat in 0 1 2; do
  uv run --frozen sensemem-bench eval \
    --config eval.yaml --tasks locomo-refined --limit all --seed 42 \
    --repeat-index "$repeat" --run-id "perf-r$repeat" \
    --output-path ".benchmarks/results/perf-r$repeat"
done
```

Each run creates fresh stores. Query embeddings warm up before measurement; generation does not.
Report variability and actual warmup counts. Leave response caching and resume off. Reserve remote
services for publishable measurements and record their hardware separately from the client.

Task/arm `performance` distributions report count, average, p50, p95, and p99. Throughput uses
successful work divided by the union of successful active intervals. Failed attempts stay in error
and attempt-latency counts. Missing observations are `null`, with coverage/completeness fields.

| Block | Measurement boundary |
| --- | --- |
| `ingest` | Accepted items, searchable SDK batch latency, amortized work, and throughput; commit/apply is included, batched flush/outbox acknowledgement is outside the measured call |
| `answer` | Caller completion latency/TTFT, nested SDK latency, and model first-token/first-chunk times |
| `ask_retrieval_core` | Retrieval performed inside the answer |
| `search_e2e` | Standalone warm-store public-search replay after all formal answers; separate `diagnostic` usage |
| `capture`, `time_to_searchable_ms`, `formation` | Capture acknowledgement, commit-to-searchable time, and settlement work; populated only in capture mode |
| `asr` | Successful inference/audio durations, real-time factor, speedup, latency, and failures |
| `compile` | Compile latency and bundle character/item/media counts |
| `recall` | Plan, shape, fallback, replan, incomplete/non-selective counts, exhaustive rows, and planner latency |
| `nodes` | Operation, stage, and model spans with identities, timing, status, and attribution |
| `token_usage` | Measured total and per-module tokens, request counts, modalities, and completeness |

Caller TTFT includes harness admission waiting and ends at the first non-empty text delta.
Generation first-chunk time can include an empty chunk. Per-sample `latency_ms` starts after
admission. Buffered answers have no caller TTFT. Search replay is not in-answer retrieval; failed
or missing planned replays make its distribution incomplete when replay is enabled.

ASR real-time factor is successful compute seconds / audio seconds; lower is better. Missing
durations make the full ratio unavailable. Token totals are never estimated: `reported_*` fields
are lower bounds when components are incomplete, and replay usage stays under `diagnostic`.
`token_usage.product` excludes judge usage. Bundle characters are not provider tokens.

`resources` covers client process CPU, lifetime resident-memory high-water mark, storage growth,
and sampled local GPU readings. GPU readings are device-wide and non-exclusive. Optional
`benchmark.server_metrics` URLs provide process-global vLLM deltas, excluding judging but including
search replay. Shared traffic may contaminate them; absent telemetry is marked unavailable.

Add performance limits to a compatible `--compare` run:

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

Fractions are maximum relative regressions; latency/tokens are lower-is-better, throughput is
higher-is-better. Results appear in `performance_comparisons`. These budgets require matching
schema/runner, input digests, models/composition, concurrency, warmup, runtimes, and hardware;
all baseline tasks must remain present. Cached, resumed, errored, or incomplete relevant
measurements are refused. The same mapping can live under `benchmark.performance_budgets`.

## Cache and resume

`--use-cache .benchmarks/response-cache` reuses answers and judge scores.
`--resume` reuses committed ingestion from the same explicit `--run-id`. They recover different
work; resume alone still regenerates answers. Enable response caching on the original run, then
rerun the same command with `--resume` after interruption.

Checkpoints are always written after committed batches. A unit rebuilds if its input/configuration
identity changed, its store vanished, or its state passed the earliest unanswered causal cutoff.
Rebuilding respects the live-owner lock. Write-failure counts survive resume; interruption may
duplicate at most one uncheckpointed batch.

Existing final outputs are protected unless `--overwrite` is given. Resume accepts a crash copy
but refuses final `results.jsonl`/`samples.jsonl`. Cache namespaces include question inputs,
model/configuration identities, arm budgets, answer policy, and deliberation. Cached and resumed
runs cannot supply performance baselines.

## Export AMA-Bench submissions

Complete a judged `ama-bench` run with the official `Qwen/Qwen3-32B` judge and `--limit all`:

```bash
uv run --frozen sensemem-bench ama-bench export \
  --dataset .benchmarks/ama-bench/test/open_end_qa_set.jsonl \
  --samples .benchmarks/results/ama-bench/samples.jsonl \
  --output .benchmarks/results/ama-bench/submission.jsonl
```

Export requires pinned inputs/scoring and one successful product prediction per question; partial,
duplicate, failed, mismatched, and predict-only results are rejected. Rows contain `episode_id`,
`question_uuid_list`, `answer_list`, and `llm_as_judge_score_list`. Result `official_evaluation`
and `leaderboard` blocks retain evaluator and weighted leaderboard aggregates; token F1 is separate.
Submit the local file through the
[official leaderboard](https://huggingface.co/spaces/AMA-bench/AMA-bench-Leaderboard).

## Export raw LoCoMo predictions

Use this utility when an external evaluator needs raw predictions instead of integrated scores:

```bash
uv run --frozen sensemem-bench locomo-refined \
  --config eval.yaml \
  --dataset .benchmarks/locomo-refined/data/raw/locomo_refined.json \
  --output .benchmarks/results/locomo-raw/predictions.jsonl \
  --data-root .benchmarks/data --run-id locomo-raw-001 --limit 1
```

The sibling `.manifest.jsonl` records digests, model identities, counts, platform, and isolated
store paths. Existing artifacts need `--overwrite`; existing run stores need `--resume`.

## Rejudge ICM predictions

Rejudge logged answers without opening models for ingestion or regenerating predictions:

```bash
uv run --frozen python -m sensemem.benchmarks.rejudge_icm \
  --predictions icm-samples.jsonl --annotations icm-questions.jsonl \
  --output icm-rejudged.json --model gpt-4o-mini
```

Inputs must contain only current-schema ICM rows with original questions and references retained
by `--log-samples`. Annotations join by ID and must match both text and references. Output retains
prediction hashes, prior scores/protocols, new judge identity, and input hashes; existing output
is protected. Judge failures are unscored, empty predictions score zero, and failed generation or
ingestion is skipped. Report all denominators and compare within the same judge protocol.

The command uses `OPENAI_API_KEY`; override with `--api-key-env` and `--base-url` as needed.
Only judge calls are made. A non-official judge is labelled as a proxy; rejudging cannot change
the input protocol the reader originally received.
