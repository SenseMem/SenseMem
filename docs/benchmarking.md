# Benchmarking

Measure how well your agent uses memory with `sensemem-bench eval`. The harness calls the public
SDK on isolated stores and reports answer quality, retrieval diagnostics, latency, and resource
use. Baseline arms and paired comparisons help you choose models and evaluate changes.

Start with a bounded run, then evaluate the complete dataset under the same conditions.
For existing measurements, see [benchmark results](benchmarks/results.md).

## 1. Install and choose a task

From the repository checkout:

```bash
uv sync --locked --default-index https://pypi.org/simple \
  --extra benchmarks --extra local --extra openai
uv run --frozen sensemem-bench eval --list-tasks
```

The listing shows task names, groups, source revisions, and local-data readiness. Use the
[task guide](benchmarks/tasks.md) to choose a dataset and check its model and media requirements.
Review the upstream dataset terms before downloading; they are separate from SenseMem's license.

Each independent memory unit gets its own physical store. Keep benchmark data separate from an
application's live `data_dir`.

## 2. Configure the models

Save a configuration as `eval.yaml`. This example uses a local embedding endpoint and separate
OpenAI-compatible reader and judge endpoints; replace the URLs, model names, and vector dimension
with those your services provide:

```yaml
embedding:
  provider: openai
  base_url: http://127.0.0.1:8000/v1
  api_key: local-server-ignores-this
  model: your-embedding-model
  dimension: 1024
generation:
  provider: openai
  base_url: https://reader.example/v1
  model: your-reader-model
benchmark:
  judge:
    base_url: https://judge.example/v1
    model: qwen3-14b
```

Set `SENSEMEM_GENERATION_API_KEY` for the reader and `SENSEMEM_JUDGE_API_KEY` for the judge.
The reader key falls back to `OPENAI_API_KEY`; an unset judge key reuses the reader key.
`qwen3-14b` is the required LoCoMo-Refined judge. Other tasks may require a different judge.

The `sensemem` arm requires an explicit `embedding` section. The runner replaces `data_dir` with
isolated benchmark directories. Settings resolve as CLI flag → file → environment → default.
See the [annotated configuration](examples/eval.example.yaml) for all fields and the
[configuration reference](configuration.md) for model capabilities and provider settings.

## 3. Run a bounded evaluation

```bash
uv run --frozen sensemem-bench eval \
  --config eval.yaml \
  --tasks locomo-refined \
  --arms sensemem,blind \
  --limit 1 \
  --seed 42 \
  --run-id locomo-smoke \
  --output-path .benchmarks/results/locomo-smoke
```

`sensemem` answers using memory; `blind` answers without memory and measures the same reader's
prior knowledge. Both use the same selected questions and scorer.

`--limit 1` selects one task-specific unit. For LoCoMo-Refined, that is one conversation and
all its questions. Report the emitted `question_count` to show the evaluated population.
`--limit` accepts an integer count, a fraction between zero and one, or `all`.

Missing supported data downloads automatically. Video preparation needs `ffmpeg` and `ffprobe`;
M3-Bench web videos also need `yt-dlp`. To use existing data, pass `--no-download`. The
[data reference](benchmarks/reference.md#data-and-isolation) covers integrity checks and overrides.

Once the bounded run works, evaluate the complete dataset in a new run:

```bash
uv run --frozen sensemem-bench eval \
  --config eval.yaml \
  --tasks locomo-refined \
  --arms sensemem,blind \
  --limit all \
  --seed 42 \
  --run-id locomo-baseline \
  --output-path .benchmarks/results/locomo-baseline
```

Progress appears on stderr. By default, each task is judged and reported when its answers finish.
For interrupted runs, see [cache and resume](benchmarks/reference.md#cache-and-resume).

## 4. Read the results

Each completed output directory contains:

| File | Contents |
| --- | --- |
| `results.jsonl` | One run record containing task/arm scores, confidence intervals, controls, performance, and resource use |
| `samples.jsonl` | One row per question per arm, with prediction, metrics, evidence diagnostics, and errors |
| `config.yaml` | Resolved settings with credentials omitted; a run manifest, not a reusable input configuration |

Check these fields before quoting a task score:

| Field | What to check |
| --- | --- |
| `primary_metric`, `score.mean` | Name the metric and its scale; scales differ across tasks |
| `score_valid`, `dataset_coverage` | Confirm the selected inputs and measured run are valid |
| `question_count`, `score.cluster_count` | Report questions and independent memory units |
| `score.confidence_interval_95` | Report uncertainty; it is `null` with fewer than two independent units |
| `official_metric`, `scorer_protocol`, `judge_model` | Check that the metric uses the required scorer and judge |
| `controls` | Read the blind score and, where gold source labels exist, recall and random-ranker controls |

Report scores with their recorded input and scoring protocols. Results record
`cross_harness_comparable: false`; use matching protocols for comparisons with another harness.
Use full, verified inputs for a full-dataset score.

Retrieval controls are unavailable for datasets without compatible gold source IDs. Such runs
report missing controls explicitly; use them for answer evaluation without claiming retrieval
quality. See [controls and diagnostics](benchmarks/reference.md#controls-and-diagnostics).

`--log-samples` additionally retains original questions, references, prompts, and judge responses.
Artifacts can contain licensed or private content; keep raw samples local unless their terms
permit sharing.

## 5. Compare a change

Run the candidate against the baseline using the same task selection, seed, scorer, and judge:

```bash
uv run --frozen sensemem-bench eval \
  --config eval.yaml \
  --tasks locomo-refined \
  --arms sensemem,blind \
  --limit all \
  --seed 42 \
  --compare .benchmarks/results/locomo-baseline \
  --fail-on-regression \
  --regression-threshold 0.01 \
  --run-id locomo-candidate \
  --output-path .benchmarks/results/locomo-candidate
```

`--compare` pairs SenseMem predictions by stable sample ID and rejects incompatible inputs,
scorers, or judges. For fractional accuracy, `0.01` is one percentage point. With
`--fail-on-regression`, answer or ingest failures also cause a nonzero exit.

Use `--arms` to compare memory strategies; use `--compare` to check a change against an earlier
SenseMem run. Report paired confidence intervals and the recorded noise floor. For performance
claims, use at least three fresh runs with identical services and hardware, without response
caching or resume; see [performance measurements](benchmarks/reference.md#performance-measurements).

## Reference

| Need | Guide |
| --- | --- |
| Choose a dataset, judge, or input protocol | [Tasks and protocol notes](benchmarks/tasks.md) |
| Tune a run, interpret diagnostics, resume, or export | [Evaluation reference](benchmarks/reference.md) |
| Measure storage, SDK concurrency, or consolidation | [Local measurements](benchmarks/local.md) |
| Host the participant Add/Search API | [Agent Memory Leaderboard adapter](agent-memory-leaderboard.md) |
| Check reader behavior with a small real-model suite | [Model quality gate](model-quality.md) |
