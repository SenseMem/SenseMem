# Benchmark tasks

Choose a workload that matches your agent: conversation history, personalization, personal
media, or long video memory. The tables list selectors, primary metrics, and required judges.

Follow [Benchmarking](../benchmarking.md) for setup and execution. Use
`uv run --frozen sensemem-bench eval --list-tasks` for exact variants and local-data readiness.
Group selectors expand to several tasks.

## Choose a workload

The judge column names the model required by the pinned scoring protocol. A different judge is
allowed but marks the affected metric non-official. Deterministic metrics need no judge.
Baseline arms are always non-official.

### Dialogue and agent histories

| Selector or group | Workload | Primary metric | Judge |
| --- | --- | --- | --- |
| `ama-bench` | Agent action/observation histories | `accuracy` | `Qwen/Qwen3-32B` |
| `locomo-refined` | Multi-session dialogue with published image captions | `llm_judge` | `qwen3-14b` |
| `memlens` | 32K–256K dialogue, images, temporal reasoning, updates, and refusal | `accuracy` | `qwen3-235b-judge` |
| `longmemeval-v2` | Web and enterprise trajectories with state screenshots | `accuracy` | Phrase/choice rules; `gpt-5.2` for judged types |
| `mosaic` | Text, image, audio, and video conversation histories, including refusal | Answerable-only `judge_score` (0–1) | `qwen/qwen3.7-plus` |
| `personamem-v3` | Cross-app personalization, preference changes, repetition, and proactive actions | `accuracy_pct_micro` (0–100), with per-family scores | `gpt-5.5` for judged families |

### Personal media archives

| Selector or group | Workload | Primary metric | Judge |
| --- | --- | --- | --- |
| `atm-bench` | Email, images, and video; raw and processed-text SGM variants | `accuracy` | `gpt-5-mini` for open-ended rows; deterministic number/list scoring |
| `mem-gallery` | Image-grounded personal dialogue | `f1` | Deterministic; optional `llm_judge` uses `qwen2.5-72b-instruct` |

### Video histories

| Selector or group | Workload | Primary metric | Judge |
| --- | --- | --- | --- |
| `egotempo` | Temporal QA over Ego4D clips | `accuracy` | `gemini-1.5-flash` |
| `mm-lifelong` | Day-to-month video memory and temporal grounding | `answer_accuracy` | `gpt-5` |
| `supermemory-vqa` | Causal multi-video QA, including unanswerable cases | `qa_accuracy` | Deterministic choices |
| `m3-bench` | Robot and web-video histories | `accuracy` | `gpt-4o-2024-11-20` |
| `icm-bench` | Incremental video recall, retrieval, and profiles | `accuracy` | `gemini-3-flash-preview` |
| `videozerobench` | Five levels of QA and temporal/spatial localization | Level-3 `accuracy` | Deterministic QA and IoU |

## Data requirements

Supported annotations and media download automatically, except for authorized Ego4D media.
Video tasks require `ffmpeg` and `ffprobe`; M3 web also requires `yt-dlp`. Start with one task:
the `all` group can require hundreds of gigabytes.

LoCoMo-Refined, MemLens, LongMemEval-V2, and PersonaMem-v3 need no timed-media
preparation. Image-bearing tasks still require compatible embedding and reader
models. Raw ATM variants need a `vision` backend to index media descriptions; without it, their
indexed text contains only source IDs.

Review each upstream license before download. In particular, LoCoMo-Refined and PersonaMem-v3
are CC BY-NC 4.0; MM-Lifelong restricts use to academic purposes; VideoZeroBench is
CC BY-NC-ND 4.0; ICM data is CC BY-NC-SA 4.0. See the
[scorer notices](../../src/sensemem/benchmarks/_official/NOTICE.md) for copied code and prompts.

## Protocol notes

Use these notes to choose the right input and scoring condition. The task catalog and generated
manifests retain exact source and protocol identities.

### Dialogue tasks

- **LoCoMo-Refined:** `--limit` counts conversations and keeps all their questions. Published
  captions are ingested as text. The release excludes original LoCoMo adversarial category 5.
- **MemLens:** `--limit` counts questions. Released images contain answers absent from captions;
  an empty `--media-root` deliberately selects a caption-only condition and must be reported.
  Answer-session labels form groups: retrieving any turn in an answer session satisfies that group.

### LongMemEval-V2

Select `longmemeval-v2-small` or `longmemeval-v2-medium`; the group selects both. `--limit` and
`--offset` select questions before grouping identical ordered haystacks into shared stores.
Trajectories retain goals, outcomes, actions, thoughts, accessibility trees, and screenshots.
Gold answers and evaluator specifications remain scorer-side; no retrieval labels are supplied.

Even a one-question run needs its complete haystack and domain screenshot archive. Both domains
total about 5.9 GB of archives plus a 1.2 GB trajectory JSONL. Offline layouts retain
`screenshots/<trajectory_id>/<step>.png` and `question_screenshots/`; extracted screenshots may
replace archives. `--media-root TASK=/path/to/release` overrides this media root. Missing
referenced screenshots fail the run.

Scoring reproduces the six released evaluator functions. The last `\boxed{...}` is extracted;
exact `UNKNOWN` scores zero. Flaw-awareness and gotcha judges use `gpt-5.2`, medium reasoning,
and a 4,096-token completion limit. This harness does not calculate leaderboard LAFS gains.

### MOSAIC

The native multimodal task covers 923 questions across 70 persona histories. Its headline is
answerable-only `judge_score`; refusal rates use separate answerable and unanswerable populations.
See the [EmbeddingGemma 2 / Qwen Omni recipe](mosaic-native.md) for pinned inputs, native chunking,
model settings, complete run commands, and measured results.

### PersonaMem-v3

Only events strictly before the question timestamp enter memory. `profile.json` is scorer-only.
`--limit` and `--offset` select personas before their engagement logs are parsed. Calendar
additions, updates, and removals retain the same entry ID so memory can link their changes.
The product arm uses `Memory.compile` followed by a harness assistant prompt that permits general
knowledge while requiring support for personal facts. Its `answer_policy` is `null`; it exposes
no ranked retrieval list. This differs from the grounded-reader `compile` baseline.

Scored families use the released personalization rubric, four task-specific judges, and ranking
formula recorded in the task catalog. Ranking reports `ndcg_at_5`: positive gain +2, filler +1, hard
negative −2; historical hidden-persona slates without hard negatives use filler gain zero.
Ranking questions retain any published prior conversation.

Local recommendation after a location change uses the official deterministic
`geo_shift_correctness`: current city or region earns 1, neither current nor previous location
earns 0.5, and mentioning the previous city earns 0. Responses with fewer than 15 distinct
ASCII tokens of at least three characters earn 0.

The five proactive-action families request the official JSON decision (`should_act`,
`action_class`, `content`, `evidence_cited`, `reasoning`). These are simulated chat decisions;
the harness executes no actions. Generation sees causal memory and an identical instruction
for act and restraint cases. Hidden trigger evidence, JITAI cards, and instance polarity go
only to the `gpt-5.5` judge. The headline `proactive_action_score` combines 70% deterministic
decision correctness with 30% judged justification, including the official hard-failure and
message-length rules. Without a judge, only deterministic diagnostics are reported.

After [configuring the embedding, reader, and judge](../benchmarking.md#2-configure-the-models),
use `benchmark.judge.model: gpt-5.5` and start with one persona:

```bash
uv run --frozen sensemem-bench eval \
  --config eval.yaml --tasks personamem-v3 --arms sensemem,blind --limit 1
```

Novel suggestions use the official JSON recommendation prompt, deterministic fatigue and
recent-hashtag checks, and the 0–3 alignment judge. A fatigued-topic overlap or dominant reuse
of recent hashtags fails the row; otherwise an alignment score of at least 2 passes.
Unprompted moments retain their empty user query without disclosing the fatigued topic or gold.

Both repetition families execute the complete released sequence, including intervening filler
turns. Each response enters the next prompt. Generation and judging run once per cluster and arm;
published sibling rows reuse those results. As in the pinned runner, the first selected row's
memory snapshot is shared by the cluster. Only target turns are judged. The first
`n_allowed_repetitions + 1` target turns are warmup; the headline is mean tail restraint on a
0–10 scale. Empty or non-substantive responses score zero. Failed judgments retain their original
positions and are excluded from the tail mean, with the failure reported.

Lifecycle ranking uses the released location and calendar context without revealing phase,
stop condition, or matching labels. The runner pairs pre/post answers by persona and canonical
ID, computes `lifecycle_score = pre.match_rate_at_3 - post.match_rate_at_3`, and maps its signed
range to 0–100, with zero delta equal to 50. Each pair contributes once to the micro score;
both original answer rows retain the paired result.

Mistake prevention keeps warn/foil polarity and expected warning terms scorer-side. The official
warning detector, mention-coverage threshold, leak check, and substantive-foil gate produce each
row's `correct`. The pinned aggregator uses these per-row values in the micro score; the
`mistake_prevention` summary separately reports complete-pair correctness, warn recall, foil
precision, and paired F1, including counts of incomplete pairs.

`accuracy_pct_micro` is available when every eligible selected scoring unit has its required
metric. Unpaired lifecycle phases remain in the answer artifacts and are explicitly excluded
from scoring, following the upstream aggregator's missing-metric rule. The pinned release has
349 complete lifecycle pairs and 169 unpaired phases; `score_coverage` reports excluded counts
and reasons. Missing judges or unscored eligible rows withhold the headline and report coverage
plus supported-subset results. Use the recorded adapter and scorer versions when comparing runs;
this protocol uses adapter `personamem_v3_official_v3`, scorer
`personamem_v3_ad80a3b1b322_v3`, and result schema 25. Earlier artifacts use different populations
and protocols.

### ATM-Bench and Mem-Gallery

ATM uses one official Oracle instruction for every question type, including `Unknown` refusal.
`number` questions can request dates, times, ranges, units, or currencies. Answer scoring follows
the pinned release without repairing its references or ID-list edge cases. The strict judge
schema requests boolean `accuracy`; official conversion also accepts the string `"true"`.
Retrieval and joint metrics are separate SenseMem diagnostics.

To add released caption, OCR, location, and tag annotations to raw ATM media:

```bash
uv run --frozen python benchmarks/atm_source_manifest.py \
  --inputs .benchmarks \
  --output .benchmarks/atm-source-annotations.json
uv run --frozen sensemem-bench eval \
  --config eval.yaml --tasks atm-bench-main \
  --media-manifest .benchmarks/atm-source-annotations.json \
  --data-root .benchmarks/atm-source-run --no-download
```

The builder needs a complete release, keeps raw assets, and reads no QA labels. Report this as
raw media plus released source annotations, separately from raw-only and SGM conditions.
Use a multimodal configuration and a fresh data root.

Mem-Gallery's query places the original question before its official category constraint.
Compare matching adapter/query versions. Its official retrieval formulas at K=10 differ from
generic labelled-source coverage; empty-clue rows score zero.

### ICM-Bench

Input is raw video plus speakerless ASR, including calibration. All 829 released ASR files are
required; only the ten clips without published transcripts may omit them. Recall/Retrieval
questions see `before_clip` inclusively; Profile sees the full album. Clip positions are causal
coordinates, not dates or video offsets. Character IDs, speaker labels, gold answers, and evidence
annotations stay outside model input. Clip evidence labels may be incomplete.

The scorer uses the pinned user-only semantic-equivalence prompt and exact `Yes` parser,
protocol `icm_semantic_equivalence_10f02babe3c7_user_only_v2`. Compare runs with matching protocols.
Report effective modalities and identity settings: frame conversion does not provide native
audio or face/voice binding.
After extraction, the downloaded tar may be removed for offline runs; clips and annotations must
remain available.

### VideoZeroBench

`--limit` and `--offset` select original questions before expanding their applicable levels.
`question_count` counts expanded requests; the headline and category breakdowns use only Level-3.

| Level | Request input | Score |
| --- | --- | --- |
| 1 | Question plus gold temporal intervals and boxes | Exact-answer rules |
| 2 | Question plus gold temporal intervals | Exact-answer rules |
| 3 | Question without evidence hints | Headline `accuracy` (0–1) |
| 4 | Temporal-localization instruction without gold intervals | Union tIoU; joint score requires correct Level-3 and tIoU > 0.3 |
| 5 | Gold key times, spatial instruction, no boxes | Mean union-box vIoU; joint score also requires vIoU > 0.3 |

Named `Level-*` metrics use 0–100. Joint scores divide by all selected original questions; mean
IoUs use only applicable evidence. Malformed predictions score zero; failed requests remain in
their denominators. Coordinates are predicted in [0,1000] and converted to [0,1]. All seven named
metrics receive paired comparisons in percentage points; unavailable metrics retain their reason.

The video archive is about 9.7 GB and extracts to `videozerobench/compressed/`. For existing
data, use `--task-data videozerobench=/path/to/VideoZeroBench_500_v0.json` and
`--media-root videozerobench=/path/to/compressed`. Models must support video or suitable
conversions. Spatial localization needs visual input. This memory/retrieval route differs from
the upstream frame sampler and priority-keyframe injection, even with identical scoring formulas.

### Other video tasks

- **M3 web:** permanently unavailable YouTube videos are listed in the media manifest. Remaining
  samples run, but incomplete coverage makes `score_valid` and full-dataset comparability false.
  Network/authentication failures stop acquisition. An operator-supplied copy can restore coverage.
- **SuperMemory-VQA:** `qa_mrr` is unavailable because the backend exposes no option scores.
- **MM-Lifelong:** the default group includes day/week/month validation; select month-train
  explicitly. Temporal `ref_at_300` measures intervals, separately from source-ID recall.
