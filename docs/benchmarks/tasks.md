# Benchmark tasks

Choose a workload that matches your agent: conversation history, personalization, personal media, or long video memory. The tables list selectors, primary metrics, and required judges.

Follow [Benchmarking](../benchmarking.md) for setup and execution. Use `uv run --frozen sensemem-bench eval --list-tasks` for exact variants and local-data readiness. Group selectors expand to several tasks.

## Choose a workload

The judge column names the model required by the pinned scoring protocol. A different judge is allowed but marks the affected metric non-official. Deterministic metrics need no judge, and baseline arms are always non-official.

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

Supported annotations and media download automatically, except for authorized Ego4D media. Video tasks require `ffmpeg` and `ffprobe`; M3 web also requires `yt-dlp`. Start with one task: the `all` group can require hundreds of gigabytes.

LoCoMo-Refined, MemLens, LongMemEval-V2, and PersonaMem-v3 need no timed-media preparation. Image-bearing tasks still require compatible embedding and reader models. Raw ATM variants need a `vision` backend to index media descriptions; without it, indexed text contains only source IDs.

Review each upstream license before download (LoCoMo-Refined and PersonaMem-v3 are CC BY-NC 4.0; MM-Lifelong is academic-use only; VideoZeroBench is CC BY-NC-ND 4.0; ICM data is CC BY-NC-SA 4.0). See the [scorer notices](../../src/sensemem/benchmarks/_official/NOTICE.md) for upstream attributions.

## Protocol notes

### Dialogue tasks

- **LoCoMo-Refined:** `--limit` counts conversations and keeps all their questions. Published captions are ingested as text.
- **MemLens:** `--limit` counts questions. Released images contain answers absent from captions; passing an empty `--media-root` selects a caption-only condition. Answer-session labels form groups where retrieving any turn in the session satisfies the group.

### LongMemEval-V2

Select `longmemeval-v2-small` or `longmemeval-v2-medium` (`longmemeval-v2` selects both). `--limit` and `--offset` select questions before grouping identical ordered haystacks into shared stores. Trajectories include goals, outcomes, actions, thoughts, accessibility trees, and screenshots (~5.9 GB screenshot archives plus 1.2 GB trajectory JSONL). Use `--media-root TASK=/path/to/release` for offline layouts (`screenshots/<trajectory_id>/<step>.png` and `question_screenshots/`). Scoring extracts the last `\boxed{...}` and uses `gpt-5.2` (medium reasoning, 4,096 completion tokens) for judged types.

### MOSAIC

MOSAIC covers 923 questions (784 answerable, 139 unanswerable) across 70 multimodal persona histories containing text, images, audio, and video. Each persona gets its own isolated physical store; text chunks split losslessly at 16,000 characters while preserving source IDs. The headline metric is answerable-only `judge_score`, with separate correct-refusal and false-refusal rates. Configure multimodal embedding and generation endpoints (such as [EmbeddingGemma 2 through vLLM](../configuration/models.md#embeddinggemma-2-through-vllm)) and run with `--tasks mosaic`.

### PersonaMem-v3

Only events strictly before the question timestamp enter memory; `profile.json` is scorer-only. `--limit` and `--offset` select personas before engagement logs are parsed. Calendar updates retain stable entry IDs. The `sensemem` arm compiles context via `Memory.compile` and passes it to a harness assistant prompt.

Scored families cover personalization rubrics, ranking (`ndcg_at_5`), geographic-shift local recommendations (`geo_shift_correctness`), proactive action decisions (`proactive_action_score`: 70% deterministic decision + 30% `gpt-5.5` justification), novel suggestions, multi-turn repetition clusters, paired pre/post lifecycle ranking, and mistake-prevention warn/foil pairs. Unpaired lifecycle phases are retained in artifacts and excluded from `accuracy_pct_micro`, which is reported only when all eligible selected units have complete scoring coverage.

After [configuring the embedding, reader, and judge](../benchmarking.md#2-configure-the-models) with `benchmark.judge.model: gpt-5.5`, start with one persona:

```bash
uv run --frozen sensemem-bench eval \
  --config eval.yaml --tasks personamem-v3 --arms sensemem,blind --limit 1
```

### ATM-Bench and Mem-Gallery

ATM uses the official Oracle instruction for every question type (including `Unknown` refusal) and strict boolean `accuracy` judging for open-ended questions. To combine raw ATM media with released caption, OCR, location, and tag annotations:

```bash
uv run --frozen python benchmarks/atm_source_manifest.py \
  --inputs .benchmarks \
  --output .benchmarks/atm-source-annotations.json
uv run --frozen sensemem-bench eval \
  --config eval.yaml --tasks atm-bench-main \
  --media-manifest .benchmarks/atm-source-annotations.json \
  --data-root .benchmarks/atm-source-run --no-download
```

Mem-Gallery places the original question before its official category constraint and reports token F1 plus official retrieval metrics at K=10.

### ICM-Bench

Input is raw video plus speakerless ASR (all 829 released ASR files are required; only the ten clips without published transcripts may omit them). Recall and Retrieval questions see `before_clip` inclusively; Profile sees the full album. Scoring uses the pinned user-only semantic-equivalence prompt and exact `Yes` parser.

### VideoZeroBench

`--limit` and `--offset` select original questions before expanding applicable levels; the headline metric uses Level-3 `accuracy`:

| Level | Request input | Score |
| --- | --- | --- |
| 1 | Question plus gold temporal intervals and boxes | Exact-answer rules |
| 2 | Question plus gold temporal intervals | Exact-answer rules |
| 3 | Question without evidence hints | Headline `accuracy` (0–1) |
| 4 | Temporal-localization instruction without gold intervals | Union tIoU (joint requires Level-3 correct and tIoU > 0.3) |
| 5 | Gold key times, spatial instruction, no boxes | Mean union-box vIoU (joint requires vIoU > 0.3) |

The video archive (~9.7 GB) extracts to `videozerobench/compressed/`. For existing data, pass `--task-data videozerobench=/path/to/VideoZeroBench_500_v0.json` and `--media-root videozerobench=/path/to/compressed`.

### Other video tasks

- **M3 web:** permanently unavailable YouTube videos are listed in the media manifest; incomplete coverage sets `score_valid` to `false`.
- **SuperMemory-VQA:** `qa_mrr` is unavailable because the backend exposes no per-option scores.
- **MM-Lifelong:** the default group includes day/week/month validation (`month-train` is selected explicitly); `ref_at_300` measures temporal intervals separately from source-ID recall.
