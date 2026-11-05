# Provider fields

Return to [Configuration](../configuration.md) for composition and ownership.
This page owns OpenAI-compatible connection, completion, routing, and transport controls.
[Model configuration](models.md) covers embedding, reranking, speech, and visual-description recipes.

## Connections and response format

OpenAI slots use `base_url`, `api_key`, `timeout`, and `max_retries`, each defaulting to
`None` and delegating to SDK defaults/environment lookup. Each slot constructs a separate client.
An unset key uses standard lookup including `OPENAI_API_KEY`; explicit keys are masked in
representations but remain secrets in their input files.

Use an absolute HTTP(S) SDK base URL ending in `/v1`. The SDK appends operation paths.
`response_format: json_schema` is the default for structural calls: formation, consolidation,
descriptions, diagnosis, and recall planning. Select `json_object` explicitly for a compatible
server lacking strict schema support. JSON syntax still requires local structural/semantic checks.
Answers remain text unless structured answering is enabled.

Every chat request sends `store: false` by default. `store: true` opts into provider application
state storage; direct adapters use `generation_store`. This does not disable abuse-monitoring
retention or establish Zero Data Retention; see the provider's
[data controls](https://developers.openai.com/api/docs/guides/your-data). Local deletion cannot erase
provider copies. REST/MCP inbound authentication belongs to the host.

## Embedding fields

| Field | OpenAI default | Meaning |
| --- | --- | --- |
| `model` | `text-embedding-3-small` | Deployed model |
| `dimension` | `1536` | Required response width |
| `space` | `None` | Explicit identity override; must represent the actual recipe |
| `modalities` | `[text]` | At least one atomic modality |
| `request_format` | `input` | Embeddings array or chat-style `messages`; part of space identity |
| `image_max_pixels` | `None` | Optional resizing policy; part of embedding identity |

## Completion fields

These OpenAI slots share the following defaults unless noted:

| Field | Default |
| --- | --- |
| `model` | `gpt-5-mini-2025-08-07` |
| `modalities` | `[text]`; vision defaults to `[image]` and allows image/video only |
| `temperature`, `seed`, `max_tokens` | `None` |
| `response_format` | `json_schema` |
| `store`, `stream` | `False` |
| `extra_body` | `None` |
| `image_max_pixels` | `None` for generation, formation, and vision |

`consolidation` uses the shared completion fields without an image-resizing option.
Vision adds `video_frames=4` (2–64). Generation adds:

| Field | Default | Use |
| --- | --- | --- |
| `answer_style` | `default` | Standard reader or answer-only `concise` |
| `answer_format` | `text` | Text or validated `structured` draft |
| `source_metadata_key` | `source_id` | Application identifier field for structured source lists |
| `request_max_bytes` | `None` | Optional complete answer-request JSON byte bound |
| `video_limit` | `8` | Retrieved-video cap; `None` disables this count |
| `min_video_seconds` | `None` | Convert shorter video to four ordered stills; needs image support |
| `media_policy` | `all` | Attach eligible media, or select `on_demand` |
| `media_max_items` | `2` | On-demand selection cap, 1–8 |
| `media_max_tokens` | `8192` | Positive output-token ceiling for media diagnosis |
| `routes` | Empty | Specialized generation models |
| `offline_gazetteer` | `None` | Pinned, caller-supplied local map index |

Modalities are atomic `text`, `image`, `video`, and `audio`. Positive fields must exceed zero;
retry counts are non-negative; temperature is [0,2] and seed is [0,`2**63 - 1`].
`extra_body` cannot replace model, messages, response format, sampling, token limits, streaming,
or storage controls.

### Sampling and streaming

Known reasoning models omit incompatible automatic temperature/seed controls; explicit unsupported
values raise `ValidationError` before inference. `max_tokens` maps to `max_completion_tokens`
for reasoning models and includes reasoning tokens. Reserve enough budget for visible output.

The adapter recognizes GPT-5/GPT-6 and o1/o3/o4 reasoning families. GPT-5 Chat and unrelated
compatible names use generic controls. Supported later models can use temperature with
`reasoning_effort: none`; original GPT-5, GPT-6 Astra, and GPT-6.1 Sol cannot. GPT-5.1/5.2 aliases
and snapshots default to none. Known reasoning models reject seeds.

Slot `stream: true` collects provider SSE before parsing and requires the terminal usage chunk.
It can help gateways that time out on buffered JSON. This transport setting is independent of
the public `ask_stream()` surface and benchmark reporting/answer-surface controls.

### Answer presentation

`answer_style: concise` asks for requested values/entities without explanatory prose or citations,
while preserving refusal and evidence policy. The model checks roles, dates, units, and list
conditions; these are prompt instructions, not deterministic validators. Evaluate the style for
the chosen model/task. Formation, vision, and consolidation do not accept it.

`answer_format: structured` validates a typed draft against final fitted evidence, then renders
it. Source lists resolve local aliases through `source_metadata_key`; supply that metadata when
application IDs must be returned. Missing IDs or invalid drafts raise `ModelError`.

Date/calculation drafts need timestamp or exact-quote witnesses. Supported operations are scalar,
sum, difference, distinct count, and date-interval union. Dates retain `source_value` plus
granularity/format: `iso8601` preserves timezone; `utc_datetime` converts an explicitly zoned
timestamp to UTC without a suffix. Numeric/duration drafts select with-unit or number presentation.
These checks do not establish relevance, real-event uniqueness, or set completeness. Other answers
use text drafts. Structured streaming buffers validation before yielding rendered text.
Returned hits are the declared support subset in kernel-selected order; use search for ranking.

## Generation routes

```yaml
generation:
  provider: openai
  model: text-model
  modalities: [text]
  routes:
    - provider: openai
      model: image-model
      modalities: [text, image]
    - provider: openai
      model: video-model
      modalities: [text, video]
```

Each route accepts the usual generation fields. Routing considers question and evidence together,
prefers native support, then the narrowest declared set, then declaration order. Nested routes
flatten in order. A failed call does not switch models. T+I and T+V routes do not imply T+I+V.
`Memory.capabilities.generation_routes` reports executable native combinations; source-backed
conversion is a separate path. Injected `EmbeddingRouter` routes must share space and dimension.

## Media selection and byte limits

`media_policy: on_demand` makes one text-only diagnosis call for eligible memory media. At most
64 candidate attachments are offered; up to `media_max_items` are selected for missing details or
conflicting descriptions. Questions, retained text, and provenance remain. Failed/invalid diagnosis
adds no media; provider rejection does not rerun diagnosis. This adds model requests/tokens.
Diagnosis uses the lower of `max_tokens` (when set) and `media_max_tokens`; a truncated response
selects no media and reports its diagnostic failure.
Direct composition uses `EvidenceGeneration(model, selector=model, media_max_items=2)`.
For direct adapters, `generation_media_max_tokens` sets the independent diagnosis ceiling;
`generation_max_tokens` remains the overall output ceiling.

`request_max_bytes` bounds the final answer JSON, including base64, UTF-8 text, provider fields,
and streaming fields, excluding headers/framing. Existing fitting is preserved when unset or the
body already fits. When over the bound:

1. Retain the question, mandatory question attachments, and fitted source text.
2. Admit retrieved media in retrieval order. Images may resize to JPEG; a video too large even
   alone may become 16 ordered stills when images are supported.
3. Declare lost detail/motion/audio/timing and omitted media. A native video that fits alone is
   omitted if earlier media consumed its room.
4. Fail locally with `payload_too_large` if mandatory text/media/metadata still cannot fit.

Source-bound GPS/capture time and enabled map context selected before the cap remain untrusted
metadata. Omitted assets carry `media_omitted=true` and source asset IDs without attachment aliases.
There is no HTTP 413 retry. This answer-only option changes neither stored assets nor recipes.
Input/output token limits remain separate; see [grounding settings](settings.md#evidence-budgets).

A provider technical-duration error retries once without that modality's evidence media and with
declared omissions. Safety refusals, including recognized gateway inspection errors, instead raise
`model_refused` without removing media; semantic error codes take priority over message matching.
Source modality describes original text/image/video/audio/omni; ordered stills remain video sources.
Per-attachment modality entries describe only assets actually supplied.

### Image resizing

`image_max_pixels` accepts 1–64,000,000. Headers reject sources above 64 million pixels without
changing Pillow's global limit. Images over the configured pixel or 20 MiB inline-item limit are
EXIF-oriented, converted to RGB JPEG quality 85/4:2:0, and proportionally resized with LANCZOS.
The pixel budget halves as needed until encoding fits; original bytes survive when already within
both limits. Request fitting counts actual base64 bytes and retains the 64 MiB aggregate limit.

Conversion caches only within a request. Stored bytes/IDs/digests stay unchanged. Undecodable or
oversized images fail embedding/formation/description; answering can omit retrieved media, while
mandatory question media must succeed. Integrity failures remain errors. The option joins embedding
space even with explicit `space`, and vision/formation recipe identity. Embedding changes need
fresh-store replay; derivation changes do not automatically backfill existing records.

## Offline map context

`generation.offline_gazetteer` needs `gazetteer` and `openai` extras plus an immutable local index:

```yaml
generation:
  provider: openai
  model: your-vision-model
  modalities: [text, image]
  offline_gazetteer:
    index_path: /absolute/path/named-geometries.sqlite3
    index_sha256: "<complete lowercase SHA256>"
    radius_m: 500
    limit: 8
    budget_chars: 16000
```

Path and complete SHA are required. Radius is positive and at most 500 metres, limit is 1–8,
and the serialized-character budget is positive. Only integrity-checked original image GPS is used.
Candidates are untrusted map context, not proof of photo subject, historical presence, or GPS accuracy.
Only retained attachments receive context, under one global whole-candidate budget.

The resource makes no network lookup and changes no place labels, records, embeddings, descriptions,
formation, or retrieval. The answer provider receives admitted context. Report this additional
knowledge source in comparisons. The [gazetteer API](../api/python-adapters.md#offline-gazetteer)
owns index formats, provenance, immutability checks, regional bounds, and geometry semantics.

## Local provider fields

| Slot/provider | Required | Defaults |
| --- | --- | --- |
| Embedding / Sentence Transformers | `model` | `revision=None`, `dimension=None`, `device=None`, `batch_size=32` |
| Reranking / Sentence Transformers | `model` | `revision=None`, `device=None`, `batch_size=32`, `max_length=1024` |
| Reranking / Qwen3-VL | `model` | `revision=None`, `device=None`, `batch_size=32`, `max_length=8192`, `gpu_memory_utilization=0.9`, `image_limit=8`, `video_limit=2`, `image_max_pixels=262144`, `video_frames=8` |
| Reranking / vLLM service | `model`, `base_url` | `api_key=None`, `timeout=60.0`, `batch_size=32`, `modalities=[text,image,video]`, `media_candidates=None`, `media_weight=0.5` |
| Speech / FunASR | None | `device=auto`, `ncpu=4`, `vad_device=None`, `speaker_device=None` |
| Speech / OpenAI | None | `model=whisper-1`, `space=None`, plus connection fields |
| Face / OpenCV | `detector_model`, `recognizer_model` | `score_threshold=0.9`, `nms_threshold=0.3`, `top_k=5000`, `frame_interval_ms=1000`, `max_video_frames=300` |

Face thresholds accept [0,1]. See [model setup](models.md) for loading, revisions, devices, and
modality behavior, and [identity thresholds](settings.md#identity-thresholds) for calibration.
