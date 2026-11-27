# Provider fields

Configure endpoints, completion controls, generation routes, and media delivery for bundled providers. [Configuration](../configuration.md) covers composition and ownership; [model setup](models.md) covers embedding, reranking, speech, and visual-description recipes.

## Connections and response format

OpenAI slots accept `base_url`, `api_key`, `timeout`, and `max_retries`, each defaulting to `None` and delegating to the official OpenAI SDK defaults and environment lookup (`OPENAI_API_KEY`). Use an absolute HTTP(S) SDK base URL ending in `/v1`.

`response_format` defaults to `json_schema` for structural calls (formation, consolidation, descriptions, diagnosis, and recall planning). Set `response_format: json_object` explicitly for compatible servers that do not support strict JSON Schema. Answers remain text unless `answer_format: structured` is set.

Every chat request sends `store: false` by default (`generation_store` on direct adapters). Opt into provider application-state storage with `store: true`; see the provider's [data controls](https://developers.openai.com/api/docs/guides/your-data).

## Embedding fields

| Field | OpenAI default | Meaning |
| --- | --- | --- |
| `model` | `text-embedding-3-small` | Deployed model |
| `dimension` | `1536` | Required response width |
| `space` | `None` | Explicit identity override matching the actual recipe |
| `modalities` | `[text]` | At least one atomic modality |
| `request_format` | `input` | Embeddings array or chat-style `messages` (part of space identity) |
| `image_max_pixels` | `None` | Optional resizing policy (part of embedding identity) |

## Completion fields

OpenAI completion slots (`generation`, `formation`, `consolidation`, `vision`) share these defaults unless noted:

| Field | Default |
| --- | --- |
| `model` | `gpt-5-mini-2025-08-07` |
| `modalities` | `[text]` (`vision` defaults to `[image]` and accepts only `image`/`video`) |
| `temperature`, `seed`, `max_tokens` | `None` |
| `response_format` | `json_schema` |
| `store`, `stream` | `False` |
| `extra_body` | `None` |
| `image_max_pixels` | `None` (for `generation`, `formation`, and `vision`) |

`vision` adds `video_frames=4` (`2`–`64`). `generation` adds:

| Field | Default | Use |
| --- | --- | --- |
| `answer_style` | `default` | Standard reader or answer-only `concise` |
| `answer_format` | `text` | Plain text or validated `structured` draft |
| `image_source_prose` | `include` | Keep image-source text in `content`, or `context_when_supplied` to move it to `auxiliary_source_text` when all source images are supplied |
| `evidence_operation_tool` | `False` | Enable native `submit_evidence_operation` tool for source selection, counting, and arithmetic (requires `answer_format: text`) |
| `source_metadata_key` | `source_id` | Application identifier field for structured and executed source lists |
| `request_max_bytes` | `None` | Optional complete answer-request JSON byte cap |
| `video_limit` | `8` | Retrieved-video cap (`None` disables) |
| `min_video_seconds` | `None` | Convert shorter videos to four ordered stills (requires `image` modality) |
| `media_policy` | `all` | Attach all eligible media, or select `on_demand` |
| `media_max_items` | `2` | On-demand selection cap (`1`–`8`) |
| `media_max_tokens` | `8192` | Positive output-token ceiling for media diagnosis |
| `routes` | Empty | Specialized generation models |
| `offline_gazetteer` | `None` | Pinned, caller-supplied local map index |

Modalities are atomic `text`, `image`, `video`, and `audio`. `temperature` accepts `[0, 2]` and `seed` accepts `[0, 2**63 - 1]`. `extra_body` cannot override `model`, `messages`, `response_format`, sampling, token limits, streaming, or storage controls.

### Sampling and streaming

Reasoning model families (`o1`/`o3`/`o4`, `gpt-5` reasoning variants) map `max_tokens` to `max_completion_tokens` (which includes reasoning tokens) and reject incompatible automatic `temperature`/`seed` controls; explicit unsupported values fail validation before inference.

Slot `stream: true` collects provider SSE before parsing (requiring the terminal usage chunk) and is independent of the public `ask_stream()` API.

### Answer presentation

- **Concise answers (`answer_style: concise`):** instructs the model to return requested values or entities without explanatory prose while preserving abstention and evidence rules.
- **Structured answers (`answer_format: structured`):** validates a typed draft (scalar, sum, difference, distinct count, date-interval union, or source list resolved through `source_metadata_key`) against fitted evidence before rendering. Structured streaming buffers validation before yielding text.
- **Image source prose (`image_source_prose: context_when_supplied`):** for originally image-only sources whose images all survive fitting, moves accompanying text to `auxiliary_source_text` so visible details are checked against pixels while non-visual background stays available. Falls back cleanly to ordinary `content` if the extra view exceeds request bounds or any image is omitted.
- **Source-bound evidence operations (`evidence_operation_tool: true`):** offers `submit_evidence_operation` on providers supporting native function calling for source selection, distinct counting, and witnessed arithmetic. Validated operations render directly; invalid initial proposals fall back to at most one final read of the fitted evidence without tools. `extra_body` cannot override `tools`, `tool_choice`, or `parallel_tool_calls` when enabled.

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

Routing checks question and evidence together, preferring complete native support, then the narrowest declared modality set, then declaration order. A failed call does not switch models. `Memory.capabilities.generation_routes` reports executable native combinations.

## Media selection and byte limits

`media_policy: on_demand` runs one text-only diagnosis call over up to 64 candidate attachments to select up to `media_max_items` for missing details or conflicting descriptions, bounded by `min(max_tokens, media_max_tokens)`. Direct composition uses `EvidenceGeneration(model, selector=model, media_max_items=2)`.

### Complete request-byte limit

When `request_max_bytes` is set and an answer payload exceeds it:

1. Retain the question, mandatory question attachments, and fitted source text.
2. Admit retrieved media in retrieval order, resizing images to JPEG or converting an oversized video to 16 ordered stills when image input is supported.
3. Mark omitted media explicitly with `media_omitted=true` and asset IDs.
4. Fail locally with `payload_too_large` if mandatory content still exceeds the bound.

Provider technical-duration errors retry once without that modality's evidence media; safety refusals raise `model_refused` immediately without dropping media.

### Image resizing

`image_max_pixels` accepts `1`–`64,000,000`. Images exceeding the configured pixel budget or 20 MiB inline limit are EXIF-oriented, encoded as RGB JPEG (quality 85, 4:2:0), and proportionally resized until they fit within the 20 MiB per-item and 64 MiB aggregate inline limits. Original stored assets remain unchanged. `image_max_pixels` participates in embedding, vision, and formation recipe identities.

## Offline map context

`generation.offline_gazetteer` requires the `gazetteer` and `openai` extras plus an immutable local SQLite index:

```yaml
generation:
  provider: openai
  model: your-vision-model
  modalities: [text, image]
  offline_gazetteer:
    index_path: ./data/named-geometries.sqlite3
    index_sha256: "<complete lowercase SHA256>"
    radius_m: 500
    limit: 8
    budget_chars: 16000
```

`index_path` and complete `index_sha256` are required; `radius_m` is `(0, 500]`, `limit` is `1`–`8`, and `budget_chars` is positive. Only verified original image EXIF GPS coordinates are queried, making no network calls and mutating no stored records. See [Offline gazetteer](../api/python-adapters.md#offline-gazetteer) for index format details.

## Local provider fields

| Slot/provider | Required | Defaults |
| --- | --- | --- |
| Embedding / Sentence Transformers | `model` | `revision=None`, `dimension=None`, `device=None`, `batch_size=32`, `dtype=auto`, `modalities=None` |
| Reranking / Sentence Transformers | `model` | `revision=None`, `device=None`, `batch_size=32`, `max_length=1024` |
| Reranking / Qwen3-VL | `model` | `revision=None`, `device=None`, `batch_size=32`, `max_length=8192`, `gpu_memory_utilization=0.9`, `image_limit=8`, `video_limit=2`, `image_max_pixels=262144`, `video_frames=8` |
| Reranking / vLLM service | `model`, `base_url` | `api_key=None`, `timeout=60.0`, `batch_size=32`, `modalities=[text,image,video]`, `media_candidates=None`, `media_weight=0.5` |
| Speech / FunASR | None | `device=auto`, `ncpu=4`, `vad_device=None`, `speaker_device=None` |
| Speech / OpenAI | None | `model=whisper-1`, `space=None`, plus connection fields |
| Face / OpenCV | `detector_model`, `recognizer_model` | `score_threshold=0.9`, `nms_threshold=0.3`, `top_k=5000`, `frame_interval_ms=1000`, `max_video_frames=300` |

Face thresholds accept `[0, 1]`. See [model setup](models.md) for loading and devices, and [identity thresholds](settings.md#identity-thresholds) for calibration.
