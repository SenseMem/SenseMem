# Model setup

Select an embedding model, add reranking or perception where useful, and keep stored model spaces compatible across restarts. This guide gives runnable configurations and runtime requirements for each bundled model path.

Start with [extras and composition](../configuration.md). [Provider fields](providers.md) lists connection controls and defaults.

## Embedding choices

Every store needs an embedding backend. Space and dimension must match when reopening.

| Provider | Selection and identity |
| --- | --- |
| `sentence-transformers` | Required `model`; optional full 40-character `revision` (omission resolves the current Hub commit with remote code disabled) |
| `openai` | Explicit deployed `model`, optional `space`, `dimension`, `modalities`, and `request_format` |

Pin a cached `revision` for offline reproducibility. Native media routes use original assets; matching speech or vision backends supply labelled text fallback for unsupported media, and missing required conversion raises `unsupported_modality`.

A directory without persisted speech derivations can add or change speech on reopen. Once speech data is stored, any configured speech backend must match the stored recipe; omitting the speech backend preserves cached transcripts and identities.

### EmbeddingGemma 2

Install `sensemem[local]`:

```json
{
  "data_dir": "./data/embeddinggemma2",
  "embedding": {
    "provider": "sentence-transformers",
    "model": "google/embeddinggemma-2",
    "dimension": 768,
    "device": "cuda",
    "dtype": "auto",
    "batch_size": 8
  }
}
```

The `local` extra uses Sentence Transformers 6.1+ and Transformers 5.18+. `dtype="auto"` selects BF16 on CUDA GPUs with native support and FP32 otherwise; explicit `float32` or `bfloat16` overrides automatic selection, while FP16 is rejected. Text queries and documents receive the checkpoint's retrieval prompts automatically while pure media receives no prefix. PyAV decodes video frames and 16 kHz mono float32 audio.

An optional `modalities` list omits unused encoders before weights allocate (`["text"]` loads the 270M text encoder; `["text", "image"]` or `["text", "video"]` drops audio; `["text", "audio"]` drops vision; omission loads all 740M weights). Supported dimensions are 128, 256, 512, and 768 within an 8,192-token shared context. Because BF16 and FP32 have distinct vector-space identities, pin `dtype` when reopening across different hardware. See the [official best practices](https://huggingface.co/google/embeddinggemma-2#best-practices) and the [precision check](../model-quality.md#check-embeddinggemma-2-loading).

### EmbeddingGemma 2 through vLLM

Install `sensemem[openai]` and deploy [the model server](../deployment.md#embeddinggemma-2-model-server) separately:

```json
{
  "data_dir": "./data/embeddinggemma2-vllm",
  "embedding": {
    "provider": "openai",
    "model": "google/embeddinggemma-2",
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key": "EMPTY",
    "dimension": 768,
    "modalities": ["text", "image", "audio", "video"],
    "request_format": "messages"
  }
}
```

The exact `google/embeddinggemma-2` model name automatically applies the checkpoint's query (`"task: search result | query: "`) and document (`"title: none | text: "`) prefixes, preserves interleaved text/media order, enables BOS/EOS tokens, and validates 768-dimensional vectors. Keep the checkpoint's bundled chat template on the vLLM server. Local and served EmbeddingGemma 2 recipes use distinct vector-space identities.

## Semantic reranking

Reranking is off unless configured. It runs after authoritative hydration, filters, relevance gating, and ambiguity checks, before the final `limit`. It changes order without modifying stored embeddings or bounded `SearchHit.score`; original rank breaks ties, and `search(..., explain=True)` exposes `reranking_score`. Changing rerankers requires reopening `Memory`, not re-embedding.

### Remote vLLM service

The `openai` extra supplies HTTP transport:

```yaml
reranking:
  provider: vllm
  model: your-reranker-model
  base_url: https://reranker.example
  api_key: replace-with-your-key
  batch_size: 32
  request_concurrency: 32
  media_candidates: 40
  media_weight: 0.5
  timeout: 240.0
```

`base_url` must be an absolute HTTP(S) base URL without query or fragment; the client appends `/score` using vLLM's `queries`/`documents` protocol. An unset `api_key` sends no authorization header and does not read `OPENAI_API_KEY`. `request_concurrency` bounds active HTTP batches (default `32`). Omit `media_candidates` for direct multimodal scoring, or set it to refine a text-ranked shortlist combined via `media_weight` in `(0, 1]`.

### Local Qwen3-VL

```bash
uv add "sensemem[local,reranking]"
```

```yaml
reranking:
  provider: qwen3-vl
  model: Qwen/Qwen3-VL-Reranker-8B
  device: cuda
  batch_size: 32
  max_length: 8192
  gpu_memory_utilization: 0.9
  image_limit: 8
  video_limit: 2
  image_max_pixels: 262144
  video_frames: 8
  video_cache_bytes: 268435456
```

Requires Linux x86-64/CUDA (`device` omitted or `"cuda"`; select GPUs via `CUDA_VISIBLE_DEVICES`). Text, images, and videos form query/candidate pairs following the [official vLLM reranker recipe](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/examples/reranker_vllm.ipynb); raw audio uses transcripts. Oversized tokenized pairs or media counts raise `payload_too_large` without silent truncation. `video_cache_bytes` bounds decoded frame-array caching (default 256 MiB; `0` disables).

Create `Memory` inside a `main()` guard so vLLM multiprocessing spawn works safely:

```python
import json
from pathlib import Path

from sensemem import Memory


def main() -> None:
    config = json.loads(Path("sensemem.json").read_text(encoding="utf-8"))
    with Memory.from_config(config) as memory:
        print(memory.search(("Find this scene", Path("query.mp4")), limit=5).hits)


if __name__ == "__main__":
    main()
```

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn uv run --frozen python run_memory.py
```

See [vLLM multiprocessing](https://github.com/vllm-project/vllm/blob/main/docs/usage/troubleshooting.md#python-multiprocessing).

### Text reranking and scoring

`provider: sentence-transformers` uses the `local` extra (`batch_size=32`, `max_length=1024`, remote code disabled). Long documents send query-focused windows alongside opening context while stored text remains unchanged. Reranking failures raise `ModelError(stage="reranking")` explicitly.

## Local speech with FunASR

The `local` extra installs the pinned FunASR package (requiring Git) with Fun-ASR-Nano (Chinese, English, and Japanese), FSMN-VAD, and CAM++ speaker embeddings via ModelScope Hub 0.6.1+:

```yaml
speech:
  provider: funasr
  device: cuda:0
  ncpu: 2
  vad_device: cpu
  speaker_device: cpu
```

`device` accepts `auto`, `cpu`, `cuda`, or `cuda:index`; `vad_device` and `speaker_device` inherit `device` when omitted. Videos without an audio track return empty speech analysis without invoking ASR. Positive `ncpu` sets process-wide PyTorch/BLAS thread budgets; inference per adapter is serialized. Device placement and `ncpu` do not alter `transcription_space`.

## Visual descriptions

A `vision` slot appends labelled captions and OCR to indexed text while preserving original media:

```yaml
vision:
  provider: openai
  model: your-vision-model
  modalities: [image]
```

Each write batch describes not-yet-described visuals in one completion, storing sections under `[visual description:<asset_id>]` and `[facts:<asset_id>]`. When speech is configured, video clip descriptions also receive diarised transcripts and may propose typed `SpeakerName` suggestions via `VisionAnalysisBackend`.

| Control | Behavior |
| --- | --- |
| `modalities` | `image` and/or `video`; undeclared media stay undescribed |
| `video_frames` | `2`–`64`, default `4` locally decoded ordered stills |
| Inline bounds | 20 MiB per item, 64 MiB aggregate across outbound images/stills |
| Malformed reply | One bounded correction attempt using only schema and expected count |
| Failure | Optional if native media or transcript remains usable; otherwise raises classified `ModelError` |

Setting `settings.query_vision=True` temporarily describes query media during search to add description/OCR routes without writing durable records or descriptions; see [runtime settings](settings.md).
