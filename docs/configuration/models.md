# Model setup

Choose the [extras and composition](../configuration.md) first. Connection fields and defaults
are in [Provider fields](providers.md). This page covers model recipes and runtime requirements.

## Embedding choices

Every store needs an embedding backend. Space and dimension must match when reopening.

| Provider | Selection and identity |
| --- | --- |
| `sentence-transformers` | Required model; optional full 40-character revision. Omission resolves the current Hub commit before immediate loading with remote code disabled |
| `openai` | Explicit deployed model/space/dimension and modalities; request format is part of identity |

An omitted local revision needs Hub access; pin a cached commit for reproducible offline loading.
Native media routes use original assets. Matching speech/vision backends can supply labelled text
fallback for unsupported media. Missing/empty required conversion raises `unsupported_modality`.
A transcript does not establish visual understanding. Atomic-only encoders keep supported atomic
keys while omitting an unsupported joint aggregate.

A directory without persisted speech derivations can enable/replace speech on reopen. Once stored,
a configured producer's recipe must match. Omitting that producer preserves cached transcripts,
identities, and recipe. A diagnostic model name alone does not invalidate an explicitly shared space.

### EmbeddingGemma 2

Install `sensemem[local]`:

```json
{
  "data_dir": "./data/embeddinggemma2",
  "embedding": {
    "provider": "sentence-transformers",
    "model": "google/embeddinggemma-2",
    "dimension": 768,
    "device": "cpu",
    "batch_size": 8
  }
}
```

The local extra requires Sentence Transformers 6.1+ and Transformers 5.18+. The adapter loads
FP32 on CPU or CUDA; injected encoders must use FP32/BF16, never FP16. Text queries/documents
receive the checkpoint's retrieval prompts automatically; pure media receives no prefix. Text,
images, audio, video, and ordered combinations share a space. PyAV preserves model video sampling.

Dimensions are 128/256/512/768 with renormalization after shortening. The shared context is 8192
tokens; split larger inputs. The text prompt recipe has its own identity. The Apache-2.0 checkpoint
loads with `trust_remote_code=False` and supplies embedding, not recognition or generation.

### EmbeddingGemma 2 through vLLM

Install `sensemem[openai]` on the application and deploy
[the model server](../deployment.md#embeddinggemma-2-model-server) separately:

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

The exact model name enables query prefix `"task: search result | query: "` and document prefix
`"title: none | text: "`; aliases do not. Mixed inputs retain attachment order, pure media has no
prefix, BOS/EOS tokens are enabled, and media uses inline data URLs. Keep the checkpoint's chat
template on the server.

Messages mode validates native 768 dimensions and requests no Matryoshka truncation. Its identity
differs from generic messages, unprompted stores, and the local recipe. Use fresh-store replay when
switching. An explicit space must describe the actual server recipe; record the resolved server
revision and validate numerical parity before treating local/served embeddings as equivalent.

## Semantic reranking

Reranking is off unless configured. It runs after authoritative hydration, scope/type/time
filters, relevance gating, and ambiguity checks, before the final result limit. It changes order,
not stored embeddings or bounded `SearchHit.score`. Original rank breaks ties;
`search(explain=True)` exposes the separate `reranking_score`.

### Remote vLLM service

The `openai` extra supplies transport:

```yaml
reranking:
  provider: vllm
  model: Qwen3-VL-Rerank-8B
  base_url: http://127.0.0.1:8002
  api_key: EMPTY
  batch_size: 32
  media_candidates: 40
  media_weight: 0.5
  timeout: 240.0
```

The absolute HTTP(S) server base must have no query/fragment; the client appends `/rerank`.
An unset key sends no authorization header and does not read `OPENAI_API_KEY`.
Counts/timeouts are positive, media weight is (0,1], and modalities include text/image/video.

The server owns revision, context, templates, and preprocessing; local revision/device/max-length
fields are unsupported. Omit `media_candidates` for direct multimodal pairs; setting it refines a
text-ranked shortlist and combines probability logits using `media_weight`. Record server identity.

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
```

Text and supported original images/videos automatically form query/candidate pairs, including
media-only inputs. Raw audio is unsupported; transcripts can supply text. The adapter follows the
[official pooling/classification recipe](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/examples/reranker_vllm.ipynb)
with its no/yes head and one classify call for the candidate pool. Batch size bounds concurrent sequences.

`max_length` bounds the complete tokenized pair including template/suffix. Oversized text or
expanded visual input raises `payload_too_large` at reranking without truncation, media removal,
or retry. Media counts combine query and candidate: eight images/two videos by default.
Processed images default to 262144 pixels; videos sample 2–64 ordered frames (default eight),
retaining timestamps without model resampling. Query decoding is reused; files stay leased.

The supported prebuilt runtime is Linux x86-64/CUDA with the lockfile's CUDA 13.0 PyTorch family.
Device is omitted or `cuda`; use `CUDA_VISIBLE_DEVICES` for selection.
`gpu_memory_utilization` is in (0,1] and defaults to 0.9.

Create models inside a main guard because vLLM may spawn after the embedder initializes CUDA:

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

Setting spawn alone does not replace the main guard. See
[vLLM multiprocessing](https://docs.vllm.ai/en/latest/usage/troubleshooting/#python-multiprocessing).

### Text reranking and scoring

`sentence-transformers` uses the `local` extra, `batch_size=32`, and `max_length=1024`.
An omitted revision resolves to an immutable commit; explicit revisions are full 40-character
commits and remote code is disabled. Its cross-encoder owns truncation, while Qwen rejects overflow.

Long documents supply exact query-related windows with opening context and matched middle/tail
passages; stored text and original reader evidence stay unchanged. Supported media accompanies
those windows. Unsupported candidates retain positions, and a query with no supported content skips
scoring. Models return aligned finite scores: Qwen sigmoid values are in [0,1], while text
cross-encoder logits can be negative. Neither is answer confidence. Failures raise classified
`ModelError(stage="reranking")` instead of silent fallback. Changing rerankers needs reopen, not replay.

## Local speech with FunASR

The `local` extra installs FunASR at
[`940b0d2463141a2e9ace8a9fd965b4aea93f2db5`](https://github.com/modelscope/FunASR/commit/940b0d2463141a2e9ace8a9fd965b4aea93f2db5),
so installation needs Git. The recipe pins Fun-ASR-Nano, FSMN-VAD, and CAM++; Nano covers Chinese,
English, and Japanese, rather than the MLT checkpoint's separate 31-language contract.

```yaml
speech:
  provider: funasr
  device: cuda:0
  ncpu: 2
  vad_device: cpu
  speaker_device: cpu
```

ASR device accepts auto/cpu/cuda/cuda:index. Auto selects the upstream accelerator or CPU fallback;
VAD/speaker placement accepts cpu/cuda/cuda:index and otherwise inherits ASR. Placement is fixed at
first load and preserves pinned VAD limits. CPU submodels trade accelerator memory for latency.

Positive `ncpu` changes process-wide PyTorch/BLAS budgets; another loaded model can replace them.
It does not enable concurrent transcriber calls: the loaded FunASR pipeline serializes inference.
Source revision and complete model recipe define transcription space, not placement/CPU budget.
Changing a persisted recipe needs fresh-store replay.

## Visual descriptions

A `vision` slot adds labelled captions/OCR to indexed text while preserving original media:

```yaml
vision:
  provider: openai
  model: gpt-5-mini-2025-08-07
  modalities: [image]
```

Each write batch describes applicable, not-yet-described visuals in one completion. Captions make
otherwise wordless images searchable lexically and provide fallback to text-only encoders.
Derived sections use `[visual description:<asset_id>]`; facts use `[facts:<asset_id>]`.
Original media still enters native routes.

Descriptions cover shown content, exact readable text, counts, places, clocks, ordered video
events, and tags. Visible instructions remain untrusted content. With speech configured, clip
descriptions also receive diarised transcripts. Bundled `Fact:` lines produce separately marked
facts; explicit transcript-supported speaker names can register identities. Custom plain
`describe()` text has no identity authority: proposals need optional `describe_analysis()`,
`VisualDescription`, and `SpeakerName`. Description caches use typed-v1.

| Control | Behavior |
| --- | --- |
| `modalities` | Only image/video; undeclared assets remain undescribed |
| `video_frames` | 2–64, default 4; ordered locally decoded stills including the last frame, never uploaded video |
| Reply shape | Exactly one nonempty caption per visual, not per frame; request carries pixels and ordinals, no stored IDs/paths |
| Inline bounds | 20 MiB per item, 64 MiB aggregate across outbound images/stills |
| Malformed reply | One correction attempt using only rejected JSON shape and expected count, never rejected caption text |
| Transport retries | Owned by SDK `max_retries`; kernel adds no waits/attempts |
| Failure | Optional if native media/transcript remains usable; otherwise preserve the original classified model error |

Unreadable video yields no caption. Failed writes are not cached; re-adding an embedded record
does not backfill descriptions. A failed capture settlement stays pending. Telemetry records
`sensemem.vision.failed_batches` and separate description tokens.
Supported generic models receive default zero temperature/fixed seed; known reasoning models omit
automatic sampling controls. Recipe/model identities must accompany comparisons.

`settings.query_vision=True` additionally describes query media temporarily and searches full
description/OCR routes beside native routes. Dense/hybrid can embed descriptions; lexical uses
full-text only. It creates no durable memory, identity, or description, caches attempts only within
the operation, and preserves original search on failure. Required caption fallback still runs
without this option. See [runtime settings](settings.md).
