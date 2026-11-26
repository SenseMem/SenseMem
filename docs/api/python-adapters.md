# Bundled Python model adapters

Choose a bundled adapter for local embedding, model endpoints, reranking, speech, faces, or map context. Pass inference adapters to the [Memory constructor](python-sdk.md#construction) or `MemoryPlugins`; compose the optional tool-using agent separately with `AgentRunner`.

[Backend protocols](python-backends.md) define inference methods; [model setup](../configuration/models.md) provides declarative examples and runtime requirements.

| Adapter | Role |
| --- | --- |
| [SentenceTransformersEmbedder](#sentencetransformersembedder) | Local embedding |
| [SentenceTransformersReranker](#sentencetransformersreranker) | Local text ranking |
| [VLLMReranker](#vllmreranker) | Hosted multimodal ranking |
| [Qwen3VLReranker](#qwen3vlreranker) | Local multimodal ranking |
| [FunASRTranscriber](#funasrtranscriber) | Speech/transcription |
| [OpenCVFaceAnalyzer](#opencvfaceanalyzer) | Local face analysis |
| [OpenAIEmbedder](#openaiembedder) | Compatible embedding endpoint |
| [OpenAITranscriber](#openaitranscriber) | Compatible transcription endpoint |
| [OpenAICompletion](#openaicompletion) | Completion, formation, vision, planning, event segmentation, and consolidation |
| [OpenAIAgent](#openaiagent) | Optional multimodal agent steps with native tool calls |
| [Offline gazetteer](#offline-gazetteer) | Explicit map-source enrichment |
| [Named recipes](#named-recipes) | Published compositions and probes |

## SentenceTransformersEmbedder

```text
SentenceTransformersEmbedder(
    encoder=None,
    *,
    model_id: str,
    revision: str | None = None,
    dimension: int | None = None,
    device: str | None = None,
    batch_size: int = 32,
    dtype: Literal["auto", "float32", "bfloat16"] = "auto",
    modalities: Iterable[Modality] | None = None,
)

SentenceTransformersEmbedder.load(...) -> SentenceTransformersEmbedder
```

Loads the checkpoint immediately with remote code disabled unless an `encoder` is supplied. Explicit `revision` is a 40-character commit; omission resolves the Hub revision. For EmbeddingGemma 2, `dtype="auto"` uses BF16 on capable CUDA GPUs and FP32 otherwise (FP16 is rejected), and `modalities` selects which encoders to load (`128`/`256`/`512`/`768` dimensions). See [EmbeddingGemma 2 setup](../configuration/models.md#embeddinggemma-2).

## SentenceTransformersReranker

```text
SentenceTransformersReranker(
    encoder=None,
    *,
    model_id: str,
    revision: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 1024,
)

SentenceTransformersReranker.load(...) -> SentenceTransformersReranker
```

Loads a pinned cross-encoder with remote code disabled and serializes per-adapter inference. Predict requests raw logits without softmax; `max_length` bounds pairs. `Memory` owns the adapter, and changing it requires reopening rather than re-embedding.

## VLLMReranker

```text
VLLMReranker(
    client: httpx2.Client | None = None,
    *,
    model_id: str,
    base_url: str,
    api_key: str | None = None,
    timeout: float = 60.0,
    batch_size: int = 32,
    request_concurrency: int = 32,
    modalities: Iterable[Modality] = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
    media_candidates: int | None = None,
    media_weight: float = 0.5,
)
```

Calls `base_url/score` with vLLM's `queries`/`documents` protocol and restores finite scores in input order. Owns and closes its HTTP client (waiting for active calls). `request_concurrency` bounds concurrent batches; optional `media_candidates`/`media_weight` enable text-shortlist multimodal logit fusion. See [remote reranking](../configuration/models.md#remote-vllm-service).

## Qwen3VLReranker

```text
Qwen3VLReranker(
    engine=None,
    *,
    model_id: str,
    revision: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 8192,
    gpu_memory_utilization: float = 0.9,
    image_limit: int = 8,
    video_limit: int = 2,
    image_max_pixels: int = 262144,
    video_frames: int = 8,
    video_cache_bytes: int = 268435456,
)

Qwen3VLReranker.load(...) -> Qwen3VLReranker
```

Requires the `reranking` extra on Linux x86-64/CUDA (`device` is `None` or `"cuda"`; select GPUs via `CUDA_VISIBLE_DEVICES` inside a `main()` guard). Classifies text, image, and video pairs with pinned revisions and remote code disabled, caching decoded video frames up to `video_cache_bytes`. Oversized inputs raise `payload_too_large`. See [local Qwen3-VL](../configuration/models.md#local-qwen3-vl).

## FunASRTranscriber

```text
FunASRTranscriber(
    recipe: FunASRRecipe = DEFAULT_FUNASR_RECIPE,
    *,
    device: str = "auto",
    ncpu: int = 4,
    vad_device: str | None = None,
    speaker_device: str | None = None,
)

FunASRRecipe(
    model_id: str,
    vad_model: str,
    speaker_model: str | None,
    model_revision: str | None = None,
    vad_revision: str | None = None,
    speaker_revision: str | None = None,
    punctuation_model: str | None = None,
    punctuation_revision: str | None = None,
    vad_max_single_segment_ms: int | None = None,
    hub: str = "ms",
    trust_remote_code: bool = False,
)
```

`DEFAULT_FUNASR_MODEL_ID` and `DEFAULT_FUNASR_RECIPE` publish the speech defaults. `transcription_space` encodes the pinned FunASR revision and recipe; device placement and `ncpu` do not alter it. See [speech setup](../configuration/models.md#local-speech-with-funasr).

## OpenCVFaceAnalyzer

```text
OpenCVFaceAnalyzer(
    detector_model: str | Path,
    recognizer_model: str | Path,
    *,
    score_threshold: float = 0.9,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    frame_interval_ms: int = 1000,
    max_video_frames: int = 300,
)
```

Supply caller-managed detector and recognizer model files and calibrate thresholds against your target camera and population; see [identity thresholds](../configuration/settings.md#identity-thresholds).

## OpenAIEmbedder

```text
OpenAIEmbedder(
    client: OpenAI | None = None,
    *,
    owns_client: bool = False,
    embedding_model: str = "text-embedding-3-small",
    embedding_space: str | None = None,
    embedding_dimension: int = 1536,
    embedding_request_format: Literal["input", "messages"] = "input",
    embedding_capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
    image_max_pixels: int | None = None,
)
```

Implements `EmbeddingBackend`. Its space identifies model, dimension, task prefix, normalization, and media transport recipe.

## OpenAITranscriber

```text
OpenAITranscriber(
    client: OpenAI | None = None,
    *,
    owns_client: bool = False,
    transcription_model: str = "whisper-1",
    transcription_space: str | None = None,
    transcription_prompt: str | None = None,
    transcription_keywords: Sequence[str] | None = None,
    transcription_languages: Sequence[str] | None = None,
    transcription_capabilities: frozenset[Modality] = frozenset({Modality.AUDIO, Modality.VIDEO}),
)
```

Implements `TranscriptionBackend`. Prompt, keyword, and language hints participate in `transcription_space`.

## OpenAICompletion

```text
OpenAICompletion(
    client: OpenAI | None = None,
    *,
    owns_client: bool = False,
    generation_model: str = "gpt-5-mini-2025-08-07",
    generation_capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
    generation_seed: int | None = None,
    generation_temperature: float | None = None,
    generation_max_tokens: int | None = None,
    generation_media_max_tokens: int = 8192,
    generation_response_format: Literal["json_object", "json_schema"] = "json_schema",
    generation_store: bool = False,
    generation_min_video_seconds: float | None = None,
    generation_video_limit: int | None = 8,
    generation_extra_body: Mapping[str, object] | None = None,
    generation_stream: bool = False,
    generation_answer_style: Literal["default", "concise"] = "default",
    generation_answer_format: Literal["text", "structured"] = "text",
    generation_evidence_operation_tool: bool = False,
    generation_image_source_prose: Literal["include", "context_when_supplied"] = "include",
    generation_source_metadata_key: str = "source_id",
    vision_video_frames: int = 4,
    image_max_pixels: int | None = None,
    generation_request_max_bytes: int | None = None,
)
```

Each OpenAI adapter accepts one SDK client (`owns_client=False` by default so caller-supplied clients stay open on adapter close; `owns_client=True` transfers closure). `OpenAICompletion` also implements `EventSegmentationBackend` (`event_model`, `event_recipe`) for [`AsyncEventStream`](python-streams.md#asynceventstream).

| Direct control | Declarative counterpart |
| --- | --- |
| `embedding_request_format` | `embedding.request_format` |
| `generation_answer_style` / `generation_answer_format` | `generation.answer_style` / `answer_format` |
| `generation_image_source_prose` | `generation.image_source_prose` |
| `generation_evidence_operation_tool` | `generation.evidence_operation_tool` |
| `generation_source_metadata_key` | `generation.source_metadata_key` |
| `generation_request_max_bytes` | `generation.request_max_bytes` |
| `generation_media_max_tokens` | `generation.media_max_tokens` |
| `generation_min_video_seconds` / `generation_video_limit` | `generation.min_video_seconds` / `video_limit` |
| `vision_video_frames` | `vision.video_frames` |
| `generation_stream` | `stream` on the corresponding completion slot |

See [provider fields](../configuration/providers.md) for answer presentation (`concise`, `structured`, `image_source_prose`, `evidence_operation_tool`), sampling, image resizing, and request-byte fitting.

### Image source metadata

Answering and descriptions read only validated GPS and EXIF capture time from original image bytes (preserving explicit UTC offsets while omitting camera IDs and free-form tags). Metadata stays untrusted and source-bound without adding durable fields.

## OpenAIAgent

```text
OpenAIAgent(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str = DEFAULT_AGENT_INSTRUCTIONS,
    capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
    max_tokens: int | None = None,
    max_tokens_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_completion_tokens",
    strict_tools: bool = False,
    audio_data_url: bool = False,
)
```

Implements `AgentBackend.step` over Chat Completions for [`AgentRunner`](python-agents.md), preserving ordered media and prior tool messages with `store=False`. `OpenAIAgent.close()` leaves caller-owned SDK clients open.

## Offline gazetteer

Install `sensemem[openai,gazetteer]` and supply an immutable local SQLite map index (using an open caller-owned OpenAI `client`):

```python
from sensemem import (
    EvidenceGeneration,
    GazetteerEnricher,
    Modality,
    OfflineGazetteer,
    OpenAICompletion,
)

gazetteer = OfflineGazetteer(
    "./data/named-geometries.sqlite3",
    index_sha256="<complete lowercase SHA256>",
    radius_m=500,
    limit=8,
    budget_chars=16000,
)
answerer = EvidenceGeneration(
    OpenAICompletion(
        client=client,
        generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
    ),
    enricher=GazetteerEnricher(gazetteer),
)
```

`OfflineGazetteer.nearby(latitude, longitude)` queries the SHA-verified read-only SQLite RTree index (`features`, `bounds`, `metadata`) with no network calls and returns detached candidate dictionaries (`status`, `candidates`, source provenance, `global_complete=False`, `photo_identity_confidence="unknown"`). Supports both single-region (`offline-osm-named-geometries-v1-aeqd-wgs84-step0005`) and multi-region (`offline-osm-named-geometries-v2-aeqd-wgs84-step0005`) indexes with geometry densified to at most 0.0005 degrees. Lookups exceeding 10,000 prefilter features or 32 MiB of geometry return `lookup_incomplete=True`. Declarative configuration uses `generation.offline_gazetteer`; see [offline map context](../configuration/providers.md#offline-map-context).

## OpenAI request behavior

Direct methods use `ModelInput`; public `Memory` methods accept `ContentInput` directly:

```text
embed(inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT) -> tuple[tuple[float, ...], ...]
form(inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]
answer(question: ModelInput, hits: Sequence[SearchHit], *, answer_policy: AnswerPolicy = "strict", exhaustive: bool = False) -> AnswerResult
stream_answer(question: ModelInput, hits: Sequence[SearchHit], *, answer_policy: AnswerPolicy = "strict", exhaustive: bool = False) -> Generator[str, None, AnswerResult]
transcribe(assets: Sequence[AssetRef]) -> tuple[str, ...]
```

Media data URLs are bounded at 20 MiB each and 64 MiB aggregate. Oversized videos fall back to four ordered stills for embedding and sixteen for answering (with explicit lost-motion/audio/timing notes) while stored assets remain unchanged. Grounded answer text is capped at 4 MiB; oversized individual embedding keys elide with `sensemem.embedding.elided_parts` while usable keys keep the record searchable.

## Named recipes

Named recipes live in `sensemem.recipes`:

```text
import sensemem.recipes as recipes

recipes.names() -> tuple[str, ...]
recipes.slots(name: str) -> tuple[str, ...]
recipes.require_slot(name: str, slot: str) -> None
recipes.probe(name: str) -> str
recipes.describe(name: str) -> dict[str, object]
recipes.embedder(name: str, *, load: bool = False) -> EmbeddingBackend
recipes.answerer(name: str, *, load: bool = False) -> GenerationBackend
recipes.former(name: str, *, load: bool = False) -> FormationBackend
recipes.consolidator(name: str, *, load: bool = False) -> ConsolidationBackend
recipes.vision(name: str, *, load: bool = False) -> VisionDescriptionBackend
recipes.transcriber(name: str, *, load: bool = False) -> SpeechBackend | TranscriptionBackend
```

Each factory returns a caller-owned adapter (`load=True` runs its loader probe). `describe()` returns per-slot `classes` and `models` mappings.
