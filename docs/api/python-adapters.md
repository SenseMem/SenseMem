# Bundled Python model adapters

Use these objects with the [Memory constructor](python-sdk.md#construction) or `MemoryPlugins`.
Their inference methods implement the [backend protocols](python-backends.md).
For declarative examples and runtime requirements, see [model setup](../configuration/models.md).

| Adapter | Role |
| --- | --- |
| [SentenceTransformersEmbedder](#sentencetransformersembedder) | Local embedding |
| [SentenceTransformersReranker](#sentencetransformersreranker) | Local text ranking |
| [VLLMReranker](#vllmreranker) | Hosted multimodal ranking |
| [Qwen3VLReranker](#qwen3vlreranker) | Local multimodal ranking |
| [FunASRTranscriber](#funasrtranscriber) | Speech/transcription |
| [OpenCVFaceAnalyzer](#opencvfaceanalyzer) | Local face analysis |
| [OpenAIModels](#openaimodels) | Compatible embedding/completion/transcription capabilities |
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
)

SentenceTransformersEmbedder.load(
    model_id: str,
    *,
    revision: str | None = None,
    dimension: int | None = None,
    device: str | None = None,
    batch_size: int = 32,
) -> SentenceTransformersEmbedder
```

A supplied encoder implements supports, `get_embedding_dimension`, `encode_query`, and `encode_document`.
Without one, constructor/load immediately load the checkpoint. Explicit revision is a complete
40-character commit; omission resolves the Hub revision and needs access. An injected encoder
needs an immutable model _commit_hash or explicit revision. Remote model code is disabled.

EmbeddingGemma 2 loads FP32 and supports 128/256/512/768 dimensions without optional Matryoshka
metadata. Checkpoint query/document prompts and model-specific input limits are documented in
[model setup](../configuration/models.md#embeddinggemma-2).

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

SentenceTransformersReranker.load(
    model_id: str,
    *,
    revision: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 1024,
) -> SentenceTransformersReranker
```

Loads a pinned cross-encoder when no encoder is supplied. Omitted revision resolves through Hub;
an injected encoder also needs an explicit revision to avoid resolution. Loading disables remote
code. Inference/batching serialize per adapter. Predict requests raw logits with identity activation
and no softmax. `Max_length` bounds the pair with cross-encoder truncation. `Memory` owns the adapter;
changing it requires reopen, not re-embedding.

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
    modalities: Iterable[Modality] = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
    media_candidates: int | None = None,
    media_weight: float = 0.5,
)
```

Uses the openai extra's HTTP transport for `base_url`/rerank and restores finite input-order scores.
It owns its client, including an injected client, and waits for active calls on close.
Server deployment owns revision/media/context policy. Optional `media_candidates`/`media_weight`
add bounded text/media probability-logit fusion; usage counts both passes.
See [remote reranking](../configuration/models.md#remote-vllm-service).

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
)

Qwen3VLReranker.load(
    model_id: str,
    *,
    revision: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 8192,
    gpu_memory_utilization: float = 0.9,
    image_limit: int = 8,
    video_limit: int = 2,
    image_max_pixels: int = 262144,
    video_frames: int = 8,
) -> Qwen3VLReranker
```

Requires the reranking extra on Linux x86-64/CUDA. Device is `None` or cuda; CPU/MPS/cuda:N are
rejected. Select GPUs through `CUDA_VISIBLE_DEVICES`. Create the owner inside a main guard for spawn.

The vLLM pooling runner classifies all text/native-image/native-video pairs with pinned model and
tokenizer revisions, remote code disabled, and checkpoint yes/no sigmoid scores. Batch size limits
concurrent sequences. Oversized tokenized pairs or media counts raise `payload_too_large` at
reranking with no truncation/removal/retry. Query decoding is reused, source timing retained,
and transcripts can represent unsupported raw audio. Close shuts down the engine.
[Model setup](../configuration/models.md#local-qwen3-vl) owns complete input limits and sampling.

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

`DEFAULT_FUNASR_MODEL_ID` and `DEFAULT_FUNASR_RECIPE` publish the speech defaults.
`FunASRRecipe.auto_model_arguments`() returns dict[str,object]. The transcription space includes the
immutable FunASR revision and complete recipe; execution placement/CPU budget do not change it.
Ncpu affects process-wide PyTorch/BLAS threads; VAD/speaker devices can differ.
See [speech setup](../configuration/models.md#local-speech-with-funasr).

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

Supply caller-managed detector/recognizer model files and evaluate matching thresholds against
the intended camera/population. See [identity calibration](../configuration/settings.md#identity-thresholds).

## OpenAIModels

```text
OpenAIModels(
    client: OpenAI | None = None,
    *,
    embedding_client: OpenAI | None = None,
    generation_client: OpenAI | None = None,
    transcription_client: OpenAI | None = None,
    embedding_model: str = "text-embedding-3-small",
    embedding_space: str | None = None,
    embedding_dimension: int = 1536,
    embedding_request_format: Literal["input", "messages"] = "input",
    generation_model: str = "gpt-5-mini-2025-08-07",
    transcription_model: str = "whisper-1",
    transcription_space: str | None = None,
    transcription_prompt: str | None = None,
    transcription_keywords: Sequence[str] | None = None,
    transcription_languages: Sequence[str] | None = None,
    embedding_capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
    generation_capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
    transcription_capabilities: frozenset[Modality] = frozenset({Modality.AUDIO, Modality.VIDEO}),
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
    generation_source_metadata_key: str = "source_id",
    vision_video_frames: int = 4,
    image_max_pixels: int | None = None,
    generation_request_max_bytes: int | None = None,
)
```

The common client fills unset operation-specific clients. Missing clients fail only when called.
Close leaves caller-supplied OpenAI clients open. Pass the same object only to desired `Memory` slots.

| Direct control | Declarative counterpart |
| --- | --- |
| `embedding_request_format` | `embedding.request_format` |
| `generation_answer_style` / `generation_answer_format` | `generation.answer_style` / `answer_format` |
| `generation_source_metadata_key` | `generation.source_metadata_key` |
| `generation_request_max_bytes` | `generation.request_max_bytes` |
| `generation_media_max_tokens` | `generation.media_max_tokens` |
| `generation_min_video_seconds` / `generation_video_limit` | `generation.min_video_seconds` / `video_limit` |
| `vision_video_frames` | `vision.video_frames` |
| `generation_stream` | stream on the corresponding completion slot |

Transcription prompt/keywords/languages are provider hints. Source identities and recipes include
the corresponding input options. [Provider fields](../configuration/providers.md) owns connection,
sampling, routing, response format, storage, image fitting, and request-byte behavior.

### Structured answer witnesses

A direct adapter returns supporting hits in draft order; `Memory` intersects them with fitted evidence
and preserves kernel order. Labels must refer to final fitted evidence; source lists require the
configured metadata key. Invalid labels/missing IDs/unsupported witnesses raise `ModelError`.

Numeric/date witnesses are exact uniquely located quotes with matching literals, or permitted
`occurred_at` date witnesses. Arithmetic uses validated operands; durations declare closed/half-open
intervals. This does not prove relevance, event uniqueness, unit conversion, or set completeness.
Dates require `source_value` plus granularity/format. Iso8601 preserves timezone; `utc_datetime` needs
explicit zone and minute/second precision, rendering UTC without suffix. Number/duration formats
are `with_unit` or number and preserve unit checks. Missing format is `response_invalid`.
Structured streaming buffers the draft and yields rendered text, never raw JSON.

### Image source metadata

Answering/descriptions read only validated GPS and capture time from original image EXIF, preserving
an explicit UTC offset when available. Unzoned capture time remains local and never replaces
caller occurrence. Camera IDs/comments/free-form fields are excluded. Invalid metadata is omitted
without rejecting pixels, while source bytes must match SHA.

Metadata stays untrusted and source-bound even after request resizing; it adds no `AssetRef`/durable
fields. No address lookup occurs unless the offline gazetteer is enabled. Existing stores need no
re-ingest for answer metadata; caption recipe identity changes, embedding identity does not.

## Offline gazetteer

This construction example assumes an open, caller-owned OpenAI client named `client`. Keep it
open for the owning `Memory` lifetime and close it afterward.

```python
from sensemem import (
    EvidenceGeneration,
    GazetteerEnricher,
    Modality,
    OfflineGazetteer,
    OpenAIModels,
)

gazetteer = OfflineGazetteer(
    "/absolute/path/named-geometries.sqlite3",
    index_sha256="<complete lowercase SHA256>",
    radius_m=500,
    limit=8,
    budget_chars=16000,
)
answerer = EvidenceGeneration(
    OpenAIModels(
        generation_client=client,
        generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
    ),
    enricher=GazetteerEnricher(gazetteer),
)
```

Declare the modalities supported by the provider. Pass the answerer through the existing `Memory` constructor or
`MemoryPlugins`. Declarative configuration exposes the same resource through
`generation.offline_gazetteer`; see [configuration](../configuration/providers.md).
Install `sensemem[openai,gazetteer]`; importing `sensemem` does not import Shapely or pyproj.

`OfflineGazetteer.nearby(latitude, longitude)` returns a detached dictionary with `status`,
`candidates`, source/license provenance, `global_complete=False` and
`photo_identity_confidence="unknown"`. Candidate fields include OSM type/ID, kind, geometry type,
original names and aliases, explicit address tags, metric distance, coverage/boundary flags and
spatial relation. Names and map data are untrusted. Proximity refers to the recorded coordinate,
not an assertion about what a photo depicts. The source snapshot may postdate capture.
`provenance` returns a detached dictionary; `index_sha256` and `budget_chars` are read-only
properties. No persistent database connection is held between lookups. Multiple instances may
share this immutable external index; it is not a SenseMem `data_dir` or an isolation boundary.

The single-region v1 index has SQLite tables `metadata(key,value)`,
`features(id,osm_type,osm_id,kind,tags_json,geometry_wkb)` and RTree
`bounds(id,min_lon,max_lon,min_lat,max_lat)`. Metadata values are JSON. Its recipe is exactly
`offline-osm-named-geometries-v1-aeqd-wgs84-step0005`; source geometry must be densified to at
most 0.0005 degrees. Source metadata requires an HTTPS URL, source SHA256, ISO snapshot date,
`extract_bbox_lonlat` and optional header fields. Attribution is
`© OpenStreetMap contributors; ODbL 1.0`, with license URL
`https://www.openstreetmap.org/copyright`. The index declares
`geometry_segment_max_degrees=0.0005` and `complete_world_or_region_map=false`.
No alternate schema is migrated or guessed, and the SDK does not build an index from raw OSM.

A multi-region index uses the explicit recipe
`offline-osm-named-geometries-v2-aeqd-wgs84-step0005`. Its `features` table adds
`source_index` after the six v1 columns; `bounds` is unchanged. Metadata replaces `source`
with `sources`, an ordered list of at least two complete source mappings with the same
provenance fields described above. Mixing `source` and `sources`, an invalid source index,
or duplicate `(osm_type, osm_id)` features rejects the index. Each feature belongs to exactly
one source; the adapter does not choose between duplicate copies of an OSM feature.

The constructor is unchanged. For v2, `nearby` and `provenance` return the complete `sources`
list instead of `source`, and each candidate carries its original zero-based `source_index`.
A lookup first admits sources whose own extract bounds contain the coordinate, then sorts
their geometry candidates together by distance before applying the same candidate limit.
The answer context includes `map_sources` in stable source-index order for only the sources
represented by the candidates actually admitted under the budget. Each entry identifies its
source index, URL, PBF SHA, snapshot date and replication timestamp; there is no synthesized
single date or source file. Source headers and candidates share the character budget,
including every repeated metadata occurrence. V1 retains its existing single-source fields.

Supply a complete, checkpointed main SQLite snapshot and keep it immutable. Only that main file
is SHA-verified and queried, using SQLite's immutable read-only mode; adjacent WAL/SHM files
are ignored. Its file identity, size and modification time are subsequently checked, including
cached lookups.
Invalid options, missing dependencies, index or provenance raise
`ValidationError`; changed/unavailable indexes and SQLite read failures raise `StorageError`.
Coordinates and provider bodies are not logged. A lookup exceeding 10000 prefilter features or
32 MiB of geometry returns unknown with `lookup_incomplete=True`, rather than claiming that a
partial scan found the nearest place. Empty/out-of-region results also remain unknown.
The added map context has a separate whole-candidate character budget; the existing total answer
byte bound remains in force. Enabling this resource changes the available external knowledge,
requires task-specific quality evaluation and must be declared in comparisons.

## OpenAI request behavior

Direct methods use `ModelInput`; public `Memory` methods accept `ContentInput` directly.

```text
embed(
    inputs: Sequence[ModelInput],
    task: EmbedTask = EmbedTask.DOCUMENT,
) -> tuple[tuple[float, ...], ...]

form(inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]

answer(
    question: ModelInput,
    hits: Sequence[SearchHit],
    *,
    answer_policy: AnswerPolicy = "strict",
    exhaustive: bool = False,
) -> AnswerResult

stream_answer(
    question: ModelInput,
    hits: Sequence[SearchHit],
    *,
    answer_policy: AnswerPolicy = "strict",
    exhaustive: bool = False,
) -> Generator[str, None, AnswerResult]

transcribe(assets: Sequence[AssetRef]) -> tuple[str, ...]
```

Close releases the adapter's own resources, not injected OpenAI clients.

Structural requests default to strict JSON schema; JSON-object mode is explicit and retains
validation. Strict rejection never silently downgrades. `Generation_store` is a strict boolean,
default false, and cannot be overridden through provider fields. Sampling/token serialization and
provider SSE are documented in [Provider fields](../configuration/providers.md#sampling-and-streaming).

Media data URLs are bounded at 20 MiB each and 64 MiB aggregate, roughly 15/48 MiB untransformed
on-disk bytes. Embedding oversized video uses four locally decoded ordered stills or fails if
decoding cannot meet bounds; answering can use sixteen stills with explicit lost motion/audio/
timing labels. Stored assets remain unchanged. Required original-whole media needs a custom
provider upload adapter.

Grounded answer text, including system/user text and attachment labels, is capped at 4 MiB.
Question capacity is reserved before ranked evidence; overflow media may remain as text. A single
oversized embedding key can be elided while usable keys keep the write searchable, recorded as
`sensemem.`embedding.elided_parts``. Refusing every key fails. Technical context-length video fallback
records `sensemem.`embedding.video_sampled_inputs`` and preserves original assets for answering.

## Named recipes

Named recipes are available through the supported `sensemem.recipes` submodule, not as a root
re-export:

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
recipes.transcriber(
    name: str,
    *,
    load: bool = False,
) -> SpeechBackend | TranscriptionBackend
```

Each construction function returns an object the caller owns and closes. `load=True` additionally
exercises the recipe's published probe (`weights`, `import`, or `client`).
