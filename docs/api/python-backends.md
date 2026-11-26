# Python backend protocols

Connect a custom model through a narrow inference protocol while retaining SenseMem's storage, IDs, evidence, and visibility rules. Backends are runtime-checkable and must be thread-safe. Import public protocols from `sensemem`; [bundled adapters](python-adapters.md) provide reference implementations.

For query planning, candidate discovery, retrieval scoring, context selection, and ordered ingestion stages, see [computation plugin protocols](python-policies.md).

## Declarations and ownership

| Capability | Required properties |
| --- | --- |
| Embedding | `embedding_capabilities`, `embedding_model`, `embedding_space`, `embedding_dimension` |
| Reranking | `reranking_capabilities`, `reranking_model` |
| Transcription/speech | `transcription_capabilities`, `transcription_model`, `transcription_space` |
| Face | `face_capabilities`, `face_model`, `face_space`, `face_analysis_space` |
| Vision description | `vision_capabilities`, `vision_model`, `vision_space` |
| Formation | `formation_capabilities`, `formation_model`, `formation_space` |
| Consolidation | `consolidation_model`, `consolidation_recipe` |
| Event segmentation | `event_model`, `event_recipe` |
| Generation | `generation_capabilities` |

Resource-owning backends implement `close()`; optional routing/admission protocols add no resource. A `*_space` property identifies the complete recipe—change it whenever the model, prompt, or preprocessing changes so cached derivations never mix incompatible outputs.

## EmbeddingBackend

```text
EmbeddingBackend.embed(
    inputs: Sequence[ModelInput],
    task: EmbedTask = EmbedTask.DOCUMENT,
) -> tuple[tuple[float, ...], ...]
```

Return one vector per input. The kernel checks count, width, numeric values, and finite nonzero magnitude, then normalizes once for retrieval and persistence. Optional `EmbeddingInputSupport.embedding_routes` declares exact native modality combinations and per-modality count limits; unsupported aggregates still contribute supported atomic keys.

## RerankingBackend

```text
RerankingBackend.rerank(
    query: ModelInput,
    documents: Sequence[ModelInput],
) -> tuple[float, ...]
```

Return aligned finite scores in document order. The kernel prepares supported content and skips unsupported inputs; invalid score counts or non-finite values raise `response_invalid` at `stage="reranking"`. Scores are request-local and recorded as `RetrievalCandidateTrace.reranking_score`.

## GenerationBackend and streaming

```text
GenerationBackend.answer(
    question: ModelInput,
    hits: Sequence[SearchHit],
    *,
    answer_policy: AnswerPolicy = "strict",
    exhaustive: bool = False,
) -> AnswerResult

StreamingGenerationBackend.stream_answer(
    question: ModelInput,
    hits: Sequence[SearchHit],
    *,
    answer_policy: AnswerPolicy = "strict",
    exhaustive: bool = False,
) -> Generator[str, None, AnswerResult]
```

The kernel always passes `answer_policy` and `exhaustive`. Streaming yields text deltas and returns `AnswerResult` as the generator completion value. `ModelInput` carries normalized text, resolved assets, and request-only source/evidence metadata; when routing produces text, the kernel appends the resolved clock line (`Reference time for relative dates`). Temporary `image_retrieval_matches` may identify up to three image-only neighbors per query image inside scoped candidates.

### Routing and evidence composition

`GenerationRouter` selects native support first, then the narrowest backend covering question and evidence, then caller order. `GenerationRoutingBackend` exposes `select(question, hits, convertible=frozenset())` and `generation_routes`. Routing may omit derived captions, transcripts, or interpretations when a native-only backend accepts their original media, while never discarding original text parts.

`EmbeddingRouter` requires equal space and dimension across routes and preserves input order. `EvidenceGeneration(backend, selector=None, enricher=None, media_max_items=2)` composes media selection and source metadata enrichment internally during `answer()` and `stream_answer()`:

```text
MediaSelectionBackend.select_media(
    payload: Mapping[str, object], *, count: int, max_items: int,
) -> tuple[int, ...]

SourceMetadataBackend.enrich(assets: Sequence[AssetRef]) -> Mapping[str, dict[str, object]]

GenerationMediaBackend.grounding_assets(
    question: ModelInput, hits: Sequence[SearchHit],
) -> tuple[AssetRef, ...]
```

## Perception protocols

### TranscriptionBackend and SpeechBackend

```text
TranscriptionBackend.transcribe(assets: Sequence[AssetRef]) -> tuple[str, ...]
SpeechBackend.analyze(assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]
```

`SpeechBackend` returns `SpeechAnalysis(turns, speakers)` with `SpeechTurn` and `SpeakerEmbedding`; `TranscriptionBackend` returns aligned text.

### FaceBackend

```text
FaceBackend.analyze(assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]
```

Returns `FaceAnalysis` containing `FaceEmbedding` observations.

### VisionDescriptionBackend and VisionAnalysisBackend

```text
VisionDescriptionBackend.describe(inputs: Sequence[ModelInput]) -> tuple[str, ...]
VisionAnalysisBackend.describe_analysis(inputs: Sequence[ModelInput]) -> tuple[VisualDescription, ...]
```

`VisualDescription(text="", facts=(), speaker_names=())` carries observation text and typed `SpeakerName` proposals referring to source-local `speaker_N` labels. Plain strings have no naming authority; model-proposed names carry `MODEL_INFERENCE` and require two independent groups before projection.

## EventSegmentationBackend

```text
EventSegmentationBackend.segment_events(
    observations: Sequence[EventObservation],
    *,
    end_of_stream: bool = False,
) -> tuple[EventSegment, ...]
```

`EventObservation(content: ModelInput, occurred_at, occurred_end)` supplies a timed media window. `EventSegment(end_chunk, summary, complete=True)` closes a complete prefix ending at exclusive offset `end_chunk` (starting at 1), with a nonempty summary of at most 16,384 characters. Return one trailing `EventSegment(..., complete=False)` ending at the window length to summarize an unfinished suffix across rolling windows. `end_of_stream=True` requests a final assessment without forcing unfinished activities to close. See [automatic event segmentation](../omni-streaming-and-interaction-memory.md#automatic-event-segmentation).

## FormationBackend

```text
FormationBackend.form(
    inputs: Sequence[FormationInput],
) -> tuple[tuple[FormationProposal, ...], ...]
```

Receives committed sources and returns one proposal tuple per input in order without writing storage. Resolve relative dates against `occurred_at`, then `created_at`. `evidence_ids` is `None` (primary singleton) or a nonempty unique ordered conjunction including the primary input and citing only the current batch or that target's active history. Canonical self-citation refuses the individual proposal while valid proposals survive; citations omitting the primary input or outside the batch and history fail the derived commit.

Automatic formation and consolidation proposals must carry `MODEL_INFERENCE` basis, even when the source is host-marked `USER_STATEMENT`. To confirm a typed claim explicitly, use `Memory.apply(MemoryOperation(intent=MemoryIntent.CONSOLIDATE, evidence_ids=(source.id,), proposal=FormationProposal(..., basis=EvidenceBasis.USER_STATEMENT)))`.

## ConsolidationBackend

```text
ConsolidationBackend.consolidate(
    evidence: Sequence[MemoryRecord],
    *,
    trigger: MemoryTrigger,
    query_failure: QueryFailure | None = None,
) -> tuple[MemoryOperation, ...]
```

Receives the kernel's bounded evidence set and may cite only shown standing records (`IDENTIFY` uses `IdentityClaim`, not `FormationProposal`). Explicit hidden/retired assertions are target-only. See [control operations](python-control.md).

## Optional recall planning

Lives in `sensemem.models.base` and `sensemem.recall` (`OpenAICompletion` implements it):

```text
plan_recall(
    question: str,
    *,
    reference_at: datetime,
    corpus_digest: str,
    attempted: str = "",
) -> str | None
```

Returns JSON text for the kernel to validate and execute; unusable plans fall back to ordinary ranking. See [recall planning](../configuration/settings.md#recall-planning).

## Backend values

| Backend value | Fields |
| --- | --- |
| `ModelInput` | `text`, `assets`, `image_retrieval_matches=()`, request-only `source_metadata={}`, `evidence_sources=()`; derived `modality` and `modalities` properties |
| `EventObservation` | `content`, `occurred_at`, `occurred_end` |
| `EventSegment` | `end_chunk`, `summary`, `complete=True` |
| `VisualRetrievalMatch` | `query_asset_id`, `memory_id`, `source_asset_id`, positive `rank`, `cosine_similarity` in `[-1, 1]` |
| `SpeechTurn` | `start_ms`, `end_ms`, `text`, `speaker_label` |
| `SpeakerEmbedding` | `speaker_label`, `values` |
| `SpeechAnalysis` | `turns`, `speakers` |
| `FaceEmbedding` | `face_label`, `values`, `bounding_box`, `observed_at_ms` |
| `FaceAnalysis` | `faces` |
| `FormationInput` | `memory_id`, `content`, `context`, `history=()`, `history_truncated=False`, `occurred_at=None`, `occurred_end=None`, `created_at=None` |
| `FormationProposal` | `kind`, `content`, `basis`, `subject`, `predicate`, `value`, `confidence`, `valid_from`, `valid_until`, `spatial`, `cue_modality`, `valence`, `arousal`, optional `evidence_ids` |
