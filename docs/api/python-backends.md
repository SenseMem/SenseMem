# Python backend protocols

Connect a custom model through a narrow inference protocol while retaining SenseMem's storage,
IDs, evidence, and visibility rules. Backends are runtime-checkable and must be thread-safe.
Import public protocols from `sensemem`; [bundled adapters](python-adapters.md) provide reference
implementations.

For retrieval scoring, context selection, and ordered ingestion stages, use the separate
[computation plugin protocols](python-policies.md). Those protocols propose decisions while the
kernel retains evidence, budget, and transaction admission.

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

Resource-owning backends implement close; optional routing/admission protocols add no resource.
A space identifies the whole recipe. Change it when a stored derivation's model/prompt/input recipe
changes. For example, captions cache by asset content and `vision_space`; the bundled adapter hashes
the model, controls, and caption prompt.

The kernel depends on these protocols rather than provider SDK objects. The bundled `OpenAIEmbedder`, `OpenAITranscriber`, and `OpenAICompletion` implement
embedding, transcription, and completion roles through explicit official OpenAI SDK clients. Its templates, request
serialization and fitting, media encoding and integrity checks, and domain-output parsing live in
separate internal components; [the model boundary](../architecture.md#model-boundary) lists their
responsibilities. The component split preserves these public signatures and the current prompts.

The official SDK handles authentication, HTTP transport, timeouts, and network retries. SenseMem
retains one bounded semantic retry for a malformed HTTP 200 JSON result, with both attempts
accounted for; a refusal or truncation does not enter that retry. Stream accumulation reuses the
SDK's `ChatCompletionStreamState`, with local response-contract and resource-lifetime checks.

## EmbeddingBackend

```text
EmbeddingBackend.embed(
    inputs: Sequence[ModelInput],
    task: EmbedTask = EmbedTask.DOCUMENT,
) -> tuple[tuple[float, ...], ...]
```

Return one vector per input. The kernel checks count, width, numeric values, and finite nonzero
magnitude, then normalizes once for retrieval/persistence. Direct bundled embed callers receive
the model's vector scale. Sentence Transformers capability probes must return booleans;
probe exceptions remain failures.

Optional `EmbeddingInputSupport.embedding_routes` declares exact native modality combinations and
per-modality count limits. Without it, every nonempty subset of atomic capabilities is assumed
native with no declared count limit. Unsupported aggregates can still contribute supported atomic keys.

## RerankingBackend

```text
RerankingBackend.rerank(
    query: ModelInput,
    documents: Sequence[ModelInput],
) -> tuple[float, ...]
```

Return aligned finite scores in document order, never replacement source IDs. The kernel prepares
supported content and skips unsupported inputs; direct adapter calls reject unsupported modalities.
Bad score count/type/finiteness raises `response_invalid` at reranking. Failures are explicit.
Scores are request-local, not calibrated answer confidence or a durable output space.
`RetrievalCandidateTrace` records `reranking_score` separately from bounded `final_score`.

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

The kernel always supplies `answer_policy` and exhaustive, including defaults. Streaming yields
deltas and returns `AnswerResult` as the generator completion value; missing results or bare hit
tuples are invalid. Exhaustive describes the executed predicate evidence, not global knowledge.

`ModelInput` includes normalized text, resolved assets, and request-only source/evidence metadata.
The latter do not affect embedding-key equality and remain untrusted.
An answerer gets the resolved clock as the final Reference time for relative dates line when
routing produces text. It must not substitute its own clock when the line is absent.

Temporary `image_retrieval_matches` can identify up to three image-only neighbors per query image
inside scoped candidates. They reuse stored keys/vectors, make no model call, and establish
neither identity nor class labels. Incomplete key layouts yield no association. The kernel removes
matches for omitted records/assets; attachment labels preserve gaps after omission.

### Routing and evidence composition

`GenerationRouter` selects native support first, then the narrowest backend covering question and
evidence, then caller order. `GenerationRoutingBackend` exposes `select(question, hits, convertible=frozenset())`
and nonempty `generation_routes` whose union matches atomic capability. `EvidenceGeneration` preserves
that selection when nested inside/outside routers.

Routing may omit derived captions, transcripts, or model interpretations when a native-only
backend accepts their original media. It never discards original text parts to make an input
fit: a backend unable to consume those words raises `ModelError` with `unsupported_modality`.
Embedding keys follow the same distinction, including after capture settlement and reopening.

`EmbeddingRouter` requires equal space/dimension, dispatches supported inputs, and preserves order.
Sentence Transformers snapshots joint/message support at construction. `Memory` closes distinct
nested resources once; compositions also allow direct close.

`EvidenceGeneration(backend, selector=None, enricher=None, media_max_items=2)` combines media selection
and source metadata independently of provider. `OpenAICompletion` supplies selection/asset admission;
`GazetteerEnricher` supplies optional map context.

`answer()` and `stream_answer()` perform this preparation automatically after route selection.
Preparation may invoke the configured media-selection backend or metadata enricher, and remains
internal to those generation operations.

The extension protocols are:

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
TranscriptionBackend.transcribe(
    assets: Sequence[AssetRef],
) -> tuple[str, ...]

SpeechBackend.analyze(
    assets: Sequence[AssetRef],
) -> tuple[SpeechAnalysis, ...]
```

Speech returns `SpeechAnalysis`(turns,speakers), with `SpeechTurn` and `SpeakerEmbedding`.
Plain transcription returns aligned text.

### FaceBackend

```text
FaceBackend.analyze(
    assets: Sequence[AssetRef],
) -> tuple[FaceAnalysis, ...]
```

Returns `FaceAnalysis` containing `FaceEmbedding` observations.

### VisionDescriptionBackend and VisionAnalysisBackend

```text
VisionDescriptionBackend.describe(
    inputs: Sequence[ModelInput],
) -> tuple[str, ...]

VisionAnalysisBackend.describe_analysis(
    inputs: Sequence[ModelInput],
) -> tuple[VisualDescription, ...]
```

`VisualDescription(text="", facts=(), speaker_names=())` carries observation text and explicit proposals.
`SpeakerName` refers to local `speaker_N` labels. Plain strings have no naming authority. The kernel
validates typed names and logs sourced, reversible `MODEL_INFERENCE` identity operations. One
recording cannot confirm a name; independent support controls visibility, and only host
registration makes `IdentityProfile.confirmed` true. Typed caches use the current
`VisualDescription` representation.

## EventSegmentationBackend

```text
EventSegmentationBackend.segment_events(
    observations: Sequence[EventObservation],
    *,
    end_of_stream: bool = False,
) -> tuple[EventSegment, ...]
```

`EventObservation(content: ModelInput, occurred_at, occurred_end)` supplies an ordered, timed
media window. `EventSegment(end_chunk, summary, complete=True)` closes one complete event from the
start of the window or the previous segment's end. `end_chunk` is an exclusive observation offset,
starting at one; offsets must increase and stay within the window. A nonempty summary of at most
16,384 characters must describe the event without inventing unseen completion.

For an unfinished suffix, return one final `EventSegment(..., complete=False)` ending at the window
length. Its summary preserves the ongoing goal, observed steps, and unresolved outcome. The runtime
can replace that analyzed prefix with one explicitly labelled, untrusted model interpretation in
the next assessment, alongside new original media. Its assets repeat the last already analyzed
original chunk as motion and sound context; this overlap adds no boundary offset. The synopsis
counts as one observation when
addressing boundaries. Every original chunk remains retained outside the model window. An empty
tuple is accepted, but repeated empty results eventually fill the bounded analysis window; a
continuation or complete prefix is then required before more input can be accepted.

`end_of_stream=True` asks for a final assessment, not permission to declare a truncated action
complete. Observations and embedded speech, text, or images are untrusted evidence and cannot
change the segmentation policy. `AsyncEventStream` validates boundaries and summaries, then
commits the confirmed prefixes through the ordinary memory path. The backend has no storage
authority. Its `event_model` and `event_recipe` identify the model and full segmentation recipe in
each event's metadata. The caller owns the backend and its resource lifetime.

`OpenAICompletion` implements this protocol through its configured completion model and media
capabilities. See [automatic event segmentation](../omni-streaming-and-interaction-memory.md#automatic-event-segmentation)
for the product path.

## FormationBackend

```text
FormationBackend.form(
    inputs: Sequence[FormationInput],
) -> tuple[tuple[FormationProposal, ...], ...]
```

Inputs are committed sources; return one proposal tuple per input in order. A former never writes
storage. The kernel checks source modality/spatial frame, IDs, and evidence before committing.

`FormationInput` retains occurrence/end/creation timestamps; supplied times must be timezone-aware
and an end needs an earlier start. Resolve relative dates against occurrence, then creation.
For example, yesterday observed on March 2 is March 1 even if stored later. Missing timestamps
and undated evidence do not authorize a wall-clock guess. The SDK supplies committed `created_at`.

`evidence_ids` is `None` for the primary singleton or a nonempty unique ordered conjunction containing
the primary input. It can cite that exact batch or that target's active history, never another
target's private history. Separate clauses are alternatives. Losing a member retires the whole
conjunction; provenance does not establish independent or complete semantic support.

History is text-only, unique, excludes the primary target, and reports truncation. Retired
background can inform interpretation but not support an affirmative proposal. Commit revalidates
citations atomically. Canonical self-citation refuses the complete proposal without manufacturing
a smaller clause; other valid proposals survive. Place and metadata inheritance requires agreement
across cited sources; uncited background has no effect.

The bundled former receives aliases, enriched content/assets, occurrence/creation, basis,
confidence, explicit validity, and spatial frame/anchor. It receives no identity roster.
Names/IDs appear only if earlier recognition rendered them into content. Bounded history carries
recorded/occurrence time and typed status. After proposals, the kernel binds textual subjects to
visible names and propagates source context. See [typed memory](../typed-memory.md).
The default is eight history rows within 6,000 content characters; set
`MemorySettings(formation_history_max_rows=0)` to disable it. Custom formers must respect active
witness and naming-citation rules when using this context.

Automatic formation and consolidation proposals must have `MODEL_INFERENCE` basis, including
when their source is host-marked `USER_STATEMENT`. The kernel rejects any other basis from an
automatic backend. Source provenance, exact copied text, and a model-selected reporter do not
confirm the extracted claim's meaning or who asserted it. The bundled wire schema accepts neither
`statement_quote` nor `basis`; the adapter always constructs an inference.

An application can explicitly confirm a typed claim with
`Memory.apply(MemoryOperation(intent=MemoryIntent.CONSOLIDATE, evidence_ids=(source.id,),
proposal=FormationProposal(..., basis=EvidenceBasis.USER_STATEMENT)))`.
That host operation retains the ordinary evidence and kind-specific validation. The source's
`ObservationContext.basis` remains unchanged by automatic extraction.

## ConsolidationBackend

```text
ConsolidationBackend.consolidate(
    evidence: Sequence[MemoryRecord],
    *,
    trigger: MemoryTrigger,
    query_failure: QueryFailure | None = None,
) -> tuple[MemoryOperation, ...]
```

Receives the kernel's bounded evidence set and may cite only that set. `IDENTIFY` uses `IdentityClaim`,
not `FormationProposal`; the kernel builds its typed naming assertion.
Explicitly selected hidden/retired assertions are target-only; never cite them as evidence.
The optional `QueryFailure` preserves the failed request and its retrieval constraints for diagnosis.
[Control operations](python-control.md) own eligibility, authority, and rollback.

## Optional recall planning

This extension lives in `sensemem.models.base` and `sensemem.recall` rather than root exports.
`OpenAICompletion` implements it; answerers may omit it and use ordinary retrieval with a warning when
planning was requested.

```text
plan_recall(
    question: str,
    *,
    reference_at: datetime,
    corpus_digest: str,
    attempted: str = "",
) -> str | None
```

Return JSON text, not an executed plan. The kernel validates and falls back for unusable or empty
steps. First attempted is empty; a later abstained round can carry a bounded note/evidence excerpts.
No keep/unresolved controls exist. A non-abstained answer stops. See
[recall settings](../configuration/settings.md#recall-planning).

## Backend values

| Backend value | Fields |
| --- | --- |
| `ModelInput` | `text`, `assets`, `image_retrieval_matches=()`, request-only `source_metadata={}`, `evidence_sources=()`; derived `modality` and `modalities` properties |
| `EventObservation` | `content`, `occurred_at`, `occurred_end` |
| `EventSegment` | `end_chunk`, `summary`, `complete=True`; only complete results establish finality |
| `VisualRetrievalMatch` | `query_asset_id`, `memory_id`, `source_asset_id`, positive `rank`, `cosine_similarity` in `[-1, 1]` |
| `SpeechTurn` | `start_ms`, `end_ms`, `text`, `speaker_label` |
| `SpeakerEmbedding` | `speaker_label`, `values` |
| `SpeechAnalysis` | `turns`, `speakers` |
| `FaceEmbedding` | `face_label`, `values`, `bounding_box`, `observed_at_ms` |
| `FaceAnalysis` | `faces` |
| `FormationInput` | `memory_id`, `content`, `context`, `history=()`, `history_truncated=False`, `occurred_at=None`, `occurred_end=None`, `created_at=None` |
| `FormationProposal` | `kind`, `content`, `basis`, `subject`, `predicate`, `value`, `confidence`, `valid_from`, `valid_until`, `spatial`, `cue_modality`, `valence`, `arousal`, optional `evidence_ids` |
