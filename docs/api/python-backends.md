# Python backend protocols

Connect a custom model through a narrow inference protocol while retaining SenseMem's storage,
IDs, evidence, and visibility rules. Backends are runtime-checkable and must be thread-safe.
Import public protocols from `sensemem`; [bundled adapters](python-adapters.md) provide reference
implementations.

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
| Generation | `generation_capabilities` |

Resource-owning backends implement close; optional routing/admission protocols add no resource.
A space identifies the whole recipe. Change it when a stored derivation's model/prompt/input recipe
changes. For example, captions cache by asset content and `vision_space`; the bundled adapter hashes
the model, controls, and caption prompt.

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

`EmbeddingRouter` requires equal space/dimension, dispatches supported inputs, and preserves order.
Sentence Transformers snapshots joint/message support at construction. `Memory` closes distinct
nested resources once; compositions also allow direct close.

`EvidenceGeneration(backend, selector=None, enricher=None, media_max_items=2)` combines media selection
and source metadata independently of provider. `OpenAIModels` supplies selection/asset admission;
`GazetteerEnricher` supplies optional map context:

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
validates typed names and logs reversible identity operations. Typed caches recompute old formats.

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
Names/IDs appear only if earlier recognition rendered them into content. Opt-in history carries
recorded/occurrence time and typed status. After proposals, the kernel binds textual subjects to
visible names and propagates source context. See [typed memory](../typed-memory.md).

## ConsolidationBackend

```text
ConsolidationBackend.consolidate(
    evidence: Sequence[MemoryRecord],
    *,
    trigger: MemoryTrigger,
) -> tuple[MemoryOperation, ...]
```

Receives the kernel's bounded evidence set and may cite only that set. `IDENTIFY` uses `IdentityClaim`,
not `FormationProposal`; the kernel builds its typed naming assertion.
[Control operations](python-control.md) own eligibility, authority, and rollback.

## Optional recall planning

This extension lives in `sensemem.models.base` and `sensemem.recall` rather than root exports.
`OpenAIModels` implements it; answerers may omit it and use ordinary retrieval with a warning when
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
| `VisualRetrievalMatch` | `query_asset_id`, `memory_id`, `source_asset_id`, positive `rank`, `cosine_similarity` in `[-1, 1]` |
| `SpeechTurn` | `start_ms`, `end_ms`, `text`, `speaker_label` |
| `SpeakerEmbedding` | `speaker_label`, `values` |
| `SpeechAnalysis` | `turns`, `speakers` |
| `FaceEmbedding` | `face_label`, `values`, `bounding_box`, `observed_at_ms` |
| `FaceAnalysis` | `faces` |
| `FormationInput` | `memory_id`, `content`, `context`, `history=()`, `history_truncated=False`, `occurred_at=None`, `occurred_end=None`, `created_at=None` |
| `FormationProposal` | `kind`, `content`, `basis`, `subject`, `predicate`, `value`, `confidence`, `valid_from`, `valid_until`, `spatial`, `cue_modality`, `valence`, `arousal`, optional `evidence_ids` |
