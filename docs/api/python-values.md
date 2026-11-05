# Python public values

These immutable values and literal/enum contracts are re-exported from `sensemem`.
Use the [SDK index](python-sdk.md) to find methods consuming or returning them.

`MemoryRecord.content` preserves caller text at the front, followed by derived sections marked
`[transcript:<asset_id>]`, `[visual description:<asset_id>]`, `[facts:<asset_id>]`, or
`[speech identities:<asset_id>]`. Add derives before its first write; settlement appends after
capture. Original media remains content-addressed in assets.

## Records and retrieval

| Value | Fields |
| --- | --- |
| `Blob` | `data`, `media_type`, `name` |
| `AssetRef` | `id`, `modality`, `media_type`, `size_bytes`, `sha256`, `name`, `path`; `is_resolved` property |
| `StreamInput` | `content`, `occurred_at`, `occurred_end`, `metadata`, `memory_type`, `context`, `transcript`, `description` |
| `MemoryRecord` | `id`, `content`, `created_at`, `occurred_at`, `occurred_end`, `metadata`, `assets`, `modality`, `memory_type`, `context`, `place_id`, `forgotten_at` |
| `PendingCapture` | `memory_id`, `enqueued_at`, `attempts`, `last_error`, `awaiting` |
| `SearchHit` | all visible memory fields plus `score` |
| `AnswerResult` | `answer`, `hits`, `abstained`, `abstention_reason` |
| `AnswerChunk` | `text`, `result` |
| `Page` | `items`, `next_cursor` |
| `SearchResult` | `hits`, optional `trace=None` |
| `RetrievalTrace` | `candidates`, `candidate_limit`, `exhaustive`, `ambiguous` |
| `RetrievalCandidateTrace` | `memory_id`, `index_ids`, `dense_relevance`, `dense_confidence`, `lexical_relevance`, `lexical_rerank_bonus`, `lexical_match`, `gate_relevance`, `base_relevance`, `reinforcement_factor`, `temporal_factor`, `retention_factor`, `final_score`, `reranking_score=None`, `rank`, `rejected_by` |
| `RetrievalMode` | `hybrid`, `dense`, `lexical` instance candidate policy |

## Typed context and scope

| Value | Fields |
| --- | --- |
| `SpatialContext` | `frame_id`, `anchor`, `x`, `y`, `z`, `orientation_xyzw`, `position_uncertainty_m` |
| `ObservationContext` | `basis`, `source_id`, `confidence`, `valid_from`, `valid_until`, `spatial`, `place_id` |
| `MemoryContext` | `kind`, `basis`, `confidence`, `valid_from`, `valid_until`, `recorded_at`, `visible`, `retired_at`, `lineage_id`, `source_id`, `subject`, `predicate`, `value`, `evidence_ids`, `supersedes_id`, `model_id`, `recipe`, `identity_id`, `spatial`, `cue_modality`, `valence`, `arousal` |
| `RetrievalScope` | `valid_at`, `known_at`, `near`, `radius_m`, `place_id`, `identity_id` |

## Identity and retention

| Value | Fields |
| --- | --- |
| `SpeakerSegment` | `asset_id`, `start_ms`, `end_ms`, `text`, `speaker_id`, `speaker_name`, `identity_score` |
| `IdentityProfile` | `identity_id`, `name`, `relationship`, `confirmed`, `evidence_ids`; the last two are derived from the current visible naming assertion, never stored |
| `IdentityErasure` | `identity_id`, `alias_ids`, `face_exemplars`, `voice_exemplars`, `face_observations`, `speech_segments` |
| `ConsentClaim` | `identity_id`, `state`, `note`: one person's own statement, carried on the `CONSENT` operation that records it |
| `ExportBundle` | `exported_at`, `identity_id`, `identities`, `records`, `operations` |
| `RetentionPolicy` | `media_days`, `forgotten_days`, `capture_failure_days`; every field optional, and unset means no policy rather than zero days |
| `RetentionReport` | `dry_run`, `media_memory_ids`, `forgotten_memory_ids`, `cascade_memory_ids`, `asset_ids`, `capture_memory_ids`; `deleted` property |
| `FaceObservation` | `asset_id`, `bounding_box`, `identity_id`, `identity_name`, `identity_score`, `observed_at_ms` |
| `PerceptionResult` | typed `speech` segments, `faces` observations and per-asset `coverage` |
| `PerceptionCoverage` | `kind`, `asset_id`, optional skip `reason`; `processed` property |

## Compiled context

| Value | Fields |
| --- | --- |
| `ContextBudget` | `max_chars`, `max_items`, `max_media_items`, `memory_types`, `min_confidence`, `freshness`, `max_latency_ms`, `selected_proofs` |
| `ContextProof` | `anchor_id`, complete `nodes`; derived `confidence`, `footprint`, `independent_of()`, and `render()` |
| `ContextProofNode` | `memory_id`, `confidence`, `sources`, `capture_id` |
| `ContextConflict` | `lineage_id`, `subject`, `predicate`, `values`, `memory_ids` |
| `ContextUnknown` | `kind` (a `ContextUnknownKind`), `detail` |
| `NamedActor` | `identity_id`, `name`, `memory_ids`, `naming_assertion_id`: an identity a currently visible naming assertion names, reached through a compiled bundle's `actors` evidence rather than the assertion itself |
| `ProvisionalActor` | `identity_id`, `memory_ids` |
| `AffectCue` | every `SearchHit` field plus `event_ids`: one affect entry of a compiled bundle, carrying the events formed from the same observations the cue cites in its own `context.evidence_ids` |
| `TextSpanPiece` | `role` (`context` or `body`), half-open `start_codepoint`, `end_codepoint`, exact `source_text`, and `sha256` |
| `TextSpanSelector` | `parent_content_sha256`, `embedding_input_sha256`, `recipe_version`, ordered `pieces` |
| `ContextExcerpt` | `source_memory_id`, structured `matched_index_id`, partial `content`, `selector`, `score`, `created_at`, `occurred_at`, `occurred_end`, `memory_type`, `context`, `place_id` |
| `ContextSymbol` | request-local `symbol`, `namespace`, `stable_id`, `coverage`, structural `roles`, and `selector` only for partial memory coverage |
| `ContextCitation` | decoded `memory_id`, full or partial `coverage`, and `selector` exactly when partial; reference-only construction is rejected |
| `ContextPresentation` | compact `text`, typed `symbols`, exact `chars`; pure `resolve()` identity decoding and eligibility-checking `resolve_citation()` |
| `ContextBundle` | `goal`, `reference_at`, `budget`, `actors`, `relationships`, `scene`, `episodes`, `facts`, `procedures`, `affect`, `traits`, `conflicts`, `unknowns`, `occurred_from`, `occurred_until`, `frames`, `places`, `omitted`, `chars`, `elapsed_ms`, `deadline_exceeded`, `excerpts`, `proofs`; `hits` property, stable-ID `render()`, and opt-in `compact()` |

## Control operations

| Value | Fields |
| --- | --- |
| `MemoryOperation` | `intent`, `evidence_ids`, `target_ids`, `proposal`, `claim`, `consent`, `identity`, `rationale` |
| `IdentityClaim` | `identity_id`, `name`, `relationship` |
| `IdentityChange` | `identity_id`, `moved_ids` |
| `MemoryOperationRecord` | `operation_id`, `operation`, `trigger`, `applied_at`, `model_id`, `recipe`, `created_ids`, `changed_ids`, `forgotten_ids`, `superseded`, `rolled_back_at`, `outcome`, `outcome_note` |
| `ConsolidationCandidate` | `trigger`, `memory_ids`, `evidence_count` |
| `ConsolidationReport` | `operations`, `rejected` as `(MemoryOperation, reason)` pairs, `weighed`, `rounds`, `skipped`, `model_calls`; `applied` property |

## Stream packets and results

| Value | Fields |
| --- | --- |
| `StreamEvent` | `phase`, `item`, `stream_id` |
| `StreamCommit` | `record`, `prefetch`, `retrieval_error`, `stream_id`, `pending_settlement` |
| `PCMChunk` | `data`, `sample_rate_hz`, `channels`, `sample_width_bytes`, `stream_id`, `occurred_at` |
| `VADPacket` | `active`, `stream_id`, `occurred_at` |
| `ASRPartial` | `text`, `stream_id`, `occurred_at` |
| `AcousticBoundary` | `boundary`, `stream_id`, `occurred_at` |
| `VisionFrame` | `image`, `stream_id`, `occurred_at` |
| `VisionPartial` | `text`, `stream_id`, `occurred_at` |
| `SceneBoundary` | `boundary`, `stream_id`, `occurred_at` |
| `PrefetchResult` | `hits` |

## Composition values

| Value | Fields |
| --- | --- |
| `EmbeddingRoute` | exact `modalities`, `max_assets_per_modality=None` |
| `MemoryCapabilities` | `embedding`, `embedding_model`, `embedding_space`, `embedding_dimension`, `embedding_routes`, `generation`, `generation_routes`, `transcription`, `vision`, `face`, `formation`, `reranking`, `reranking_model`, `generation_model`, `transcription_space`, `vision_model`, `face_model`, `formation_model`, `consolidation_model`, `speaker_recognition`; `operations` property, `supports_embedding(modalities, assets_per_modality=1)`, `supports_generation(modalities)` and `document()` |

## Abstention signal

`abstained` is a protocol signal, not a refusal rate. The bundled grounded prompt asks for the
opaque token `[insufficient_evidence]`, and the adapter reports `abstained` when that bracketed
token appears anywhere in the answer, when the bare token is the whole answer, or when the answer
starts with the English fallback sentence. A model that refuses in its own words some other way is
still an ordinary answer. `abstention_reason` is `no_evidence` when retrieval returned nothing and
`insufficient_evidence` when hits were retrieved but could not ground an answer; the reported
`answer` is the readable fallback sentence rather than the token. Measure refusal rate separately
if that is the quantity needed.

## Enums

| Enum | Values |
| --- | --- |
| `Modality` | `text`, `image`, `video`, `audio`, `omni` |
| `AnswerPolicy` (literal alias) | `strict`, `best_effort` |
| `MemoryType` | `semantic`, `episodic`, `procedural` |
| `RetrievalMode` | `hybrid`, `dense`, `lexical` |
| `IndexQuantization` | `none`, `fp16`, `int8`, `rabitq` |
| `AbstentionReason` | `no_evidence`, `insufficient_evidence` |
| `RetrievalRejection` | `stale_index`, `occurrence_range`, `missing_memory`, `memory_type`, `minimum_relevance`, `ambiguity`, `limit` |
| `EmbedTask` | `retrieval.query`, `retrieval.document` |
| `MemoryKind` | `observation`, `entity`, `event`, `state`, `relation`, `affect`, `trait`, `response_policy` |
| `MemoryIntent` | `reinforce`, `consolidate`, `correct`, `forget`, `identify`, `consent`, `merge` |
| `MemoryOutcome` | `confirmed`, `refuted` |
| `ConsentState` | `granted`, `withheld`, `withdrawn` |
| `MemoryTrigger` | `manual`, `evidence`, `feedback`, `contradiction`, `query_failure`, `pressure`, `idle` |
| `EvidenceBasis` | `observation`, `user_statement`, `model_inference`, `response_feedback` |
| `SpatialAnchor` | `observer`, `subject` |
| `ContextUnknownKind` | `scope_empty`, `section_empty`, `budget_excluded`, `candidates_exhausted`, `modality_unsupported`, `stage_skipped`, `consent_withheld`, `evidence_unavailable` |
| `ContextSymbolNamespace` | `memory`, `identity` |
| `ContextSymbolCoverage` | `full`, `partial`, `reference_only` |
| `ContextSymbolRole` | `hit`, `excerpt`, `evidence`, `affect_event`, `actor`, `observed_in`, `naming_assertion`, `conflict` |
| `StreamPhase` | `update`, `final`, `cancel` |
| `AudioBoundary` | `start`, `end`, `cancel` |
| `VisionBoundary` | `start`, `end`, `cancel` |

## Memory capabilities

`capabilities` publishes what the composition declared — the modality set each backend accepts,
the embedding model, space and dimension, the optional model identities, and whether speaker
recognition is available. It also publishes `reranking` and `reranking_model`, including when
reranking is absent. It is a field read: it cannot raise or block.
It returns the immutable snapshot assembled when opening `Memory`, also used by internal routing.
Changing plugin declarations requires closing and reopening the instance.
`generation` lists atomic modalities; `generation_routes` lists the native coverage of each
executable route. `supports_generation(frozenset({Modality.TEXT, Modality.IMAGE}))` checks whether
one route covers that entire combination. Separate T+I and T+V routes do not advertise T+I+V.
This is native coverage before source-backed caption/transcript conversion, which can fail.

`embedding_routes` contains `EmbeddingRoute(modalities, max_assets_per_modality=None)` values.
Unlike generation route coverage, each embedding route names one **exact** native combination.
`None` means no declared per-modality asset count limit; `1` permits one image and one video,
but not two images. Use `capabilities.supports_embedding(modalities, assets_per_modality=1)`
to check a native aggregate; the count is the maximum number of assets of any one media kind.
A text-only input has count zero. This uses the same immutable declaration as the embedding plane.
An unsupported aggregate may still be indexed and searched through supported atomic keys.

```python
from sensemem import Modality

joint = frozenset({Modality.TEXT, Modality.IMAGE})
native_joint = memory.capabilities.supports_embedding(joint)
native_two_images = memory.capabilities.supports_embedding(joint, assets_per_modality=2)
```

`document()` is the shared JSON-ready renderer for SDK/doctor, REST health, and MCP instructions.
Network surfaces narrow operations to those they serve; SDK and doctor retain the full declaration.

## Root import inventory

The supported names exported by `sensemem` are:

| Group | Names |
| --- | --- |
| `Memory` | `Memory`, `AsyncMemory`, `AsyncOmniPrefetch`, `AsyncCaptureStream`, `AsyncAudioStream`, `AsyncVisionStream` |
| Composition | `SenseMemConfig`, `MemoryComposition`, `MemoryConfig`, `MemoryPlugins`, `resolve_memory_config` |
| Content and records | `ContentAtom`, `ContentInput`, `Blob`, `AssetRef`, `StreamInput`, `MemoryRecord`, `SearchHit`, `AnswerResult`, `AnswerChunk`, `Page`, `ObservationContext`, `MemoryContext`, `RetrievalScope`, `SpatialContext`, `SpeakerSegment`, `IdentityProfile`, `IdentityClaim`, `IdentityErasure`, `FaceObservation`, `MemoryCapabilities`, `PendingCapture`, `PrefetchResult`, `StreamCommit`, `SearchResult`, `PerceptionResult`, `PerceptionCoverage`, `PerceptionSkipReason`, `RetrievalTrace`, `RetrievalCandidateTrace`, `FormationProposal`, `ContextBudget`, `ContextBundle`, `ContextExcerpt`, `ContextPresentation`, `ContextSymbol`, `ContextCitation`, `TextSpanSelector`, `TextSpanPiece`, `ContextConflict`, `ContextUnknown`, `AffectCue`, `NamedActor`, `ProvisionalActor`, `IdentityChange`, `MemoryOperation`, `MemoryOperationRecord`, `ConsolidationReport`, `ConsolidationCandidate`, `ConsentClaim`, `ExportBundle`, `RetentionPolicy`, `RetentionReport` |
| Delivery certificates | `ContextProof`, `ContextProofNode` |
| Stream input | `AudioStreamPacket`, `PCMChunk`, `VADPacket`, `ASRPartial`, `AcousticBoundary`, `VisionStreamPacket`, `VisionFrame`, `VisionPartial`, `SceneBoundary`, `StreamEvent` |
| Enums and literal aliases | `AnswerPolicy`, `PerceptionKind`, `Modality`, `MemoryType`, `EvidenceBasis`, `MemoryKind`, `MemoryIntent`, `MemoryTrigger`, `SpatialAnchor`, `ContextUnknownKind`, `ContextSymbolNamespace`, `ContextSymbolCoverage`, `ContextSymbolRole`, `AbstentionReason`, `IndexQuantization`, `RetrievalMode`, `RetrievalRejection`, `StreamPhase`, `AudioBoundary`, `VisionBoundary`, `EmbedTask`, `MemoryOutcome`, `ConsentState` |
| Backend protocols and values | `EmbeddingBackend`, `EmbeddingRoute`, `RerankingBackend`, `GenerationBackend`, `StreamingGenerationBackend`, `TranscriptionBackend`, `SpeechBackend`, `VisionDescriptionBackend`, `VisionAnalysisBackend`, `EmbeddingInputSupport`, `MediaSelectionBackend`, `SourceMetadataBackend`, `GenerationMediaBackend`, `GenerationRoutingBackend`, `VisualDescription`, `SpeakerName`, `FaceBackend`, `FormationBackend`, `ConsolidationBackend`, `ModelInput`, `VisualRetrievalMatch`, `FormationInput`, `SpeechTurn`, `SpeakerEmbedding`, `SpeechAnalysis`, `FaceEmbedding`, `FaceAnalysis` |
| Bundled adapters | `SentenceTransformersEmbedder`, `SentenceTransformersReranker`, `Qwen3VLReranker`, `VLLMReranker`, `OpenAIModels`, `GenerationRouter`, `EmbeddingRouter`, `EvidenceGeneration`, `GazetteerEnricher`, `OfflineGazetteer`, `OpenCVFaceAnalyzer`, `FunASRTranscriber`, `FunASRRecipe`, `DEFAULT_FUNASR_MODEL_ID`, `DEFAULT_FUNASR_RECIPE` |
| Exceptions | `SenseMemError`, `ValidationError`, `MemoryNotFoundError`, `IdentityNotFoundError`, `ModelError`, `ModelOutputTruncatedError`, `StorageError`, `IndexUnavailableError` |
