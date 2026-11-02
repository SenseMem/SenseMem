# Changelog

Notable release changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
MindBridge is pre-1.0; backward compatibility is not maintained.

## Unreleased

This tree targets `0.2.0` and replaces the unreleased service-oriented `0.1.0` design.
The [documentation index](docs/README.md) describes the current contracts.

### Added

- `mindbridge-bench eval --tasks videozerobench` downloads pinned VideoZeroBench inputs, reuses
  per-video memory and media preparation, and runs five diagnostic levels with deterministic
  bilingual answer, temporal and spatial scoring. Primary accuracy uses Level-3; terminal output
  and paired comparisons cover all seven named metrics across Level-1 through Level-5. Named
  joint metrics retain upstream thresholds and denominators. Comparisons record `score_scale`
  so regression thresholds retain their meaning for percentage scores. Reports identify the
  memory transport's differences from the upstream frame-sampling protocol.
- Native `embedding_routes`, typed `EmbeddingRoute` asset-count limits and `supports_embedding`
  share their declarations with execution. Capability documents also publish `reranking` and
  `reranking_model`. `AsyncMemory` now mirrors `forget_identity`.

- Automatic source-backed modality conversion across writes, queries, answering and formation;
  explicit `GenerationRouter`, same-space `EmbeddingRouter`, and joint-input support contracts.
  `generation_routes` and `supports_generation` describe executable combinations: native T+I
  and T+V routes do not imply T+I+V.
- Per-asset perception `coverage` preserves successful empty analysis and skip reasons on
  Python, REST, MCP and CLI.
- Typed visual facts and speaker naming proposals, with identity behavior owned by the identity
  plane. Provider-neutral `EvidenceGeneration` composes media selection and source enrichment.
- Image-bearing `ask` inputs carry temporary `VisualRetrievalMatch` values from existing
  image-only embedding keys. Answer transport labels only associations whose source media
  actually reached the reader; similarities do not become durable facts or verified identities.
- Optional `OfflineGazetteer` and `generation.offline_gazetteer` add bounded, explicitly untrusted
  external map candidates from a pinned read-only local OSM index to answer evidence. The default
  remains disabled; no coordinates leave the machine for lookup and no memory facts are written.
- Public `Memory` and `AsyncMemory` facades for ordered text, image, video, audio, and composite
  observations. Semantic, episodic, and procedural roles share event-time intervals, metadata,
  spatial context, explicit retrieval scopes, and typed results.
- Deferred ingestion through `capture()`, `settle()`, and durable pending captures; streaming
  observations, grounded answer streaming, and asynchronous speculative prefetch.
- Context compilation with bounded evidence closure, identity and consent obligations, exact raw
  excerpts, affect cues, compact Python presentation, and opt-in selected proof delivery.
  Independent-evidence projection remains opt-in and pinned to the store.
- A memory control plane for consolidation, reinforcement, cognitive forgetting, correction,
  rollback, candidate scheduling, and auditable operations. Derived assertions retain explicit
  provenance and temporal versions.
- Face and speaker recognition, named identities, identity linking and erasure, sourced affect
  hypotheses, visual descriptions, automatic typed formation, and bounded optional history.
- Opt-in `query_vision` adds transient query-image description and OCR retrieval routes through
  the configured vision backend. `vision.video_frames` selects 2–64 description stills, default 4;
  non-default counts participate in the visual-description recipe identity.
- OpenAI embedding, generation, and vision adapters accept opt-in `image_max_pixels` bounds for
  request-local image conversion without changing stored assets. Embedding and answering can
  send oversized videos as bounded ordered stills, with explicit sampling limitations.
- Optional `generation.answer_style: concise` requests only the question's answer without
  unrelated dates, explanations or citations. The default reader prompt and both evidence-policy
  refusal rules are preserved; benchmark baseline arms honor the same configured style.
- Explicit backend composition through `MemoryPlugins`, `MemoryConfig`, declarative
  configuration, and named recipes. Local Sentence Transformers, FunASR, OpenCV face, and
  caller-supplied OpenAI SDK adapters load their optional dependencies lazily.
- Qwen3-VL reranking uses vLLM's pooling classifier and automatically combines query and candidate
  text, original images and sampled videos. `RerankingBackend` declares `reranking_capabilities`
  and accepts `ModelInput` pairs; the separate media interface and manual modality switch are removed.
  Install the `reranking` extra; the shared lock uses CUDA 13 PyTorch builds. See
  [semantic reranking](docs/configuration.md#semantic-reranking) for runtime controls and limits.
- REST under `/v1`, typed MCP tools, and the `mindbridge` product CLI over the shared SDK.
  Capability documents describe the operations and modalities available on each surface.
- OpenTelemetry stage spans, provider request IDs, token usage, streamed time to first token,
  retrieval explanations, and classified retryable errors without ordinary content logging.
- The `mindbridge-bench` CLI with pinned dataset adapters and official scoring protocols,
  physically isolated stores, response caches, resume support, per-task artifacts, diagnostics,
  rejudging, and local storage measurements. See [benchmarking](docs/benchmarking.md).
- A real-model quality gate over fixed answering fixtures, isolated optional-dependency checks,
  and wheel/sdist validation across the supported Python 3.11–3.14 range.

### Changed

- Remove adapter-only plain-string embedding and generation inputs; direct model calls now use
  `ModelInput` consistently. The public memory methods still accept text. Structured answer
  parsing requires the `format` field already required by its wire schema. Benchmark rejudging
  preserves saved local metrics without repairing scores from obsolete artifacts.
- Validate plugin protocols once at assembly for every constructor path, before opening storage.
  Normalize and validate answer evidence at the result boundary; invalid plugin output reports
  `response_invalid` instead of failing later during evidence reconciliation.
- Preserve cached speech and its recipe when its producer is omitted. Configured incompatible
  recipes remain rejected. Embedding names are diagnostic; space and dimension decide compatibility.
- Replace `EmbeddingInputSupport.supports_embedding_input` with immutable `embedding_routes`.
  Remove `streaming_generation`: all configured answerers support `ask_stream`, with one complete
  delta when their selected backend cannot generate incrementally. These are breaking API changes.
- Separate retrieval query preparation, authoritative candidate expansion and pure score
  calculation without changing scoring policy or source leases.
- Distinguish invalid vector representations from programming `ValueError`s so corrupt persisted
  vectors retain the storage-error path without broadly wrapping internal failures.

- Expert constructor policy uses `config=MemoryConfig(...)`; backend keyword injection and
  `from_plugins` remain. Remove flattened policy keywords and repeated policy forwarding.
- Use `register_identity` for face and voice IDs; remove `register_speaker` and its speaker-only
  exception. Unknown identities consistently raise `IdentityNotFoundError`.
- Every search returns `SearchResult` with `hits` and optional `trace`; use
  `search(..., explain=True)` instead of `search_with_trace` and `TracedSearchResult`.
- `consolidate` processes one evidence set. `consolidate_pending` and CLI `consolidate-pending`
  own candidate scheduling, `max_rounds` and `idle`. Both return `ConsolidationReport`, replacing
  `deliberate` and `DeliberationReport`.
- `perceive` returns typed speech and face observations, replacing `speech` and `faces` methods.
  REST uses `/v1/perception`; MCP uses `perceive_memory`, with thirteen tools in total.
- Share completed stream-item dispatch across synchronous, asynchronous, and live capture paths;
  preserve distinct batch and streaming convenience methods and their resource lifecycles.
- Keep evidence behavior in the kernel, generic resource ownership outside the kernel, and
  structured provider-response parsing beside its adapter. Model clients no longer import
  kernel implementation.
- SDK clients own vision transport retries; remove kernel retries, fixed waits, and the
  `retried_batches` metric. The adapter classifies aborted generation as `provider_unavailable`.
- Preserve original errors when caption conversion is required. Optional enhancement failures
  retain usable native media or transcripts. Translate concrete storage failures while
  preserving programming errors and persisted-data validation.
- Surface Sentence Transformers capability-probe failures, remove repeated configuration and
  scope checks, and use one immutable capability snapshot for routing and public inspection.
- Normalize embeddings in the kernel only. Direct bundled-adapter calls return the model's
  vector scale; memory retrieval and normalized FP32 persistence retain their contract.
- Prepare lexical speech queries through the shared transcription path, preserve every text key
  for atomic-only encoders, and route each attachment against its own derived representation.
  Stored captions remain usable without a loaded producer; empty transcripts cannot satisfy a
  media fallback. Naming and identity-link reindexing prepare vectors outside the write lock,
  revalidate affected records at commit, and preserve unattributed speech turns.
- Run speech inference and embedding outside the instance write lock, atomically revalidating and
  committing speech identities with records. Settle independent captures concurrently and batch
  plain text; roll index scheduling forward as capacity becomes free. Overlap independent temporal
  routes, cloud transcription requests, and voice/face inference.
- Allow optional transcription recipe initialization until speech derivations have been persisted.
  Visual caches use typed descriptions and are recomputed under the new cache identity.
- Remove `OpenAIModels.generation_media_policy`, `generation_media_max_items` and
  `generation_gazetteer` constructor arguments. Use `EvidenceGeneration`, a selector, and
  `GazetteerEnricher`; declarative `generation` media/geography keys are preserved. Plain caption
  strings no longer cause identity writes. See [plugin architecture](docs/plugin-architecture.md).
- Require Python 3.11–3.14 and Pillow 13; update HTTPX2, Pydantic, MCP and the server/telemetry
  dependency chains. Remove CairoSVG and the CI system-library installation it needed.
- Remove the Jina-specific Sentence Transformers adaptation and `jina-omni` recipe. Local models
  use the standard Sentence Transformers loader with remote code disabled. MindBridge evaluation
  and serving require an explicit embedding configuration; existing stores keep their original vector-space
  identity and need a fresh directory when changing models.
- Operation-log decoding requires the complete current payload. Removed missing-field defaults,
  outdated benchmark object-shape fallbacks and non-protocol MCP error-field handling; older
  incomplete payloads must be replaced by re-ingesting source observations. CLI `apply --operation`
  uses the same complete payload, including explicit nulls and empty arrays.
- OpenAI completion defaults pin `gpt-5-mini-2025-08-07`, use strict JSON Schema, and explicitly
  disable provider application-state storage with `generation_store=False` (`store` in config).
  JSON-only compatible endpoints must select `json_object` explicitly. Stored formation, vision
  and consolidation recipes change; use a fresh directory and replay when adopting these defaults.
- Model operations cannot mutate host response policies or their transitive active evidence,
  including through implicit lineage replacement. Content-safety refusals keep `model_refused`
  semantics without media-removal retries, and deliberation logs never include exception chains.
- Formation explicitly separates factual source content from embedded instructions that impersonate
  runtime authority, replace observation clocks or dictate output fields.
- Relevant pull requests run answering and workflow quality gates covering formation timestamps,
  recall planning, image prompt injection, policy preservation and Chinese answering. See the
  [model quality gate](docs/operations.md#model-quality-gate) for required CI secrets and model pins.
- Answer requests identify original record and attachment modalities, including video sources
  shown as ordered stills or retained only as text after media omission.
- Local speech uses a pinned FunASR main revision with corrected speaker attribution and bounded
  CAM++ clustering work. `ncpu`, `vad_device`, and `speaker_device` expose thread and placement
  controls. The new transcription space requires a fresh directory and replay; see
  [local speech configuration](docs/configuration.md#local-speech-with-funasr).
- SQLite is authoritative for records, FP32 embeddings, metadata, and the durable index outbox.
  Zvec 0.8 provides a derived dense/full-text search projection. Writes commit SQLite before
  flushing Zvec and acknowledging outbox work; missing indexes rebuild without model calls.
- One physical `data_dir` has one live owner. POSIX directory and database permissions, fork-use
  rejection, operation leases, bounded runtime resources, and deterministic close enforce the
  embedded runtime's lifecycle. Separate memory domains require separate directories.
- SQLite schema 20 records predicate-resolution provenance. Startup accepts only the current
  schema and matching model identities. Current quantization changes rebuild from canonical
  vectors; changing an embedding or transcription space requires a fresh directory and replay.
- Public facades validate, wire, and forward to explicitly constructed kernel planes. Local
  storage uses one connection pool with table families; model and transport adapters remain
  separate, and product modules do not import benchmark code.
- Retrieval shares authoritative hydration and ranking across search, answering, compilation,
  and control. Relevance and singleton ambiguity gates can abstain; multimodal keys, lexical
  coverage, event-time filters, decay, and reinforcement contribute under explicit settings.
- Answering exposes `strict` and `best_effort` policies, bounded supplemental evidence, optional
  recall planning, and opt-in semantic query expansion. Planned reads retain their resolved time
  bounds and completeness limits. These options do not establish a general quality improvement.
- Semantic query evidence can replace the supplemental ranked tail while retaining the original
  `limit` window and matched-set limits. With query expansion enabled, strict abstentions can
  retry internally using bounded JSON evidence excerpts; streaming emits only the final round.
  Successful query routes remain eligible across rounds within the same evidence budget, so a
  later query can retain an earlier bridge fact.
- OpenAI completion slots support explicit JSON Schema outputs alongside JSON object mode.
  Known reasoning models use supported sampling controls and completion-token parameters.
  Model output is validated before durable writes; stored evidence remains untrusted input.
- Benchmark artifacts require the current schemas: evaluation results and samples use schema 22.
  PersonaMem-v3 uses compiled context; MEMLENS session labels supply grouped retrieval evidence.
  Resolved configuration, effective controls, answer policies, source questions, and cache
  identities make the actual evaluation protocol inspectable.
- Benchmark `--gen-kwargs` accepts finite temperatures from 0 through 2, default 0, with consistent
  sampling flags and a fixed run seed. Product and baseline requests share these controls, and
  effective artifacts and cache identities record them; conflicting YAML controls remain rejected.
- Wheel packages retain runtime code, typing metadata, and third-party notices while excluding
  contributor instructions.

### Fixed

- Mem-Gallery queries keep the question inside the official template instead of leaving its
  question heading followed only by a format constraint. Adapter v3 invalidates old answer caches.
- ATM judge verdicts follow the official pre-summary conversion, accepting boolean and string
  `true`. Scorer v3 separates those verdicts from numeric deterministic-list scores.
- Coarse current-clock declarations such as "It's June 2020 now" leave natural-language time
  preferences unset, preserving relevant earlier memories and explicit occurrence filters.
- Selective recall reads use spare explicit evidence budgets for the original ranked tail,
  retaining contextual follow-ups while preserving unbudgeted depth and predicate row caps.
- Sequence answers order selected evidence by event time and insertion order. ATM numeric
  instructions preserve complete dates, times and ranges.
- Judge requests omit unsupported automatic sampling controls. Judge failures retain local
  answer metrics, and saved `JudgeError` predictions can restore those metrics and be rejudged
  without repeating generation.
- Evaluation shares model capacity dynamically between ingestion and answers. Bounded idle
  SQLite pools and child-worker admission preserve progress across independent directories;
  cancellation waits for in-flight work and store cleanup. See [benchmarking](docs/benchmarking.md).
- Candidate metadata decoding releases its SQLite connection before constructing results.
  Evaluation bootstrap statistics avoid flattened sample copies and repeated paired draws while
  preserving seeded results and cluster weighting.
- Directory-owned SQLite pools share a bounded idle budget, preserving each directory's base
  allocation while reusing spare capacity for bursts. Active admission reclaims idle handles,
  and directory ownership outlasts eviction cleanup.
- Index replay, missing-index recovery, stale search-ID hydration, dead-row maintenance, and
  rejected-open metadata handling preserve SQLite authority and recovery work.
- Concurrent retrieval, deletion, asset leasing, streamed ingestion, and pooled reads preserve
  snapshot and resource-lifecycle invariants.
- Formation and control writes recheck evidence activity inside their SQLite transaction.
  Private history aliases and causal version selection prevent cross-target history confusion;
  unsupported proposals cannot grant consent or host policy authority.
- Evidence independence, proof budgets, exact excerpts, compiled character accounting, and
  consent-aware actor delivery respect their explicit bounds.
- Multilingual lexical retrieval, bounded temporal matching, singleton ambiguity handling, media
  fallback, video speech, source event times, and attachment references retain usable evidence.
- Model refusals, provider context limits, output truncation, billing failures, and unavailable
  providers map to explicit errors. Stream cleanup preserves failure classification and usage.
- Benchmark answer policies, budgets, cache identity, scorer protocols, progress, resume,
  telemetry, and resource cleanup reflect the configured run.

### Removed

- Unused generic token-F1 and ungraded NDCG helpers and the withdrawn raw-video experiment proposal.
  Dataset-specific official scorers and the historical measurement records remain available.
- Unused single-item storage wrappers and the unused media-size query; internal callers use
  the existing batch operations.
- PostgreSQL/pgvector, Celery, Redis, S3, service workers, and the service-specific APIs and
  benchmark runners from the superseded design.
- Tenant, user, run, and other implicit isolation identifiers; metadata remains application data.
- Schema migrations, historical Jina embedding spaces and index recipes, automatic re-embedding,
  pending migration state, and the pre-place content-ID fallback.
- Old result readers, permissive cache formats, task-name and CLI underscore aliases,
  `--memory-config`, `pretrained` aliases, `--model mindbridge`, and `num_fewshot`.
  Evaluation uses hyphenated options, `generation_model` overrides, and judge `model` overrides.
- `MemorySettings`, `mindbridge.sdk`, the combined `ModelBackend`, provider credential
  configuration, the custom OpenAI HTTP client, and older-SDK transport probing.
- Video-MME, EgoLifeQA, EgoMemReason, and MemEye tasks, their unused adapters and submission
  artifacts, duplicate scoring runners, the unused evaluation journal, and unused registries.
  Video-MME-v2 remains supported.
- Completed implementation audits and intermediate development notes duplicated by current
  guides and regression tests. Research protocols, measurements, and source pins stay local-only;
  third-party scorer notices remain in the benchmark package.

### Limits

- The embedded runtime does not supply distributed writers, authentication, quotas, rate limiting,
  secure erasure, or a runtime plugin registry. Hosts own deployment and access policy.
- Automatic semantic support verification, learned reranking, long-media segmentation, and
  general natural-language retraction are not established capabilities.
- Provider input limits and optional experimental policies remain explicit; consult
  [configuration](docs/configuration.md), [operations](docs/operations.md), and the
  [API references](docs/README.md#integrate) before integrating.
