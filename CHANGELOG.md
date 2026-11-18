# Changelog

Release changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Review the breaking changes before upgrading a deployment. The
[documentation index](docs/README.md) describes the current supported contracts.

## Unreleased

Changes for version `0.3.0`.

### Breaking changes

- Rename Python values to distinguish storage category, semantic meaning and configuration:
  `MemoryType` → `MemoryCategory`, `MemoryKind` → `RecordKind`, `MemoryContext` →
  `RecordSemantics`, `MemoryConfig` → `MemorySettings`, `SenseMemConfig` → `MemorySpec`,
  `QueryPlan` → `QueryRepresentationPlan`, and `RecallPlan` → `RecallProgram`.
  Object construction and `from_plugins()` now take `settings=`. No compatibility aliases remain.
  Wire enum values, `memory_type` fields and the SQLite schema retain their existing meanings.
- Replace `OpenAIModels` with `OpenAIEmbedder`, `OpenAITranscriber`, and `OpenAICompletion`,
  each accepting one `client`. Caller-owned clients remain open unless ownership is transferred.
  Recipe descriptions expose a per-slot `classes` mapping in place of the single `class` field.
- Rename `AgentRunner.arun()` to `run_async()` and `ContextProof.independent_of()` to
  `structurally_independent_of()`. The latter compares certificate structure; bundle evidence
  scoring still applies the configured policy. Evidence preparation is internal to
  `EvidenceGeneration.answer()` and `stream_answer()`; remove the public `prepare()` helper.
  See [agents](docs/api/python-agents.md) and [proof scoring](docs/context-compilation.md#selected-delivery-certificates).
- Unify declarative construction under `MemorySpec`: every backend field accepts a bundled
  provider or registered factory. Remove `RegisteredMemoryConfig` and the nested `plugins` shape.
  Rename `resolve_memory_config()` to `build_memory_composition(spec, ...)` to make adapter
  construction and resource ownership explicit; close the composition or transfer its plugins
  into `Memory`. See [SDK construction](docs/api/python-sdk.md#construction).
- Require context selectors to return `ContextSelection(anchor_ids, diversify=False)`.
  One compiler admission ledger owns all delivery budgets. See the [maintainer map](docs/maintaining.md).
  Evaluation options are immutable, run statistics are isolated, and runner v28 invalidates old caches.

- Separate continuous-event intake from model assessment with `AsyncEventStream.append()`
  and `process()`. Checkpoint format v5 retains pending input as disk references with a separate
  `max_pending_bytes` budget; `append()` acknowledges durable media without waiting for a model.
  Earlier checkpoints require their original runtime to finish or replay retained source input.
- Publish `ContextBundle.proof_scoring` and `ContextProofNode.source_group_id` across SDK, REST,
  and MCP. Selected delivery now follows the instance's existing capture-group or independent-pair
  policy, including default scores that require three or more groups. Complete AND proofs and
  budgets remain required; stored scores, projection recipes, and SQLite schema do not change.
  See [selected certificates](docs/context-compilation.md#selected-delivery-certificates).
- Advance evidence projection recipes to `complete-v2` and `independent-v2`. Matching
  host-confirmed identity names remain required AND premises without counting as observations
  of a model-inferred claim. Preserve default singleton/joint scoring and full delivery proofs;
  earlier projection recipes require replay into a fresh directory. This projection recipe is
  independent of the SQLite schema.
- Advance OpenAI event segmentation to recipe v3. Requests preserve interleaved source parts
  and distinguish original text from model-derived interpretations, with explicit native media
  references. Finish pending checkpoints with their original segmenter recipe before starting
  new streams. Checkpoint format is v5. See [continuous events](docs/omni-streaming-and-interaction-memory.md).
- Require `MODEL_INFERENCE` for all automatic formation and consolidation proposals. A
  host-marked source and an exact quote no longer promote an extracted state or trait to
  `USER_STATEMENT`; remove the internal `statement_quote` wire field. Hosts can explicitly
  confirm typed claims with `apply(CONSOLIDATE)`. The changed formation recipe requires source
  replay into a fresh directory. See [claim authority](docs/typed-memory.md#state-replacement).
- Advance the remote EmbeddingGemma 2 messages recipe to `special-tokens-v3`, preserving
  interleaved text/media positions when adding retrieval prefixes. Replay sources into a fresh
  directory to replace v2 embeddings; explicit custom spaces remain caller-owned. See
  [served EmbeddingGemma 2](docs/configuration/models.md#embeddinggemma-2-through-vllm).
- Add `ContextProjection`, `ContextBundle.projections`, and text-only citation coverage to
  SDK, REST, and MCP compilation. Oversized raw media observations can deliver their complete
  labelled text while explicitly omitting native media; projections cannot satisfy evidence
  proofs. See [context compilation](docs/context-compilation.md).
- Add computation plugin roles to `Memory` and `MemoryPlugins`: query preparation, candidate
  discovery, retrieval scoring, context selection, and ordered ingestion stages. External
  candidate IDs remain subject to SQLite hydration and request scope. Stage recipes are pinned in SQLite metadata for
  nonempty stores; a changed or missing marker requires replay into a fresh directory, including
  when no stages are configured. Bundled and application
  factory declarations now share one local catalog. See [computation plugins](docs/api/python-policies.md).
- Extend typed affect formation and stored contexts with target/elicitor relations and exact
  reporting witnesses. Formation proposals may carry batch-local references; compiled
  attribution requires complete scoped affirmation/denial evidence under the ordinary budget.
  The changed prompt/schema recipe has not completed real-model attribution qualification.
  See [affective memory](docs/affective-memory.md).
- Default formation to eight causal history rows within 6,000 content characters. Existing
  text can now reach the formation provider on later writes, increasing input scope and cost;
  set `formation_history_max_rows: 0` to opt out. Active-witness and naming-citation rules still
  apply. See [formation history](docs/configuration/lifecycle.md#bounded-formation-history).
- Extend Python, REST and MCP context compilation with temporary `supplements` and include
  `supplements` / `omitted_supplements` in context bundles. See [context compilation](docs/context-compilation.md).
- Default local EmbeddingGemma 2 to BF16 on CUDA GPUs with native support. BF16 has a separate
  vector-space identity; PyAV audio decoding changes the local recipe and requires source replay. See
  [model setup](docs/configuration/models.md#embeddinggemma-2).
- Use vLLM's `/score` endpoint with `queries`/`documents` for remote reranking, avoiding
  responses that echo full base64 media. See [remote reranking](docs/configuration/models.md#remote-vllm-service).
- Remove BEAM, CL-Bench, WorldMemArena, Video-MME-v2, and OpenEQA tasks from
  `sensemem-bench eval`. See the [supported tasks](docs/benchmarks/tasks.md).
- Rename the distribution, import package, commands, configuration classes, exceptions, and
  telemetry to SenseMem: `sensemem`, `sensemem-bench`, `MemorySpec`, `SenseMemError`, and
  `SENSEMEM_*` benchmark variables. See [upgrading](docs/deployment.md#upgrading-to-sensemem).
- Require Python 3.11–3.14 and Zvec 0.8.1 or newer within 0.8.x. The shared local/reranking
  environment uses CUDA 13 PyTorch builds on Linux x86-64; see [configuration](docs/configuration.md).
- Accept only SQLite schema 25 and current stored representation identities. Remove redundant
  selector and search-index outbox indexes already covered by primary keys. Earlier directories
  require their original runtime to read source data and re-ingest into a fresh directory; see
  [representation changes](docs/operations.md#changing-the-runtime-or-stored-representations).
  Changing current index quantization rebuilds Zvec from stored FP32 vectors.
- Preserve ordered text/media parts in records and version the changed embedding recipes.
  Source replay into a separate directory can rebuild retained observations under a new model.
- Keep stream interpretations in a distinct `model_interpretation` content part, excluding them
  from original-source replay and host-statement authorization. Cross-location event summaries
  remain whole-event metadata instead of being asserted as facts of every local part.
- Accept only continuous-event checkpoint format v5. Finish v1–v4 checkpoints with their original
  runtime before upgrading; the new interpretation and context-aware event identities have no migration.
- Version OpenAI formation and consolidation recipes with `source-parts-v2`; current input,
  read-only history and consolidation retain derived content roles in their wire payloads.
- Default automatic answer reinforcement to `False`; hosts can reinforce confirmed useful
  outcomes or explicitly opt into backend-declared answer feedback.
- Remove implicit face/voice co-occurrence linking and `identity_link_min_assets` /
  `link_identities` controls. Host-authored identity merges confirm bindings, and unlink decisions
  persist across recognition and restart.
- Pass typed `query_failure` context to consolidation backends and report unresolved queries.
  Custom backends must accept the updated [consolidation protocol](docs/api/python-backends.md).
- Use predicate registry v2 for reviewed current-drink aliases; grounded host statements can
  update a preferred drink without leaving its previous value current.
- Use evaluation artifact schema 25 with `sensemem` arm names and `sensemem_version`.
  Keep text-only projection evidence separate from complete source coverage and original-media
  delivery, including in response caches.
  Earlier artifacts remain records of their original runs and are not current runner inputs.
- Retire the original `longmemeval-s` dialogue benchmark and scorer. Use
  `longmemeval-v2-small` or `longmemeval-v2-medium` for agent-trajectory evaluation; their input
  protocols and scores differ from the retired task.
- Consolidate SDK policy under `settings=MemorySettings(...)`. Search returns `SearchResult`;
  `perceive` returns speech, face, and coverage collections; `consolidate` and
  `consolidate_pending` return `ConsolidationReport`. See the [SDK reference](docs/api/python-sdk.md).
- Default OpenAI completion slots to strict JSON Schema and `store=False`. JSON-only servers
  must explicitly select `json_object`. Changed model/recipe identities can require source replay;
  see [provider configuration](docs/configuration/providers.md).

### Added

- Connected demo views for retained original media, evolving typed interpretations and their
  native operation history, with explicit confirmation, correction, reversible forgetting,
  ASR-driven speculative recall and optional visual recall. The example explicitly configures
  formation and consolidation; original-media review uses the native grounded answer path.
  See the [guided experiences and qualification limits](docs/live-demo.md).
- Explicit typed tool registration shared by fixed pipelines and an optional provider-independent
  agent loop, with an OpenAI SDK adapter and asynchronous temporary context supply. See
  [plugin composition](docs/plugin-architecture.md), [tools](docs/api/python-tools.md) and
  [agents](docs/api/python-agents.md).
- Host-registered configuration factories for every backend slot and arbitrary tools, plus
  reusable perception/identity/search/context and transcription tool adapters. See
  [application factories](docs/configuration/plugins.md).
- Verified original media reads through the SDK, REST, and an MCP resource template; cursor
  pagination for deferred-capture backlogs; and goal-directed context recall with bounded plans.
- Automatic semantic event segmentation for continuous audio/video through `AsyncEventStream`,
  preserving original chunks, event summaries, and complete event intervals. See
  [automatic event segmentation](docs/omni-streaming-and-interaction-memory.md#automatic-event-segmentation).
- A task-oriented documentation map, a minimal getting-started path, and consolidated
  [benchmark results](docs/benchmarks/results.md).
- EmbeddingGemma 2 precision selection and selective vision/audio loading through SDK and
  declarative configuration, with an opt-in pinned-model parity check. See
  [loading quality](docs/model-quality.md#check-embeddinggemma-2-loading).
- Optional image-source context and native evidence operations for grounded answers, including
  source selection, distinct counting and witnessed arithmetic. See
  [provider configuration](docs/configuration/providers.md).
- Native MOSAIC evaluation with pinned data and judge prompts, plus an EmbeddingGemma 2 /
  Qwen Omni [reproduction recipe and results](docs/benchmarks/mosaic-native.md).
- Bounded concurrent HTTP reranking batches, local video-frame caching and media preparation
  during inference. See [reranker configuration](docs/configuration/models.md#semantic-reranking).
- Optional `--no-search-replay` / `benchmark.run.search_replay: false` for evaluations that omit
  standalone search-latency measurement. The choice is recorded in result metadata.
- Ordered text, image, video, audio, and mixed observations through synchronous and asynchronous
  SDKs, with immutable media, content-addressed IDs, and explicit model capabilities.
- Deferred `capture`/`settle` ingestion, pending-work inspection, completed-observation streams,
  speculative recall, and audio/vision capture reducers with explicit finality.
- Hybrid retrieval with memory roles, event intervals, bitemporal and spatial scopes, optional
  decay, reinforcement, and bounded explanations.
- Grounded answering and answer streaming, explicit abstention policies, optional recall planning
  and query expansion, and validated structured answer rendering.
- Evidence-closed context compilation, conflicts and unknowns, exact raw-text excerpts, compact
  Python presentation, and optional selected delivery certificates.
- Typed formation and a memory control plane for consolidation, correction, cognitive forgetting,
  operation history, outcomes, and rollback. Response policies require host authorization.
- Speech and face observations, reversible identity linking and naming, host-recorded consent,
  subject export, biometric erasure, and explicit retention passes.
- Local Sentence Transformers and Qwen3-VL rerankers; native/derived modality routing;
  provider-neutral media selection; optional pinned offline map enrichment for answers.
- SDK-aligned REST, thirteen default MCP tools with authority switches, and a JSON product CLI.
- Dataset evaluation with isolated stores, baseline arms, official or explicitly identified
  scorers, response caches, resume, quality diagnostics, and performance comparisons.
  See [benchmarking](docs/benchmarking.md) for tasks and protocol boundaries.
- OpenTelemetry stage, model, and transport measurements, plus credential-free contract checks
  and separate real-model answering/workflow gates.

### Fixed

- Reject incomplete operation-log payloads instead of filling missing effects with defaults.
  The current schema and payload fields are unchanged. See [operations](docs/operations.md#changing-the-runtime-or-stored-representations).
- Align control-plane benchmark fixtures with model-inferred state provenance and identify
  derived claims through their source evidence instead of retrieval ranking.
- Upgrade Zvec to 0.8.1 for corrected RaBitQ graph-query scoring and reduced memory on eligible
  FP32 HNSW builds. Current index recipes remain unchanged.
- Measure local-index recall against independent original FP32 cosine scores from SQLite,
  including quantization error, and record dependency versions for local index and SDK load
  measurements. See [local benchmarks](docs/benchmarks/local.md).
- Retain original-text subject labels as unbound formation candidates when bounded history
  contains a matching registered name. Stable identity binding still requires the cited naming
  premise; same-name identities and unbound claims keep separate lineage keys.
- Preserve refreshed capture work when an older settlement or formation call completes;
  stale work cannot overwrite refreshed content or acknowledge its pending task.
- Defer vision-stream descriptions until settlement in capture mode, keeping original frames
  durable even when the description backend is unavailable.
- Preserve the requested memory type in planned neighbor reads and related-evidence expansion.
- Return perception coverage through REST and align local/remote CLI pending-capture fields.
- Complete PersonaMem-v3 evaluation with proactive decisions, geographic-shift scoring,
  threaded repetition clusters, novel-suggestion judging, lifecycle pairing, and warning/foil
  summaries. Report the micro score only with complete selected scoring coverage; preserve
  ranking dialogue, calendar entry IDs, and selected-persona history loading.
  See [PersonaMem-v3 evaluation](docs/benchmarks/tasks.md#personamem-v3).
- Remove the ES-MemEval task, downloader registration, prompts, and scorers.
- Preserve explicitly configured evaluation reader modalities and routes when loading native
  media; require ModelScope Hub 0.6.1+ so local speech resolves its pinned model commits.
- Skip ASR for videos without an audio track, retaining their visual evidence and mixed-batch
  speech alignment. The local extra includes PyAV for stream inspection.
- Split oversized MOSAIC text chunks into separate SDK records with the same source ID, retaining
  their complete text within the SDK's total per-record limit.
- Serialize modality sets deterministically so evaluation caches and ingest checkpoints work
  across processes and interpreter hash seeds.
- Restrict image metadata warning escalation to Pillow decompression warnings so concurrent
  speech decoding is not failed by unrelated deprecations.
- Cache successful evaluation answers as they complete so interruption does not discard a
  completed answer while other questions in the same unit are still running.
- Project time-filter columns in Zvec full-text queries to avoid Arrow batch schema errors on
  persisted indexes while retaining the same temporal predicates and rankings.
- Recognize TokenHub SSE error `502001` as a retryable upstream failure without exposing its body.
- Preserve event identity, requested list membership, and source-relative time precision in the
  optional concise answer style.
- Preserve SQLite authority through index failures, missing-index rebuilds, stale index IDs,
  concurrent writes, media leasing, and bounded resource cleanup.
- Deliver derived answer evidence only with its complete eligible source closure. Packing,
  media omission, and provider fallbacks preserve that obligation.
- Revalidate formation and control evidence at commit. Reject self-citation, unauthorized
  response-policy changes, inferred consent, and stale or inconsistent operations.
- Preserve current identity attribution when naming, splitting, erasing, correcting, or
  withdrawing supporting evidence.
- Classify provider refusals, context overflow, output truncation, exhausted quota, overload,
  and transport failures consistently without exposing provider response bodies.
- Retain failed evaluation rows in their denominators, preserve deterministic scores after judge
  failures, and distinguish cached, resumed, partial, and non-comparable measurements.

### Removed

- PostgreSQL/pgvector, Celery, Redis, S3, service workers, logical tenant/request scoping,
  automatic schema migration, and obsolete SDK/CLI aliases.
- One-off policy replay, corroboration ablation, investigation diagnostics, and saved ICM rejudging from
  distributed benchmarks. Supported SDK behavior remains covered by focused regression tests.

### Deployment notes

- One physical `data_dir` has one live owner. Hosts provide authentication, authorization, TLS,
  quotas, filesystem protection, and backup policy.
- Deletion removes live records and unreferenced media; it does not guarantee secure erasure
  from storage remnants, operation payloads, snapshots, or backups.
- Qualify model and retrieval choices for your application with the
  [model quality gate](docs/model-quality.md) and [benchmark harness](docs/benchmarking.md).
