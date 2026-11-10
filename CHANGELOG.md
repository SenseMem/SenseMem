# Changelog

Release changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Review the breaking changes before upgrading a deployment. The
[documentation index](docs/README.md) describes the current supported contracts.

## Unreleased

Changes for version `0.3.0`.

### Breaking changes

- Default local EmbeddingGemma 2 to BF16 on CUDA GPUs with native support. BF16 has a separate
  vector-space identity; PyAV audio decoding changes the local recipe and requires source replay. See
  [model setup](docs/configuration/models.md#embeddinggemma-2).
- Use vLLM's `/score` endpoint with `queries`/`documents` for remote reranking, avoiding
  responses that echo full base64 media. See [remote reranking](docs/configuration/models.md#remote-vllm-service).
- Remove BEAM, CL-Bench, WorldMemArena, Video-MME-v2, and OpenEQA tasks from
  `sensemem-bench eval`. See the [supported tasks](docs/benchmarks/tasks.md).
- Rename the distribution, import package, commands, configuration classes, exceptions, and
  telemetry to SenseMem: `sensemem`, `sensemem-bench`, `SenseMemConfig`, `SenseMemError`, and
  `SENSEMEM_*` benchmark variables. See [upgrading](docs/deployment.md#upgrading-to-sensemem).
- Require Python 3.11–3.14 and Zvec 0.8. The shared local/reranking environment uses CUDA 13
  PyTorch builds on Linux x86-64; see [configuration](docs/configuration.md).
- Accept only SQLite schema 21 and current stored representation identities. Earlier directories
  require their original runtime to read source data and re-ingest into a fresh directory.
  Changing current index quantization rebuilds Zvec from stored FP32 vectors.
- Use evaluation artifact schema 23 with `sensemem` arm names and `sensemem_version`.
  Earlier artifacts remain records of their original runs and are not current runner inputs.
- Retire the original `longmemeval-s` dialogue benchmark and scorer. Use
  `longmemeval-v2-small` or `longmemeval-v2-medium` for agent-trajectory evaluation; their input
  protocols and scores differ from the retired task.
- Consolidate SDK policy under `config=MemoryConfig(...)`. Search returns `SearchResult`;
  `perceive` returns speech, face, and coverage collections; `consolidate` and
  `consolidate_pending` return `ConsolidationReport`. See the [SDK reference](docs/api/python-sdk.md).
- Default OpenAI completion slots to strict JSON Schema and `store=False`. JSON-only servers
  must explicitly select `json_object`. Changed model/recipe identities can require source replay;
  see [provider configuration](docs/configuration/providers.md).

### Added

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
  scorers, response caches, resume, quality diagnostics, rejudging, and performance comparisons.
  See [benchmarking](docs/benchmarking.md) for tasks and protocol boundaries.
- OpenTelemetry stage, model, and transport measurements, plus credential-free contract checks
  and separate real-model answering/workflow gates.

### Fixed

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

### Deployment notes

- One physical `data_dir` has one live owner. Hosts provide authentication, authorization, TLS,
  quotas, filesystem protection, and backup policy.
- Deletion removes live records and unreferenced media; it does not guarantee secure erasure
  from storage remnants, operation payloads, snapshots, or backups.
- Qualify model and retrieval choices for your application with the
  [model quality gate](docs/model-quality.md) and [benchmark harness](docs/benchmarking.md).
