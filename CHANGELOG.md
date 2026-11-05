# Changelog

Notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
SenseMem is pre-1.0; backward compatibility is not maintained. This file describes the final
release behavior; the [documentation index](docs/README.md) links to detailed contracts.

## Unreleased

This checkout targets `0.3.0` and replaces the unreleased service-oriented `0.1.0` design.

### Breaking changes

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
  see [provider configuration](docs/configuration.md#provider-fields).

### Added

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
  New adapters include VideoZeroBench and LongMemEval-V2; see [benchmarking](docs/benchmarking.md).
- OpenTelemetry stage, model, and transport measurements, plus credential-free contract checks
  and separate real-model answering/workflow gates.

### Fixed

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
- Superseded benchmark adapters, unused experiment launchers, compatibility readers, and
  intermediate implementation specifications. Private research remains local-only.

### Known limits

- One physical `data_dir` has one live owner. Hosts provide authentication, authorization, TLS,
  quotas, filesystem protection, and backup policy.
- Deletion removes live records and unreferenced media; it does not guarantee secure erasure
  from storage remnants, operation payloads, snapshots, or backups.
- Model proposals and citations are not proofs of factual support. Optional inference and
  retrieval policies require task-specific quality evaluation before a deployment relies on them.
