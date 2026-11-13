# SenseMem documentation

Build agents that remember across sessions and use the right evidence for the current task.
SenseMem connects durable multimodal memory, evolving assertions, retrieval, and context
construction through one runtime.

## Start here

1. **[Quick start](quickstart.md):** install SenseMem, store a fact, and retrieve it.
2. **[Core concepts](concepts.md):** understand observations, evidence, models, and a memory domain.
3. **[Configuration](configuration.md):** choose models and enable the capabilities you need.
4. **[Integrate](api/python-sdk.md):** connect your application through the SDK, REST, MCP, or CLI.

Read [product capabilities](product-capabilities.md) for use cases and the complete capability map.
These guides describe SenseMem 0.3.0; the [changelog](../CHANGELOG.md#unreleased) lists release changes.

## Choose by task

| I want to… | Read |
| --- | --- |
| Give an agent useful context within a prompt budget | [Context compilation](context-compilation.md) |
| Remember events, preferences, and changing facts | [Memory types, time, and decay](memory-types-time-and-decay.md) · [Typed memory](typed-memory.md) |
| Consolidate, correct, forget, or undo a change | [Memory management](memory-management.md) |
| Capture live audio, vision, or interaction | [Streaming and interaction memory](omni-streaming-and-interaction-memory.md) |
| Connect a local WeMM endpoint | [Local WeMM integration](use-local-wemm.md) |
| Serve EmbeddingGemma 2 through vLLM | [Model server](deployment.md#embeddinggemma-2-model-server) · [Client configuration](configuration/models.md#embeddinggemma-2-through-vllm) |
| Add text or multimodal reranking | [Semantic reranking](configuration/models.md#semantic-reranking) |
| Choose a working evaluation configuration | [Example catalog](examples/README.md) |
| Run the three promotional demos and prepare English videos | [Promotional demos](examples/promo-demos.md) |
| See measured results or evaluate my own setup | [Benchmark results](benchmarks/results.md) · [Benchmarking](benchmarking.md) |
| Deploy, back up, or recover memory | [Deployment](deployment.md) · [Operations](operations.md) |
| Diagnose an error or empty search | [Troubleshooting](troubleshooting.md) |

## Understand memory

- [Product capabilities](product-capabilities.md) — what you can build and which capabilities to enable.
- [Core concepts](concepts.md) — content, records, evidence, model capabilities, and directory ownership.
- [Memory types, time, and decay](memory-types-time-and-decay.md) — cognitive roles, event time,
  historical validity, spatial scope, ranking decay, and reinforcement.
- [Typed memory and evidence](typed-memory.md) — entities, events, states, support clauses,
  conflict handling, and naming.
- [Affective memory](affective-memory.md) — sourced affect cues, corroborated traits, and
  host-authorized response guidance.
- [Memory management](memory-management.md) — deliberate maintenance, scheduling, audit, and rollback.
- [Context compilation](context-compilation.md) — structured evidence, budgets, source closure,
  conflicts, excerpts, and rendering.
- [Streaming and interaction memory](omni-streaming-and-interaction-memory.md) — completed
  observations, speculative recall, and explicit update/final/cancel lifecycles.

## Configure and integrate

- [Configuration](configuration.md) — installation, model slots, composition, and resource ownership.
  Detailed references: [provider fields](configuration/providers.md),
  [model setup](configuration/models.md), [runtime settings](configuration/settings.md), and
  [formation, consolidation, and retention](configuration/lifecycle.md).
- [SDK reference](api/python-sdk.md) — construction, lifecycle, content, errors, and API navigation.
  Detailed references: [memory operations](api/python-memory.md),
  [identity and data lifecycle](api/python-identity.md), [control operations](api/python-control.md),
  [async and streams](api/python-streams.md), [public values](api/python-values.md),
  [backend protocols](api/python-backends.md), and [bundled adapters](api/python-adapters.md).
- [REST API](api/rest.md) — routes under `/v1`, request/response shapes, SSE, errors, and limits.
- [MCP tools](api/mcp.md) — thirteen tools, authority switches, schemas, and transport behavior.
- [Command line](api/cli.md) — local and remote operations, JSON input/output, diagnostics, and exit codes.
- [Local WeMM integration](use-local-wemm.md) — a multimodal embedding endpoint with optional speech
  and formation.
- [Example catalog](examples/README.md) — prerequisites and customization for every committed YAML.

## Deploy and operate

- [Deployment](deployment.md) — embedded, REST, and MCP topologies and model-service placement.
- [Operations](operations.md) — lifecycle, capture backlog, backup, restore, index maintenance, and retention.
- [Runtime limits](runtime-limits.md) — admission budgets, workers, cancellation, and sizing.
- [Observability](observability.md) — traces, stage timing, tokens, provenance, and diagnostics.
- [Troubleshooting](troubleshooting.md) — concrete actions for startup, retrieval, media, and model errors.
- [Security](../SECURITY.md) — private reporting, trust boundaries, data handling, and deployment controls.

## Evaluate quality

- [Benchmark results](benchmarks/results.md) — measured multimodal configurations and scoring scope.
- [Benchmarking](benchmarking.md) — configure, run, interpret, and compare an evaluation.
- [Benchmark tasks](benchmarks/tasks.md) — workload selection, data requirements, and scoring protocols.
- [Evaluation reference](benchmarks/reference.md) — run controls, baseline arms, diagnostics,
  performance, resume, and exports.
- [Local measurements](benchmarks/local.md) — index storage, SDK concurrency, and consolidation.
- [MOSAIC native multimodal recipe](benchmarks/mosaic-native.md) — service setup and full-run commands.
- [Model quality gate](model-quality.md) — focused real-model checks and CI configuration.
- [Agent Memory Leaderboard](agent-memory-leaderboard.md) — participant Add/Search hosting.

## Understand and extend the design

- [Architecture](architecture.md) — code layout, storage authority, write/retrieval paths, and concurrency.
- [Design principles](design-principles.md) — Smarter / Swifter / Stronger and the criteria behind the design.
- [Context OS direction](context-os.md) — the lifecycle connecting capture, memory management,
  and context compilation.
- [Plugin architecture](plugin-architecture.md) — typed capabilities, kernel responsibilities,
  and extension requirements.

For development setup, documentation conventions, and quality gates, read
[Contributing](../CONTRIBUTING.md).
