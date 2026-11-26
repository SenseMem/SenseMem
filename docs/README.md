# SenseMem documentation

Build agents that remember across sessions and use the right evidence for the current task.
SenseMem connects durable multimodal memory, evolving assertions, retrieval, and context
construction through one runtime.

## Start here

1. Use the [quick start](quickstart.md) to install SenseMem, store a fact, and retrieve it.
2. Read [core concepts](concepts.md) to understand observations, evidence, and directory ownership.
3. Choose models and optional capabilities through [configuration](configuration.md).
4. Pick an [interface](#connect-through-your-preferred-interface) for your application.

These guides describe SenseMem 0.3.0. See the [changelog](../CHANGELOG.md#unreleased) before upgrading.

## Choose by task

| I want to… | Read |
| --- | --- |
| Replay timed audio/video, form events, and play recalled media | [Example catalog](examples/README.md) |
| Give an agent useful context within a prompt budget | [Context compilation](context-compilation.md) |
| Remember events, preferences, and changing facts | [Memory types, time, and decay](memory-types-time-and-decay.md) · [Typed memory](typed-memory.md) |
| Consolidate, correct, forget, or undo a change | [Memory management](memory-management.md) |
| Capture live audio, vision, or interaction | [Streaming and interaction memory](omni-streaming-and-interaction-memory.md) |
| Try a continuous webcam and voice conversation | [Local audio/video demo](live-demo.md) |
| Share capabilities between a pipeline and a tool-using agent | [Tools](api/python-tools.md) · [Agents](api/python-agents.md) · [Plugin factories](configuration/plugins.md) |
| Connect a local WeMM endpoint | [WeMM setup](configuration/models.md#local-wemm-embedding-service) |
| Serve EmbeddingGemma 2 through vLLM | [Model server](deployment.md#embeddinggemma-2-model-server) · [Client configuration](configuration/models.md#embeddinggemma-2-through-vllm) |
| Add text or multimodal reranking | [Semantic reranking](configuration/models.md#semantic-reranking) |
| Choose a working evaluation configuration | [Example catalog](examples/README.md) |
| See measured results or evaluate my own setup | [Benchmark results](benchmarks/results.md) · [Benchmarking](benchmarking.md) |
| Deploy, back up, or recover memory | [Deployment](deployment.md) · [Operations](operations.md) |
| Diagnose an error or empty search | [Troubleshooting](troubleshooting.md) |

## Choose capabilities

An embedding backend enables durable capture and retrieval. Add optional backends for the
operations your application needs. Modality coverage follows the configured backends, and the
kernel validates proposed changes before committing them.

| Capability | What it enables | Guide |
| --- | --- | --- |
| Durable capture | Immediate, batched, streamed, and deferred writes with retained original media | [Memory operations](api/python-memory.md) |
| Retrieval | Dense, lexical, or hybrid search with filters, optional reranking, and diagnostics | [Retrieval settings](configuration/settings.md#retrieval-and-grounding) |
| Context compilation | Evidence with item, character, and media budgets, conflicts, and unknowns | [Context compilation](context-compilation.md) |
| Grounded answers | Buffered or streamed generation with evidence and abstention | [Answering](api/python-memory.md#ask) |
| Typed memory | Model-proposed entities, events, states, relations, affect, and traits | [Typed memory](typed-memory.md) |
| Memory maintenance | Explicit consolidation, correction, reinforcement, reversible forgetting, and audit history | [Memory management](memory-management.md) |
| Time and place | Event intervals, valid and known time, symbolic places, and same-frame spatial filters | [Time and decay](memory-types-time-and-decay.md) |
| Local identities | Voice and face observations, sourced naming, host-confirmed links, and erasure | [Identity API](api/python-identity.md) |
| Continuous input | Automatic complete-event segmentation or host-supplied finality for audio and video | [Streaming](omni-streaming-and-interaction-memory.md) |
| Data lifecycle | Host-recorded consent, subject export, and explicit retention passes | [Data-subject operations](api/python-identity.md#data-subject-rights) |

## Connect through your preferred interface

| Surface | Use | Reference |
| --- | --- | --- |
| Python SDK | Synchronous and asynchronous integration with the full memory lifecycle | [Python SDK](api/python-sdk.md) |
| REST | Share one owner over HTTP under `/v1` | [REST API](api/rest.md) |
| MCP | Give an agent host up to thirteen tools with configurable authority | [MCP tools](api/mcp.md) |
| CLI | Operate locally or call an existing REST owner | [CLI](api/cli.md) |

`Memory.capabilities`, REST `GET /healthz`, MCP instructions, and `sensemem doctor` describe the
configured capabilities as exposed through each interface. The [memory control plane](architecture.md#public-and-trust-boundaries)
runs through the SDK in the owner process. The local CLI can call that SDK directly.

## Understand memory

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
- [Application plugin factories](configuration/plugins.md) — configure third-party backends and tools.
- [Computation plugins](api/python-policies.md) — replace retrieval scoring and context selection,
  and compose ordered ingestion stages around the consistency kernel.
- [SDK reference](api/python-sdk.md) — construction, lifecycle, content, errors, and API navigation.
  Detailed references: [memory operations](api/python-memory.md),
  [identity and data lifecycle](api/python-identity.md), [control operations](api/python-control.md),
  [async and streams](api/python-streams.md), [public values](api/python-values.md),
  [backend protocols](api/python-backends.md), and [bundled adapters](api/python-adapters.md).
- [Python tools](api/python-tools.md) — register typed computations and reuse existing perception.
- [Optional agents](api/python-agents.md) — choose a core model and compose tool calls with async context.
- [REST API](api/rest.md) — routes under `/v1`, request/response shapes, SSE, errors, and limits.
- [MCP tools](api/mcp.md) — thirteen tools, authority switches, schemas, and transport behavior.
- [Command line](api/cli.md) — local and remote operations, JSON input/output, diagnostics, and exit codes.
- [Example catalog](examples/README.md) — notebook setup and configuration recipes.

## Deploy and operate

- [Deployment](deployment.md) — embedded, REST, and MCP topologies and model-service placement.
- [Operations](operations.md) — lifecycle, capture backlog, backup, restore, index maintenance, and retention.
- [Runtime limits](operations.md#admission-limits) — admission budgets, workers, cancellation, and sizing.
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
- [Plugin admission](architecture.md#admission-rule) — extension requirements for new kernel capabilities.
- [Prompt contracts](model-quality.md#prompt-contracts) — runtime prompt ownership and quality checks.

For development setup, documentation conventions, and quality gates, read
[Contributing](../CONTRIBUTING.md).
