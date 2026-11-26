# SenseMem documentation

Build agents that remember across sessions and use the right evidence for the current task. SenseMem connects durable multimodal memory, evolving assertions, retrieval, and context construction through one runtime.

## Start here

1. **[Quick start](quickstart.md):** install SenseMem, store a fact, and retrieve it.
2. **[Core concepts](concepts.md):** understand observations, evidence, models, and a memory domain.
3. **[Configuration](configuration.md):** choose models and enable the capabilities you need.
4. **[Integrate](api/python-sdk.md):** connect your application through the SDK, REST, MCP, or CLI.

## Choose by task

| I want to… | Read |
| --- | --- |
| Replay timed audio/video, form events, and play recalled media | [Best practices notebook](../examples/best_practices.ipynb) |
| Give an agent useful context within a prompt budget | [Context compilation](context-compilation.md) |
| Remember events, preferences, and changing facts | [Memory types, time, and decay](memory-types-time-and-decay.md) · [Typed memory](typed-memory.md) |
| Consolidate, correct, forget, or undo a change | [Memory management](memory-management.md) |
| Capture live audio, vision, or interaction | [Streaming and interaction memory](omni-streaming-and-interaction-memory.md) |
| Try a continuous webcam and voice conversation | [Local audio/video demo](live-demo.md) |
| Share capabilities between a pipeline and a tool-using agent | [Tools](api/python-tools.md) · [Agents](api/python-agents.md) · [Plugin factories](configuration/plugins.md) |
| Serve EmbeddingGemma 2 through vLLM | [Model server](deployment.md#embeddinggemma-2-model-server) · [Client configuration](configuration/models.md#embeddinggemma-2-through-vllm) |
| Add text or multimodal reranking | [Semantic reranking](configuration/models.md#semantic-reranking) |
| Evaluate my own setup on benchmarks | [Benchmarking](benchmarking.md) · [Evaluation template](examples/eval.example.yaml) |
| Deploy, back up, or recover memory | [Deployment](deployment.md) · [Operations](operations.md) |
| Diagnose an error or empty search | [Troubleshooting](troubleshooting.md) |

## Understand memory

- [Core concepts](concepts.md) — content, records, evidence, model capabilities, and directory ownership.
- [Memory types, time, and decay](memory-types-time-and-decay.md) — cognitive roles, event time, historical validity, spatial scope, ranking decay, and reinforcement.
- [Typed memory and evidence](typed-memory.md) — entities, events, states, support clauses, conflict handling, and naming.
- [Affective memory](affective-memory.md) — sourced affect cues, corroborated traits, and host-authorized response guidance.
- [Memory management](memory-management.md) — deliberate maintenance, scheduling, audit, and rollback.
- [Context compilation](context-compilation.md) — structured evidence, budgets, source closure, conflicts, excerpts, and rendering.
- [Streaming and interaction memory](omni-streaming-and-interaction-memory.md) — completed observations, speculative recall, and explicit update/final/cancel lifecycles.

## Configure and integrate

- [Configuration](configuration.md) — installation, model slots, composition, and resource ownership. Detailed references: [provider fields](configuration/providers.md), [model setup](configuration/models.md), [runtime settings](configuration/settings.md), and [formation, consolidation, and retention](configuration/lifecycle.md).
- [Application plugin factories](configuration/plugins.md) — configure third-party backends, extension rules, and tools.
- [Computation plugins](api/python-policies.md) — replace query planning, candidate discovery, retrieval scoring, context selection, and ordered ingestion stages.
- [SDK reference](api/python-sdk.md) — construction, lifecycle, content, errors, and API navigation. Detailed references: [memory operations](api/python-memory.md), [identity and data lifecycle](api/python-identity.md), [control operations](api/python-control.md), [async and streams](api/python-streams.md), [public values](api/python-values.md), [backend protocols](api/python-backends.md), and [bundled adapters](api/python-adapters.md).
- [Python tools](api/python-tools.md) — register typed computations and reuse existing perception.
- [Optional agents](api/python-agents.md) — choose a core model and compose tool calls with async context.
- [REST API](api/rest.md) — routes under `/v1`, request/response shapes, SSE, errors, and limits.
- [MCP tools](api/mcp.md) — thirteen tools, authority switches, schemas, and transport behavior.
- [Command line](api/cli.md) — local and remote operations, JSON input/output, diagnostics, and exit codes.

## Deploy and operate

- [Deployment](deployment.md) — embedded, REST, and MCP topologies and model-service placement.
- [Operations](operations.md) — lifecycle, capture backlog, backup, restore, index maintenance, and retention.
- [Runtime limits](runtime-limits.md) — admission budgets, workers, cancellation, and sizing.
- [Observability](observability.md) — traces, stage timing, tokens, provenance, and diagnostics.
- [Troubleshooting](troubleshooting.md) — concrete actions for startup, retrieval, media, and model errors.
- [Security](../SECURITY.md) — private reporting, trust boundaries, data handling, and deployment controls.

## Evaluate quality

- [Benchmarking](benchmarking.md) — configure, run, interpret, and compare an evaluation.
- [Benchmark tasks](benchmarks/tasks.md) — workload selection, data requirements, and scoring protocols.
- [Evaluation reference](benchmarks/reference.md) — run controls, baseline arms, diagnostics, performance, resume, and exports.
- [Local measurements](benchmarks/local.md) — index storage, SDK concurrency, and consolidation.
- [Model quality gate](model-quality.md) — focused real-model checks and CI configuration.

## Understand and extend the runtime

- [Architecture](architecture.md) — code layout, storage authority, write/retrieval paths, and concurrency.
- [Maintain prompts](prompts.md) — runtime components, official benchmark wording, and validation.

For development setup, documentation conventions, and quality gates, read [Contributing](../CONTRIBUTING.md).
