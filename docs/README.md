# SenseMem documentation

SenseMem is a **unified context and memory layer for agents, across devices, modalities, and models**.
It connects incoming context, evolving memory, task-specific recall, and context construction.
Smarter / Swifter / Stronger guide its [product goals](design-principles.md#smarter--swifter--stronger).

Start with the quick start for a working example, use the references for exact contracts, and
follow the operational guides for deployment and recovery.

These guides describe SenseMem `0.3.0` on Python 3.11–3.14. See the
[changelog](../CHANGELOG.md#unreleased) for release changes. Direction pages identify design goals
separately from supported behavior.

## Choose a path

| Goal | Start here | Continue with |
| --- | --- | --- |
| Understand what SenseMem can do | [Product capabilities](product-capabilities.md) | [Design principles](design-principles.md) |
| Try SenseMem end to end | [Quick start](quickstart.md) | [Core concepts](concepts.md) |
| Build a Python integration | [Configuration](configuration.md) | [Python SDK](api/python-sdk.md) |
| Rerank text, images and videos | [Semantic reranking](configuration.md#semantic-reranking) | [Python SDK](api/python-sdk.md) |
| Use a local WeMM embedding service | [Local WeMM deployment](use-local-wemm.md) | [Configuration](configuration.md) |
| Use EmbeddingGemma 2 through vLLM | [Model server](deployment.md#embeddinggemma-2-model-server) | [Configuration](configuration.md#embeddinggemma-2-through-vllm) |
| Give an agent a bounded context view | [Context compilation](context-compilation.md) | [Python SDK](api/python-sdk.md) |
| Expose memory to another process | [REST](api/rest.md), [MCP](api/mcp.md), or [CLI](api/cli.md) | [Deployment](deployment.md) |
| Run a durable instance | [Deployment](deployment.md) | [Operations](operations.md), [architecture](architecture.md), and [troubleshooting](troubleshooting.md) |
| Evaluate memory quality | [Benchmarking](benchmarking.md) | [Example evaluation configuration](examples/eval.example.yaml) |
| Understand or extend the design | [Design principles](design-principles.md) | [Plugin architecture](plugin-architecture.md) |
| Select a runnable configuration | [Example catalog](examples/README.md) | [Configuration](configuration.md) and [benchmarking](benchmarking.md) |
| Follow where the product is going | [Context OS direction](context-os.md) | [Design principles](design-principles.md) |

## Learn

1. [Product capabilities](product-capabilities.md) — see the implemented product surface, use
   cases, boundaries, and extension model.
2. [Quick start](quickstart.md) — install SenseMem and exercise its core capabilities.
3. [Core concepts](concepts.md) — understand records, content, retrieval, and directory ownership.
4. [Configuration](configuration.md) — select bundled adapters or inject application backends.
5. [Memory types, time, and decay](memory-types-time-and-decay.md) — control cognitive role and
   temporal ranking.
6. [Omni streaming and interaction memory](omni-streaming-and-interaction-memory.md) — ingest
   completed observations and derive grounded interaction records.
7. [Context compilation](context-compilation.md) — compile a bounded, structured context bundle for
   one goal instead of a flat hit list.

## Integrate

- [Python SDK](api/python-sdk.md) — complete `sensemem` root-import contract.
- [REST API](api/rest.md) — `/v1` requests, responses, errors, and limits.
- [MCP tools](api/mcp.md) — the tool schemas and transport boundary.
- [Command line](api/cli.md) — commands, input forms, JSON output, and exit codes.
- [Local WeMM deployment](use-local-wemm.md) — configure a local WeMM embedding service with the
  public SDK.
- [Agent Memory Leaderboard](agent-memory-leaderboard.md) — host the benchmark Add/Search format
  with physically isolated user stores.

## Deploy and operate

- [Architecture](architecture.md) — storage authority, write and retrieval paths, concurrency, and
  model boundaries.
- [Deployment](deployment.md) — embedded, REST, MCP, and edge process shapes.
- [Operations](operations.md) — health, capture backlog, backup, recovery, retention, index
  maintenance, telemetry, and the answering/workflow model quality gates.
- [Troubleshooting](troubleshooting.md) — diagnose startup, retrieval, content, and provider
  failures.
- [Security](../SECURITY.md) — trust boundaries, data exposure, and deployment hardening.

## Understand the direction

- [Context OS direction](context-os.md) — the long-term product boundary, fast and slow context
  planes, memory management, and context compilation.
- [Product goals and design principles](design-principles.md) — the unified Context & Memory
  positioning and Smarter / Swifter / Stronger criteria. States direction, not implemented status.
- [Plugin architecture](plugin-architecture.md) — the kernel/plugin boundary and the admission rule
  a new public capability must satisfy.
- [Affective memory](affective-memory.md) — sourced affect cues, trait visibility, response-policy
  authority, and the limits of model inference.

## Evaluate

- [Benchmarking](benchmarking.md) — behavior evaluation, local-index measurement, and SDK load tests.
- [Model quality gate](operations.md#model-quality-gate) — synthetic real-model regression cases,
  explicit snapshots, safe artifacts, and optional scheduled evaluation.
- [Example catalog](examples/README.md) — prerequisites and intended use for every committed YAML
  configuration, including the annotated evaluation template.

For repository setup and quality gates, see [CONTRIBUTING.md](../CONTRIBUTING.md). When a fact
changes, update its owning page and link to it elsewhere instead of copying it.
