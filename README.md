# SenseMem

[![CI](https://github.com/SenseMem/SenseMem/actions/workflows/ci.yml/badge.svg)](https://github.com/SenseMem/SenseMem/actions/workflows/ci.yml)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-3DA639.svg)](LICENSE)

**A unified context and memory layer for agents, across devices, modalities, and models.**

SenseMem turns information from digital interactions and the physical world into evolving memory,
then builds evidence-backed context for an agent's current task. Applications connect chat,
software tools, and device observations through one content contract.

Context is the information an agent needs now. Memory is information retained and organized across
time. SenseMem connects them in a continuous loop:

**Receive context → form and maintain memory → recall for a task → build useful context.**

The current release is an embedded, local-first runtime for ordered text, images, video, and audio,
alone or in supported combinations. It preserves original evidence and its time, place, and
identity context. The same `Memory` kernel powers the Python SDK, REST, MCP, and CLI.

Authoritative records and source media stay local. Inference uses explicitly configured local
adapters, OpenAI-compatible endpoints, or custom typed backends; a remote backend receives the
inputs required for its operation. Deployment depends on support for Python, dependencies, and the
selected models. Device capture and synchronization remain application responsibilities.

The documentation describes version `0.3.0`. Start with the [quick start](docs/quickstart.md)
or browse the [documentation index](docs/README.md).

## Why SenseMem

Three principles guide the product:

| Principle | Commitment |
| --- | --- |
| **Smarter** | Improve memory formation, enrichment, revision, organization, and recall so agents receive more useful context. |
| **Swifter** | Let users compose capabilities, choose models, and extend implementations through explicit plugin contracts. |
| **Stronger** | Deploy across hardware classes and pursue state-of-the-art results across a broad range of memory and multimodal tasks. |

SenseMem lets you:

- **Preserve multimodal evidence.** Store ordered text, images, video, and audio as one
  observation, with original media retained locally.
- **Retrieve meaningful memories.** Combine dense and lexical search with semantic, episodic,
  and procedural roles, event time, validity, identity, and spatial scope.
- **Build grounded responses.** `ask()` returns evidence and explicit abstention;
  `compile()` produces structured context within declared item, character, and media budgets.
- **Control when inference runs.** `add()` returns searchable; `capture()` commits before model
  work and lets the host schedule enrichment with `settle()`.
- **Manage memory explicitly.** Optional formation and consolidation propose evidence-linked
  changes; the kernel validates them and records reversible operations.
- **Recover local state.** SQLite and original media are authoritative. A durable outbox keeps
  the rebuildable Zvec search projection recoverable.

Use SenseMem inside a Python application, or expose the same owner through REST, MCP, or the CLI.
Applications choose local or remote model backends through typed protocols. Model choice controls
modality support, quality, inference cost, and hardware requirements.

The host owns agent planning, device integration, and access control. SenseMem owns memory
semantics, evidence, durable local consistency, retrieval, and context construction. See
[Product capabilities](docs/product-capabilities.md),
[Design principles](docs/design-principles.md), and [Architecture](docs/architecture.md).

## Installation

SenseMem supports Python 3.11 through 3.14. The base package provides local storage, retrieval,
public types and Zvec. Model runtimes and
transports live in optional extras; the [configuration guide](docs/configuration.md#1-install-only-the-surfaces-you-use)
maintains their complete dependency catalog. In a Python project, choose the integrations you need:

```bash
uv add "sensemem[local]"          # local embedding and speech
uv add "sensemem[openai]"         # OpenAI-compatible model endpoints
uv add "sensemem[local,server]"   # local models and REST
uv add "sensemem[openai,mcp]"     # remote models and MCP
```

With `pip`, the same extras apply:

```bash
python -m pip install "sensemem[local]"
```

`sensemem[all]` installs the complete optional dependency surface. It does not configure model
providers, download benchmark datasets, or supply the YuNet and SFace model files.

Use the base `sensemem` package when supplying your own embedder. Installing a transport extra
does not configure a model backend.

`SentenceTransformersEmbedder` loads the selected model at construction with
`trust_remote_code=False`. The example below selects EmbeddingGemma 2 with Apache-2.0 weights.
Review the selected model's license and record its resolved identity when comparing deployments.

See [Configuration](docs/configuration.md) for provider fields, custom backends, credentials, and
durable embedding-space rules.

## Basic usage

Save the following as `example.py` after installing `sensemem[local]`. It stores and retrieves
a fact without a database service or model API key:

```python
from sensemem import Memory

config = {
    "data_dir": "./data/sensemem-demo",
    "embedding": {
        "provider": "sentence-transformers",
        "model": "google/embeddinggemma-2",
    },
    "settings": {
        "minimum_relevance": 0.0,
        "ambiguity_margin": 0.0,
    },
}

with Memory.from_config(config) as memory:
    stored = memory.add(
        "The spare key is in the blue toolbox.",
        metadata={"source": "workshop-note"},
    )
    duplicate = memory.add(
        "The spare key is in the blue toolbox.",
        metadata={"source": "workshop-note"},
    )
    hits = memory.search("Where is the spare key?", limit=1).hits

    assert duplicate.id == stored.id
    assert hits and hits[0].id == stored.id
    print(hits[0].content)
```

```bash
uv run python example.py
```

The first run may pause while the model downloads. Success prints
`The spare key is in the blue toolbox.`. The zero relevance and ambiguity settings make this
one-record checkpoint deterministic; remove them to restore the defaults (`0.10` and `0.01`)
before tuning a real corpus.

Run the script again to reopen the same directory. Repeating the same canonical content, metadata,
time, type, and context returns the same record instead of creating a duplicate. The context manager
closes model, SQLite, and index resources.

The same `add()` call accepts local `Path`, inline `Blob`, existing `AssetRef`, or an ordered
combination. Add a generation backend to call `ask()`; add speech or face backends for local
identity analysis; use `AsyncMemory` and the stream helpers for continuous capture.

Continue with the [Quick start](docs/quickstart.md) and
[Python SDK reference](docs/api/python-sdk.md).

## Capabilities and documentation

| Capability | Implemented surface | Guide |
| --- | --- | --- |
| Product overview | Use cases, capability boundaries, and extension model | [Product capabilities](docs/product-capabilities.md) |
| Omni-modal memory | Text, image, video, audio, arbitrary mixed records, assets, and content identity | [Core concepts](docs/concepts.md) |
| Retrieval and time | Dense/lexical retrieval, trace, memory roles, bitemporal and spatial scope, decay | [Memory types, time, and decay](docs/memory-types-time-and-decay.md) |
| Grounded generation | Optional answer backend, evidence hits, abstention, and evidence budgets | [Configuration](docs/configuration.md) |
| Formation | Optional entity, event, state, relation, affect, and trait derivation; host-authorized response policies | [Automatic formation](docs/configuration/lifecycle.md#automatic-memory-formation) |
| Streaming | Durable streams, omni prefetch, audio/vision reducers, interaction memory | [Streaming and interaction memory](docs/omni-streaming-and-interaction-memory.md) |
| Identity | Speech and face analysis, naming, corroborated linking, unlinking, and erasure | [Python SDK](docs/api/python-identity.md#cross-modal-identity-binding) |
| Context compilation | Budgeted, partitioned context bundles with conflicts and declared capabilities | [Context compilation](docs/context-compilation.md) |
| Memory control plane | Deferred capture settlement, model-proposed consolidation, reversible forgetting, and the operation log | [Memory types, time, and decay](docs/memory-types-time-and-decay.md) |
| Interfaces | Complete SDK; SDK-aligned REST, MCP, and CLI operations | [Python](docs/api/python-sdk.md) · [REST](docs/api/rest.md) · [MCP](docs/api/mcp.md) · [CLI](docs/api/cli.md) |
| Deployment and recovery | Ownership, backup, restore, reindex, telemetry, and security boundaries | [Deployment](docs/deployment.md) · [Operations](docs/operations.md) |
| Extensibility | Narrow typed model protocols with no global plugin registry | [Plugin architecture](docs/plugin-architecture.md) |

The [documentation index](docs/README.md) is the complete task-oriented map.

## Benchmarks

`sensemem-bench eval` evaluates memory quality and end-to-end behavior through the public SDK.
Its executable catalog covers long-horizon conversation,
personalization, multimodal video and lifelog QA, embodied scene memory, and grounded retrieval.
Separate utilities produce raw LoCoMo-Refined predictions and measure the local SQLite-to-Zvec
index path. Copy the [evaluation example](docs/examples/eval.example.yaml) to `eval.yaml` and
configure the embedding, generation, and judge endpoints and credentials before running it.

```bash
sensemem-bench eval --list-tasks
sensemem-bench eval \
  --config eval.yaml \
  --tasks locomo-refined \
  --model-args generation_model=gpt-5-mini-2025-08-07 \
  --limit 1 \
  --seed 42 \
  --output-path .benchmarks/results/locomo-one-conversation
```

`--limit 1` selects one LoCoMo conversation and evaluates all its questions; it is not one model
request. Results record dataset and implementation pins, digests, confidence intervals, latency,
resources, tokens, abstentions, mandatory controls, and a noise floor.

Compare results with their dataset, scorer, model, runtime, hardware, and validity fields.
[Benchmarking](docs/benchmarking.md) documents dataset preparation and reproducible comparisons.

## Operational boundaries

- One physical `data_dir` has one live owner; metadata is not an isolation boundary.
- REST and network MCP add no authentication, authorization, TLS, quotas, or rate limits.
- Backups require SQLite and original assets; Zvec alone is disposable.
- Embedding model, revision, dimension, and recipe form durable identity.
- SenseMem does not fetch remote URLs; the host must download and validate media.
- Stored content remains untrusted model input; inspect `AnswerResult.hits` for high-impact use.

Read [Security](SECURITY.md), [Deployment](docs/deployment.md),
[Operations](docs/operations.md), and [Troubleshooting](docs/troubleshooting.md).

## Citation

If you use SenseMem in academic or technical work, cite the version or commit you evaluated:

```bibtex
@software{sensemem_2026,
  author  = {{SenseMem contributors}},
  title   = {SenseMem: A Unified Context and Memory Layer for Agents},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/SenseMem/SenseMem}
}
```

## Acknowledgements

SenseMem builds on SQLite, Zvec, Pydantic, and OpenTelemetry. Optional surfaces integrate
Sentence Transformers, FunASR, OpenCV, the official OpenAI SDK, FastAPI/Starlette, and MCP.
We thank their maintainers and research communities. Benchmark sources, revisions, protocol notes,
and retained licenses are listed in [Benchmarking](docs/benchmarking.md) and the bundled
[scorer notices](src/sensemem/benchmarks/_official/NOTICE.md).

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers the locked `uv` environment,
Python 3.11–3.14 quality gates, documentation checks, architecture constraints, and pull-request
expectations. Participation follows the [Code of Conduct](CODE_OF_CONDUCT.md); report security
issues privately through [SECURITY.md](SECURITY.md).

## License

SenseMem-authored code and documentation are available under the
[Apache License 2.0](LICENSE). Models, datasets, and vendored benchmark material retain their own
terms; review the licenses of every model and dataset selected for your deployment.
