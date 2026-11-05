# Product goals and design principles

SenseMem connects incoming context, durable memory, task-specific recall, and context construction
through one embedded runtime. Applications supply goals, actions, and source integrations; SenseMem
maintains memory semantics, evidence, retrieval, and local consistency.

This page explains the design criteria. [Product capabilities](product-capabilities.md) describes
supported behavior, [architecture](architecture.md) describes its implementation, and
[Context OS direction](context-os.md) describes the broader lifecycle goals.

## Smarter / Swifter / Stronger

| Principle | Goal | How to assess it |
| --- | --- | --- |
| **Smarter** | Give agents useful context through formation, enrichment, revision, organization, and recall. | Compare downstream task success, evidence accuracy, contradiction handling, and context cost with the same models, inputs, and budgets. |
| **Swifter** | Make capabilities easy to compose, models easy to select, and integrations easy to extend. | Measure integration effort and verify that optional components can be enabled independently. |
| **Stronger** | Support more deployment environments and improve quality across memory and multimodal tasks. | Publish tested configurations with reproducible quality, latency, and resource measurements. |

Hardware coverage and state-of-the-art quality are goals evaluated per configuration. Neither a
model's capabilities nor a score on one dataset establishes support or superiority elsewhere.

## One memory surface across modalities

Text, images, video, and audio use the same `add`, `search`, `ask`, and `compile` operations.
Ordered combinations preserve the caller's text and original media as one observation.
Application-specific capture objects normalize into this content contract at the source boundary.

Routing uses declared capabilities and the modalities actually present. Text-only requests do not
need multimodal inference. Native media routes preserve original signals; source-labelled captions
or transcripts can supply text for backends that require it. Unsupported content fails explicitly
when no usable route exists.

Spatial metadata augments these observations. It does not imply native 3D-scene support, sensor
drivers, localization, or coordinate transforms. See the [content contract](api/python-sdk.md#content-contract).

## Evidence before interpretation

Raw observations remain distinct from derived assertions. Model output carries source evidence,
basis, confidence, model, recipe, and temporal context. The kernel validates proposals before they
affect durable state; a model never grants itself authority to change runtime policy.

`search()` returns evidence. `compile()` selects an evidence-closed context bundle and reports
conflicts and missing information. `ask()` adds grounded generation and explicit abstention.
Citations and confidence values remain inspectable claims, rather than proofs of truth.

Corrections preserve the distinction between what was observed and what was inferred. Cognitive
forgetting, validity expiration, ranking decay, and physical deletion have different effects and
must remain distinguishable in the API and operation history. See
[memory types, time, and decay](memory-types-time-and-decay.md).

## Explicit composition around one kernel

Replaceable computation belongs behind narrow, typed backend protocols. The kernel owns identity,
validation, evidence policy, routing, persistence, and final hydration. A plugin's protocol grants
no storage authority; custom Python code still runs with the host process's privileges.

Applications choose bundled providers through `Memory.from_config()`, pass a `MemoryPlugins`
bundle, or inject backend objects directly. Composition stays fixed for one `Memory` lifetime.
Changing it requires closing and reopening the owner; changing a durable representation also
requires a compatible space or source replay into a fresh directory.

Optional capabilities add no model call when omitted. New capabilities need a concrete adapter,
a reachable product use case, explicit lifecycle and failure behavior, and tests through the public
API. Reuse an existing protocol before adding another. [Plugin architecture](plugin-architecture.md)
defines this boundary and its admission rule.

## One execution plane across interfaces

The Python SDK, REST, MCP, and CLI call the same `Memory` operations. Transports decode inputs and
serialize results; they do not choose independent retrieval policies, models, or storage semantics.
Shared operations preserve IDs, defaults, pagination, idempotency, and error meanings.

The common path is small: create one owner, add observations, retrieve evidence, and optionally
compile context or ask a question. Advanced operations remain discoverable in typed signatures,
OpenAPI, MCP schemas, and structured CLI output.

Transport authority is explicit. REST and MCP expose documented subsets and switches; memory
management and index maintenance stay with the owning process. Model capabilities and granted
administrative authority are different concerns. The [surface map](product-capabilities.md#python-rest-mcp-and-cli-surfaces)
lists the supported operations.

## Durable local authority

SQLite owns records, FP32 embeddings, store metadata, and pending index work. Immutable original
media stays in the local asset store. Zvec is a derived projection that can be rebuilt without
re-embedding stored content.

Durable writes commit SQLite before changing Zvec. Outbox work is acknowledged only after the
corresponding index flush succeeds. Retrieval hydrates candidates from SQLite, so a stale index ID
cannot resurrect a deleted record.

One physical `data_dir` has one live owner. Applications needing separate memory domains allocate
separate directories; metadata never substitutes for isolation or access control. These
[architecture invariants](architecture.md) also apply to optimized and asynchronous paths.

## Observable performance and quality

Measure the path an application experiences: input preparation, configured perception, embedding,
durable commit, search visibility, retrieval, and generation. Capture acknowledgement and time to
searchable are separate milestones. Report warm and cold behavior, latency percentiles,
throughput, memory, storage, and accelerator use on named hardware.

Performance changes must preserve durability, evidence, and modality coverage. Batching,
quantization, caching, and concurrency earn their complexity by improving a measured bottleneck.
Quality comparisons use the public SDK with fixed model, data, scorer, and runtime identities.
[Benchmarking](benchmarking.md) defines the reporting and comparison rules.

## Privacy and host authority

Local authoritative storage does not imply local inference. Every remote backend receives inputs
for its configured operation. Credentials, memory content, model responses, and provider exception
bodies stay out of ordinary logs.

The host grants sensitive operations, collects consent, controls source capture, and protects
network and filesystem access. Inferred affect cannot authorize a response policy, and a model
cannot manufacture consent. Physical deletion changes live state but does not guarantee secure
erasure from snapshots or backups. See [Security](../SECURITY.md).

## Reuse and scope

Use official SDKs, maintained model runtimes, existing dependencies, and the standard library before
building infrastructure. New dependencies require a public use case and belong in the narrowest
optional extra.

SenseMem does not provide agent planning, robot control, device synchronization, a hosted database,
or a worker service. Add abstractions when a concrete requirement demonstrates a gap in the
existing boundary. Keep benchmark code outside product dependencies and exercise product behavior
through the public SDK.
