# Product capabilities

SenseMem gives agents durable memory and task-specific context through one embedded Python
runtime. Applications provide observations and model backends; SenseMem stores evidence, recalls
it, and returns search results, a context bundle, or a grounded answer.

## When to use it

- Keep facts, events, preferences, and procedures across assistant sessions.
- Recall evidence from text, images, audio, and video with compatible model backends.
- Give an agent a bounded context bundle while it retains control of prompts and actions.
- Run local memory behind Python, HTTP, MCP, or a command line.

Phones, smart glasses, and robots can supply observations through application adapters. Device
capture, synchronization, localization, and sensor drivers belong to the host application.
This page describes SenseMem 0.3; [design principles](design-principles.md) covers future direction.

## The memory loop

```text
Observation -> durable record -> optional typed formation
                                  |
Task -> search -> evidence --------+
     -> compile -> context for the host agent
     -> ask -> grounded answer with evidence
```

`add` returns searchable records. `capture` acknowledges a durable observation before model work;
`settle` enriches it and makes it searchable. Optional formation adds evidence-linked assertions.
The host explicitly schedules consolidation, reviews operation history, and can roll changes back.

## Capabilities and requirements

| Capability | Supported behavior | Requirement and guide |
| --- | --- | --- |
| Durable capture | Single, batched, streamed, and deferred writes; immutable original media | Embedding backend; [core concepts](concepts.md) |
| Retrieval | Dense, lexical, or hybrid search; scope/time filters; optional reranking and diagnostics | [Retrieval settings](configuration/settings.md) |
| Context compilation | Typed sections, item/character/media budgets, conflicts, and unknowns | [Context compilation](context-compilation.md) |
| Grounded answers | Buffered or streamed answers, evidence, and abstention | Generation backend; [Python memory API](api/python-memory.md) |
| Typed memory | Observations, entities, events, states, relations, affect, traits, and host-authorized response policies | Optional formation; [typed memory](typed-memory.md) |
| Memory maintenance | Consolidate, correct, reinforce, forget, audit, and roll back | Proposals need a consolidation backend; [memory management](memory-management.md) |
| Time and place | Event intervals, valid/known time, symbolic places, and same-frame spatial filters | Caller-supplied context; [memory types and time](memory-types-time-and-decay.md) |
| Local identities | Voice/face observations, naming, corroborated linking, unlinking, and erasure | Speech and/or face backend; [identity API](api/python-identity.md) |
| Continuous input | Explicit update/final/cancel lifecycle for audio, vision, and interaction streams | Host-normalized packets; [streaming guide](omni-streaming-and-interaction-memory.md) |
| Data lifecycle | Host-recorded consent, subject export, deletion, and retention passes | Host authority; [data-subject operations](api/python-identity.md#data-subject-rights) |

Modality coverage follows declared backend capabilities. A transcript supports speech content;
visual or spatial claims need corresponding evidence. Evidence IDs establish provenance, and
source coverage alone does not prove that a model's answer is correct.

## Python, REST, MCP, and CLI surfaces

All adapters call the same `Memory` kernel. Network exposure is opt-in.

| Surface | Use | Reference |
| --- | --- | --- |
| Python | Full synchronous and asynchronous integration | [SDK](api/python-sdk.md) |
| REST | Share one owner with HTTP clients under `/v1` | [REST](api/rest.md) |
| MCP | Expose thirteen tools to an agent host | [MCP](api/mcp.md) |
| CLI | Local operations or requests to an existing REST owner | [CLI](api/cli.md) |

Adapters expose different subsets of the SDK. Consult their references for exact routes, tools,
administrative opt-ins, and errors. `Memory.capabilities`, `GET /healthz`, MCP instructions, and
`sensemem doctor` describe the configured capabilities through the same document.

## Deployment and isolation

One physical `data_dir` has one live owner. Separate applications or security domains use separate
directories. Other processes call the owner through a transport; directories do not synchronize.
Metadata is application payload and provides no access-control boundary.

SQLite and original media are authoritative; Zvec is rebuildable. Storage remains local while
configured inference can be local or remote. Back up SQLite and assets together. The host supplies
authentication, authorization, encryption, TLS, quotas, and service scheduling.

See [deployment](deployment.md) for topology and [operations](operations.md) for runbooks.

## Extend a composition

Use bundled providers through `Memory.from_config`, grouped application backends through
`Memory.from_plugins`, or individual backend objects through `Memory(...)`.
[Configuration](configuration.md) explains the choices;
[backend protocols](api/python-backends.md) define custom inference implementations.

Composition is fixed for one instance's lifetime. Backends declare inference capabilities; the
kernel retains validation, stable IDs, durability, evidence policy, and final SQLite hydration.
Custom transports call the public SDK. SQLite, media storage, and Zvec are internal components.

## Limits

- SenseMem is an embedded runtime with one writer-owner per directory. Shared-volume multi-writer
  operation and network filesystems are unsupported.
- Model quality depends on the selected backends and workload. Evaluate changes with the
  [model quality gate](model-quality.md) and [benchmarks](benchmarking.md).
- The host agent owns goals, tools, actions, and scheduling. Stored procedures and response-policy
  memories are evidence, never authority to change runtime policy.
- Spatial metadata does not provide coordinate transforms or native 3D-scene understanding.
- There is no bundled cross-device sync, device SDK, plugin discovery, or live backend swap.

Start with the [quick start](quickstart.md), then select the capabilities your application needs.
