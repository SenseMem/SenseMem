# Product capabilities

SenseMem gives agents continuity across sessions and modalities. It preserves observations,
organizes evidence, and retrieves the context an application needs for its next task—all through
one embedded runtime.

Use it to build assistants that remember preferences and events, agents that recall original
photos and recordings, and applications that assemble useful context from a growing history.

## What your application gains

**Durable memory without a database service.** Records, embeddings, and original media live in
your application's data directory. Reopen it to continue; rebuild the search index from stored
embeddings when needed.

**One contract for text and media.** The same operations accept text, images, audio, video,
and ordered combinations. Compatible backends process original media; transcripts and captions
provide source-labelled text routes.

**Task-ready context with traceable evidence.** Retrieve ranked hits, compile a structured
context bundle, or generate a grounded answer. Time, place, identity, and evidence remain
available throughout the path.

**Memory that adapts to new information.** Optional formation derives typed assertions;
explicit maintenance reinforces, consolidates, corrects, or forgets records. An operation log
supports inspection and rollback.

## The memory loop

```text
Observations -> durable memory -> optional formation and maintenance
                     |
Task ----------------+-> search  -> ranked evidence
                     +-> compile -> structured context for your agent
                     +-> ask     -> grounded answer with evidence
```

`add()` returns searchable records. `capture()` saves observations before inference;
`settle()` enriches and indexes them on the host's schedule. Both preserve the same original
evidence and use the same configured model stages.

## Choose capabilities

| Capability | What it enables | Configuration or guide |
| --- | --- | --- |
| Durable capture | Single, batched, streamed, and deferred writes with immutable original media | Embedding backend; [core concepts](concepts.md) |
| Retrieval | Dense, lexical, or hybrid search with scope/time filters, reranking, and diagnostics | [Retrieval settings](configuration/settings.md) |
| Context compilation | Typed sections, item/character/media budgets, conflicts, and unknowns | [Context compilation](context-compilation.md) |
| Grounded answers | Buffered or streamed answers with evidence and abstention | Generation backend; [memory API](api/python-memory.md) |
| Typed memory | Entities, events, states, relations, affect, and traits derived from observations | Formation backend; [typed memory](typed-memory.md) |
| Memory maintenance | Consolidation, correction, reinforcement, reversible forgetting, and audit history | Consolidation backend for proposals; [memory management](memory-management.md) |
| Time and place | Event intervals, valid/known time, symbolic places, and same-frame spatial filters | Caller-supplied context; [time and decay](memory-types-time-and-decay.md) |
| Local identities | Voice/face observations, naming, corroborated linking, unlinking, and erasure | Speech and/or face backend; [identity API](api/python-identity.md) |
| Continuous input | Update/final/cancel lifecycles for audio, vision, and interaction streams | Host-normalized packets; [streaming](omni-streaming-and-interaction-memory.md) |
| Data lifecycle | Host-recorded consent, subject export, deletion, and retention passes | [Data-subject operations](api/python-identity.md#data-subject-rights) |

Enable optional capabilities independently. Modality coverage follows the selected backends'
declarations; the kernel validates their outputs before durable state changes. Response-policy
records require host authorization.

## Connect through your preferred interface

| Surface | Use | Reference |
| --- | --- | --- |
| SDK | Full synchronous and asynchronous integration | [SDK](api/python-sdk.md) |
| REST | Share one owner with HTTP clients under `/v1` | [REST](api/rest.md) |
| MCP | Give an agent host thirteen tools with configurable authority | [MCP](api/mcp.md) |
| CLI | Operate locally or call an existing REST owner | [CLI](api/cli.md) |

These adapters share the same kernel and expose the operations listed in their references.
`Memory.capabilities`, `GET /healthz`, MCP instructions, and `sensemem doctor` describe the
configured capabilities through a common document.

## Deploy with clear ownership

One physical `data_dir` has one live owner. Allocate separate directories for independent data
or security domains; let other processes call the owner through a transport.
SQLite and original media are authoritative, and Zvec is rebuildable.

Choose local or remote inference explicitly. The host controls authentication, authorization,
TLS, encryption, quotas, and scheduling. Use a local filesystem and back up SQLite and assets
together. See [deployment](deployment.md) and [operations](operations.md).

Phones, glasses, robots, and software tools can supply observations through application
adapters. The host handles capture, device drivers, synchronization, localization, goals,
and actions. SenseMem handles memory semantics, evidence, retrieval, and context construction.
Spatial filters compare coordinates within a declared frame.

## Compose and extend

Use `Memory.from_config()` for bundled providers, `Memory.from_plugins()` for a capability
bundle, or `Memory(...)` for individual backends. Narrow typed protocols let you choose models
without changing the memory API. Reopen the instance to change its composition.

The kernel keeps validation, stable IDs, durability, evidence policy, and final SQLite hydration.
[Configuration](configuration.md) explains composition;
[backend protocols](api/python-backends.md) define custom inference implementations;
[plugin architecture](plugin-architecture.md) explains extension rules.

This guide describes SenseMem 0.3.0. Start with the [quick start](quickstart.md), or explore the
long-term goals in [design principles](design-principles.md).
