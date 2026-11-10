# Core concepts

SenseMem turns observations into persistent evidence and brings that evidence back to an agent's
current task. One `Memory` instance handles storage, retrieval, context compilation, and
grounded answers. Your application supplies observations, model backends, and the agent's goals.

This guide explains the concepts behind the [quick start](quickstart.md). Use
[product capabilities](product-capabilities.md) to choose features,
[architecture](architecture.md) for consistency guarantees, and the
[SDK reference](api/python-sdk.md) for exact signatures.

## Context and memory

Context is information needed for the current task. Memory is information retained and organized
across time. Applications choose what to retain through `add()` or `capture()`; queries and
speculative recall snapshots can remain transient. Optional formation and explicit maintenance
organize stored evidence over time.

To return memory to the current task, `search()` retrieves evidence, `compile()` constructs a
bounded structured context bundle, and `ask()` generates an evidence-backed answer. The agent
owns its prompt, goals, and actions. SenseMem connects them to retained evidence through the
memory loop: capture, organize, recall, and construct context.

## From input to evidence

```text
content -> model preparation -> SQLite commit -> Zvec projection
query   -> selected dense/lexical routes -> SQLite hydration -> ranked evidence
```

SQLite is authoritative for records, embeddings, metadata, and pending index work.
Zvec is a derived search projection. Search results are hydrated from SQLite, and a missing Zvec
index can be rebuilt from stored embeddings without re-embedding content.

Treat `Memory` as one embedded runtime boundary. Do not build application state by
reading SQLite, `assets/`, or Zvec directly.

`capture()` commits the source before model work; `settle()` later enriches and indexes it.
`get()` and `list()` can read an unsettled source, while retrieval waits for enrichment. A pending
row may instead owe formation only and already be searchable: inspect its `awaiting` field.
The host schedules settlement; opening, searching, and closing never do it automatically.

## Content becomes a record

Python accepts one content atom or an ordered sequence of atoms:

| Atom | Use |
| --- | --- |
| `str` | Application text |
| `pathlib.Path` | A local media file copied into the asset store |
| `Blob` | In-memory media bytes with an explicit MIME type |
| `AssetRef` | Media already stored in the same data directory |

`add`, `search`, and `ask` share this `ContentInput` shape. `add_stream` also accepts `StreamInput`,
so each completed observation can supply event time, metadata, memory type, transcript or visual
description, and an `ObservationContext`. A URL-shaped `str` remains text; SenseMem never
downloads it. Fetch remote media in the application, then pass a `Blob` or local `Path`. REST and
MCP accept neither remote URLs nor server filesystem paths.

Each record has one modality. Text without media is `text`; text plus one media family keeps that
media modality; multiple media families produce `omni`. For example, a caption and photo form
one `image` record; text, a photo, and a voice recording form one `omni` record.

`ObservationContext` is caller input for provenance, confidence, world-time validity, metric pose,
and symbolic `place_id`. The returned `MemoryRecord` exposes the authoritative stored
`MemoryContext` and place separately. Its stable ID covers ordered canonical content, media
digests, metadata, event time, memory type, and optional observation context. Repeating the same
logical add returns the existing record without another model call. Media bytes are independently
de-duplicated by SHA-256, so records can share one immutable asset.

`delete()` removes the authoritative record and queues removal from the search projection. Shared
media remains until no record or active lease references it. API deletion is not secure erasure
from SQLite free pages, WAL files, snapshots, or backups; see the
[security model](../SECURITY.md#storage-deletion-and-backups).

Metadata is JSON application payload. SenseMem stores and returns it but does not
interpret it as a retrieval filter, tenant identifier, authorization rule, or isolation boundary.

## A data directory is one memory domain

```text
data_dir/
├── state.sqlite3
├── assets/
├── .sensemem.lock
└── zvec/
```

| Path | Role |
| --- | --- |
| `state.sqlite3` | Authoritative records, embeddings, metadata, and durable index outbox |
| `assets/` | Authoritative immutable media bytes |
| `zvec/` | Rebuildable search projection |
| `.sensemem.lock` | Operating-system lock target; its presence does not imply a live owner |

Exactly one live `Memory` may own a physical directory. Different live owners must
use different directories.

Allocate separate directories for applications, accounts, benchmark cases, or any
other data domains that must not share records. Use filesystem permissions and the deployment
boundary for access control. See [deployment](deployment.md) and [operations](operations.md) for
process, backup, and recovery procedures.

## Models are capabilities

Every memory requires an `EmbeddingBackend`. Other capabilities are optional:

| Capability | Enables |
| --- | --- |
| `GenerationBackend` | Grounded `ask()` answers |
| `TranscriptionBackend` or `SpeechBackend` | Audio/video text fallback and optional speech analysis |
| `FaceBackend` | Local face observations and identity evidence |
| `VisionDescriptionBackend` | Captions not-yet-described visual assets on every write (`add`/`add_many`/`settle`) |
| `FormationBackend` | Typed records derived from committed observations |
| `ConsolidationBackend` | Explicit, bounded memory-management proposals validated by the kernel |

Routing follows each backend's declared atomic modalities. Unsupported media fails explicitly
instead of being discarded.

Formation is opt-in. The declarative `formation: openai` provider or an injected
`FormationBackend` may propose finer typed records only after the source observation commits; it
never replaces the caller's raw record or its `MemoryType`.

Use `Memory.from_config()` for bundled adapters, `Memory.from_plugins()` for an explicit capability
bundle, or `Memory(...)` for direct injection. `Memory` closes backend objects it receives; an
adapter may leave its caller-supplied SDK client open. [Configuration](configuration.md) explains
the three paths, formation controls, and ownership rules.

## Retrieval can return no evidence

`search()` returns a `SearchResult`; its `hits` field is a `tuple[SearchHit, ...]`.
The relevance gate keeps weak matches out of an agent's context. An empty `hits` tuple is normal
when no candidate clears
`minimum_relevance`. With `limit=1`, SenseMem may also withhold an unresolved top-two tie. A score
ranks hits against one query and does not move with the `limit` you ask for; it is not a global
probability.

`ask()` uses the same retrieval path before calling a configured generation backend. Its
`AnswerResult` reports the answer, the canonical hits used, and explicit abstention state. Calling
`ask()` without an answerer is a configuration error.

`compile()` instead returns a structured, evidence-closed context bundle without calling a
generation model. Its character and item budgets differ from `ask()`'s grounding window. Use
[context compilation](context-compilation.md) when the host constructs its own prompt.

Handle empty search results and answer abstention as ordinary outcomes. Use
`search(..., explain=True)` for one local ranking investigation; normal telemetry intentionally omits
candidate identifiers.

## Continue by task

- Choose adapters and policies in [configuration](configuration.md).
- Model roles, event time, typed assertions, spatial scope, and decay in
  [memory types, time, and decay](memory-types-time-and-decay.md).
- Ingest completed observations or continuous capture in
  [omni streaming and interaction memory](omni-streaming-and-interaction-memory.md).
- Use [architecture](architecture.md) for transaction, recovery, and concurrency details.
