# Python SDK

Import supported public SDK values from `sensemem`. `Memory` is synchronous;
`AsyncMemory` wraps the same instance semantics. REST/MCP/CLI expose documented subsets.

## Find an API

| Task | Reference |
| --- | --- |
| Construct, configure, or close an instance | [Construction](#construction), [lifecycle](#lifecycle-and-ownership) |
| Supply text/media or per-item metadata | [Content contract](#content-contract) |
| Write, capture, search, answer, compile, or delete | [Memory operations](python-memory.md) |
| Recognize/name people, record consent, export, or retain | [Identity and data-subject operations](python-identity.md) |
| Consolidate, correct, forget, audit, or roll back | [Control operations](python-control.md) |
| Await operations or consume live streams | [Async and streaming](python-streams.md) |
| Look up returned fields, enums, or root exports | [Public values](python-values.md) |
| Implement inference or select a bundled model | [Backend protocols](python-backends.md), [adapters](python-adapters.md) |
| Handle failures and input bounds | [Errors and limits](#errors-and-limits) |

## First use

Install the [local extra](../configuration.md#1-install-only-the-surfaces-you-use) for this example:

```python
from sensemem import Memory, SentenceTransformersEmbedder

with Memory(
    "./data/assistant",
    embedder=SentenceTransformersEmbedder(model_id="google/embeddinggemma-2"),
) as memory:
    record = memory.add("The spare key is in the blue toolbox.")
    hits = memory.search("Where is the spare key?").hits
```

For declarative files/providers, follow [Configuration](../configuration.md).

## Construction

```text
Memory(
    data_dir: str | Path = ".sensemem",
    *,
    embedder: EmbeddingBackend,
    answerer: GenerationBackend | None = None,
    transcriber: SpeechBackend | TranscriptionBackend | None = None,
    vision_describer: VisionDescriptionBackend | None = None,
    face_analyzer: FaceBackend | None = None,
    former: FormationBackend | None = None,
    consolidator: ConsolidationBackend | None = None,
    reranker: RerankingBackend | None = None,
    config: MemoryConfig | None = None,
    tracer: opentelemetry.trace.Tracer | None = None,
) -> None

Memory.from_plugins(
    data_dir: str | Path = ".sensemem",
    *,
    plugins: MemoryPlugins,
    config: MemoryConfig | None = None,
    tracer: Tracer | None = None,
) -> Memory

Memory.from_config(
    config: SenseMemConfig | Mapping[str, object],
    *,
    tracer: Tracer | None = None,
) -> Memory

resolve_memory_config(
    value: SenseMemConfig | Mapping[str, object],
) -> MemoryComposition
```

`embedder` is required. The other slots independently enable answering, speech/transcription,
visual description, faces, formation, consolidation, or reranking. An omitted slot does not inherit
another slot's capability. `MemoryConfig` carries policies; `tracer` is a separate dependency.

`MemoryPlugins` is a value bundle with the same backend slots. Construction validates protocols
and declarations before storage opens. `Memory.from_config` constructs the closed catalog of
bundled providers. `resolve_memory_config` returns a `MemoryComposition(data_dir, plugins, settings)`;
close it unless its plugins have been transferred to an owner.

See [runtime settings](../configuration/settings.md) for defaults and
[backend contracts](python-backends.md) for custom implementations. Answer evidence budgets,
closure, and reinforcement are defined with [ask](python-memory.md#answer-grounding).

## Lifecycle and ownership

One physical directory has one live owner. A second raises `StorageError(reason="data_dir_in_use")`.
Use separate directories or send requests to the existing REST owner. Metadata provides no logical
account/request isolation. An instance belongs to the process that opened it and cannot cross fork.

`Memory` closes each distinct supplied backend once; an adapter may document separate ownership of
a caller-supplied provider client. Use a context manager or close explicitly; repeated closes are
harmless.

```text
close(*, timeout_seconds: float = 30.0) -> None
```

Timeout must be positive and finite. Expiration raises retryable
`StorageError(reason="timeout", stage="close")` while cleanup continues with ownership retained.
New work is rejected once closing begins. Another close waits for the same cleanup.
Provider timeouts remain necessary for running synchronous calls.

## Content contract

```python
ContentAtom = str | pathlib.Path | Blob | AssetRef
ContentInput = ContentAtom | Sequence[ContentAtom]
```

| Input | Meaning |
| --- | --- |
| `str` | Text, including strings resembling paths or URLs |
| `Path` | Local regular media file; suffix determines media type |
| `Blob` | Nonempty image/video/audio bytes with media type and optional name |
| `AssetRef` | Existing media in the same directory, identified by lowercase 64-character SHA-256 |
| Ordered sequence | One composite record preserving text/media order |

SenseMem does not fetch remote URLs. Returned asset references include modality, media type, size,
digest, name, and local path. Metadata must be JSON-compatible with nonempty string keys.
Event timestamps are timezone-aware; end requires an earlier start. Roles are semantic/episodic/
procedural. Stable IDs cover canonical ordered content, media, metadata, event bounds, role,
and observation context. Repeating completed canonical input needs no new model call; pending
capture or formation can still run.

Per-item streams use:

```text
StreamInput(
    content: ContentInput,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    context: ObservationContext | None = None,
    transcript: str | None = None,
    description: str | None = None,
)
```

## Errors and limits

All stable exceptions derive from `SenseMemError`:

| Exception | Code | Default reason |
| --- | --- | --- |
| `SenseMemError` | `sensemem_error` | `None` |
| `ValidationError` | `validation_error` | `input_invalid` |
| `MemoryNotFoundError` | `memory_not_found` | `memory_not_found` |
| `IdentityNotFoundError` | `identity_not_found` | `identity_not_found` |
| `ModelError` | `model_error` | `None` |
| `ModelOutputTruncatedError` | `model_output_truncated` | `output_truncated` |
| `StorageError` | `storage_error` | `None` |
| `IndexUnavailableError` | `index_unavailable` | `None` |

Errors may carry reason, stage, and subject. Retryability comes only from the stable reason:
`capacity_exceeded`, `connection_failed`, `data_dir_in_use`, `flush_failed`, `index_missing`,
`provider_unavailable`, `rate_limited`, or timeout. It is never inferred from message text.
Truncation is a model error; index unavailability is a storage error.

Provider refusals map to `model_refused` without exposing refusal text. Quota exhaustion requires
operator action; `provider_unavailable` follows the SDK retry budget. `Context_length_exceeded` is
permanent provider context overflow; `payload_too_large` is a byte limit or HTTP 413, including
non-JSON proxy responses. Neither silently trims selected evidence for a retry. Output truncation
is separate. See [the shared error vocabulary](rest.md#codes-and-reasons).

| Input bound | Value |
| --- | --- |
| Ordered content parts | 128 |
| Normalized combined text | 65,536 characters |
| Serialized metadata per memory | 262,144 UTF-8 bytes |
| Local media asset | 512 MiB |
| Search/answer/list limit | 1–100 |
| Identity/speaker name | 255 printable characters |
| Add-many inputs | 1024 |

These are local ceilings; backend and REST/MCP limits may be narrower. One operation/process shares
a 512 MiB materialization budget. Oversized input is invalid; contention is retryable capacity
exceeded. Capture backlog caps at 10,000 per directory, while duplicate capture remains idempotent.
[Runtime limits](../runtime-limits.md) defines worker/store/connection admission and cancellation.
[Adapter limits](python-adapters.md#openai-request-behavior) defines media degradation and payloads.
