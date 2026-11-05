# Python async and streaming APIs

Async methods preserve synchronous arguments, defaults, results, and errors.
See [the streaming guide](../omni-streaming-and-interaction-memory.md) for complete capture examples.

| Need | API |
| --- | --- |
| Await ordinary memory operations | [AsyncMemory](#asyncmemory) |
| Coalesce speculative queries | [AsyncOmniPrefetch](#asyncomniprefetch) |
| Commit update/final/cancel events | [AsyncCaptureStream](#asynccapturestream) |
| Reduce audio or vision packets | [AsyncAudioStream](#asyncaudiostream), [AsyncVisionStream](#asyncvisionstream) |

## AsyncMemory

`AsyncMemory(memory: Memory)` wraps one open `Memory`, which it owns and closes;
`AsyncMemory.from_plugins()` and `AsyncMemory.from_config()` take the same parameters as their
`Memory` counterparts and open one for you. Its operations have the same keyword parameters,
defaults, and result values as `Memory`, including `forget_identity`. Finite operations are awaited
and `close()` is asynchronous. Its stream boundary is:

```text
add_stream(
    contents: AsyncIterable[ContentInput | StreamInput],
    *,
    capture: bool = False,
) -> AsyncIterator[MemoryRecord]
```

The facade submits the synchronous core to bounded worker pools and preserves tracing context.
It does not turn synchronous provider clients into native async clients. Cancelling an awaiter
does not stop work already running; capacity stays reserved until that work finishes.

## AsyncOmniPrefetch

`AsyncOmniPrefetch` coalesces evolving query snapshots for one turn:

```text
AsyncOmniPrefetch(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    occurred_from: datetime | None = None,
    occurred_until: datetime | None = None,
) -> None

latest: PrefetchResult | None
submit(query: ContentInput) -> None
async finalize(query: ContentInput | None = None) -> PrefetchResult
async close() -> None
```

Only one search runs at once and only the newest queued snapshot survives. `finalize` returns the
result for the exact final snapshot and closes the helper. `PrefetchResult` contains only `hits`;
submission numbers are internal. Snapshots reject mutable `Path` atoms; use `Blob` or
`AssetRef`.

Use `finalize()` when results must match the final query; `latest` remains speculative.

## Async capture streams

### AsyncCaptureStream

```text
AsyncCaptureStream(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    max_streams: int = 32,
    capture: bool = False,
) -> None
```

### AsyncAudioStream

```text
AsyncAudioStream(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    context: StreamContext = None,
    reference_at: datetime | None = None,
    max_streams: int = 32,
    capture: bool = False,
) -> None
```

### AsyncVisionStream

```text
AsyncVisionStream(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    context: StreamContext = None,
    reference_at: datetime | None = None,
    max_streams: int = 32,
    capture: bool = False,
) -> None
```

### consume

```text
AsyncCaptureStream.consume(
    events: AsyncIterable[StreamEvent],
) -> AsyncIterator[StreamCommit]

AsyncAudioStream.consume(
    packets: AsyncIterable[AudioStreamPacket],
) -> AsyncIterator[StreamCommit]

AsyncVisionStream.consume(
    packets: AsyncIterable[VisionStreamPacket],
) -> AsyncIterator[StreamCommit]
```

`AsyncCaptureStream` treats `update` as speculative retrieval, `final` as the durable write, and
`cancel` as discard. `AsyncAudioStream` reduces `PCMChunk`, `VADPacket`, `ASRPartial`, and
`AcousticBoundary`; `AsyncVisionStream` reduces `VisionFrame`, `VisionPartial`, and
`SceneBoundary`. `StreamContext` is an `ObservationContext`, a callable sampled once per completed
observation, or `None`. Up to `max_streams` independent `stream_id` values may be active.
`capture=True` commits each final through `capture` instead of `add`, so acknowledgement leaves
the model path: automatic retrieval is skipped, and every `StreamCommit` reports
`pending_settlement=True`, `prefetch=None`, and `retrieval_error=None`. New records require
`settle` before retrieval; capturing an existing record preserves its readiness. Inspect
`pending_captures()` for actual outstanding work. Use `AsyncOmniPrefetch` separately if speculative
retrieval is needed. The default stays the strong `add`. See
[omni streaming and interaction memory](../omni-streaming-and-interaction-memory.md) for event
semantics and complete examples.
