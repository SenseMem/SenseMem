# Python async and streaming APIs

Use memory in an async application, segment continuous media into complete events, prefetch context during a changing turn, and commit finalized observations. Async methods preserve the synchronous API's arguments, defaults, results, and errors.

See [the streaming guide](../omni-streaming-and-interaction-memory.md) for complete capture examples and update/final/cancel semantics.

| Need | API |
| --- | --- |
| Await ordinary memory operations | [AsyncMemory](#asyncmemory) |
| Coalesce speculative queries | [AsyncOmniPrefetch](#asyncomniprefetch) |
| Receive temporary context from an application-owned coroutine or agent | [AsyncContextSource](#asynccontextsource) |
| Understand continuous media and commit complete events | [AsyncEventStream](#asynceventstream) |
| Commit update/final/cancel events | [AsyncCaptureStream](#asynccapturestream) |
| Reduce audio or vision packets | [AsyncAudioStream](#asyncaudiostream), [AsyncVisionStream](#asyncvisionstream) |

## AsyncMemory

`AsyncMemory(memory: Memory)` wraps and owns one open `Memory`; `AsyncMemory.from_plugins()` and `AsyncMemory.from_config()` mirror their `Memory` counterparts. Operations keep identical parameters, defaults, and return types, executed on bounded worker pools while preserving tracing context:

```text
add_stream(
    contents: AsyncIterable[ContentInput | StreamInput],
    *,
    capture: bool = False,
) -> AsyncIterator[MemoryRecord]
```

Cancelling an awaiter does not interrupt already-running synchronous work; capacity stays reserved until it finishes.

`StreamInput.interpretation` stores a model's lossy interpretation as a separate `model_interpretation` content part excluded from source replay. Derived fields (`transcript`, `description`, `interpretation`) do not change the original record's content-addressed ID; supplying updated derived text refreshes that derived kind without reactivating retired or withdrawn records.

## AsyncOmniPrefetch

```text
AsyncOmniPrefetch(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryCategory | None = None,
    reference_at: datetime | None = None,
    occurred_from: datetime | None = None,
    occurred_until: datetime | None = None,
    budget: ContextBudget | None = None,
) -> None

latest: PrefetchResult | None
submit(query: ContentInput) -> int
updates() -> AsyncIterator[PrefetchUpdate]
async finalize(query: ContentInput | None = None) -> PrefetchResult
async close() -> None
```

Runs one retrieval at a time and replaces any unstarted queued snapshot with the latest submission. `submit()` returns a request-local revision; `updates()` yields non-superseded `PrefetchUpdate(revision, result)` completions; `finalize()` returns the exact result for the final snapshot and closes the helper. Passing `budget=ContextBudget(...)` runs `compile()` instead of `search()` (returning `PrefetchResult.bundle`) and disallows search-only `memory_type`/`occurred_from`/`occurred_until`. Snapshots reject mutable `Path` atoms—pass text, `Blob`, or `AssetRef`.

## AsyncContextSource

`AsyncContextSource` runs an application-supplied async context producer independently of the main pipeline and returns `ContextSupplement` values for `AsyncMemory.compile()`:

```python
from sensemem import AsyncContextSource, ContextSupplement


async def provide_context(query):
    observation = await my_agent.observe(query)
    return (ContextSupplement(observation, source="my-agent"),)


source = AsyncContextSource(provide_context)
revision = source.submit("What should I bring?")
try:
    result = await source.finalize()
    bundle = await async_memory.compile("What should I bring?", supplements=result.supplements)
finally:
    await source.close()
```

```text
AsyncContextSource(
    provider: Callable[[ContentInput], Awaitable[Sequence[ContextSupplement]]],
)
submit(query: ContentInput) -> int
latest: ContextSourceResult | None
async finalize(query: ContentInput | None = None) -> ContextSourceResult
async close() -> None
```

One supplier runs at a time, superseding older unstarted snapshots and discarding stale completions. `close()` and cancelling `finalize()` cancel the active async supplier and wait for cleanup. Submissions reject mutable `Path` atoms.

## AsyncEventStream

```text
AsyncEventStream(
    memory: AsyncMemory,
    *,
    segmenter: EventSegmentationBackend,
    limit: int = 10,
    context: StreamContext = None,
    reference_at: datetime | None = None,
    max_streams: int = 32,
    max_chunks: int = 32,
    max_buffer_bytes: int = 64 * 1024 * 1024,
    max_pending_bytes: int = 64 * 1024 * 1024,
    capture: bool = False,
    checkpoint_path: Path | None = None,
) -> None

positions: dict[str, datetime]

async AsyncEventStream.append(chunk: EventChunk) -> None
AsyncEventStream.process(*, end_of_stream: bool = False) -> AsyncIterator[StreamCommit]
AsyncEventStream.consume(chunks: AsyncIterable[EventChunk]) -> AsyncIterator[StreamCommit]
```

`EventChunk(content, occurred_at, occurred_end, stream_id="default", context=None)` holds one media `Blob` or a synchronized audio + image/video pair with timezone-aware, contiguous, nonoverlapping intervals per stream.

- `append()` requires `checkpoint_path` and flushes original media and its pending reference before returning, without waiting for model work. Concurrent appends serialize while one `process()` assesses earlier input.
- `process()` coalesces pending chunks within `max_chunks` and `max_buffer_bytes` and yields `StreamCommit` records for confirmed events. Call `process(end_of_stream=True)` after final input arrives to request a final assessment without forcing unfinished activities to close.
- `consume()` combines source pulling and processing (with or without a checkpoint).

Each confirmed event writes one or more `EPISODIC` records (splitting at `max_chunks`, 127 media parts, `max_buffer_bytes`, or chunk-context changes) sharing `event_segmentation.event_id`, storing original media in content and the unverified synopsis in `interpretation`/metadata. Unfinished suffixes exceeding the active window roll forward via a labelled `complete=False` continuation plus the last analyzed original chunk while keeping all originals on disk.

With `checkpoint_path`, accepted media and validated progress persist in the manifest and `<checkpoint_path>.media/` directory across process restarts. Checkpoint manifests require an explicit integer `"version": 5` (missing, non-integer, or unsupported versions raise `ValidationError`). Reopen with the same `data_dir`, segmenter recipe, `max_chunks`, and `max_buffer_bytes`, and inspect `positions` for per-stream resumption timestamps. Only one helper may own a checkpoint, and overlapping `process()`/`consume()` calls are rejected. `capture=True` defers write enrichment to `settle()` while event segmentation still runs. Cancellation drains any active model call or started commit cleanly. See [automatic event segmentation](../omni-streaming-and-interaction-memory.md#automatic-event-segmentation).

## Async capture streams

### AsyncCaptureStream

```text
AsyncCaptureStream(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryCategory | None = None,
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
    memory_type: MemoryCategory | None = None,
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
    memory_type: MemoryCategory | None = None,
    context: StreamContext = None,
    reference_at: datetime | None = None,
    max_streams: int = 32,
    capture: bool = False,
) -> None
```

### consume

```text
AsyncCaptureStream.consume(events: AsyncIterable[StreamEvent]) -> AsyncIterator[StreamCommit]
AsyncAudioStream.consume(packets: AsyncIterable[AudioStreamPacket]) -> AsyncIterator[StreamCommit]
AsyncVisionStream.consume(packets: AsyncIterable[VisionStreamPacket]) -> AsyncIterator[StreamCommit]
```

`AsyncCaptureStream` maps `update` to speculative retrieval, `final` to durable commit, and `cancel` to discard. `AsyncAudioStream` reduces `PCMChunk`, `VADPacket`, `ASRPartial`, and `AcousticBoundary`; `AsyncVisionStream` reduces `VisionFrame`, `VisionPartial`, and `SceneBoundary`. `StreamContext` is a fixed `ObservationContext`, a zero-argument sampler called once per closed observation, or `None`. Passing `capture=True` commits finals via `capture()` and skips automatic retrieval (`pending_settlement=True`, `prefetch=None`, `retrieval_error=None`).
