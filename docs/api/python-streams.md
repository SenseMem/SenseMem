# Python async and streaming APIs

Use memory in an async application, segment continuous media into complete events, prefetch
context during a changing turn, and commit finalized observations. Async methods preserve the
synchronous API's arguments, defaults, results, and errors.

See [the streaming guide](../omni-streaming-and-interaction-memory.md) for complete capture examples
and update/final/cancel semantics.

| Need | API |
| --- | --- |
| Await ordinary memory operations | [AsyncMemory](#asyncmemory) |
| Coalesce speculative queries | [AsyncOmniPrefetch](#asyncomniprefetch) |
| Receive temporary context from an application-owned coroutine or agent | [AsyncContextSource](#asynccontextsource) |
| Understand continuous media and commit complete events | [AsyncEventStream](#asynceventstream) |
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

`StreamInput.interpretation` carries a model's lossy interpretation separately from original
`content`. Both blocking and deferred writes store it as a `model_interpretation` content part,
with an explicit derived-evidence marker. It survives settlement and reopening, but source replay
does not turn it into original text. Supply the model, recipe, source span and uncertainty in
application metadata; an interpretation does not establish that every supplied modality was
fully perceived. `AsyncEventStream` supplies these fields for event summaries.

Derived fields do not participate in the original observation's content-addressed ID. Supplying a
changed `transcript`, `description`, or `interpretation` refreshes that derived kind on the same
record and preserves kinds omitted from the new item. Ordinary duplicate `add`/`capture` calls
preserve existing enrichment. Blocking writes replace old retrieval keys before returning;
capture-routed refreshes withdraw old keys and return to pending enrichment until `settle()`
recomputes them. A derivation refresh preserves existing retirement and explicit withdrawal.
It cannot make an old source or a retracted claim active again.

## AsyncOmniPrefetch

`AsyncOmniPrefetch` coalesces evolving query snapshots for one turn:

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

Only one search runs at once and only the newest queued snapshot survives. `finalize` returns the
result for the exact final snapshot and closes the helper. `submit()` returns a request-local
revision, reusing the revision for an identical successful or pending snapshot. `updates()` yields
`PrefetchUpdate(revision, result)` when a matching completion arrives, filtering completions already
superseded by a submission. It retains no update queue: a slow reader receives the newest matching
completion. The iterator ends after finalization or closure drains the active operation.

`PrefetchResult` contains `hits` and an optional `bundle`. Without `budget`, the helper calls
`search()` and `bundle` is `None`. With `budget`, it calls `compile()` and returns the same compiled
bundle with its policy, evidence closure, media, and uncertainty; finalization can reuse this exact
completion without another compile. In compiled mode `budget` controls item/type selection;
`limit` is a search-only setting, and `memory_type`, `occurred_from`, and `occurred_until` must be
omitted rather than silently ignored. Snapshots reject mutable `Path` atoms; use `Blob` or `AssetRef`.

Use `finalize()` when results must match the final query; `latest` remains speculative.
Speculative updates never create observation records or reinforce recalled evidence. As with ordinary
search and compilation, the configured runtime can retain failed-query control signals.

## AsyncContextSource

`AsyncContextSource` calls a supplier that the application provides. It does not create an agent,
search memory, or persist results. The supplier may run tools or an Omni Agentic workflow and
return text supplements; the application decides when to merge them into a compiled bundle:

```python
from sensemem import AsyncContextSource, ContextSupplement


async def provide_context(query):
    observation = await my_agent.observe(query)
    return (ContextSupplement(observation, source="my-agent"),)


source = AsyncContextSource(provide_context)
revision = source.submit("What should I bring?")
try:
    # The application can keep doing independent pipeline work here.
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

Each submission returns a request-local revision. `ContextSourceResult` contains that `revision`
and immutable `supplements`. `latest` is nonblocking; compare its revision with the submitted one
before consuming it. One supplier runs at a time, and a newer queued snapshot replaces the previous
queued snapshot. A completion for an input already superseded is discarded. `finalize()` closes
submission and returns only the exact final revision, propagating its failure rather than reusing
an older result. An identical submission is reused unless it failed.

`close()` discards queued work, cancels the native async supplier, and waits for cleanup.
Cancelling `finalize()` also cancels its supplier. The application owns provider clients and any
work the provider starts outside this coroutine. Like prefetch, submissions reject mutable `Path`
atoms; use immutable text, `Blob`, or `AssetRef` snapshots.

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

AsyncEventStream.process(
    *,
    end_of_stream: bool = False,
) -> AsyncIterator[StreamCommit]

AsyncEventStream.consume(
    chunks: AsyncIterable[EventChunk],
) -> AsyncIterator[StreamCommit]
```

`EventChunk(content, occurred_at, occurred_end, stream_id="default", context=None)` carries one immutable
encoded image/audio/video `Blob` or two synchronized blobs pairing audio with image/video.
`context=None` optionally carries a chunk's `ObservationContext`, including its place and pose.
Times are timezone-aware, ordered, contiguous, and nonoverlapping within each stream. The application
supplies independently decodable observations, such as codec groups remuxed at video keyframes;
a wall-clock chunk timer is not required. The segmenter chooses complete event boundaries.
Boundary precision is limited to chunk edges. Raw-file decoding and sensor capture belong to the
application.

`append()` requires `checkpoint_path`. It returns after the original media and its pending manifest
reference are flushed, without waiting for segmentation, embedding, or other model work. This
acknowledges durable input, not a completed event or searchable memory. Concurrent appends are
serialized at acceptance; one `process()` can assess earlier input while more is appended.

`process()` drains accepted input through a single semantic processor. Adjacent pending observations
are assessed together within the model-window and active-byte budgets. An idle processor returns
without implying EOF. Use `process(end_of_stream=True)` after the source is durably accepted to
assess its final suffix; EOF still does not force semantic completion. The application schedules
this processor independently from source acceptance and consumes its commits. Overlapping
processors, overlapping consumers, or a concurrent `consume()` and `process()` are rejected.

`consume()` remains the combined source-and-processing convenience interface. It assesses input as
it pulls the source and requests EOF when that source ends. It also works without a checkpoint;
use `append()` and `process()` when source acknowledgement must stay independent of model latency.

Each completed event becomes one or more `EPISODIC` records with a shared
`event_segmentation.event_id`. Storage parts split at `max_chunks` original chunks, 127 media
items, or chunk-context changes;
`part_index` and `part_count` identify the full set. Metadata includes the complete event interval,
model, recipe, whole-event `summary`, `summary_coverage="unverified"`, `supplied_modalities`, and
each part's chunk intervals, media counts, original `asset_ids`, and contexts. Each record's interval
and typed context describe its own part, preserving spatial retrieval.
Event IDs include per-chunk source context, so different sources or places with identical media
and intervals remain different events.

Original parts contain media rather than model text. A single-part synopsis is stored through
`interpretation`, with explicit model provenance and unverified modality coverage. For a multipart
event, each part's interpretation contains structural interval information only; the global synopsis
is metadata and does not establish facts at that part's place or time. Every synopsis detail remains
an unverified model interpretation, including a positive sound claim or exact count. Distinguish
the value a synopsis reports from a fact established by the corresponding original media or
independent evidence. An omitted sound or count is unknown, not established absent. Source replay
retains original media and excludes interpretation text. Assessments retain all original event
chunks while they fit `max_chunks` and the active byte budget. Longer events use
the synopsis fallback; completion then does not reanalyze all earlier media to verify the final synopsis.

Incomplete suffixes survive EOF. `max_chunks` bounds one model assessment, including an untrusted
continuation summary, and each completed storage part's original chunk count; its minimum is two.
That summary observation also carries the last analyzed
original chunk as context, without adding a boundary offset. With a checkpoint,
`max_buffer_bytes` bounds one active model window or storage part; older unfinished originals
remain lightweight disk references. Storage parts also split at this byte limit. Without a
checkpoint, the limit bounds all retained unfinished and unacknowledged original media. `max_pending_bytes`
separately bounds durably staged media awaiting admission to that active window; it is not a model
context limit. Pending references reopen without loading their original bytes; size and digest
integrity are checked when processing loads them. Exceeding a budget fails explicitly without
discarding previously accepted input. `max_streams` bounds distinct IDs over the helper's lifetime.
Limits never force completion.
`stream_id` separates transient state and is not an isolation boundary.

`checkpoint_path` automatically saves accepted original media before inference and restores pending
assessments and unacknowledged storage parts after process exit. The manifest and sibling
`<checkpoint_path>.media/` directory form one checkpoint. Reopen it with the same physical memory
directory, segmenter recipe, `max_chunks`, and `max_buffer_bytes`. Retained source files are integrity-checked
incrementally on reopen without loading the entire event. `positions` returns each stream's last accepted end timestamp for
source resumption. Only one helper may own a checkpoint. Without a path, only that live helper
retains unfinished state between calls. See [capture recovery](../omni-streaming-and-interaction-memory.md#resume-continuous-capture).

Checkpoint format v5 pins `max_chunks` and `max_buffer_bytes` to preserve storage-part boundaries
during recovery; a different bound is rejected before processing. It retains the durable `pending`
list and each media reference's `size_bytes`. Earlier v1–v4 manifests are rejected; the runtime
contains no format migration. Finish an old
checkpoint with its original runtime before upgrading, or back up its manifest, media, and memory
store together for explicit operator handling. Do not replace a checkpoint in an active instance.
The bundled event recipe is v3, preserving within-observation part order, source roles, and native
media references. Changing that recipe also requires finishing pending checkpoints with their
original segmenter. Checkpoint format and model recipe are separate versioned contracts.

`context` fills missing contexts when chunks arrive. Default writes use `add()`; `capture=True` defers
write enrichment and skips automatic retrieval, but event segmentation still calls the model.
Capture-mode commits have the same pending-settlement fields described below. The caller owns
the segmenter's lifetime. See [automatic event segmentation](../omni-streaming-and-interaction-memory.md#automatic-event-segmentation)
for an example and [the backend contract](python-backends.md#eventsegmentationbackend) for custom
segmenters.

Cancellation drains the active model call and preserves resumable input and validated assessments.
A final write already started follows the capture reducer's commit rule.

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
retrieval is needed. By default, finals use `add()` and yield searchable records. See
[omni streaming and interaction memory](../omni-streaming-and-interaction-memory.md) for event
semantics and complete examples.
