# Omni streaming and interaction memory

Connect continuous audio and video to durable event memory. `AsyncEventStream` understands successive media chunks and commits model-confirmed complete events while preserving their original media and time intervals. Applications with their own finality signals can instead use `StreamEvent`:

1. An `UPDATE` is a complete current snapshot used only for speculative search.
2. A `FINAL` is a completed observation eligible for exactly one durable write.
3. A `CANCEL` or end-of-stream before finality writes nothing.

`AsyncAudioStream` and `AsyncVisionStream` map canonical capture values onto that contract. Sensor capture, reconnection, packet decoding, pixel conversion, synchronization, and encoding remain application or adapter work.

![Streaming finality separates speculative recall from durable capture; personalization separates affect, traits, and authorized response guidance.](assets/tech-report/streaming-personalization.png)

## Choose the smallest stream API

| Need | API |
| --- | --- |
| Lazily add completed independent observations | `Memory.add_stream()` or `AsyncMemory.add_stream()` |
| Deliver an answer while the model is still writing it | `Memory.ask_stream()` or `AsyncMemory.ask_stream()` |
| Acknowledge already completed observations without waiting for models | `Memory.capture()` with `Memory.settle()`, or explicit-finality stream APIs with `capture=True` |
| Search while one query snapshot is changing | `AsyncOmniPrefetch` |
| Understand successive audio/video chunks and find complete events | `AsyncEventStream` |
| Acknowledge continuous media before semantic models finish | `AsyncEventStream.append()` with a checkpoint and host-scheduled `process()` |
| Associate speculative snapshots with final commits | `AsyncCaptureStream` |
| Normalize PCM, VAD, and ASR state | `AsyncAudioStream` |
| Normalize encoded frames and visual descriptions | `AsyncVisionStream` |

Every reducer takes an `AsyncMemory` (`async_memory` below). Partial query snapshots must be immutable: prefetch rejects `Path` atoms whose bytes could change during search—pass `Blob` or `AssetRef`. For streamed answer output, see [`ask_stream`](api/python-memory.md#ask_stream).

## Automatic event segmentation

Supply independently decodable `EventChunk` values in time order (for example, codec groups beginning at video keyframes). The segmenter receives a bounded media window and returns complete events from its beginning; ongoing actions stay pending across pauses.

```python
from pathlib import Path

from sensemem import AsyncEventStream, Blob, EventChunk


async def video_chunks():
    for encoded_clip, occurred_at, occurred_end in codec_groups:
        yield EventChunk(
            Blob(encoded_clip, "video/mp4"),
            occurred_at=occurred_at,
            occurred_end=occurred_end,
        )


events = AsyncEventStream(
    async_memory,
    segmenter=segmenter,
    checkpoint_path=Path("capture-checkpoint.json"),
)
async for commit in events.consume(video_chunks()):
    print(commit.record.id, commit.record.occurred_at, commit.record.occurred_end)
```

`segmenter` is a caller-owned `EventSegmentationBackend`, such as `OpenAICompletion` configured for a media-capable endpoint. To pair synchronized camera and microphone streams, pass a two-element tuple of one audio `Blob` and one image/video `Blob` with a shared interval.

For continuous sources that must keep accepting media while a model is busy, separate durable acceptance from semantic processing:

```python
async def receive_media(chunks):
    async for chunk in chunks:
        await events.append(chunk)
        # Durable acceptance completes before model assessment.


async def analyze_available(*, source_ended=False):
    async for commit in events.process(end_of_stream=source_ended):
        print(commit.record.id)
```

`append()` requires `checkpoint_path` and returns once original media and its pending reference are flushed. The host schedules one `process()` loop independently; `process()` coalesces arrived chunks within the active window and byte budgets, and `process(end_of_stream=True)` requests a final assessment without forcing unfinished activities to close. See the [live demo](live-demo.md#implementation) for a complete continuous adapter.

Each confirmed event writes one or more `EPISODIC` records (splitting at `max_chunks`, 127 media parts, `max_buffer_bytes`, or a change in chunk context). Original content holds only captured media; the synopsis is stored separately as a `model_interpretation` part with `summary_coverage="unverified"` and `supplied_modalities`. Metadata under `event_segmentation` records `model`, `recipe`, shared `event_id`, `part_index`, `part_count`, the complete event interval, `summary`, and per-part chunk intervals, `asset_ids`, and `context`. Original media can be read via [`read_asset`](api/python-memory.md#read_asset) and replayed without copying model interpretations.

Supply `EventChunk(..., context=ObservationContext(...))` or a `context` sampler callback to preserve per-chunk place and pose at arrival time. While an unfinished event fits within `max_chunks` and `max_buffer_bytes`, assessments retain all original chunks; longer events roll forward using an explicitly labelled `complete=False` continuation synopsis plus the last analyzed chunk as context while keeping all original chunks on disk for final storage.

Bounds default to `max_streams=32`, `max_chunks=32` (minimum 2), `max_buffer_bytes=64 MiB` (active analysis window or storage part when checkpointed), and `max_pending_bytes=64 MiB` (durably staged input awaiting analysis). Each stream's chunks must have timezone-aware, contiguous, nonoverlapping intervals. `capture=True` defers write enrichment to `settle()` while segmentation still runs on the event stream. See [`AsyncEventStream`](api/python-streams.md#asynceventstream) and [`EventSegmentationBackend`](api/python-backends.md#eventsegmentationbackend).

### Resume continuous capture

With `checkpoint_path`, accepted chunks flush to a private manifest and content-addressed files in `<checkpoint_path>.media/` before inference. Reopen the same `data_dir` with the same `checkpoint_path`, segmenter recipe, `max_chunks`, and `max_buffer_bytes`, then call `process()` (or `consume()`). `events.positions` maps each `stream_id` to its last accepted end timestamp for resuming the source. Only one helper may own a checkpoint, and overlapping `process()`/`consume()` calls are rejected. Back up the manifest, `<checkpoint_path>.media/`, and `data_dir` together.

## Add completed observations

`Memory.add_stream()` consumes a lazy iterable of `ContentInput` or `StreamInput` values, committing each item before pulling the next so a later failure preserves the committed prefix.

```python
from datetime import datetime, timedelta, timezone

from sensemem import Blob, MemoryCategory, StreamInput

started = datetime(2026, 8, 31, 9, tzinfo=timezone.utc)
observation = StreamInput(
    (
        "The user is looking for the red toolbox.",
        Blob(frame_bytes, "image/png", "frame.png"),
        Blob(audio_bytes, "audio/wav", "speech.wav"),
    ),
    occurred_at=started,
    occurred_end=started + timedelta(seconds=2),
    metadata={"source": "companion_robot", "sequence": 17},
    memory_type=MemoryCategory.EPISODIC,
)

for source_record in memory.add_stream((observation,)):
    print(source_record.id)
```

`AsyncMemory.add_stream()` accepts an `AsyncIterable` and yields with `async for`. Keep mutable ASR hypotheses and overwritten file paths out of `add_stream()`.

## Fast capture

Use `capture()` when observations are final but the caller cannot block on transcription, embedding, indexing, and formation:

```python
for observation in burst:
    memory.capture(observation)

while memory.settle(limit=32):
    pass
```

Inspect `pending_captures()` for `attempts`, `last_error`, and `awaiting` (`"enrichment"` vs `"formation"`). Retry a record past its retry ceiling with `settle(memory_ids=...)`. See [memory operations](api/python-memory.md).

## Prefetch a changing query

`AsyncOmniPrefetch` wraps an open `AsyncMemory` and overlaps retrieval with an unfinished turn without storing partial input. Every submission is the complete current snapshot:

```python
from sensemem import AsyncOmniPrefetch

prefetch = AsyncOmniPrefetch(async_memory, limit=8)
prefetch.submit((partial_text, latest_frame, current_audio_window))
prefetch.submit((newer_text, latest_frame, current_audio_window))

result = await prefetch.finalize((final_text, final_frame, final_audio_window))
hits = result.hits
```

One retrieval runs at a time; newer submissions replace unstarted queued snapshots. Consume `updates()` for non-superseded `PrefetchUpdate(revision, result)` completions. Pass `budget=ContextBudget(...)` instead of `limit` to prefetch a `ContextBundle` directly. Call `await prefetch.close()` if a turn is abandoned so any dispatched search drains cleanly.

## Capture-event reducer

`AsyncCaptureStream` tracks independent `stream_id` lifecycles:

| Phase | Payload | Retrieval | Durable write |
| --- | --- | --- | --- |
| `UPDATE` | Complete current immutable `ContentInput` snapshot | Speculative and coalesced | Never |
| `FINAL` | Exact final `ContentInput` or `StreamInput` | Exact-final confirmation | Once |
| `CANCEL` | None | Discarded and drained | Never |
| EOF before `FINAL` | — | Every open stream discarded and drained | Never |

```python
from sensemem import AsyncCaptureStream, StreamEvent, StreamPhase


async def camera_events():
    yield StreamEvent(StreamPhase.UPDATE, left_frame, stream_id="camera-left")
    yield StreamEvent(StreamPhase.UPDATE, right_frame, stream_id="camera-right")
    yield StreamEvent(StreamPhase.FINAL, left_clip, stream_id="camera-left")
    yield StreamEvent(StreamPhase.FINAL, right_clip, stream_id="camera-right")


capture = AsyncCaptureStream(async_memory, limit=8)
async for commit in capture.consume(camera_events()):
    print(commit.stream_id, commit.record.id, commit.prefetch, commit.retrieval_error)
```

`max_streams` defaults to 32. Passing `capture=True` to `AsyncCaptureStream`, `AsyncAudioStream`, `AsyncVisionStream`, or `add_stream()` commits finals via `capture()` and skips automatic retrieval (`pending_settlement=True`, `prefetch=None`):

```python
audio = AsyncAudioStream(async_memory, limit=8, capture=True)
async for commit in audio.consume(audio_packets()):
    assert commit.pending_settlement

while await async_memory.settle(limit=32):
    pass
```

## Native audio protocol

`AsyncAudioStream` reduces `PCMChunk`, `VADPacket`, `ASRPartial`, and `AcousticBoundary` into WAV observations:

```python
from sensemem import (
    AcousticBoundary,
    ASRPartial,
    AsyncAudioStream,
    AudioBoundary,
    PCMChunk,
    VADPacket,
)


async def audio_packets():
    yield VADPacket(True, stream_id="headset")
    yield PCMChunk(first_pcm, sample_rate_hz=16_000, stream_id="headset")
    yield ASRPartial("where is the red", stream_id="headset")
    yield PCMChunk(second_pcm, sample_rate_hz=16_000, stream_id="headset")
    yield ASRPartial("where is the red toolbox", stream_id="headset")
    yield AcousticBoundary(AudioBoundary.END, stream_id="headset")


audio = AsyncAudioStream(async_memory, limit=8)
async for commit in audio.consume(audio_packets()):
    print(commit.stream_id, commit.record.id)
```

Native-audio embedders receive changing WAV snapshots directly; text-only embedders use `ASRPartial` or a configured transcription backend. `VADPacket(active=False)` and `AudioBoundary.END` finalize; `CANCEL` and incomplete EOF never write.

## Native vision protocol

`AsyncVisionStream` reduces `VisionFrame`, `VisionPartial`, and `SceneBoundary` and keeps the latest frame per scene as its durable keyframe:

```python
from sensemem import (
    AsyncVisionStream,
    Blob,
    SceneBoundary,
    VisionBoundary,
    VisionFrame,
    VisionPartial,
)


async def vision_packets():
    yield VisionFrame(Blob(first_jpeg, "image/jpeg"), stream_id="camera-left")
    yield VisionPartial("a red toolbox beside the door", stream_id="camera-left")
    yield VisionFrame(Blob(latest_jpeg, "image/jpeg"), stream_id="camera-left")
    yield SceneBoundary(VisionBoundary.END, stream_id="camera-left")


vision = AsyncVisionStream(async_memory, limit=8)
async for commit in vision.consume(vision_packets()):
    print(commit.stream_id, commit.record.id)
```

With a text-only embedder, `VisionPartial` or a configured `VisionDescriptionBackend` supplies text; otherwise finalization fails explicitly rather than discarding frames.

## Interaction-derived records and observability

Organize interaction memories with `SEMANTIC` (preferences/facts), `EPISODIC` (situated observations), and `PROCEDURAL` (response guidance returned as evidence, never executed). For typed formation, affect cues, corroborated traits, and host-authorized response policies, see [typed memory](typed-memory.md) and [affective memory](affective-memory.md).

Every completed stream item emits `sensemem.add` (or `sensemem.capture` when `capture=True`), and every started speculative search emits `sensemem.search`; see [observability](observability.md).
