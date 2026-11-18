"""Async observation streams: semantic events, omni prefetch, capture, audio, and vision."""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TypeVar, cast

from sensemem._content import content_atoms, snapshot_content
from sensemem._validation import nonempty_text, positive_int
from sensemem.content import Blob, ContentAtom, ContentInput
from sensemem.context_values import ContextBudget
from sensemem.events import EventChunk
from sensemem.exceptions import SenseMemError, ValidationError
from sensemem.infrastructure.local.event_checkpoint import EventCheckpoint
from sensemem.kernel.content import declared_atom_modality
from sensemem.kernel.events import EventSegmentation
from sensemem.kernel.temporal import resolved_reference_at, search_occurrence_range
from sensemem.kernel.validation import optional_memory_type, validate_limit
from sensemem.memory import AsyncMemory
from sensemem.models.base import EventSegmentationBackend
from sensemem.spatial import ObservationContext
from sensemem.stream_values import (
    AcousticBoundary,
    ASRPartial,
    AudioBoundary,
    AudioStreamPacket,
    PCMChunk,
    PrefetchResult,
    PrefetchUpdate,
    SceneBoundary,
    StreamCommit,
    StreamEvent,
    StreamInput,
    StreamPhase,
    VADPacket,
    VisionBoundary,
    VisionFrame,
    VisionPartial,
    VisionStreamPacket,
)
from sensemem.types import MemoryCategory, Modality

_T = TypeVar("_T")


@dataclass(slots=True)
class _AudioStreamState:
    pcm: bytearray
    sample_rate_hz: int | None = None
    channels: int | None = None
    sample_width_bytes: int | None = None
    transcript: str = ""
    occurred_at: datetime | None = None


@dataclass(slots=True)
class _VisionStreamState:
    image: Blob | None = None
    description: str = ""
    occurred_at: datetime | None = None
    last_occurred_at: datetime | None = None


class AsyncOmniPrefetch:
    """Coalesce evolving omni query snapshots into one in-flight retrieval per turn."""

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryCategory | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        budget: ContextBudget | None = None,
    ) -> None:
        validate_limit(limit, maximum=100)
        occurred_from, occurred_until = search_occurrence_range(
            occurred_from,
            occurred_until,
        )
        self._memory = memory
        self._limit = limit
        self._memory_type = optional_memory_type(memory_type)
        self._reference_at = resolved_reference_at(reference_at)
        self._occurred_from = occurred_from
        self._occurred_until = occurred_until
        if budget is not None and not isinstance(budget, ContextBudget):
            raise ValidationError("prefetch budget must be ContextBudget")
        if budget is not None and any(
            value is not None for value in (memory_type, occurred_from, occurred_until)
        ):
            raise ValidationError("compiled prefetch uses budget filters instead of search filters")
        self._budget = budget
        self._revision = 0
        self._submitted: tuple[int, ContentInput] | None = None
        self._pending: tuple[int, ContentInput] | None = None
        self._latest: PrefetchResult | None = None
        self._latest_revision = 0
        self._failure: tuple[int, Exception] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._changed = asyncio.Event()

    @property
    def latest(self) -> PrefetchResult | None:
        """Return the newest completed result without waiting."""
        return self._latest

    def submit(self, query: ContentInput) -> int:
        """Queue a complete snapshot and return its request-local revision."""
        if self._closed:
            raise ValidationError("prefetch is closed")
        loop = asyncio.get_running_loop()
        snapshot = snapshot_content(query)
        if self._submitted is not None and self._submitted[1] == snapshot:
            failed = self._failure is not None and self._failure[0] == self._submitted[0]
            if not failed:
                return self._submitted[0]
        self._revision += 1
        revision = self._revision
        self._submitted = (revision, snapshot)
        self._pending = self._submitted
        self._failure = None
        if self._worker is None:
            self._worker = loop.create_task(self._run())
        return revision

    async def updates(self) -> AsyncIterator[PrefetchUpdate]:
        """Yield matching completions as they arrive, without retaining an update queue."""
        seen = 0
        while True:
            if (
                self._latest is not None
                and self._latest_revision > seen
                and self._submitted is not None
                and self._latest_revision == self._submitted[0]
            ):
                seen = self._latest_revision
                yield PrefetchUpdate(seen, self._latest)
            elif self._closed and self._worker is None:
                return
            else:
                self._changed.clear()
                await self._changed.wait()

    async def finalize(self, query: ContentInput | None = None) -> PrefetchResult:
        """Finish the turn and return a result for the exact final snapshot."""
        if self._closed:
            raise ValidationError("prefetch is closed")
        if query is None:
            if self._submitted is None:
                raise ValidationError("prefetch has no submitted query")
        else:
            self.submit(query)
        assert self._submitted is not None
        target = self._submitted[0]
        self._closed = True
        self._changed.set()
        worker = self._worker
        if worker is not None:
            await asyncio.shield(worker)
        if self._latest is not None and self._latest_revision == target:
            return self._latest
        if self._failure is not None and self._failure[0] == target:
            raise self._failure[1]
        raise RuntimeError("prefetch completed without its final revision")

    async def close(self) -> None:
        """Discard queued snapshots and drain the operation that may already be running."""
        self._closed = True
        self._pending = None
        self._changed.set()
        worker = self._worker
        if worker is not None:
            await asyncio.shield(worker)

    async def _run(self) -> None:
        try:
            while self._pending is not None:
                revision, query = self._pending
                self._pending = None
                try:
                    if self._budget is None:
                        result = await self._memory.search(
                            query,
                            limit=self._limit,
                            memory_type=self._memory_type,
                            reference_at=self._reference_at,
                            occurred_from=self._occurred_from,
                            occurred_until=self._occurred_until,
                        )
                        completed = PrefetchResult(hits=result.hits)
                    else:
                        bundle = await self._memory.compile(
                            query, budget=self._budget, reference_at=self._reference_at
                        )
                        completed = PrefetchResult(hits=bundle.hits, bundle=bundle)
                except Exception as error:
                    self._failure = (revision, error)
                    self._changed.set()
                    continue
                self._latest = completed
                self._latest_revision = revision
                self._changed.set()
        finally:
            self._worker = None
            self._changed.set()


class AsyncCaptureStream:
    """Reduce associated capture streams into retrieval and durable memories.

    `capture=True` commits each `FINAL` through `Memory.capture()` instead of `Memory.add()`, so
    automatic retrieval is skipped and every `StreamCommit` reports `pending_settlement` with
    `prefetch=None`. The host then owes `settle()`; the default stays the strong `add()`.
    """

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryCategory | None = None,
        reference_at: datetime | None = None,
        max_streams: int = 32,
        capture: bool = False,
    ) -> None:
        if not isinstance(memory, AsyncMemory):
            raise ValidationError("memory must be an AsyncMemory")
        validate_limit(limit, maximum=100)
        if not isinstance(capture, bool):
            raise ValidationError("capture must be a boolean")
        self._memory = memory
        self._limit = limit
        self._memory_type = optional_memory_type(memory_type)
        self._reference_at = resolved_reference_at(reference_at)
        self._max_streams = positive_int(max_streams, "max_streams")
        self._capture = capture

    async def consume(  # noqa: C901 - the three-state reducer is intentionally inline
        self,
        events: AsyncIterable[StreamEvent],
    ) -> AsyncIterator[StreamCommit]:
        """Yield only final observations; partials, cancellation, and EOF never write."""
        if not isinstance(events, AsyncIterable):
            raise ValidationError("events must be an async iterable of StreamEvent values")
        prefetches: dict[str, AsyncOmniPrefetch] = {}
        try:
            async for event in events:
                if not isinstance(event, StreamEvent):
                    raise ValidationError("events must contain StreamEvent values")
                stream_id = event.stream_id
                if event.phase is StreamPhase.UPDATE:
                    prefetch = self._prefetch_for(prefetches, stream_id)
                    assert event.item is not None and not isinstance(event.item, StreamInput)
                    if not self._capture:
                        prefetch.submit(event.item)
                    continue
                if event.phase is StreamPhase.CANCEL:
                    cancelled_prefetch = (
                        prefetches.pop(stream_id) if stream_id in prefetches else None
                    )
                    if cancelled_prefetch is not None:
                        await cancelled_prefetch.close()
                    continue
                assert event.item is not None
                prefetch = self._prefetch_for(prefetches, stream_id)
                item = event.item
                retrieval_error: Exception | None = None
                retrieval: PrefetchResult | None = None
                if not self._capture:
                    try:
                        retrieval = await prefetch.finalize(self._final_query(item))
                    except Exception as error:
                        retrieval_error = error
                        # An early rejection can leave an UPDATE search in flight.
                        await prefetch.close()
                prefetches.pop(stream_id, None)
                # ponytail: FINAL is the commit point; a cancellable storage transaction would
                # be needed to revoke it safely once the worker thread has started.
                commit = asyncio.create_task(
                    self._memory._write_stream_item(item, capture=self._capture)
                )
                cancelled = False
                while not commit.done():
                    try:
                        await asyncio.shield(commit)
                    except asyncio.CancelledError:
                        cancelled = True
                    except Exception:
                        if cancelled:
                            raise asyncio.CancelledError from None
                        raise
                try:
                    record = commit.result()
                except Exception:
                    if cancelled:
                        raise asyncio.CancelledError from None
                    raise
                if cancelled:
                    raise asyncio.CancelledError
                if retrieval_error is not None:
                    yield StreamCommit(
                        record=record,
                        prefetch=None,
                        retrieval_error=(
                            retrieval_error.code
                            if isinstance(retrieval_error, SenseMemError)
                            else "retrieval_failed"
                        ),
                        stream_id=stream_id,
                        pending_settlement=self._capture,
                    )
                else:
                    yield StreamCommit(
                        record=record,
                        prefetch=retrieval,
                        stream_id=stream_id,
                        pending_settlement=self._capture,
                    )
        finally:
            for prefetch in tuple(prefetches.values()):
                await prefetch.close()

    def _prefetch_for(
        self,
        prefetches: dict[str, AsyncOmniPrefetch],
        stream_id: str,
    ) -> AsyncOmniPrefetch:
        prefetch = prefetches.get(stream_id)
        if prefetch is not None:
            return prefetch
        if len(prefetches) >= self._max_streams:
            raise ValidationError("capture stream exceeds max_streams")
        prefetch = AsyncOmniPrefetch(
            self._memory,
            limit=self._limit,
            memory_type=self._memory_type,
            reference_at=self._reference_at,
        )
        prefetches[stream_id] = prefetch
        return prefetch

    def _final_query(self, item: ContentInput | StreamInput) -> ContentInput:
        if not isinstance(item, StreamInput) or (
            item.transcript is None and item.description is None
        ):
            return item.content if isinstance(item, StreamInput) else item
        capabilities = self._memory.capabilities.embedding
        if Modality.TEXT not in capabilities:
            return item.content
        routed: list[ContentAtom] = [
            value for value in (item.transcript, item.description) if value is not None
        ]
        for atom in content_atoms(item.content):
            modality = declared_atom_modality(atom)
            if (
                modality is Modality.AUDIO
                and item.transcript is not None
                and modality not in capabilities
            ):
                continue
            if (
                modality in {Modality.IMAGE, Modality.VIDEO}
                and item.description is not None
                and modality not in capabilities
            ):
                continue
            routed.append(atom)
        return routed[0] if len(routed) == 1 else tuple(routed)


StreamContext = ObservationContext | Callable[[], ObservationContext | None] | None


"""Provenance for streamed observations: one fixed value, or a sampler read at each boundary."""


def _stream_context(context: StreamContext) -> Callable[[], ObservationContext | None]:
    """Normalize a fixed context or a per-observation sampler into one sampler.

    A capture stream outlives the observations it commits, so a robot's pose is not a property
    of the stream. A callable is read once per closed observation, which is the only moment at
    which the adapter knows which interval the pose belongs to. A fixed `ObservationContext`
    stays accepted because a static camera really does have one.
    """
    if context is None:
        return lambda: None
    if isinstance(context, ObservationContext):
        return lambda: context
    if not callable(context):
        raise ValidationError("context must be an ObservationContext or a callable returning one")
    return context


class AsyncEventStream:
    """Understand continuous timed media and commit complete semantic events automatically.

    Capture windows are independently encoded audio, images, video, or synchronized sound and
    visual media. The segmenter decides finality, at window boundaries, from the whole unfinished
    event. EOF triggers one final assessment and retains any still-incomplete suffix. The caller
    owns the segmenter; the memory instance owns execution, storage, and ordinary enrichment.
    """

    def __init__(
        self,
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
    ) -> None:
        if not isinstance(memory, AsyncMemory):
            raise ValidationError("memory must be an AsyncMemory")
        if not isinstance(segmenter, EventSegmentationBackend):
            raise ValidationError("segmenter must be an EventSegmentationBackend")
        nonempty_text(segmenter.event_model, "event_model")
        nonempty_text(segmenter.event_recipe, "event_recipe")
        self._memory = memory
        self._context = _stream_context(context)
        max_streams = positive_int(max_streams, "max_streams")
        max_chunks = positive_int(max_chunks, "max_chunks")
        max_buffer_bytes = positive_int(max_buffer_bytes, "max_buffer_bytes")
        max_pending_bytes = positive_int(max_pending_bytes, "max_pending_bytes")
        if checkpoint_path is not None and not isinstance(checkpoint_path, Path):
            raise ValidationError("checkpoint_path must be a Path")
        if max_chunks < 2:
            raise ValidationError("max_chunks must allow a continuation and one new chunk")
        core = memory._memory
        self._plane = EventSegmentation(
            tracer=core._tracer,
            segmenter=segmenter,
            lifecycle=core._lifecycle,
            materializer=core._materializer,
            hydrator=core._hydrator,
            max_streams=max_streams,
            max_chunks=max_chunks,
            max_buffer_bytes=max_buffer_bytes,
            max_pending_bytes=max_pending_bytes,
            checkpoint=(
                None
                if checkpoint_path is None
                else EventCheckpoint(checkpoint_path, data_dir=core.data_dir)
            ),
        )
        self._consuming = False
        self._processing = False
        self._checkpoint_path = checkpoint_path
        self._capture = AsyncCaptureStream(
            memory,
            limit=limit,
            memory_type=MemoryCategory.EPISODIC,
            reference_at=reference_at,
            max_streams=max_streams,
            capture=capture,
        )

    @property
    def positions(self) -> dict[str, datetime]:
        """Return each stream's accepted capture end; resume source input at this boundary."""
        return self._plane.positions

    async def consume(self, chunks: AsyncIterable[EventChunk]) -> AsyncIterator[StreamCommit]:
        """Yield storage parts of complete events; accepted input survives interrupted calls."""
        if not isinstance(chunks, AsyncIterable):
            raise ValidationError("chunks must be an async iterable of EventChunk values")
        if self._consuming or self._processing:
            raise ValidationError("event stream already has an active consumer")
        self._consuming = True
        try:
            async with aclosing(self._process(end_of_stream=False)) as processing:
                async for commit in processing:
                    yield commit
            async for chunk in chunks:
                await self._append(chunk, deferred=False)
                async with aclosing(self._process(end_of_stream=False)) as processing:
                    async for commit in processing:
                        yield commit
            async with aclosing(self._process(end_of_stream=True)) as processing:
                async for commit in processing:
                    yield commit
        finally:
            self._consuming = False

    async def append(self, chunk: EventChunk) -> None:
        """Flush a window to the native checkpoint without waiting for any model work.

        Requires ``checkpoint_path``. Concurrent appends are serialized at durable acceptance;
        one ``process()`` may assess earlier input concurrently. ``max_pending_bytes`` bounds
        unassessed staged media separately from the active ``max_buffer_bytes`` budget.
        """
        if self._checkpoint_path is None:
            raise ValidationError("append requires checkpoint_path for durable acceptance")
        await self._append(chunk, deferred=True)

    async def _append(self, chunk: EventChunk, *, deferred: bool) -> None:
        if not isinstance(chunk, EventChunk):
            raise ValidationError("chunks must contain EventChunk values")
        if chunk.context is None:
            chunk = replace(chunk, context=self._context())
        await self._drain_call(partial(self._plane.stage, chunk, deferred=deferred))

    async def process(self, *, end_of_stream: bool = False) -> AsyncIterator[StreamCommit]:
        """Drain arrived input through one bounded semantic processor.

        Adjacent queued windows are assessed together within ``max_chunks`` and the active
        byte budget. Idle input does not imply EOF. Explicit EOF assesses the final suffix
        without forcing completion; source bytes remain resumable when a model fails.
        """
        if self._consuming:
            raise ValidationError("event stream already has an active consumer")
        async with aclosing(self._process(end_of_stream=end_of_stream)) as processing:
            async for commit in processing:
                yield commit

    async def _process(self, *, end_of_stream: bool) -> AsyncGenerator[StreamCommit, None]:
        if not isinstance(end_of_stream, bool):
            raise ValidationError("end_of_stream must be a boolean")
        if self._processing:
            raise ValidationError("event stream already has an active processor")
        self._processing = True
        try:
            async with (
                aclosing(self._events(end_of_stream=end_of_stream)) as events,
                aclosing(
                    cast(AsyncGenerator[StreamCommit, None], self._capture.consume(events))
                ) as captures,
            ):
                async for commit in captures:
                    await self._drain_call(self._plane.acknowledge)
                    yield commit
        finally:
            self._processing = False

    async def _events(self, *, end_of_stream: bool) -> AsyncGenerator[StreamEvent, None]:
        plane = self._plane
        finished: set[str] = set()
        while (event := plane.peek()) is not None:
            yield event
        while assessed := await self._drain_call(
            partial(plane.advance, end_of_stream=end_of_stream)
        ):
            stream_id, final = assessed
            if final:
                finished.add(stream_id)
            while (event := plane.peek()) is not None:
                yield event
        if end_of_stream:
            for stream_id in plane.pending_streams:
                if stream_id not in finished:
                    await self._drain_call(partial(plane.finish, stream_id))
                    while (event := plane.peek()) is not None:
                        yield event

    async def _drain_call(self, call: Callable[[], _T]) -> _T:
        assessment = asyncio.create_task(self._memory._memory._execution.run_in_worker(call))
        cancelled = False
        # The next consume may reuse this state or close the caller's segmenter. Drain active
        # model calls and checkpoint writes before cancellation releases that ownership boundary.
        while not assessment.done():
            try:
                await asyncio.shield(assessment)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                if not cancelled:
                    raise
        if cancelled:
            if not assessment.cancelled():
                assessment.exception()
            raise asyncio.CancelledError
        return assessment.result()


class AsyncAudioStream:
    """Normalize PCM, VAD, ASR, and acoustic boundaries into associated capture events."""

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryCategory | None = None,
        context: StreamContext = None,
        reference_at: datetime | None = None,
        max_streams: int = 32,
        capture: bool = False,
    ) -> None:
        if not isinstance(memory, AsyncMemory):
            raise ValidationError("memory must be an AsyncMemory")
        self._max_streams = positive_int(max_streams, "max_streams")
        self._written_type = optional_memory_type(memory_type) or MemoryCategory.SEMANTIC
        self._context = _stream_context(context)
        capabilities = memory.capabilities.embedding
        self._native_audio = Modality.AUDIO in capabilities
        self._native_text = Modality.TEXT in capabilities
        self._max_pcm_bytes = memory._memory._storage.assets.max_bytes - 44
        self._capture = AsyncCaptureStream(
            memory,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            max_streams=max_streams,
            capture=capture,
        )

    async def consume(
        self,
        packets: AsyncIterable[AudioStreamPacket],
    ) -> AsyncIterator[StreamCommit]:
        """Yield one associated commit for each VAD or acoustic end boundary."""
        if not isinstance(packets, AsyncIterable):
            raise ValidationError("packets must be an async iterable of audio stream values")
        async for commit in self._capture.consume(self._events(packets)):
            yield commit

    async def _events(  # noqa: C901 - packet kinds form one explicit stream state machine
        self,
        packets: AsyncIterable[AudioStreamPacket],
    ) -> AsyncIterator[StreamEvent]:
        states: dict[str, _AudioStreamState] = {}
        async for packet in packets:
            if not isinstance(packet, (PCMChunk, VADPacket, ASRPartial, AcousticBoundary)):
                raise ValidationError("packets contain an invalid audio stream value")
            stream_id = packet.stream_id
            if isinstance(packet, AcousticBoundary):
                if packet.boundary is AudioBoundary.CANCEL:
                    states.pop(stream_id, None)
                    yield StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
                    continue
                if packet.boundary is AudioBoundary.END:
                    final = self._finish(states, stream_id, packet.occurred_at)
                    if final is not None:
                        yield final
                    continue
                current = states.get(stream_id)
                if current is not None and (current.pcm or current.transcript):
                    raise ValidationError("audio stream started before its prior boundary")
                if current is None:
                    self._state_for(states, stream_id, packet.occurred_at)
                else:
                    current.occurred_at = packet.occurred_at
                continue
            if isinstance(packet, VADPacket):
                if not packet.active:
                    final = self._finish(states, stream_id, packet.occurred_at)
                    if final is not None:
                        yield final
                else:
                    self._state_for(states, stream_id, packet.occurred_at)
                continue
            state = self._state_for(states, stream_id, packet.occurred_at)
            if isinstance(packet, ASRPartial):
                state.transcript = packet.text
            else:
                self._append_pcm(state, packet)
            query = self._query(state)
            if query is not None:
                yield StreamEvent(StreamPhase.UPDATE, query, stream_id)

    def _state_for(
        self,
        states: dict[str, _AudioStreamState],
        stream_id: str,
        occurred_at: datetime | None,
    ) -> _AudioStreamState:
        state = states.get(stream_id)
        if state is not None:
            if state.occurred_at is None:
                state.occurred_at = occurred_at
            return state
        if len(states) >= self._max_streams:
            raise ValidationError("audio stream exceeds max_streams")
        state = _AudioStreamState(bytearray(), occurred_at=occurred_at)
        states[stream_id] = state
        return state

    def _append_pcm(self, state: _AudioStreamState, packet: PCMChunk) -> None:
        sample_format = (
            packet.sample_rate_hz,
            packet.channels,
            packet.sample_width_bytes,
        )
        current_format = (
            state.sample_rate_hz,
            state.channels,
            state.sample_width_bytes,
        )
        if state.sample_rate_hz is None:
            state.sample_rate_hz, state.channels, state.sample_width_bytes = sample_format
        elif current_format != sample_format:
            raise ValidationError("PCM format changed before an audio boundary")
        if len(state.pcm) + len(packet.data) > self._max_pcm_bytes:
            raise ValidationError("PCM stream exceeds the local asset size limit")
        state.pcm.extend(packet.data)

    def _query(self, state: _AudioStreamState) -> ContentInput | None:
        if self._native_audio and state.pcm:
            audio = _pcm_blob(state)
            if self._native_text and state.transcript:
                return (state.transcript, audio)
            return audio
        return state.transcript or None

    def _finish(
        self,
        states: dict[str, _AudioStreamState],
        stream_id: str,
        occurred_end: datetime | None,
    ) -> StreamEvent | None:
        state = states.pop(stream_id, None)
        if state is None:
            return None
        occurred_at, occurred_end = _audio_interval(state, occurred_end)
        # Sampled here, at the boundary, so the pose belongs to the interval being committed.
        context = self._context()
        if state.pcm:
            item = StreamInput(
                _pcm_blob(state),
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
                transcript=state.transcript or None,
            )
        elif state.transcript:
            item = StreamInput(
                state.transcript,
                occurred_at=occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
            )
        else:
            return StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
        return StreamEvent(StreamPhase.FINAL, item, stream_id)


class AsyncVisionStream:
    """Normalize image frames, visual descriptions, and scene boundaries."""

    def __init__(
        self,
        memory: AsyncMemory,
        *,
        limit: int = 10,
        memory_type: MemoryCategory | None = None,
        context: StreamContext = None,
        reference_at: datetime | None = None,
        max_streams: int = 32,
        capture: bool = False,
    ) -> None:
        if not isinstance(memory, AsyncMemory):
            raise ValidationError("memory must be an AsyncMemory")
        self._memory = memory
        self._max_streams = positive_int(max_streams, "max_streams")
        self._written_type = optional_memory_type(memory_type) or MemoryCategory.SEMANTIC
        self._context = _stream_context(context)
        capabilities = memory.capabilities.embedding
        self._native_image = Modality.IMAGE in capabilities
        self._native_text = Modality.TEXT in capabilities
        self._capture = AsyncCaptureStream(
            memory,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            max_streams=max_streams,
            capture=capture,
        )
        self._capture_mode = capture

    async def consume(
        self,
        packets: AsyncIterable[VisionStreamPacket],
    ) -> AsyncIterator[StreamCommit]:
        """Yield one associated commit for each completed visual scene."""
        if not isinstance(packets, AsyncIterable):
            raise ValidationError("packets must be an async iterable of vision stream values")
        async for commit in self._capture.consume(self._events(packets)):
            yield commit

    async def _events(  # noqa: C901 - packet kinds form one explicit stream state machine
        self,
        packets: AsyncIterable[VisionStreamPacket],
    ) -> AsyncIterator[StreamEvent]:
        states: dict[str, _VisionStreamState] = {}
        async for packet in packets:
            if not isinstance(packet, (VisionFrame, VisionPartial, SceneBoundary)):
                raise ValidationError("packets contain an invalid vision stream value")
            stream_id = packet.stream_id
            if isinstance(packet, SceneBoundary):
                if packet.boundary is VisionBoundary.CANCEL:
                    states.pop(stream_id, None)
                    yield StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
                    continue
                if packet.boundary is VisionBoundary.END:
                    final = await self._finish(states, stream_id, packet.occurred_at)
                    if final is not None:
                        yield final
                    continue
                current = states.get(stream_id)
                if current is not None and (current.image is not None or current.description):
                    raise ValidationError("vision stream started before its prior boundary")
                if current is None:
                    self._state_for(states, stream_id, packet.occurred_at)
                else:
                    current.occurred_at = packet.occurred_at
                    current.last_occurred_at = packet.occurred_at
                continue
            state = self._state_for(states, stream_id, packet.occurred_at)
            if isinstance(packet, VisionPartial):
                state.description = packet.text
            else:
                # ponytail: retain one keyframe per scene; add a bounded sampler only when
                # multi-frame retrieval quality demonstrates that the latest frame is insufficient.
                state.image = packet.image
            query = self._query(state)
            if query is not None:
                yield StreamEvent(StreamPhase.UPDATE, query, stream_id)

    def _state_for(
        self,
        states: dict[str, _VisionStreamState],
        stream_id: str,
        occurred_at: datetime | None,
    ) -> _VisionStreamState:
        state = states.get(stream_id)
        if state is None:
            if len(states) >= self._max_streams:
                raise ValidationError("vision stream exceeds max_streams")
            state = _VisionStreamState(occurred_at=occurred_at)
            states[stream_id] = state
        elif state.occurred_at is None:
            state.occurred_at = occurred_at
        if occurred_at is not None:
            state.last_occurred_at = occurred_at
        return state

    def _query(self, state: _VisionStreamState) -> ContentInput | None:
        if self._native_image and state.image is not None:
            if self._native_text and state.description:
                return (state.description, state.image)
            return state.image
        return state.description or None

    async def _finish(
        self,
        states: dict[str, _VisionStreamState],
        stream_id: str,
        occurred_end: datetime | None,
    ) -> StreamEvent | None:
        state = states.pop(stream_id, None)
        if state is None:
            return None
        occurred_end = occurred_end or state.last_occurred_at
        if (
            not self._capture_mode
            and state.image is not None
            and not state.description
            and not self._native_image
            and self._native_text
            and self._memory._memory._backends.vision_describer is not None
        ):
            state.description = (
                await self._memory._memory._execution.run_in_worker(
                    self._memory._memory._vision.describe, (state.image,)
                )
            )[0]
        # Sampled here, at the boundary, so the pose belongs to the scene being committed.
        context = self._context()
        if state.image is not None:
            item = StreamInput(
                state.image,
                occurred_at=state.occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
                description=state.description or None,
            )
        elif state.description:
            item = StreamInput(
                state.description,
                occurred_at=state.occurred_at,
                occurred_end=occurred_end,
                memory_type=self._written_type,
                context=context,
            )
        else:
            return StreamEvent(StreamPhase.CANCEL, stream_id=stream_id)
        return StreamEvent(StreamPhase.FINAL, item, stream_id)


def _pcm_blob(state: _AudioStreamState) -> Blob:
    assert (
        state.sample_rate_hz is not None
        and state.channels is not None
        and state.sample_width_bytes is not None
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(state.channels)
        audio.setsampwidth(state.sample_width_bytes)
        audio.setframerate(state.sample_rate_hz)
        audio.writeframes(state.pcm)
    return Blob(output.getvalue(), "audio/wav", "capture.wav")


def _audio_interval(
    state: _AudioStreamState,
    occurred_end: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    occurred_at = state.occurred_at
    if not state.pcm:
        return occurred_at, occurred_end
    assert (
        state.sample_rate_hz is not None
        and state.channels is not None
        and state.sample_width_bytes is not None
    )
    frames = len(state.pcm) // (state.channels * state.sample_width_bytes)
    duration = timedelta(seconds=frames / state.sample_rate_hz)
    if occurred_at is None and occurred_end is not None:
        occurred_at = occurred_end - duration
    elif occurred_at is not None and occurred_end is None:
        occurred_end = occurred_at + duration
    return occurred_at, occurred_end
