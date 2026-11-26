"""Restartable semantic events with bounded model windows and lossless storage parts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from threading import RLock

from opentelemetry.trace import Tracer

from sensemem._content import MAX_CONTENT_PARTS
from sensemem._telemetry import mark_model_requests
from sensemem.content import Blob, ContentAtom
from sensemem.events import EventChunk, EventSegment
from sensemem.exceptions import ModelError, StorageError, ValidationError
from sensemem.infrastructure.local.event_checkpoint import (
    Checkpoint,
    CheckpointChunk,
    CheckpointEvent,
    CheckpointStream,
    EventCheckpoint,
)
from sensemem.kernel.hydration import Hydrator
from sensemem.kernel.lifecycle import Lifecycle
from sensemem.kernel.materialization import Materializer
from sensemem.kernel.tracing import Traced
from sensemem.models.base import EventObservation, EventSegmentationBackend, ModelInput
from sensemem.prompts import events as event_prompts
from sensemem.spatial import ObservationContext
from sensemem.stream_values import StreamEvent, StreamInput, StreamPhase
from sensemem.types import MemoryCategory


@dataclass(slots=True)
class _EventState:
    chunks: list[EventChunk | CheckpointChunk] = field(default_factory=list)
    last_end: datetime | None = None
    continuation: str | None = None
    continuation_chunks: int = 0
    needs_assessment: bool = False


@dataclass(slots=True)
class _ReadyEvent:
    chunks: tuple[EventChunk | CheckpointChunk, ...]
    summary: str
    max_chunks: int
    next_part: int = 0
    max_part_bytes: int | None = None
    parts: tuple[tuple[EventChunk | CheckpointChunk, ...], ...] = field(init=False)
    event_id: str | None = None

    def __post_init__(self) -> None:
        """Keep chunks intact; a location change also starts a separate retrieval record."""
        parts: list[list[EventChunk | CheckpointChunk]] = []
        media_count = 0
        media_bytes = 0
        for chunk in self.chunks:
            if (
                not parts
                or len(parts[-1]) >= self.max_chunks
                or parts[-1][-1].context != chunk.context
                or media_count + len(chunk.media) > MAX_CONTENT_PARTS - 1
                or (
                    self.max_part_bytes is not None
                    and media_bytes + _chunk_size(chunk) > self.max_part_bytes
                )
            ):
                parts.append([])
                media_count = 0
                media_bytes = 0
            parts[-1].append(chunk)
            media_count += len(chunk.media)
            media_bytes += _chunk_size(chunk)
        self.parts = tuple(tuple(part) for part in parts)


class EventSegmentation(Traced):
    """Own input, continuation, and delivery state until each storage part is acknowledged."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        segmenter: EventSegmentationBackend,
        lifecycle: Lifecycle,
        materializer: Materializer,
        hydrator: Hydrator,
        max_streams: int,
        max_chunks: int,
        max_buffer_bytes: int,
        max_pending_bytes: int,
        checkpoint: EventCheckpoint | None = None,
    ) -> None:
        super().__init__(tracer)
        self._segmenter = segmenter
        self._lifecycle = lifecycle
        self._materializer = materializer
        self._hydrator = hydrator
        self._max_streams = max_streams
        self._max_chunks = max_chunks
        self._max_buffer_bytes = max_buffer_bytes
        self._max_pending_bytes = max_pending_bytes
        self._checkpoint = checkpoint
        self._lock = RLock()
        self._states: dict[str, _EventState] = {}
        self._ready: list[_ReadyEvent] = []
        self._pending: list[EventChunk | CheckpointChunk] = []
        self._buffer_bytes = 0
        self._pending_bytes = 0
        saved = None if checkpoint is None else checkpoint.load()
        if saved is not None:
            self._restore(saved)

    @property
    def pending_streams(self) -> tuple[str, ...]:
        return tuple(key for key, state in self._states.items() if state.chunks)

    @property
    def positions(self) -> dict[str, datetime]:
        with self._lock:
            result = {
                key: state.last_end
                for key, state in self._states.items()
                if state.last_end is not None
            }
            result.update((chunk.stream_id, chunk.occurred_end) for chunk in self._pending)
            return result

    def stage(self, chunk: EventChunk, *, deferred: bool) -> None:
        """Accept source bytes independently of the sole semantic processor."""
        with self._lock:
            self._lifecycle.require_open()
            positions = self.positions
            if chunk.stream_id not in positions and len(positions) >= self._max_streams:
                raise ValidationError("event stream exceeds max_streams")
            if (previous := positions.get(chunk.stream_id)) is not None and (
                chunk.occurred_at != previous
            ):
                raise ValidationError("event chunks must have contiguous ordered capture intervals")
            size = sum(len(blob.data) for blob in chunk.media)
            if size > self._max_buffer_bytes:
                raise ValidationError("event stream exceeds max_buffer_bytes")
            if self._pending_bytes + size > self._max_pending_bytes:
                raise ValidationError(
                    "event stream exceeds max_pending_bytes", reason="capacity_exceeded"
                )
            if not deferred:
                state = self._states.get(chunk.stream_id, _EventState())
                self._require_capacity(state, size)
            reference = chunk if self._checkpoint is None else self._checkpoint.pack(chunk)
            self._pending.append(reference)
            self._pending_bytes += size
            try:
                self._save()
            except StorageError:
                self._pending.pop()
                self._pending_bytes -= size
                raise

    def advance(self, *, end_of_stream: bool) -> tuple[str, bool] | None:
        """Coalesce arrived windows, release acceptance ownership, then assess one snapshot."""
        with self._lock:
            if retry := next(
                (key for key, state in self._states.items() if state.needs_assessment), None
            ):
                stream_id = retry
                state = self._states[stream_id]
            elif self._pending:
                stream_id = self._pending[0].stream_id
                state = self._states.get(stream_id, _EventState())
                previous_count = len(state.chunks)
                previous_end = state.last_end
                room = self._max_chunks - (
                    len(state.chunks) - state.continuation_chunks + bool(state.continuation)
                )
                self._require_capacity(state, _chunk_size(self._pending[0]))
                admitted: list[EventChunk | CheckpointChunk] = []
                size = 0
                for reference in self._pending[:room]:
                    if reference.stream_id != stream_id:
                        break
                    part_size = _chunk_size(reference)
                    active_bytes = (
                        self._buffer_bytes + size + part_size
                        if self._checkpoint is None
                        else self._analysis_window(state, (*admitted, reference))[1]
                    )
                    if active_bytes > self._max_buffer_bytes:
                        break
                    admitted.append(reference)
                    size += part_size
                state.chunks.extend(admitted)
                state.last_end = admitted[-1].occurred_end
                state.needs_assessment = True
                self._states[stream_id] = state
                del self._pending[: len(admitted)]
                self._pending_bytes -= size
                if self._checkpoint is None:
                    self._buffer_bytes += size
                try:
                    self._save()
                except StorageError:
                    del state.chunks[previous_count:]
                    state.last_end = previous_end
                    state.needs_assessment = False
                    self._pending[:0] = admitted
                    self._pending_bytes += size
                    if self._checkpoint is None:
                        self._buffer_bytes -= size
                    if previous_end is None:
                        self._states.pop(stream_id)
                    raise
            else:
                return None
            final = end_of_stream and not any(
                reference.stream_id == stream_id for reference in self._pending
            )
        self._segment(stream_id, state, end_of_stream=final)
        return stream_id, final

    def _require_capacity(self, state: _EventState, size: int) -> None:
        active_bytes = self._buffer_bytes + size
        if self._checkpoint is not None:
            _, active_bytes = self._analysis_window(state, extra_bytes=size, extra_count=1)
        if active_bytes > self._max_buffer_bytes:
            raise ValidationError(
                "event stream exceeds max_buffer_bytes", reason="capacity_exceeded"
            )
        if (
            len(state.chunks) - state.continuation_chunks + bool(state.continuation)
            >= self._max_chunks
        ):
            raise ModelError(
                "event segmenter must summarize the unfinished window before more input",
                reason="response_invalid",
            )

    def _analysis_window(
        self,
        state: _EventState,
        additional: tuple[EventChunk | CheckpointChunk, ...] = (),
        *,
        extra_bytes: int = 0,
        extra_count: int = 0,
    ) -> tuple[int, int]:
        """Choose original context or a continuation anchor within one analysis budget."""
        size = sum(_chunk_size(chunk) for chunk in state.chunks)
        size += sum(_chunk_size(chunk) for chunk in additional) + extra_bytes
        compressed = (
            state.continuation_chunks
            if (
                len(state.chunks) + len(additional) + extra_count > self._max_chunks
                or (self._checkpoint is not None and size > self._max_buffer_bytes)
            )
            else 0
        )
        if compressed:
            size -= sum(_chunk_size(chunk) for chunk in state.chunks[: compressed - 1])
        return compressed, size

    def finish(self, stream_id: str) -> None:
        """EOF assesses finality; an incomplete suffix remains available for resumption."""
        self._segment(stream_id, self._states[stream_id], end_of_stream=True)

    def peek(self) -> StreamEvent | None:
        if not self._ready:
            return None
        ready = self._ready[0]
        parts = ready.parts
        event_chunks = tuple(self._load(chunk) for chunk in parts[ready.next_part])
        if ready.event_id is None:
            identity = hashlib.sha256()
            identity.update(self._segmenter.event_recipe.encode())
            for chunk in ready.chunks:
                identity.update(chunk.occurred_at.isoformat().encode())
                identity.update(chunk.occurred_end.isoformat().encode())
                identity.update(
                    json.dumps(
                        None if chunk.context is None else _context_metadata(chunk.context),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                )
                digests = (
                    (hashlib.sha256(blob.data).hexdigest() for blob in chunk.media)
                    if isinstance(chunk, EventChunk)
                    else (blob.digest for blob in chunk.media)
                )
                for digest in digests:
                    identity.update(bytes.fromhex(digest))
            ready.event_id = identity.hexdigest()
        content: list[ContentAtom] = [blob for chunk in event_chunks for blob in chunk.media]
        event_start = ready.chunks[0].occurred_at.isoformat()
        event_end = ready.chunks[-1].occurred_end.isoformat()
        part_start = event_chunks[0].occurred_at.isoformat()
        part_end = event_chunks[-1].occurred_end.isoformat()
        modalities = sorted(
            {blob.media_type.split("/", 1)[0] for chunk in ready.chunks for blob in chunk.media}
        )
        if len(parts) == 1:
            interpretation = event_prompts.synopsis(ready.summary)
        else:
            interpretation = event_prompts.grouping(
                ready.next_part + 1,
                len(parts),
                part_start=part_start,
                part_end=part_end,
                event_start=event_start,
                event_end=event_end,
            )
        item = StreamInput(
            tuple(content),
            occurred_at=event_chunks[0].occurred_at,
            occurred_end=event_chunks[-1].occurred_end,
            memory_type=MemoryCategory.EPISODIC,
            context=event_chunks[0].context,
            interpretation=interpretation,
            metadata={
                "event_segmentation": {
                    "model": self._segmenter.event_model,
                    "recipe": self._segmenter.event_recipe,
                    "event_id": ready.event_id,
                    "part_index": ready.next_part,
                    "part_count": len(parts),
                    "occurred_at": event_start,
                    "occurred_end": event_end,
                    "summary": ready.summary,
                    "summary_coverage": "unverified",
                    "supplied_modalities": modalities,
                    "chunks": [
                        {
                            "occurred_at": chunk.occurred_at.isoformat(),
                            "occurred_end": chunk.occurred_end.isoformat(),
                            "media_count": len(chunk.media),
                            "asset_ids": [
                                hashlib.sha256(blob.data).hexdigest() for blob in chunk.media
                            ],
                            "context": (
                                None if chunk.context is None else _context_metadata(chunk.context)
                            ),
                        }
                        for chunk in event_chunks
                    ],
                }
            },
        )
        return StreamEvent(StreamPhase.FINAL, item, event_chunks[0].stream_id)

    def acknowledge(self) -> None:
        with self._lock:
            ready = self._ready[0]
            previous_part = ready.next_part
            ready.next_part += 1
            if ready.next_part == len(ready.parts):
                self._ready.pop(0)
                if self._checkpoint is None:
                    self._buffer_bytes -= sum(_chunk_size(chunk) for chunk in ready.chunks)
            try:
                saved = self._save()
            except StorageError:
                if ready.next_part == len(ready.parts):
                    self._ready.insert(0, ready)
                    if self._checkpoint is None:
                        self._buffer_bytes += sum(_chunk_size(chunk) for chunk in ready.chunks)
                ready.next_part = previous_part
                raise
            if saved is not None:
                assert self._checkpoint is not None
                self._checkpoint.prune(saved)

    def _load(self, chunk: EventChunk | CheckpointChunk) -> EventChunk:
        if isinstance(chunk, EventChunk):
            return chunk
        assert self._checkpoint is not None
        return self._checkpoint.unpack(chunk)

    def _segment(self, stream_id: str, state: _EventState, *, end_of_stream: bool) -> None:
        chunks = tuple(state.chunks)
        if not chunks:
            return
        with self._lifecycle.operation(model_work=True) as operation:
            observations: list[EventObservation] = []
            offsets: list[int] = []
            compressed_prefix, _ = self._analysis_window(state)
            if compressed_prefix:
                assert state.continuation is not None
                previous = self._load(chunks[state.continuation_chunks - 1])
                previous_content = self._hydrator.model_input(
                    self._materializer.materialize_content(previous.media, operation)
                )
                observations.append(
                    EventObservation(
                        ModelInput(
                            text=event_prompts.continuation(
                                state.continuation,
                                occurred_at=previous.occurred_at,
                                occurred_end=previous.occurred_end,
                            ),
                            assets=previous_content.assets,
                        ),
                        chunks[0].occurred_at,
                        chunks[state.continuation_chunks - 1].occurred_end,
                    )
                )
                offsets.append(state.continuation_chunks)
            for index, reference in enumerate(chunks[compressed_prefix:]):
                chunk = self._load(reference)
                observations.append(
                    EventObservation(
                        self._hydrator.model_input(
                            self._materializer.materialize_content(chunk.media, operation)
                        ),
                        chunk.occurred_at,
                        chunk.occurred_end,
                    )
                )
                offsets.append(compressed_prefix + index + 1)
            with self._model_trace(
                "event_segmentation",
                "event.segment",
                model=self._segmenter.event_model,
                batch_size=len(observations),
                modalities=(
                    modality for item in observations for modality in item.content.modalities
                ),
            ):
                mark_model_requests(1)
                try:
                    proposals = self._segmenter.segment_events(
                        tuple(observations), end_of_stream=end_of_stream
                    )
                except ModelError:
                    raise
                except Exception as error:
                    raise ModelError("event segmentation failed", reason="model_failed") from error
                proposals = _validated_segments(proposals, len(observations))
        self._commit_assessment(state, chunks, proposals, offsets)

    def _commit_assessment(
        self,
        state: _EventState,
        chunks: tuple[EventChunk | CheckpointChunk, ...],
        proposals: tuple[EventSegment, ...],
        offsets: list[int],
    ) -> None:
        """Publish boundaries atomically before any completed originals reach storage."""
        with self._lock:
            previous_ready = len(self._ready)
            previous_continuation = state.continuation
            previous_continuation_chunks = state.continuation_chunks
            previous_assessment = state.needs_assessment
            start = 0
            continuation = None
            for proposal in proposals:
                end = offsets[proposal.end_chunk - 1]
                if proposal.complete:
                    self._ready.append(
                        _ReadyEvent(
                            chunks[start:end],
                            proposal.summary,
                            self._max_chunks,
                            max_part_bytes=(
                                None if self._checkpoint is None else self._max_buffer_bytes
                            ),
                        )
                    )
                    start = end
                else:
                    continuation = proposal.summary
            del state.chunks[:start]
            if continuation is not None:
                state.continuation = continuation
                state.continuation_chunks = len(state.chunks)
            elif start:
                state.continuation = None
                state.continuation_chunks = 0
            state.needs_assessment = False
            try:
                self._save()
            except StorageError:
                state.chunks[:0] = chunks[:start]
                state.continuation = previous_continuation
                state.continuation_chunks = previous_continuation_chunks
                state.needs_assessment = previous_assessment
                del self._ready[previous_ready:]
                raise

    def _save(self) -> Checkpoint | None:
        checkpoint = self._checkpoint
        if checkpoint is None:
            return None
        saved = Checkpoint(
            model=self._segmenter.event_model,
            recipe=self._segmenter.event_recipe,
            data_dir=checkpoint.data_dir,
            max_buffer_bytes=self._max_buffer_bytes,
            max_chunks=self._max_chunks,
            streams=tuple(
                CheckpointStream(
                    stream_id=stream_id,
                    chunks=tuple(
                        chunk if isinstance(chunk, CheckpointChunk) else checkpoint.pack(chunk)
                        for chunk in state.chunks
                    ),
                    last_end=state.last_end,
                    continuation=state.continuation,
                    continuation_chunks=state.continuation_chunks,
                    needs_assessment=state.needs_assessment,
                )
                for stream_id, state in self._states.items()
            ),
            events=tuple(
                CheckpointEvent(
                    chunks=tuple(
                        chunk if isinstance(chunk, CheckpointChunk) else checkpoint.pack(chunk)
                        for chunk in event.chunks
                    ),
                    summary=event.summary,
                    next_part=event.next_part,
                )
                for event in self._ready
            ),
            pending=tuple(
                reference if isinstance(reference, CheckpointChunk) else checkpoint.pack(reference)
                for reference in self._pending
            ),
        )
        checkpoint.save(saved)
        return saved

    def _validate_checkpoint(self, saved: Checkpoint) -> None:
        """Reject incompatible ownership or delivery boundaries before loading source files."""
        assert self._checkpoint is not None
        if saved.data_dir != self._checkpoint.data_dir:
            raise ValidationError("event checkpoint belongs to a different data_dir")
        if (saved.model, saved.recipe) != (
            self._segmenter.event_model,
            self._segmenter.event_recipe,
        ):
            raise ValidationError("event checkpoint requires the same segmenter model and recipe")
        if saved.max_buffer_bytes != self._max_buffer_bytes:
            raise ValidationError("event checkpoint requires the same max_buffer_bytes")
        if saved.max_chunks != self._max_chunks:
            raise ValidationError("event checkpoint requires the same max_chunks")
        if len(saved.streams) > self._max_streams:
            raise ValidationError("event checkpoint exceeds max_streams")

    def _restore(self, saved: Checkpoint) -> None:
        self._validate_checkpoint(saved)
        for stream in saved.streams:
            chunks: list[EventChunk | CheckpointChunk] = list(stream.chunks)
            self._verify_chunks(stream.chunks)
            if stream.continuation is not None:
                EventSegment(1, stream.continuation, complete=False)
            if (
                stream.stream_id in self._states
                or any(chunk.stream_id != stream.stream_id for chunk in chunks)
                or not 0 <= stream.continuation_chunks <= len(chunks)
                or (stream.continuation is None) != (stream.continuation_chunks == 0)
                or len(chunks) - stream.continuation_chunks + bool(stream.continuation)
                > self._max_chunks
                or any(
                    first.occurred_end != second.occurred_at for first, second in pairwise(chunks)
                )
                or (chunks and chunks[-1].occurred_end != stream.last_end)
            ):
                raise ValidationError("event checkpoint stream state is invalid")
            state = _EventState(
                chunks,
                stream.last_end,
                stream.continuation,
                stream.continuation_chunks,
                stream.needs_assessment,
            )
            if self._analysis_window(state)[1] > self._max_buffer_bytes:
                raise ValidationError("event checkpoint exceeds max_buffer_bytes")
            self._states[stream.stream_id] = state
        for event in saved.events:
            self._verify_chunks(event.chunks)
            ready = _ReadyEvent(
                event.chunks,
                event.summary,
                self._max_chunks,
                next_part=event.next_part,
                max_part_bytes=self._max_buffer_bytes,
            )
            if (
                not ready.chunks
                or not 0 <= ready.next_part < len(ready.parts)
                or any(chunk.stream_id != ready.chunks[0].stream_id for chunk in ready.chunks)
                or any(
                    first.occurred_end != second.occurred_at
                    for first, second in pairwise(ready.chunks)
                )
            ):
                raise ValidationError("event checkpoint delivery state is invalid")
            EventSegment(1, ready.summary)
            self._ready.append(ready)
        self._restore_pending(saved.pending)

    def _verify_chunks(self, chunks: tuple[CheckpointChunk, ...]) -> None:
        assert self._checkpoint is not None
        for chunk in chunks:
            _validate_pending_reference(chunk)
            if _chunk_size(chunk) > self._max_buffer_bytes:
                raise ValidationError("event checkpoint exceeds max_buffer_bytes")
            self._checkpoint.verify(chunk)

    def _restore_pending(self, pending: tuple[CheckpointChunk, ...]) -> None:
        positions = self.positions
        for reference in pending:
            _validate_pending_reference(reference)
            if (previous := positions.get(reference.stream_id)) is not None and (
                reference.occurred_at != previous
            ):
                raise ValidationError("event checkpoint pending intervals are not contiguous")
            positions[reference.stream_id] = reference.occurred_end
            self._pending.append(reference)
            self._pending_bytes += _chunk_size(reference)
        if len(positions) > self._max_streams:
            raise ValidationError("event checkpoint exceeds max_streams")
        if self._pending_bytes > self._max_pending_bytes:
            raise ValidationError("event checkpoint exceeds max_pending_bytes")


def _chunk_size(chunk: EventChunk | CheckpointChunk) -> int:
    if isinstance(chunk, EventChunk):
        return sum(len(blob.data) for blob in chunk.media)
    return sum(blob.size_bytes for blob in chunk.media)


def _validate_pending_reference(chunk: CheckpointChunk) -> None:
    """Validate source metadata without loading staged media into the analysis buffer."""
    EventChunk(
        tuple(Blob(b"reference", blob.media_type, blob.name) for blob in chunk.media),
        chunk.occurred_at,
        chunk.occurred_end,
        chunk.stream_id,
        chunk.context,
    )
    if any(
        blob.size_bytes <= 0
        or len(blob.digest) != 64
        or any(character not in "0123456789abcdef" for character in blob.digest)
        for blob in chunk.media
    ):
        raise ValidationError("event checkpoint pending media reference is invalid")


def _context_metadata(context: ObservationContext) -> dict[str, object]:
    pose = context.spatial
    return {
        "basis": context.basis.value,
        "source_id": context.source_id,
        "confidence": context.confidence,
        "valid_from": None if context.valid_from is None else context.valid_from.isoformat(),
        "valid_until": None if context.valid_until is None else context.valid_until.isoformat(),
        "place_id": context.place_id,
        "spatial": (
            None
            if pose is None
            else {
                "frame_id": pose.frame_id,
                "anchor": pose.anchor.value,
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "orientation_xyzw": pose.orientation_xyzw,
                "position_uncertainty_m": pose.position_uncertainty_m,
            }
        ),
    }


def _validated_segments(proposals: object, count: int) -> tuple[EventSegment, ...]:
    if not isinstance(proposals, tuple) or any(
        not isinstance(proposal, EventSegment) for proposal in proposals
    ):
        raise ModelError("event segmentation returned invalid output", reason="response_invalid")
    previous = 0
    result: list[EventSegment] = []
    for index, proposal in enumerate(proposals):
        if not previous < proposal.end_chunk <= count or (
            not proposal.complete and (index != len(proposals) - 1 or proposal.end_chunk != count)
        ):
            raise ModelError(
                "event segmentation returned invalid boundaries", reason="response_invalid"
            )
        previous = proposal.end_chunk
        result.append(proposal)
    return tuple(result)
