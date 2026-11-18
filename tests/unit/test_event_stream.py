"""Continuous capture commits semantic event prefixes without losing source windows."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    AsyncEventStream,
    AsyncMemory,
    Blob,
    EmbedTask,
    EventChunk,
    EventObservation,
    EventSegment,
    Memory,
    MemoryCategory,
    Modality,
    ModelError,
    ModelInput,
    ObservationContext,
    StreamCommit,
    ValidationError,
)

REFERENCE_AT = datetime(2026, 11, 14, 8, tzinfo=timezone.utc)


class ScriptedSegmenter:
    event_model = "event-test-snapshot"
    event_recipe = "event-test-snapshot:complete-prefix-v1"

    def __init__(self, replies: Sequence[object]) -> None:
        self._replies = iter(replies)
        self.calls: list[tuple[tuple[EventObservation, ...], bool]] = []

    def segment_events(
        self,
        observations: Sequence[EventObservation],
        *,
        end_of_stream: bool = False,
    ) -> tuple[EventSegment, ...]:
        self.calls.append((tuple(observations), end_of_stream))
        reply = next(self._replies)
        if isinstance(reply, Exception):
            raise reply
        # Illegal replies deliberately exercise the public model boundary's validation.
        return cast(tuple[EventSegment, ...], reply)


class CountingEmbedder(TinyEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        return super().embed(inputs, task)


def _chunk(
    offset: int,
    *,
    stream_id: str = "default",
    paired: bool = False,
    data: bytes | None = None,
) -> EventChunk:
    audio = Blob(data or f"{stream_id}-sound-{offset}".encode(), "audio/wav")
    content = (
        (audio, Blob(f"{stream_id}-frame-{offset}".encode(), "image/png")) if paired else audio
    )
    return EventChunk(
        content,
        REFERENCE_AT + timedelta(seconds=offset),
        REFERENCE_AT + timedelta(seconds=offset + 1),
        stream_id,
    )


async def _chunks(*values: EventChunk) -> AsyncIterator[EventChunk]:
    for value in values:
        yield value


@pytest.mark.asyncio
async def test_one_complete_event_preserves_silence_and_camera_changes(tmp_path: Path) -> None:
    segmenter = ScriptedSegmenter([(), (), (EventSegment(3, "Prepare and serve tea"),)])
    windows = (
        _chunk(0, paired=True),
        _chunk(1, paired=True, data=b"silent pause"),
        _chunk(2, paired=True),
    )
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(memory, segmenter=segmenter).consume(
                _chunks(*windows)
            )
        ]

        assert [len(call[0]) for call in segmenter.calls] == [1, 2, 3]
        assert all(not eof for _, eof in segmenter.calls)
        assert len(commits) == 1
        record = commits[0].record
        assert json.loads(record.content.rsplit("\n", 1)[-1]) == "Prepare and serve tea"
        assert not any(part.kind == "text" for part in record.parts)
        assert sum(part.kind == "model_interpretation" for part in record.parts) == 1
        assert record.memory_type is MemoryCategory.EPISODIC
        assert (record.occurred_at, record.occurred_end) == (
            REFERENCE_AT,
            REFERENCE_AT + timedelta(seconds=3),
        )
        assert {asset.id for asset in record.assets} == {
            hashlib.sha256(blob.data).hexdigest() for chunk in windows for blob in chunk.media
        }
        assert len(record.assets) == 6
        assert {asset.modality for asset in record.assets} == {Modality.AUDIO, Modality.IMAGE}
        assert record.metadata["event_segmentation"] == {
            "model": segmenter.event_model,
            "recipe": segmenter.event_recipe,
            "event_id": cast(dict[str, object], record.metadata["event_segmentation"])["event_id"],
            "part_index": 0,
            "part_count": 1,
            "occurred_at": windows[0].occurred_at.isoformat(),
            "occurred_end": windows[-1].occurred_end.isoformat(),
            "summary": "Prepare and serve tea",
            "summary_coverage": "unverified",
            "supplied_modalities": ["audio", "image"],
            "chunks": [
                {
                    "occurred_at": chunk.occurred_at.isoformat(),
                    "occurred_end": chunk.occurred_end.isoformat(),
                    "media_count": 2,
                    "asset_ids": [hashlib.sha256(blob.data).hexdigest() for blob in chunk.media],
                    "context": None,
                }
                for chunk in windows
            ],
        }
        assert len((await memory.list()).items) == 1


@pytest.mark.asyncio
async def test_multiple_events_partition_the_completed_prefix_and_keep_the_suffix(
    tmp_path: Path,
) -> None:
    segmenter = ScriptedSegmenter(
        [
            (),
            (),
            (),
            (EventSegment(1, "Open the cupboard"), EventSegment(3, "Make tea")),
            (EventSegment(2, "Wash the cup"),),
        ]
    )
    windows = tuple(_chunk(offset, paired=True) for offset in range(5))
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(memory, segmenter=segmenter).consume(
                _chunks(*windows)
            )
        ]

        assert [
            cast(dict[str, object], commit.record.metadata["event_segmentation"])["summary"]
            for commit in commits
        ] == [
            "Open the cupboard",
            "Make tea",
            "Wash the cup",
        ]
        assert [(commit.record.occurred_at, commit.record.occurred_end) for commit in commits] == [
            (REFERENCE_AT, REFERENCE_AT + timedelta(seconds=1)),
            (REFERENCE_AT + timedelta(seconds=1), REFERENCE_AT + timedelta(seconds=3)),
            (REFERENCE_AT + timedelta(seconds=3), REFERENCE_AT + timedelta(seconds=5)),
        ]
        asset_ids = [asset.id for commit in commits for asset in commit.record.assets]
        assert len(asset_ids) == len(set(asset_ids)) == 10
        assert set(asset_ids) == {
            hashlib.sha256(blob.data).hexdigest() for chunk in windows for blob in chunk.media
        }
        final_observations, eof = segmenter.calls[-1]
        assert not eof
        assert [value.occurred_at for value in final_observations] == [
            windows[3].occurred_at,
            windows[4].occurred_at,
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_at_eof", [False, True])
async def test_eof_assesses_pending_evidence_once_without_assuming_completion(
    tmp_path: Path, complete_at_eof: bool
) -> None:
    last_reply = (EventSegment(2, "Finish packing"),) if complete_at_eof else ()
    segmenter = ScriptedSegmenter([(), (), last_reply])
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(memory, segmenter=segmenter).consume(
                _chunks(_chunk(0), _chunk(1))
            )
        ]

        assert [(len(items), eof) for items, eof in segmenter.calls] == [
            (1, False),
            (2, False),
            (2, True),
        ]
        assert len(commits) == int(complete_at_eof)
        assert len((await memory.list()).items) == int(complete_at_eof)


@pytest.mark.asyncio
async def test_eof_keeps_a_completed_prefix_without_promoting_its_pending_suffix(
    tmp_path: Path,
) -> None:
    segmenter = ScriptedSegmenter([(), (EventSegment(1, "Put down the bag"),), ()])
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        (commit,) = [
            commit
            async for commit in AsyncEventStream(memory, segmenter=segmenter).consume(
                _chunks(_chunk(0), _chunk(1))
            )
        ]

        assert json.loads(commit.record.content.rsplit("\n", 1)[-1]) == "Put down the bag"
        assert commit.record.occurred_end == REFERENCE_AT + timedelta(seconds=1)
        (pending,), eof = segmenter.calls[-1]
        assert eof
        assert pending.occurred_at == REFERENCE_AT + timedelta(seconds=1)
        assert len((await memory.list()).items) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        (EventSegment(1, "Valid prefix"), EventSegment(3, "Out of range")),
        (EventSegment(1, "First"), EventSegment(1, "Duplicate")),
        (EventSegment(2, "First"), EventSegment(1, "Descending")),
        [EventSegment(1, "Wrong container")],
        ("wrong item",),
    ],
)
async def test_invalid_model_batch_is_rejected_before_any_prefix_commits(
    tmp_path: Path, invalid: object
) -> None:
    segmenter = ScriptedSegmenter([(), invalid])
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        with pytest.raises(ModelError) as failure:
            _ = [
                commit
                async for commit in AsyncEventStream(memory, segmenter=segmenter).consume(
                    _chunks(_chunk(0), _chunk(1))
                )
            ]

        assert failure.value.reason == "response_invalid"
        assert (await memory.list()).items == ()


@pytest.mark.asyncio
async def test_model_failure_preserves_only_already_committed_events(tmp_path: Path) -> None:
    segmenter = ScriptedSegmenter(
        [(EventSegment(1, "Close the door"),), RuntimeError("provider failed")]
    )
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        commits: list[StreamCommit] = []
        with pytest.raises(ModelError) as failure:
            async for commit in AsyncEventStream(memory, segmenter=segmenter).consume(
                _chunks(_chunk(0), _chunk(1))
            ):
                commits.append(commit)

        assert failure.value.reason == "model_failed"
        assert len(commits) == 1
        assert json.loads(commits[0].record.content.rsplit("\n", 1)[-1]) == "Close the door"
        assert [record.id for record in (await memory.list()).items] == [commits[0].record.id]
        assert all(not eof for _, eof in segmenter.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_source_failure_or_cancellation_does_not_finalize_pending_event(
    tmp_path: Path, cancelled: bool
) -> None:
    segmenter = ScriptedSegmenter([()])

    async def interrupted() -> AsyncIterator[EventChunk]:
        yield _chunk(0)
        if cancelled:
            raise asyncio.CancelledError
        raise RuntimeError("capture failed")

    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
            _ = [
                commit
                async for commit in AsyncEventStream(memory, segmenter=segmenter).consume(
                    interrupted()
                )
            ]

        assert len(segmenter.calls) == 1
        assert not segmenter.calls[0][1]
        assert (await memory.list()).items == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("at_eof", [False, True])
@pytest.mark.parametrize("model_fails", [False, True])
async def test_cancellation_drains_an_active_segmenter_before_returning(
    tmp_path: Path, at_eof: bool, model_fails: bool
) -> None:
    started = threading.Event()
    released = threading.Event()
    finished = threading.Event()

    class BlockingSegmenter:
        event_model = "blocking-event-test"
        event_recipe = "blocking-event-test:v1"

        def segment_events(
            self,
            observations: Sequence[EventObservation],
            *,
            end_of_stream: bool = False,
        ) -> tuple[EventSegment, ...]:
            if at_eof and not end_of_stream:
                return ()
            started.set()
            try:
                assert released.wait(5), "test did not release the active segmenter"
                if model_fails:
                    raise RuntimeError("provider failed after cancellation")
                return (EventSegment(len(observations), "Completed event"),)
            finally:
                finished.set()

    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:

        async def collect() -> list[StreamCommit]:
            return [
                commit
                async for commit in AsyncEventStream(memory, segmenter=BlockingSegmenter()).consume(
                    _chunks(_chunk(0))
                )
            ]

        task = asyncio.create_task(collect())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done()
            assert not finished.is_set()

            released.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
            assert finished.is_set()
            assert (await memory.list()).items == ()
        finally:
            released.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)


@pytest.mark.asyncio
async def test_interleaved_streams_keep_independent_evidence_and_commit_ids(tmp_path: Path) -> None:
    segmenter = ScriptedSegmenter(
        [(), (), (EventSegment(2, "Left event"),), (EventSegment(2, "Right event"),)]
    )
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(memory, segmenter=segmenter).consume(
                _chunks(
                    _chunk(0, stream_id="left"),
                    _chunk(0, stream_id="right"),
                    _chunk(1, stream_id="left"),
                    _chunk(1, stream_id="right"),
                )
            )
        ]

        assert [
            (
                commit.stream_id,
                cast(dict[str, object], commit.record.metadata["event_segmentation"])["summary"],
            )
            for commit in commits
        ] == [
            ("left", "Left event"),
            ("right", "Right event"),
        ]
        for (observations, _), stream_id in zip(
            segmenter.calls, ("left", "right", "left", "right"), strict=True
        ):
            assert {
                asset.id for observation in observations for asset in observation.content.assets
            } == {
                hashlib.sha256(f"{stream_id}-sound-{offset}".encode()).hexdigest()
                for offset in range(len(observations))
            }
        for index, stream_id in enumerate(("left", "right")):
            assert {asset.id for asset in commits[index].record.assets} == {
                hashlib.sha256(f"{stream_id}-sound-{offset}".encode()).hexdigest()
                for offset in (0, 1)
            }


@pytest.mark.asyncio
@pytest.mark.parametrize("completed_prefix", [False, True])
@pytest.mark.parametrize("next_offset", [0, 2])
async def test_capture_intervals_stay_contiguous_after_pending_or_completed_events(
    tmp_path: Path, completed_prefix: bool, next_offset: int
) -> None:
    segmenter = ScriptedSegmenter(
        [(EventSegment(1, "Completed first event"),) if completed_prefix else ()]
    )
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        with pytest.raises(ValidationError, match="contiguous"):
            _ = [
                commit
                async for commit in AsyncEventStream(memory, segmenter=segmenter).consume(
                    _chunks(_chunk(0), _chunk(next_offset))
                )
            ]

        assert len(segmenter.calls) == 1
        assert len((await memory.list()).items) == int(completed_prefix)


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", ["streams", "bytes"])
async def test_buffer_limits_reject_without_forcing_an_event_boundary(
    tmp_path: Path, bound: str
) -> None:
    segmenter = ScriptedSegmenter([()])
    first = _chunk(0, data=b"one")
    second = _chunk(
        1, stream_id="other" if bound in {"streams", "bytes"} else "default", data=b"two"
    )
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(
            memory,
            segmenter=segmenter,
            max_streams=1 if bound == "streams" else 32,
            max_buffer_bytes=3 if bound == "bytes" else 64 * 1024 * 1024,
        )
        with pytest.raises(ValidationError):
            _ = [commit async for commit in stream.consume(_chunks(first, second))]

        assert len(segmenter.calls) == 1
        assert (await memory.list()).items == ()


@pytest.mark.asyncio
async def test_stream_id_limit_also_bounds_completed_stream_history(tmp_path: Path) -> None:
    segmenter = ScriptedSegmenter([(EventSegment(1, "Completed event"),)])
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        with pytest.raises(ValidationError, match="max_streams"):
            _ = [
                commit
                async for commit in AsyncEventStream(
                    memory, segmenter=segmenter, max_streams=1
                ).consume(_chunks(_chunk(0), _chunk(0, stream_id="other")))
            ]

        assert len(segmenter.calls) == 1
        stored = (await memory.list()).items
        assert len(stored) == 1
        assert json.loads(stored[0].content.rsplit("\n", 1)[-1]) == "Completed event"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, -1, True])
async def test_stream_configuration_requires_positive_integer_buffer_limits(
    tmp_path: Path, value: int
) -> None:
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        segmenter = ScriptedSegmenter([])
        for name in ("streams", "chunks", "bytes", "pending_bytes"):
            with pytest.raises(ValidationError, match="positive integer"):
                AsyncEventStream(
                    memory,
                    segmenter=segmenter,
                    max_streams=value if name == "streams" else 32,
                    max_chunks=value if name == "chunks" else 32,
                    max_buffer_bytes=value if name == "bytes" else 1024,
                    max_pending_bytes=value if name == "pending_bytes" else 1024,
                )
        assert segmenter.calls == []


@pytest.mark.asyncio
async def test_context_is_sampled_once_per_received_chunk(tmp_path: Path) -> None:
    segmenter = ScriptedSegmenter(
        [(), (EventSegment(1, "First event"), EventSegment(2, "Second event"))]
    )
    samples: list[str] = []

    def context() -> ObservationContext:
        source_id = f"observation-{len(samples) + 1}"
        samples.append(source_id)
        return ObservationContext(source_id=source_id)

    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(
                memory, segmenter=segmenter, context=context
            ).consume(_chunks(_chunk(0), _chunk(1)))
        ]

        assert samples == ["observation-1", "observation-2"]
        assert [
            commit.record.context.source_id
            for commit in commits
            if commit.record.context is not None
        ] == samples


@pytest.mark.asyncio
async def test_capture_defers_embeddings_but_still_understands_events(tmp_path: Path) -> None:
    segmenter = ScriptedSegmenter([(EventSegment(1, "Store the keys"),)])
    embedder = CountingEmbedder()
    async with AsyncMemory(Memory(tmp_path, embedder=embedder)) as memory:
        (commit,) = [
            commit
            async for commit in AsyncEventStream(memory, segmenter=segmenter, capture=True).consume(
                _chunks(_chunk(0, paired=True))
            )
        ]

        assert len(segmenter.calls) == 1
        assert embedder.calls == 0
        assert commit.pending_settlement
        assert commit.prefetch is None
        assert commit.retrieval_error is None
        assert len((await memory.get(commit.record.id)).assets) == 2
        assert len(await memory.pending_captures()) == 1
        assert await memory.settle(limit=1) == 1
        assert embedder.calls > 0
        assert await memory.pending_captures() == ()
