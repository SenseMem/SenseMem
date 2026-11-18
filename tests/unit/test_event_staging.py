"""Durable event input stays independent of semantic model processing."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import AsyncGenerator, Sequence
from dataclasses import replace
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from typing import cast

import pytest
from _feature_support import TinyEmbedder
from test_event_stream import ScriptedSegmenter, _chunk, _chunks

from sensemem import (
    AsyncEventStream,
    AsyncMemory,
    EventChunk,
    EventObservation,
    EventSegment,
    Memory,
    StorageError,
    StreamCommit,
    ValidationError,
)
from sensemem.infrastructure.local.event_checkpoint import CheckpointChunk, EventCheckpoint


class BlockingSegmenter(ScriptedSegmenter):
    def __init__(self) -> None:
        super().__init__([(), (EventSegment(3, "Finish making tea"),)])
        self.entered = threading.Event()
        self.release = threading.Event()

    def segment_events(
        self,
        observations: Sequence[EventObservation],
        *,
        end_of_stream: bool = False,
    ) -> tuple[EventSegment, ...]:
        if not self.calls:
            self.entered.set()
            if not self.release.wait(10):
                raise RuntimeError("test did not release semantic assessment")
        return super().segment_events(observations, end_of_stream=end_of_stream)


@pytest.mark.asyncio
async def test_append_requires_a_durable_checkpoint(tmp_path: Path) -> None:
    segmenter = ScriptedSegmenter([])
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(memory, segmenter=segmenter)

        with pytest.raises(ValidationError, match="checkpoint"):
            await stream.append(_chunk(0))

        assert stream.positions == {}
        assert segmenter.calls == []
        assert (await memory.list()).items == ()


@pytest.mark.asyncio
async def test_append_survives_restart_without_running_the_segmenter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "capture.json"
    boundaries = (0, 700, 2400, 5600)
    started = _chunk(0).occurred_at
    windows = tuple(
        replace(
            _chunk(offset, paired=True),
            occurred_at=started + timedelta(milliseconds=start),
            occurred_end=started + timedelta(milliseconds=end),
        )
        for offset, (start, end) in enumerate(pairwise(boundaries))
    )
    original_unpack = EventCheckpoint.unpack
    unpacked = 0

    def observe_unpack(self: EventCheckpoint, chunk: CheckpointChunk) -> EventChunk:
        nonlocal unpacked
        unpacked += 1
        return original_unpack(self, chunk)

    monkeypatch.setattr(EventCheckpoint, "unpack", observe_unpack)
    unused = ScriptedSegmenter([])
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(memory, segmenter=unused, checkpoint_path=checkpoint)
        for chunk in windows:
            await stream.append(chunk)

        assert stream.positions == {"default": windows[-1].occurred_end}
        assert unused.calls == []
        assert unpacked == 0
        assert (await memory.list()).items == ()

    segmenter = ScriptedSegmenter([(EventSegment(3, "Finish making tea"),)])
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        restored = AsyncEventStream(memory, segmenter=segmenter, checkpoint_path=checkpoint)
        assert restored.positions == {"default": windows[-1].occurred_end}
        assert unpacked == 0
        (commit,) = [value async for value in restored.process()]
        # Originals are loaded for the bounded assessment, then again for its storage part;
        # accepted source references do not retain either loaded copy.
        assert unpacked == 2 * len(windows)
        assert [(len(items), eof) for items, eof in segmenter.calls] == [(3, False)]
        assert [
            (observation.occurred_at, observation.occurred_end)
            for observation in segmenter.calls[0][0]
        ] == [(chunk.occurred_at, chunk.occurred_end) for chunk in windows]
        assert (commit.record.occurred_at, commit.record.occurred_end) == (
            windows[0].occurred_at,
            windows[-1].occurred_end,
        )
        assert [asset.id for asset in commit.record.assets] == [
            hashlib.sha256(blob.data).hexdigest() for chunk in windows for blob in chunk.media
        ]
        assert [
            (await memory.read_asset(commit.record.id, asset.id)).data
            for asset in commit.record.assets
        ] == [blob.data for chunk in windows for blob in chunk.media]

        replayed = AsyncEventStream(memory, segmenter=segmenter, checkpoint_path=checkpoint)
        assert [value async for value in replayed.process()] == []
        assert len((await memory.list()).items) == 1


@pytest.mark.asyncio
async def test_append_accepts_input_while_semantic_assessment_is_blocked(tmp_path: Path) -> None:
    segmenter = BlockingSegmenter()
    windows = tuple(_chunk(offset, paired=True) for offset in range(3))
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(
            memory, segmenter=segmenter, checkpoint_path=tmp_path / "capture.json"
        )
        await stream.append(windows[0])

        async def process() -> list[StreamCommit]:
            return [value async for value in stream.process()]

        processing = asyncio.create_task(process())
        try:
            assert await asyncio.to_thread(segmenter.entered.wait, 3)
            with pytest.raises(ValidationError, match="active processor"):
                _ = [value async for value in stream.process()]
            with pytest.raises(ValidationError, match="boolean"):
                _ = [value async for value in stream.process(end_of_stream=cast(bool, "yes"))]
            await asyncio.wait_for(stream.append(windows[1]), 2)
            await asyncio.wait_for(stream.append(windows[2]), 2)
            assert not processing.done()
            assert stream.positions == {"default": windows[-1].occurred_end}
        finally:
            segmenter.release.set()
            commits = await asyncio.wait_for(processing, 10)

        assert [(len(items), eof) for items, eof in segmenter.calls] == [
            (1, False),
            (3, False),
        ]
        assert len(commits) == 1
        assert len(commits[0].record.assets) == 6
        assert len((await memory.list()).items) == 1


@pytest.mark.asyncio
async def test_process_coalesces_backlog_within_the_analysis_window(tmp_path: Path) -> None:
    windows = tuple(_chunk(offset, data=str(offset).encode()) for offset in range(5))
    segmenter = ScriptedSegmenter(
        [
            (EventSegment(2, "Make tea"),),
            (EventSegment(2, "Wash up"),),
            (EventSegment(1, "Put away the cup"),),
        ]
    )
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(
            memory,
            segmenter=segmenter,
            checkpoint_path=tmp_path / "capture.json",
            max_chunks=2,
            max_buffer_bytes=2,
        )
        for chunk in windows:
            await stream.append(chunk)

        commits = [value async for value in stream.process()]

        assert [(len(items), eof) for items, eof in segmenter.calls] == [
            (2, False),
            (2, False),
            (1, False),
        ]
        assert [commit.record.occurred_at for commit in commits] == [
            windows[index].occurred_at for index in (0, 2, 4)
        ]
        assert [asset.id for commit in commits for asset in commit.record.assets] == [
            hashlib.sha256(chunk.media[0].data).hexdigest() for chunk in windows
        ]


@pytest.mark.asyncio
async def test_pending_byte_limit_rejects_without_acknowledging_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "capture.json"
    first, second = _chunk(0, data=b"one"), _chunk(1, data=b"two")
    unused = ScriptedSegmenter([])
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(
            memory,
            segmenter=unused,
            checkpoint_path=checkpoint,
            max_pending_bytes=3,
        )
        await stream.append(first)
        with pytest.raises(ValidationError, match="max_pending_bytes") as rejected:
            await stream.append(second)
        assert rejected.value.reason == "capacity_exceeded"
        assert rejected.value.retryable is True

        assert stream.positions == {"default": first.occurred_end}
        assert unused.calls == []

    segmenter = ScriptedSegmenter([(EventSegment(1, "First action"),)])
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        restored = AsyncEventStream(
            memory, segmenter=segmenter, checkpoint_path=checkpoint, max_pending_bytes=3
        )
        with pytest.raises(ValidationError, match="max_pending_bytes"):
            await restored.append(second)

        assert restored.positions == {"default": first.occurred_end}
        assert segmenter.calls == []
        assert len([value async for value in restored.process()]) == 1
        await restored.append(second)
        assert restored.positions == {"default": second.occurred_end}


@pytest.mark.asyncio
async def test_failed_manifest_write_keeps_append_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    segmenter = ScriptedSegmenter([(EventSegment(1, "Finish repair"),)])
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(
            memory, segmenter=segmenter, checkpoint_path=tmp_path / "capture.json"
        )

        def cannot_save(self: EventCheckpoint, checkpoint: object) -> None:
            raise StorageError("checkpoint filesystem is full")

        with monkeypatch.context() as patch:
            patch.setattr(EventCheckpoint, "save", cannot_save)
            with pytest.raises(StorageError):
                await stream.append(_chunk(0))

        assert stream.positions == {}
        assert segmenter.calls == []
        await stream.append(_chunk(0))
        assert len([value async for value in stream.process()]) == 1
        assert len((await memory.list()).items) == 1


@pytest.mark.asyncio
async def test_processing_eof_keeps_an_unfinished_event_for_resumption(tmp_path: Path) -> None:
    checkpoint = tmp_path / "capture.json"
    segmenter = ScriptedSegmenter([(EventSegment(1, "Still repairing", complete=False),)])
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(memory, segmenter=segmenter, checkpoint_path=checkpoint)
        await stream.append(_chunk(0))
        assert [value async for value in stream.process(end_of_stream=True)] == []
        assert [(len(items), eof) for items, eof in segmenter.calls] == [(1, True)]
        assert (await memory.list()).items == ()

    recovered = ScriptedSegmenter([(EventSegment(2, "Finish repair"),)])
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        restored = AsyncEventStream(memory, segmenter=recovered, checkpoint_path=checkpoint)
        await restored.append(_chunk(1))
        (commit,) = [value async for value in restored.process()]
        assert len(commit.record.assets) == 2
        assert commit.record.occurred_at == _chunk(0).occurred_at
        assert commit.record.occurred_end == _chunk(1).occurred_end


@pytest.mark.asyncio
@pytest.mark.parametrize("driver", ["process", "consume"])
async def test_explicit_iterator_close_releases_processing_ownership(
    tmp_path: Path, driver: str
) -> None:
    segmenter = ScriptedSegmenter(
        [(EventSegment(1, "First activity"),), (EventSegment(1, "Second activity"),)]
    )
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(
            memory, segmenter=segmenter, checkpoint_path=tmp_path / "capture.json"
        )
        if driver == "process":
            await stream.append(_chunk(0))
            processing = stream.process()
        else:
            processing = stream.consume(_chunks(_chunk(0)))
        await anext(processing)
        await cast(AsyncGenerator[StreamCommit, None], processing).aclose()
        await stream.append(_chunk(1))
        assert len([value async for value in stream.process()]) == 1
        assert len((await memory.list()).items) == 2
