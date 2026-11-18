"""Reviewed camera windows retain original evidence before audiovisual activity completion."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

import pytest
from _feature_support import TinyEmbedder
from PIL import Image

from sensemem import (
    AsyncMemory,
    ContentInput,
    EmbedTask,
    Memory,
    MemoryRecord,
    ModelInput,
    StorageError,
    StreamCommit,
    StreamInput,
)

if TYPE_CHECKING:
    from examples.live.perception import PerceptionFrame, PerceptionResult
    from examples.live.visual_memory import VisualMemoryCapture

NOW = datetime(2026, 11, 17, tzinfo=timezone.utc)
SOURCE = "live:visual-memory-fixture"


@pytest.fixture(autouse=True)
def example_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))


def _frames(start: int = 1, count: int = 2) -> tuple[PerceptionFrame, ...]:
    from examples.live.perception import PerceptionFrame

    frames: list[PerceptionFrame] = []
    for sequence in range(start, start + count):
        encoded = BytesIO()
        Image.new("RGB", (16, 16), (sequence % 256, 60, 160)).save(encoded, format="JPEG")
        frames.append(
            PerceptionFrame(
                sequence=sequence,
                captured_at=NOW + timedelta(milliseconds=sequence * 100),
                timestamp_ms=sequence * 100,
                jpeg=encoded.getvalue(),
            )
        )
    return tuple(frames)


def _window(
    start: int = 1, *, changed: bool = False, epoch: int = 0, count: int = 2
) -> PerceptionResult:
    from examples.live.perception import FrameEvidence, PerceptionContext, PerceptionResult

    frames = _frames(start, count)
    return PerceptionResult(
        summary="The key was placed inside the green box.",
        changed=changed,
        action="context",
        utterance=None,
        confidence=0.95,
        evidence=tuple(
            FrameEvidence(sequence=frame.sequence, observation="A key and green box are visible.")
            for frame in frames
        ),
        reason="Keep the visual context current.",
        contradicted_claim=None,
        correction=None,
        source_id=SOURCE,
        frame_sequences=tuple(frame.sequence for frame in frames),
        occurred_at=frames[0].captured_at,
        occurred_end=frames[-1].captured_at,
        timestamp_ms=frames[0].timestamp_ms,
        end_timestamp_ms=frames[-1].timestamp_ms,
        context=PerceptionContext(),
        target_response_id=None,
        current=True,
        latest_sequence=frames[-1].sequence,
        expires_at=NOW + timedelta(seconds=8),
        epoch=epoch,
    )


def _submit(
    capture: VisualMemoryCapture,
    start: int = 1,
    *,
    changed: bool = False,
    epoch: int = 0,
    count: int = 2,
) -> None:
    result = _window(start, changed=changed, epoch=epoch, count=count)
    capture.submit(result, _frames(start, count), perception_id=f"observation-{start}-{epoch}")


@pytest.mark.asyncio
@pytest.mark.parametrize("frame_count", [1, 2])
async def test_original_images_times_and_interpretation_survive_capture_settle_and_reopen(
    tmp_path: Path, frame_count: int
) -> None:
    from examples.live.visual_memory import VisualMemoryCapture

    commits: list[StreamCommit] = []
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        capture = VisualMemoryCapture(
            memory, SOURCE, committed=commits.append, model_id="visual-review-fixture-v1"
        )
        _submit(capture, count=frame_count)
        await capture.close()
        assert capture.failure is None
        (commit,) = commits
        assert commit.pending_settlement and commit.prefetch is None
        record = commit.record
        assert record.occurred_at == NOW + timedelta(milliseconds=100)
        assert record.occurred_end == (
            None if frame_count == 1 else NOW + timedelta(milliseconds=200)
        )
        assert record.context is not None and record.context.source_id == SOURCE
        assert len(record.assets) == frame_count
        frames = _frames(count=frame_count)
        for asset, frame in zip(record.assets, frames, strict=True):
            assert (await memory.read_asset(record.id, asset.id)).data == frame.jpeg
        assert [part.kind for part in record.parts] == [
            *["asset"] * frame_count,
            "model_interpretation",
        ]
        assert record.parts[-1].value.endswith("The key was placed inside the green box.")
        observation = record.metadata["visual_observation"]
        assert isinstance(observation, dict)
        assert observation["perception_id"] == commit.stream_id == "observation-1-0"
        assert observation["frame_sequences"] == list(range(1, frame_count + 1))
        assert observation["model_id"] == "visual-review-fixture-v1"
        assert observation["frames"] == [
            {
                "sequence": frame.sequence,
                "captured_at": frame.captured_at.isoformat(),
                "timestamp_ms": frame.timestamp_ms,
            }
            for frame in frames
        ]
        assert len(await memory.pending_captures()) == 1
        assert await memory.settle() == 1
        assert not await memory.pending_captures()
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as reopened:
        assert (await reopened.get(record.id)).parts == record.parts
        assert (await reopened.read_asset(record.id, record.assets[0].id)).data == frames[0].jpeg


@pytest.mark.asyncio
async def test_unchanged_reviews_and_duplicate_windows_skip_but_changes_and_new_epochs_persist(
    tmp_path: Path,
) -> None:
    from examples.live.visual_memory import VisualMemoryCapture

    commits: list[StreamCommit] = []
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        capture = VisualMemoryCapture(memory, SOURCE, committed=commits.append)
        _submit(capture)
        assert capture.task is not None
        await capture.task
        _submit(capture, changed=True)  # Same observed frames, regardless of a re-review.
        _submit(capture, 3)  # A fresh review reports no meaningful visual change.
        _submit(capture, 5, changed=True)
        assert capture.task is not None
        await capture.task
        _submit(capture, 7, epoch=1)  # The first observation after reopening the camera.
        await capture.close()
        assert [commit.stream_id for commit in commits] == [
            "observation-1-0",
            "observation-5-0",
            "observation-7-1",
        ]


@pytest.mark.asyncio
async def test_overload_coalesces_bounded_work_while_retaining_first_and_latest_window(
    tmp_path: Path,
) -> None:
    from examples.live.visual_memory import VisualMemoryCapture

    commits: list[StreamCommit] = []
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        capture = VisualMemoryCapture(memory, SOURCE, committed=commits.append)
        _submit(capture)
        for start in range(3, 52, 2):
            _submit(capture, start, changed=True)
        assert len(capture.pending) == 2
        assert not commits
        await capture.close()
        assert [commit.stream_id for commit in commits] == ["observation-1-0", "observation-51-0"]


@pytest.mark.asyncio
async def test_slow_native_settlement_does_not_block_accepting_the_next_visual_change(
    tmp_path: Path,
) -> None:
    from examples.live.visual_memory import VisualMemoryCapture

    class SlowEmbedder(TinyEmbedder):
        entered = Event()
        release = Event()

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            self.entered.set()
            assert self.release.wait(5), "the test did not release settlement"
            return super().embed(inputs, task)

    commits: list[StreamCommit] = []
    embedder = SlowEmbedder()
    async with AsyncMemory(Memory(tmp_path, embedder=embedder)) as memory:
        capture = VisualMemoryCapture(memory, SOURCE, committed=commits.append)
        _submit(capture)
        assert capture.task is not None
        await capture.task
        settlement = asyncio.create_task(memory.settle())
        try:
            assert await asyncio.to_thread(embedder.entered.wait, 5)
            _submit(capture, 3, changed=True)
            assert len(capture.pending) == 1
            assert not settlement.done()
        finally:
            embedder.release.set()
            await settlement
            await capture.close()
        assert len(commits) == 2


@pytest.mark.asyncio
async def test_failed_capture_reports_safe_code_and_a_later_changed_window_can_commit(
    tmp_path: Path,
) -> None:
    from examples.live.visual_memory import VisualMemoryCapture

    class FailingMemory(AsyncMemory):
        fail = True

        async def _write_stream_item(
            self, content: ContentInput | StreamInput, *, capture: bool, index: int | None = None
        ) -> MemoryRecord:
            if self.fail:
                self.fail = False
                raise StorageError("private storage diagnostic")
            return await super()._write_stream_item(content, capture=capture, index=index)

    commits: list[StreamCommit] = []
    async with FailingMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        capture = VisualMemoryCapture(memory, SOURCE, committed=commits.append)
        _submit(capture)
        assert capture.task is not None
        await capture.task
        assert capture.failure == "visual_capture_failed"
        assert not capture.processing and not commits
        _submit(capture, 3, changed=True)
        await capture.close()
        assert capture.failure is None
        assert [commit.stream_id for commit in commits] == ["observation-3-0"]
