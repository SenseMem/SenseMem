"""Public SDK events retain original evidence across long activities and process failures."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from _feature_support import TinyEmbedder
from test_event_stream import CountingEmbedder, ScriptedSegmenter, _chunk, _chunks

from sensemem import (
    AsyncEventStream,
    AsyncMemory,
    EventObservation,
    EventSegment,
    Memory,
    MemoryRecord,
    MemorySettings,
    ModelError,
    ObservationContext,
    RetrievalScope,
    SpatialAnchor,
    SpatialContext,
    StorageError,
    StreamInput,
    ValidationError,
)
from sensemem.infrastructure.local.event_checkpoint import Checkpoint, EventCheckpoint


class ContinuingSegmenter:
    event_model = "long-activity-fixture"
    event_recipe = "long-activity-fixture:v1"

    def __init__(self, complete_after: int) -> None:
        self.complete_after = complete_after
        self.calls: list[tuple[EventObservation, ...]] = []

    def segment_events(
        self,
        observations: Sequence[EventObservation],
        *,
        end_of_stream: bool = False,
    ) -> tuple[EventSegment, ...]:
        self.calls.append(tuple(observations))
        return (
            EventSegment(
                len(observations),
                "Repair the bicycle; the wheel is back in place."
                if len(self.calls) == self.complete_after
                else "Repairing the bicycle; the wheel is still off and the outcome is unknown.",
                complete=len(self.calls) == self.complete_after,
            ),
        )


@pytest.mark.asyncio
async def test_long_activity_keeps_one_event_and_all_media_with_bounded_analysis(
    tmp_path: Path,
) -> None:
    segmenter = ContinuingSegmenter(130)
    windows = tuple(_chunk(index, paired=True) for index in range(130))
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(
                memory,
                segmenter=segmenter,
                max_chunks=2,
                checkpoint_path=tmp_path / "capture.json",
                capture=True,
            ).consume(_chunks(*windows))
        ]
        assert len(commits) == 65
        descriptors = [
            cast(dict[str, object], commit.record.metadata["event_segmentation"])
            for commit in commits
        ]
        assert len({descriptor["event_id"] for descriptor in descriptors}) == 1
        assert [descriptor["part_index"] for descriptor in descriptors] == list(range(65))
        assert {descriptor["part_count"] for descriptor in descriptors} == {65}
        assert {descriptor["occurred_at"] for descriptor in descriptors} == {
            windows[0].occurred_at.isoformat()
        }
        assert {descriptor["occurred_end"] for descriptor in descriptors} == {
            windows[-1].occurred_end.isoformat()
        }
        assert [len(commit.record.assets) for commit in commits] == [4] * 65
        assert all(len(cast(list[object], descriptor["chunks"])) <= 2 for descriptor in descriptors)
        assert [
            (chunk["occurred_at"], chunk["occurred_end"])
            for descriptor in descriptors
            for chunk in cast(list[dict[str, object]], descriptor["chunks"])
        ] == [
            (window.occurred_at.isoformat(), window.occurred_end.isoformat()) for window in windows
        ]
        assert len({descriptor["summary"] for descriptor in descriptors}) == 1
        assert {descriptor["summary_coverage"] for descriptor in descriptors} == {"unverified"}
        assert {
            asset.path.read_bytes()
            for commit in commits
            for asset in commit.record.assets
            if asset.path is not None
        } == {blob.data for window in windows for blob in window.media}
        assert [len(call) for call in segmenter.calls] == [1] + [2] * 129
        # Each new chunk is analyzed once, with only its immediately preceding original
        # chunk repeated as context rather than repeatedly analyzing the growing event.
        assert sum(len(item.content.assets) for call in segmenter.calls for item in call) == 518
        assert all(not item.content.text for item in segmenter.calls[1])
        assert all("NOT ORIGINAL EVIDENCE" in call[0].content.text for call in segmenter.calls[2:])
        for index, call in enumerate(segmenter.calls[2:], start=2):
            assert "not a new chunk or an additional boundary offset" in call[0].content.text
            assert tuple(asset.sha256 for asset in call[0].content.assets) == tuple(
                hashlib.sha256(blob.data).hexdigest() for blob in windows[index - 1].media
            )
        assert len((await memory.list(limit=100)).items) == 65
    assert not tuple((tmp_path / "capture.json.media").iterdir())


@pytest.mark.asyncio
async def test_checkpoint_spools_long_continuation_and_resumes_byte_bounded_parts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "capture.json"
    windows = tuple(_chunk(index, data=bytes([index + 1]) * 40) for index in range(8))
    budget = 80
    assert sum(len(blob.data) for chunk in windows for blob in chunk.media) > budget
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        first = ContinuingSegmenter(100)
        stream = AsyncEventStream(
            memory,
            segmenter=first,
            max_buffer_bytes=budget,
            checkpoint_path=checkpoint,
            capture=True,
        )
        for window in windows[:4]:
            await stream.append(window)
            assert [commit async for commit in stream.process()] == []
        assert [len(call) for call in first.calls] == [1, 2, 2, 2]
        assert (await memory.list()).items == ()
        saved = json.loads(checkpoint.read_text())
        assert type(saved["version"]) is int and saved["version"] == 5
        assert saved["max_buffer_bytes"] == budget
        assert len(saved["streams"][0]["chunks"]) == 4

    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        with pytest.raises(ValidationError, match="same max_buffer_bytes"):
            AsyncEventStream(memory, segmenter=ContinuingSegmenter(4), checkpoint_path=checkpoint)
        last = ContinuingSegmenter(4)
        resumed = AsyncEventStream(
            memory,
            segmenter=last,
            max_buffer_bytes=budget,
            checkpoint_path=checkpoint,
            capture=True,
        )
        commits = []
        for window in windows[4:]:
            await resumed.append(window)
            commits.extend([commit async for commit in resumed.process()])
        assert [len(call) for call in last.calls] == [2] * 4
        assert len(commits) == 4
        assert [len(commit.record.assets) for commit in commits] == [2] * 4
        assert [
            (await memory.read_asset(commit.record.id, asset.id)).data
            for commit in commits
            for asset in commit.record.assets
        ] == [blob.data for window in windows for blob in window.media]
        descriptors = [commit.record.metadata["event_segmentation"] for commit in commits]
        assert all(isinstance(descriptor, dict) for descriptor in descriptors)
        assert [
            cast(dict[str, object], descriptor)["part_index"] for descriptor in descriptors
        ] == [
            0,
            1,
            2,
            3,
        ]
        assert (
            len({cast(dict[str, object], descriptor)["event_id"] for descriptor in descriptors})
            == 1
        )
        assert resumed.positions == {"default": windows[-1].occurred_end}
    assert not tuple((tmp_path / "capture.json.media").iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["assessment", "acknowledgement"])
async def test_checkpoint_publish_failure_preserves_retryable_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    checkpoint = tmp_path / "capture.json"
    segmenter = ScriptedSegmenter([(EventSegment(1, "Finished repair"),)] * 2)
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(
            memory, segmenter=segmenter, checkpoint_path=checkpoint, capture=True
        )
        await stream.append(_chunk(0))
        original_save = EventCheckpoint.save
        failed = False

        def interrupted_save(self: EventCheckpoint, saved: Checkpoint) -> None:
            nonlocal failed
            target = (
                bool(saved.events)
                if phase == "assessment"
                else not (saved.pending or saved.events or saved.streams[0].chunks)
            )
            if target and not failed:
                failed = True
                raise StorageError("checkpoint filesystem is full")
            original_save(self, saved)

        with monkeypatch.context() as patch:
            patch.setattr(EventCheckpoint, "save", interrupted_save)
            with pytest.raises(StorageError, match="filesystem is full"):
                _ = [commit async for commit in stream.process()]
        assert failed
        stored = len((await memory.list()).items)
        assert stored == int(phase == "acknowledgement")
        assert tuple((tmp_path / "capture.json.media").iterdir())
        commits = [commit async for commit in stream.process()]
        assert len(commits) == 1
        assert len((await memory.list()).items) == 1
        assert len(segmenter.calls) == (2 if phase == "assessment" else 1)
    assert not tuple((tmp_path / "capture.json.media").iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", ["bytes", "chunks"])
async def test_bounded_storage_parts_resume_after_a_lost_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    checkpoint = tmp_path / "capture.json"
    windows = tuple(_chunk(index, data=bytes([index + 1]) * 40) for index in range(6))
    second_id: str | None = None
    max_chunks = 2 if bound == "chunks" else 32
    max_buffer_bytes = 80 if bound == "bytes" else 1024
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=CountingEmbedder())) as memory:
        stream = AsyncEventStream(
            memory,
            segmenter=ContinuingSegmenter(6),
            max_buffer_bytes=max_buffer_bytes,
            max_chunks=max_chunks,
            checkpoint_path=checkpoint,
            capture=False,
        )
        for window in windows[:-1]:
            await stream.append(window)
            assert [commit async for commit in stream.process()] == []
        await stream.append(windows[-1])
        original = memory._write_stream_item
        writes = 0

        async def interrupted_write(item: StreamInput, *, capture: bool) -> MemoryRecord:
            nonlocal writes, second_id
            record = await original(item, capture=capture)
            writes += 1
            if writes == 2:
                second_id = record.id
                raise StorageError("SQLite committed before acknowledgement was lost")
            return record

        with monkeypatch.context() as patch:
            patch.setattr(memory, "_write_stream_item", interrupted_write)
            with pytest.raises(StorageError):
                _ = [commit async for commit in stream.process()]
        assert len((await memory.list()).items) == 2
        assert json.loads(checkpoint.read_text())["events"][0]["next_part"] == 1
    embedder = CountingEmbedder()
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=embedder)) as memory:
        assert embedder.calls == 0
        segmenter = ContinuingSegmenter(100)
        with pytest.raises(ValidationError, match="same max_chunks"):
            AsyncEventStream(
                memory,
                segmenter=segmenter,
                max_chunks=max_chunks + 1,
                max_buffer_bytes=max_buffer_bytes,
                checkpoint_path=checkpoint,
            )
        stream = AsyncEventStream(
            memory,
            segmenter=segmenter,
            max_buffer_bytes=max_buffer_bytes,
            max_chunks=max_chunks,
            checkpoint_path=checkpoint,
            capture=False,
        )
        commits = [commit async for commit in stream.process()]
        assert len(commits) == 2
        assert commits[0].record.id == second_id
        assert segmenter.calls == []
        records = (await memory.list()).items
        assert len(records) == 3
        assert {
            (await memory.read_asset(record.id, asset.id)).data
            for record in records
            for asset in record.assets
        } == {blob.data for window in windows for blob in window.media}
    reopened_embedder = CountingEmbedder()
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=reopened_embedder)) as memory:
        assert len((await memory.list()).items) == 3
        assert reopened_embedder.calls == 0
    assert not tuple((tmp_path / "capture.json.media").iterdir())


@pytest.mark.asyncio
async def test_original_event_windows_are_compressed_only_after_the_configured_bound(
    tmp_path: Path,
) -> None:
    segmenter = ContinuingSegmenter(4)
    windows = tuple(_chunk(index, paired=True) for index in range(4))
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        records = [
            commit.record
            async for commit in AsyncEventStream(
                memory, segmenter=segmenter, max_chunks=3, capture=True
            ).consume(_chunks(*windows))
        ]
        assert len(records) == 2
        assert [len(record.assets) for record in records] == [6, 2]
        assert [len(call) for call in segmenter.calls] == [1, 2, 3, 2]
        for count, call in enumerate(segmenter.calls[:3], start=1):
            assert all(not item.content.text for item in call)
            assert [asset.sha256 for item in call for asset in item.content.assets] == [
                hashlib.sha256(blob.data).hexdigest()
                for chunk in windows[:count]
                for blob in chunk.media
            ]
        last = segmenter.calls[-1]
        assert "NOT ORIGINAL EVIDENCE" in last[0].content.text
        assert last[0].occurred_at == windows[0].occurred_at
        assert last[0].occurred_end == windows[2].occurred_end
        assert [asset.sha256 for item in last for asset in item.content.assets] == [
            hashlib.sha256(blob.data).hexdigest() for chunk in windows[2:] for blob in chunk.media
        ]


@pytest.mark.asyncio
async def test_event_location_parts_preserve_chunk_context_and_scope(tmp_path: Path) -> None:
    kitchen = ObservationContext(
        place_id="kitchen",
        spatial=SpatialContext(frame_id="home", anchor=SpatialAnchor.SUBJECT, x=1, y=2),
    )
    lounge = ObservationContext(
        place_id="lounge",
        spatial=SpatialContext(frame_id="home", anchor=SpatialAnchor.SUBJECT, x=10, y=2),
    )
    windows = (replace(_chunk(0), context=kitchen), replace(_chunk(1), context=lounge))
    segmenter = ScriptedSegmenter([(), (EventSegment(2, "Prepare and drink tea"),)])
    async with AsyncMemory(
        Memory(
            tmp_path / "memory",
            embedder=TinyEmbedder(),
            settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
        )
    ) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(memory, segmenter=segmenter).consume(
                _chunks(*windows)
            )
        ]
        assert [commit.record.place_id for commit in commits] == ["kitchen", "lounge"]
        assert [
            commit.record.context.spatial for commit in commits if commit.record.context is not None
        ] == [kitchen.spatial, lounge.spatial]
        assert [
            hit.id
            for hit in (await memory.search("tea", scope=RetrievalScope(place_id="kitchen"))).hits
        ] == [commits[0].record.id]
        descriptors = [
            cast(dict[str, object], commit.record.metadata["event_segmentation"])
            for commit in commits
        ]
        assert descriptors[0]["event_id"] == descriptors[1]["event_id"]
        assert descriptors[0]["part_count"] == 2
        assert all("Prepare and drink tea" not in commit.record.content for commit in commits)
        assert all(descriptor["summary"] == "Prepare and drink tea" for descriptor in descriptors)
        assert all(descriptor["summary_coverage"] == "unverified" for descriptor in descriptors)
        saved_chunks = cast(list[dict[str, object]], descriptors[0]["chunks"])
        assert cast(dict[str, object], saved_chunks[0]["context"])["place_id"] == "kitchen"
        assert saved_chunks[0]["asset_ids"] == [asset.id for asset in commits[0].record.assets]


@pytest.mark.asyncio
async def test_event_synopsis_is_unverified_interpretation_and_never_replayed_as_original_text(
    tmp_path: Path,
) -> None:
    summary = "A square reached its target."
    async with AsyncMemory(Memory(tmp_path / "source", embedder=TinyEmbedder())) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(
                memory,
                segmenter=ScriptedSegmenter([(EventSegment(1, summary),)]),
                capture=True,
            ).consume(_chunks(_chunk(0, paired=True)))
        ]
        record_id = commits[0].record.id
    with (
        Memory(tmp_path / "source", embedder=TinyEmbedder()) as source,
        Memory(tmp_path / "destination", embedder=TinyEmbedder()) as destination,
    ):
        record = source.get(record_id)
        assert [part.kind for part in record.parts] == ["asset", "asset", "model_interpretation"]
        assert record.content.endswith(json.dumps(summary, ensure_ascii=False))
        assert "Summary coverage: unverified" in record.content
        assert "not evidence of its absence" in record.content
        metadata = cast(dict[str, object], record.metadata["event_segmentation"])
        assert metadata["supplied_modalities"] == ["audio", "image"]
        assert metadata["summary_coverage"] == "unverified"
        assert source.settle() == 1
        bundle = source.compile("square target")
        assert bundle.hits and bundle.hits[0].parts[-1].kind == "model_interpretation"
        assert "Summary coverage: unverified" in bundle.render()
        replayed = tuple(source.replay_sources(destination))
        assert len(replayed) == 1
        assert replayed[0].content == ""
        assert [part.kind for part in replayed[0].parts] == ["asset", "asset"]
        assert {source.read_asset(record.id, asset.id).data for asset in record.assets} == {
            destination.read_asset(replayed[0].id, asset.id).data for asset in replayed[0].assets
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("axis", ["place", "source"])
async def test_identical_media_from_distinct_capture_contexts_have_distinct_event_ids(
    tmp_path: Path, axis: str
) -> None:
    contexts = (
        (ObservationContext(place_id="kitchen"), ObservationContext(place_id="lounge"))
        if axis == "place"
        else (ObservationContext(source_id="camera-a"), ObservationContext(source_id="camera-b"))
    )
    original = _chunk(0, paired=True)
    windows = tuple(
        replace(original, stream_id=f"capture-{index}", context=context)
        for index, context in enumerate(contexts)
    )
    segmenter = ScriptedSegmenter([(EventSegment(1, "A finished activity"),)] * 2)
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        records = [
            commit.record
            async for commit in AsyncEventStream(memory, segmenter=segmenter, capture=True).consume(
                _chunks(*windows)
            )
        ]
        assert len({record.id for record in records}) == 2
        descriptors = [
            cast(dict[str, object], record.metadata["event_segmentation"]) for record in records
        ]
        assert len({descriptor["event_id"] for descriptor in descriptors}) == 2
        assert [descriptor["part_count"] for descriptor in descriptors] == [1, 1]


@pytest.mark.asyncio
async def test_provider_failure_checkpoints_input_before_model_and_retries_on_reopen(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "capture.json"
    context = ObservationContext(
        source_id="capture-fixture",
        place_id="kitchen",
        valid_from=_chunk(0).occurred_at,
        spatial=SpatialContext(frame_id="home", anchor=SpatialAnchor.SUBJECT, x=1, y=2),
    )
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        with pytest.raises(ModelError):
            _ = [
                commit
                async for commit in AsyncEventStream(
                    memory,
                    segmenter=ScriptedSegmenter([RuntimeError("provider failed")]),
                    checkpoint_path=checkpoint,
                    context=context,
                ).consume(_chunks(_chunk(0, paired=True)))
            ]
        assert (await memory.list()).items == ()
        assert json.loads(checkpoint.read_text())["streams"][0]["needs_assessment"] is True
    recovered = ScriptedSegmenter(
        [(EventSegment(1, "Still repairing", complete=False),), (EventSegment(2, "Finish repair"),)]
    )
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        resumed = AsyncEventStream(
            memory, segmenter=recovered, checkpoint_path=checkpoint, context=context
        )
        assert resumed.positions == {"default": _chunk(0).occurred_end}
        commits = [commit async for commit in resumed.consume(_chunks(_chunk(1, paired=True)))]
        assert len(commits) == 1
        assert len(commits[0].record.assets) == 4
        assert commits[0].record.place_id == "kitchen"
        assert commits[0].record.context is not None
        assert commits[0].record.context.spatial == context.spatial
        assert commits[0].record.context.valid_from == context.valid_from
        assert [len(call[0]) for call in recovered.calls] == [1, 2]
        assert recovered.calls[0][1] is False


@pytest.mark.asyncio
async def test_incomplete_eof_is_checkpointed_without_promoting_it_to_a_memory(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "capture.json"
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        segmenter = ScriptedSegmenter(
            [
                (EventSegment(1, "Still repairing", complete=False),),
                (EventSegment(1, "Still repairing", complete=False),),
            ]
        )
        assert [
            commit
            async for commit in AsyncEventStream(
                memory, segmenter=segmenter, checkpoint_path=checkpoint
            ).consume(_chunks(_chunk(0)))
        ] == []
        assert (await memory.list()).items == ()
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(
                memory,
                segmenter=ScriptedSegmenter([(EventSegment(2, "Finish repair"),)]),
                checkpoint_path=checkpoint,
            ).consume(_chunks(_chunk(1)))
        ]
        assert len(commits) == 1
        assert len(commits[0].record.assets) == 2


@pytest.mark.asyncio
async def test_recovery_replays_an_unacknowledged_durable_part_without_duplicate_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "capture.json"
    windows = (
        replace(_chunk(0), context=ObservationContext(place_id="kitchen")),
        replace(_chunk(1), context=ObservationContext(place_id="lounge")),
    )
    second_id: str | None = None
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        original = memory._write_stream_item
        writes = 0

        async def interrupted_write(item: StreamInput, *, capture: bool) -> MemoryRecord:
            nonlocal writes, second_id
            record = await original(item, capture=capture)
            writes += 1
            if writes == 2:
                second_id = record.id
                raise StorageError("process lost acknowledgement after SQLite committed")
            return record

        with monkeypatch.context() as patch:
            patch.setattr(memory, "_write_stream_item", interrupted_write)
            with pytest.raises(StorageError):
                _ = [
                    commit
                    async for commit in AsyncEventStream(
                        memory,
                        segmenter=ScriptedSegmenter([(), (EventSegment(2, "Finish making tea"),)]),
                        checkpoint_path=checkpoint,
                    ).consume(_chunks(*windows))
                ]
        assert len((await memory.list()).items) == 2
    recovered = ScriptedSegmenter([])
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        commits = [
            commit
            async for commit in AsyncEventStream(
                memory, segmenter=recovered, checkpoint_path=checkpoint
            ).consume(_chunks())
        ]
        assert [commit.record.id for commit in commits] == [second_id]
        assert len((await memory.list()).items) == 2
        assert recovered.calls == []


@pytest.mark.asyncio
async def test_full_analysis_window_requires_continuation_but_preserves_retryable_input(
    tmp_path: Path,
) -> None:
    segmenter = ScriptedSegmenter([(), (), (EventSegment(2, "Finish repair"),)])
    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(memory, segmenter=segmenter, max_chunks=2)
        with pytest.raises(ModelError, match="summarize"):
            _ = [
                commit async for commit in stream.consume(_chunks(_chunk(0), _chunk(1), _chunk(2)))
            ]
        assert (await memory.list()).items == ()
        commits = [commit async for commit in stream.consume(_chunks())]
        assert len(commits) == 1
        assert len(commits[0].record.assets) == 2


@pytest.mark.parametrize(
    "version",
    [pytest.param(None, id="missing"), 1, 2, 3, 4, 5.0, "5", True],
)
def test_invalid_event_checkpoint_version_is_rejected_without_changing_the_original(
    tmp_path: Path, version: int | float | str | None
) -> None:
    backend = ScriptedSegmenter([])
    with Memory(tmp_path / "memory", embedder=TinyEmbedder()) as core:
        checkpoint = tmp_path / "capture.json"
        manifest: dict[str, object] = {
            "model": backend.event_model,
            "recipe": backend.event_recipe,
            "data_dir": str(core.data_dir),
            "max_buffer_bytes": 64 * 1024 * 1024,
            "max_chunks": 32,
            "streams": [],
            "events": [],
            "pending": [],
        }
        if version is not None:
            manifest["version"] = version
        checkpoint.write_text(json.dumps(manifest))
        original = checkpoint.read_bytes()
        with pytest.raises(ValidationError, match="unsupported format"):
            AsyncEventStream(AsyncMemory(core), segmenter=backend, checkpoint_path=checkpoint)
        assert backend.calls == []
        assert core.list().items == ()
        assert checkpoint.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["recipe", "media", "schema", "directory", "continuation"])
async def test_checkpoint_rejects_incompatible_or_damaged_state(tmp_path: Path, fault: str) -> None:
    checkpoint = tmp_path / "capture.json"
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        assert [
            commit
            async for commit in AsyncEventStream(
                memory, segmenter=ScriptedSegmenter([(), ()]), checkpoint_path=checkpoint
            ).consume(_chunks(_chunk(0)))
        ] == []
        if fault == "media":
            next((tmp_path / "capture.json.media").iterdir()).write_bytes(b"corrupted")
        else:
            saved = json.loads(checkpoint.read_text())
            if fault == "continuation":
                saved["streams"][0]["continuation"] = " "
                saved["streams"][0]["continuation_chunks"] = 1
            else:
                field = {"recipe": "recipe", "schema": "version", "directory": "data_dir"}[fault]
                saved[field] = "incompatible"
            checkpoint.write_text(json.dumps(saved))
        with pytest.raises(StorageError if fault == "media" else ValidationError):
            AsyncEventStream(memory, segmenter=ScriptedSegmenter([]), checkpoint_path=checkpoint)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["reordered", "mixed_stream"])
async def test_checkpoint_rejects_invalid_completed_event_evidence_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    checkpoint = tmp_path / "capture.json"
    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:

        async def unavailable_store(item: StreamInput, *, capture: bool) -> MemoryRecord:
            raise StorageError("fixture store unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(memory, "_write_stream_item", unavailable_store)
            with pytest.raises(StorageError):
                _ = [
                    commit
                    async for commit in AsyncEventStream(
                        memory,
                        segmenter=ScriptedSegmenter([(), (EventSegment(2, "Finished activity"),)]),
                        checkpoint_path=checkpoint,
                        capture=True,
                    ).consume(_chunks(_chunk(0), _chunk(1)))
                ]
        saved = json.loads(checkpoint.read_text())
        if fault == "reordered":
            saved["events"][0]["chunks"].reverse()
        else:
            saved["events"][0]["chunks"][1]["stream_id"] = "different-capture"
        checkpoint.write_text(json.dumps(saved))
        recovered = ScriptedSegmenter([])
        with pytest.raises(ValidationError, match="delivery state"):
            AsyncEventStream(memory, segmenter=recovered, checkpoint_path=checkpoint)
        assert recovered.calls == []
        assert (await memory.list()).items == ()


@pytest.mark.asyncio
async def test_checkpoint_failure_does_not_advance_the_source_or_call_a_model(
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
                _ = [commit async for commit in stream.consume(_chunks(_chunk(0)))]
        assert stream.positions == {}
        assert segmenter.calls == []
        assert len([commit async for commit in stream.consume(_chunks(_chunk(0)))]) == 1


def test_a_process_exit_during_inference_leaves_accepted_media_restartable(tmp_path: Path) -> None:
    program = """
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from _feature_support import TinyEmbedder
from test_event_stream import _chunk, _chunks
from sensemem import AsyncEventStream, AsyncMemory, Memory
class Crash:
    event_model = 'event-test-snapshot'
    event_recipe = 'event-test-snapshot:complete-prefix-v1'
    def segment_events(self, observations, *, end_of_stream=False):
        os._exit(7)
async def run():
    async with AsyncMemory(Memory(sys.argv[1], embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(memory, segmenter=Crash(), checkpoint_path=Path(sys.argv[2]))
        _ = [commit async for commit in stream.consume(_chunks(_chunk(0)))]
asyncio.run(run())
"""
    checkpoint = tmp_path / "capture.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(tmp_path / "memory"),
            str(checkpoint),
            str(Path(__file__).parent),
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 7, result.stderr.decode()

    async def recover() -> None:
        async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
            assert (await memory.list()).items == ()
            commits = [
                commit
                async for commit in AsyncEventStream(
                    memory,
                    segmenter=ScriptedSegmenter([(EventSegment(1, "Finish repair"),)]),
                    checkpoint_path=checkpoint,
                ).consume(_chunks())
            ]
            assert len(commits) == 1
            assert len(commits[0].record.assets) == 1

    asyncio.run(recover())
