"""Native event completion drives bounded background context and proactive evidence."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import Literal

import pytest
from _feature_support import TinyEmbedder
from fixtures.events import ScriptedSegmenter
from fixtures.events import make_chunk as _chunk
from fixtures.live import ControlledMemory
from PIL import Image

from examples.live.capture import EventCapture
from examples.live.recall import LiveRecall
from examples.live.updates import LiveEvent, LiveUpdates, Trigger
from sensemem import (
    AsyncCaptureStream,
    AsyncEventStream,
    Blob,
    EventObservation,
    EventSegment,
    Memory,
    MemorySettings,
    ModelError,
    ObservationContext,
    StreamCommit,
    StreamEvent,
    StreamInput,
    StreamPhase,
)

NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)
SOURCE_ID = "live:" + "a" * 32


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[ControlledMemory]:
    value = ControlledMemory(
        Memory(
            tmp_path / "memory",
            embedder=TinyEmbedder(),
            settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
        )
    )
    async with value:
        await value.add("The spare keys are in the blue toolbox.")
        yield value


async def _commit(
    memory: ControlledMemory,
    event_id: str,
    *,
    part_index: int = 0,
    part_count: int = 1,
    modality: Literal["audio", "image", "video"] = "image",
) -> StreamCommit:
    record = await memory.capture(
        Blob(
            f"{event_id}:{part_index}".encode(),
            {"audio": "audio/wav", "image": "image/png", "video": "video/mp4"}[modality],
        ),
        occurred_at=NOW,
        occurred_end=NOW + timedelta(seconds=2),
        context=ObservationContext(source_id=SOURCE_ID),
        metadata={
            "event_segmentation": {
                "event_id": event_id,
                "summary": f"Observed activity {event_id}",
                "summary_coverage": "unverified",
                "part_index": part_index,
                "part_count": part_count,
                "occurred_at": NOW.isoformat(),
                "occurred_end": (NOW + timedelta(seconds=2)).isoformat(),
                "supplied_modalities": [modality],
            }
        },
    )
    return StreamCommit(record, None, pending_settlement=True)


async def test_native_events_publish_context_without_an_optional_response_assessor(
    memory: ControlledMemory,
) -> None:

    received: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall = LiveRecall(memory)
    recall.sender = received.put
    updates = LiveUpdates(memory, recall)
    updates.observe(await _commit(memory, "native-only"))
    event = await asyncio.wait_for(received.get(), 3)
    context = await asyncio.wait_for(received.get(), 3)
    assert event["type"] == "memory_event"
    assert context["type"] == "context"
    assert context["event_id"] == event["event_id"]
    assert updates.assessor is None and received.empty()
    await updates.close()
    await recall.close()


@pytest.mark.parametrize("frame_count", [1, 2], ids=["point", "window"])
async def test_committed_visual_window_triggers_event_while_audiovisual_activity_is_unfinished(
    memory: ControlledMemory,
    tmp_path: Path,
    frame_count: int,
) -> None:

    received: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    assessed: list[LiveEvent] = []
    recall = LiveRecall(memory)
    recall.sender = received.put

    async def assess(event: LiveEvent, revision: int, trigger: Trigger) -> dict[str, object]:
        assessed.append(event)
        assert revision == 0 and trigger == "visual"
        return {
            "candidate_id": event.event_id,
            "utterance": "I saw where you placed the key.",
            "memory_revision": revision,
        }

    updates = LiveUpdates(memory, recall, assess)
    activity = AsyncEventStream(
        memory,
        segmenter=ScriptedSegmenter(
            [(EventSegment(1, "The surrounding activity is ongoing.", complete=False),)]
        ),
        capture=True,
        context=ObservationContext(source_id=SOURCE_ID),
        checkpoint_path=tmp_path / "ongoing-activity.json",
    )
    await activity.append(_chunk(0))
    assert [commit async for commit in activity.process()] == []
    assert updates.latest is None and received.empty()

    originals: list[bytes] = []
    for color in ("green", "blue")[:frame_count]:
        encoded = BytesIO()
        Image.new("RGB", (16, 16), color).save(encoded, format="JPEG")
        originals.append(encoded.getvalue())
    summary = "A key is visible inside the green box."
    started = NOW + timedelta(seconds=2)
    ended = NOW + timedelta(seconds=3) if frame_count > 1 else None

    async def observed_window() -> AsyncIterator[StreamEvent]:
        yield StreamEvent(StreamPhase.UPDATE, Blob(originals[0], "image/jpeg"), "visual-window")
        assert updates.latest is None and received.empty()
        yield StreamEvent(
            StreamPhase.FINAL,
            StreamInput(
                tuple(
                    Blob(original, "image/jpeg", f"key-location-{index}.jpg")
                    for index, original in enumerate(originals)
                ),
                occurred_at=started,
                occurred_end=ended,
                context=ObservationContext(source_id=SOURCE_ID),
                interpretation=summary,
                metadata={
                    "visual_observation": {
                        "perception_id": "visual-window",
                        "boundary": "observed_window",
                    }
                },
            ),
            "visual-window",
        )

    commits = [
        commit
        async for commit in AsyncCaptureStream(memory, capture=True).consume(observed_window())
    ]
    (commit,) = commits
    assert commit.pending_settlement and "event_segmentation" not in commit.record.metadata
    assert commit.record.occurred_at == started and commit.record.occurred_end == ended
    assert commit.record.context is not None and commit.record.context.source_id == SOURCE_ID
    assert len(commit.record.assets) == frame_count
    for asset, original in zip(commit.record.assets, originals, strict=True):
        assert (await memory.read_asset(commit.record.id, asset.id)).data == original
    updates.observe(commit)
    event = await asyncio.wait_for(received.get(), 3)
    remaining = [await asyncio.wait_for(received.get(), 3) for _ in range(2)]
    context = next(message for message in remaining if message["type"] == "context")
    candidate = next(message for message in remaining if message["type"] == "proactive")
    assert event["type"] == "memory_event" and event["event_id"] == "visual-window"
    assert event["source_id"] == SOURCE_ID
    assert event["evidence_ids"] == [commit.record.id]
    assert event["occurred_at"] == started.isoformat()
    assert event["occurred_end"] == (ended or started).isoformat()
    assert str(event["summary"]).endswith(summary)
    assert context["event_id"] == candidate["event_id"] == event["event_id"]
    assert context["evidence_ids"] == candidate["evidence_ids"] == [commit.record.id]
    assert candidate["trigger"] == "visual"
    assert assessed[0].modalities == ("image",)
    assert activity.positions == {"default": _chunk(0).occurred_end}
    await updates.close()
    await recall.close()


@pytest.mark.parametrize(
    ("modality", "expected_trigger"), [("video", "visual"), ("audio", "audio")]
)
async def test_all_native_parts_are_required_and_pending_observations_stay_durable(
    memory: ControlledMemory,
    modality: Literal["audio", "video"],
    expected_trigger: str,
) -> None:

    received: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    assessed: list[LiveEvent] = []
    recall = LiveRecall(memory)
    recall.sender = received.put

    async def assess(event: LiveEvent, revision: int, trigger: Trigger) -> dict[str, object]:
        assessed.append(event)
        assert revision == 0 and trigger == expected_trigger
        return {
            "candidate_id": event.event_id,
            "utterance": "要我帮你找钥匙吗?",
            "reason": "fixture",
            "memory_revision": revision,
        }

    updates = LiveUpdates(memory, recall, assess)
    first = await _commit(memory, "group", part_index=0, part_count=2, modality=modality)
    second = await _commit(memory, "group", part_index=1, part_count=2, modality=modality)
    updates.observe(first)
    assert updates.latest is None and updates.task is None
    updates.observe(second)
    event = await asyncio.wait_for(received.get(), 3)
    remaining = [await asyncio.wait_for(received.get(), 3) for _ in range(2)]
    context = next(message for message in remaining if message["type"] == "context")
    candidate = next(message for message in remaining if message["type"] == "proactive")
    assert event["type"] == "memory_event" and event["summary_verified"] is False
    assert event["source_id"] == SOURCE_ID
    assert event["evidence_ids"] == [first.record.id, second.record.id]
    assert context["type"] == "context" and context["memory_revision"] == 0
    assert candidate["type"] == "proactive"
    assert candidate["trigger"] == expected_trigger
    assert candidate["evidence_ids"] == event["evidence_ids"]
    assert assessed[0].evidence_ids == (first.record.id, second.record.id)
    assert assessed[0].modalities == (modality,)
    assert {pending.memory_id for pending in await memory.pending_captures()} == {
        first.record.id,
        second.record.id,
    }
    updates.observe(second)
    assert received.empty()
    await updates.close()
    await recall.close()


async def test_late_older_event_cannot_replace_a_newer_window_from_the_same_source(
    memory: ControlledMemory,
) -> None:

    received: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall = LiveRecall(memory)
    recall.sender = received.put
    updates = LiveUpdates(memory, recall)
    current = LiveEvent(
        event_id="current-visual-window",
        source_id=SOURCE_ID,
        evidence_ids=("current-evidence",),
        summary="The key is in the green box.",
        occurred_at=NOW,
        occurred_end=NOW + timedelta(seconds=3),
        modalities=("image",),
    )
    updates._enqueue(current)
    assert (await asyncio.wait_for(received.get(), 3))["event_id"] == current.event_id
    assert (await asyncio.wait_for(received.get(), 3))["type"] == "context"
    work = updates._work

    older = replace(
        current,
        event_id="late-audiovisual-event",
        occurred_end=NOW + timedelta(seconds=2),
        modalities=("video",),
    )
    updates._enqueue(older)
    assert updates.latest is current and updates._work is work
    assert updates._pending_event is None and received.empty()

    another_source = replace(older, source_id="live:" + "b" * 32)
    updates._enqueue(another_source)
    event = await asyncio.wait_for(received.get(), 3)
    context = await asyncio.wait_for(received.get(), 3)
    assert event["type"] == "memory_event" and event["event_id"] == older.event_id
    assert event["source_id"] == another_source.source_id
    assert context["type"] == "context" and context["event_id"] == older.event_id
    assert updates.latest is another_source and received.empty()
    await updates.close()
    await recall.close()


async def test_latest_event_coalesces_without_withdrawing_revalidated_past_events(
    memory: ControlledMemory,
) -> None:

    messages: list[dict[str, object]] = []
    entered: asyncio.Queue[str] = asyncio.Queue()
    assessed: list[tuple[str, int]] = []
    release = asyncio.Event()
    ready = asyncio.Event()
    invalidated = asyncio.Event()
    recall = LiveRecall(memory)

    async def send(message: dict[str, object]) -> None:
        messages.append(message)
        if message["type"] == "proactive" and message["event_id"] == "latest":
            ready.set()
        if message == {"type": "invalidated", "memory_revision": 2}:
            invalidated.set()

    async def assess(event: LiveEvent, revision: int, trigger: Trigger) -> dict[str, object]:
        assessed.append((event.event_id, revision))
        entered.put_nowait(event.event_id)
        if event.event_id == "first":
            await release.wait()
        return {
            "candidate_id": event.event_id,
            "utterance": "fixture",
            "memory_revision": updates.memory_revision,
        }

    recall.sender = send
    updates = LiveUpdates(memory, recall, assess)
    first = await _commit(memory, "first")
    updates.observe(first)
    assert await asyncio.wait_for(entered.get(), 3) == "first"
    worker = updates.task
    updates.memory_changed(1)
    updates.observe(await _commit(memory, "superseded"))
    updates.memory_changed(2)
    latest = await _commit(memory, "latest")
    updates.observe(latest)
    assert updates.task is worker
    await asyncio.wait_for(invalidated.wait(), 3)
    assert not release.is_set()
    release.set()
    await asyncio.wait_for(ready.wait(), 3)
    await updates.close()
    await recall.close()
    assert assessed == [("first", 0), ("latest", 2)]
    assert memory.calls == ["Observed activity first", "Observed activity latest"]
    candidates = [message for message in messages if message["type"] == "proactive"]
    assert [candidate["event_id"] for candidate in candidates] == ["first", "latest"]
    assert [candidate["evidence_ids"] for candidate in candidates] == [
        [first.record.id],
        [latest.record.id],
    ]
    assert all(candidate["memory_revision"] == 2 for candidate in candidates)
    assert {"type": "invalidated", "memory_revision": 2} in messages


@pytest.mark.parametrize("revalidated", [False, True])
@pytest.mark.parametrize("newer_event", [False, True])
async def test_review_runs_alongside_context_and_requires_current_evidence_revalidation(
    memory: ControlledMemory,
    revalidated: bool,
    newer_event: bool,
) -> None:

    memory.block = True
    messages: list[dict[str, object]] = []
    entered: asyncio.Queue[int] = asyncio.Queue()
    release = asyncio.Event()
    ready = asyncio.Event()
    invalidated = asyncio.Event()
    context_ready = asyncio.Event()
    recall = LiveRecall(memory)

    async def send(message: dict[str, object]) -> None:
        messages.append(message)
        if message["type"] == "proactive":
            ready.set()
        if message == {"type": "invalidated", "memory_revision": 1}:
            invalidated.set()
        if message["type"] == "context":
            context_ready.set()

    async def assess(event: LiveEvent, revision: int, trigger: Trigger) -> dict[str, object] | None:
        assert trigger == ("visual" if revision == 0 or newer_event else "memory")
        entered.put_nowait(revision)
        if revision != 0:
            return None
        await release.wait()
        return {
            "candidate_id": event.event_id,
            "utterance": "fixture",
            "context": "independently reviewed original media and cited memories",
            "memory_revision": updates.memory_revision if revalidated else revision,
        }

    recall.sender = send
    updates = LiveUpdates(memory, recall, assess)
    original = await _commit(memory, "same-event")
    updates.observe(original)
    await asyncio.wait_for(memory.started.get(), 3)
    assert await asyncio.wait_for(entered.get(), 3) == 0
    updates.memory_changed(1)
    if newer_event:
        updates.observe(await _commit(memory, "newer-unrelated-event"))
    await asyncio.wait_for(invalidated.wait(), 3)
    release.set()
    if revalidated:
        await asyncio.wait_for(ready.wait(), 3)
        assert not memory.release.is_set()
    memory.release.set()
    assert await asyncio.wait_for(entered.get(), 3) == 1
    await asyncio.wait_for(context_ready.wait(), 3)
    await updates.close()
    await recall.close()
    candidates = [message for message in messages if message["type"] == "proactive"]
    assert len(candidates) == int(revalidated)
    if revalidated:
        assert candidates[0]["event_id"] == "same-event"
        assert candidates[0]["source_id"] == SOURCE_ID
        assert candidates[0]["evidence_ids"] == [original.record.id]
        assert candidates[0]["memory_revision"] == 1
        assert (
            candidates[0]["context"] == "independently reviewed original media and cited memories"
        )
    assert all(
        message["memory_revision"] == 1 for message in messages if message["type"] == "context"
    )
    assert all(
        message["event_id"] == ("newer-unrelated-event" if newer_event else "same-event")
        for message in messages
        if message["type"] == "context"
    )


async def test_changed_revision_rejects_context_that_was_already_in_flight(
    memory: ControlledMemory,
) -> None:

    memory.block = True
    messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    assessed: asyncio.Queue[int] = asyncio.Queue()
    recall = LiveRecall(memory)
    recall.sender = messages.put

    async def assess(event: LiveEvent, revision: int, trigger: Trigger) -> None:
        assessed.put_nowait(revision)

    updates = LiveUpdates(memory, recall, assess)
    updates.observe(await _commit(memory, "scene"))
    await memory.started.get()
    updates.memory_changed(1)
    memory.release.set()
    assert await asyncio.wait_for(assessed.get(), 3) == 0
    assert await asyncio.wait_for(assessed.get(), 3) == 1
    contexts: list[dict[str, object]] = []
    while not contexts:
        message = await asyncio.wait_for(messages.get(), 3)
        if message["type"] == "context":
            contexts.append(message)
    await updates.close()
    await recall.close()
    while not messages.empty():
        message = messages.get_nowait()
        if message["type"] == "context":
            contexts.append(message)
    assert len(contexts) == 1 and contexts[0]["memory_revision"] == 1


async def test_provisional_native_segments_never_become_events_and_capture_keeps_appending(
    memory: ControlledMemory,
    tmp_path: Path,
) -> None:

    entered = asyncio.Event()
    release = asyncio.Event()
    assessed: list[LiveEvent] = []
    recall = LiveRecall(memory)

    async def assess(event: LiveEvent, revision: int, trigger: Trigger) -> None:
        assessed.append(event)
        entered.set()
        await release.wait()

    updates = LiveUpdates(memory, recall, assess)
    segmenter = ScriptedSegmenter(
        [
            (EventSegment(1, "Activity is unfinished", complete=False),),
            (EventSegment(2, "Activity is complete"),),
            (),
            (),
        ]
    )
    stream = AsyncEventStream(
        memory,
        segmenter=segmenter,
        capture=True,
        context=ObservationContext(source_id=SOURCE_ID),
        checkpoint_path=tmp_path / "capture.json",
    )
    await stream.append(_chunk(0))
    assert [commit async for commit in stream.process()] == []
    assert updates.latest is None and not assessed
    capture = EventCapture(
        stream,
        asyncio.Event(),
        updates.observe,
    )
    await capture.push(_chunk(1))
    await asyncio.wait_for(entered.wait(), 3)
    await asyncio.wait_for(capture.push(_chunk(2)), 3)
    assert capture.stream.positions["default"] == _chunk(2).occurred_end
    assert len(assessed) == 1
    closing = asyncio.create_task(updates.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing
    await capture.finish()
    await capture.wait()
    await recall.close()


async def test_capture_pushes_failure_and_unfinished_eof_state_without_committing_a_memory(
    memory: ControlledMemory,
    tmp_path: Path,
) -> None:

    states: list[tuple[bool, bool]] = []
    commits: list[StreamCommit] = []
    segmenter = ScriptedSegmenter(
        [
            ModelError("private model response", reason="timeout"),
            (EventSegment(1, "Activity remains unfinished", complete=False),),
            (EventSegment(2, "Activity remains unfinished", complete=False),),
        ]
    )
    capture = EventCapture(
        AsyncEventStream(
            memory,
            segmenter=segmenter,
            capture=True,
            checkpoint_path=tmp_path / "failed-capture.json",
        ),
        asyncio.Event(),
        observe=commits.append,
        changed=lambda: states.append((capture.processing, capture.failure is not None)),
    )
    before = (await memory.list()).items
    await capture.push(_chunk(0))
    with pytest.raises(ModelError, match="private model response"):
        await asyncio.wait_for(capture.wait(), 3)
    assert states == [(True, False), (False, True)]
    assert capture.stream.positions["default"] == _chunk(0).occurred_end
    await capture.push(_chunk(1))
    await capture.finish()
    await asyncio.wait_for(capture.wait(), 3)
    assert states[-2:] == [(True, True), (False, False)]
    assert capture.closed and capture.failure is None
    assert not commits and (await memory.list()).items == before


async def test_capture_state_pushes_do_not_wait_for_semantic_analysis_to_accept_input(
    memory: ControlledMemory,
    tmp_path: Path,
) -> None:

    class BlockingSegmenter:
        event_model = "blocked-live-state-fixture"
        event_recipe = "blocked-live-state-fixture:v1"

        def __init__(self) -> None:
            self.entered = Event()
            self.release = Event()

        def segment_events(
            self,
            observations: Sequence[EventObservation],
            *,
            end_of_stream: bool = False,
        ) -> tuple[EventSegment, ...]:
            self.entered.set()
            assert self.release.wait(5), "test did not release semantic analysis"
            return ()

    states: list[bool] = []
    segmenter = BlockingSegmenter()
    capture = EventCapture(
        AsyncEventStream(
            memory,
            segmenter=segmenter,
            capture=True,
            checkpoint_path=tmp_path / "blocked-capture.json",
        ),
        asyncio.Event(),
        changed=lambda: states.append(capture.processing),
    )
    try:
        await capture.push(_chunk(0))
        assert await asyncio.to_thread(segmenter.entered.wait, 3)
        assert states == [True]
        await asyncio.wait_for(capture.push(_chunk(1)), 3)
        assert capture.stream.positions["default"] == _chunk(1).occurred_end
        assert not segmenter.release.is_set()
    finally:
        segmenter.release.set()
        await capture.finish()
        await asyncio.wait_for(capture.wait(), 3)
    assert states[-1] is False and capture.failure is None
