"""Speculative live recall stays attributable, read-only, and exact at finality."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder
from fastapi import WebSocketDisconnect

from examples.live.recall import LiveRecall, RecallInput
from sensemem import (
    AsyncMemory,
    AsyncOmniPrefetch,
    Blob,
    ContentInput,
    ContextBudget,
    ContextBundle,
    ContextSupplement,
    Memory,
    MemorySettings,
    ModelError,
    PrefetchResult,
    RetrievalScope,
    ValidationError,
)

PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZ1kAAAAASUVORK5CYII="
)


class ControlledMemory(AsyncMemory):
    def __init__(self, memory: Memory) -> None:
        super().__init__(memory)
        self.calls: list[ContentInput] = []
        self.started: asyncio.Queue[ContentInput] = asyncio.Queue()
        self.completed: asyncio.Queue[ContextBundle] = asyncio.Queue()
        self.release = asyncio.Event()
        self.block = False
        self.fail: str | None = None

    async def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        allow_partial_sources: bool = False,
        supplements: Sequence[ContextSupplement] = (),
    ) -> ContextBundle:
        self.calls.append(goal)
        self.started.put_nowait(goal)
        if self.block and len(self.calls) == 1:
            await self.release.wait()
        if goal == self.fail:
            raise ModelError("private provider failure", reason="timeout")
        result = await super().compile(
            goal,
            budget=budget,
            reference_at=reference_at,
            scope=scope,
            allow_partial_sources=allow_partial_sources,
            supplements=supplements,
        )
        self.completed.put_nowait(result)
        return result


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[ControlledMemory]:
    value = ControlledMemory(
        Memory(
            tmp_path,
            embedder=TinyEmbedder(),
            settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
        )
    )
    async with value:
        await value.add("The spare key is in the blue toolbox.")
        yield value


async def test_native_compiled_prefetch_coalesces_snapshots_without_storing_input(
    memory: ControlledMemory,
) -> None:
    memory.block = True
    before = (await memory.list()).items
    helper = AsyncOmniPrefetch(memory, budget=ContextBudget(max_items=4))
    updates = helper.updates()
    arriving = asyncio.ensure_future(anext(updates))
    image = Blob(base64.b64decode(PNG.partition(",")[2]), "image/png")
    first: list[str | Blob] = ["old", image]
    assert helper.submit(first) == 1
    first[0] = "mutated"
    await memory.started.get()
    assert helper.submit("discarded before dispatch") == 2
    revision = helper.submit("Where is the spare key?")
    memory.release.set()
    update = await asyncio.wait_for(arriving, 3)
    assert update.revision == revision == 3
    assert update.result.bundle is not None
    assert update.result.bundle.goal == "Where is the spare key?"
    assert update.result.hits == update.result.bundle.hits
    assert memory.calls == [("old", image), "Where is the spare key?"]
    assert await helper.finalize("Where is the spare key?") is update.result
    with pytest.raises(StopAsyncIteration):
        await anext(updates)
    assert (await memory.list()).items == before


async def test_failed_exact_final_never_delivers_an_older_success(memory: ControlledMemory) -> None:
    helper = AsyncOmniPrefetch(memory, budget=ContextBudget())
    updates = helper.updates()
    helper.submit("Where is the spare key?")
    old = await anext(updates)
    assert old.revision == 1
    memory.fail = "final authoritative input"
    with pytest.raises(ModelError, match="private provider failure"):
        await helper.finalize(memory.fail)
    with pytest.raises(StopAsyncIteration):
        await anext(updates)
    assert helper.latest is old.result


async def _present(result: PrefetchResult) -> dict[str, object]:
    assert result.bundle is not None
    return {"context": result.bundle.render(), "hits": [hit.id for hit in result.hits]}


async def test_live_results_arrive_during_partial_speech_and_reuse_exact_final(
    memory: ControlledMemory,
) -> None:

    recall = LiveRecall(memory)
    messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall.sender = messages.put
    recall.presenter = _present
    before = (await memory.list()).items
    await recall.update(RecallInput(type="update", turn_id="audio_1", text="Where is"))
    assert (await messages.get())["type"] == "pending"
    partial = await asyncio.wait_for(messages.get(), 3)
    assert partial["type"] == "result" and partial["final"] is False
    assert partial["input_final"] is False
    await recall.update(
        RecallInput(type="final", turn_id="audio_1", text="Where is the spare key?")
    )
    assert (await messages.get())["type"] == "pending"
    final_input = await asyncio.wait_for(messages.get(), 3)
    assert final_input["revision"] == 2 and final_input["input_final"] is True
    while not final_input["final"]:
        final_input = await asyncio.wait_for(messages.get(), 3)
    result = await recall.finalize("audio_1", "rewritten query must not replace the transcript")
    assert final_input["final"] is True and final_input["revision"] == 2
    assert messages.empty()
    assert result is not None and result.bundle is not None
    assert result.bundle.goal == "Where is the spare key?"
    assert memory.calls == ["Where is", "Where is the spare key?"]
    assert (await memory.list()).items == before
    await recall.close()


async def test_authoritative_final_background_failure_is_safe_without_a_tool_call(
    memory: ControlledMemory,
) -> None:

    memory.fail = "exact final"
    recall = LiveRecall(memory)
    messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall.sender = messages.put
    recall.presenter = _present
    await recall.update(RecallInput(type="final", turn_id="final", text="exact final"))
    assert (await messages.get())["type"] == "pending"
    assert await asyncio.wait_for(messages.get(), 3) == {
        "type": "error",
        "turn_id": "final",
        "code": "recall_failed",
        "final": True,
    }
    assert memory.calls == ["exact final"]
    await recall.close()


async def test_corrected_final_replaces_background_retrieval_without_waiting_for_old_work(
    memory: ControlledMemory,
) -> None:

    memory.block = True
    recall = LiveRecall(memory)
    messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall.sender = messages.put
    recall.presenter = _present
    await recall.update(RecallInput(type="final", turn_id="same", text="misheard final"))
    await memory.started.get()
    await recall.update(RecallInput(type="final", turn_id="same", text="correct final"))
    assert (await messages.get())["type"] == "pending"
    assert (await messages.get())["type"] == "pending"
    completed = await asyncio.wait_for(messages.get(), 3)
    assert completed["type"] == "result" and completed["final"]
    assert completed["query"] == "correct final"
    memory.release.set()
    await recall.close()
    assert messages.empty()


async def test_authoritative_http_transcript_supersedes_a_late_socket_hypothesis(
    memory: ControlledMemory,
) -> None:

    memory.block = True
    recall = LiveRecall(memory)
    await recall.update(RecallInput(type="update", turn_id="audio_1", text="blue box"))
    await memory.started.get()
    finalizing = asyncio.create_task(
        recall.finalize("audio_1", "model rewrite", transcript="Where is the spare key?")
    )
    await asyncio.sleep(0)
    memory.release.set()
    result = await finalizing
    assert result is not None and result.bundle is not None
    assert result.bundle.goal == "Where is the spare key?"
    assert memory.calls == ["blue box", "Where is the spare key?"]
    await recall.close()


async def test_superseded_turn_cannot_replace_the_current_turns_visible_evidence(
    memory: ControlledMemory,
) -> None:

    memory.block = True
    recall = LiveRecall(memory)
    messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall.sender = messages.put
    recall.presenter = _present
    await recall.update(RecallInput(type="update", turn_id="old", text="old input"))
    await memory.started.get()
    await recall.update(RecallInput(type="final", turn_id="new", text="new final input"))
    memory.release.set()
    result = await recall.finalize("new", "ignored rewrite")
    await recall.close()
    delivered = []
    while not messages.empty():
        delivered.append(messages.get_nowait())
    results = [item for item in delivered if item["type"] == "result"]
    assert results and all(item["turn_id"] == "new" for item in results)
    assert results[-1]["final"] is True
    assert result is not None and result.bundle is not None
    assert result.bundle.goal == "new final input"


async def test_visual_snapshot_keeps_original_media_through_authoritative_final(
    memory: ControlledMemory,
) -> None:

    recall = LiveRecall(memory)
    supplied = RecallInput(type="update", turn_id="visual", text="this object", image=PNG)
    await recall.update(supplied)
    result = await recall.finalize("visual", "ignored", transcript="Have you seen this object?")
    assert result is not None
    final_query = memory.calls[-1]
    assert isinstance(final_query, tuple)
    assert final_query[0] == "Have you seen this object?"
    expected = supplied.snapshot()
    assert isinstance(expected, tuple)
    assert final_query[1] == expected[1]
    assert (await memory.list()).items[0].content == "The spare key is in the blue toolbox."
    await recall.close()


async def test_empty_authoritative_transcript_retracts_partial_evidence(
    memory: ControlledMemory,
) -> None:

    recall = LiveRecall(memory)
    messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall.sender = messages.put
    await recall.update(RecallInput(type="update", turn_id="empty", text="unstable partial"))
    await recall.update(RecallInput(type="final", turn_id="empty", text=""))
    assert await recall.finalize("empty", "fallback") is None
    assert (await messages.get())["type"] == "pending"
    assert await messages.get() == {"type": "cleared", "turn_id": "empty"}
    await recall.close()


async def test_socket_disconnect_during_result_delivery_does_not_leak_a_failed_watcher(
    memory: ControlledMemory,
) -> None:

    disconnected = asyncio.Event()

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "result":
            disconnected.set()
            raise WebSocketDisconnect(code=1000)

    recall = LiveRecall(memory)
    recall.sender = send
    recall.presenter = _present
    await recall.update(RecallInput(type="update", turn_id="closed", text="Where is the key?"))
    await disconnected.wait()
    await recall.close()
    assert recall.sender is None and not recall.turns


async def test_memory_change_invalidates_cached_final_and_refreshes_the_same_visual_turn(
    memory: ControlledMemory,
) -> None:

    recall = LiveRecall(memory)
    messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall.sender = messages.put
    recall.presenter = _present
    supplied = RecallInput(type="final", turn_id="same_view", text="Where is the key?", image=PNG)
    await recall.update(supplied)
    cached = await recall.finalize("same_view", "Where is the key?")
    assert cached is not None and cached.hits
    await memory.forget((cached.hits[0].id,))
    sender = recall.sender
    recall.invalidate()
    assert await recall.finalize("same_view", "Where is the key?") is None
    assert recall.sender is sender and not recall.closed
    await recall.update(supplied)
    refreshed = await recall.finalize("same_view", "Where is the key?")
    assert refreshed is not None and not refreshed.hits
    assert len(memory.calls) == 2
    await recall.close()
    assert not recall.retiring
    final_revisions = []
    while not messages.empty():
        message = messages.get_nowait()
        if message.get("final"):
            final_revisions.append(message["revision"])
    assert final_revisions == [1, 2]


async def test_memory_settlement_refreshes_exact_final_context_without_an_http_tool(
    memory: ControlledMemory,
) -> None:

    recall = LiveRecall(memory)
    messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall.sender = messages.put
    recall.presenter = _present
    await recall.update(RecallInput(type="final", turn_id="settled", text="Where are the keys?"))
    assert (await messages.get())["type"] == "pending"
    first = await asyncio.wait_for(messages.get(), 3)
    while not first.get("final"):
        first = await asyncio.wait_for(messages.get(), 3)
    assert first["hits"]
    await memory.forget(tuple(item.id for item in (await memory.list()).items))
    recall.invalidate(refresh=True)
    assert recall.latest_text == "Where are the keys?"
    refreshed = await asyncio.wait_for(messages.get(), 3)
    assert refreshed["final"] and refreshed["query"] == "Where are the keys?"
    assert refreshed["revision"] == 2 and refreshed["hits"] == []
    assert memory.calls == ["Where are the keys?", "Where are the keys?"]
    await recall.close()


async def test_repeated_memory_changes_coalesce_refresh_and_new_speech_takes_priority(
    memory: ControlledMemory,
) -> None:

    memory.block = True
    recall = LiveRecall(memory)
    messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall.sender = messages.put
    recall.presenter = _present
    await recall.update(RecallInput(type="final", turn_id="old", text="old final"))
    await memory.started.get()
    assert (await messages.get())["type"] == "pending"
    recall.invalidate(refresh=True)
    worker = recall.refreshing
    for _ in range(4):
        recall.invalidate(refresh=True)
    assert recall.refreshing is worker and recall.latest_text == "old final"
    assert len(recall.retiring) == 1 and memory.calls == ["old final"]
    await recall.update(RecallInput(type="final", turn_id="new", text="new user speech"))
    assert (await messages.get())["type"] == "pending"
    completed = await asyncio.wait_for(messages.get(), 3)
    assert completed["query"] == "new user speech" and completed["final"]
    memory.release.set()
    await recall.close()
    assert messages.empty()
    assert memory.calls == ["old final", "new user speech"]


async def test_invalidation_drains_old_work_without_delivering_it_to_a_reused_turn_id(
    memory: ControlledMemory,
) -> None:

    memory.block = True
    recall = LiveRecall(memory)
    messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    recall.sender = messages.put
    recall.presenter = _present
    await recall.update(RecallInput(type="update", turn_id="same_view", text="old input"))
    await memory.started.get()
    recall.invalidate()
    assert recall.retiring and not memory.release.is_set()
    await recall.update(RecallInput(type="update", turn_id="same_view", text="new input"))
    pending = await messages.get()
    assert pending["query"] == "old input" and pending["type"] == "pending"
    assert (await messages.get())["query"] == "new input"
    current = await asyncio.wait_for(messages.get(), 3)
    assert current["query"] == "new input" and current["type"] == "result"
    memory.release.set()
    await recall.close()
    assert messages.empty()


@pytest.mark.parametrize("failure", [False, True])
async def test_invalidation_during_finalization_requests_a_fresh_compile(
    memory: ControlledMemory, failure: bool
) -> None:

    memory.block = True
    if failure:
        memory.fail = "Where is the key?"
    recall = LiveRecall(memory)
    await recall.update(RecallInput(type="final", turn_id="final", text="Where is the key?"))
    await memory.started.get()
    finalizing = asyncio.create_task(recall.finalize("final", "Where is the key?"))
    await asyncio.sleep(0)
    recall.invalidate()
    memory.release.set()
    assert await finalizing is None
    await recall.close()


def test_recall_rejects_ambiguous_or_invalid_encoded_images() -> None:

    for image in ("https://remote.example/frame.png", "data:image/png;base64,!invalid", PNG + "!"):
        with pytest.raises(ValidationError):
            RecallInput(type="update", turn_id="invalid", text="scene", image=image).snapshot()
