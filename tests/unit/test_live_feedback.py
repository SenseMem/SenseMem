"""Live feedback reviews originals without stalling input or inventing user facts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Self, cast

import pytest
from _feature_support import TinyEmbedder

from examples.live import feedback as module
from examples.live.feedback import MemoryFeedback
from examples.live.memories import memory_cards
from examples.live.recall import LiveRecall
from sensemem import (
    AbstentionReason,
    AnswerPolicy,
    AnswerResult,
    AsyncMemory,
    Blob,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemoryRecord,
    MemorySettings,
    Modality,
    ModelError,
    ModelInput,
    ObservationContext,
    RecordKind,
    SearchHit,
)

SOURCE = "live:" + "a" * 32
NOW = datetime(2026, 11, 17, tzinfo=timezone.utc)


class Reviewer:
    generation_model = "feedback-source-fixture:v1"
    generation_capabilities = frozenset(
        {Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}
    )

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.release.set()
        self.calls: list[tuple[str, tuple[bytes, ...]]] = []
        self.abstain = False
        self.fail = False

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        originals = []
        for asset in question.assets:
            assert asset.path is not None
            originals.append(asset.path.read_bytes())
        self.calls.append((question.text, tuple(originals)))
        self.started.set()
        assert self.release.wait(5), "source review did not drain"
        if self.fail:
            raise ModelError("private provider details must not leave the worker", reason="timeout")
        return AnswerResult(
            "无法从原片确认。" if self.abstain else "钥匙放在蓝色盒子里。",
            hits=tuple(hits),
            abstained=self.abstain,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE if self.abstain else None,
        )

    def close(self) -> None:
        self.release.set()


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[tuple[AsyncMemory, Reviewer]]:
    reviewer = Reviewer()
    async with AsyncMemory(
        Memory(
            tmp_path,
            embedder=TinyEmbedder(),
            answerer=reviewer,
            settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
        )
    ) as memory:
        try:
            yield memory, reviewer
        finally:
            reviewer.release.set()


async def event(
    memory: AsyncMemory,
    *records: MemoryRecord,
    query: str = "钥匙放在哪里",
    revision: int = 0,
    kind: str = "sensory_recall",
) -> dict[str, object]:

    return {
        "type": kind,
        "source_id": SOURCE,
        "memory_revision": revision,
        "status": "ready",
        "query": query,
        "hits": await memory_cards(memory, records),
    }


async def drain(feedback: MemoryFeedback) -> None:
    assert feedback.task is not None
    await asyncio.wait_for(asyncio.shield(feedback.task), timeout=5)


async def test_review_is_nonblocking_coalesces_pending_and_preserves_original_media(
    memory: tuple[AsyncMemory, Reviewer],
) -> None:

    store, reviewer = memory
    original = b"original video packets, including the microphone"
    first = await store.add(
        ("unverified event synopsis", Blob(original, "video/mp4")), occurred_at=NOW
    )
    middle = await store.add("obsolete pending text")
    latest = await store.add("latest pending text")
    recall = LiveRecall(store)
    published: list[dict[str, object]] = []

    async def publish(message: dict[str, object]) -> None:
        published.append(message)

    recall.sender = publish
    feedback = MemoryFeedback(store, recall.publish, SOURCE)
    recall.observer = feedback.observe
    reviewer.release.clear()
    try:
        # The original result is delivered before the source-review gate is released.
        await asyncio.wait_for(recall.publish(await event(store, first)), timeout=1)
        assert published[0]["type"] == "sensory_recall"
        assert await asyncio.to_thread(reviewer.started.wait, 2)
        feedback.observe(await event(store, middle, query="obsolete"))
        feedback.observe(await event(store, latest, query="latest"))
        assert not any(item["type"] == "memory_feedback" for item in published)
        reviewer.release.set()
        await drain(feedback)
        results = [item for item in published if item["type"] == "memory_feedback"]
        assert [item["query"] for item in results] == ["钥匙放在哪里", "latest"]
        assert len(reviewer.calls) == 1
        question, delivered = reviewer.calls[0]
        assert delivered == (original,)
        assert "unverified event synopsis" not in question
        assert "not the current scene" in question and "not the original asset" in question
        assert results[0]["facts"] == [
            {
                "memory_id": first.id,
                "text": "历史记录原片复核(模型): 钥匙放在蓝色盒子里。",
                "occurred_at": NOW.isoformat(),
                "verification": "source_review",
                "asset_ids": [first.assets[0].id],
            }
        ]
        assert await store.read_asset(first.id, first.assets[0].id) == Blob(original, "video/mp4")
    finally:
        reviewer.release.set()
        await feedback.close()
        await recall.close()


async def test_slow_original_review_gets_delivery_ttl_without_redating_the_source(
    memory: tuple[AsyncMemory, Reviewer], monkeypatch: pytest.MonkeyPatch
) -> None:

    store, reviewer = memory
    record = await store.add(
        ("unverified summary", Blob(b"original", "audio/wav")), occurred_at=NOW
    )
    published: list[dict[str, object]] = []
    clock = datetime.now(timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz: object = None) -> Self:
            return cls.fromisoformat(clock.isoformat())

    monkeypatch.setattr(module, "datetime", Clock)

    async def publish(message: dict[str, object]) -> None:
        published.append(message)

    feedback = module.MemoryFeedback(store, publish, SOURCE)
    reviewer.release.clear()
    try:
        feedback.observe(
            {**await event(store, record), "expires_at": (clock + timedelta(seconds=1)).isoformat()}
        )
        assert await asyncio.to_thread(reviewer.started.wait, 2)
        clock += timedelta(seconds=30)
        reviewer.release.set()
        await drain(feedback)
        assert len(published) == 1
        facts = cast(list[dict[str, object]], published[0]["facts"])
        assert facts[0]["occurred_at"] == NOW.isoformat()
        assert datetime.fromisoformat(str(published[0]["expires_at"])) > clock
    finally:
        reviewer.release.set()
        await feedback.close()


async def test_new_revision_drops_old_review_and_accepts_only_current_source(
    memory: tuple[AsyncMemory, Reviewer],
) -> None:

    store, reviewer = memory
    first = await store.add(("old summary", Blob(b"old original", "audio/wav")))
    latest = await store.add("new authoritative record")
    published: list[dict[str, object]] = []

    async def publish(message: dict[str, object]) -> None:
        published.append(message)

    feedback = MemoryFeedback(store, publish, SOURCE)
    reviewer.release.clear()
    try:
        feedback.observe(await event(store, first))
        assert await asyncio.to_thread(reviewer.started.wait, 2)
        feedback.invalidate(1)
        feedback.observe(await event(store, first))
        feedback.observe({**await event(store, latest, revision=1), "source_id": "another-call"})
        feedback.observe(await event(store, latest, query="new", revision=1))
        reviewer.release.set()
        await drain(feedback)
        assert len(published) == 1
        assert published[0]["memory_revision"] == 1
        assert published[0]["query"] == "new"
        assert published[0]["facts"] == [
            {
                "memory_id": latest.id,
                "text": "历史记录文本/未核实: new authoritative record",
                "occurred_at": None,
                "verification": "record_text",
                "asset_ids": [],
            }
        ]
    finally:
        reviewer.release.set()
        await feedback.close()


async def test_forgetting_during_review_withdraws_fact_without_host_revision_hook(
    memory: tuple[AsyncMemory, Reviewer],
) -> None:

    store, reviewer = memory
    record = await store.add(("unverified summary", Blob(b"historical video", "video/mp4")))
    published: list[dict[str, object]] = []

    async def publish(message: dict[str, object]) -> None:
        published.append(message)

    feedback = MemoryFeedback(store, publish, SOURCE)
    reviewer.release.clear()
    try:
        feedback.observe(await event(store, record))
        assert await asyncio.to_thread(reviewer.started.wait, 2)
        await store.forget((record.id,))
        reviewer.release.set()
        await drain(feedback)
        assert len(published) == 1
        assert published[0]["status"] == "ready"
        assert published[0]["facts"] == published[0]["hits"] == []
    finally:
        reviewer.release.set()
        await feedback.close()


async def test_pending_and_current_agent_outputs_cannot_feed_back_as_external_facts(
    memory: tuple[AsyncMemory, Reviewer],
) -> None:

    store, reviewer = memory
    pending = await store.capture("still awaiting native settlement")
    own_reply = await store.add(
        "Agent said the keys are in a box, not a user assertion",
        context=ObservationContext(source_id=SOURCE),
        metadata={"dialogue": {"origin": "realtime", "speaker": "assistant"}},
    )
    prior_reply = await store.add(
        "I suggested bringing a notebook.",
        context=ObservationContext(source_id="prior-call"),
        metadata={"dialogue": {"origin": "realtime", "speaker": "assistant"}},
    )
    published: list[dict[str, object]] = []

    async def publish(message: dict[str, object]) -> None:
        published.append(message)

    feedback = MemoryFeedback(store, publish, SOURCE)
    try:
        feedback.observe(await event(store, own_reply, pending))
        await drain(feedback)
        assert published[-1]["facts"] == []
        feedback.observe(await event(store, prior_reply, kind="context"))
        await drain(feedback)
        assert published[-1]["facts"] == [
            {
                "memory_id": prior_reply.id,
                "text": "Agent记录文本/未核实: I suggested bringing a notebook.",
                "occurred_at": None,
                "verification": "record_text",
                "asset_ids": [],
            }
        ]
        assert reviewer.calls == []
    finally:
        await feedback.close()


async def test_native_hidden_and_retired_state_overrides_stale_visible_hit_cards(
    memory: tuple[AsyncMemory, Reviewer],
) -> None:

    store, reviewer = memory
    source = await store.add("A single patient conversation.")
    operation = await store.apply(
        MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=(source.id,),
            proposal=FormationProposal(
                kind=RecordKind.TRAIT,
                content="The user is patient.",
                subject="user",
                predicate="personality",
                value="patient",
                confidence=0.9,
            ),
        )
    )
    hidden = await store.get(operation.created_ids[0])
    assert hidden.context is not None and not hidden.context.visible
    message = await event(store, hidden)
    await store.apply(MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(hidden.id,)))
    retired = await store.get(hidden.id)
    assert retired.context is not None and retired.context.retired_at is not None
    published: list[dict[str, object]] = []

    async def publish(value: dict[str, object]) -> None:
        published.append(value)

    feedback = MemoryFeedback(store, publish, SOURCE)
    try:
        feedback.observe(message)
        await drain(feedback)
        assert published[0]["facts"] == published[0]["hits"] == []
        assert reviewer.calls == []
    finally:
        await feedback.close()


async def test_feedback_bounds_text_and_records_without_promoting_excerpts_to_verified_facts(
    memory: tuple[AsyncMemory, Reviewer],
) -> None:

    store, reviewer = memory
    records = tuple(
        [await store.add(f"记录{index}: " + "未经核实的历史文字" * 100) for index in range(3)]
    )
    published: list[dict[str, object]] = []

    async def publish(message: dict[str, object]) -> None:
        published.append(message)

    feedback = MemoryFeedback(store, publish, SOURCE)
    try:
        feedback.observe(await event(store, *records))
        await drain(feedback)
        facts = cast(list[dict[str, object]], published[0]["facts"])
        assert [fact["memory_id"] for fact in facts] == [record.id for record in records[:2]]
        total = 0
        for fact in facts:
            text = fact["text"]
            assert isinstance(text, str)
            assert text.startswith("历史记录文本/未核实: ") and text.endswith("…")
            total += len(text)
            assert fact["verification"] == "record_text"
            assert fact["asset_ids"] == []
        assert total <= 160
        assert reviewer.calls == []
    finally:
        await feedback.close()


@pytest.mark.parametrize("outcome", ["empty", "abstained", "error"])
async def test_empty_abstained_and_failed_reviews_have_explicit_safe_outcomes(
    memory: tuple[AsyncMemory, Reviewer], outcome: str
) -> None:

    store, reviewer = memory
    record = await store.add(("unverified synopsis", Blob(b"original source", "video/mp4")))
    reviewer.abstain, reviewer.fail = outcome == "abstained", outcome == "error"
    published: list[dict[str, object]] = []

    async def publish(message: dict[str, object]) -> None:
        published.append(message)

    feedback = MemoryFeedback(store, publish, SOURCE)
    try:
        feedback.observe(await event(store, *(() if outcome == "empty" else (record,))))
        await drain(feedback)
        assert len(published) == 1
        result = published[0]
        assert result["facts"] == []
        assert result["status"] == ("error" if outcome == "error" else "ready")
        assert result["abstained"] == (outcome == "abstained")
        assert "private provider" not in repr(result)
        feedback.observe(result)
        assert feedback.task is not None and feedback.task.done()
    finally:
        await feedback.close()
