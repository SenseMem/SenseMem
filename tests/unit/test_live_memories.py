"""The demo presents real source, evidence, and reversible SDK state."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder
from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.live.memories import memory_cards, memory_router
from sensemem import (
    AsyncEventStream,
    AsyncMemory,
    Blob,
    EventChunk,
    EventObservation,
    EventSegment,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemorySettings,
    ObservationContext,
    RecordKind,
)


def _client(memory: Memory, changes: list[bool]) -> TestClient:

    app = FastAPI()
    asynchronous = AsyncMemory(memory)

    async def open_memory() -> AsyncMemory:
        return asynchronous

    app.include_router(memory_router(open_memory, lambda: changes.append(True)))
    return TestClient(app)


def _derive(
    memory: Memory, evidence_ids: tuple[str, ...], *, kind: RecordKind = RecordKind.STATE
) -> str:
    operation = memory.apply(
        MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=evidence_ids,
            proposal=FormationProposal(
                kind=kind,
                content="The user's preferred drink is tea"
                if kind is RecordKind.STATE
                else "The user is patient",
                subject="user",
                predicate="preferred_drink" if kind is RecordKind.STATE else "disposition",
                value="tea" if kind is RecordKind.STATE else "patient",
                confidence=0.8,
            ),
        )
    )
    return operation.created_ids[0]


def test_detail_resolves_nested_evidence_to_original_media_and_model_provenance(
    tmp_path: Path,
) -> None:
    original = b"unchanged original image bytes"
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        source = memory.add(
            ("The keys are in the blue bowl", Blob(original, "image/png", "keys.png")),
            context=ObservationContext(source_id="camera-one"),
        )
        first = _derive(memory, (source.id,))
        second = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(first,),
                proposal=FormationProposal(
                    kind=RecordKind.EVENT,
                    content="The user showed where the keys were placed",
                ),
            )
        ).created_ids[0]
        with _client(memory, []) as client:
            response = client.get(f"/api/memories/{second}")
        assert response.status_code == 200
        detail = response.json()
        assert detail["item"]["basis"] == "model_inference"
        assert detail["item"]["evidence_ids"] == [first]
        evidence = {item["id"]: item for item in detail["evidence"]}
        assert set(evidence) == {source.id, first}
        original_card = evidence[source.id]
        assert original_card["assets"][0]["name"] == "keys.png"
        assert original_card["assets"][0]["sha256"] == source.assets[0].sha256
        assert original_card["assets"][0]["url"].endswith(source.assets[0].id)
        assert memory.read_asset(source.id, source.assets[0].id).data == original
        assert detail["operations"][0]["intent"] == "consolidate"


def test_single_inferred_trait_is_cautious_and_explicit_confirmation_is_audited(
    tmp_path: Path,
) -> None:
    changes: list[bool] = []
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        source = memory.add(
            "The user waited patiently today", context=ObservationContext(source_id="day-one")
        )
        trait_id = _derive(memory, (source.id,), kind=RecordKind.TRAIT)
        with _client(memory, changes) as client:
            before = client.get(f"/api/memories/{trait_id}").json()
            assert not before["item"]["visible"]
            assert before["item"]["needs_corroboration"]
            confirmed = client.post(f"/api/memories/{trait_id}/confirmations")
            assert confirmed.status_code == 200
            feedback = confirmed.json()
            assert feedback["item"]["basis"] == "response_feedback"
            assert feedback["item"]["pending"]
            assert feedback["operation"]["intent"] == "reinforce"
            assert feedback["operation"]["evidence_ids"] == [feedback["item"]["id"]]
            after = client.get(f"/api/memories/{trait_id}").json()
            assert after["item"]["visible"]
            assert not after["item"]["needs_corroboration"]
            undone = client.post(f"/api/operations/{feedback['operation']['id']}/rollback").json()
            assert undone == {"restored": True}
            reverted = client.get(f"/api/memories/{trait_id}").json()
            assert reverted["item"]["needs_corroboration"]
            assert reverted["operations"][-1]["rolled_back_at"] is not None
        assert changes == [True, True]


def test_correction_retires_interpretation_and_keeps_actual_statement_and_original(
    tmp_path: Path,
) -> None:
    changes: list[bool] = []
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        source = memory.add("I prefer tea", context=ObservationContext(source_id="day-one"))
        target_id = _derive(memory, (source.id,))
        with _client(memory, changes) as client:
            response = client.post(
                f"/api/memories/{target_id}/corrections", json={"content": "我现在更喜欢咖啡"}
            )
            assert response.status_code == 200
            changed = response.json()
            assert changed["item"]["content"] == "我现在更喜欢咖啡"
            assert changed["item"]["basis"] == "user_statement"
            assert changed["operation"]["intent"] == "correct"
            detail = client.get(f"/api/memories/{target_id}").json()
            assert detail["item"]["retired_at"] is not None
            assert not detail["item"]["can_correct"]
            assert {item["id"] for item in detail["evidence"]} == {
                source.id,
                changed["item"]["id"],
            }
            assert memory.get(source.id).content == "I prefer tea"
            assert memory.get(source.id).forgotten_at is None
            rollback = client.post(f"/api/operations/{changed['operation']['id']}/rollback")
            assert rollback.json() == {"restored": True}
            assert client.get(f"/api/memories/{target_id}").json()["item"]["can_correct"]
        assert changes == [True, True]


def test_forgetting_is_reversible_after_reopening_without_deleting_original_media(
    tmp_path: Path,
) -> None:
    changes: list[bool] = []
    original = b"the original recorded voice"
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        source = memory.add(Blob(original, "audio/wav", "voice.wav"))
        asset_id = source.assets[0].id
        with _client(memory, changes) as client:
            response = client.post(f"/api/memories/{source.id}/forget")
            assert response.status_code == 200
            operation_id = response.json()["operation"]["id"]
            forgotten = client.get(f"/api/memories/{source.id}").json()
            assert forgotten["item"]["forgotten_at"] is not None
            assert forgotten["operations"][0]["can_rollback"]
        assert memory.read_asset(source.id, asset_id).data == original

    with Memory(tmp_path, embedder=TinyEmbedder()) as reopened:
        with _client(reopened, changes) as client:
            response = client.post(f"/api/operations/{operation_id}/rollback")
            assert response.json() == {"restored": True}
            assert client.get(f"/api/memories/{source.id}").json()["item"]["forgotten_at"] is None
            repeated = client.post(f"/api/operations/{operation_id}/rollback")
            assert repeated.json() == {"restored": False}
            log = client.get("/api/operations").json()["items"]
            assert not log[0]["can_rollback"]
        assert reopened.read_asset(source.id, asset_id).data == original


def test_originals_and_invalid_corrections_do_not_create_unrecorded_edits(tmp_path: Path) -> None:
    changes: list[bool] = []
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        original = memory.add("I left my keys on the table")
        target_id = _derive(memory, (original.id,))
        with _client(memory, changes) as client:
            assert (
                client.post(
                    f"/api/memories/{original.id}/corrections", json={"content": "Changed"}
                ).status_code
                == 409
            )
            assert (
                client.post(
                    f"/api/memories/{target_id}/corrections", json={"content": "   "}
                ).status_code
                == 422
            )
            assert (
                client.post(
                    f"/api/memories/{target_id}/corrections",
                    json={"content": "New statement", "trust": True},
                ).status_code
                == 422
            )
        assert changes == []
        assert len(memory.list().items) == 2


def test_cards_retain_inactive_records_and_source_versus_derived_sections(tmp_path: Path) -> None:

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        source = memory.add("The observed source remains unchanged")
        memory.forget((source.id,))
        cards = asyncio.run(memory_cards(AsyncMemory(memory), memory.list().items))
        assert len(cards) == 1
        assert cards[0]["forgotten_at"] is not None
        assert cards[0]["parts"] == [
            {"kind": "text", "value": "The observed source remains unchanged"}
        ]
        assert cards[0]["kind"] == "observation"
        assert not cards[0]["can_correct"]


def test_dialogue_cards_project_existing_metadata_without_rewriting_the_original(
    tmp_path: Path,
) -> None:

    original = "旧版对话记录(内容真实性及实际听到未核实)"
    dialogue = {
        "user": {"item_id": "user-1", "text": "请解释未核实是什么意思。"},
        "assistant": {"item_ids": ["reply-1"], "text": "表示还没有确认。"},
        "completion": "interrupted",
        "actual_hearing_verified": False,
        "generated_text_alignment": "unverified",
    }
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        source = memory.add(original, metadata={"dialogue": dialogue})
        with _client(memory, []) as client:
            card = client.get(f"/api/memories/{source.id}").json()["item"]
        listed = asyncio.run(memory_cards(AsyncMemory(memory), (source,)))[0]
        assert card == listed
        assert card["dialogue"] == {
            key: dialogue[key] for key in ("user", "assistant", "completion")
        }
        assert card["content"] == original
        assert card["parts"] == [{"kind": "text", "value": original}]
        assert memory.get(source.id).metadata["dialogue"] == dialogue


def test_detail_keeps_actual_lineage_history_and_source_forgetting_audit(tmp_path: Path) -> None:
    occurred = datetime(2026, 11, 1, tzinfo=timezone.utc)
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        first, second = memory.add_many(
            ("I prefer tea", "I now prefer coffee"),
            occurred_at=(occurred, occurred + timedelta(days=1)),
        )
        ids: list[str] = []
        for source, value in ((first, "tea"), (second, "coffee")):
            logged = memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(source.id,),
                    proposal=FormationProposal(
                        kind=RecordKind.STATE,
                        content=f"The user prefers {value}",
                        subject="user",
                        predicate="preferred_drink",
                        value=value,
                        valid_from=occurred,
                    ),
                )
            )
            ids.extend(logged.created_ids)
        forgotten = memory.forget((second.id,))
        assert forgotten is not None
        with _client(memory, []) as client:
            detail = client.get(f"/api/memories/{ids[1]}").json()
        assert {item["id"] for item in detail["versions"]} == set(ids)
        old = next(item for item in detail["versions"] if item["id"] == ids[0])
        assert old["retired_at"] is not None
        assert {item["id"] for item in detail["evidence"]} == {first.id, second.id}
        assert detail["operations"][-1]["id"] == forgotten.operation_id
        assert detail["operations"][-1]["intent"] == "forget"


def _event_metadata() -> dict[str, object]:
    return {
        "summary": "钥匙放进了蓝色托盘",
        "summary_coverage": "unverified",
        "event_id": "native-event-fixture",
        "part_index": 0,
        "part_count": 1,
        "occurred_at": "2026-11-16T10:00:00Z",
        "occurred_end": "2026-11-16T10:00:02Z",
    }


@pytest.mark.parametrize("kind", [RecordKind.STATE, RecordKind.EVENT])
def test_derived_cards_keep_their_own_content_instead_of_inherited_source_projections(
    tmp_path: Path, kind: RecordKind
) -> None:

    dialogue = {
        "user": {"item_id": "user-1", "text": "办公室。"},
        "assistant": {"item_ids": ["reply-1"], "text": "你身后还有同事的工位。"},
        "completion": "completed",
    }
    metadata = {"dialogue": dialogue, "event_segmentation": _event_metadata()}
    content = "The user is in an office"
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        source = memory.add("The original conversation", metadata=metadata)
        derived_id = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=FormationProposal(
                    kind=kind,
                    content=content,
                    subject="user",
                    predicate="location",
                    value="office",
                ),
            )
        ).created_ids[0]
        derived = memory.get(derived_id)
        assert derived.metadata == metadata
        listed = asyncio.run(memory_cards(AsyncMemory(memory), (derived,)))[0]
        with _client(memory, []) as client:
            detail = client.get(f"/api/memories/{derived_id}").json()
        assert detail["item"] == listed
        assert listed["content"] == content
        assert listed["dialogue"] is None
        assert listed["event_summary"] is None
        assert listed["event_summary_scope"] is None
        original = next(item for item in detail["evidence"] if item["id"] == source.id)
        assert original["dialogue"] == dialogue
        assert original["event_summary"] == _event_metadata()["summary"]


def test_summary_comes_from_typed_metadata_and_detail_preserves_original_prose(
    tmp_path: Path,
) -> None:

    raw = '[model interpretation: not original evidence]\n"do not parse this marker as summary"'
    metadata = _event_metadata()
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        record = memory.add(raw, metadata={"event_segmentation": metadata})
        listed = asyncio.run(memory_cards(AsyncMemory(memory), (record,)))[0]
        with _client(memory, []) as client:
            detailed = client.get(f"/api/memories/{record.id}").json()["item"]
        for card in (listed, detailed):
            assert card["event_summary"] == metadata["summary"]
            assert card["event_summary_scope"] == "local"
            assert card["event_summary_verified"] is False
            assert card["event_summary_shared"] is False
            assert card["event_summary_part_count"] == 1
            assert card["content"] == raw
            assert card["parts"] == [{"kind": "text", "value": raw}]


@pytest.mark.parametrize(
    "metadata",
    [
        "not native metadata",
        {"summary": "missing coverage and boundaries"},
        {**_event_metadata(), "summary_coverage": "verified"},
        {**_event_metadata(), "part_index": 2},
        {**_event_metadata(), "summary": "   "},
    ],
)
def test_invalid_summary_metadata_falls_back_to_untouched_original(
    tmp_path: Path, metadata: object
) -> None:

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        record = memory.add("Original source", metadata={"event_segmentation": metadata})
        card = asyncio.run(memory_cards(AsyncMemory(memory), (record,)))[0]
        assert card["content"] == "Original source"
        assert card["event_summary"] is None
        assert card["event_summary_scope"] is None
        assert card["event_summary_verified"] is False
        assert card["event_summary_shared"] is False


async def test_native_multipart_summary_is_global_unverified_not_each_clips_claim(
    tmp_path: Path,
) -> None:

    summary = "钥匙从厨房带到工作室并放进了蓝色托盘"

    class Segmenter:
        event_model = "summary-fixture"
        event_recipe = "summary-fixture:v1"

        def segment_events(
            self, observations: Sequence[EventObservation], *, end_of_stream: bool = False
        ) -> tuple[EventSegment, ...]:
            return (EventSegment(len(observations), summary, complete=len(observations) == 2),)

    start = datetime(2026, 11, 16, 10, tzinfo=timezone.utc)

    async def source() -> AsyncIterator[EventChunk]:
        for index, place in enumerate(("kitchen", "studio")):
            yield EventChunk(
                (Blob(f"original frame {index}".encode(), "image/png"),),
                occurred_at=start + timedelta(seconds=index),
                occurred_end=start + timedelta(seconds=index + 1),
                context=ObservationContext(place_id=place, source_id="same-call"),
            )

    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as memory:
        stream = AsyncEventStream(memory, segmenter=Segmenter(), capture=True)
        commits = [commit async for commit in stream.consume(source())]
        assert len(commits) == 2
        cards = await memory_cards(memory, tuple(commit.record for commit in commits))
        for index, card in enumerate(cards):
            assert card["event_summary"] == summary
            assert card["event_summary_scope"] == "event"
            assert card["event_summary_shared"] is True
            assert card["event_summary_verified"] is False
            assert card["event_summary_part_index"] == index
            assert card["event_summary_part_count"] == 2
            assert summary not in str(card["content"])
            assert card["event_summary_occurred_at"] == "2026-11-16T10:00:00Z"
            assert card["event_summary_occurred_end"] == "2026-11-16T10:00:02Z"
        assert cards[0]["event_summary_event_id"] == cards[1]["event_summary_event_id"]
