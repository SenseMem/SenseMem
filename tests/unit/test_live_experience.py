"""The local app wires speculative recall, original media, and native maintenance."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import wave
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from threading import Event

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder
from fastapi.testclient import TestClient
from fixtures.live import make_recording as _recording
from PIL import Image

from examples.live import app as demo
from examples.live.app import LiveRuntime, create_app
from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AsyncMemory,
    Blob,
    ConsolidationReport,
    ContentInput,
    ContextBudget,
    ContextBundle,
    ContextSupplement,
    EventChunk,
    EventObservation,
    EventSegment,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryOperation,
    MemoryRecord,
    MemorySettings,
    MemorySpec,
    MemoryTrigger,
    Modality,
    ModelError,
    ModelInput,
    QueryFailure,
    RecordKind,
    RetrievalScope,
    SearchHit,
)


class Segmenter:
    event_model = "experience-fixture"
    event_recipe = "experience-fixture:v1"

    def segment_events(
        self, observations: Sequence[EventObservation], *, end_of_stream: bool = False
    ) -> tuple[EventSegment, ...]:
        return tuple(
            EventSegment(index + 1, "A complete observed activity")
            for index in range(len(observations))
        )


class ObservedMemory(AsyncMemory):
    def __init__(self, memory: Memory) -> None:
        super().__init__(memory)
        self.queries: list[ContentInput] = []

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
        self.queries.append(goal)
        return await super().compile(
            goal,
            budget=budget,
            reference_at=reference_at,
            scope=scope,
            allow_partial_sources=allow_partial_sources,
            supplements=supplements,
        )


class SourceReviewer:
    generation_capabilities = ATOMIC_MODALITIES

    def __init__(self) -> None:
        self.questions: list[str] = []
        self.originals: list[bytes] = []

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        self.questions.append(question.text)
        selected = tuple(hit for hit in hits if hit.assets)
        for hit in selected:
            for asset in hit.assets:
                assert asset.path is not None
                self.originals.append(asset.path.read_bytes())
        return AnswerResult("The original frame shows keys in the blue bowl.", hits=selected)

    def close(self) -> None:
        pass


def _config(path: Path, *, generation: bool = False, lifecycle: bool = False) -> MemorySpec:
    return MemorySpec.model_validate(
        {
            "data_dir": path,
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai", "model": "fixture"} if generation else None,
            "formation": {"provider": "openai", "model": "fixture"} if lifecycle else None,
            "consolidation": {"provider": "openai", "model": "fixture"} if lifecycle else None,
        }
    )


def test_camera_reviews_save_originals_and_become_searchable_while_av_event_is_unfinished(  # noqa: C901 - one camera-to-native-memory round trip
    tmp_path: Path,
) -> None:

    class VisualBackend(Segmenter):
        generation_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

        def __init__(self) -> None:
            self.event_started = Event()
            self.event_finality: list[bool] = []

        def segment_events(
            self, observations: Sequence[EventObservation], *, end_of_stream: bool = False
        ) -> tuple[EventSegment, ...]:
            self.event_finality.append(end_of_stream)
            self.event_started.set()
            return (EventSegment(len(observations), "The activity is continuing.", complete=False),)

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: AnswerPolicy = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            assert answer_policy == "best_effort" and not hits
            sequences = tuple(int(asset.id.partition(":")[2]) for asset in question.assets)
            return AnswerResult(
                json.dumps(
                    {
                        "summary": "The key is on the tray."
                        if sequences[-1] == 1
                        else "The key is inside the green box.",
                        "changed": sequences[-1] == 2,
                        "action": "context",
                        "utterance": None,
                        "confidence": 0.95,
                        "evidence": [
                            {"sequence": sequence, "observation": "The key is visible."}
                            for sequence in sequences
                        ],
                        "reason": "Update the visible key location.",
                        "contradicted_claim": None,
                        "correction": None,
                    }
                )
            )

        def close(self) -> None:
            pass

    backend = VisualBackend()
    opened: list[AsyncMemory] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, VisualBackend]:
        memory = AsyncMemory(
            Memory(
                config.data_dir,
                embedder=TinyEmbedder(),
                settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
            )
        )
        opened.append(memory)
        return memory, backend

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        source_id = client.post("/api/session").json()["source_id"]
        start = datetime.now(timezone.utc) - timedelta(seconds=6)
        with (
            client.websocket_connect("/api/media") as media,
            client.websocket_connect("/api/recall") as recall,
            client.websocket_connect("/api/perception") as vision,
        ):
            media.send_json({"type": "start", "started_at": start.isoformat()})
            media.send_bytes(_recording(frame_count=48))
            assert backend.event_started.wait(5), "the original A/V stream must be active"
            vision.send_json({"type": "start", "started_at": start.isoformat()})
            assert vision.receive_json()["type"] == "ready"
            originals: list[bytes] = []
            committed_ids: list[str] = []
            assert client.portal is not None
            for sequence in (1, 2, 3):
                image = BytesIO()
                Image.new("RGB", (16, 16), (sequence * 40, 60, 160)).save(image, "JPEG")
                originals.append(image.getvalue())
                vision.send_json(
                    {
                        "type": "frame",
                        "sequence": sequence,
                        "timestamp_ms": sequence * 100,
                        "captured_at": (start + timedelta(milliseconds=sequence * 100)).isoformat(),
                        "image": "data:image/jpeg;base64,"
                        + base64.b64encode(originals[-1]).decode(),
                    }
                )
                assert vision.receive_json() == {"type": "ack", "sequence": sequence}
                observed = recall.receive_json()
                while observed["type"] != "perception":
                    observed = recall.receive_json()
                assert observed["frame_sequences"][-1] == sequence
                if sequence == 3:
                    break  # An unchanged scene is still reviewed, without a duplicate memory.
                saved = recall.receive_json()
                while saved["type"] != "visual_memory":
                    saved = recall.receive_json()
                assert saved["status"] == "saved"
                assert saved["perception_id"] == observed["perception_id"]
                assert saved["source_id"] == source_id
                committed_ids.extend(saved["memory_ids"])
                deadline = time.monotonic() + 5
                while True:
                    state = client.get("/api/memories").json()
                    if not state["processing"] and all(
                        item["awaiting"] is None for item in state["items"]
                    ):
                        break
                    assert time.monotonic() < deadline, "visual records must settle before A/V EOF"
                    time.sleep(0.01)
                record = client.portal.call(opened[0].get, saved["memory_ids"][0])
                assert record.context is not None and record.context.source_id == source_id
                assert record.occurred_at == start + timedelta(milliseconds=100)
                assert record.occurred_end == (
                    None if sequence == 1 else start + timedelta(milliseconds=sequence * 100)
                )
                assert [part.kind for part in record.parts] == [
                    *["asset"] * sequence,
                    "model_interpretation",
                ]
                assert record.parts[-1].value.endswith(observed["summary"])
                metadata = record.metadata["visual_observation"]
                assert isinstance(metadata, dict)
                assert metadata["perception_id"] == observed["perception_id"]
                assert metadata["model_id"] == backend.event_model
                assert metadata["frame_sequences"] == list(range(1, sequence + 1))
                for asset, expected in zip(record.assets, originals, strict=True):
                    downloaded = client.get(f"/api/memories/{record.id}/assets/{asset.id}")
                    assert downloaded.status_code == 200 and downloaded.content == expected
                recalled = client.post("/api/search", json={"query": "Where is the key?"}).json()
                assert record.id in {hit["id"] for hit in recalled["hits"]}
                assert backend.event_finality and not any(backend.event_finality)
            assert len(client.get("/api/memories").json()["items"]) == len(committed_ids) == 2


def test_partial_recall_precedes_complete_input_and_final_search_keeps_authoritative_words(
    tmp_path: Path,
) -> None:

    opened: list[ObservedMemory] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        local = Memory(
            config.data_dir,
            embedder=TinyEmbedder(),
            settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
        )
        local.add("The spare keys are in the blue bowl by the door.")
        memory = ObservedMemory(local)
        opened.append(memory)
        return memory, Segmenter()

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        assert client.post("/api/session").status_code == 200
        with client.websocket_connect("/api/recall") as socket:
            socket.send_json({"type": "update", "turn_id": "utterance-one", "text": "上次钥匙"})
            pending = socket.receive_json()
            assert pending["type"] == "pending"
            assert pending["final"] is False
            preview = socket.receive_json()
            assert preview["type"] == "result"
            assert preview["input_final"] is False
            assert preview["query"] == "上次钥匙"
            assert len(client.get("/api/memories").json()["items"]) == 1

            final_words = "上次我给你看过的备用钥匙放在哪里"
            socket.send_json({"type": "final", "turn_id": "utterance-one", "text": final_words})
            assert socket.receive_json()["type"] == "pending"
            result = client.post(
                "/api/search",
                json={
                    "query": "a model-rewritten and incomplete key query",
                    "transcript": final_words,
                    "turn_id": "utterance-one",
                },
            )
            assert result.status_code == 200
            assert result.json()["query"] == final_words
            assert "media_review" not in result.json()
            delivered = socket.receive_json()
            while not delivered["final"]:
                delivered = socket.receive_json()
            assert delivered["query"] == final_words
            assert delivered["input_final"]
        assert opened[0].queries == ["上次钥匙", final_words]
        assert len(client.get("/api/memories").json()["items"]) == 1
        assert client.delete("/api/session").status_code == 204


@pytest.mark.parametrize(("generation", "with_media"), [(True, True), (False, True), (True, False)])
def test_media_review_uses_native_answer_and_returns_its_actual_hits(
    tmp_path: Path, generation: bool, with_media: bool
) -> None:

    reviewer = SourceReviewer()
    source_ids: list[str] = []
    original = b"unchanged camera frame fixture"

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        local = Memory(
            config.data_dir,
            embedder=TinyEmbedder(),
            answerer=reviewer,
            settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
        )
        source = local.add(
            ("The spare keys are in the blue bowl.", Blob(original, "image/png"))
            if with_media
            else "The spare keys are in the blue bowl."
        )
        source_ids.append(source.id)
        local.add("The user also has an umbrella by the door.")
        return AsyncMemory(local), Segmenter()

    with TestClient(
        create_app(_config(tmp_path, generation=generation), memory_factory=factory)
    ) as client:
        result = client.post("/api/search", json={"query": "Where are the spare keys?"})
        assert result.status_code == 200
        body = result.json()
        if generation and with_media:
            assert body["media_review"]["answer"] == (
                "The original frame shows keys in the blue bowl."
            )
            assert body["media_review"]["source_ids"] == source_ids
            assert [hit["id"] for hit in body["hits"]] == source_ids
            assert len(reviewer.questions) == 1
            assert reviewer.questions[0].partition("\n\n")[0] == "Where are the spare keys?"
            assert reviewer.originals == [original]
        else:
            assert "media_review" not in body
            assert reviewer.questions == []
            assert reviewer.originals == []


def test_forgetting_invalidates_cached_final_recall_and_restore_searches_fresh(
    tmp_path: Path,
) -> None:

    source_ids: list[str] = []
    opened: list[ObservedMemory] = []
    query = "备用钥匙放在哪里"

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        local = Memory(
            config.data_dir,
            embedder=TinyEmbedder(),
            settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
        )
        source_ids.append(local.add("The spare keys are in the blue bowl.").id)
        memory = ObservedMemory(local)
        opened.append(memory)
        return memory, Segmenter()

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        assert client.post("/api/session").status_code == 200
        with client.websocket_connect("/api/recall") as socket:
            socket.send_json({"type": "final", "turn_id": "cached-turn", "text": query})
            assert socket.receive_json()["type"] == "pending"
            preview = socket.receive_json()
            while not preview.get("final"):
                preview = socket.receive_json()
            assert preview["type"] == "result"
            assert [hit["id"] for hit in preview["hits"]] == source_ids
            supplied = {
                "query": "rewritten tool query",
                "turn_id": "cached-turn",
                "transcript": query,
            }
            cached = client.post("/api/search", json=supplied)
            assert cached.status_code == 200
            assert [hit["id"] for hit in cached.json()["hits"]] == source_ids
            assert opened[0].queries == [query]

            forgotten = client.post(f"/api/memories/{source_ids[0]}/forget")
            assert forgotten.status_code == 200
            operation_id = forgotten.json()["operation"]["id"]
            current = client.post("/api/search", json=supplied)
            assert current.status_code == 200
            assert current.json()["query"] == query
            assert current.json()["hits"] == []
            assert opened[0].queries == [query, query]

            restored = client.post(f"/api/operations/{operation_id}/rollback")
            assert restored.json() == {"restored": True}
            refreshed = client.post("/api/search", json=supplied)
            assert refreshed.status_code == 200
            assert refreshed.json()["query"] == query
            assert [hit["id"] for hit in refreshed.json()["hits"]] == source_ids
            assert opened[0].queries == [query, query, query]


@pytest.mark.parametrize("forget_source", [True, False])
def test_media_review_rechecks_sources_while_unrelated_dialogue_can_continue(
    tmp_path: Path, forget_source: bool
) -> None:

    class BlockingReviewer(SourceReviewer):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: AnswerPolicy = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            self.started.set()
            assert self.release.wait(10), "the test did not release original-media review"
            return super().answer(
                question, hits, answer_policy=answer_policy, exhaustive=exhaustive
            )

    reviewer = BlockingReviewer()
    source_ids: list[str] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        local = Memory(
            config.data_dir,
            embedder=TinyEmbedder(),
            answerer=reviewer,
            settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
        )
        source = local.add(("The keys are in the blue bowl.", Blob(b"original frame", "image/png")))
        source_ids.append(source.id)
        return AsyncMemory(local), Segmenter()

    with (
        TestClient(
            create_app(_config(tmp_path, generation=True), memory_factory=factory)
        ) as client,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        session = client.post("/api/session").json()
        response = executor.submit(
            client.post, "/api/search", json={"query": "Where are the keys?"}
        )
        assert reviewer.started.wait(5)
        try:
            if forget_source:
                changed = client.post(f"/api/memories/{source_ids[0]}/forget")
                assert changed.status_code == 200
            else:
                changed = client.post(
                    "/api/dialogue",
                    json={
                        "source_id": session["source_id"],
                        "response_id": "unrelated-response",
                        "started_at": "2026-11-17T00:00:00Z",
                        "completed_at": "2026-11-17T00:00:01Z",
                        "origin": "realtime",
                        "assistant": {"item_ids": ["unrelated-item"], "text": "好的。"},
                    },
                )
                assert changed.status_code == 202
        finally:
            reviewer.release.set()
        rejected = response.result(timeout=5)
        assert reviewer.questions
        if forget_source:
            assert rejected.status_code == 409
            assert rejected.json() == {
                "detail": "Memory changed during retrieval; please ask again."
            }
            assert "answer" not in rejected.json()
            assert "hits" not in rejected.json()
        else:
            assert rejected.status_code == 200
            assert rejected.json()["media_review"]["answer"].endswith("blue bowl.")
            assert rejected.json()["memory_revision"] > session["memory_revision"]


async def test_native_stream_groups_same_call_as_one_source_and_new_call_independently(
    tmp_path: Path,
) -> None:

    audio = BytesIO()
    with wave.open(audio, "wb") as encoded:
        encoded.setnchannels(1)
        encoded.setsampwidth(2)
        encoded.setframerate(16_000)
        encoded.writeframes(bytes(1600))

    runtime: LiveRuntime = LiveRuntime(
        _config(tmp_path),
        lambda config: (AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())), Segmenter()),
        "wss://realtime.example.test/v1/realtime",
    )
    start = datetime(2026, 11, 16, tzinfo=timezone.utc)
    try:
        await runtime.start_capture()
        memory = await runtime.open_memory()
        capture = runtime.capture
        assert capture is not None
        for index in range(2):
            await capture.push(
                EventChunk(
                    (Blob(audio.getvalue(), "audio/wav"),),
                    occurred_at=start + timedelta(seconds=index),
                    occurred_end=start + timedelta(seconds=index + 1),
                )
            )
        await runtime.finish_capture()
        await capture.wait()
        first_call = (await memory.list()).items
        assert len(first_call) == 2
        first_groups = {
            record.context.source_id for record in first_call if record.context is not None
        }
        assert len(first_groups) == 1
        assert all(group is not None and group.startswith("live:") for group in first_groups)
        await runtime.start_capture()
        next_capture = runtime.capture
        assert next_capture is not None
        await next_capture.push(
            EventChunk(
                (Blob(audio.getvalue(), "audio/wav"),),
                occurred_at=start + timedelta(seconds=10),
                occurred_end=start + timedelta(seconds=11),
            )
        )
        await runtime.finish_capture()
        await next_capture.wait()
        all_groups = {
            record.context.source_id
            for record in (await memory.list()).items
            if record.context is not None
        }
        assert len(all_groups) == 2
    finally:
        await runtime.close()


async def test_partial_settlement_failure_still_invalidates_recall_and_pushes_updates(
    tmp_path: Path,
) -> None:

    class PartialFailure(AsyncMemory):
        async def settle(
            self,
            *,
            limit: int = 100,
            max_attempts: int = 3,
            memory_ids: Sequence[str] | None = None,
        ) -> int:
            count = await super().settle(
                limit=limit, max_attempts=max_attempts, memory_ids=memory_ids
            )
            if count:
                raise ModelError("one batch member failed", reason="model_failed")
            return count

    runtime = LiveRuntime(
        _config(tmp_path),
        lambda config: (
            PartialFailure(Memory(config.data_dir, embedder=TinyEmbedder())),
            Segmenter(),
        ),
        "wss://realtime.example.test/v1/realtime",
    )
    try:
        memory = await runtime.open_memory()
        record = await memory.capture("An independently successful batch member")
        runtime.work.set()
        async with runtime.changes:
            await asyncio.wait_for(
                runtime.changes.wait_for(lambda: runtime.memory_revision > 0),
                timeout=5,
            )
        assert runtime.view_revision > 0
        assert record.id in {hit.id for hit in (await memory.search(record.content)).hits}
    finally:
        await runtime.close()


class PreferenceFormer:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "live-formation-fixture"
    formation_space = "live-formation-fixture:v1"

    def __init__(self) -> None:
        self.calls = 0

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.calls += 1
        return tuple(
            (
                FormationProposal(
                    kind=RecordKind.STATE,
                    content="The user prefers tea",
                    subject="user",
                    predicate="preferred_drink",
                    value="tea",
                ),
            )
            for _value in inputs
        )

    def close(self) -> None:
        pass


@pytest.mark.parametrize("fail", [False, True])
def test_settlement_keeps_processing_visible_and_notifies_views_until_model_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:

    class BlockingFormer(PreferenceFormer):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()

        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            self.started.set()
            assert self.release.wait(10), "the test did not release native formation"
            if fail:
                raise ModelError("private formation response", reason="timeout")
            return super().form(inputs)

    former = BlockingFormer()
    notices: list[tuple[int, bool]] = []
    started_notice, finished_notice = Event(), Event()
    notify = demo.LiveRuntime._notify_changes

    async def observe_notice(runtime: LiveRuntime) -> None:
        await notify(runtime)
        notices.append((runtime.view_revision, runtime.maintaining))
        (started_notice if runtime.maintaining else finished_notice).set()

    monkeypatch.setattr(demo.LiveRuntime, "_notify_changes", observe_notice)

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        local = Memory(config.data_dir, embedder=TinyEmbedder(), former=former)
        local.capture("I prefer tea")
        return AsyncMemory(local), Segmenter()

    with TestClient(demo.create_app(_config(tmp_path), memory_factory=factory)) as client:
        try:
            client.get("/api/memories")
            assert former.started.wait(5)
            assert started_notice.wait(5)
            assert not finished_notice.is_set()
            assert client.get("/api/memories").json()["processing"] is True
        finally:
            former.release.set()
        assert finished_notice.wait(5)
        state = client.get("/api/memories").json()
        assert state["processing"] is False
        assert notices[0][1] is True and notices[-1][1] is False
        assert notices[-1][0] > notices[0][0]
        assert "private formation response" not in str(state)


class Maintenance:
    consolidation_model = "live-maintenance-fixture"
    consolidation_recipe = "live-maintenance-fixture:v1"

    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.called = Event()

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
        query_failure: QueryFailure | None = None,
    ) -> tuple[MemoryOperation, ...]:
        self.called.set()
        if self.fail:
            raise ModelError("private model response must not leave the backend", reason="timeout")
        return ()

    def close(self) -> None:
        pass


async def test_close_drains_native_captures_without_starting_another_consolidation(
    tmp_path: Path,
) -> None:

    class CountingMemory(AsyncMemory):
        def __init__(self, memory: Memory) -> None:
            super().__init__(memory)
            self.maintenance_calls = 0

        async def consolidate_pending(
            self, *, limit: int = 32, max_rounds: int = 4, idle: bool = False
        ) -> ConsolidationReport:
            self.maintenance_calls += 1
            return await super().consolidate_pending(limit=limit, max_rounds=max_rounds, idle=idle)

    former, maintenance = PreferenceFormer(), Maintenance(False)
    opened: list[CountingMemory] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        local = Memory(
            config.data_dir,
            embedder=TinyEmbedder(),
            former=former,
            consolidator=maintenance,
        )
        local.capture("I prefer tea")
        memory = CountingMemory(local)
        opened.append(memory)
        return memory, Segmenter()

    runtime = LiveRuntime(
        _config(tmp_path, lifecycle=True),
        factory,
        "wss://realtime.example.test/v1/realtime",
    )
    try:
        memory = await runtime.open_memory()
        async with runtime.changes:
            await asyncio.wait_for(
                runtime.changes.wait_for(
                    lambda: opened[0].maintenance_calls == 1 and not runtime.maintaining
                ),
                timeout=5,
            )
        assert maintenance.called.is_set() and former.calls == 1
        tail = await memory.capture("I prefer tea on Friday")
        assert tail.id in {item.memory_id for item in await memory.pending_captures()}
        await runtime.close()
        assert opened[0].maintenance_calls == 1
        assert former.calls == 2
        with Memory(tmp_path, embedder=TinyEmbedder()) as reopened:
            assert not reopened.pending_captures()
            assert tail.id in {hit.id for hit in reopened.search(tail.content, limit=10).hits}
    finally:
        await runtime.close()


@pytest.mark.parametrize("fail", [False, True])
def test_worker_runs_native_formation_and_bounded_maintenance_with_safe_failure_state(
    tmp_path: Path, fail: bool
) -> None:

    former = PreferenceFormer()
    maintenance = Maintenance(fail)

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        local = Memory(
            config.data_dir,
            embedder=TinyEmbedder(),
            former=former,
            consolidator=maintenance,
        )
        local.capture("I prefer tea")
        return AsyncMemory(local), Segmenter()

    with TestClient(
        create_app(_config(tmp_path, lifecycle=True), memory_factory=factory)
    ) as client:
        public = client.get("/api/config").json()
        assert public["formation_enabled"]
        assert public["maintenance_enabled"]
        client.get("/api/memories")
        assert maintenance.called.wait(5)
        deadline = time.monotonic() + 5
        while True:
            state = client.get("/api/memories").json()
            if not state["processing"] and state["maintenance_failed"] is fail:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert former.calls == 1
        assert any(item["kind"] == "state" for item in state["items"])
        assert "private model response" not in str(state)
