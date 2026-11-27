"""Completed demo replies become durable, attributed native conversation evidence."""

from __future__ import annotations

import base64
import io
import re
import struct
import time
import wave
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import cast

import pytest
from _feature_support import TinyEmbedder
from fastapi.testclient import TestClient

from examples.live.app import LiveRuntime, create_app
from examples.live.perception import FrameEvidence, PerceptionContext, PerceptionResult
from examples.live.vision import LiveVision
from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AsyncMemory,
    EmbedTask,
    EventObservation,
    EventSegment,
    EvidenceBasis,
    Memory,
    MemoryCategory,
    MemoryRecord,
    MemorySettings,
    MemorySpec,
    Modality,
    ModelInput,
    SearchHit,
    StorageError,
)


class Segmenter:
    event_model = "dialogue-fixture"
    event_recipe = "dialogue-fixture:v1"

    def segment_events(
        self, observations: Sequence[EventObservation], *, end_of_stream: bool = False
    ) -> tuple[EventSegment, ...]:
        return ()


def _config(path: Path) -> MemorySpec:
    return MemorySpec.model_validate({"data_dir": path, "embedding": {"provider": "openai"}})


def _payload(source_id: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "response_id": "response-1",
        "started_at": "2026-11-16T10:00:00Z",
        "completed_at": "2026-11-16T10:00:03Z",
        "user": {"item_id": "user-1", "text": "我刚才请你提醒什么?"},
        "assistant": {
            "item_ids": ["assistant-1", "assistant-2"],
            "text": "我建议明天带上紫色笔记本和备用电池。",
        },
    }


def test_native_reply_is_saved_without_inventing_a_user_transcript(tmp_path: Path) -> None:

    opened: list[Memory] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        memory = Memory(config.data_dir, embedder=TinyEmbedder())
        opened.append(memory)
        return AsyncMemory(memory), Segmenter()

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        source_id = client.post("/api/session").json()["source_id"]
        payload = _payload(source_id)
        payload.pop("user")
        payload["origin"] = "realtime"
        payload["feedback_receipts"] = [
            {
                "input_id": "av_12_12000",
                "feedback_id": "evidence-1",
                "memory_revision": 2,
                "memory_ids": ["memory-1"],
                "candidate_id": None,
                "action": "quiet",
                "event_id": "evt-1",
                "operation": "replace",
            }
        ]
        result = client.post("/api/dialogue", json=payload)
        assert result.status_code == 202
        record = opened[0].get(result.json()["id"])
        dialogue = cast(dict[str, object], record.metadata["dialogue"])
        assert dialogue["user"] is None
        assert dialogue["origin"] == "realtime"
        assert dialogue["feedback_receipts"] == payload["feedback_receipts"]
        assert dialogue["feedback_semantic_adoption_verified"] is False
        assert dialogue["feedback_receipt_basis"] == "client_reported_native_input_acknowledgment"
        assert record.content == "SenseMem:\n我建议明天带上紫色笔记本和备用电池。"
        assert dialogue["input_transcript_basis"] == "absent"
        assert client.post("/api/dialogue", json=payload).json() == result.json()
        memory_id = record.id

    with Memory(tmp_path, embedder=TinyEmbedder()) as reopened:
        assert reopened.get(memory_id).metadata["dialogue"] == dialogue


@pytest.mark.parametrize("status", ["completed", "interrupted", "stopped"])
def test_native_rendered_pcm_is_attributed_preserved_and_replayable_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:

    opened: list[Memory] = []
    runtimes: list[LiveRuntime] = []
    pcm = struct.pack("<6h", -32768, -1000, 0, 1000, 16384, 32767) * 4000
    start_capture = LiveRuntime.start_capture

    async def start(runtime: LiveRuntime) -> str:
        runtimes.append(runtime)
        return await start_capture(runtime)

    monkeypatch.setattr(LiveRuntime, "start_capture", start)

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        memory = Memory(config.data_dir, embedder=TinyEmbedder())
        opened.append(memory)
        return AsyncMemory(memory), Segmenter()

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        payload = _payload(client.post("/api/session").json()["source_id"])
        payload.pop("user")
        payload.update(
            origin="realtime",
            assistant={
                "item_ids": ["native-response"],
                "text": "生成字幕仍可能包含尚未播放的结尾。",
            },
            playback={
                "sample_rate": 24000,
                "pcm_s16le": base64.b64encode(pcm).decode(),
                "rendered_samples": len(pcm) // 2,
                "status": status,
            },
        )
        response = client.post("/api/dialogue", json=payload)
        assert response.status_code == 202
        record = opened[0].get(response.json()["id"])
        assert record.modality is Modality.AUDIO and len(record.assets) == 1
        retained = opened[0].read_asset(record.id, record.assets[0].id)
        with wave.open(io.BytesIO(retained.data), "rb") as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (
                24000,
                1,
                2,
            )
            assert audio.getnframes() == 24000
            assert audio.readframes(audio.getnframes()) == pcm
        assert record.content == "SenseMem:\n生成字幕仍可能包含尚未播放的结尾。"
        dialogue = cast(dict[str, object], record.metadata["dialogue"])
        assert dialogue["speaker"] == "assistant" and dialogue["user"] is None
        assert dialogue["completion"] == status
        assert dialogue["playback_clock"] == "web_audio_render_clock"
        assert dialogue["actual_hearing_verified"] is False
        assert dialogue["generated_text_alignment"] == "unverified"
        assert (runtimes[0].latest_dialogue is not None) == (status == "completed")
        assert dialogue["playback"] == {
            "sample_rate": 24000,
            "rendered_samples": 24000,
            "status": status,
        }
        assert client.post("/api/dialogue", json=payload).json() == response.json()
        assert len(opened[0].list().items) == 1
        memory_id, asset_id = record.id, record.assets[0].id
    with Memory(tmp_path, embedder=TinyEmbedder()) as reopened:
        assert reopened.get(memory_id).metadata["dialogue"] == dialogue
        assert reopened.read_asset(memory_id, asset_id).data == retained.data


def test_native_interrupted_audio_without_a_generated_subtitle_is_saved(tmp_path: Path) -> None:

    opened: list[Memory] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        memory = Memory(config.data_dir, embedder=TinyEmbedder())
        opened.append(memory)
        return AsyncMemory(memory), Segmenter()

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        payload = _payload(client.post("/api/session").json()["source_id"])
        payload.pop("user")
        payload.update(
            origin="realtime",
            assistant={"item_ids": ["native-audio-only"], "text": ""},
            playback={
                "sample_rate": 24000,
                "pcm_s16le": "AQD//w==",
                "rendered_samples": 2,
                "status": "interrupted",
            },
        )
        response = client.post("/api/dialogue", json=payload)
        assert response.status_code == 202
        record = opened[0].get(response.json()["id"])
        assert record.content == "SenseMem:\nVoice reply"
        dialogue = cast(dict[str, object], record.metadata["dialogue"])
        assert dialogue["completion"] == "interrupted"
        assert dialogue["playback"] == {
            "sample_rate": 24000,
            "rendered_samples": 2,
            "status": "interrupted",
        }
        assert len(record.assets) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_rate": 16000},
        {"sample_rate": 24000.0},
        {"pcm_s16le": "not base64"},
        {"pcm_s16le": "A Q A A A A=="},
        {"pcm_s16le": "AAA="},
        {"rendered_samples": 3},
        {"rendered_samples": 0},
        {"rendered_samples": 7_200_001},
        {"rendered_samples": True},
        {"status": "heard"},
        {"extra": "invalid"},
    ],
)
def test_native_invalid_playback_is_rejected_without_capturing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:

    async def unexpected_capture(*_args: object, **_kwargs: object) -> MemoryRecord:
        pytest.fail("invalid playback reached memory capture")

    monkeypatch.setattr(AsyncMemory, "capture", unexpected_capture)
    with TestClient(
        create_app(
            _config(tmp_path),
            memory_factory=lambda config: (
                AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())),
                Segmenter(),
            ),
        )
    ) as client:
        payload = _payload("live:" + "a" * 32)
        payload.pop("user")
        payload.update(
            origin="realtime",
            playback={
                "sample_rate": 24000,
                "pcm_s16le": "AQAAAA==",
                "rendered_samples": 2,
                "status": "interrupted",
                **changes,
            },
        )
        assert client.post("/api/dialogue", json=payload).status_code == 422
        assert client.get("/api/memories").json()["items"] == []


def test_native_empty_or_oversized_playback_does_not_reach_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    async def unexpected_capture(*_args: object, **_kwargs: object) -> MemoryRecord:
        pytest.fail("invalid playback reached memory capture")

    monkeypatch.setattr(AsyncMemory, "capture", unexpected_capture)
    with TestClient(
        create_app(
            _config(tmp_path),
            memory_factory=lambda config: (
                AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())),
                Segmenter(),
            ),
        )
    ) as client:
        payload = _payload("live:" + "a" * 32)
        payload.pop("user")
        payload.update(origin="realtime", assistant={"item_ids": ["empty"], "text": ""})
        assert client.post("/api/dialogue", json=payload).status_code == 422
        payload["playback"] = {
            "sample_rate": 24000,
            "pcm_s16le": "",
            "rendered_samples": 0,
            "status": "interrupted",
        }
        assert client.post("/api/dialogue", json=payload).status_code == 422
        assert (
            client.post("/api/dialogue", content=b"x" * (20 * 1024 * 1024 + 1)).status_code == 413
        )
        assert client.get("/api/memories").json()["items"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_id", "fake-candidate"),
        ("event_id", "fake-event"),
        ("evidence_ids", ["fake-evidence"]),
    ],
)
def test_native_reply_rejects_unattributed_trigger_evidence(
    tmp_path: Path, field: str, value: object
) -> None:

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        return AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())), Segmenter()

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        payload = _payload(client.post("/api/session").json()["source_id"])
        payload.pop("user")
        payload.update({"origin": "realtime", field: value})
        assert client.post("/api/dialogue", json=payload).status_code == 422


def test_realtime_reply_retains_final_asr_as_client_reported_text(tmp_path: Path) -> None:

    opened: list[Memory] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        memory = Memory(config.data_dir, embedder=TinyEmbedder())
        opened.append(memory)
        return AsyncMemory(memory), Segmenter()

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        payload = _payload(client.post("/api/session").json()["source_id"])
        payload["origin"] = "realtime"
        response = client.post("/api/dialogue", json=payload)
        assert response.status_code == 202
        record = opened[0].get(response.json()["id"])
        dialogue = cast(dict[str, object], record.metadata["dialogue"])
        assert dialogue["user"] == payload["user"]
        assert dialogue["input_transcript_basis"] == "client_reported_final_asr"
        assert record.content == (
            "User:\n我刚才请你提醒什么?\n\nSenseMem:\n我建议明天带上紫色笔记本和备用电池。"
        )
        assert dialogue["actual_hearing_verified"] is False


def test_dialogue_acknowledges_before_embedding_and_survives_restart(tmp_path: Path) -> None:

    entered, release = Event(), Event()
    opened: list[Memory] = []

    class BlockingEmbedder(TinyEmbedder):
        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            if task is EmbedTask.DOCUMENT:
                entered.set()
                assert release.wait(10), "test did not release the embedding model"
            return super().embed(inputs, task)

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        local = Memory(
            config.data_dir,
            embedder=BlockingEmbedder(),
            settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
        )
        opened.append(local)
        return AsyncMemory(local), Segmenter()

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        session = client.post("/api/session")
        assert session.status_code == 200
        source_id = session.json()["source_id"]
        assert re.fullmatch(r"live:[0-9a-f]{32}", source_id)
        assert client.post("/api/session").json() == {
            "source_id": source_id,
            "memory_revision": 0,
        }
        payload = _payload(source_id)
        user = cast(dict[str, object], payload["user"])
        assistant = cast(dict[str, object], payload["assistant"])
        with ThreadPoolExecutor(max_workers=1) as requests:
            pending = requests.submit(client.post, "/api/dialogue", json=payload)
            try:
                assert entered.wait(5), "dialogue capture did not schedule native settlement"
                response = pending.result(timeout=2)
                assert response.status_code == 202
                memory_id = response.json()["id"]
                record = opened[0].get(memory_id)
                assert record.memory_type is MemoryCategory.EPISODIC
                assert record.context is not None
                assert record.context.basis is EvidenceBasis.OBSERVATION
                assert record.context.source_id == source_id
                assert record.occurred_at == datetime.fromisoformat(str(payload["started_at"]))
                assert record.occurred_end == datetime.fromisoformat(str(payload["completed_at"]))
                assert record.content == (
                    f"User:\n{user['text']}\n\nSenseMem:\n{assistant['text']}"
                )
                dialogue = cast(dict[str, object], record.metadata["dialogue"])
                for field in ("source_id", "response_id", "user", "assistant"):
                    assert dialogue[field] == payload[field]
                for field in ("started_at", "completed_at"):
                    assert datetime.fromisoformat(str(dialogue[field])) == datetime.fromisoformat(
                        str(payload[field])
                    )
                assert opened[0].pending_captures(memory_ids=(memory_id,))
                assert client.post("/api/dialogue", json=payload).json() == {"id": memory_id}
                another = {**payload, "response_id": "response-2"}
                second = client.post("/api/dialogue", json=another)
                assert second.status_code == 202
                second_id = second.json()["id"]
                assert second_id != memory_id
                assert len(opened[0].list().items) == 2
                assert not release.is_set()
            finally:
                release.set()
        deadline = time.monotonic() + 5
        while opened[0].pending_captures():
            assert time.monotonic() < deadline, "native settlement did not finish"
            time.sleep(0.01)
        result = client.post("/api/search", json={"query": str(assistant["text"])})
        assert result.status_code == 200
        assert memory_id in {hit["id"] for hit in result.json()["hits"]}
        assert client.delete("/api/session").status_code == 204
        next_session = client.post("/api/session")
        assert next_session.status_code == 200
        assert next_session.json()["source_id"] != source_id

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
    ) as reopened:
        assert reopened.get(memory_id).metadata == record.metadata
        assert reopened.pending_captures() == ()
        assert memory_id in {hit.id for hit in reopened.search(str(assistant["text"])).hits}
        assert str(assistant["text"]) in reopened.compile(str(assistant["text"])).render()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", "live:bad"),
        ("source_id", "live:" + "A" * 32),
        ("response_id", ""),
        ("response_id", "   "),
        ("response_id", "r" * 129),
        ("response_id", 1),
        ("started_at", "2026-11-16T10:00:00"),
        ("completed_at", "2026-11-16T09:59:59Z"),
        ("completed_at", "2026-11-16T10:00:03"),
        ("extra", "ignored fields must be rejected"),
        ("user.item_id", ""),
        ("user.item_id", " " * 2),
        ("user.item_id", "u" * 129),
        ("user.text", "  \n\t"),
        ("user.text", "u" * 4001),
        ("user.extra", "ignored fields must be rejected"),
        ("assistant.item_ids", []),
        ("assistant.item_ids", ["item"] * 9),
        ("assistant.item_ids", [""]),
        ("assistant.item_ids", ["  "]),
        ("assistant.item_ids", ["a" * 129]),
        ("assistant.text", " \n\t"),
        ("assistant.text", "a" * 8001),
        ("assistant.extra", "ignored fields must be rejected"),
    ],
)
def test_dialogue_rejects_invalid_input_before_capture(
    tmp_path: Path, field: str, value: object, monkeypatch: pytest.MonkeyPatch
) -> None:

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        return AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())), Segmenter()

    async def unexpected_capture(*_args: object, **_kwargs: object) -> MemoryRecord:
        pytest.fail("invalid dialogue reached memory capture")

    monkeypatch.setattr(AsyncMemory, "capture", unexpected_capture)

    payload = deepcopy(_payload("live:" + "a" * 32))
    if "." in field:
        parent, key = field.split(".", 1)
        cast(dict[str, object], payload[parent])[key] = value
    else:
        payload[field] = value
    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        response = client.post("/api/dialogue", json=payload)
        assert response.status_code == 422


def test_dialogue_storage_failure_does_not_expose_provider_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    opened: list[AsyncMemory] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        memory = AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder()))
        opened.append(memory)
        return memory, Segmenter()

    async def fail_capture(*_args: object, **_kwargs: object) -> MemoryRecord:
        raise StorageError("private provider body: secret dialogue and credential")

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        source_id = client.post("/api/session").json()["source_id"]
        monkeypatch.setattr(opened[0], "capture", fail_capture)
        response = client.post("/api/dialogue", json=_payload(source_id))
        assert response.status_code == 503
        assert response.json() == {"detail": "The memory operation could not be completed."}
        assert "private" not in response.text
        assert "credential" not in response.text


@pytest.fixture
def visual_dialogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, Memory, dict[str, object]]]:

    class VisualSegmenter(Segmenter):
        generation_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: AnswerPolicy = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            pytest.fail("a dialogue HTTP contract must not invoke a model")

        def close(self) -> None:
            pass

    opened: list[Memory] = []
    runtimes: list[LiveRuntime] = []
    generation = VisualSegmenter()

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        memory = Memory(config.data_dir, embedder=TinyEmbedder())
        opened.append(memory)
        return AsyncMemory(memory), generation

    start_capture = LiveRuntime.start_capture

    async def issue(runtime: LiveRuntime) -> str:
        runtimes.append(runtime)
        source_id = await start_capture(runtime)
        assert runtime.recall is not None
        vision = LiveVision(
            await runtime.open_memory(),
            generation,
            runtime.recall,
            source_id,
            runtime.visual_initiatives,
        )
        now = datetime.now(timezone.utc)
        await vision._observed(
            PerceptionResult(
                summary="MODEL_ONLY_SUMMARY_MUST_NOT_BECOME_A_DURABLE_EVENT",
                changed=True,
                action="wait",
                utterance="The proposed words are not proof of what the Agent actually said.",
                confidence=0.95,
                evidence=(
                    FrameEvidence(sequence=1, observation="A key is in the tray."),
                    FrameEvidence(sequence=2, observation="The key is in the bowl."),
                ),
                reason="A completed visible placement.",
                contradicted_claim=None,
                correction=None,
                source_id=source_id,
                frame_sequences=(1, 2),
                occurred_at=now - timedelta(seconds=2),
                occurred_end=now - timedelta(seconds=1),
                timestamp_ms=1000.0,
                end_timestamp_ms=2000.0,
                context=PerceptionContext(memory_revision=runtime.memory_revision),
                target_response_id=None,
                current=False,
                latest_sequence=3,
                expires_at=now + timedelta(seconds=7),
                epoch=0,
            )
        )
        await vision.close()
        return source_id

    monkeypatch.setattr(LiveRuntime, "start_capture", issue)
    app = create_app(_config(tmp_path), memory_factory=factory)
    with TestClient(app) as client:
        source_id = client.post("/api/session").json()["source_id"]
        candidate = next(iter(runtimes[0].visual_initiatives.values()))
        payload = _payload(source_id)
        payload.pop("user")
        payload.update(
            origin="perception",
            candidate_id=candidate["candidate_id"],
            event_id=candidate["perception_id"],
            evidence_ids=[],
            perception={
                "frame_sequences": candidate["frame_sequences"],
                "occurred_at": candidate["occurred_at"],
                "occurred_end": candidate["occurred_end"],
            },
        )
        yield client, opened[0], payload


def test_perception_dialogue_requires_actual_issued_timed_frame_provenance(
    visual_dialogue: tuple[TestClient, Memory, dict[str, object]],
) -> None:
    client, memory, payload = visual_dialogue
    assert memory.list().items == (), "temporary observation alone must not be captured"
    accepted = client.post("/api/dialogue", json=payload)
    assert accepted.status_code == 202
    record = memory.get(accepted.json()["id"])
    dialogue = cast(dict[str, object], record.metadata["dialogue"])
    assert dialogue["origin"] == "perception" and dialogue["user"] is None
    stored = cast(dict[str, object], dialogue["perception"])
    supplied = cast(dict[str, object], payload["perception"])
    assert stored["frame_sequences"] == supplied["frame_sequences"]
    for field in ("occurred_at", "occurred_end"):
        assert datetime.fromisoformat(str(stored[field])) == datetime.fromisoformat(
            str(supplied[field])
        )
    assert dialogue["evidence_ids"] == []
    assert dialogue["event_id"] == payload["event_id"]
    assert record.context is not None and record.context.source_id == payload["source_id"]
    assert record.context.basis is EvidenceBasis.OBSERVATION


def test_perception_http_rejects_unissued_or_altered_source_and_frame_identity(
    visual_dialogue: tuple[TestClient, Memory, dict[str, object]],
) -> None:
    client, memory, payload = visual_dialogue
    for changes in (
        {"candidate_id": "never-issued"},
        {"event_id": "different-observation"},
        {"source_id": "live:" + "b" * 32},
        {
            "perception": {
                **cast(dict[str, object], payload["perception"]),
                "frame_sequences": [1, 3],
            }
        },
        {
            "perception": {
                **cast(dict[str, object], payload["perception"]),
                "occurred_at": "2026-11-16T09:00:00Z",
            }
        },
    ):
        rejected = client.post("/api/dialogue", json={**payload, **changes})
        assert rejected.status_code == 422
    assert memory.list().items == ()


def test_perception_dialogue_rejects_fake_native_evidence_and_noncanonical_frame_sequences(
    visual_dialogue: tuple[TestClient, Memory, dict[str, object]],
) -> None:
    client, memory, payload = visual_dialogue
    for sequences in ([], [2, 1], [1, 1], [0, 2], [True, 2], ["1", 2], [1.0, 2]):
        altered = deepcopy(payload)
        cast(dict[str, object], altered["perception"])["frame_sequences"] = sequences
        assert client.post("/api/dialogue", json=altered).status_code == 422
    for changes in (
        {"evidence_ids": ["fake-native-memory"]},
        {"perception": None},
        {"user": {"item_id": "invented-user", "text": "A fabricated ASR turn"}},
    ):
        assert client.post("/api/dialogue", json={**payload, **changes}).status_code == 422
    assert memory.list().items == ()


def test_perception_capture_retains_agent_text_and_temporary_source_metadata(
    visual_dialogue: tuple[TestClient, Memory, dict[str, object]],
) -> None:
    client, memory, payload = visual_dialogue
    spoken = "刚才画面里的钥匙被放进了碗里。"
    payload["assistant"] = {"item_ids": ["actual-generated-item"], "text": spoken}
    response = client.post("/api/dialogue", json=payload)
    assert response.status_code == 202
    record = memory.get(response.json()["id"])
    assert record.content == f"SenseMem:\n{spoken}"
    assert "event_segmentation" not in record.metadata
    assert set(record.metadata) == {"dialogue"}
    dialogue = cast(dict[str, object], record.metadata["dialogue"])
    assert dialogue["assistant"] == payload["assistant"] and dialogue["user"] is None
    assert dialogue["origin"] == "perception"
    stored = cast(dict[str, object], dialogue["perception"])
    supplied = cast(dict[str, object], payload["perception"])
    assert stored["frame_sequences"] == supplied["frame_sequences"]
    for field in ("occurred_at", "occurred_end"):
        assert datetime.fromisoformat(str(stored[field])) == datetime.fromisoformat(
            str(supplied[field])
        )
    assert client.post("/api/dialogue", json=payload).json() == response.json()
    assert len(memory.list().items) == 1
