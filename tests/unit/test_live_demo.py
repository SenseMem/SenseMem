"""The local example retains streamed A/V while model work remains asynchronous."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import cast

import av
import numpy as np
import pytest
import uvicorn
from _feature_support import TinyEmbedder
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import examples.live.app as live_app
from examples.live import app as demo
from examples.live import prompts as demo_prompts
from examples.live.app import DemoStaticFiles, LiveRuntime, create_app
from sensemem import (
    AsyncEventStream,
    AsyncMemory,
    EmbedTask,
    EventObservation,
    EventSegment,
    EventSegmentationBackend,
    Memory,
    MemorySettings,
    MemorySpec,
    ModelError,
    ModelInput,
    ObservationContext,
    Page,
    SenseMemError,
    StorageError,
)
from sensemem.configuration import OpenAIGenerationConfig


def _config(path: Path) -> MemorySpec:
    return MemorySpec.model_validate({"data_dir": path, "embedding": {"provider": "openai"}})


def _recording(seconds: int = 1) -> bytes:
    encoded = BytesIO()
    with av.open(
        encoded,
        "w",
        format="mp4",
        options={"movflags": "empty_moov+default_base_moof+frag_keyframe"},
    ) as target:
        video = target.add_stream("libvpx-vp9", rate=8)
        video.width = video.height = 16
        video.pix_fmt = "yuv420p"
        video.gop_size = 8
        audio = target.add_stream("libopus", rate=48_000)
        audio.layout = "mono"
        for index in range(seconds * 8):
            pixels = np.zeros((16, 16, 3), dtype=np.uint8)
            pixels[:, :, 0] = index * 25 % 256
            for packet in video.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                target.mux(packet)
        for packet in video.encode():
            target.mux(packet)
        sound = av.AudioFrame.from_ndarray(
            np.zeros((1, seconds * 48_000), dtype=np.float32), format="flt"
        )
        sound.sample_rate = 48_000
        sound.pts = 0
        for packet in audio.encode(sound):
            target.mux(packet)
        for packet in audio.encode():
            target.mux(packet)
    return encoded.getvalue()


def _packets(data: bytes) -> list[tuple[str, bytes]]:
    with av.open(BytesIO(data), mode="r") as source:
        return [
            (packet.stream.type, bytes(packet))
            for packet in source.demux()
            if packet.dts is not None
        ]


def _upload(
    client: TestClient,
    data: bytes,
    *,
    started_at: str = "2026-11-16T10:00:00Z",
    fragment_size: int = 257,
) -> dict[str, object]:
    with client.websocket_connect("/api/media") as socket:
        socket.send_json({"type": "start", "started_at": started_at})
        for offset in range(0, len(data), fragment_size):
            socket.send_bytes(data[offset : offset + fragment_size])
        socket.send_json({"type": "end"})
        return cast(dict[str, object], socket.receive_json())


def _wait_until(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "background processing did not reach its expected state"
        time.sleep(0.01)


class ObservedMemory(AsyncMemory):
    def __init__(self, memory: Memory) -> None:
        super().__init__(memory)
        self.finished = Event()

    async def settle(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 3,
        memory_ids: Sequence[str] | None = None,
    ) -> int:
        try:
            count = await super().settle(
                limit=limit, max_attempts=max_attempts, memory_ids=memory_ids
            )
        except SenseMemError:
            self.finished.set()
            raise
        if count:
            self.finished.set()
        return count


class Segmenter:
    event_model = "demo-event-fixture"
    event_recipe = "demo-event-fixture:v1"

    def __init__(self, replies: Sequence[tuple[EventSegment, ...] | ModelError] = ()) -> None:
        self.replies = iter(replies)
        self.calls: list[tuple[tuple[EventObservation, ...], bool]] = []

    def segment_events(
        self,
        observations: Sequence[EventObservation],
        *,
        end_of_stream: bool = False,
    ) -> tuple[EventSegment, ...]:
        self.calls.append((tuple(observations), end_of_stream))
        reply = next(self.replies, ())
        if isinstance(reply, ModelError):
            raise reply
        return reply


class CompletingSegmenter(Segmenter):
    def segment_events(
        self,
        observations: Sequence[EventObservation],
        *,
        end_of_stream: bool = False,
    ) -> tuple[EventSegment, ...]:
        self.calls.append((tuple(observations), end_of_stream))
        return (
            EventSegment(
                len(observations),
                "The camera recorded a complete activity."
                if end_of_stream
                else "The activity is still ongoing.",
                complete=end_of_stream,
            ),
        )


def test_frontend_assets_revalidate_without_reusing_changed_modules(tmp_path: Path) -> None:

    with TestClient(
        create_app(
            _config(tmp_path / "memory"),
            memory_factory=lambda config: (
                AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())),
                Segmenter(),
            ),
        )
    ) as client:
        for path in ("/", "/app.js", "/memory-state.js", "/style.css"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-cache"
            unchanged = client.get(path, headers={"If-None-Match": response.headers["etag"]})
            assert unchanged.status_code == 304
            assert unchanged.headers["cache-control"] == "no-cache"

    asset = tmp_path / "module.js"
    asset.write_text("export const revision = 'before';", encoding="utf-8")
    app = FastAPI()
    app.mount("/", DemoStaticFiles(directory=tmp_path))
    with TestClient(app) as client:
        previous = client.get("/module.js")
        asset.write_text("export const revision = 'after the update';", encoding="utf-8")
        changed = client.get("/module.js", headers={"If-None-Match": previous.headers["etag"]})
        assert changed.status_code == 200
        assert changed.text == "export const revision = 'after the update';"
        assert changed.headers["etag"] != previous.headers["etag"]
        assert changed.headers["cache-control"] == "no-cache"


def test_stream_capture_settle_recall_and_replay_share_one_owner(tmp_path: Path) -> None:

    opened: list[ObservedMemory] = []
    segmenter = CompletingSegmenter()

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        memory = ObservedMemory(
            Memory(
                config.data_dir,
                embedder=TinyEmbedder(),
                settings=MemorySettings(minimum_relevance=0),
            )
        )
        opened.append(memory)
        return memory, segmenter

    directory = tmp_path / "companion"
    recording = _recording(seconds=2)
    with TestClient(create_app(_config(directory), memory_factory=factory)) as client:
        first_session = client.post("/api/session").json()
        assert first_session["source_id"].startswith("live:")
        assert client.post("/api/session").json() == first_session
        assert len(opened) == 1
        with pytest.raises(StorageError):
            Memory(directory, embedder=TinyEmbedder())
        with Memory(tmp_path / "independent", embedder=TinyEmbedder()) as separate:
            separate.add("Independent memory")
        assert _upload(client, recording) == {"type": "accepted"}
        assert opened[0].finished.wait(5)
        items = client.get("/api/memories").json()["items"]
        assert len(items) == 1
        memory_id = items[0]["id"]
        assert datetime.fromisoformat(items[0]["occurred_at"]) == datetime(
            2026, 11, 16, 10, tzinfo=timezone.utc
        )
        assert items[0]["occurred_end"] > items[0]["occurred_at"]
        assert items[0]["pending"] is False
        retained: list[tuple[str, bytes]] = []
        for asset in items[0]["assets"]:
            response = client.get(asset["url"])
            assert response.status_code == 200
            retained.extend(_packets(response.content))
        assert Counter(retained) == Counter(_packets(recording))
        found = client.post("/api/search", json={"query": "What did the camera record?"})
        assert found.status_code == 200
        assert [item["id"] for item in found.json()["hits"]] == [memory_id]
        assert type(found.json()["memory_revision"]) is int
        assert "untrusted evidence" in found.json()["context"]
        assert client.delete("/api/session").status_code == 204
        assert client.post("/api/session").status_code == 200
        assert len(opened) == 1

    with Memory(directory, embedder=TinyEmbedder()) as reopened:
        segmentation = reopened.get(memory_id).metadata["event_segmentation"]
        assert isinstance(segmentation, dict)
        assert segmentation["model"] == segmenter.event_model
        assert not reopened.pending_captures()


def test_s2s_configuration_exposes_native_audio_context_and_tool_contract(tmp_path: Path) -> None:

    with TestClient(
        create_app(
            _config(tmp_path),
            memory_factory=lambda config: (
                AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())),
                Segmenter(),
            ),
        )
    ) as client:
        config = client.get("/api/config").json()
        assert config["realtime_ready"] is True
        assert "frame_interval_ms" not in config
        assert config["perception_ready"] is False
        assert "chunk_seconds" not in config and "clip_seconds" not in config
        session = config["session"]
        assert session == {
            "type": "realtime",
            "instructions": demo_prompts.REALTIME_INSTRUCTIONS,
            "output_modalities": ["audio"],
            "extensions": ["speech_to_speech.input_audio_transcription.snapshot"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"language": "auto"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.35,
                        "silence_duration_ms": 600,
                        "create_response": False,
                        "interrupt_response": True,
                    },
                },
                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
            },
            "tools": [
                {
                    "type": "function",
                    "name": "search_memory",
                    "description": demo_prompts.SEARCH_MEMORY_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "minLength": 1, "maxLength": 2000}
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                }
            ],
            "tool_choice": "auto",
            "reasoning": {"effort": "none"},
        }
        assert config["realtime_model"] is None
        assert config["speech_service"] == "speech-to-speech"
        assert config["capabilities"] == {
            "input_audio": True,
            "input_video": False,
            "output_audio": True,
            "output_text": True,
            "input_transcription": True,
            "function_calling": True,
            "dynamic_context": True,
        }
        assert config["memory_context_instructions"] == demo_prompts.MEMORY_CONTEXT_INSTRUCTIONS
        assert (
            config["recall_unavailable_instructions"]
            == demo_prompts.RECALL_UNAVAILABLE_INSTRUCTIONS
        )
        assert (
            config["context_unavailable_instructions"]
            == demo_prompts.CONTEXT_UNAVAILABLE_INSTRUCTIONS
        )
        assert (
            config["visual_unavailable_instructions"]
            == demo_prompts.VISUAL_UNAVAILABLE_INSTRUCTIONS
        )
        assert config["proactive_instructions"] == demo_prompts.PROACTIVE_INSTRUCTIONS
        assert client.post("/api/connect", content=b"obsolete transport").status_code in {404, 405}


def test_local_speech_needs_no_modelbest_key_and_memory_credentials_remain_server_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setenv("SENSEMEM_GENERATION_API_KEY", "fixture-memory-key")
    monkeypatch.setenv("SENSEMEM_GENERATION_BASE_URL", "https://memory.example.test/api/v1")
    monkeypatch.delenv("MAP_API_KEY", raising=False)
    captured: list[MemorySpec] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        captured.append(config)
        return AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())), Segmenter()

    config = MemorySpec.model_validate(
        {
            "data_dir": tmp_path,
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai"},
        }
    )
    with TestClient(create_app(config, memory_factory=factory)) as client:
        public = client.get("/api/config")
        assert public.json()["realtime_ready"] is True
        assert "fixture-memory-key" not in public.text
        assert "private provider body" not in public.text
        assert client.post("/api/session").status_code == 200
        generation = captured[0].generation
        assert isinstance(generation, OpenAIGenerationConfig) and generation.api_key is not None
        assert generation.api_key.get_secret_value() == "fixture-memory-key"
        assert generation.base_url == "https://memory.example.test/api/v1"


def test_call_start_returns_only_source_and_revision_without_reading_a_history_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        retained_id = memory.add(
            "Existing memory remains retained, not copied into session init."
        ).id

    async def history(_memory: AsyncMemory, *, limit: int = 100, cursor: str | None = None) -> Page:
        raise AssertionError("Starting a call must not enumerate an initial history snapshot.")

    monkeypatch.setattr(AsyncMemory, "list", history)
    with TestClient(
        create_app(
            _config(tmp_path),
            memory_factory=lambda config: (
                AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())),
                Segmenter(),
            ),
        )
    ) as client:
        started = client.post("/api/session")
        assert started.status_code == 200
        issued = started.json()
        assert set(issued) == {"source_id", "memory_revision"}
        assert issued["source_id"].startswith("live:") and len(issued["source_id"]) == 37
        assert type(issued["memory_revision"]) is int and issued["memory_revision"] >= 0
        assert client.post("/api/session").json() == issued
        assert client.delete("/api/session").status_code == 204
        restarted = client.post("/api/session").json()
        assert set(restarted) == {"source_id", "memory_revision"}
        assert restarted["source_id"] != issued["source_id"]
        assert restarted["memory_revision"] >= issued["memory_revision"]

    with Memory(tmp_path, embedder=TinyEmbedder()) as reopened:
        assert reopened.get(retained_id).content.startswith("Existing memory remains retained")


def test_demo_opens_memory_at_startup_and_rejects_invalid_capture(tmp_path: Path) -> None:

    opened: list[AsyncMemory] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        memory = AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder()))
        opened.append(memory)
        return memory, Segmenter()

    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        assert len(opened) == 1
        assert client.get("/api/config").status_code == 200
        assert client.get("/api/memories").status_code == 200
        with client.websocket_connect("/api/media") as socket:
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()
            assert closed.value.code == 1008
        assert client.post("/api/chunks", content=b"old transport").status_code in {404, 405}
        assert (
            client.post("/api/search", json={"query": "x", "frame": "invalid"}).status_code == 422
        )
        assert client.post("/api/search", json={"query": 42}).status_code == 422
        assert client.delete("/api/session").status_code == 204


def test_demo_refuses_startup_with_an_unsupported_memory_schema(tmp_path: Path) -> None:

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        record = memory.add("Retained startup fixture")
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        connection.execute(f"PRAGMA user_version = {version - 1}")

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        return AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())), Segmenter()

    with (
        pytest.raises(StorageError) as caught,
        TestClient(create_app(_config(tmp_path), memory_factory=factory)),
    ):
        pytest.fail("the server must not accept requests without its memory runtime")
    assert caught.value.reason == "schema_unsupported"
    assert caught.value.stage == "open"
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == version - 1
        assert connection.execute("SELECT count(*) FROM memory_records").fetchone()[0] == 1
        connection.execute(f"PRAGMA user_version = {version}")
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        assert memory.get(record.id).content == "Retained startup fixture"


def test_entrypoint_bounds_connection_shutdown_before_draining_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    directory = tmp_path / "memory"
    config_path = tmp_path / "config.json"
    config_path.write_text(_config(directory).model_dump_json(), encoding="utf-8")
    launched: list[tuple[str, int, bool, int]] = []

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        return AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())), Segmenter()

    def run(
        app: FastAPI,
        *,
        host: str,
        port: int,
        access_log: bool,
        timeout_graceful_shutdown: int,
    ) -> None:
        launched.append((host, port, access_log, timeout_graceful_shutdown))
        with TestClient(app) as client:
            assert client.get("/api/memories").status_code == 200

    monkeypatch.setattr(demo, "open_event_memory", factory)
    monkeypatch.setattr(uvicorn, "run", run)
    assert demo.main(["--config", str(config_path), "--data-dir", str(directory)]) == 0
    assert launched == [("127.0.0.1", 8787, False, 5)]
    with Memory(directory, embedder=TinyEmbedder()) as memory:
        assert memory.list().items == ()


def test_empty_media_source_can_finish_without_a_model_call(tmp_path: Path) -> None:

    segmenter = Segmenter()
    with TestClient(
        create_app(
            _config(tmp_path),
            memory_factory=lambda config: (
                AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())),
                segmenter,
            ),
        )
    ) as client:
        assert client.post("/api/session").status_code == 200
        assert _upload(client, b"") == {"type": "accepted"}
        assert client.delete("/api/session").status_code == 204
    assert segmenter.calls == []


def test_model_backlog_does_not_block_stream_acceptance_stop_or_reconnect(tmp_path: Path) -> None:

    entered, release = Event(), Event()
    opened: list[ObservedMemory] = []

    class BlockingSegmenter(CompletingSegmenter):
        def segment_events(
            self,
            observations: Sequence[EventObservation],
            *,
            end_of_stream: bool = False,
        ) -> tuple[EventSegment, ...]:
            if not self.calls:
                entered.set()
                assert release.wait(15), "test did not release the segmentation model"
            return super().segment_events(observations, end_of_stream=end_of_stream)

    segmenter = BlockingSegmenter()

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        memory = ObservedMemory(Memory(config.data_dir, embedder=TinyEmbedder()))
        opened.append(memory)
        return memory, segmenter

    recording = _recording(seconds=30)
    with TestClient(create_app(_config(tmp_path), memory_factory=factory)) as client:
        try:
            assert client.post("/api/session").status_code == 200
            started = time.monotonic()
            assert _upload(client, recording, fragment_size=113) == {"type": "accepted"}
            assert time.monotonic() - started < 5
            assert entered.wait(2)
            assert not release.is_set()
            manifests = list((tmp_path / "live-events").glob("*.json"))
            assert len(manifests) == 1
            saved = json.loads(manifests[0].read_text())
            assert len(saved["pending"]) > 8
            assert all(
                blob["size_bytes"] > 0 for item in saved["pending"] for blob in item["media"]
            )
            assert client.get("/api/memories").json()["processing"] is True
            started = time.monotonic()
            assert client.delete("/api/session").status_code == 204
            assert client.post("/api/session").status_code == 200
            assert time.monotonic() - started < 2
            assert client.get("/api/memories").json()["processing"] is True
            assert len(opened) == 1
        finally:
            release.set()
        assert opened[0].finished.wait(5)
        _wait_until(lambda: not client.get("/api/memories").json()["processing"])
        items = client.get("/api/memories").json()["items"]
        assert items
        retained: list[tuple[str, bytes]] = []
        for item in items:
            for asset in item["assets"]:
                retained.extend(_packets(client.get(asset["url"]).content))
        assert Counter(retained) == Counter(_packets(recording))
        assert max(len(observations) for observations, _eof in segmenter.calls) == 4
        assert all(len(item["assets"]) <= 4 for item in items)


async def test_shutdown_finishes_deferred_capture_work(tmp_path: Path) -> None:

    runtime = LiveRuntime(
        _config(tmp_path),
        lambda config: (
            AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())),
            Segmenter(),
        ),
        "ws://realtime.example.test/v1/realtime",
    )
    memory = await runtime.open_memory()
    record = await memory.capture("Remember the blue toolbox.")
    runtime.work.set()
    await asyncio.wait_for(runtime.close(), timeout=5)
    with Memory(tmp_path, embedder=TinyEmbedder()) as reopened:
        assert reopened.get(record.id).content == "Remember the blue toolbox."
        assert not reopened.pending_captures()


def test_failed_processing_is_visible_without_provider_exception_bodies(tmp_path: Path) -> None:

    class FailingEmbedder(TinyEmbedder):
        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            raise ModelError("private provider body", reason="rate_limited")

    memory = Memory(tmp_path, embedder=FailingEmbedder())
    memory.capture("An observation waiting to become searchable.")
    observed = ObservedMemory(memory)
    with TestClient(
        create_app(_config(tmp_path), memory_factory=lambda _config: (observed, Segmenter()))
    ) as client:
        # Startup schedules one attempt; starting a call would schedule another retry.
        assert observed.finished.wait(5)
        response = client.get("/api/memories")
        card = response.json()["items"][0]
        assert card["pending"] is True
        assert card["error"] == {
            "code": "processing_failed",
            "attempts": 1,
            "awaiting": "enrichment",
        }
        assert "private provider body" not in response.text


def test_historical_checkpoints_are_untouched_while_new_calls_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    recording = _recording()
    with TestClient(
        create_app(
            _config(tmp_path),
            memory_factory=lambda config: (
                AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())),
                Segmenter(),
            ),
        )
    ) as client:
        assert client.post("/api/session").status_code == 200
        assert _upload(client, recording) == {"type": "accepted"}
        assert client.delete("/api/session").status_code == 204
    directory = tmp_path / "live-events"
    assert len(list(directory.glob("*.json"))) == 1
    (directory / "invalid.json").write_bytes(b"not a checkpoint")
    historical = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    read_bytes = Path.read_bytes

    def read_current_file(path: Path) -> bytes:
        assert path not in historical, "the demo must not read historical event checkpoints"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_current_file)
    segmenter: Segmenter = CompletingSegmenter()
    opened: list[ObservedMemory] = []

    def reopen(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        memory = ObservedMemory(Memory(config.data_dir, embedder=TinyEmbedder()))
        opened.append(memory)
        return memory, segmenter

    with TestClient(create_app(_config(tmp_path), memory_factory=reopen)) as client:
        assert client.get("/api/memories").json()["items"] == []
        assert segmenter.calls == []
        assert client.post("/api/session").status_code == 200
        response = client.get("/api/memories").json()
        assert "recovery_failed" not in response
        assert response["processing"] is False
        assert segmenter.calls == []
        assert _upload(client, recording) == {"type": "accepted"}
        assert opened[0].finished.wait(5)
        items = client.get("/api/memories").json()["items"]
        assert len(items) == 1
        asset = client.get(items[0]["assets"][0]["url"])
        assert Counter(_packets(asset.content)) == Counter(_packets(recording))
        assert segmenter.calls[-1][1] is True
        assert client.delete("/api/session").status_code == 204
    assert {path: read_bytes(path) for path in historical} == historical

    saved = items[0]["id"]
    segmenter = Segmenter()
    with TestClient(create_app(_config(tmp_path), memory_factory=reopen)) as client:
        assert [item["id"] for item in client.get("/api/memories").json()["items"]] == [saved]
        assert client.post("/api/session").status_code == 200
        assert segmenter.calls == []
    assert {path: read_bytes(path) for path in historical} == historical


def test_incomplete_event_is_retained_after_media_eof(tmp_path: Path) -> None:

    segmenter = Segmenter()
    with TestClient(
        create_app(
            _config(tmp_path),
            memory_factory=lambda config: (
                AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())),
                segmenter,
            ),
        )
    ) as client:
        assert client.post("/api/session").status_code == 200
        assert _upload(client, _recording()) == {"type": "accepted"}
        _wait_until(lambda: not client.get("/api/memories").json()["processing"])
        assert client.get("/api/memories").json()["items"] == []
        assert segmenter.calls[-1][1] is True
    checkpoint = json.loads(next((tmp_path / "live-events").glob("*.json")).read_text())
    assert checkpoint["streams"][0]["chunks"]


def test_native_capacity_failure_is_a_socket_error_and_not_an_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    def bounded_stream(
        memory: AsyncMemory,
        *,
        segmenter: EventSegmentationBackend,
        max_chunks: int,
        capture: bool,
        context: ObservationContext | None = None,
        checkpoint_path: Path,
    ) -> AsyncEventStream:
        return AsyncEventStream(
            memory,
            segmenter=segmenter,
            max_chunks=max_chunks,
            capture=capture,
            context=context,
            checkpoint_path=checkpoint_path,
            max_pending_bytes=1,
        )

    monkeypatch.setattr(live_app, "AsyncEventStream", bounded_stream)
    with TestClient(
        live_app.create_app(
            _config(tmp_path),
            memory_factory=lambda config: (
                AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())),
                Segmenter(),
            ),
        )
    ) as client:
        assert client.post("/api/session").status_code == 200
        with client.websocket_connect("/api/media") as socket:
            socket.send_json({"type": "start", "started_at": "2026-11-16T10:00:00Z"})
            socket.send_bytes(_recording())
            socket.send_json({"type": "end"})
            assert socket.receive_json() == {"type": "error", "code": "capacity_exceeded"}
            with pytest.raises(WebSocketDisconnect):
                socket.receive_json()
