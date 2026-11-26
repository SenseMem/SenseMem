"""Independent visual and sensory workers preserve native input and voice priority."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import struct
import wave
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Self, cast

import pytest
from _feature_support import TinyEmbedder
from fastapi import WebSocket, WebSocketDisconnect
from fixtures.live import Segmenter
from fixtures.live import live_config as _config
from PIL import Image
from websockets.asyncio.client import ClientConnection

from examples.live import vision
from examples.live import vision as module
from examples.live.app import LiveRuntime
from examples.live.perception import PerceptionContext, PerceptionResult
from examples.live.realtime import RealtimeRelay
from examples.live.recall import LiveRecall
from examples.live.vision import LiveVision
from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AsyncMemory,
    Blob,
    ContentInput,
    ContextBudget,
    Memory,
    Modality,
    ModelError,
    ModelInput,
    PrefetchResult,
    SearchHit,
)


class Socket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.outgoing: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.sent: list[dict[str, object]] = []
        self.codes: list[int] = []

    async def accept(self) -> None:
        pass

    async def receive_text(self) -> str:
        value = await self.incoming.get()
        if value is None:
            raise WebSocketDisconnect()
        return value

    async def send_json(self, event: dict[str, object]) -> None:
        self.sent.append(event)
        self.outgoing.put_nowait(event)

    async def close(self, code: int) -> None:
        self.codes.append(code)

    def push(self, event: dict[str, object]) -> None:
        self.incoming.put_nowait(json.dumps(event))


class Reviewer:
    generation_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()
        self.inputs: list[ModelInput] = []
        self.originals: list[tuple[bytes, ...]] = []
        self.invalid = False
        self.closed = False

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        assert answer_policy == "best_effort" and not hits
        self.inputs.append(question)
        self.originals.append(
            tuple(cast(Path, asset.path).read_bytes() for asset in question.assets)
        )
        self.entered.set()
        assert self.release.wait(3), "test must release the model worker"
        if self.invalid:
            return AnswerResult("private malformed provider output")
        return AnswerResult(
            json.dumps(
                {
                    "summary": "The observed cup is blue.",
                    "changed": False,
                    "action": "context",
                    "utterance": None,
                    "confidence": 0.95,
                    "evidence": [
                        {
                            "sequence": int(question.assets[0].id.partition(":")[2]),
                            "observation": "A blue cup is visible.",
                        }
                    ],
                    "reason": "Only update context.",
                    "contradicted_claim": None,
                    "correction": None,
                }
            )
        )

    def close(self) -> None:
        self.closed = True


class Queries:
    def __init__(self, result: PrefetchResult) -> None:
        self.result = result
        self.inputs: list[ContentInput] = []
        self.started: asyncio.Queue[ContentInput] = asyncio.Queue()
        self.release = asyncio.Event()
        self.active = self.max_active = self.closed = 0
        self.error: ModelError | None = None

    def helper(self, memory: AsyncMemory, *, budget: ContextBudget) -> Query:
        assert budget.max_media_items == 8
        return Query(self)


class Query:
    def __init__(self, owner: Queries) -> None:
        self.owner = owner
        self.snapshot: ContentInput = ""

    def submit(self, query: ContentInput) -> int:
        self.snapshot = query
        return 1

    async def finalize(self) -> PrefetchResult:
        owner = self.owner
        owner.inputs.append(self.snapshot)
        owner.started.put_nowait(self.snapshot)
        owner.active += 1
        owner.max_active = max(owner.max_active, owner.active)
        try:
            await owner.release.wait()
            if owner.error is not None:
                raise owner.error
            return owner.result
        finally:
            owner.active -= 1

    async def close(self) -> None:
        self.owner.closed += 1


@dataclass
class Fixture:
    memory: AsyncMemory
    vision: LiveVision
    recall: LiveRecall
    reviewer: Reviewer
    queries: Queries
    messages: asyncio.Queue[dict[str, object]]
    socket: Socket
    start: datetime

    def frame(self, sequence: int) -> dict[str, object]:
        encoded = io.BytesIO()
        Image.new("RGB", (8, 8), (sequence, 60, 160)).save(encoded, "JPEG")
        return {
            "type": "frame",
            "sequence": sequence,
            "timestamp_ms": sequence * 100,
            "captured_at": (self.start + timedelta(milliseconds=sequence * 100)).isoformat(),
            "image": "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode(),
        }


@pytest.fixture
async def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Fixture]:

    async with AsyncMemory(Memory(tmp_path / "memory", embedder=TinyEmbedder())) as memory:
        bundle = await memory.compile("fixture")
        queries = Queries(PrefetchResult(hits=bundle.hits, bundle=bundle))
        monkeypatch.setattr(vision, "AsyncOmniPrefetch", queries.helper)
        recall = LiveRecall(memory)
        messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        recall.sender = messages.put
        reviewer = Reviewer()
        worker = vision.LiveVision(memory, reviewer, recall, "live:fixture")
        start = datetime.now(timezone.utc) - timedelta(seconds=3)
        value = Fixture(memory, worker, recall, reviewer, queries, messages, Socket(), start)
        try:
            yield value
        finally:
            reviewer.release.set()
            queries.release.set()
            await worker.close()
            await recall.close()


async def _message(
    queue: asyncio.Queue[dict[str, object]], event_type: str, *, status: str | None = None
) -> dict[str, object]:
    while True:
        message = await asyncio.wait_for(queue.get(), 3)
        if message["type"] == event_type and (status is None or message.get("status") == status):
            return message


async def _start(fixture: Fixture) -> asyncio.Task[None]:
    task = asyncio.create_task(fixture.vision.serve(cast(WebSocket, fixture.socket)))
    fixture.socket.push({"type": "start", "started_at": fixture.start.isoformat()})
    await _message(fixture.socket.outgoing, "ready")
    return task


async def test_slow_sensory_recall_preserves_source_time_without_reviving_expired_vision(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:

    f = fixture
    clock = datetime.now(timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz: object = None) -> Self:
            return cls.fromisoformat(clock.isoformat())

    monkeypatch.setattr(module, "datetime", Clock)
    task = await _start(f)
    try:
        f.socket.push(f.frame(1))
        await _message(f.socket.outgoing, "ack")
        await asyncio.wait_for(f.queries.started.get(), 3)
        f.vision.observation = {"expires_at": (clock + timedelta(seconds=1)).isoformat()}
        clock += timedelta(seconds=30)
        f.queries.release.set()
        ready = await _message(f.messages, "sensory_recall", status="ready")
        assert ready["occurred_end"] == f.frame(1)["captured_at"]
        assert datetime.fromisoformat(str(ready["expires_at"])) > clock
        context = json.loads(str(ready["context"]))
        assert context["temporary_visual_observation"] is None
        historical = json.loads(context["historical_memory"])
        assert historical["input_occurred_end"] == ready["occurred_end"]
        assert (await f.memory.list()).items == ()
    finally:
        f.reviewer.release.set()
        f.socket.incoming.put_nowait(None)
        await task


async def test_ack_and_native_workers_progress_independently_without_persisting_frames(
    fixture: Fixture,
) -> None:
    f = fixture
    before = (await f.memory.list()).items
    task = await _start(f)
    try:
        f.socket.push(f.frame(1))
        assert (await _message(f.socket.outgoing, "ack"))["sequence"] == 1
        await asyncio.wait_for(f.queries.started.get(), 3)
        assert await asyncio.to_thread(f.reviewer.entered.wait, 3)
        for sequence in range(2, 10):
            f.socket.push(f.frame(sequence))
            assert (await _message(f.socket.outgoing, "ack"))["sequence"] == sequence
        assert len(f.reviewer.inputs) == len(f.queries.inputs) == 1
        assert f.queries.max_active == 1
        assert f.vision.perception.pending[-1].sequence == 9
        f.queries.release.set()
        ready = await _message(f.messages, "sensory_recall", status="ready")
        assert ready["frame_sequences"] == [1], "completed older snapshots must remain useful"
        assert ready["occurred_end"] == f.frame(1)["captured_at"]
        latest = await _message(f.messages, "sensory_recall", status="ready")
        assert latest["frame_sequences"] == [9]
        assert len(f.queries.inputs) == 2 and f.queries.max_active == 1
        f.reviewer.release.set()
        first = await _message(f.messages, "perception")
        assert first["frame_sequences"] == [1] and first["current"] is False
        second = await _message(f.messages, "perception")
        assert cast(list[int], second["frame_sequences"])[-1] == 9
        original = base64.b64decode(cast(str, f.frame(1)["image"]).partition(",")[2])
        assert f.reviewer.originals[0] == (original,)
        assert isinstance(f.queries.inputs[0], tuple)
        first_query = cast(tuple[str | Blob, ...], f.queries.inputs[0])
        assert isinstance(first_query[-1], Blob) and first_query[-1].data == original
        assert (await f.memory.list()).items == before
    finally:
        f.socket.incoming.put_nowait(None)
        await task


@pytest.mark.parametrize("invalid", ["clock", "sequence", "encoding", "schema"])
async def test_frame_clock_sequence_and_schema_fail_with_safe_error(
    fixture: Fixture, invalid: str
) -> None:
    f = fixture
    task = await _start(f)
    frame = f.frame(1)
    if invalid == "sequence":
        f.socket.push(frame)
        await _message(f.socket.outgoing, "ack")
    elif invalid == "clock":
        frame["timestamp_ms"] = 1000
    elif invalid == "encoding":
        frame["image"] = "data:image/jpeg;base64,not-base64"
    else:
        frame["provider_body"] = "must not echo this input"
    f.socket.push(frame)
    await asyncio.wait_for(task, 3)
    assert f.socket.sent[-1] == {"type": "error", "code": "invalid_perception_input"}
    assert f.socket.codes == [1008]


async def test_pcm_snapshots_use_exact_original_samples_and_bounded_native_audio_query(
    fixture: Fixture,
) -> None:
    f = fixture
    f.vision.started_at = f.start
    packet = (8192).to_bytes(2, "little", signed=True) * 320
    for sequence in range(1, 13):
        f.vision.append_audio(packet, sequence, sequence * 20)
    assert f.vision.task is None, "less than 4000 new samples do not dispatch a query"
    f.vision.append_audio(packet, 13, 260)
    first = cast(tuple[str | Blob, ...], await asyncio.wait_for(f.queries.started.get(), 3))
    assert isinstance(first[-1], Blob) and first[-1].media_type == "audio/wav"
    with wave.open(io.BytesIO(first[-1].data)) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        assert audio.readframes(audio.getnframes()) == packet * 13
    for sequence in range(14, 114):
        f.vision.append_audio(packet, sequence, sequence * 20)
    assert f.vision.audio_bytes == 64_000
    assert len(f.queries.inputs) == 1 and f.queries.max_active == 1
    f.queries.release.set()
    ready = await _message(f.messages, "sensory_recall", status="ready")
    assert ready["frame_sequences"] == []
    last = await _message(f.messages, "sensory_recall", status="ready")
    assert last["occurred_end"] == (f.start + timedelta(milliseconds=2260)).isoformat()
    assert len(f.queries.inputs) == 2 and f.queries.max_active == 1
    query = cast(tuple[str | Blob, ...], f.queries.inputs[-1])
    assert isinstance(query[-1], Blob)
    with wave.open(io.BytesIO(query[-1].data)) as audio:
        assert audio.readframes(audio.getnframes()) == packet * 100
    assert (await f.memory.list()).items == ()


async def test_audio_tap_forwards_original_protocol_before_observation(fixture: Fixture) -> None:

    f = fixture
    f.vision.started_at = f.start
    expected = struct.pack("<5h", 8192, -8192, 32767, -32768, 0) * 3200
    raw = json.dumps(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(expected).decode(),
            "event_id": "av_1_1000.0",
            "custom_extension": "kept verbatim",
        }
    )
    trace: list[tuple[str, object]] = []

    class Input:
        def __init__(self) -> None:
            self.messages = iter(
                (
                    {"type": "websocket.receive", "text": raw},
                    {"type": "websocket.disconnect"},
                )
            )

        async def receive(self) -> dict[str, str]:
            return next(self.messages)

    class Connection:
        async def send(self, value: str) -> None:
            trace.append(("forward", value))

    def observe(data: bytes, sequence: int, timestamp_ms: float) -> None:
        assert trace == [("forward", raw)]
        trace.append(("tap", (data, sequence, timestamp_ms)))
        f.vision.append_audio(data, sequence, timestamp_ms)

    relay = RealtimeRelay(cast(WebSocket, Input()), observe)
    await relay._send(cast(ClientConnection, Connection()))
    assert trace == [("forward", raw), ("tap", (expected, 1, 1000.0))]
    query = cast(tuple[str | Blob, ...], await asyncio.wait_for(f.queries.started.get(), 3))
    assert isinstance(query[-1], Blob)
    with wave.open(io.BytesIO(query[-1].data)) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        assert audio.getnframes() == 16000
        assert audio.readframes(audio.getnframes()) == expected


async def test_camera_reset_suppresses_both_inflight_outputs_without_erasing_issued_provenance(
    fixture: Fixture,
) -> None:
    f = fixture
    issued: dict[str, dict[str, object]] = {
        "spoken": {"candidate_id": "spoken", "source_id": f.vision.source_id}
    }
    f.vision.issued = issued
    task = await _start(f)
    f.socket.push(f.frame(1))
    await _message(f.socket.outgoing, "ack")
    await asyncio.wait_for(f.queries.started.get(), 3)
    assert await asyncio.to_thread(f.reviewer.entered.wait, 3)
    f.socket.push({"type": "camera", "enabled": False})
    await _message(f.messages, "perception_reset")
    f.socket.incoming.put_nowait(None)
    await task
    f.queries.release.set()
    f.reviewer.release.set()
    await f.vision.close()
    assert f.vision.issued is issued and "spoken" in issued
    assert f.messages.empty(), "camera epoch/close must suppress old query and model results"


async def test_memory_revision_and_expired_observations_cannot_issue_actions(
    fixture: Fixture,
) -> None:

    f = fixture
    now = datetime.now(timezone.utc)
    data: dict[str, object] = {
        "summary": "The key moved into the bowl.",
        "changed": True,
        "action": "wait",
        "utterance": "刚才钥匙放进碗里了。",
        "confidence": 0.95,
        "evidence": (
            {"sequence": 1, "observation": "Key in tray"},
            {"sequence": 2, "observation": "Key in bowl"},
        ),
        "reason": "Concrete recent placement",
        "contradicted_claim": None,
        "correction": None,
        "source_id": f.vision.source_id,
        "frame_sequences": (1, 2),
        "occurred_at": now - timedelta(seconds=2),
        "occurred_end": now - timedelta(seconds=1),
        "timestamp_ms": 1000.0,
        "end_timestamp_ms": 2000.0,
        "context": PerceptionContext(memory_revision=0),
        "target_response_id": None,
        "current": False,
        "latest_sequence": 3,
        "expires_at": now + timedelta(seconds=7),
        "epoch": 0,
    }
    result = PerceptionResult.model_validate(data)
    f.vision.invalidate(1)
    await f.vision._observed(result)
    assert (await f.messages.get())["type"] == "perception"
    while not f.messages.empty():
        assert f.messages.get_nowait()["type"] != "visual_response"
    assert not f.vision.issued
    expired = result.model_copy(
        update={
            "context": PerceptionContext(memory_revision=1),
            "expires_at": now - timedelta(seconds=1),
        }
    )
    await f.vision._observed(expired)
    assert (await f.messages.get())["type"] == "perception"
    expired_context = await _message(f.messages, "visual_context")
    assert json.loads(str(expired_context["context"]))["temporary_visual_observation"] is None
    while not f.messages.empty():
        assert f.messages.get_nowait()["type"] != "visual_response"
    assert not f.vision.issued
    valid = result.model_copy(update={"context": PerceptionContext(memory_revision=1)})
    await f.vision._observed(valid)
    assert (await f.messages.get())["type"] == "perception"
    candidate = await _message(f.messages, "visual_response")
    assert candidate["type"] == "visual_response" and candidate["memory_revision"] == 1
    assert json.loads(str(candidate["context"]))["temporary_visual_observation"] is not None
    assert candidate["candidate_id"] in f.vision.issued


async def test_voice_disconnect_returns_while_native_visual_work_drains(
    fixture: Fixture, tmp_path: Path
) -> None:

    f = fixture
    runtime = LiveRuntime(
        _config(tmp_path / "unused"),
        lambda _: (f.memory, Segmenter()),
        "wss://realtime.example.test/v1/realtime",
    )
    task = await _start(f)
    f.socket.push(f.frame(1))
    await _message(f.socket.outgoing, "ack")
    await asyncio.wait_for(f.queries.started.get(), 3)
    assert await asyncio.to_thread(f.reviewer.entered.wait, 3)
    runtime.vision, runtime.recall = f.vision, f.recall
    relay = runtime.realtime = RealtimeRelay(cast(WebSocket, f.socket))
    try:
        await asyncio.wait_for(runtime.disconnect(), 1)
        assert relay.stopping and runtime.realtime is None and runtime.vision is None
        assert runtime.recall_closing and not f.queries.release.is_set()
        assert not f.reviewer.release.is_set()
    finally:
        f.socket.incoming.put_nowait(None)
        await task
        f.queries.release.set()
        f.reviewer.release.set()
        await asyncio.gather(*runtime.recall_closing)
    assert f.reviewer.closed is False, "the shared model backend remains owned by the runtime"


async def test_native_failures_and_invalid_model_output_expose_only_safe_categories(
    fixture: Fixture,
) -> None:
    f = fixture
    f.queries.error = ModelError("private provider response body", reason="timeout")
    f.queries.release.set()
    f.reviewer.invalid = True
    f.reviewer.release.set()
    task = await _start(f)
    try:
        f.socket.push(f.frame(1))
        await _message(f.socket.outgoing, "ack")
        events: list[dict[str, object]] = []
        while not (
            any(
                event["type"] == "sensory_recall" and event.get("status") == "error"
                for event in events
            )
            and any(event["type"] == "perception_error" for event in events)
        ):
            events.append(await asyncio.wait_for(f.messages.get(), 3))
        sensory = next(event for event in events if event.get("status") == "error")
        assert "context" not in sensory and "error" not in sensory
        failure = next(event for event in events if event["type"] == "perception_error")
        assert failure == {
            "type": "perception_error",
            "source_id": "live:fixture",
            "code": "invalid_output",
        }
        assert not f.vision.issued
    finally:
        f.socket.incoming.put_nowait(None)
        await task
