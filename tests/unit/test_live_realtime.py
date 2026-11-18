"""The same-origin relay preserves s2s events and owns one upstream session."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest
from _feature_support import TinyEmbedder
from fastapi import WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect, WebSocketState
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from sensemem import AsyncMemory, EventObservation, EventSegment, Memory, MemorySpec


@pytest.fixture(autouse=True)
def example_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))


class Segmenter:
    event_model = "relay-fixture"
    event_recipe = "relay-fixture:v1"

    def segment_events(
        self, observations: Sequence[EventObservation], *, end_of_stream: bool = False
    ) -> tuple[EventSegment, ...]:
        return ()


def _memory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
    return AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())), Segmenter()


@contextmanager
def _native_server(handler: Callable[[ServerConnection], None]) -> Iterator[str]:
    with serve(handler, "127.0.0.1", 0) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}/v1/realtime"
        finally:
            server.shutdown()
            thread.join(5)
            assert not thread.is_alive()


def test_native_json_audio_video_and_disconnect_survive_the_local_relay(tmp_path: Path) -> None:
    from examples.live.app import create_app

    received: list[str] = []
    paths: list[str] = []
    released = Event()
    caption = '{"type":"response.output_audio_transcript.delta","delta":"你好"}'
    audio = '{"type":"response.output_audio.delta","delta":"AAAAAA=="}'

    def native(socket: ServerConnection) -> None:
        assert socket.request is not None
        paths.append(socket.request.path)
        assert socket.request.headers.get("Authorization") is None
        socket.send('{"type":"session.created","session":{"id":"native-fixture"}}')
        try:
            for text in socket:
                assert isinstance(text, str)
                received.append(text)
                if json.loads(text)["type"] == "session.update":
                    socket.send('{"type":"session.updated","session":{"id":"native-fixture"}}')
                    continue
                socket.send(caption)
                socket.send(audio)
        except ConnectionClosed:
            pass
        finally:
            released.set()

    with _native_server(native) as url:
        config = MemorySpec.model_validate(
            {"data_dir": tmp_path, "embedding": {"provider": "openai"}}
        )
        with TestClient(
            create_app(
                config,
                memory_factory=_memory,
                realtime_url=url,
            )
        ) as client:
            assert client.post("/api/session").status_code == 200
            with client.websocket_connect("/api/realtime") as socket:
                assert socket.receive_json()["type"] == "session.created"
                initialize = json.dumps(
                    {
                        "type": "session.update",
                        "session": client.get("/api/config").json()["session"],
                    }
                )
                socket.send_text(initialize)
                assert socket.receive_json() == {
                    "type": "session.updated",
                    "session": {"id": "native-fixture"},
                }
                update = '{ "type":"input_audio_buffer.append", "audio":"AAAAAA==" }'
                socket.send_text(update)
                assert socket.receive_text() == caption
                assert socket.receive_text() == audio
                image = json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": "data:image/jpeg;base64,fixture",
                                }
                            ],
                        },
                    }
                )
                socket.send_text(image)
                assert socket.receive_text() == caption
                assert socket.receive_text() == audio
            assert released.wait(5)
            assert client.delete("/api/session").status_code == 204
    assert paths == ["/v1/realtime"]
    assert received == [initialize, update, image]


def test_stop_releases_only_the_owned_connection_and_allows_reconnect(tmp_path: Path) -> None:
    from examples.live.app import create_app

    released: list[Event] = []

    def native(socket: ServerConnection) -> None:
        ended = Event()
        released.append(ended)
        socket.send('{"type":"session.created"}')
        try:
            for text in socket:
                socket.send(text)
        except ConnectionClosed:
            pass
        finally:
            ended.set()

    with _native_server(native) as url:
        config = MemorySpec.model_validate(
            {"data_dir": tmp_path, "embedding": {"provider": "openai"}}
        )
        with TestClient(
            create_app(
                config,
                memory_factory=_memory,
                realtime_url=url,
            )
        ) as client:
            with client.websocket_connect("/api/realtime") as unstarted:
                with pytest.raises(WebSocketDisconnect) as rejected:
                    unstarted.receive_json()
                assert rejected.value.code == 1008
            client.post("/api/session")
            with client.websocket_connect("/api/realtime") as first:
                assert first.receive_json()["type"] == "session.created"
                with client.websocket_connect("/api/realtime") as duplicate:
                    with pytest.raises(WebSocketDisconnect) as rejected:
                        duplicate.receive_json()
                    assert rejected.value.code == 1008
                first.send_text('{"type":"session.close"}')
                assert first.receive_json()["type"] == "session.close"
                assert client.delete("/api/session").status_code == 204
                assert released[0].wait(5)
                with pytest.raises(WebSocketDisconnect):
                    first.receive_json()
                client.post("/api/session")
                with client.websocket_connect("/api/realtime") as second:
                    assert second.receive_json()["type"] == "session.created"
                    second.send_text('{"type":"input_audio_buffer.append","audio":"AAAAAA=="}')
                    assert second.receive_json()["audio"] == "AAAAAA=="
                    assert client.delete("/api/session").status_code == 204
                    assert released[1].wait(5)
    assert len(released) == 2


@pytest.mark.parametrize("kind", ["binary", "oversized"])
def test_relay_bounds_browser_messages_without_forwarding_them(tmp_path: Path, kind: str) -> None:
    from examples.live.app import create_app
    from examples.live.realtime import MAX_MESSAGE_BYTES

    received: list[str | bytes] = []

    def native(socket: ServerConnection) -> None:
        socket.send('{"type":"session.created"}')
        with suppress(ConnectionClosed):
            received.extend(socket)

    with _native_server(native) as url:
        config = MemorySpec.model_validate(
            {"data_dir": tmp_path, "embedding": {"provider": "openai"}}
        )
        with TestClient(
            create_app(
                config,
                memory_factory=_memory,
                realtime_url=url,
            )
        ) as client:
            client.post("/api/session")
            with client.websocket_connect("/api/realtime") as socket:
                socket.receive_json()
                if kind == "binary":
                    socket.send_bytes(b"unframed PCM")
                else:
                    socket.send_text("a" * (MAX_MESSAGE_BYTES + 1))
                with pytest.raises(WebSocketDisconnect) as ended:
                    socket.receive_json()
                assert ended.value.code == (1003 if kind == "binary" else 1009)
    assert received == []


def test_upstream_failure_is_safe_and_leaves_memory_available(tmp_path: Path) -> None:
    from examples.live.app import create_app

    def native(socket: ServerConnection) -> None:
        socket.close(code=1011, reason="private provider body and credential")

    with _native_server(native) as url:
        config = MemorySpec.model_validate(
            {"data_dir": tmp_path, "embedding": {"provider": "openai"}}
        )
        with TestClient(
            create_app(
                config,
                memory_factory=_memory,
                realtime_url=url,
            )
        ) as client:
            client.post("/api/session")
            with client.websocket_connect("/api/realtime") as socket:
                failure = socket.receive_json()
                assert failure["type"] == "error"
                assert failure["error"]["code"] == "realtime_unavailable"
                assert "private" not in str(failure)
                with pytest.raises(WebSocketDisconnect) as ended:
                    socket.receive_json()
                assert ended.value.code == 1011
            assert client.delete("/api/session").status_code == 204
            assert client.get("/api/memories").status_code == 200


def test_local_realtime_needs_no_cloud_key(tmp_path: Path) -> None:
    from examples.live.app import create_app

    config = MemorySpec.model_validate({"data_dir": tmp_path, "embedding": {"provider": "openai"}})
    with TestClient(create_app(config, memory_factory=_memory)) as client:
        assert client.get("/api/config").json()["realtime_ready"] is True
        assert client.get("/api/memories").status_code == 200


async def test_browser_disconnect_during_final_close_releases_upstream_without_raising() -> None:
    from examples.live.realtime import RealtimeRelay

    released = Event()

    def native(socket: ServerConnection) -> None:
        try:
            with suppress(ConnectionClosed):
                for _message in socket:
                    pass
        finally:
            released.set()

    class DisconnectedBrowser:
        application_state = WebSocketState.CONNECTED
        close_code: int | None = None

        async def receive(self) -> dict[str, object]:
            return {"type": "websocket.disconnect", "code": 1006}

        async def close(self, code: int = 1000) -> None:
            self.close_code = code
            raise WebSocketDisconnect(1006)

    browser = DisconnectedBrowser()
    relay = RealtimeRelay(cast(WebSocket, browser))
    with _native_server(native) as url:
        await relay.serve(url)
        assert released.wait(5)
    assert browser.close_code == 1000
    assert relay.task is not None and relay.task.done()


def test_native_pcm_audio_observation_retains_clock_and_leaves_16k_unchanged() -> None:
    from examples.live.realtime import RealtimeRelay

    observed: list[tuple[bytes, int, float]] = []
    relay = RealtimeRelay(cast(WebSocket, object()), lambda *values: observed.append(values))
    data = struct.pack("<5h", -32768, -16384, 0, 16384, 32767)
    audio = base64.b64encode(data).decode()
    relay._observe_audio(
        json.dumps({"type": "input_audio_buffer.append", "event_id": "av_17_240.5", "audio": audio})
    )
    assert len(observed) == 1
    pcm, sequence, timestamp = observed[0]
    assert pcm == data
    assert (sequence, timestamp) == (17, 240.5)

    for invalid in (
        "not base64",
        "AA==",
    ):
        relay._observe_audio(
            json.dumps(
                {"type": "input_audio_buffer.append", "event_id": "av_18_260", "audio": invalid}
            )
        )
    relay._observe_audio(json.dumps({"type": "input_audio_buffer.append", "audio": audio}))
    assert len(observed) == 1


def test_24k_pcm_side_path_resamples_continuously_across_packet_boundaries() -> None:
    from examples.live.realtime import RealtimeRelay

    observed: list[tuple[bytes, int, float]] = []
    relay = RealtimeRelay(cast(WebSocket, object()), lambda *values: observed.append(values))
    relay._observe_audio(
        json.dumps(
            {
                "type": "session.update",
                "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}},
            }
        )
    )
    for sequence, values in enumerate(((0,), (100, 200), (300, 400), (500,))):
        relay._observe_audio(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"av_{sequence}_{sequence * 20}",
                    "audio": base64.b64encode(struct.pack(f"<{len(values)}h", *values)).decode(),
                }
            )
        )
    joined = b"".join(values[0] for values in observed)
    assert struct.unpack(f"<{len(joined) // 2}h", joined) == (0, 150, 300, 450)
    assert [values[1:] for values in observed] == [(0, 0), (1, 20), (2, 40), (3, 60)]


@pytest.mark.asyncio
async def test_native_forwarding_precedes_original_user_audio_and_server_finality_observers() -> (
    None
):
    from examples.live.realtime import RealtimeRelay
    from examples.live.user_audio import UserAudioCapture

    order: list[tuple[str, object]] = []
    pcm = struct.pack("<3h", -300, 0, 500)
    packet = json.dumps(
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
    )
    final = json.dumps(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input-1",
            "transcript": "原始用户语音",
        }
    )

    class Capture(UserAudioCapture):
        def __init__(self) -> None:
            pass

        def append_audio(self, data: bytes, sample_rate: int) -> None:
            order.append(("pcm", (data, sample_rate)))

        def observe_event(self, text: str) -> None:
            order.append(("final", text))

    class Browser:
        def __init__(self) -> None:
            self.messages = iter(
                [
                    {"type": "websocket.receive", "text": packet},
                    {"type": "websocket.disconnect"},
                ]
            )

        async def receive(self) -> dict[str, str]:
            return next(self.messages)

        async def send_text(self, text: str) -> None:
            order.append(("browser", text))

    class Native:
        async def send(self, text: str) -> None:
            order.append(("native", text))

        async def recv(self) -> str:
            if any(kind == "final" for kind, _value in order):
                raise asyncio.CancelledError
            return final

    relay = RealtimeRelay(cast(WebSocket, Browser()), user_audio=Capture())
    relay.input_rate = 24_000
    # The native protocol has no required client event_id; original user capture still works.
    from websockets.asyncio.client import ClientConnection

    connection = cast(ClientConnection, Native())
    await relay._send(connection)
    with pytest.raises(asyncio.CancelledError):
        await relay._receive(connection)
    assert order == [
        ("native", packet),
        ("pcm", (pcm, 24_000)),
        ("browser", final),
        ("final", final),
    ]
