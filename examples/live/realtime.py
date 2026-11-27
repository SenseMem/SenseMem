"""Relay the local speech-to-speech protocol and observe its audio side path."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import struct
from collections.abc import Callable
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosedOK, WebSocketException

from examples.live.user_audio import UserAudioCapture

MAX_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_AUDIO_BASE64_CHARS = 4 * ((24_000 * 2 + 2) // 3)
DEFAULT_REALTIME_URL = "ws://127.0.0.1:8765/v1/realtime"
REALTIME_MAX_QUEUE = 8
REALTIME_OPEN_TIMEOUT_SECONDS = 30
REALTIME_CLOSE_TIMEOUT_SECONDS = 5


class RealtimeRelay:
    """Own a bounded pair of pumps and release the native session on either disconnect."""

    def __init__(
        self,
        socket: WebSocket,
        observe_audio: Callable[[bytes, int, float], None] | None = None,
        *,
        user_audio: UserAudioCapture | None = None,
    ) -> None:
        self.socket = socket
        self.observe_audio = observe_audio
        self.user_audio = user_audio
        self.task: asyncio.Task[None] | None = None
        self.stopping = False
        self.close_code = 1000
        self.input_rate = 16_000
        self.samples: list[int] = []
        self.sample_position = 0.0

    async def serve(self, url: str) -> None:
        self.task = asyncio.create_task(self._relay(url))
        try:
            await self.task
        except asyncio.CancelledError:
            if not self.stopping:
                raise

    async def close(self) -> None:
        self.stopping = True
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def _relay(self, url: str) -> None:
        try:
            async with connect(
                url,
                max_size=MAX_MESSAGE_BYTES,
                max_queue=REALTIME_MAX_QUEUE,
                open_timeout=REALTIME_OPEN_TIMEOUT_SECONDS,
                close_timeout=REALTIME_CLOSE_TIMEOUT_SECONDS,
            ) as connection:
                pumps = (
                    asyncio.create_task(self._send(connection)),
                    asyncio.create_task(self._receive(connection)),
                )
                try:
                    done, _pending = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                finally:
                    for task in pumps:
                        task.cancel()
                    await asyncio.gather(*pumps, return_exceptions=True)
        except (WebSocketDisconnect, ConnectionClosedOK):
            pass
        except (WebSocketException, OSError, TimeoutError, UnicodeDecodeError):
            self.close_code = 1011
            await self.socket.send_json(
                {
                    "type": "error",
                    "error": {
                        "type": "server_error",
                        "code": "realtime_unavailable",
                        "message": "The local speech-to-speech service is unavailable.",
                    },
                }
            )
        finally:
            if self.socket.application_state is WebSocketState.CONNECTED:
                with suppress(WebSocketDisconnect):
                    await self.socket.close(code=self.close_code)

    async def _send(self, connection: ClientConnection) -> None:
        while True:
            message = await self.socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            text = message.get("text")
            if text is None:
                self.close_code = 1003
                return
            if len(text.encode("utf-8")) > MAX_MESSAGE_BYTES:
                self.close_code = 1009
                return
            await connection.send(text)
            self._observe_audio(text)

    def _observe_audio(self, text: str) -> None:  # noqa: C901 - decode once for both PCM consumers
        # Forward first; the independent memory path receives a 16 kHz copy.
        observer = self.observe_audio
        if observer is None and self.user_audio is None:
            return
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        if event.get("type") == "session.update":
            self._update_input_rate(event)
            return
        if event.get("type") != "input_audio_buffer.append":
            return
        audio, event_id = event.get("audio"), event.get("event_id")
        if not isinstance(audio, str) or len(audio) > MAX_AUDIO_BASE64_CHARS:
            return
        data = _decode_pcm(audio)
        if data is None:
            return
        if self.user_audio is not None:
            self.user_audio.append_audio(data, self.input_rate)
        if observer is None or not isinstance(event_id, str):
            return
        clock = re.fullmatch(r"av_(\d+)_(\d+(?:\.\d+)?)", event_id)
        if clock is None:
            return
        pcm = data
        if self.input_rate == 24_000:
            pcm = self._resample_24k(data)
        observer(pcm, int(clock[1]), float(clock[2]))

    def _update_input_rate(self, event: dict[str, object]) -> None:
        session = event.get("session")
        audio_config = session.get("audio") if isinstance(session, dict) else None
        supplied = audio_config.get("input") if isinstance(audio_config, dict) else None
        audio_format = supplied.get("format") if isinstance(supplied, dict) else None
        rate = audio_format.get("rate") if isinstance(audio_format, dict) else None
        if isinstance(rate, int) and rate in (16_000, 24_000) and rate != self.input_rate:
            self.input_rate = rate
            self.samples.clear()
            self.sample_position = 0

    def _resample_24k(self, data: bytes) -> bytes:
        self.samples.extend(value for (value,) in struct.iter_unpack("<h", data))
        converted: list[int] = []
        while self.sample_position < len(self.samples):
            index = int(self.sample_position)
            fraction = self.sample_position - index
            if fraction and index + 1 == len(self.samples):
                break
            value = self.samples[index]
            if fraction:
                value += round((self.samples[index + 1] - value) * fraction)
            converted.append(value)
            self.sample_position += 1.5
        consumed = min(int(self.sample_position), len(self.samples))
        del self.samples[:consumed]
        self.sample_position -= consumed
        return struct.pack(f"<{len(converted)}h", *converted)

    async def _receive(self, connection: ClientConnection) -> None:
        while True:
            message = await connection.recv()
            text = message if isinstance(message, str) else message.decode("utf-8")
            await self.socket.send_text(text)
            if self.user_audio is not None:
                self.user_audio.observe_event(text)


def _decode_pcm(audio: str) -> bytes | None:
    try:
        data = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError):
        return None
    return data if data and len(data) % 2 == 0 else None
