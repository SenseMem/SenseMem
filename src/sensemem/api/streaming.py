"""SSE answer delivery and ownership of a pinned synchronous answer stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from concurrent.futures import Executor
from contextlib import suppress
from contextvars import Context
from typing import TypeVar, cast

from fastapi import Request
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from starlette.types import Receive, Scope, Send

from sensemem.api.errors import exception_response
from sensemem.api.responses import AnswerResponse
from sensemem.kernel.execution import StreamWorker
from sensemem.records import AnswerChunk, AnswerResult

_STREAM_EXHAUSTED = object()
_FutureResult = TypeVar("_FutureResult")


async def _first_chunk_or_disconnect(
    stream: Generator[AnswerChunk, None, AnswerResult],
    *,
    connection: Request,
    worker: Executor,
    context: Context,
) -> object:
    loop = asyncio.get_running_loop()

    async def wait_for_disconnect() -> None:
        while (await connection.receive()).get("type") != "http.disconnect":
            pass

    pending_chunk = loop.run_in_executor(
        worker, context.run, lambda: next(stream, _STREAM_EXHAUSTED)
    )
    disconnected = asyncio.create_task(wait_for_disconnect())
    try:
        done, _pending = await asyncio.wait(
            (pending_chunk, disconnected),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnected not in done:
            return pending_chunk.result()
        raise asyncio.CancelledError
    except BaseException:
        # A running synchronous provider call cannot be interrupted safely. Queue close on its
        # pinned worker so disconnects and task cancellation release the response after it returns.
        pending_chunk.add_done_callback(_discard_future_result)
        closing = loop.run_in_executor(worker, context.run, stream.close)
        closing.add_done_callback(_discard_future_result)
        raise
    finally:
        disconnected.cancel()
        with suppress(asyncio.CancelledError):
            await disconnected


def _discard_future_result(future: asyncio.Future[_FutureResult]) -> None:
    with suppress(BaseException):
        future.result()


async def _answer_events(
    first: object,
    rest: Generator[AnswerChunk, None, AnswerResult],
    *,
    worker: Executor,
    context: Context,
    close: Callable[[], Awaitable[None]],
) -> AsyncGenerator[bytes, None]:
    loop = asyncio.get_running_loop()
    chunk = first

    try:
        while chunk is not _STREAM_EXHAUSTED:
            answer_chunk = cast(AnswerChunk, chunk)
            terminal = answer_chunk.result is not None
            if terminal:
                await close()
            yield _answer_event(answer_chunk)
            if terminal:
                return
            chunk = await loop.run_in_executor(
                worker, context.run, lambda: next(rest, _STREAM_EXHAUSTED)
            )
        raise RuntimeError("answer stream ended without a terminal result")
    except Exception as error:
        trace.get_current_span().set_status(StatusCode.ERROR)
        response = exception_response(error)
        yield _sse_event("error", json.loads(bytes(response.body)))


class _AnswerStreamingResponse(StreamingResponse):
    """Own the prefetched synchronous stream for the complete ASGI send lifecycle."""

    def __init__(
        self,
        first: object,
        rest: Generator[AnswerChunk, None, AnswerResult],
        *,
        worker: Executor,
        context: Context,
    ) -> None:
        self._rest = rest
        self._worker = worker
        self._context = context
        self._closed = False
        self._events = _answer_events(
            first, rest, worker=worker, context=context, close=self._close_stream
        )
        super().__init__(
            self._events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette sends `http.response.start` before it enters the body iterator. Closing only
            # in `_answer_events` therefore leaks the already-prefetched Memory operation whenever
            # headers or the first body frame cannot be sent.
            try:
                await self._events.aclose()
            finally:
                await self._close_stream()

    async def _close_stream(self) -> None:
        if self._closed:
            return
        self._closed = True
        await _close_answer_stream(self._rest, worker=self._worker, context=self._context)


async def _close_answer_stream(
    stream: Generator[AnswerChunk, None, AnswerResult],
    *,
    worker: Executor,
    context: Context,
) -> None:
    loop = asyncio.get_running_loop()
    try:
        closing = loop.run_in_executor(worker, context.run, stream.close)
        closing.add_done_callback(_discard_future_result)
        with suppress(asyncio.CancelledError):
            await asyncio.shield(closing)
    finally:
        if isinstance(worker, StreamWorker):
            with suppress(asyncio.CancelledError):
                await asyncio.shield(asyncio.wrap_future(worker.release()))
        else:
            worker.shutdown(wait=False)


def _answer_event(chunk: AnswerChunk) -> bytes:
    if chunk.result is None:
        payload: dict[str, object] = {"text": chunk.text}
        event = "delta"
    else:
        payload = AnswerResponse.model_validate(chunk.result).model_dump(mode="json")
        event = "result"
    return _sse_event(event, payload)


def _sse_event(event: str, payload: object) -> bytes:
    return (
        f"event: {event}\ndata: "
        + json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n\n"
    ).encode("utf-8")
