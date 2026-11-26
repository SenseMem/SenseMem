"""Thin synchronous FastAPI adapter for one local Memory instance."""

from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncIterator,
    Generator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager, suppress
from contextvars import copy_context
from datetime import datetime
from time import monotonic, perf_counter_ns
from typing import Annotated, Protocol

from fastapi import APIRouter, FastAPI, Query, Request, Response, status
from fastapi import Path as PathParameter
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode, Tracer
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sensemem._capacity import Capacity
from sensemem.api.content import (
    content_input,
    context_budget,
    context_supplements,
)
from sensemem.api.deliberation import deliberation_lifespan
from sensemem.api.errors import (
    REASON_STATUS,
    error_response,
    error_responses,
    exception_response,
    register_error_handlers,
)
from sensemem.api.responses import (
    AnswerResponse,
    ConsentResponse,
    ContextBundleResponse,
    DeleteResponse,
    ExportResponse,
    ForgetResponse,
    HealthResponse,
    IdentityResponse,
    MemoryBatchResponse,
    MemoryResponse,
    PageResponse,
    PendingCapturesResponse,
    PerceptionResponse,
    RecordConsentResponse,
    RegisterResponse,
    ReinforceResponse,
    RetentionResponse,
    SearchResponse,
    SettleResponse,
    UnlinkResponse,
    _capabilities_response,
    _operation_response,
    bundle_response,
    export_response,
)
from sensemem.api.schemas import (
    AnswerRequest,
    ConsentRequest,
    ContextRequest,
    IdentityRegisterRequest,
    MemoryBatchCreate,
    MemoryCreate,
    PerceptionRequest,
    QueryRequest,
    ReinforceRequest,
    RetentionRequest,
    SettleRequest,
)
from sensemem.api.streaming import (
    STREAM_EXHAUSTED,
    AnswerStreamingResponse,
    _first_chunk_or_disconnect,
)
from sensemem.capabilities import MemoryCapabilities
from sensemem.content import Blob, ContentInput
from sensemem.context_bundle import ContextBundle
from sensemem.context_values import ContextBudget, ContextSupplement
from sensemem.control_values import ExportBundle, MemoryOperationRecord, RetentionReport
from sensemem.exceptions import StorageError
from sensemem.identity import (
    ConsentState,
    IdentityErasure,
    IdentityProfile,
    PerceptionKind,
    PerceptionResult,
)
from sensemem.kernel.execution import StreamWorkers
from sensemem.memory import Memory
from sensemem.records import (
    AnswerChunk,
    AnswerResult,
    MemoryRecord,
    Page,
    PendingCapture,
    SearchResult,
)
from sensemem.spatial import ObservationContext, RetrievalScope
from sensemem.types import AnswerPolicy, MemoryCategory

_MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
_REQUEST_BODY_TIMEOUT_SECONDS = 30.0
_RESPONSE_SEND_TIMEOUT_SECONDS = 30.0
_MemoryId = Annotated[str, PathParameter(min_length=1)]
_HTTP_TTFH = "sensemem.transport.server_time_to_headers_ms"
_HTTP_TTFB = "sensemem.transport.server_time_to_first_body_byte_ms"
_HTTP_TOTAL = "sensemem.transport.server_total_ms"


class _MemoryAccessor(Protocol):
    def __call__(self) -> _Memory: ...


class _Memory(Protocol):
    @property
    def capabilities(self) -> MemoryCapabilities: ...

    def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryCategory = MemoryCategory.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord: ...

    def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        occurred_at: Sequence[datetime | None] | None = None,
        occurred_end: Sequence[datetime | None] | None = None,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
        memory_type: MemoryCategory = MemoryCategory.SEMANTIC,
        context: Sequence[ObservationContext | None] | None = None,
    ) -> tuple[MemoryRecord, ...]: ...

    def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryCategory | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
        explain: bool = False,
    ) -> SearchResult: ...

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryCategory | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        answer_policy: AnswerPolicy = "strict",
    ) -> AnswerResult: ...

    def ask_stream(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryCategory | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        answer_policy: AnswerPolicy = "strict",
    ) -> Generator[AnswerChunk, None, AnswerResult]: ...

    def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        allow_partial_sources: bool = False,
        supplements: Sequence[ContextSupplement] = (),
    ) -> ContextBundle: ...

    def reinforce(self, memory_ids: Sequence[str]) -> int: ...

    def get(self, memory_id: str) -> MemoryRecord: ...

    def read_asset(self, memory_id: str, asset_id: str) -> Blob: ...

    def list(self, *, limit: int = 100, cursor: str | None = None) -> Page: ...

    def delete(self, memory_id: str) -> bool: ...

    def capture(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryCategory = MemoryCategory.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord: ...

    def settle(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 3,
        memory_ids: Sequence[str] | None = None,
    ) -> int: ...

    def pending_captures(
        self,
        *,
        limit: int = 100,
        memory_ids: Sequence[str] | None = None,
        cursor: str | None = None,
    ) -> tuple[PendingCapture, ...]: ...

    def perceive(
        self, memory_id: str, *, kinds: frozenset[PerceptionKind] | None = None
    ) -> PerceptionResult: ...

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None: ...

    def identity(self, identity_id: str) -> IdentityProfile | None: ...

    def unlink_identity(self, alias_id: str) -> str | None: ...

    def forget_identity(self, identity_id: str) -> IdentityErasure: ...

    def record_consent(
        self,
        identity_id: str,
        state: ConsentState,
        *,
        note: str | None = None,
    ) -> MemoryOperationRecord | None: ...

    def consent(self, identity_id: str) -> ConsentState | None: ...

    def export(
        self,
        *,
        identity_id: str | None = None,
        memory_ids: Sequence[str] | None = None,
    ) -> ExportBundle: ...

    def apply_retention(self, *, dry_run: bool = False) -> RetentionReport: ...


class _ResponseSendTimeout(Exception):
    pass


class _RequestCapacity:
    """Refuse overload before buffering bodies or waiting for a framework worker."""

    def __init__(self, app: ASGIApp, *, capacity: Capacity) -> None:
        self.app = app
        self._slots = capacity

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_api_path(str(scope.get("path", ""))):
            await self.app(scope, receive, send)
            return
        started = False

        async def bounded_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            try:
                await asyncio.wait_for(send(message), timeout=_RESPONSE_SEND_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                raise _ResponseSendTimeout from None

        if not self._slots.acquire():
            response = exception_response(
                StorageError(
                    "local request capacity exceeded", reason="capacity_exceeded", stage="admission"
                )
            )
            with suppress(_ResponseSendTimeout):
                await response(scope, receive, bounded_send)
            return
        try:
            await self.app(scope, receive, bounded_send)
        except _ResponseSendTimeout:
            return
        except asyncio.TimeoutError:
            if not started:
                response = exception_response(
                    StorageError("request body timed out", reason="timeout", stage="transport.read")
                )
                with suppress(_ResponseSendTimeout):
                    await response(scope, receive, bounded_send)
        finally:
            self._slots.release()


class _RequestBodyLimit:
    """Bound /v1 request bodies before framework parsing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_api_path(str(scope.get("path", ""))):
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope.get("headers", []))
        if content_length is not None and content_length > _MAX_REQUEST_BODY_BYTES:
            await _request_too_large()(scope, receive, send)
            return

        body = bytearray()
        deadline = monotonic() + _REQUEST_BODY_TIMEOUT_SECONDS
        while True:
            message = await asyncio.wait_for(receive(), timeout=max(0.0, deadline - monotonic()))
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > _MAX_REQUEST_BODY_BYTES:
                await _request_too_large()(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


class _RequestTelemetry:
    """Measure the server side of one /v1 request, including streamed delivery."""

    def __init__(self, app: ASGIApp, *, tracer: Tracer) -> None:
        self.app = app
        self.tracer = tracer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path", ""))
        if scope["type"] != "http" or not _is_api_path(path):
            await self.app(scope, receive, send)
            return

        started = perf_counter_ns()
        response_started = False
        first_body_sent = False
        response_bytes = 0
        attributes = {
            "sensemem.span.kind": "transport",
            "http.request.method": str(scope.get("method", "")),
            "sensemem.transport.response_mode": (
                "streaming" if path == "/v1/answers/stream" else "buffered"
            ),
        }
        with self.tracer.start_as_current_span(
            "sensemem.http.request",
            kind=SpanKind.SERVER,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:

            async def observed_send(message: Message) -> None:
                nonlocal response_started, first_body_sent, response_bytes
                now = perf_counter_ns()
                if message["type"] == "http.response.start":
                    response_started = True
                    status_code = int(message["status"])
                    span.set_attribute("http.response.status_code", status_code)
                    span.set_attribute(_HTTP_TTFH, (now - started) / 1_000_000)
                    if status_code >= 500:
                        span.set_status(StatusCode.ERROR)
                elif message["type"] == "http.response.body":
                    body = message.get("body", b"")
                    response_bytes += len(body)
                    if body and not first_body_sent:
                        first_body_sent = True
                        span.set_attribute(_HTTP_TTFB, (now - started) / 1_000_000)
                await send(message)

            try:
                await self.app(scope, receive, observed_send)
            except BaseException:
                span.set_status(StatusCode.ERROR)
                raise
            finally:
                route = scope.get("route")
                if isinstance(route, Route):
                    span.set_attribute("http.route", route.path)
                span.set_attribute(_HTTP_TOTAL, (perf_counter_ns() - started) / 1_000_000)
                span.set_attribute("http.response.body.size", response_bytes)
                span.set_attribute("sensemem.transport.response_started", response_started)


# Which of MemoryCapabilities.operations a /v1 route can actually reach. Consolidation --
# like cognitive forgetting and operation rollback, which are not derived capabilities at all --
# has no route by design (docs/architecture.md), so /healthz must not advertise it; perceive
# reaches perceiveMemory only when the host opts in. An operation added to the derivation stays
# unadvertised here until it is given a route.
_ROUTED_OPERATIONS = frozenset({"ask", "transcribe", "describe_vision", "formation"})
_EMBODIED_ROUTED_OPERATIONS = frozenset({"perceive"})


def create_app(
    *,
    memory: _Memory,
    identity_operations: bool = False,
    embodied_operations: bool = False,
    tracer: Tracer | None = None,
    deliberate_every: float | None = None,
) -> FastAPI:
    """Create an unauthenticated API over one caller-owned memory instance.

    identity_operations and embodied_operations mirror the same-named switches on
    build_mcp_server. Both default to off: REST is a network surface, and naming, erasing,
    or analyzing a person is host authority that no route grants unless the host opts in. When a
    switch is off the routes it gates are never registered, so a caller gets 404 rather than 403 --
    it cannot discover through the error what an enabled deployment would offer.

    deliberate_every is a number of seconds after which this process runs consolidate()
    on the memory it serves, off by default. The control plane stays SDK-only -- no route and no
    tool exposes it -- but the signals the routes write, above all the QUERY_FAILURE rows a
    failed search, ask, or compile records, can only be consumed by the process that
    owns the directory. Without this a REST-only or MCP-only deployment collects them and drops
    them. It is refused when no consolidation backend is configured, because the loop would then
    raise on every round.
    """
    served = _ROUTED_OPERATIONS | (
        _EMBODIED_ROUTED_OPERATIONS if embodied_operations else frozenset()
    )
    execution = memory._execution if isinstance(memory, Memory) else None
    stream_workers = execution.streams if execution is not None else StreamWorkers()
    background_lifespan = (
        None if deliberate_every is None else deliberation_lifespan(memory, deliberate_every)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            if background_lifespan is None:
                yield
            else:
                async with background_lifespan(app):
                    yield
        finally:
            if execution is None:
                await asyncio.to_thread(stream_workers.close)

    app = FastAPI(
        title="SenseMem",
        version="0.3.0",
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.add_middleware(_RequestBodyLimit)
    app.add_middleware(
        _RequestCapacity,
        capacity=execution.requests if execution is not None else Capacity(32),
    )
    app.add_middleware(
        _RequestTelemetry,
        tracer=tracer or trace.get_tracer("sensemem.api"),
    )

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        operation_id="health",
        responses=error_responses(status.HTTP_500_INTERNAL_SERVER_ERROR),
    )
    def health() -> HealthResponse:
        # Read from the injected instance on every call, so a composition swapped behind this
        # process is reported rather than a snapshot taken at construction.
        return HealthResponse(
            capabilities=_capabilities_response(memory.capabilities, served=served)
        )

    app.include_router(
        _v1_router(
            memory,
            identity_operations=identity_operations,
            embodied_operations=embodied_operations,
            stream_workers=stream_workers,
        )
    )
    return app


def _v1_router(  # noqa: C901 - one literal public route registry
    memory: _Memory,
    *,
    identity_operations: bool,
    embodied_operations: bool,
    stream_workers: StreamWorkers,
) -> APIRouter:
    """Register every /v1 route against one caller-owned memory instance."""

    def current_service() -> _Memory:
        return memory

    router = APIRouter(prefix="/v1")
    standard_statuses = (
        status.HTTP_413_CONTENT_TOO_LARGE,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    standard_errors = error_responses(*standard_statuses)
    model_errors = error_responses(
        *standard_statuses,
        status.HTTP_501_NOT_IMPLEMENTED,
        status.HTTP_502_BAD_GATEWAY,
    )
    not_found_errors = error_responses(*standard_statuses, status.HTTP_404_NOT_FOUND)
    streaming_errors: dict[int | str, dict[str, object]] = {
        **model_errors,
        status.HTTP_200_OK: {
            "description": "SSE answer deltas followed by one grounded result event",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
    }

    @router.post(
        "/memories",
        response_model=MemoryResponse,
        status_code=status.HTTP_201_CREATED,
        operation_id="createMemory",
        responses=model_errors,
    )
    def create_memory(request: MemoryCreate) -> MemoryResponse:
        record = current_service().add(
            content_input(request.content),
            occurred_at=request.occurred_at,
            occurred_end=request.occurred_end,
            metadata=request.metadata,
            memory_type=request.memory_type,
            context=request.context,
        )
        return MemoryResponse.model_validate(record)

    @router.post(
        "/memories/batch",
        response_model=MemoryBatchResponse,
        status_code=status.HTTP_201_CREATED,
        operation_id="createMemories",
        responses=model_errors,
    )
    def create_memories(request: MemoryBatchCreate) -> MemoryBatchResponse:
        return MemoryBatchResponse.model_validate(
            {
                "memories": current_service().add_many(
                    tuple(content_input(content) for content in request.contents),
                    occurred_at=request.occurred_at,
                    occurred_end=request.occurred_end,
                    metadata=request.metadata,
                    memory_type=request.memory_type,
                    context=request.context,
                )
            }
        )

    @router.get(
        "/memories",
        response_model=PageResponse,
        operation_id="listMemories",
        responses=standard_errors,
    )
    def list_memories(
        limit: Annotated[int, Query(ge=1, le=100)] = 100,
        cursor: Annotated[str | None, Query(min_length=1)] = None,
    ) -> PageResponse:
        return PageResponse.model_validate(current_service().list(limit=limit, cursor=cursor))

    @router.post(
        "/memories/search",
        response_model=SearchResponse,
        operation_id="searchMemories",
        responses=model_errors,
    )
    def search_memories(request: QueryRequest) -> SearchResponse:
        return _search(current_service(), request)

    @router.post(
        "/memories/reinforce",
        response_model=ReinforceResponse,
        operation_id="reinforceMemories",
        responses=standard_errors,
    )
    def reinforce_memories(request: ReinforceRequest) -> ReinforceResponse:
        return ReinforceResponse(reinforced=current_service().reinforce(request.memory_ids))

    @router.get(
        "/memories/{memory_id}",
        response_model=MemoryResponse,
        operation_id="getMemory",
        responses=not_found_errors,
    )
    def get_memory(memory_id: _MemoryId) -> MemoryResponse:
        return MemoryResponse.model_validate(current_service().get(memory_id))

    @router.get(
        "/memories/{memory_id}/assets/{asset_id}",
        operation_id="readAsset",
        response_class=Response,
        responses=not_found_errors,
    )
    def read_asset(memory_id: _MemoryId, asset_id: _MemoryId) -> Response:
        blob = current_service().read_asset(memory_id, asset_id)
        return Response(content=blob.data, media_type=blob.media_type)

    @router.delete(
        "/memories/{memory_id}",
        response_model=DeleteResponse,
        operation_id="deleteMemory",
        responses=standard_errors,
    )
    def delete_memory(memory_id: _MemoryId) -> DeleteResponse:
        return DeleteResponse(deleted=current_service().delete(memory_id))

    @router.post(
        "/answers",
        response_model=AnswerResponse,
        operation_id="answer",
        responses=model_errors,
    )
    def answer(request: AnswerRequest) -> AnswerResponse:
        return AnswerResponse.model_validate(
            current_service().ask(
                content_input(request.question),
                limit=request.limit,
                memory_type=request.memory_type,
                reference_at=request.reference_at,
                scope=request.scope,
                answer_policy=request.answer_policy,
            )
        )

    @router.post(
        "/answers/stream",
        operation_id="answerStream",
        responses=streaming_errors,
        response_class=StreamingResponse,
    )
    async def answer_stream(request: AnswerRequest, connection: Request) -> StreamingResponse:
        chunks = iter(
            current_service().ask_stream(
                content_input(request.question),
                limit=request.limit,
                memory_type=request.memory_type,
                reference_at=request.reference_at,
                scope=request.scope,
                answer_policy=request.answer_policy,
            )
        )
        worker = stream_workers.borrow()
        context = copy_context()

        try:
            first = await _first_chunk_or_disconnect(
                chunks,
                connection=connection,
                worker=worker,
                context=context,
            )
        except BaseException:
            worker.shutdown(wait=False)
            raise
        if first is STREAM_EXHAUSTED:
            worker.shutdown(wait=False)
            raise RuntimeError("answer stream ended without a terminal result")
        return AnswerStreamingResponse(
            first,
            chunks,
            worker=worker,
            context=context,
        )

    _add_context_route(router, current_service, responses=model_errors)
    _add_capture_routes(
        router,
        current_service,
        model_errors=model_errors,
        standard_errors=standard_errors,
    )
    _add_gated_routes(
        router,
        current_service,
        identity_operations=identity_operations,
        embodied_operations=embodied_operations,
        standard_statuses=standard_statuses,
        standard_errors=standard_errors,
        not_found_errors=not_found_errors,
    )
    return router


def _add_gated_routes(
    router: APIRouter,
    current_service: _MemoryAccessor,
    *,
    identity_operations: bool,
    embodied_operations: bool,
    standard_statuses: tuple[int, ...],
    standard_errors: dict[int | str, dict[str, object]],
    not_found_errors: dict[int | str, dict[str, object]],
) -> None:
    """Register the opt-in identity and embodied routes, split out to keep _v1_router readable."""
    if embodied_operations:
        embodied_errors = error_responses(
            *standard_statuses,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_501_NOT_IMPLEMENTED,
            status.HTTP_502_BAD_GATEWAY,
        )
        _add_embodied_routes(router, current_service, responses=embodied_errors)
    if identity_operations:
        _add_identity_routes(
            router,
            current_service,
            not_found_errors=not_found_errors,
            standard_errors=standard_errors,
        )


def _add_capture_routes(
    router: APIRouter,
    current_service: _MemoryAccessor,
    *,
    model_errors: dict[int | str, dict[str, object]],
    standard_errors: dict[int | str, dict[str, object]],
) -> None:
    """Register the fast-capture plane: capture, settle, and pending_captures.

    These are always on, unlike the identity and embodied routes below: they are ordinary
    application operations on the caller's own records, not administrative authority over a
    person.
    """

    @router.post(
        "/capture",
        response_model=MemoryResponse,
        status_code=status.HTTP_201_CREATED,
        operation_id="captureMemory",
        responses=model_errors,
    )
    def capture_memory(request: MemoryCreate) -> MemoryResponse:
        record = current_service().capture(
            content_input(request.content),
            occurred_at=request.occurred_at,
            occurred_end=request.occurred_end,
            metadata=request.metadata,
            memory_type=request.memory_type,
            context=request.context,
        )
        return MemoryResponse.model_validate(record)

    @router.post(
        "/settle",
        response_model=SettleResponse,
        operation_id="settleCaptures",
        responses=model_errors,
    )
    def settle_captures(request: SettleRequest) -> SettleResponse:
        return SettleResponse(
            settled=current_service().settle(
                limit=request.limit,
                max_attempts=request.max_attempts,
                memory_ids=request.memory_ids,
            )
        )

    @router.get(
        "/pending_captures",
        response_model=PendingCapturesResponse,
        operation_id="pendingCaptures",
        responses=standard_errors,
    )
    def pending_captures_route(
        limit: Annotated[int, Query(ge=1, le=100)] = 100,
        memory_ids: Annotated[list[str] | None, Query(min_length=1, max_length=100)] = None,
        cursor: str | None = None,
    ) -> PendingCapturesResponse:
        items = current_service().pending_captures(
            limit=limit, memory_ids=memory_ids, cursor=cursor
        )
        return PendingCapturesResponse.model_validate(
            {"items": items, "next_cursor": items[-1].cursor if len(items) == limit else None}
        )


def _add_embodied_routes(
    router: APIRouter,
    current_service: _MemoryAccessor,
    *,
    responses: dict[int | str, dict[str, object]],
) -> None:
    """Register source-local multimodal perception, gated by embodied_operations."""

    @router.post(
        "/perception",
        response_model=PerceptionResponse,
        operation_id="perceiveMemory",
        responses=responses,
    )
    def perceive_memory(request: PerceptionRequest) -> PerceptionResponse:
        result = current_service().perceive(request.memory_id, kinds=request.kinds)
        return PerceptionResponse.model_validate(result)


def _add_identity_routes(
    router: APIRouter,
    current_service: _MemoryAccessor,
    *,
    not_found_errors: dict[int | str, dict[str, object]],
    standard_errors: dict[int | str, dict[str, object]],
) -> None:
    """Register naming, reading, unlinking, and erasing identities.

    Gated by the host's identity_operations: naming and erasing a person is host authority,
    exercised through the caller, exactly as it is on MCP.
    """

    @router.post(
        "/identities",
        response_model=RegisterResponse,
        operation_id="registerIdentity",
        responses=not_found_errors,
    )
    def register_identity(request: IdentityRegisterRequest) -> RegisterResponse:
        current_service().register_identity(
            request.identity_id,
            request.name,
            relationship=request.relationship,
        )
        return RegisterResponse(registered=True)

    @router.get(
        "/identities/{identity_id}",
        response_model=IdentityResponse,
        operation_id="getIdentity",
        responses=standard_errors,
    )
    def get_identity(identity_id: _MemoryId) -> IdentityResponse:
        return IdentityResponse(identity=current_service().identity(identity_id))

    @router.post(
        "/identities/{alias_id}/unlink",
        response_model=UnlinkResponse,
        operation_id="unlinkIdentity",
        responses=standard_errors,
    )
    def unlink_identity(alias_id: _MemoryId) -> UnlinkResponse:
        return UnlinkResponse(restored_identity_id=current_service().unlink_identity(alias_id))

    @router.delete(
        "/identities/{identity_id}",
        response_model=ForgetResponse,
        operation_id="forgetIdentity",
        responses=not_found_errors,
    )
    def forget_identity(identity_id: _MemoryId) -> ForgetResponse:
        return ForgetResponse(erasure=current_service().forget_identity(identity_id))

    @router.post(
        "/identities/{identity_id}/consent",
        response_model=RecordConsentResponse,
        operation_id="recordConsent",
        responses=not_found_errors,
    )
    def record_consent(identity_id: _MemoryId, request: ConsentRequest) -> RecordConsentResponse:
        record = current_service().record_consent(
            identity_id,
            request.state,
            note=request.note,
        )
        return RecordConsentResponse(
            operation=None if record is None else _operation_response(record)
        )

    @router.get(
        "/identities/{identity_id}/consent",
        response_model=ConsentResponse,
        operation_id="getConsent",
        responses=standard_errors,
    )
    def get_consent(identity_id: _MemoryId) -> ConsentResponse:
        return ConsentResponse(consent=current_service().consent(identity_id))

    @router.get(
        "/export",
        response_model=ExportResponse,
        operation_id="exportSubject",
        responses=not_found_errors,
    )
    def export_subject(
        identity_id: Annotated[str | None, Query(min_length=1)] = None,
        memory_ids: Annotated[list[str] | None, Query(min_length=1, max_length=100)] = None,
    ) -> ExportResponse:
        return export_response(
            current_service().export(identity_id=identity_id, memory_ids=memory_ids)
        )

    @router.post(
        "/retention",
        response_model=RetentionResponse,
        operation_id="applyRetention",
        responses=standard_errors,
    )
    def apply_retention(request: RetentionRequest) -> RetentionResponse:
        return RetentionResponse.model_validate(
            current_service().apply_retention(dry_run=request.dry_run)
        )


def _add_context_route(
    router: APIRouter,
    current_service: _MemoryAccessor,
    *,
    responses: dict[int | str, dict[str, object]],
) -> None:
    """Register the compiler route, split out only to keep _v1_router readable."""

    @router.post(
        "/context",
        response_model=ContextBundleResponse,
        operation_id="compileContext",
        responses=responses,
    )
    def compile_context(request: ContextRequest) -> ContextBundleResponse:
        return bundle_response(
            current_service().compile(
                content_input(request.goal),
                budget=context_budget(request.budget),
                reference_at=request.reference_at,
                scope=request.scope,
                allow_partial_sources=request.allow_partial_sources,
                supplements=context_supplements(request.supplements),
            )
        )


def _search(service: _Memory, request: QueryRequest) -> SearchResponse:
    result = service.search(
        content_input(request.query),
        limit=request.limit,
        memory_type=request.memory_type,
        reference_at=request.reference_at,
        occurred_from=request.occurred_from,
        occurred_until=request.occurred_until,
        scope=request.scope,
        explain=request.explain,
    )
    return SearchResponse.model_validate(result)


def _content_length(headers: Sequence[tuple[bytes, bytes]]) -> int | None:
    values = [value for key, value in headers if key.lower() == b"content-length"]
    if len(values) != 1:
        return None
    try:
        value = int(values[0])
    except ValueError:
        return None
    return value if value >= 0 else None


def _is_api_path(path: str) -> bool:
    return path == "/v1" or path.startswith("/v1/")


def _request_too_large() -> Response:
    return error_response(
        # Read from the shared table, so this cannot drift from the provider-side raise site that
        # reports the same reason.
        REASON_STATUS["payload_too_large"],
        "request_too_large",
        f"request body must not exceed {_MAX_REQUEST_BODY_BYTES} bytes",
        reason="payload_too_large",
    )
