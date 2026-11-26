"""The typed MCP tool surface over one local memory boundary."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from time import perf_counter_ns
from typing import Annotated, ParamSpec, TypeVar, cast
from uuid import uuid4

from mcp.server import MCPServer

# MCP 2 exposes its default OTel middleware and trace extractor only from private _otel modules.
from mcp.server._otel import OpenTelemetryMiddleware
from mcp.server.context import CallNext, HandlerResult, ServerMiddleware, ServerRequestContext
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared._otel import extract_trace_context
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from opentelemetry import trace as otel_trace
from opentelemetry.trace import SpanKind, StatusCode, Tracer
from pydantic import (
    AwareDatetime,
    Field,
    JsonValue,
    StringConstraints,
)

from sensemem._capacity import Capacity
from sensemem.api.content import (
    Content,
    ContextBudgetInput,
    ContextSupplementInput,
    Limit,
    content_input,
    context_budget,
    context_supplements,
)
from sensemem.api.deliberation import deliberation_lifespan
from sensemem.api.messages import error_message
from sensemem.api.responses import (
    AnswerResponse,
    ContextBundleResponse,
    DeleteResponse,
    ForgetResponse,
    IdentityResponse,
    MemoryResponse,
    PageResponse,
    PerceptionResponse,
    RegisterResponse,
    ReinforceResponse,
    SearchResponse,
    UnlinkResponse,
)
from sensemem.capabilities import MemoryCapabilities
from sensemem.context_values import (
    ContextBudget,
)
from sensemem.exceptions import SenseMemError, StorageError
from sensemem.identity import (
    PerceptionKind,
)
from sensemem.memory import Memory
from sensemem.prompts import mcp as prompts
from sensemem.spatial import ObservationContext, RetrievalScope
from sensemem.types import AnswerPolicy, MemoryCategory

LOGGER = logging.getLogger(__name__)
_MCP_TOTAL = "sensemem.transport.server_total_ms"
_Identifier = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]
_Cursor = Annotated[str, StringConstraints(min_length=1)]
# The advertised tool defaults are read from the SDK value, so the prose and the schema cannot
# drift from ContextBudget.
_BUDGET = ContextBudget()
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_NON_IDEMPOTENT_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_IDEMPOTENT_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
# Reinforcement accumulates: each call raises access_count and moves the ranking factor, so a
# repeated call is not the same as one call and the honest hint is idempotent_hint=False.
_COUNTED_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
_DELETE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
# Erasure is destructive and, unlike a delete, cannot be repeated: the second call reports the
# person as unknown, so a retry after a dropped response is not the same call.
_ERASE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)
_TOOL_ARGUMENTS = {
    "add_memory": frozenset(
        {"content", "occurred_at", "occurred_end", "metadata", "memory_type", "context"}
    ),
    "search_memories": frozenset(
        {
            "query",
            "limit",
            "memory_type",
            "reference_at",
            "occurred_from",
            "occurred_until",
            "scope",
            "explain",
        }
    ),
    "ask_memory": frozenset(
        {"question", "limit", "memory_type", "reference_at", "scope", "answer_policy"}
    ),
    "compile_context": frozenset(
        {"goal", "budget", "reference_at", "scope", "allow_partial_sources", "supplements"}
    ),
    "get_memory": frozenset({"memory_id"}),
    "list_memories": frozenset({"limit", "cursor"}),
    "delete_memory": frozenset({"memory_id"}),
    "perceive_memory": frozenset({"memory_id", "kinds"}),
    "register_identity": frozenset({"identity_id", "name", "relationship"}),
    "get_identity": frozenset({"identity_id"}),
    "unlink_identity": frozenset({"alias_id"}),
    "forget_identity": frozenset({"identity_id"}),
    "reinforce_memories": frozenset({"memory_ids"}),
}
# The error envelope MCP shares with REST.
_ERROR_FIELDS = frozenset(
    {"code", "reason", "retryable", "stage", "subject", "message", "trace_id", "issues"}
)


def _error_codes(root: type[SenseMemError]) -> frozenset[str]:
    codes = {root.code}
    for subclass in root.__subclasses__():
        codes |= _error_codes(subclass)
    return frozenset(codes)


# Derived rather than listed. A hand-maintained set silently loses every new exception class to the
# middleware's validation_error overwrite, which is how model_output_truncated was destroyed.
_STABLE_ERROR_CODES = _error_codes(SenseMemError) | {"internal_error"}

_P = ParamSpec("_P")
_T = TypeVar("_T")


def build_mcp_server(
    memory: Memory,
    *,
    identity_operations: bool = True,
    embodied_operations: bool = True,
    write_operations: bool = True,
    tracer: Tracer | None = None,
    deliberate_every: float | None = None,
) -> MCPServer[None]:
    """Expose the typed agent tool surface without taking ownership of memory.

    Every tool is one call on the injected Memory: the embodied and identity tools are
    reachable here for the same reason the common-path tools are, because this server runs in the
    process that holds it.

    The three switches are the host's choice about which authority an agent holds. All default to
    True, which is the published thirteen-tool surface; each False withholds its group by never
    registering it, because a tool an agent cannot see is better than one it must be trusted not
    to call, and the server instructions say which groups are missing.

    - identity_operations: naming, reading, unlinking and erasing a person.
    - embodied_operations: perceive_memory. Recognition keeps face and voice identities
      separate. Only a host-authored SDK operation may confirm a binding.
    - write_operations: add_memory, delete_memory and reinforce_memories.

    With all three False the surface is exactly the five read tools -- search_memories,
    ask_memory, compile_context, get_memory and list_memories -- which is
    recall and compile alone.

    deliberate_every is the same opt-in slow loop create_app takes, off by default: a
    number of seconds after which this process runs consolidate() on the memory it serves, in
    the server lifespan. No tool reaches the control plane, and one physical data_dir has one
    live owner, so without it an MCP-only deployment records QUERY_FAILURE signals that
    nothing ever reads. It is refused when no consolidation backend is configured, or when the
    interval is not a positive number of seconds.
    """
    transport_telemetry = _request_telemetry(tracer or otel_trace.get_tracer("sensemem.api.mcp"))
    server: MCPServer[None] = MCPServer(
        "sensemem",
        title="SenseMem Memory",
        description=prompts.SERVER_DESCRIPTION,
        # Read once here because MCP fixes instructions at construction: an agent learns the
        # configured composition when it connects instead of spending a tool call or discovering
        # a missing backend by failing. /healthz reports the same view per request, so a
        # composition swapped behind a long-lived server stays visible there.
        instructions=_instructions(
            memory.capabilities,
            identity_operations=identity_operations,
            embodied_operations=embodied_operations,
            write_operations=write_operations,
        ),
        version="0.3.0",
        lifespan=(
            None if deliberate_every is None else deliberation_lifespan(memory, deliberate_every)
        ),
        middleware=[
            transport_telemetry,
            _request_admission(memory),
            cast(ServerMiddleware[None], _strict_tool_arguments),
        ],
    )
    # MCP 2 installs its generic OpenTelemetry middleware outside every user middleware. Keep the
    # privacy-bounded SenseMem transport span as the sole server boundary: retaining both counts
    # every request twice and lets the generic span publish arbitrary tool names and request IDs.
    server.middleware[:] = [
        transport_telemetry,
        *(
            middleware
            for middleware in server.middleware
            if middleware is not transport_telemetry
            and not isinstance(middleware, OpenTelemetryMiddleware)
        ),
    ]

    @server.tool(description=prompts.SEARCH_MEMORIES_DESCRIPTION, annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def search_memories(
        query: Annotated[Content, Field(description=prompts.QUERY_DESCRIPTION)],
        limit: Annotated[
            Limit,
            Field(description=prompts.SEARCH_MEMORIES_LIMIT_DESCRIPTION),
        ] = 10,
        memory_type: Annotated[
            MemoryCategory | None, Field(description=prompts.MEMORY_TYPE_FILTER_DESCRIPTION)
        ] = None,
        reference_at: Annotated[
            AwareDatetime | None, Field(description=prompts.REFERENCE_AT_DESCRIPTION)
        ] = None,
        occurred_from: Annotated[
            AwareDatetime | None,
            Field(description=prompts.SEARCH_MEMORIES_OCCURRED_FROM_DESCRIPTION),
        ] = None,
        occurred_until: Annotated[
            AwareDatetime | None,
            Field(description=prompts.SEARCH_MEMORIES_OCCURRED_UNTIL_DESCRIPTION),
        ] = None,
        scope: Annotated[
            RetrievalScope | None, Field(description=prompts.SCOPE_DESCRIPTION)
        ] = None,
        explain: Annotated[
            bool,
            Field(description=prompts.SEARCH_MEMORIES_EXPLAIN_DESCRIPTION),
        ] = False,
    ) -> SearchResponse:
        result = memory.search(
            content_input(query),
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
            explain=explain,
        )
        return SearchResponse.model_validate(result)

    @server.tool(description=prompts.ASK_MEMORY_DESCRIPTION, annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def ask_memory(
        question: Annotated[
            Content,
            Field(description=prompts.ASK_MEMORY_QUESTION_DESCRIPTION),
        ],
        limit: Annotated[
            Limit,
            Field(description=prompts.ASK_MEMORY_LIMIT_DESCRIPTION),
        ] = 5,
        memory_type: Annotated[
            MemoryCategory | None, Field(description=prompts.MEMORY_TYPE_FILTER_DESCRIPTION)
        ] = None,
        reference_at: Annotated[
            AwareDatetime | None, Field(description=prompts.REFERENCE_AT_DESCRIPTION)
        ] = None,
        scope: Annotated[
            RetrievalScope | None, Field(description=prompts.SCOPE_DESCRIPTION)
        ] = None,
        answer_policy: Annotated[
            AnswerPolicy, Field(description=prompts.ANSWER_POLICY_DESCRIPTION)
        ] = "strict",
    ) -> AnswerResponse:
        result = memory.ask(
            content_input(question),
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            answer_policy=answer_policy,
        )
        return AnswerResponse.model_validate(result)

    # The same annotation search_memories carries, because it is the same side effect: the
    # shared retrieval path may cache a transcript for spoken query media. Compiling stores no
    # memory, but it is not read-only, so the honest hint is the one search already publishes.
    @server.tool(description=prompts.COMPILE_CONTEXT_DESCRIPTION, annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def compile_context(
        goal: Annotated[Content, Field(description=prompts.GOAL_DESCRIPTION)],
        budget: Annotated[
            ContextBudgetInput | None,
            Field(
                description=prompts.budget_description(
                    max_chars=_BUDGET.max_chars, max_items=_BUDGET.max_items
                )
            ),
        ] = None,
        reference_at: Annotated[
            AwareDatetime | None, Field(description=prompts.REFERENCE_AT_DESCRIPTION)
        ] = None,
        scope: Annotated[
            RetrievalScope | None, Field(description=prompts.SCOPE_DESCRIPTION)
        ] = None,
        allow_partial_sources: Annotated[
            bool,
            Field(
                strict=True,
                description=prompts.COMPILE_CONTEXT_ALLOW_PARTIAL_SOURCES_DESCRIPTION,
            ),
        ] = False,
        supplements: Annotated[
            tuple[ContextSupplementInput, ...],
            Field(description=prompts.COMPILE_CONTEXT_SUPPLEMENTS_DESCRIPTION),
        ] = (),
    ) -> ContextBundleResponse:
        return ContextBundleResponse.model_validate(
            memory.compile(
                content_input(goal),
                budget=context_budget(budget),
                reference_at=reference_at,
                scope=scope,
                allow_partial_sources=allow_partial_sources,
                supplements=context_supplements(supplements),
            ).document()
        )

    @server.resource(
        "sensemem://memories/{memory_id}/assets/{asset_id}",
        name="memory_asset",
        description=prompts.MEMORY_ASSET_DESCRIPTION,
        mime_type="application/octet-stream",
    )
    def read_memory_asset(memory_id: str, asset_id: str) -> bytes:
        return memory.read_asset(memory_id, asset_id).data

    @server.tool(description=prompts.GET_MEMORY_DESCRIPTION, annotations=_READ_ONLY)
    @_stable_errors
    def get_memory(
        memory_id: Annotated[_Identifier, Field(description=prompts.MEMORY_ID_DESCRIPTION)],
    ) -> MemoryResponse:
        return MemoryResponse.model_validate(memory.get(memory_id))

    @server.tool(description=prompts.LIST_MEMORIES_DESCRIPTION, annotations=_READ_ONLY)
    @_stable_errors
    def list_memories(
        limit: Annotated[
            Limit,
            Field(description=prompts.LIST_MEMORIES_LIMIT_DESCRIPTION),
        ] = 100,
        cursor: Annotated[
            _Cursor | None,
            Field(description=prompts.LIST_MEMORIES_CURSOR_DESCRIPTION),
        ] = None,
    ) -> PageResponse:
        page = memory.list(limit=limit, cursor=cursor)
        return PageResponse.model_validate(page)

    _register_write_tools(server, memory, enabled=write_operations)
    _register_embodied_tools(server, memory, enabled=embodied_operations)
    _register_identity_tools(server, memory, enabled=identity_operations)
    return server


def _register_write_tools(server: MCPServer[None], memory: Memory, *, enabled: bool) -> None:
    """Register the tools that mutate stored memory, unless the host withheld them.

    Adding a record, deleting one, and moving the ranker are the three tools whose purpose is a
    durable change. A host that wants an agent to consume context and nothing else builds the
    server with write_operations=False. ask_memory stays: it is a recall surface, and the
    reinforcement it records for the memories its answer cited is the owner's
    reinforce_on_answer setting rather than a capability this switch grants.
    """
    if not enabled:
        return

    @server.tool(description=prompts.ADD_MEMORY_DESCRIPTION, annotations=_IDEMPOTENT_WRITE)
    @_stable_errors
    def add_memory(
        content: Annotated[Content, Field(description=prompts.CONTENT_DESCRIPTION)],
        occurred_at: Annotated[
            AwareDatetime | None,
            Field(description=prompts.ADD_MEMORY_OCCURRED_AT_DESCRIPTION),
        ] = None,
        occurred_end: Annotated[
            AwareDatetime | None,
            Field(description=prompts.ADD_MEMORY_OCCURRED_END_DESCRIPTION),
        ] = None,
        metadata: Annotated[
            dict[str, JsonValue] | None,
            Field(description=prompts.ADD_MEMORY_METADATA_DESCRIPTION),
        ] = None,
        memory_type: Annotated[
            MemoryCategory,
            Field(description=prompts.ADD_MEMORY_MEMORY_TYPE_DESCRIPTION),
        ] = MemoryCategory.SEMANTIC,
        context: Annotated[
            ObservationContext | None,
            Field(description=prompts.ADD_MEMORY_CONTEXT_DESCRIPTION),
        ] = None,
    ) -> MemoryResponse:
        record = memory.add(
            content_input(content),
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )
        return MemoryResponse.model_validate(record)

    @server.tool(description=prompts.DELETE_MEMORY_DESCRIPTION, annotations=_DELETE)
    @_stable_errors
    def delete_memory(
        memory_id: Annotated[_Identifier, Field(description=prompts.MEMORY_ID_DESCRIPTION)],
    ) -> DeleteResponse:
        return DeleteResponse(deleted=memory.delete(memory_id))

    @server.tool(description=prompts.REINFORCE_MEMORIES_DESCRIPTION, annotations=_COUNTED_WRITE)
    @_stable_errors
    def reinforce_memories(
        memory_ids: Annotated[
            tuple[_Identifier, ...],
            Field(
                min_length=1,
                max_length=100,
                description=prompts.REINFORCE_MEMORIES_MEMORY_IDS_DESCRIPTION,
            ),
        ],
    ) -> ReinforceResponse:
        return ReinforceResponse(reinforced=memory.reinforce(memory_ids))


def _register_embodied_tools(server: MCPServer[None], memory: Memory, *, enabled: bool) -> None:
    """Register the embodied tools, unless the host withheld them.

    Recognition caches source-local observations. Binding a voice to a face requires an
    explicit host-authored SDK operation and is never inferred by a tool call.
    """
    if not enabled:
        return

    @server.tool(description=prompts.PERCEIVE_MEMORY_DESCRIPTION, annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def perceive_memory(
        memory_id: Annotated[_Identifier, Field(description=prompts.MEMORY_ID_DESCRIPTION)],
        kinds: Annotated[
            frozenset[PerceptionKind] | None,
            Field(
                description=prompts.PERCEIVE_MEMORY_KINDS_DESCRIPTION,
                min_length=1,
            ),
        ] = None,
    ) -> PerceptionResponse:
        return PerceptionResponse.model_validate(memory.perceive(memory_id, kinds=kinds))


def _register_identity_tools(
    server: MCPServer[None],
    memory: Memory,
    *,
    enabled: bool,
) -> None:
    """Register the identity tools, unless the host withheld them.

    Naming a person, splitting a wrong merge and erasing them are host authority, and every one
    of them is recorded in the operation log: naming and splitting are reversible through
    rollback(), and erasure leaves an irreversible row so the request stays auditable.
    Reading who somebody is changes nothing and logs nothing. A host that does not want an agent
    holding that authority builds the server with identity_operations=False, and then these
    four tools are never registered at all -- an agent cannot call a tool it cannot see, which
    is stronger than trusting it not to.
    """
    if not enabled:
        return

    @server.tool(description=prompts.REGISTER_IDENTITY_DESCRIPTION, annotations=_IDEMPOTENT_WRITE)
    @_stable_errors
    def register_identity(
        identity_id: Annotated[_Identifier, Field(description=prompts.IDENTITY_ID_DESCRIPTION)],
        name: Annotated[_Identifier, Field(description=prompts.PERSON_NAME_DESCRIPTION)],
        relationship: Annotated[
            _Identifier | None, Field(description=prompts.RELATIONSHIP_DESCRIPTION)
        ] = None,
    ) -> RegisterResponse:
        memory.register_identity(identity_id, name, relationship=relationship)
        return RegisterResponse(registered=True)

    @server.tool(description=prompts.GET_IDENTITY_DESCRIPTION, annotations=_READ_ONLY)
    @_stable_errors
    def get_identity(
        identity_id: Annotated[_Identifier, Field(description=prompts.IDENTITY_ID_DESCRIPTION)],
    ) -> IdentityResponse:
        return IdentityResponse(identity=memory.identity(identity_id))

    @server.tool(description=prompts.UNLINK_IDENTITY_DESCRIPTION, annotations=_DELETE)
    @_stable_errors
    def unlink_identity(
        alias_id: Annotated[
            _Identifier,
            Field(description=prompts.UNLINK_IDENTITY_ALIAS_ID_DESCRIPTION),
        ],
    ) -> UnlinkResponse:
        return UnlinkResponse(restored_identity_id=memory.unlink_identity(alias_id))

    @server.tool(description=prompts.FORGET_IDENTITY_DESCRIPTION, annotations=_ERASE)
    @_stable_errors
    def forget_identity(
        identity_id: Annotated[_Identifier, Field(description=prompts.IDENTITY_ID_DESCRIPTION)],
    ) -> ForgetResponse:
        return ForgetResponse(erasure=memory.forget_identity(identity_id))


def _request_admission(memory: Memory) -> ServerMiddleware[None]:
    slots = memory._execution.requests if isinstance(memory, Memory) else Capacity(32)

    async def admit(
        context: ServerRequestContext[None, object], call_next: CallNext
    ) -> HandlerResult:
        if context.method != "tools/call":
            return await call_next(context)
        if not slots.acquire():
            return _envelope_result(
                _tool_error(
                    StorageError(
                        "local request capacity exceeded",
                        reason="capacity_exceeded",
                        stage="admission",
                    )
                )
            )
        try:
            return await call_next(context)
        finally:
            slots.release()

    return cast(ServerMiddleware[None], admit)


def _stable_errors(
    operation: Callable[_P, _T],
) -> Callable[_P, _T]:
    @wraps(operation)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return operation(*args, **kwargs)
        except Exception as error:
            raise ToolError(_error_json(error)) from None

    if operation.__doc__ is not None:
        guarded.__doc__ = inspect.cleandoc(operation.__doc__)
    return guarded


async def _strict_tool_arguments(
    context: ServerRequestContext[None, object],
    call_next: CallNext,
) -> HandlerResult:
    params = context.params or {}
    if context.method == "tools/call":
        allowed = _TOOL_ARGUMENTS.get(str(params.get("name", "")))
        if allowed is None:
            return _error_result("validation_error", "tool does not exist", reason="unknown_field")
        arguments = params.get("arguments")
        if isinstance(arguments, Mapping):
            unknown = set(arguments).difference(allowed)
            if unknown:
                return _error_result(
                    "validation_error",
                    "tool arguments contain unknown fields",
                    reason="unknown_field",
                    issues=[
                        {
                            "location": ["arguments", name],
                            "message": "Extra inputs are not permitted",
                            "type": "extra_forbidden",
                        }
                        for name in sorted(unknown)
                    ],
                )
    result = await call_next(context)
    if (
        context.method == "tools/call"
        and isinstance(result, dict)
        and result.get("isError") is True
    ):
        # A tool body signals failure by raising, and the MCP runtime prefixes that message with
        # "Error executing tool <name>: ". Re-emit the envelope on its own so one bare json.loads
        # of the text works for every failure, including the recoverable ones an agent parses to
        # decide whether to retry.
        envelope = _stable_envelope(result)
        return _envelope_result(
            envelope
            if envelope is not None
            else _envelope("validation_error", "tool arguments are invalid", reason="input_invalid")
        )
    return result


def _request_telemetry(tracer: Tracer) -> ServerMiddleware[None]:
    async def observe(
        context: ServerRequestContext[None, object],
        call_next: CallNext,
    ) -> HandlerResult:
        params = context.params or {}
        attributes = {
            "sensemem.span.kind": "transport",
            "rpc.system": "mcp",
            "rpc.method": context.method,
            "sensemem.transport.response_mode": "buffered",
        }
        if context.method == "tools/call":
            service = params.get("name")
            if isinstance(service, str) and service in _TOOL_ARGUMENTS:
                attributes["rpc.service"] = service
        started = perf_counter_ns()
        with tracer.start_as_current_span(
            "sensemem.mcp.request",
            kind=SpanKind.SERVER,
            attributes=attributes,
            context=extract_trace_context(context.meta),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = await call_next(context)
            except BaseException:
                span.set_status(StatusCode.ERROR)
                raise
            finally:
                span.set_attribute(_MCP_TOTAL, (perf_counter_ns() - started) / 1_000_000)
            if (isinstance(result, Mapping) and result.get("isError") is True) or (
                isinstance(result, CallToolResult) and result.is_error
            ):
                span.set_status(StatusCode.ERROR)
            return result

    return cast(ServerMiddleware[None], observe)


# Which of MemoryCapabilities.operations one of the thirteen tools can actually reach, mirroring
# app._ROUTED_OPERATIONS. Consolidation, cognitive forgetting and rollback stay with the owning
# process (docs/architecture.md), so the greeting's prose and its JSON must agree that there is no
# consolidate here; perceive reaches a tool only when the host opts in, and so do
# formation and describe_vision, which run on the write path alone -- a read-only server has
# no tool that forms, and none that captions a frame. transcribe stays unconditional: a read
# tool transcribes the audio a caller asks its question with.
_TOOLED_OPERATIONS = frozenset({"ask", "transcribe"})
_EMBODIED_TOOLED_OPERATIONS = frozenset({"perceive"})
_WRITTEN_TOOLED_OPERATIONS = frozenset({"formation", "describe_vision"})


def _instructions(
    capabilities: MemoryCapabilities,
    *,
    identity_operations: bool = True,
    embodied_operations: bool = True,
    write_operations: bool = True,
) -> str:
    """Render the declared capability view an agent reads when it connects.

    The document is MemoryCapabilities.document() -- the same object /healthz serves and
    sensemem doctor prints -- rather than a prose subset that used to omit every model
    identity and every non-embedding modality set, narrowed to the operations this server's tools
    serve so operations cannot name one an agent will not find. MCP fixes instructions at
    construction, so this is a snapshot of the composition the server was built with; /healthz
    is the live one.
    """
    served = (
        _TOOLED_OPERATIONS
        | (_EMBODIED_TOOLED_OPERATIONS if embodied_operations else frozenset())
        | (_WRITTEN_TOOLED_OPERATIONS if write_operations else frozenset())
    )
    return prompts.server_instructions(
        capabilities.document(served=served),
        identity_operations=identity_operations,
        embodied_operations=embodied_operations,
        write_operations=write_operations,
    )


def _error_json(error: Exception) -> str:
    return json.dumps(_tool_error(error), separators=(",", ":"))


def _tool_error(error: Exception) -> dict[str, object]:
    if isinstance(error, SenseMemError):
        return _envelope(
            error.code,
            error_message(error),
            reason=error.reason,
            retryable=error.retryable,
            stage=error.stage,
            # MCP runs in the owner process as the same user, so a subject naming local state is
            # already visible to the caller. Unauthenticated REST withholds the same value.
            subject=error.subject,
        )
    envelope = _envelope(
        "internal_error",
        "the memory operation failed unexpectedly",
        reason="unexpected",
    )
    LOGGER.error(
        f"unhandled MCP tool error ({type(error).__name__})",
        extra={"trace_id": envelope["trace_id"]},
    )
    return envelope


def _envelope(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    retryable: bool = False,
    stage: str | None = None,
    subject: str | None = None,
    issues: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    return {
        "code": code,
        "reason": reason,
        "retryable": retryable,
        "stage": stage,
        "subject": subject,
        "message": message,
        "trace_id": f"trace_{uuid4().hex}",
        "issues": list(issues),
    }


def _error_result(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    issues: Sequence[Mapping[str, object]] = (),
) -> CallToolResult:
    return _envelope_result(_envelope(code, message, reason=reason, issues=issues))


def _envelope_result(envelope: Mapping[str, object]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(dict(envelope)))],
        is_error=True,
    )


def _stable_envelope(result: Mapping[str, object]) -> dict[str, object] | None:
    """Recover the envelope a tool raised, ignoring any prefix the MCP runtime added to it."""
    content = result.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
            continue
        text = item["text"]
        try:
            envelope: object = json.loads(text[text.index("{") :])
        except ValueError:
            continue
        if (
            isinstance(envelope, dict)
            and set(envelope) == _ERROR_FIELDS
            and envelope.get("code") in _STABLE_ERROR_CODES
            and isinstance(envelope.get("message"), str)
        ):
            return envelope
    return None
