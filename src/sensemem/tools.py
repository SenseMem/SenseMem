"""Explicit, schema-validated capabilities shared by pipelines and agent tool loops."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel

from sensemem._resources import Closable, unique_resources
from sensemem.exceptions import ModelError, ValidationError

_Input = TypeVar("_Input", bound=BaseModel)
_Output = TypeVar("_Output", bound=BaseModel)
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")

ToolErrorCode = Literal[
    "unknown_tool",
    "invalid_arguments",
    "execution_failed",
    "invalid_output",
    "tool_closed",
    "registry_closed",
]
_ERROR_MESSAGES: Mapping[ToolErrorCode, str] = MappingProxyType(
    {
        "unknown_tool": "The requested tool is not registered.",
        "invalid_arguments": "Tool arguments did not match the input schema.",
        "execution_failed": "Tool execution failed.",
        "invalid_output": "Tool output did not match the output schema.",
        "tool_closed": "The tool is closed.",
        "registry_closed": "The tool registry is closed.",
    }
)


def _validate_tool_name(value: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValidationError(
            "tool name must contain 1 to 64 letters, digits, underscores or hyphens"
        )


def _validate_tool_call_id(value: str) -> None:
    if not isinstance(value, str) or (value and not value.strip()):
        raise ValidationError("tool call id must be text and cannot be whitespace")


def _freeze_json_value(value: object) -> object:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValidationError("tool JSON objects require text keys")
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    raise ValidationError("tool values must contain only finite JSON data")


def _freeze_json_object(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError("tool values must be JSON objects")
    return cast(Mapping[str, object], _freeze_json_value(value))


def _json_document_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_document_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_document_value(item) for item in value]
    return value


def _json_object_document(value: Mapping[str, object]) -> dict[str, object]:
    return {key: _json_document_value(item) for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One named capability and its immutable input and output JSON schemas."""

    name: str
    description: str
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_tool_name(self.name)
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValidationError("tool description must be nonempty text")
        object.__setattr__(self, "input_schema", _freeze_json_object(self.input_schema))
        object.__setattr__(self, "output_schema", _freeze_json_object(self.output_schema))

    def document(self) -> dict[str, object]:
        """Return an independent JSON-serializable description."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _json_object_document(self.input_schema),
            "output_schema": _json_object_document(self.output_schema),
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One untrusted call proposal; only the registry can resolve its tool name."""

    id: str
    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_tool_call_id(self.id)
        _validate_tool_name(self.name)
        object.__setattr__(self, "arguments", _freeze_json_object(self.arguments))

    def document(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": _json_object_document(self.arguments),
        }


@dataclass(frozen=True, slots=True)
class ToolError:
    """A safe, bounded failure classification that an agent may use to continue."""

    code: ToolErrorCode
    message: str

    def __post_init__(self) -> None:
        if self.code not in _ERROR_MESSAGES:
            raise ValidationError("tool error code is invalid")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValidationError("tool error message must be nonempty text")

    def document(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Exactly one validated JSON object or safe error for one call."""

    call_id: str
    name: str
    output: Mapping[str, object] | None = None
    error: ToolError | None = None

    def __post_init__(self) -> None:
        _validate_tool_call_id(self.call_id)
        _validate_tool_name(self.name)
        if (self.output is None) == (self.error is None):
            raise ValidationError("tool result requires exactly one output or error")
        if self.error is not None and not isinstance(self.error, ToolError):
            raise ValidationError("tool result error must be a ToolError")
        if self.output is not None:
            object.__setattr__(self, "output", _freeze_json_object(self.output))

    def document(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "output": None if self.output is None else _json_object_document(self.output),
            "error": None if self.error is None else self.error.document(),
        }


@dataclass(slots=True)
class _ToolState:
    # A resource leaf: Memory flattens ResourceGroup objects instead of calling their close().
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def _close_tool_resources(resources: Sequence[Closable]) -> None:
    failed = False
    for resource in unique_resources(resources):
        try:
            resource.close()
        except Exception:
            failed = True
    if failed:
        raise ModelError(
            "failed to close tool resources", reason="model_failed", stage="tools.close"
        )


class Tool(Generic[_Input, _Output]):
    """Wrap one model, algorithm, or lookup in validated Pydantic object contracts.

    The handler receives no kernel reference from this wrapper. Applications explicitly grant
    any authority its closure needs. Handlers must be thread-safe when callers run concurrently.
    Only resources declared here are owned and closed; a handler itself need not implement close.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[_Input],
        output_model: type[_Output],
        handler: Callable[[_Input], _Output],
        *,
        resources: Sequence[Closable] = (),
    ) -> None:
        if (
            not isinstance(input_model, type)
            or not issubclass(input_model, BaseModel)
            or not isinstance(output_model, type)
            or not issubclass(output_model, BaseModel)
        ):
            raise ValidationError("tool input and output models must be Pydantic models")
        if not callable(handler):
            raise ValidationError("tool handler must be callable")
        input_schema = input_model.model_json_schema(mode="validation")
        output_schema = output_model.model_json_schema(mode="serialization")
        if input_schema.get("type") != "object" or output_schema.get("type") != "object":
            raise ValidationError("tool input and output models must describe JSON objects")
        self._definition = ToolDefinition(
            name,
            description,
            input_schema,
            output_schema,
        )
        self._input_model = input_model
        self._output_model = output_model
        self._handler = handler
        self._resources = tuple(resources)
        self._state = _ToolState()

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    @property
    def closed(self) -> bool:
        return self._state.closed

    def call(self, arguments: Mapping[str, object], *, call_id: str = "") -> ToolResult:
        """Validate one call and return a result without exposing handler exception bodies."""
        _validate_tool_call_id(call_id)
        name = self._definition.name
        if self.closed:
            return ToolResult(call_id, name, error=ToolError("tool_closed", "The tool is closed."))
        try:
            payload = json.dumps(
                _json_object_document(_freeze_json_object(arguments)), allow_nan=False
            )
            inputs = self._input_model.model_validate_json(payload, strict=True)
        except Exception:
            return ToolResult(
                call_id,
                name,
                error=ToolError(
                    "invalid_arguments", "Tool arguments did not match the input schema."
                ),
            )
        try:
            output = self._handler(inputs)
        except Exception:
            return ToolResult(
                call_id, name, error=ToolError("execution_failed", "Tool execution failed.")
            )
        try:
            if not isinstance(output, self._output_model):
                raise ValueError("unexpected output model")
            payload = json.dumps(
                output.model_dump(mode="json", by_alias=False, warnings=False), allow_nan=False
            )
            validated = self._output_model.model_validate_json(
                payload, strict=True, by_alias=False, by_name=True
            )
            document = validated.model_dump(mode="json", by_alias=True, warnings=False)
            return ToolResult(call_id, name, output=document)
        except Exception:
            return ToolResult(
                call_id,
                name,
                error=ToolError("invalid_output", "Tool output did not match the output schema."),
            )

    def resources(self) -> tuple[Closable, ...]:
        return (self._state, *self._resources)

    def close(self) -> None:
        if not self.closed:
            _close_tool_resources(self.resources())


@runtime_checkable
class RegisteredTool(Protocol):
    """The non-generic view used to collect tools with different typed handlers."""

    @property
    def definition(self) -> ToolDefinition: ...

    def call(self, arguments: Mapping[str, object], *, call_id: str = "") -> ToolResult: ...

    def resources(self) -> Sequence[Closable]: ...

    def close(self) -> None: ...


class ToolRegistry:
    """An explicit capability catalog shared by direct pipeline calls and agent execution.

    Names resolve only against registered objects. Registration never imports packages, and a
    proposed tool name never selects a Python attribute or changes host authority. Finish active
    calls before closing the registry or its owning Memory.
    """

    def __init__(self, tools: Sequence[RegisteredTool] = ()) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._state = _ToolState()
        self._lock = RLock()
        for tool in tools:
            self.register(tool)

    @property
    def closed(self) -> bool:
        return self._state.closed

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        with self._lock:
            return tuple(tool.definition for tool in self._tools.values())

    def register(self, tool: RegisteredTool) -> None:
        with self._lock:
            if self.closed:
                raise ValidationError("tool registry is closed")
            name = tool.definition.name
            if name in self._tools:
                raise ValidationError("tool name is already registered")
            self._tools[name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        """Look up an explicitly registered object without invoking it."""
        with self._lock:
            return self._tools.get(name)

    def execute(self, call: ToolCall) -> ToolResult:
        """Resolve a model-proposed call without granting discovery or import authority."""
        return self.call(call.name, call.arguments, call_id=call.id)

    def call(self, name: str, arguments: Mapping[str, object], *, call_id: str = "") -> ToolResult:
        """Call the same registered capability directly from an application pipeline."""
        _validate_tool_name(name)
        _validate_tool_call_id(call_id)
        with self._lock:
            if self.closed:
                return ToolResult(
                    call_id,
                    name,
                    error=ToolError("registry_closed", "The tool registry is closed."),
                )
            tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                call_id,
                name,
                error=ToolError("unknown_tool", "The requested tool is not registered."),
            )
        try:
            result = tool.call(arguments, call_id=call_id)
        except Exception:
            return ToolResult(
                call_id, name, error=ToolError("execution_failed", "Tool execution failed.")
            )
        if not isinstance(result, ToolResult) or (result.call_id, result.name) != (call_id, name):
            return ToolResult(
                call_id,
                name,
                error=ToolError("invalid_output", "Tool output did not match its call."),
            )
        if result.error is not None:
            return ToolResult(
                call_id,
                name,
                error=ToolError(result.error.code, _ERROR_MESSAGES[result.error.code]),
            )
        return result

    def resources(self) -> tuple[Closable, ...]:
        with self._lock:
            return (self._state, *self._tools.values())

    def close(self) -> None:
        if not self.closed:
            _close_tool_resources(self.resources())
