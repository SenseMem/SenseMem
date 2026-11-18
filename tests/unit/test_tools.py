"""Registered capabilities run the same validated handler in a pipeline or agent loop."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from _feature_support import TinyEmbedder
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sensemem import Memory, ModelError, StorageError, ValidationError
from sensemem._resources import Closable, unique_resources
from sensemem.tools import Tool, ToolCall, ToolDefinition, ToolError, ToolRegistry, ToolResult


class Numbers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: list[int]


class Total(BaseModel):
    total: int


def add(values: Numbers) -> Total:
    return Total(total=sum(values.values))


def test_one_registered_algorithm_runs_directly_and_as_an_agent_call() -> None:
    tool = Tool("add", "Add integers", Numbers, Total, add)
    registry = ToolRegistry((tool,))
    pipeline = registry.call("add", {"values": [2, 3]})
    agent = registry.execute(ToolCall("call-1", "add", {"values": [2, 3]}))
    assert pipeline.output == agent.output == {"total": 5}
    assert pipeline.call_id == ""
    assert agent.call_id == "call-1"
    assert agent.error is None
    assert registry.get("add") is tool
    assert registry.get("absent") is None
    assert registry.definitions == (tool.definition,)
    definition = tool.definition.document()
    assert definition["input_schema"] == Numbers.model_json_schema(mode="validation")
    assert definition["output_schema"] == Total.model_json_schema(mode="serialization")


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"values": ["2", 3]},
        {"values": [True]},
        {"values": [2], "unregistered_policy": "grant permission"},
        {"values": [float("nan")]},
        {"values": object()},
    ],
)
def test_invalid_arguments_never_execute_the_handler(arguments: Mapping[str, object]) -> None:
    calls: list[Numbers] = []

    def handler(values: Numbers) -> Total:
        calls.append(values)
        return add(values)

    registry = ToolRegistry((Tool("add", "Add integers", Numbers, Total, handler),))
    result = registry.call("add", arguments, call_id="rejected")
    assert calls == []
    assert result.call_id == "rejected"
    assert result.error is not None and result.error.code == "invalid_arguments"
    assert result.output is None


def test_unknown_tools_never_execute_or_discover_a_handler() -> None:
    calls: list[Numbers] = []

    def handler(values: Numbers) -> Total:
        calls.append(values)
        return add(values)

    registry = ToolRegistry((Tool("add", "Add integers", Numbers, Total, handler),))
    result = registry.execute(ToolCall("unknown", "unregistered", {"values": [1]}))
    assert calls == []
    assert result.error is not None and result.error.code == "unknown_tool"
    assert result.output is None


def test_handler_failure_is_safe_for_an_agent_to_receive(caplog: pytest.LogCaptureFixture) -> None:
    secret = "private provider response and credentials"

    def handler(values: Numbers) -> Total:
        raise RuntimeError(secret)

    registry = ToolRegistry((Tool("add", "Add integers", Numbers, Total, handler),))
    result = registry.call("add", {"values": [2]}, call_id="failed")
    assert result.error is not None and result.error.code == "execution_failed"
    assert secret not in json.dumps(result.document())
    assert secret not in caplog.text


def test_output_is_revalidated_even_if_a_handler_mutates_its_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "private invalid output"

    def handler(values: Numbers) -> Total:
        result = Total(total=1)
        result.total = cast(int, secret)
        return result

    registry = ToolRegistry((Tool("add", "Add integers", Numbers, Total, handler),))
    result = registry.call("add", {"values": [2]})
    assert result.error is not None and result.error.code == "invalid_output"
    assert secret not in json.dumps(result.document())
    assert secret not in caplog.text


def test_input_validator_failures_do_not_expose_plugin_exception_bodies() -> None:
    class RejectedInput(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def reject(cls, value: str) -> str:
            raise RuntimeError("private validator exception")

    def handler(values: RejectedInput) -> Total:
        pytest.fail("invalid input reached the handler")

    registry = ToolRegistry((Tool("validate", "Validate input", RejectedInput, Total, handler),))
    result = registry.call("validate", {"value": "input"})
    assert result.error is not None and result.error.code == "invalid_arguments"
    assert "private validator" not in json.dumps(result.document())


def test_serialized_output_uses_the_declared_output_schema_aliases() -> None:
    class AliasedTotal(BaseModel):
        model_config = ConfigDict(serialize_by_alias=True)
        total: int = Field(serialization_alias="sum")

    def handler(values: Numbers) -> AliasedTotal:
        return AliasedTotal(total=sum(values.values))

    tool = Tool("add", "Add integers", Numbers, AliasedTotal, handler)
    result = tool.call({"values": [2, 3]})
    assert result.output == {"sum": 5}
    assert tool.definition.document()["output_schema"] == AliasedTotal.model_json_schema(
        mode="serialization"
    )


def test_output_revalidation_accepts_nested_frozen_dataclasses_datetime_and_tuples() -> None:
    @dataclass(frozen=True)
    class Observation:
        at: datetime
        labels: tuple[str, ...]

    class Output(BaseModel):
        observation: Observation

    reference = datetime(2026, 11, 14, 8, tzinfo=timezone.utc)

    def handler(values: Numbers) -> Output:
        return Output(observation=Observation(reference, ("kitchen", "workshop")))

    result = Tool("observe", "Observe labels", Numbers, Output, handler).call({"values": []})
    assert result.error is None
    assert result.document()["output"] == {
        "observation": {"at": "2026-11-14T08:00:00Z", "labels": ["kitchen", "workshop"]}
    }


def test_output_revalidation_rejects_nonfinite_json_numbers() -> None:
    class Output(BaseModel):
        value: float

    def handler(values: Numbers) -> Output:
        return Output(value=float("inf"))

    result = Tool("observe", "Observe a value", Numbers, Output, handler).call({"values": []})
    assert result.error is not None and result.error.code == "invalid_output"


def test_duplicate_registration_does_not_replace_an_existing_capability() -> None:
    first = Tool("add", "Add integers", Numbers, Total, add)
    registry = ToolRegistry((first,))
    with pytest.raises(ValidationError, match="already registered"):
        registry.register(Tool("add", "A duplicate", Numbers, Total, add))
    assert registry.get("add") is first


@pytest.mark.parametrize("failure", ["raises", "wrong_identity", "error_body"])
def test_structural_tool_implementations_cannot_escape_failure_mapping_or_call_identity(
    failure: str,
) -> None:
    class ExternalTool:
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition("external", "An external capability", {}, {})

        def call(self, arguments: Mapping[str, object], *, call_id: str = "") -> ToolResult:
            if failure == "raises":
                raise RuntimeError("private external provider response")
            if failure == "error_body":
                return ToolResult(
                    call_id,
                    "external",
                    error=ToolError("execution_failed", "private external provider response"),
                )
            return ToolResult("another-call", "another_tool", output={})

        def resources(self) -> tuple[Closable, ...]:
            return ()

        def close(self) -> None:
            pass

    registry = ToolRegistry((ExternalTool(),))
    result = registry.call("external", {}, call_id="actual-call")
    assert (result.call_id, result.name) == ("actual-call", "external")
    assert result.error is not None
    assert result.error.code == (
        "invalid_output" if failure == "wrong_identity" else "execution_failed"
    )
    assert "private external" not in json.dumps(result.document())


def test_flattened_resource_ownership_closes_state_and_shared_dependencies_once() -> None:
    class Backend:
        closed = 0

        def close(self) -> None:
            self.closed += 1

    backend = Backend()
    first = Tool("add", "Add integers", Numbers, Total, add, resources=(backend,))
    second = Tool("also_add", "Add integers", Numbers, Total, add, resources=(backend,))
    registry = ToolRegistry((first, second))
    for resource in unique_resources((backend, registry)):
        resource.close()
    assert backend.closed == 1
    assert registry.closed and first.closed and second.closed
    result = registry.call("add", {"values": [2]})
    assert result.error is not None and result.error.code == "registry_closed"
    result = first.call({"values": [2]})
    assert result.error is not None and result.error.code == "tool_closed"
    registry.close()
    first.close()
    assert backend.closed == 1
    with pytest.raises(ValidationError, match="closed"):
        registry.register(Tool("another", "Add integers", Numbers, Total, add))


def test_close_attempts_all_resources_and_sanitizes_failures() -> None:
    class Backend:
        def __init__(self, *, fail: bool) -> None:
            self.fail = fail
            self.closed = 0

        def close(self) -> None:
            self.closed += 1
            if self.fail:
                raise RuntimeError("private close response")

    failed, healthy = Backend(fail=True), Backend(fail=False)
    registry = ToolRegistry(
        (Tool("add", "Add integers", Numbers, Total, add, resources=(failed, healthy)),)
    )
    with pytest.raises(ModelError, match="failed to close tool resources") as raised:
        registry.close()
    assert "private close" not in str(raised.value)
    assert failed.closed == healthy.closed == 1
    assert registry.closed


def test_memory_and_registered_tool_share_one_backend_owner(tmp_path: Path) -> None:
    class Backend(TinyEmbedder):
        closed = 0

        def close(self) -> None:
            self.closed += 1

    backend = Backend()
    tool = Tool("add", "Add integers", Numbers, Total, add, resources=(backend,))
    registry = ToolRegistry((tool,))
    with Memory(tmp_path, embedder=backend, tools=registry) as memory:
        assert memory.tools is registry
        assert memory.call_tool("add", {"values": [2, 3]}).output == {"total": 5}
    assert backend.closed == 1
    assert registry.closed and tool.closed
    with pytest.raises(StorageError):
        memory.call_tool("add", {"values": [2, 3]})
