"""Public tool contracts preserve schemas, JSON identity, and immutable snapshots."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel, RootModel

from sensemem import (
    Blob,
    EmbedTask,
    Memory,
    Modality,
    ModelInput,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolError,
    ToolRegistry,
    ToolResult,
    ValidationError,
    memory_tools,
)


def test_definition_call_and_result_documents_are_independent_json_snapshots() -> None:
    schema: dict[str, object] = {"type": "object", "properties": {"value": {"type": "integer"}}}
    arguments: dict[str, object] = {"nested": {"values": [1, 2]}}
    definition = ToolDefinition("lookup", "Look up a value", schema, schema)
    call = ToolCall("call-1", "lookup", arguments)
    result = ToolResult("call-1", "lookup", output=arguments)
    arguments["nested"] = "changed"
    schema["properties"] = {}
    assert call.document() == {
        "id": "call-1",
        "name": "lookup",
        "arguments": {"nested": {"values": [1, 2]}},
    }
    assert result.document() == {
        "call_id": "call-1",
        "name": "lookup",
        "output": {"nested": {"values": [1, 2]}},
        "error": None,
    }
    assert definition.document()["input_schema"] == {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }
    assert json.loads(json.dumps(result.document())) == result.document()
    document = call.document()
    cast(dict[str, object], document["arguments"])["nested"] = "changed again"
    assert call.arguments["nested"] != "changed again"
    with pytest.raises(TypeError):
        cast(dict[str, object], call.arguments)["nested"] = "changed"
    nested = cast(Mapping[str, object], call.arguments["nested"])
    assert nested["values"] == (1, 2)
    with pytest.raises(TypeError):
        cast(dict[str, object], nested)["values"] = ()
    with pytest.raises(FrozenInstanceError):
        definition.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("name", ["", " ", "package.function", "../code", "x" * 65])
def test_tool_names_are_portable_explicit_identifiers(name: str) -> None:
    with pytest.raises(ValidationError, match="tool name"):
        ToolCall("call", name, {})


@pytest.mark.parametrize("arguments", [{"value": float("inf")}, {"value": b"bytes"}])
def test_call_values_require_finite_json_data(arguments: Mapping[str, object]) -> None:
    with pytest.raises(ValidationError, match="JSON data"):
        ToolCall("call", "lookup", arguments)


def test_results_require_exactly_one_output_or_error() -> None:
    error = ToolError("unknown_tool", "The requested tool is not registered.")
    with pytest.raises(ValidationError, match="exactly one"):
        ToolResult("call", "lookup")
    with pytest.raises(ValidationError, match="exactly one"):
        ToolResult("call", "lookup", output={}, error=error)
    assert ToolResult("call", "lookup", error=error).document()["error"] == {
        "code": "unknown_tool",
        "message": "The requested tool is not registered.",
    }


def test_scalar_model_contracts_are_rejected_before_registration() -> None:
    class Input(BaseModel):
        value: int

    class Output(RootModel[int]):
        pass

    def handler(value: Input) -> Output:
        return Output(value.value)

    with pytest.raises(ValidationError, match="JSON objects"):
        Tool("scalar", "Return an integer", Input, Output, handler)


def test_public_registration_produces_a_reachable_algorithm_without_optional_dependencies() -> None:
    class Input(BaseModel):
        text: str

    class Output(BaseModel):
        characters: int

    def count(value: Input) -> Output:
        return Output(characters=len(value.text))

    registry = ToolRegistry()
    registry.register(Tool("count", "Count text characters", Input, Output, count))
    result = registry.execute(ToolCall("call-1", "count", {"text": "SenseMem"}))
    assert result.output == {"characters": 8}
    assert result.error is None
    assert registry.definitions[0].name == "count"
    registry.close()


class _MemoryToolEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.VIDEO})
    embedding_model = embedding_space = "memory-tool-contract:v1"
    embedding_dimension = 2

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _value in inputs)

    def close(self) -> None:
        pass


def test_compile_tool_separates_complete_sources_from_text_only_delivery(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=_MemoryToolEmbedder()) as memory:
        complete = memory.add("Inspect valve V-62 tomorrow.")
        projected = memory.add(
            (
                "Inspect valve V-62 in both clips.",
                Blob(b"first clip", "video/mp4"),
                Blob(b"second clip", "video/mp4"),
            )
        )
        for tool in memory_tools(memory):
            memory.tools.register(tool)
        definition = next(
            item.document() for item in memory.tools.definitions if item.name == "compile_context"
        )
        schema = definition["output_schema"]
        assert isinstance(schema, dict)
        assert set(schema["required"]) == {
            "context",
            "source_ids",
            "excerpt_source_ids",
            "projection_source_ids",
            "omitted",
        }
        result = memory.tools.execute(ToolCall("compile", "compile_context", {"goal": "V-62"}))
        assert result.error is None and result.output is not None
        assert result.output["source_ids"] == (complete.id,)
        assert result.output["excerpt_source_ids"] == ()
        assert result.output["projection_source_ids"] == (projected.id,)
        rendered = result.output["context"]
        assert isinstance(rendered, str)
        assert "Inspect valve V-62 in both clips." in rendered
        assert "native media omitted" in rendered and "not a complete proof" in rendered
