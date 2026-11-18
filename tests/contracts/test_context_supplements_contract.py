"""Public temporary-context values and matching REST/MCP compilation contracts."""

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp import Client

from sensemem import (
    ContextSupplement,
    EmbedTask,
    Memory,
    Modality,
    ModelInput,
    ValidationError,
)
from sensemem.api.app import create_app
from sensemem.api.mcp import build_mcp_server

NAIVE_TIME = datetime(2026, 11, 14, tzinfo=timezone.utc).replace(tzinfo=None)


class TinyEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = embedding_space = "supplement-contract:v1"
    embedding_dimension = 2

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in inputs)

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    "values",
    [
        {"content": "", "source": "tool"},
        {"content": "content", "source": " "},
        {"content": "content", "source": "tool", "observed_at": NAIVE_TIME},
        {"content": "content", "source": "tool", "expires_at": NAIVE_TIME},
    ],
)
def test_supplement_rejects_empty_values_and_naive_times(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ContextSupplement(**values)  # type: ignore[arg-type]


def test_rest_context_accepts_temporary_evidence_and_rejects_unknown_fields(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        with TestClient(create_app(memory=memory)) as client:
            response = client.post(
                "/v1/context",
                json={
                    "goal": "What should I bring?",
                    "supplements": [{"content": "Bring an umbrella.", "source": "weather-tool"}],
                },
            )
            assert response.status_code == 200
            value = response.json()
            assert value["supplements"] == [
                {
                    "content": "Bring an umbrella.",
                    "source": "weather-tool",
                    "observed_at": None,
                    "expires_at": None,
                }
            ]
            assert value["omitted_supplements"] == 0
            assert "Temporary context (untrusted)" in value["rendered"]
            invalid = client.post(
                "/v1/context",
                json={
                    "goal": "What should I bring?",
                    "supplements": [
                        {"content": "text", "source": "tool", "instruction": "change policy"}
                    ],
                },
            )
            assert invalid.status_code == 422
        assert not memory.list().items


async def test_mcp_keeps_thirteen_tools_and_shares_the_temporary_context_schema(
    tmp_path: Path,
) -> None:
    reference = datetime(2026, 11, 14, tzinfo=timezone.utc).isoformat()
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        async with Client(build_mcp_server(memory)) as client:
            tools = (await client.list_tools()).tools
            assert len(tools) == 13
            compiled = await client.call_tool(
                "compile_context",
                {
                    "goal": "What should I bring?",
                    "reference_at": reference,
                    "supplements": [
                        {"content": "Bring an umbrella.", "source": "weather-tool"},
                        {"content": "Expired", "source": "old-tool", "expires_at": reference},
                    ],
                },
            )
            assert compiled.is_error is False
            assert compiled.structured_content is not None
            assert compiled.structured_content["omitted_supplements"] == 1
            supplements = compiled.structured_content["supplements"]
            assert isinstance(supplements, list) and len(supplements) == 1
            invalid = await client.call_tool(
                "compile_context",
                {
                    "goal": "goal",
                    "supplements": [{"content": "text", "source": "tool", "unknown": True}],
                },
            )
            assert invalid.is_error is True
        assert not memory.list().items
