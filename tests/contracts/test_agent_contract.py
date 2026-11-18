"""Immutable agent values and the supported package imports."""

from dataclasses import FrozenInstanceError
from inspect import iscoroutinefunction

import pytest

from sensemem import (
    AgentBackend,
    AgentResult,
    AgentRunner,
    AgentStep,
    AgentTurn,
    OpenAIAgent,
    ToolCall,
    ToolError,
    ToolResult,
    ValidationError,
)


def test_agent_values_require_final_content_or_calls() -> None:
    call = ToolCall("call-1", "read", {})
    with pytest.raises(ValidationError):
        AgentStep()
    with pytest.raises(ValidationError):
        AgentStep(content="answer", calls=(call,))
    with pytest.raises(ValidationError):
        AgentStep(calls=(call, call))
    with pytest.raises(ValidationError):
        AgentStep(calls=(ToolCall("", "read", {}),))
    with pytest.raises(FrozenInstanceError):
        AgentStep(content="answer").content = "changed"  # type: ignore[misc]


def test_agent_history_binds_results_to_exact_tool_calls() -> None:
    call = ToolCall("call-1", "read", {})
    with pytest.raises(ValidationError):
        AgentTurn(AgentStep(calls=(call,)), ())
    with pytest.raises(ValidationError):
        AgentTurn(AgentStep(calls=(call,)), (ToolResult("other", "read", output={}),))
    turn = AgentTurn(
        AgentStep(calls=(call,)),
        (ToolResult("call-1", "read", error=ToolError("unknown_tool", "Tool is unavailable.")),),
    )
    result = AgentResult("unavailable", (turn,))
    assert result.steps == 2
    assert result.tool_calls == 1
    assert AgentBackend.__name__ == "AgentBackend"
    assert isinstance(OpenAIAgent, type)
    assert isinstance(AgentRunner, type)
    assert iscoroutinefunction(AgentRunner.run_async)
