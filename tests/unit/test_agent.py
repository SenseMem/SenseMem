"""Execution checks for independently composed agents and reusable tools."""

import asyncio
from collections.abc import Sequence
from threading import Event
from traceback import format_exception

import pytest
from pydantic import BaseModel, ConfigDict

from sensemem.agents import AgentStep, AgentTurn
from sensemem.exceptions import ModelError
from sensemem.kernel.agent import AgentRunner
from sensemem.models.base import ModelInput
from sensemem.tools import Tool, ToolCall, ToolDefinition, ToolRegistry


class _Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: int


class _Output(BaseModel):
    doubled: int


class _Backend:
    def __init__(self, *steps: AgentStep) -> None:
        self.steps = list(steps)
        self.seen: list[tuple[ModelInput, tuple[ToolDefinition, ...], tuple[AgentTurn, ...]]] = []
        self.closed = 0

    def step(
        self,
        input: ModelInput,
        *,
        tools: Sequence[ToolDefinition],
        history: Sequence[AgentTurn],
    ) -> AgentStep:
        self.seen.append((input, tuple(tools), tuple(history)))
        return self.steps.pop(0)

    def close(self) -> None:
        self.closed += 1


def test_agent_repairs_unknown_tools_and_invalid_arguments_from_results() -> None:
    tool = Tool(
        "double",
        "Double a supplied integer.",
        _Input,
        _Output,
        lambda x: _Output(doubled=2 * x.value),
    )
    registry = ToolRegistry((tool,))
    backend = _Backend(
        AgentStep(calls=(ToolCall("c1", "missing", {}),)),
        AgentStep(calls=(ToolCall("c2", "double", {"value": "wrong"}),)),
        AgentStep(calls=(ToolCall("c3", "double", {"value": 7}),)),
        AgentStep(content="14"),
    )
    result = AgentRunner(backend, registry).run(ModelInput(text="Double seven."))

    assert result.content == "14"
    assert result.steps == 4
    assert result.tool_calls == 3
    assert result.turns[0].results[0].error is not None
    assert result.turns[0].results[0].error.code == "unknown_tool"
    assert result.turns[1].results[0].error is not None
    assert result.turns[1].results[0].error.code == "invalid_arguments"
    assert result.turns[2].results[0].output == {"doubled": 14}
    assert backend.seen[-1][2] == result.turns
    assert backend.seen[0][1][0].name == "double"
    assert backend.closed == 0
    assert registry.call("double", {"value": 2}).output == {"doubled": 4}


@pytest.mark.parametrize("max_steps,max_tool_calls", [(1, 4), (4, 1)])
def test_agent_bounds_stop_before_executing_an_unconsumable_batch(
    max_steps: int, max_tool_calls: int
) -> None:
    executed: list[int] = []

    def double(value: _Input) -> _Output:
        executed.append(value.value)
        return _Output(doubled=value.value * 2)

    registry = ToolRegistry((Tool("double", "Double.", _Input, _Output, double),))
    backend = _Backend(
        AgentStep(
            calls=(ToolCall("c1", "double", {"value": 1}), ToolCall("c2", "double", {"value": 2}))
        )
    )
    with pytest.raises(ModelError) as failure:
        AgentRunner(backend, registry, max_steps=max_steps, max_tool_calls=max_tool_calls).run(
            ModelInput(text="Use the tool.")
        )
    assert failure.value.reason == "agent_limit_exceeded"
    assert executed == []


def test_agent_stops_repeated_call_ids_without_reexecuting() -> None:
    executed: list[int] = []

    def double(value: _Input) -> _Output:
        executed.append(value.value)
        return _Output(doubled=value.value * 2)

    registry = ToolRegistry((Tool("double", "Double.", _Input, _Output, double),))
    step = AgentStep(calls=(ToolCall("repeat", "double", {"value": 1}),))
    with pytest.raises(ModelError) as failure:
        AgentRunner(_Backend(step, step), registry).run(ModelInput(text="Use the tool."))
    assert failure.value.reason == "response_invalid"
    assert executed == [1]


def test_agent_model_failure_is_translated_without_its_body() -> None:
    class BrokenBackend(_Backend):
        def step(
            self,
            input: ModelInput,
            *,
            tools: Sequence[ToolDefinition],
            history: Sequence[AgentTurn],
        ) -> AgentStep:
            raise RuntimeError("private provider body")

    with pytest.raises(ModelError) as failure:
        AgentRunner(BrokenBackend(), ToolRegistry()).run(ModelInput(text="Answer."))
    assert failure.value.reason == "model_failed"
    assert "private provider body" not in "".join(format_exception(failure.value))


async def test_arun_cancellation_waits_for_borrowed_tool_resources() -> None:
    entered, release, finished = Event(), Event(), Event()

    class Resource:
        closed = 0

        def close(self) -> None:
            self.closed += 1

    resource = Resource()

    def double(value: _Input) -> _Output:
        entered.set()
        assert release.wait(timeout=5)
        assert resource.closed == 0
        finished.set()
        return _Output(doubled=value.value * 2)

    registry = ToolRegistry(
        (Tool("double", "Double.", _Input, _Output, double, resources=(resource,)),)
    )
    backend = _Backend(
        AgentStep(calls=(ToolCall("c1", "double", {"value": 7}),)),
        AgentStep(content="14"),
    )
    runner = AgentRunner(backend, registry)
    task = asyncio.create_task(runner.run_async(ModelInput(text="Double seven.")))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not finished.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    assert finished.is_set()
    assert backend.closed == resource.closed == 0
    assert len(backend.seen) == 2
    assert registry.call("double", {"value": 2}).output == {"doubled": 4}
