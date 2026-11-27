"""Bounded execution of a host-selected agent and tool registry."""

import asyncio

from sensemem._validation import positive_int
from sensemem.agents import AgentBackend, AgentResult, AgentStep, AgentTurn
from sensemem.exceptions import ModelError, SenseMemError, ValidationError
from sensemem.models.base import ModelInput
from sensemem.tools import ToolRegistry


class AgentRunner:
    """Run optional tool reasoning without storage authority or resource ownership.

    The caller owns both backend and tools. A registry may therefore be shared with
    a fixed pipeline or Memory without this runner closing it or any nested model twice.
    """

    def __init__(
        self,
        backend: AgentBackend,
        tools: ToolRegistry,
        *,
        max_steps: int = 8,
        max_tool_calls: int = 16,
    ) -> None:
        if not isinstance(backend, AgentBackend):
            raise ValidationError("backend must implement AgentBackend")
        self._backend = backend
        self._tools = tools
        self._max_steps = positive_int(max_steps, "max_steps")
        self._max_tool_calls = positive_int(max_tool_calls, "max_tool_calls")

    async def run_async(self, input: ModelInput) -> AgentResult:
        """Run off-thread, draining started work before cancellation releases borrowed resources.

        Synchronous model and tool calls still require their own timeouts: cancelling an async
        waiter cannot interrupt a running thread. Cancellation wins over the drained result.
        """
        worker = asyncio.create_task(asyncio.to_thread(self.run, input))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not worker.cancelled():
                worker.exception()
            raise

    def run(self, input: ModelInput) -> AgentResult:
        """Return temporary content, or fail before a call batch would exceed either bound."""
        if not isinstance(input, ModelInput):
            raise ValidationError("agent input must be a ModelInput")
        history: list[AgentTurn] = []
        called: set[str] = set()
        definitions = self._tools.definitions
        for index in range(self._max_steps):
            try:
                step = self._backend.step(input, tools=definitions, history=tuple(history))
            except ValidationError:
                raise ModelError(
                    "agent returned invalid output", reason="response_invalid", stage="agent"
                ) from None
            except SenseMemError:
                raise
            except Exception:
                raise ModelError(
                    "agent step failed", reason="model_failed", stage="agent"
                ) from None
            if not isinstance(step, AgentStep):
                raise ModelError(
                    "agent returned invalid output", reason="response_invalid", stage="agent"
                )
            if step.content is not None:
                return AgentResult(step.content, tuple(history))
            identifiers = {call.id for call in step.calls}
            if identifiers & called:
                raise ModelError(
                    "agent repeated a tool call ID", reason="response_invalid", stage="agent"
                )
            # Do not execute tools when there is no remaining model step to consume their results.
            if index + 1 == self._max_steps or len(called) + len(step.calls) > self._max_tool_calls:
                raise ModelError(
                    "agent execution limit exceeded", reason="agent_limit_exceeded", stage="agent"
                )
            results = tuple(self._tools.execute(call) for call in step.calls)
            called.update(identifiers)
            history.append(AgentTurn(step, results))
        raise AssertionError("agent step bound must terminate the loop")
