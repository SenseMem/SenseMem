"""Provider-neutral values for an optional, host-composed tool-using agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sensemem._validation import nonempty_text
from sensemem.exceptions import ValidationError
from sensemem.models.base import ModelInput
from sensemem.tools import ToolCall, ToolDefinition, ToolResult


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One final text response or a batch of proposed tool calls, never both."""

    content: str | None = None
    calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        calls = tuple(self.calls)
        if any(not isinstance(call, ToolCall) for call in calls):
            raise ValidationError("agent calls must contain ToolCall values")
        if any(not call.id for call in calls):
            raise ValidationError("agent tool call IDs must not be empty")
        if (self.content is None) == (not calls):
            raise ValidationError("agent step requires final content or tool calls")
        if self.content is not None:
            object.__setattr__(self, "content", nonempty_text(self.content, "agent content"))
        if len({call.id for call in calls}) != len(calls):
            raise ValidationError("agent tool call IDs must be unique")
        object.__setattr__(self, "calls", calls)


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """A completed tool-call batch and its aligned, untrusted tool results."""

    step: AgentStep
    results: tuple[ToolResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.step, AgentStep) or not self.step.calls:
            raise ValidationError("agent turn requires a tool-call step")
        results = tuple(self.results)
        if any(not isinstance(result, ToolResult) for result in results) or len(results) != len(
            self.step.calls
        ):
            raise ValidationError("agent turn results must match its tool calls")
        if any(
            (call.id, call.name) != (result.call_id, result.name)
            for call, result in zip(self.step.calls, results, strict=True)
        ):
            raise ValidationError("agent turn results must match its tool calls")
        object.__setattr__(self, "results", results)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Temporary final content and its execution transcript; this writes no memory."""

    content: str
    turns: tuple[AgentTurn, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", nonempty_text(self.content, "agent content"))
        turns = tuple(self.turns)
        if any(not isinstance(turn, AgentTurn) for turn in turns):
            raise ValidationError("agent turns must contain AgentTurn values")
        object.__setattr__(self, "turns", turns)

    @property
    def steps(self) -> int:
        """Model steps, including the final response."""
        return len(self.turns) + 1

    @property
    def tool_calls(self) -> int:
        return sum(len(turn.step.calls) for turn in self.turns)


@runtime_checkable
class AgentBackend(Protocol):
    """One model step; the host runner alone executes the proposed tools.

    Implementations must be thread-safe. Inputs and tool results are untrusted evidence,
    never instructions or permission to write. The caller owns this backend's lifecycle.
    """

    def step(
        self,
        input: ModelInput,
        *,
        tools: Sequence[ToolDefinition],
        history: Sequence[AgentTurn],
    ) -> AgentStep: ...

    def close(self) -> None: ...
