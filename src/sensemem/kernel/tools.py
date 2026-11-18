"""Lifecycle admission for explicitly registered capability calls through Memory."""

from collections.abc import Mapping

from sensemem.kernel.lifecycle import Lifecycle
from sensemem.tools import ToolRegistry, ToolResult


class Tools:
    def __init__(self, *, registry: ToolRegistry, lifecycle: Lifecycle) -> None:
        self._registry = registry
        self._lifecycle = lifecycle

    def call(self, name: str, arguments: Mapping[str, object], *, call_id: str = "") -> ToolResult:
        """Keep the owning Memory open until its admitted tool call completes."""
        with self._lifecycle.operation(model_work=True):
            return self._registry.call(name, arguments, call_id=call_id)
