"""Tool base class and global registry.

Design notes:
    * Tools are async-first because shell + file I/O should not block the
      event loop when the model is busy thinking.
    * The registry is a process-wide singleton (``registry``). Tools self-
      register on import via the ``@register_tool`` decorator.
    * Tool input validation is delegated to the model via ``input_schema``
      (JSON Schema). We trust the model to send well-formed input; the
      ``execute()`` method just does the work.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..core.types import ToolResult


@dataclass
class ToolSpec:
    """The shape of a tool as exposed to the LLM.

    This mirrors Anthropic's tool definition format so ``api/client.py`` can
    pass instances straight to ``tools=[...]``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


class Tool(ABC):
    """Base class for all tools.

    Subclasses must set the three class attributes and implement ``execute``.
    They are picked up automatically by the ``@register_tool`` decorator.
    """

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Run the tool and return a string result.

        The string is sent back to the model as the tool_result content.
        Raise an exception to signal a hard failure; catch and return an
        error string for soft failures (so the model can self-correct).
        """
        raise NotImplementedError

    def to_spec(self) -> ToolSpec:
        """Return the LLM-facing description of this tool."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    async def run(self, tool_call_id: str, input: dict[str, Any]) -> ToolResult:
        """Validate + execute + wrap in a ToolResult.

        Subclasses usually don't override this — they implement ``execute``.
        """
        try:
            content = await self.execute(**input)
            return ToolResult.ok(tool_call_id, content)
        except Exception as e:  # noqa: BLE001 — we want to catch all tool errors
            return ToolResult.fail(tool_call_id, f"{type(e).__name__}: {e}")


class ToolRegistry:
    """A process-wide mapping of tool name → tool instance.

    Tools self-register at import time. The agent loop reads from
    ``registry.list_specs()`` to tell the model what's available, and
    ``registry.get(name).run(...)`` to execute a call.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        """Register a tool instance. Idempotent — re-registration is a no-op
        unless the existing entry is a different class."""
        existing = self._tools.get(tool.name)
        if existing is not None and type(existing) is not type(tool):
            raise ValueError(
                f"Tool name conflict: '{tool.name}' already registered "
                f"by {type(existing).__name__}, refusing to overwrite "
                f"with {type(tool).__name__}"
            )
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(
                f"Tool '{name}' not found. Available: {sorted(self._tools)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def list_specs(self) -> list[ToolSpec]:
        """Return LLM-facing tool descriptions for the API request."""
        return [t.to_spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def clear(self) -> None:
        """Reset the registry. Test-only — used to isolate test cases."""
        self._tools.clear()


# Global singleton. Import this in the agent loop.
registry = ToolRegistry()


def register_tool(cls: type[Tool]) -> type[Tool]:
    """Class decorator that instantiates and registers a Tool subclass.

    Usage::

        @register_tool
        class ReadTool(Tool):
            name = "Read"
            ...
    """
    instance = cls()
    registry.register(instance)
    return cls


# Re-exported alias for prettier imports elsewhere
ToolFactory = Callable[[], Awaitable[ToolResult]]
