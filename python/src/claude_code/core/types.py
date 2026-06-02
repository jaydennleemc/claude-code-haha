"""Core type definitions.

We use dataclasses rather than pydantic to keep the dependency surface small.
The shapes here mirror what the Anthropic Messages API expects/returns, so
the ``api/client.py`` layer can map directly without translation gymnastics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message role in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"


class ToolResultStatus(str, Enum):
    """Outcome of a tool invocation."""

    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ToolCall:
    """A single tool invocation requested by the assistant.

    Attributes:
        id: Anthropic-assigned tool_use block id; must be echoed back in the
            tool_result message.
        name: The tool's registered name (e.g. ``"Read"``).
        input: Parsed JSON object matching the tool's input_schema.
    """

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a ToolCall.

    Returned to the model as the ``content`` of a ``tool_result`` block.
    """

    tool_call_id: str
    content: str
    is_error: bool = False

    @classmethod
    def ok(cls, tool_call_id: str, content: str) -> "ToolResult":
        return cls(tool_call_id=tool_call_id, content=content, is_error=False)

    @classmethod
    def fail(cls, tool_call_id: str, error: str) -> "ToolResult":
        return cls(tool_call_id=tool_call_id, content=error, is_error=True)


@dataclass
class AssistantTurn:
    """One assistant response from the API.

    Contains the streamed text and any tool calls the model wants to make.
    Exactly one of ``text`` / ``tool_calls`` is typically non-empty, but
    the model can emit both (text + parallel tools) in a single turn.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_tokens

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class Message:
    """A message in the conversation history.

    For user messages, ``content`` is a string. For assistant messages it's
    a list of content blocks (text + tool_use + tool_result). We keep the
    raw structure so the API client can pass it through unchanged.
    """

    role: Role
    content: Any  # str for user, list[dict] for assistant/tool
