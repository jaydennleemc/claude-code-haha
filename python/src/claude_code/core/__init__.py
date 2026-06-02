"""Core business logic — no I/O, no external services.

The agent loop, context window, and message types live here. Everything in
this package depends only on the ``Tool`` ABC and the ``APIClient`` protocol,
so the whole agent is testable with a fake API.
"""

from .types import Message, Role, ToolCall, ToolResult, AssistantTurn
from .context import Context
from .agent import Agent

__all__ = [
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "AssistantTurn",
    "Context",
    "Agent",
]
