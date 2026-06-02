"""Pluggable tool system.

Add a new tool by subclassing ``Tool`` and decorating with ``@register_tool``.
The registry is a global singleton — tools self-register at import time, so
the agent loop can introspect what's available without explicit wiring.
"""

from .base import Tool, ToolRegistry, registry, register_tool
from . import builtin  # noqa: F401 — import for side effects (registers builtins)
from .permissions import PermissionChecker, PermissionDecision

__all__ = [
    "Tool",
    "ToolRegistry",
    "registry",
    "register_tool",
    "PermissionChecker",
    "PermissionDecision",
]
