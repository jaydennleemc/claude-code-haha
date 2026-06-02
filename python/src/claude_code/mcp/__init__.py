"""Model Context Protocol (MCP) client.

Implements the JSON-RPC 2.0 stdio transport from the MCP spec. Each
MCPClient spawns one server subprocess, discovers its tools, and wraps
each as a local :class:`Tool` instance that plugs into our normal
tool registry.

Reference: https://modelcontextprotocol.io
"""

from .client import MCPClient, MCPServerConfig, parse_server_configs
from . import transport  # noqa: F401 — re-exported for advanced use

__all__ = ["MCPClient", "MCPServerConfig", "parse_server_configs"]
