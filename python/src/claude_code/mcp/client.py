"""High-level MCP client.

Each :class:`MCPClient` owns one stdio transport (one server process). On
connect it does the MCP ``initialize`` handshake, asks for ``tools/list``,
and wraps every returned tool as a local :class:`Tool` that plugs into our
normal tool registry — so the LLM sees one unified toolset.

Spec reference: https://modelcontextprotocol.io/specification
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from ..core.types import ToolResult
from ..tools.base import Tool
from .transport import StdioTransport, TransportError

log = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for spawning one MCP server.

    Attributes:
        name: Human-readable label, also used to namespace tool names.
        command: Executable + args, e.g. ``["npx", "-y", "@mcp/server-git"]``.
        env: Extra env vars merged with ``os.environ`` when spawning.
    """

    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)


def parse_server_configs(raw: str | None) -> list[MCPServerConfig]:
    """Parse the ``MCP_SERVERS`` env var.

    Format: JSON array of ``{"name": ..., "command": ..., "args": [...], "env": {...}}``.
    Invalid JSON is logged and ignored — better to start with no servers
    than to crash on a typo.
    """
    if not raw or not raw.strip():
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("MCP_SERVERS: invalid JSON, ignoring: %s", e)
        return []
    out: list[MCPServerConfig] = []
    for item in items:
        try:
            cmd = [item["command"], *item.get("args", [])]
            out.append(
                MCPServerConfig(
                    name=item["name"],
                    command=cmd,
                    env=item.get("env", {}),
                )
            )
        except KeyError as e:
            log.warning("MCP server config missing key %s, skipping: %s", e, item)
    return out


class _MCPToolAdapter(Tool):
    """Wraps an MCP server tool so it behaves like a local :class:`Tool`.

    The tool's name is namespaced as ``mcp__{server}__{tool}`` to avoid
    collisions with built-in tools.
    """

    def __init__(self, server_name: str, tool_def: dict, client: "MCPClient") -> None:
        self._server_name = server_name
        self._def = tool_def
        self._client = client
        # Override the class attributes
        self.name = f"mcp__{server_name}__{tool_def['name']}"
        self.description = tool_def.get("description", "")
        self.input_schema = tool_def.get("inputSchema", {"type": "object", "properties": {}})

    async def execute(self, **kwargs: Any) -> str:
        return await self._client.call_tool(self._def["name"], kwargs)


class MCPClient:
    """One MCP server connection.

    Use :meth:`connect_and_register` to do the full handshake + tool
    discovery. The returned list of :class:`Tool` instances can be fed
    to the tool registry.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._transport: StdioTransport | None = None
        self._initialized = False
        self._tools_cache: list[dict] = []

    # ---- lifecycle ----

    async def connect_and_register(self) -> list[Tool]:
        """Spawn the server, initialize, list tools, return local Tool wrappers."""
        merged_env = {**os.environ, **self.config.env}
        self._transport = StdioTransport(self.config.command, env=merged_env)
        await self._transport.start()

        try:
            # MCP initialize handshake
            await self._transport.request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "claude-code-py", "version": "0.1.0"},
                },
            )
            await self._transport.notify("notifications/initialized")
            self._initialized = True

            # Discover tools
            result = await self._transport.request("tools/list", {})
            self._tools_cache = result.get("tools", [])
            log.info(
                "MCP '%s': discovered %d tools: %s",
                self.config.name,
                len(self._tools_cache),
                [t["name"] for t in self._tools_cache],
            )

            return [
                _MCPToolAdapter(self.config.name, tdef, self)
                for tdef in self._tools_cache
            ]
        except Exception:
            await self._transport.close()
            raise

    async def close(self) -> None:
        if self._transport:
            await self._transport.close()
            self._transport = None

    # ---- public methods used by the adapter ----

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Invoke a tool on the server and return its text content."""
        if not self._transport or not self._initialized:
            raise TransportError(f"MCP client '{self.config.name}' not connected")
        result = await self._transport.request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=60.0,
        )
        # MCP returns content blocks; we flatten text blocks into a string
        content = result.get("content", [])
        parts: list[str] = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                # Non-text blocks: serialize as JSON so the model still sees them
                parts.append(json.dumps(block, ensure_ascii=False))
        is_error = result.get("isError", False)
        text = "\n".join(parts) if parts else "<no output>"
        if is_error:
            raise RuntimeError(f"MCP tool '{name}' failed: {text}")
        return text


# ---------- convenience: connect to many servers ----------

async def connect_all(configs: list[MCPServerConfig]) -> list[tuple[MCPServerConfig, list[Tool] | Exception]]:
    """Connect to a list of servers in parallel, returning (config, result) pairs.

    Failures are isolated — one broken server doesn't kill the others.
    """
    async def _one(cfg: MCPServerConfig):
        client = MCPClient(cfg)
        try:
            tools = await client.connect_and_register()
            return (cfg, tools)
        except Exception as e:
            await client.close()
            return (cfg, e)

    return await asyncio.gather(*[_one(c) for c in configs])
