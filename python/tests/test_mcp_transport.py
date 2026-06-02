"""Tests for the MCP JSON-RPC stdio transport.

We spawn a real subprocess (a tiny Python script that speaks JSON-RPC) so
the wire format and framing get exercised end-to-end. No mocks for the
transport itself — that would be testing our mock.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from claude_code.mcp.transport import StdioTransport, TransportError


# A tiny echo server that responds to JSON-RPC requests.
# Written as a literal script (not a file) so the test is self-contained.
ECHO_SERVER = """
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" in msg:
        # echo back with result
        reply = {"jsonrpc": "2.0", "id": msg["id"], "result": {"echo": msg.get("params")}}
    else:
        # notification — no reply
        continue
    sys.stdout.write(json.dumps(reply) + "\\n")
    sys.stdout.flush()
"""


@pytest.fixture
def echo_server_path(tmp_path) -> Path:
    p = tmp_path / "echo_server.py"
    p.write_text(ECHO_SERVER)
    return p


class TestStdioTransport:
    @pytest.mark.asyncio
    async def test_request_returns_result(self, echo_server_path):
        transport = StdioTransport([sys.executable, str(echo_server_path)])
        await transport.start()
        try:
            result = await transport.request("ping", {"hello": "world"})
            assert result == {"echo": {"hello": "world"}}
        finally:
            await transport.close()

    @pytest.mark.asyncio
    async def test_multiple_sequential_requests(self, echo_server_path):
        transport = StdioTransport([sys.executable, str(echo_server_path)])
        await transport.start()
        try:
            r1 = await transport.request("a", {"n": 1})
            r2 = await transport.request("b", {"n": 2})
            assert r1["echo"]["n"] == 1
            assert r2["echo"]["n"] == 2
        finally:
            await transport.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, echo_server_path):
        transport = StdioTransport([sys.executable, str(echo_server_path)])
        await transport.start()
        await transport.close()
        await transport.close()  # should not raise

    @pytest.mark.asyncio
    async def test_request_after_close_raises(self, echo_server_path):
        transport = StdioTransport([sys.executable, str(echo_server_path)])
        await transport.start()
        await transport.close()
        with pytest.raises(TransportError):
            await transport.request("x", {})

    @pytest.mark.asyncio
    async def test_spawn_missing_command_raises(self):
        transport = StdioTransport(["/nonexistent/binary/that/does/not/exist"])
        with pytest.raises(Exception):
            await transport.start()
