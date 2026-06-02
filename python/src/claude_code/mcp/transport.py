"""JSON-RPC 2.0 over stdio — the MCP wire format.

MCP servers are line-delimited JSON over stdin/stdout. Each line is one
JSON-RPC 2.0 message. We need:

    * a way to send framed requests and correlate responses
    * a way to receive server-initiated notifications (no correlation needed)
    * a single in-flight request at a time per connection (MCP servers
      typically require this; pipelining is allowed but rare)

We use ``asyncio`` subprocess pipes + a background reader task.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


class TransportError(RuntimeError):
    """Raised when the MCP transport fails."""


@dataclass
class JSONRPCRequest:
    """A JSON-RPC 2.0 request to send to the server."""

    id: int
    method: str
    params: dict[str, Any] | None = None

    def encode(self) -> str:
        msg = {"jsonrpc": "2.0", "id": self.id, "method": self.method}
        if self.params is not None:
            msg["params"] = self.params
        return json.dumps(msg)


class StdioTransport:
    """A long-lived JSON-RPC 2.0 connection over a child process's stdio.

    Lifecycle::

        transport = StdioTransport(["node", "server.js"])
        await transport.start()
        try:
            result = await transport.request("initialize", {...})
            ...
        finally:
            await transport.close()
    """

    def __init__(self, command: list[str], env: dict[str, str] | None = None) -> None:
        self.command = command
        self.env = env
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._closed = False
        # Optional subscription for notifications (no id field)
        self.on_notification: Callable[[dict], None] | None = None

    # ---- lifecycle ----

    async def start(self) -> None:
        log.debug("spawning MCP server: %s", self.command)
        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
        # Fail any in-flight requests
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(TransportError("transport closed"))

    # ---- I/O ----

    async def request(self, method: str, params: dict | None = None, timeout: float = 30.0) -> Any:
        """Send a request and await the matching response."""
        if not self._proc or not self._proc.stdin:
            raise TransportError("transport not started")
        req_id = self._next_id
        self._next_id += 1

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        try:
            line = JSONRPCRequest(id=req_id, method=method, params=params).encode() + "\n"
            self._proc.stdin.write(line.encode("utf-8"))
            await self._proc.stdin.drain()
        except Exception:
            self._pending.pop(req_id, None)
            raise TransportError(f"failed to send request: {method}") from None

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TransportError(f"request '{method}' timed out after {timeout}s") from None

    async def notify(self, method: str, params: dict | None = None) -> None:
        """Send a notification (no response expected)."""
        if not self._proc or not self._proc.stdin:
            raise TransportError("transport not started")
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

    # ---- internals ----

    async def _read_loop(self) -> None:
        """Background task: read one JSON message per line, route to handler."""
        assert self._proc and self._proc.stdout
        try:
            while True:
                raw = await self._proc.stdout.readline()
                if not raw:
                    # EOF — server closed stdout
                    log.debug("MCP server stdout closed")
                    return
                try:
                    msg = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    log.warning("MCP: malformed JSON line: %r", raw[:200])
                    continue

                if "id" in msg and ("result" in msg or "error" in msg):
                    self._handle_response(msg)
                else:
                    self._handle_notification(msg)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("MCP reader crashed")

    def _handle_response(self, msg: dict) -> None:
        req_id = msg.get("id")
        fut = self._pending.pop(req_id, None)
        if fut is None or fut.done():
            return
        if "error" in msg:
            err = msg["error"]
            fut.set_exception(
                TransportError(
                    f"RPC error {err.get('code')}: {err.get('message')}"
                )
            )
        else:
            fut.set_result(msg.get("result"))

    def _handle_notification(self, msg: dict) -> None:
        if self.on_notification:
            try:
                self.on_notification(msg)
            except Exception:
                log.exception("notification handler raised")
