"""The agent loop: think → act → observe → repeat.

This is the heart of the system. Everything else exists to support it.

Design contract:
    The agent loop knows about three things only:
        1. An ``APIClient`` (the LLM)
        2. A ``ToolRegistry`` (what the LLM can call)
        3. A ``Context`` (the conversation history)

    It does NOT know about the console, MCP servers, or skills — those
    configure the above three before the loop starts.

Loop structure::

    while turn < MAX_TURNS:
        if context.needs_compact():
            context.compact()
        turn = api.create_message(...)
        if not turn.has_tool_calls:
            return turn.text              # done
        results = execute_tools(turn.tool_calls)
        context.add_assistant_turn(turn, results)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from ..api.client import APIClient, messages_from_history
from ..tools.base import ToolRegistry, registry
from ..tools.builtin import is_read_only
from ..tools.permissions import (
    PermissionChecker,
    PermissionDecision,
)
from .context import Context
from .types import AssistantTurn, ToolResult

log = logging.getLogger(__name__)


# Optional hook the UI can plug in to print progress (tool calls, results, etc.)
ProgressHook = Callable[[str, dict], None]


@dataclass
class AgentConfig:
    model: str
    max_turns: int = 50
    max_tool_parallel: int = 5


class Agent:
    """Drive the think-act-observe loop.

    Construct one per session, then call :meth:`run` for each user prompt.
    The same agent reuses its ``Context`` and tool registry across turns,
    so the conversation accumulates naturally.
    """

    def __init__(
        self,
        *,
        api: APIClient,
        context: Context,
        config: AgentConfig,
        tool_registry: ToolRegistry | None = None,
        permission_checker: PermissionChecker | None = None,
        on_progress: ProgressHook | None = None,
    ) -> None:
        self.api = api
        self.context = context
        self.config = config
        self.tools = tool_registry or registry
        self.permissions = permission_checker or PermissionChecker()
        self.on_progress = on_progress or (lambda event, data: None)

    # ---- main entry point ----

    async def run(self, user_prompt: str) -> str:
        """Send a user prompt; return the final assistant text."""
        self.context.add_user(user_prompt)
        return await self._loop()

    async def continue_session(self) -> str:
        """Continue the conversation without adding a new user message.

        Used after tool execution when we just want the model to keep
        reasoning based on the tool results that were just appended.
        """
        return await self._loop()

    # ---- loop ----

    async def _loop(self) -> str:
        for turn_idx in range(self.config.max_turns):
            if self.context.needs_compact():
                self.on_progress("compacting", {"reason": "threshold reached"})
                self.context.compact()

            self.on_progress("thinking", {"turn": turn_idx})
            assistant_turn = self._call_api()
            self.on_progress("assistant_text", {"text": assistant_turn.text})

            if not assistant_turn.has_tool_calls:
                # Record the final text turn so the model sees its own prior
                # responses if the conversation continues.
                self.context.add_assistant_turn(assistant_turn, [])
                return assistant_turn.text

            results = await self._execute_tools(assistant_turn.tool_calls)
            self.context.add_assistant_turn(assistant_turn, results)
            self.on_progress("tools_done", {"count": len(results)})

        return "[agent stopped: max turns reached]"

    # ---- helpers ----

    def _call_api(self) -> AssistantTurn:
        """Sync wrapper around the API call. SDK call is sync; tool exec is async."""
        return self.api.create_message(
            model=self.config.model,
            system=self.context.system,
            messages=self.context.as_api_messages(),
            tools=[_spec_to_dict(s) for s in self.tools.list_specs()],
        )

    async def _execute_tools(self, calls) -> list[ToolResult]:
        """Execute a batch of tool calls.

        Read-only tools (Read, Grep, Glob) run in parallel up to
        ``max_tool_parallel``. Write tools (Edit, Write, Bash) run
        sequentially to preserve ordering.
        """
        # 1. Filter by permission; unknown tools get a fail result, not a crash
        permitted: list = []
        for call in calls:
            try:
                tool = self.tools.get(call.name)
            except KeyError:
                permitted.append((call, ToolResult.fail(call.id, f"Unknown tool: {call.name}")))
                continue
            decision = self.permissions.check(tool, call.input)
            if decision == PermissionDecision.DENY:
                permitted.append((call, ToolResult.fail(call.id, "Permission denied")))
                continue
            if decision == PermissionDecision.ASK:
                permitted.append((call, ToolResult.fail(call.id, "Permission required (run interactively to grant)")))
                continue
            permitted.append((call, None))

        async def _run(call) -> ToolResult:
            tool = self.tools.get(call.name)
            self.on_progress("tool_call", {"name": call.name, "input": call.input})
            result = await tool.run(call.id, call.input)
            self.on_progress("tool_result", {"name": call.name, "is_error": result.is_error})
            return result

        # Sequential first, then parallel reads
        results_by_id: dict[str, ToolResult] = {}

        # Pre-collect denied/asked results
        for call, pr in permitted:
            if pr is not None:
                results_by_id[call.id] = pr

        # Order: writes first (serially), then reads (in parallel batches)
        writes = [c for c, pr in permitted if pr is None and not is_read_only(c.name)]
        reads = [c for c, pr in permitted if pr is None and is_read_only(c.name)]

        for call in writes:
            results_by_id[call.id] = await _run(call)

        # Batch parallel reads
        for i in range(0, len(reads), self.config.max_tool_parallel):
            batch = reads[i : i + self.config.max_tool_parallel]
            batch_results = await asyncio.gather(*[_run(c) for c in batch])
            for call, res in zip(batch, batch_results):
                results_by_id[call.id] = res

        # 3. Return in original order
        return [results_by_id[c.id] for c in calls]


def _spec_to_dict(spec) -> dict:
    """ToolSpec → dict in the Anthropic tool schema format."""
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
    }
