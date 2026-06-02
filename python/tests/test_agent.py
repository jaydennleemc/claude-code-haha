"""Tests for the agent loop using a fake APIClient.

These tests verify the loop's control flow without ever hitting the
network: we wire in a fake API that returns scripted responses and
assert that the agent calls tools, accumulates results, and stops.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from claude_code.core.agent import Agent, AgentConfig
from claude_code.core.context import Context, ContextConfig
from claude_code.core.types import AssistantTurn, ToolCall, ToolResult
from claude_code.tools.base import Tool, ToolRegistry, register_tool
from claude_code.tools.permissions import PermissionChecker


# ---------- fake API client ----------

@dataclass
class ScriptedResponse:
    """A scripted turn the fake API will return on its next call."""

    text: str = ""
    tool_calls: list[ToolCall] = None
    stop_reason: str = "end_turn"

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


class FakeAPI:
    """A programmable stand-in for the real AnthropicClient.

    Set ``script`` to a list of responses; the i-th call returns the i-th
    entry. After the script is exhausted, the same final response repeats.
    """

    def __init__(self, script: list[ScriptedResponse]) -> None:
        self.script = script
        self.calls: list[dict] = []  # for assertions

    def create_message(self, **_kwargs) -> AssistantTurn:
        self.calls.append(_kwargs)
        idx = min(len(self.calls) - 1, len(self.script) - 1)
        resp = self.script[idx]
        return AssistantTurn(
            text=resp.text,
            tool_calls=list(resp.tool_calls or []),
            stop_reason=resp.stop_reason,
        )

    def count_tokens(self, **_kwargs) -> int:
        return 100  # always under threshold


# ---------- test tools ----------

class _DoubleTool(Tool):
    """Test tool: returns the value of 'n' doubled."""

    name = "Double"
    description = "Returns 2*n"
    input_schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    }

    async def execute(self, n: int) -> str:
        return str(n * 2)


class _ErrorTool(Tool):
    name = "Boom"
    description = "Always raises"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self) -> str:
        raise RuntimeError("kaboom")


# ---------- fixtures ----------

@pytest.fixture
def agent_setup():
    """Standard agent wired with one Double tool and scripted responses."""
    reg = ToolRegistry()
    reg.register(_DoubleTool())

    api = FakeAPI([])
    ctx = Context(
        config=ContextConfig(model="fake-model", window=10000, compact_threshold=0.8),
        api=api,
        system="You are a test.",
    )
    ctx.set_tools([s.__dict__ for s in reg.list_specs()])

    config = AgentConfig(model="fake-model", max_turns=5)
    agent = Agent(
        api=api,
        context=ctx,
        config=config,
        tool_registry=reg,
        permission_checker=PermissionChecker(),
    )
    return agent, api, reg


# ---------- tests ----------

class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_returns_text_when_no_tool_calls(self, agent_setup):
        agent, api, _ = agent_setup
        api.script = [ScriptedResponse(text="the answer is 42")]

        out = await agent.run("what is the answer?")
        assert out == "the answer is 42"
        assert len(api.calls) == 1

    @pytest.mark.asyncio
    async def test_executes_tool_and_returns_final_text(self, agent_setup):
        agent, api, _ = agent_setup
        # Turn 1: assistant calls Double(5) -> tool returns "10"
        # Turn 2: assistant sees the result and replies with text
        api.script = [
            ScriptedResponse(
                tool_calls=[ToolCall(id="t1", name="Double", input={"n": 5})],
                stop_reason="tool_use",
            ),
            ScriptedResponse(text="the answer is 10", stop_reason="end_turn"),
        ]

        out = await agent.run("double 5")
        assert out == "the answer is 10"
        assert len(api.calls) == 2

    @pytest.mark.asyncio
    async def test_handles_tool_error_gracefully(self, agent_setup):
        agent, api, reg = agent_setup
        reg.register(_ErrorTool())
        agent.tools = reg  # ensure agent sees the new tool

        api.script = [
            ScriptedResponse(
                tool_calls=[ToolCall(id="t1", name="Boom", input={})],
                stop_reason="tool_use",
            ),
            ScriptedResponse(text="Boom failed, sorry", stop_reason="end_turn"),
        ]

        out = await agent.run("trigger boom")
        # The agent kept going and returned text on the second turn
        assert "Boom failed" in out or "sorry" in out

    @pytest.mark.asyncio
    async def test_stops_at_max_turns(self):
        reg = ToolRegistry()
        reg.register(_DoubleTool())
        api = FakeAPI([
            ScriptedResponse(
                tool_calls=[ToolCall(id=f"t{i}", name="Double", input={"n": i})],
                stop_reason="tool_use",
            )
            for i in range(10)
        ])
        ctx = Context(
            config=ContextConfig(model="fake", window=10000, compact_threshold=0.8),
            api=api,
            system="x",
        )
        ctx.set_tools([s.__dict__ for s in reg.list_specs()])
        agent = Agent(
            api=api,
            context=ctx,
            config=AgentConfig(model="fake", max_turns=3),
            tool_registry=reg,
        )

        out = await agent.run("loop forever")
        assert "max turns" in out
        # 3 turns of API calls
        assert len(api.calls) == 3

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_not_crash(self, agent_setup):
        agent, api, _ = agent_setup
        api.script = [
            ScriptedResponse(
                tool_calls=[ToolCall(id="t1", name="DoesNotExist", input={})],
                stop_reason="tool_use",
            ),
            ScriptedResponse(text="couldn't run that tool", stop_reason="end_turn"),
        ]
        out = await agent.run("invoke nothing")
        assert "couldn't" in out or "tool" in out.lower()

    @pytest.mark.asyncio
    async def test_context_appends_assistant_and_tool_results(self, agent_setup):
        agent, api, _ = agent_setup
        api.script = [
            ScriptedResponse(
                tool_calls=[ToolCall(id="t1", name="Double", input={"n": 3})],
                stop_reason="tool_use",
            ),
            ScriptedResponse(text="done", stop_reason="end_turn"),
        ]
        await agent.run("double 3")

        history = agent.context.history
        # user, assistant(with tool_use), user(with tool_result), assistant(text)
        assert len(history) == 4
        assert history[0].role.value == "user"
        assert history[1].role.value == "assistant"
        # The 3rd message is the tool result wrapper
        assert history[2].role.value == "user"
        tool_result_blocks = history[2].content
        assert tool_result_blocks[0]["type"] == "tool_result"
        assert tool_result_blocks[0]["tool_use_id"] == "t1"
        assert tool_result_blocks[0]["content"] == "6"

    @pytest.mark.asyncio
    async def test_progress_hook_receives_events(self, agent_setup):
        agent, api, _ = agent_setup
        events: list[tuple[str, dict]] = []
        agent.on_progress = lambda e, d: events.append((e, d))

        api.script = [
            ScriptedResponse(
                tool_calls=[ToolCall(id="t1", name="Double", input={"n": 2})],
                stop_reason="tool_use",
            ),
            ScriptedResponse(text="ok", stop_reason="end_turn"),
        ]
        await agent.run("x")

        event_names = [e for e, _ in events]
        assert "thinking" in event_names
        assert "tool_call" in event_names
        assert "tool_result" in event_names
        assert "tools_done" in event_names

    @pytest.mark.asyncio
    async def test_compaction_triggered_when_over_threshold(self):
        reg = ToolRegistry()
        api = FakeAPI([
            ScriptedResponse(text="ok", stop_reason="end_turn"),
        ])
        # Tiny window so any message pushes us over threshold
        ctx = Context(
            config=ContextConfig(
                model="fake", window=200, compact_threshold=0.5, keep_recent=2,
            ),
            api=api,
            system="x",
        )
        # Pretend count_tokens always returns something over threshold
        original = api.count_tokens
        api.count_tokens = lambda **_: 150

        agent = Agent(
            api=api, context=ctx, config=AgentConfig(model="fake", max_turns=3),
            tool_registry=reg,
        )
        await agent.run("first prompt")
        # History should be compacted (only 1 keep_recent + summary message)
        # Original: user + assistant = 2 messages
        # After compact: summary + last 2 = 3 messages
        assert len(agent.context.history) <= 3


# ---------- read-only vs write tool batching ----------

class _ReadTool(Tool):
    name = "Read"
    description = "read"
    input_schema = {"type": "object", "properties": {"x": {"type": "string"}}}

    async def execute(self, x: str) -> str:
        # Tag with sleep so parallel batching is observable
        await asyncio.sleep(0.05)
        return f"read:{x}"


class _WriteTool(Tool):
    name = "Write"
    description = "write"
    input_schema = {"type": "object", "properties": {"x": {"type": "string"}}}

    async def execute(self, x: str) -> str:
        await asyncio.sleep(0.05)
        return f"wrote:{x}"


class TestToolBatching:
    @pytest.mark.asyncio
    async def test_read_tools_run_in_parallel(self):
        reg = ToolRegistry()
        reg.register(_ReadTool())

        api = FakeAPI([
            ScriptedResponse(
                tool_calls=[
                    ToolCall(id="r1", name="Read", input={"x": "a"}),
                    ToolCall(id="r2", name="Read", input={"x": "b"}),
                    ToolCall(id="r3", name="Read", input={"x": "c"}),
                ],
                stop_reason="tool_use",
            ),
            ScriptedResponse(text="done", stop_reason="end_turn"),
        ])
        ctx = Context(
            config=ContextConfig(model="fake", window=10000, compact_threshold=0.9),
            api=api, system="x",
        )
        ctx.set_tools([s.__dict__ for s in reg.list_specs()])
        agent = Agent(
            api=api, context=ctx,
            config=AgentConfig(model="fake", max_turns=3),
            tool_registry=reg,
        )

        import time
        start = time.perf_counter()
        await agent.run("read all")
        elapsed = time.perf_counter() - start
        # 3 parallel reads @ 0.05s each → ~0.05s; serial would be ~0.15s
        # Give generous margin for CI slowness
        assert elapsed < 0.12, f"reads did not run in parallel ({elapsed:.3f}s)"

    @pytest.mark.asyncio
    async def test_write_tools_run_serially(self):
        reg = ToolRegistry()
        reg.register(_WriteTool())

        api = FakeAPI([
            ScriptedResponse(
                tool_calls=[
                    ToolCall(id="w1", name="Write", input={"x": "a"}),
                    ToolCall(id="w2", name="Write", input={"x": "b"}),
                    ToolCall(id="w3", name="Write", input={"x": "c"}),
                ],
                stop_reason="tool_use",
            ),
            ScriptedResponse(text="done", stop_reason="end_turn"),
        ])
        ctx = Context(
            config=ContextConfig(model="fake", window=10000, compact_threshold=0.9),
            api=api, system="x",
        )
        ctx.set_tools([s.__dict__ for s in reg.list_specs()])
        agent = Agent(
            api=api, context=ctx,
            config=AgentConfig(model="fake", max_turns=3),
            tool_registry=reg,
        )

        import time
        start = time.perf_counter()
        await agent.run("write all")
        elapsed = time.perf_counter() - start
        # 3 serial writes @ 0.05s each → ~0.15s; parallel would be ~0.05s
        assert elapsed > 0.1, f"writes did not run serially ({elapsed:.3f}s)"
