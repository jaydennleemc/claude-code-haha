"""Anthropic Messages API client.

Public surface:
    APIClient                    — Protocol that the agent loop depends on
    AnthropicClient              — concrete Anthropic implementation
    OpenAIClient                 — concrete OpenAI-compatible implementation (LM Studio, etc.)
    from_env()                   — build an AnthropicClient from env vars
    from_env_openai()            — build an OpenAIClient from env vars
    messages_from_history()      — convert internal Message objects to Anthropic API shape
    messages_from_history_openai() — same, but to OpenAI /chat/completions shape
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import openai

from ..core.types import AssistantTurn, ToolCall


# ---------- Public protocol (the contract the agent loop depends on) ----------

class APIClient(Protocol):
    """The minimum surface the agent loop needs from an API backend.

    Any object that exposes these methods can drive the agent — swap to
    OpenAI, a local model, or a mock for tests.
    """

    def create_message(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 8192,
    ) -> AssistantTurn: ...

    def count_tokens(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> int: ...


# ---------- Anthropic implementation ----------

@dataclass
class _PendingToolCall:
    """A tool_use block being assembled from streaming deltas."""

    id: str
    name: str
    input_json_parts: list[str]


class AnthropicClient:
    """Anthropic Messages API client with streaming aggregation."""

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    DEFAULT_MAX_TOKENS = 8192

    def __init__(
        self,
        api_key: str | None = None,
        auth_token: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if not api_key and not auth_token:
            raise ValueError(
                "Either ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN must be set"
            )
        self._client = anthropic.Anthropic(
            api_key=api_key,
            auth_token=auth_token,
            base_url=base_url,
        )

    # ---- public API ----

    def create_message(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> AssistantTurn:
        """Send a Messages request and return the aggregated turn.

        Always uses streaming — gives us progress events for free if a UI
        is attached, and the SDK handles both modes the same way internally.
        """
        text_chunks: list[str] = []
        tool_calls: dict[str, _PendingToolCall] = {}
        stop_reason = "end_turn"

        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools or [],
        ) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_calls[block.id] = _PendingToolCall(
                            id=block.id,
                            name=block.name,
                            input_json_parts=[],
                        )
                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_chunks.append(delta.text)
                    elif delta.type == "input_json_delta":
                        if tool_calls:
                            pending = list(tool_calls.values())[-1]
                            pending.input_json_parts.append(delta.partial_json)
                elif etype == "message_delta":
                    stop_reason = event.delta.stop_reason or stop_reason

        # Reassemble
        calls: list[ToolCall] = []
        for pending in tool_calls.values():
            try:
                input_obj = json.loads("".join(pending.input_json_parts))
            except Exception:
                input_obj = {}
            calls.append(ToolCall(id=pending.id, name=pending.name, input=input_obj))

        return AssistantTurn(
            text="".join(text_chunks),
            tool_calls=calls,
            stop_reason=stop_reason,
        )

    def count_tokens(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        """Use Anthropic's count_tokens endpoint for an exact count.

        Falls back to a coarse estimate if the endpoint errors (some custom
        base URLs don't implement it).
        """
        try:
            resp = self._client.messages.count_tokens(
                model=model,
                system=system,
                messages=messages,
                tools=tools or [],
            )
            return resp.input_tokens
        except Exception:
            # Estimate: ~4 chars per token, plus a flat tool-overhead term.
            char_count = len(system) + sum(
                len(str(m.get("content", ""))) for m in messages
            )
            tool_overhead = sum(len(str(t)) for t in tools) * 2
            return (char_count + tool_overhead) // 4


# ---------- helpers ----------

def from_env() -> AnthropicClient:
    """Construct a client from environment variables."""
    return AnthropicClient(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )


def messages_from_history(history: list[Any]) -> list[dict]:
    """Convert internal Message objects to the dict shape Anthropic wants."""
    out: list[dict] = []
    for m in history:
        out.append({"role": m.role.value, "content": m.content})
    return out


# ---------- OpenAI-compatible implementation (LM Studio, OpenRouter, etc.) ----------

# Map OpenAI finish_reason → our internal stop_reason vocabulary.
_OPENAI_STOP_REASON = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-shaped tool specs to OpenAI's wrapped form.

    Anthropic:  [{"name", "description", "input_schema"}]
    OpenAI:     [{"type": "function", "function": {"name", "description", "parameters"}}]
    """
    out: list[dict] = []
    for t in tools or []:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return out


def messages_from_history_openai(history: list[dict]) -> list[dict]:
    """Convert agent-loop messages (Anthropic shape) to OpenAI /chat/completions shape.

    The agent loop passes in ``Context.as_api_messages()`` output: a list of
    ``{"role", "content"}`` dicts where ``content`` for assistant messages
    is a list of ``{"type": "text"|"tool_use", ...}`` blocks and for
    follow-up user messages is a list of ``{"type": "tool_result", ...}``
    blocks. OpenAI wants tool results as standalone ``role: tool`` messages
    and tool invocations in a top-level ``tool_calls`` array.
    """
    out: list[dict] = []
    for m in history:
        role = m.get("role")
        content = m.get("content")
        if role == "user" and isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                btype = block.get("type")
                if btype == "tool_result":
                    payload = block.get("content", "")
                    if not isinstance(payload, str):
                        payload = json.dumps(payload, ensure_ascii=False)
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": payload,
                        }
                    )
                elif btype == "text":
                    text_parts.append(block.get("text", ""))
            if text_parts:
                out.append({"role": "user", "content": "\n".join(text_parts)})
        elif role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(
                                    block.get("input", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
            msg: dict = {"role": "assistant", "content": "\n".join(text_parts)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        else:
            out.append({"role": role, "content": content})
    return out


class OpenAIClient:
    """OpenAI-compatible chat completions client (LM Studio, OpenRouter, OpenAI, etc.).

    Accepts the same Anthropic-shaped tool specs and internal ``Message``
    objects as ``AnthropicClient``; conversion to the OpenAI wire format
    happens inside this class. The agent loop never has to know.
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_MAX_TOKENS = 8192

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # LM Studio accepts any non-empty key; OpenAI requires a real one.
        # We fall back to a placeholder rather than raising so that LM Studio
        # users don't have to set anything.
        self._client = openai.OpenAI(
            api_key=api_key or "lm-studio",
            base_url=base_url,
        )

    # ---- public API ----

    def create_message(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> AssistantTurn:
        openai_messages: list[dict] = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages_from_history_openai(messages))

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": openai_messages,
            "stream": True,
        }
        openai_tools = _anthropic_tools_to_openai(tools)
        if openai_tools:
            kwargs["tools"] = openai_tools

        text_chunks: list[str] = []
        # Accumulate tool calls by their streaming index. OpenAI may split
        # one tool call across many chunks and may interleave multiple
        # parallel calls in the same chunk, so we key on ``index``.
        tool_acc: dict[int, dict] = {}
        stop_reason = "end_turn"

        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is None:
                continue

            if delta.content:
                text_chunks.append(delta.content)

            for tc in (delta.tool_calls or []):
                idx = tc.index
                bucket = tool_acc.setdefault(
                    idx, {"id": "", "name": "", "arguments_parts": []}
                )
                if tc.id:
                    bucket["id"] = tc.id
                fn = tc.function
                if fn and fn.name:
                    bucket["name"] = fn.name
                if fn and fn.arguments:
                    bucket["arguments_parts"].append(fn.arguments)

            if choice.finish_reason:
                stop_reason = _OPENAI_STOP_REASON.get(
                    choice.finish_reason, choice.finish_reason
                )

        calls: list[ToolCall] = []
        for idx in sorted(tool_acc):
            b = tool_acc[idx]
            raw = "".join(b["arguments_parts"])
            try:
                input_obj = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                input_obj = {}
            calls.append(ToolCall(id=b["id"], name=b["name"], input=input_obj))

        return AssistantTurn(
            text="".join(text_chunks),
            tool_calls=calls,
            stop_reason=stop_reason,
        )

    def count_tokens(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        # OpenAI has no exact count endpoint; mirror AnthropicClient's
        # chars/4 + tool-overhead heuristic so the compaction threshold
        # behaves consistently across backends.
        char_count = len(system) + sum(
            len(str(m.get("content", ""))) for m in messages
        )
        tool_overhead = sum(len(str(t)) for t in tools) * 2
        return (char_count + tool_overhead) // 4


def from_env_openai() -> OpenAIClient:
    """Construct an OpenAIClient from environment variables.

    Recognized vars:
        OPENAI_API_KEY     — required by OpenAI; LM Studio accepts any string
        OPENAI_BASE_URL    — e.g. http://localhost:1234/v1 (LM Studio default)
    """
    return OpenAIClient(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
