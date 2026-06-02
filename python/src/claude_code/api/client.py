"""Anthropic Messages API client.

Public surface:
    APIClient            — Protocol that the agent loop depends on
    AnthropicClient      — concrete implementation
    from_env()           — build a client from env vars
    messages_from_history() — convert internal Message objects to API shape
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

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
