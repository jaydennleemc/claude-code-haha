"""Conversation context with token counting and compaction.

Responsibilities:
    * Store the message history (user + assistant + tool_result blocks).
    * Ask the API for an exact token count via ``count_tokens``.
    * Compaction: when usage exceeds the threshold, summarize older turns
      while preserving the most recent N messages verbatim.

This is a deliberately tiny implementation. Real Claude Code has
micro-compact, reactive-compact, dream-compact, snip-compact, etc. We ship
just ``compact()`` — the policy can be layered on top.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .types import AssistantTurn, Message, Role, ToolResult


class _Countable(Protocol):
    def count_tokens(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> int: ...


@dataclass
class ContextConfig:
    """Tunables for the context window manager."""

    model: str
    window: int = 200_000            # model context window in tokens
    compact_threshold: float = 0.8   # 80% of window
    keep_recent: int = 6             # verbatim messages to keep after compact
    reserve_for_output: int = 8_192  # tokens to leave free for the response


@dataclass
class Context:
    """The conversation history plus token accounting.

    Tools are passed in so the token count includes their definitions.
    The system prompt is stored separately so the LLM always sees the
    current version (skills might add to it at runtime).
    """

    config: ContextConfig
    api: _Countable
    system: str = ""
    history: list[Message] = field(default_factory=list)
    tools_schema: list[dict] = field(default_factory=list)
    _last_compact_summary: str | None = None
    _token_count: int = 0

    # ---- mutation ----

    def add_user(self, text: str) -> None:
        self.history.append(Message(role=Role.USER, content=text))

    def add_assistant_turn(self, turn: AssistantTurn, tool_results: list[ToolResult] | None = None) -> None:
        """Append one assistant turn to history.

        Anthropic requires tool_use blocks and their tool_result blocks to
        live in adjacent messages (assistant → user). We always emit the
        assistant message; the optional user-followup carries tool_results.
        """
        # Build assistant content blocks
        blocks: list[dict] = []
        if turn.text:
            blocks.append({"type": "text", "text": turn.text})
        for tc in turn.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                }
            )
        self.history.append(Message(role=Role.ASSISTANT, content=blocks))

        if tool_results:
            result_blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": tr.content,
                    "is_error": tr.is_error,
                }
                for tr in tool_results
            ]
            self.history.append(Message(role=Role.USER, content=result_blocks))

    def set_tools(self, tools_schema: list[dict]) -> None:
        self.tools_schema = tools_schema

    def set_system(self, system: str) -> None:
        self.system = system

    # ---- queries ----

    def as_api_messages(self) -> list[dict]:
        """Return history in the shape the Anthropic API wants."""
        return [{"role": m.role.value, "content": m.content} for m in self.history]

    def token_count(self) -> int:
        """Exact token count from the API; cached per call to ``add_*`` reset."""
        if self._token_count:
            return self._token_count
        self._token_count = self.api.count_tokens(
            model=self.config.model,
            system=self.system,
            messages=self.as_api_messages(),
            tools=self.tools_schema,
        )
        return self._token_count

    def usage_ratio(self) -> float:
        usable = self.config.window - self.config.reserve_for_output
        if usable <= 0:
            return 1.0
        return self.token_count() / usable

    def needs_compact(self) -> bool:
        return self.usage_ratio() >= self.config.compact_threshold

    # ---- compaction ----

    def compact(self, summarizer: "_Summarizer | None" = None) -> None:
        """Replace the oldest messages with a single summary message.

        ``summarizer`` is a callable that takes the text to summarize and
        returns a shorter string. If not provided, we do a dumb
        truncation-and-flag (preserves the most recent ``keep_recent``
        messages and drops the rest).

        The summary is itself stored in history as a user message prefixed
        with a clear "[Earlier context summary]" marker, so the model knows
        the prior conversation was compressed.
        """
        if len(self.history) <= self.config.keep_recent:
            return

        keep = self.config.keep_recent
        to_summarize = self.history[:-keep]
        kept = self.history[-keep:]

        if summarizer is not None:
            summary_text = summarizer(self._flatten(to_summarize))
        else:
            # Naive: keep the first user message + count of dropped turns
            first_user = next(
                (m for m in to_summarize if m.role == Role.USER and isinstance(m.content, str)),
                None,
            )
            snippet = (first_user.content if first_user else "")[:200]
            summary_text = (
                f"[Earlier context summary: {len(to_summarize)} prior messages were "
                f"compacted to save tokens. Last user message: '{snippet}...']"
            )

        self.history = [
            Message(role=Role.USER, content=summary_text),
            *kept,
        ]
        self._token_count = 0  # invalidate cache

    def _flatten(self, messages: list[Message]) -> str:
        """Concatenate messages into a single string for summarization."""
        parts: list[str] = []
        for m in messages:
            if isinstance(m.content, str):
                parts.append(f"{m.role.value}: {m.content}")
            else:
                parts.append(f"{m.role.value}: {json.dumps(m.content, ensure_ascii=False)}")
        return "\n\n".join(parts)


# Callable alias for the summarizer hook
_Summarizer = Any  # Callable[[str], str]
