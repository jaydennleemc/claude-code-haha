"""Anthropic API boundary.

Thin wrapper around the official SDK. The wrapper exists for three reasons:

1. **Streaming aggregation** — we turn SSE events into a single
   ``AssistantTurn`` with text + tool_calls, so the agent loop never has to
   touch the event stream.
2. **Token counting** — Anthropic exposes a ``count_tokens`` endpoint; we
   surface it for the context window manager.
3. **Testability** — the agent loop depends on the ``APIClient`` protocol,
   not on this concrete class, so tests can swap in a fake.
"""

from .client import (
    APIClient,
    AnthropicClient,
    from_env,
    messages_from_history,
)

__all__ = ["APIClient", "AnthropicClient", "from_env", "messages_from_history"]
