"""API boundary.

Thin wrappers around the Anthropic and OpenAI SDKs. The wrappers exist for
three reasons:

1. **Streaming aggregation** — we turn SSE events into a single
   ``AssistantTurn`` with text + tool_calls, so the agent loop never has to
   touch the event stream.
2. **Token counting** — Anthropic exposes a ``count_tokens`` endpoint; we
   surface it (or a coarse estimate) for the context window manager.
3. **Testability** — the agent loop depends on the ``APIClient`` protocol,
   not on a concrete class, so tests can swap in a fake.

Use ``api.factory.build_client()`` to pick the right backend from env vars.
"""

from .client import (
    APIClient,
    AnthropicClient,
    OpenAIClient,
    from_env,
    from_env_openai,
    messages_from_history,
    messages_from_history_openai,
)
from .factory import (
    Backend,
    build_client,
    build_model,
    default_model_for,
    select_format,
)

__all__ = [
    "APIClient",
    "AnthropicClient",
    "OpenAIClient",
    "Backend",
    "build_client",
    "build_model",
    "default_model_for",
    "from_env",
    "from_env_openai",
    "messages_from_history",
    "messages_from_history_openai",
    "select_format",
]
