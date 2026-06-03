"""Backend selection — picks the right API client from environment.

The agent loop only ever sees an ``APIClient`` (Protocol). This module is
the one place that knows which concrete implementation to instantiate.
Keeping the switch here means new backends (Azure, Bedrock, local llama.cpp,
etc.) only have to be wired in once.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from .client import AnthropicClient, OpenAIClient

if TYPE_CHECKING:
    from .client import APIClient

Backend = Literal["openai", "anthropic"]


def select_format() -> Backend:
    """Decide which backend to use.

    Priority:
        1. Explicit ``API_FORMAT`` env var (``openai`` or ``anthropic``).
        2. Auto-detect: ``OPENAI_BASE_URL`` containing ``:1234`` (LM Studio's
           default port) → OpenAI.
        3. Default → Anthropic.
    """
    explicit = os.environ.get("API_FORMAT", "").strip().lower()
    if explicit in ("openai", "anthropic"):
        return explicit
    if ":1234" in os.environ.get("OPENAI_BASE_URL", ""):
        return "openai"
    return "anthropic"


def default_model_for(backend: Backend) -> str:
    """Backend-appropriate default model name.

    Falls back to the other backend's ``*_MODEL`` env var so users only
    have to set one of them.
    """
    if backend == "openai":
        return (
            os.environ.get("OPENAI_MODEL")
            or os.environ.get("ANTHROPIC_MODEL")
            or OpenAIClient.DEFAULT_MODEL
        )
    return (
        os.environ.get("ANTHROPIC_MODEL")
        or AnthropicClient.DEFAULT_MODEL
    )


def build_client(backend: Backend | None = None) -> "APIClient":
    """Instantiate the right client.

    ``backend`` overrides ``select_format()`` — useful for tests.
    """
    chosen = backend or select_format()
    if chosen == "openai":
        return OpenAIClient(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
    return AnthropicClient(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )


def build_model(backend: Backend | None = None) -> str:
    """Resolve the model name, optionally honoring a ``--model`` CLI override."""
    chosen = backend or select_format()
    return default_model_for(chosen)
