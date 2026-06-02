"""Console UI — the simplest possible terminal renderer.

We don't use Rich or any color library. Just ANSI codes and a couple of
helpers. The point is that the UI layer is replaceable: swap this for a
web frontend, an IDE extension, or a test recorder without touching the
agent loop.
"""

from .console import Console

__all__ = ["Console"]
