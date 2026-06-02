"""Tiny ANSI-color terminal renderer.

Design notes:
    * We don't stream characters (no terminal cursor tricks). Each event
      gets a complete line(s) for grep-friendliness.
    * The :class:`Agent` calls us via the ``on_progress`` hook, so the
      console never has to know about the agent internals.
    * All output is plain ``print()`` with optional ANSI codes — easy to
      redirect, easy to test, easy to disable with ``NO_COLOR=1``.
"""

from __future__ import annotations

import os
import sys
from typing import Any


# ANSI codes — empty strings when stdout is not a TTY or NO_COLOR is set
def _is_color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


_USE_COLOR = _is_color_enabled()


def _wrap(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(text: str) -> str:
    return _wrap("2", text)


def bold(text: str) -> str:
    return _wrap("1", text)


def green(text: str) -> str:
    return _wrap("32", text)


def red(text: str) -> str:
    return _wrap("31", text)


def yellow(text: str) -> str:
    return _wrap("33", text)


def blue(text: str) -> str:
    return _wrap("34", text)


def magenta(text: str) -> str:
    return _wrap("35", text)


def cyan(text: str) -> str:
    return _wrap("36", text)


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


class Console:
    """Render agent events to the terminal.

    The :class:`Agent` calls ``on_progress(event, data)`` for each
    interesting event. We map events to lines of output.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._wrote_assistant_header = False

    # ---- the hook the Agent calls ----

    def on_progress(self, event: str, data: dict[str, Any]) -> None:
        handler = {
            "thinking": self._on_thinking,
            "assistant_text": self._on_assistant_text,
            "tool_call": self._on_tool_call,
            "tool_result": self._on_tool_result,
            "tools_done": self._on_tools_done,
            "compacting": self._on_compacting,
        }.get(event)
        if handler:
            handler(data)

    # ---- event handlers ----

    def _on_thinking(self, data: dict) -> None:
        if self.verbose:
            turn = data.get("turn", 0)
            print(dim(f"  [turn {turn}] thinking…"), file=sys.stderr)

    def _on_assistant_text(self, data: dict) -> None:
        text = data.get("text", "")
        if not text:
            return
        if not self._wrote_assistant_header:
            print()  # blank line before first response
            self._wrote_assistant_header = True
        # Print the assistant's text in default color (no markup)
        print(text, end="", flush=True)

    def _on_tool_call(self, data: dict) -> None:
        name = data.get("name", "?")
        inp = data.get("input", {})
        # Show a compact view of the input
        if "command" in inp:
            preview = _truncate(str(inp["command"]), 80)
            print(f"\n{yellow('→')} {bold(name)} {dim(preview)}", flush=True)
        elif "file_path" in inp:
            print(f"\n{yellow('→')} {bold(name)} {dim(str(inp['file_path']))}", flush=True)
        else:
            preview = _truncate(str(inp), 80)
            print(f"\n{yellow('→')} {bold(name)} {dim(preview)}", flush=True)

    def _on_tool_result(self, data: dict) -> None:
        if not self.verbose:
            return
        name = data.get("name", "?")
        is_error = data.get("is_error", False)
        marker = red("✗") if is_error else green("✓")
        print(f"  {marker} {dim(name)} done", file=sys.stderr, flush=True)

    def _on_tools_done(self, data: dict) -> None:
        # Nothing extra; per-tool results handle their own output in verbose mode
        pass

    def _on_compacting(self, data: dict) -> None:
        print(dim("\n  [compacting context…]"), file=sys.stderr, flush=True)

    # ---- public helpers ----

    def user_prompt(self, text: str) -> None:
        print(f"\n{cyan('●')} {bold('you')}  {text}")

    def assistant_done(self) -> None:
        # A trailing newline after streamed assistant text
        print()
        self._wrote_assistant_header = False

    def error(self, msg: str) -> None:
        print(red(f"\nerror: {msg}"), file=sys.stderr)

    def info(self, msg: str) -> None:
        print(dim(msg))
