"""Built-in tools: Read, Write, Edit, Bash.

These cover the minimum surface area for an agent to be useful in a real
codebase: inspect files, modify them, run shell commands. They are
intentionally simple — no fuzzy matching, no LSP integration, no globbing.
Just the raw primitives.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from .base import Tool, register_tool


# Tools that only read state — safe to run in parallel.
READ_ONLY_TOOLS: frozenset[str] = frozenset({"Read", "Glob", "Grep"})


def is_read_only(tool_name: str) -> bool:
    return tool_name in READ_ONLY_TOOLS


# ---------- Read ----------

@register_tool
class ReadTool(Tool):
    name = "Read"
    description = (
        "Read a file from the local filesystem. Returns the file contents "
        "with line numbers. For large files, prefer reading specific ranges."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file"},
            "offset": {"type": "integer", "description": "Line number to start at (0-based)"},
            "limit": {"type": "integer", "description": "Max number of lines to read"},
        },
        "required": ["file_path"],
    }

    async def execute(
        self,
        file_path: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {file_path}")
        if path.is_dir():
            raise IsADirectoryError(f"Is a directory: {file_path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        end = offset + limit if limit is not None else len(lines)
        snippet = lines[offset:end]
        # 1-based line numbers, like cat -n
        numbered = [f"{i + offset + 1:6d}\t{line}" for i, line in enumerate(snippet)]
        return "\n".join(numbered) if numbered else "<empty file>"


# ---------- Write ----------

@register_tool
class WriteTool(Tool):
    name = "Write"
    description = (
        "Write content to a file, replacing any existing content. Creates "
        "parent directories as needed. Use Edit for surgical changes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["file_path", "content"],
    }

    async def execute(self, file_path: str, content: str) -> str:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {file_path}"


# ---------- Edit ----------

@register_tool
class EditTool(Tool):
    name = "Edit"
    description = (
        "Surgically replace a string in a file. Both old_string and "
        "new_string must be exact matches. The old_string must be unique "
        "in the file — if it isn't, pass more context to disambiguate."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    async def execute(self, file_path: str, old_string: str, new_string: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {file_path}")

        original = path.read_text(encoding="utf-8")
        count = original.count(old_string)
        if count == 0:
            raise ValueError(
                "old_string not found in file. Make sure the match is exact "
                "(including whitespace and indentation)."
            )
        if count > 1:
            raise ValueError(
                f"old_string is not unique: appears {count} times. "
                "Provide more surrounding context to make it unique."
            )

        new_content = original.replace(old_string, new_string, 1)
        path.write_text(new_content, encoding="utf-8")
        return f"Edited {file_path}"


# ---------- Bash ----------

@register_tool
class BashTool(Tool):
    name = "Bash"
    description = (
        "Run a shell command and return its stdout/stderr. Use for git, "
        "running tests, installing dependencies, etc. Commands run in the "
        "current working directory. Long-running commands will be killed "
        "after a timeout."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds (default 30000)",
            },
        },
        "required": ["command"],
    }

    DEFAULT_TIMEOUT_MS = 30_000
    MAX_OUTPUT_BYTES = 30_000  # truncate noisy output

    async def execute(self, command: str, timeout: int | None = None) -> str:
        timeout_s = (timeout or self.DEFAULT_TIMEOUT_MS) / 1000

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"Command timed out after {timeout_s}s: {command}")

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")

        # Truncate if huge
        if len(out) > self.MAX_OUTPUT_BYTES:
            out = out[: self.MAX_OUTPUT_BYTES] + f"\n... [truncated, {len(out)} bytes total]"
        if len(err) > self.MAX_OUTPUT_BYTES:
            err = err[: self.MAX_OUTPUT_BYTES] + f"\n... [truncated, {len(err)} bytes total]"

        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        if proc.returncode != 0:
            parts.append(f"[exit code: {proc.returncode}]")

        return "\n".join(parts) if parts else "<no output>"
