"""Permission system for tool execution.

Each tool call goes through ``PermissionChecker.check()`` before execution.
Three outcomes:

    ALLOW     — run it
    DENY      — refuse, return error to model
    ASK       — prompt the user (REPL only); in headless mode, this becomes DENY

The default config trusts everything (``ALLOW``). Configure via the
``ALLOWED_TOOLS`` env var (comma-separated) for a deny-by-default posture.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from enum import Enum

from .base import Tool


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionRule:
    """One allow/deny pattern.

    Patterns support shell-style globs (``*``, ``?``) on the tool name. The
    ``arg_pattern`` field is a free-form string matched against a
    JSON-serialized form of the tool input — for example, the Bash tool
    could use ``arg_pattern="git *"`` to allow only git subcommands.
    """

    tool_name: str
    arg_pattern: str | None = None


class PermissionChecker:
    """Decide whether a tool call is permitted.

    Rules are evaluated in order; the first match wins. If no rule matches,
    the default decision is used (ASK in interactive mode, ALLOW otherwise
    for the "trust everything" default).
    """

    def __init__(
        self,
        rules: list[PermissionRule] | None = None,
        default: PermissionDecision = PermissionDecision.ALLOW,
    ) -> None:
        self.rules = rules or []
        self.default = default

    def check(self, tool: Tool, input: dict) -> PermissionDecision:
        tool_name = tool.name
        for rule in self.rules:
            if not fnmatch.fnmatch(tool_name, rule.tool_name):
                continue
            if rule.arg_pattern is None:
                return PermissionDecision.ALLOW
            # Naive match: compare against the first string field in input.
            # Real Claude Code uses structured permission rules; this is the
            # simple-but-good-enough version.
            serialized = " ".join(str(v) for v in input.values())
            if fnmatch.fnmatch(serialized, rule.arg_pattern):
                return PermissionDecision.ALLOW
        return self.default

    @classmethod
    def from_env_string(cls, raw: str) -> "PermissionChecker":
        """Parse ``"Bash:git *,Read,Write"`` → list of rules.

        ``"*"`` matches everything; missing patterns mean "any args".
        """
        if not raw.strip():
            return cls(rules=[], default=PermissionDecision.ALLOW)
        rules: list[PermissionRule] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                name, pat = token.split(":", 1)
            else:
                name, pat = token, None
            rules.append(PermissionRule(tool_name=name, arg_pattern=pat))
        return cls(rules=rules, default=PermissionDecision.DENY)
