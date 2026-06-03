"""Command-line interface — the entry point that wires everything together.

Responsibilities:
    1. Parse CLI args (argparse, headless vs REPL, model, debug)
    2. Load config from env (.env)
    3. Build the API client
    4. Load skills from the configured directory
    5. Spawn MCP servers and register their tools
    6. Build the agent loop with the assembled components
    7. Run a single prompt, or loop on stdin in REPL mode
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from . import __version__
from .api.client import APIClient
from .api.factory import build_client, build_model, select_format
from .core.agent import Agent, AgentConfig
from .core.context import Context, ContextConfig
from .mcp.client import MCPServerConfig, connect_all, parse_server_configs
from .skills.loader import format_skills_for_prompt, load_skills
from .tools.base import ToolRegistry, registry as default_registry
from .tools.permissions import PermissionChecker
from .ui.console import Console


# ---------- the system prompt ----------

BASE_SYSTEM_PROMPT = """\
You are Claude Code, an AI coding assistant running in a developer's terminal.
You can read and edit files, run shell commands, and call any tools the user
has registered. You operate in an iterative loop: think, act, observe.

Guidelines:
- Be concise. The terminal is precious real estate.
- Use tools in parallel when they're independent reads.
- When you finish a task, briefly state what you did and stop.
- If a tool fails, read the error and adjust — don't blindly retry.
"""


def build_system_prompt(skills_dir: str | Path) -> str:
    """Base prompt + injected skills section."""
    skills = load_skills(skills_dir)
    skills_section = format_skills_for_prompt(skills)
    if skills_section:
        return f"{BASE_SYSTEM_PROMPT}\n\n{skills_section}"
    return BASE_SYSTEM_PROMPT


# ---------- the wiring ----------

async def run(args: argparse.Namespace) -> int:
    """Main async entry. Returns process exit code."""
    # 1. API client
    try:
        api: APIClient = build_client()
    except ValueError as e:
        backend = select_format()
        env_var = "OPENAI_API_KEY" if backend == "openai" else "ANTHROPIC_API_KEY"
        print(f"error: {e}", file=sys.stderr)
        print(f"hint: copy .env.example to .env and set {env_var} (or {env_var}_TOKEN)", file=sys.stderr)
        return 2

    model = args.model or build_model()
    console = Console(verbose=args.debug)

    # 2. Tool registry: start with builtins, then add MCP tools
    tool_registry = ToolRegistry()
    for tool in default_registry:
        tool_registry.register(tool)

    mcp_servers = parse_server_configs(os.environ.get("MCP_SERVERS"))
    if mcp_servers:
        console.info(f"connecting to {len(mcp_servers)} MCP server(s)…")
        results = await connect_all(mcp_servers)
        for cfg, result in results:
            if isinstance(result, Exception):
                console.error(f"MCP '{cfg.name}' failed: {result}")
            else:
                console.info(f"  ✓ {cfg.name}: {len(result)} tool(s)")
                for t in result:
                    tool_registry.register(t)

    console.info(f"{len(tool_registry.names())} tool(s) available: {', '.join(tool_registry.names())}")

    # 3. Context
    ctx_config = ContextConfig(
        model=model,
        window=int(os.environ.get("CONTEXT_WINDOW", "200000")),
        compact_threshold=float(os.environ.get("COMPACT_THRESHOLD", "0.8")),
    )
    context = Context(
        config=ctx_config,
        api=api,
        system=build_system_prompt(os.environ.get("SKILLS_DIR", ".claude/skills")),
    )
    # Pre-populate the tool schema so token counting is accurate from turn 0
    context.set_tools([
        {"name": s.name, "description": s.description, "input_schema": s.input_schema}
        for s in tool_registry.list_specs()
    ])

    # 4. Permissions
    permissions = PermissionChecker.from_env_string(os.environ.get("ALLOWED_TOOLS", ""))

    # 5. Agent
    agent_config = AgentConfig(
        model=model,
        max_turns=int(os.environ.get("MAX_AGENT_TURNS", "50")),
        max_tool_parallel=int(os.environ.get("MAX_TOOL_PARALLEL", "5")),
    )
    agent = Agent(
        api=api,
        context=context,
        config=agent_config,
        tool_registry=tool_registry,
        permission_checker=permissions,
        on_progress=console.on_progress,
    )

    # 6. Run
    try:
        if args.prompt is not None:
            # Headless: single prompt, optional stdin input
            prompt = args.prompt
            if prompt == "-":
                prompt = sys.stdin.read()
            console.user_prompt(prompt)
            text = await agent.run(prompt)
            if text:
                console.assistant_done()
            return 0
        else:
            return await _repl(agent, console)
    except KeyboardInterrupt:
        console.info("\ninterrupted")
        return 130
    except Exception as e:
        console.error(str(e))
        if args.debug:
            raise
        return 1


async def _repl(agent: Agent, console: Console) -> int:
    """Interactive read-eval-print loop."""
    console.info(f"claude-code-py {__version__}  (type 'exit' or Ctrl-D to quit)")
    console.info(f"tools: {', '.join(agent.tools.names())}")
    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, lambda: input("\n> "))
        except (EOFError, KeyboardInterrupt):
            console.info("\nbye")
            return 0
        text = line.strip()
        if not text:
            continue
        if text in ("exit", "quit", ":q"):
            return 0
        if text == "/tokens":
            n = agent.context.token_count()
            ratio = agent.context.usage_ratio()
            console.info(f"tokens: {n:,} / {agent.context.config.window:,} ({ratio:.1%})")
            continue
        if text == "/tools":
            for name in agent.tools.names():
                console.info(f"  - {name}")
            continue
        console.user_prompt(text)
        try:
            response = await agent.run(text)
            if response:
                console.assistant_done()
        except Exception as e:
            console.error(str(e))


# ---------- argparse ----------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="claude-code-py",
        description="Simple Python implementation of Claude Code",
    )
    parser.add_argument(
        "-p", "--prompt",
        help="Run a single prompt and exit (use '-' to read from stdin)",
    )
    parser.add_argument(
        "--model",
        help="Override the model name from env (ANTHROPIC_MODEL or OPENAI_MODEL)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging and full tracebacks",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"claude-code-py {__version__}",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Sync entry point for the ``claude-code-py`` console script."""
    args = parse_args(argv)
    # Load .env from CWD if present, but don't override existing env
    load_dotenv(override=False)
    return asyncio.run(run(args))
