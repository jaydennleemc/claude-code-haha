# claude-code-py

A simple, well-structured **Python** implementation of the [Claude Code architecture](../docs/architecture.html).

> Built as a learning reference. Mirrors the core subsystems from the leaked TS source (agent loop, tool orchestration, MCP, skills) but stripped to the essentials.

## API backends

Two interchangeable backends, both implementing the same `APIClient` Protocol:

| Backend | Class | Use when |
|---|---|---|
| Anthropic | `AnthropicClient` | You have an Anthropic API key (or a custom proxy) |
| OpenAI-compatible | `OpenAIClient` | You have LM Studio, OpenRouter, OpenAI, vLLM, or any other `/v1/chat/completions` endpoint |

Selection is driven by the `API_FORMAT` env var, with auto-detection
fallback (`OPENAI_BASE_URL` containing `:1234` — LM Studio's default port —
implies OpenAI).

```bash
# Anthropic (default)
# .env: ANTHROPIC_API_KEY=sk-ant-xxx
claude-code-py -p "explain this code"

# LM Studio
# .env:
#   OPENAI_API_KEY=lm-studio
#   OPENAI_BASE_URL=http://localhost:1234/v1
#   OPENAI_MODEL=deepseek-r1-0528-qwen3-8b
claude-code-py -p "explain this code"

# Force the format explicitly (overrides auto-detect)
# .env: API_FORMAT=openai
```

## Architecture

```
src/claude_code/
├── __main__.py        # Entry: python -m claude_code
├── cli.py             # argparse CLI (REPL + --prompt mode)
├── core/              # Pure business logic — no I/O
│   ├── types.py       #   Message, ToolCall, ToolResult dataclasses
│   ├── context.py     #   Conversation history + token counting + compaction
│   └── agent.py       #   The think-act-observe loop
├── api/               # External boundary
│   ├── client.py      #   AnthropicClient + OpenAIClient (both stream)
│   └── factory.py     #   select_format() / build_client() — picks the backend
├── tools/             # Pluggable tool system
│   ├── base.py        #   Tool ABC + Registry + @register_tool decorator
│   ├── permissions.py #   Allow/deny rules
│   └── builtin.py     #   Read, Write, Edit, Bash
├── mcp/               # Model Context Protocol client
│   ├── transport.py   #   JSON-RPC over stdio
│   └── client.py      #   Spawns servers, wraps their tools as Tool instances
├── skills/            # Skills loaded from .md frontmatter
│   └── loader.py
└── ui/                # Display layer (swap for web/IDE)
    └── console.py
```

### Why this layout?

Each subsystem is **independently testable** and **swappable**:

- Replace `api/client.py` → swap to OpenAI / local model
- Replace `ui/console.py` → ship a web/IDE frontend
- Add to `tools/builtin.py` → register a new capability
- Add to `mcp/client.py` → connect to a new MCP server
- Add `.md` files to `skills/` → teach the agent new tricks

The `core/` package has **zero I/O dependencies** — it only knows about the `Tool` ABC and the `APIClient` protocol. That's what makes the agent loop testable with a fake API.

### How the dual backend works

The agent loop calls `api.create_message(model=..., system=..., messages=..., tools=...)` — same signature for both backends. Each client handles its own:

- **Tool schema** conversion (Anthropic flat → OpenAI `{"type": "function", "function": {...}}`)
- **Tool result** conversion (Anthropic user-message-with-blocks → OpenAI `role: "tool"` messages)
- **Streaming** aggregation (Anthropic SSE `content_block_*` events vs OpenAI `delta.tool_calls[]` with `index` keying for parallel calls)
- **Stop reason** mapping (`stop` → `end_turn`, `tool_calls` → `tool_use`, `length` → `max_tokens`)

So `core/agent.py` has no idea which backend is in use. Swapping is purely a factory concern.

## Install

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env — see "API backends" above for Anthropic vs OpenAI/LM Studio setup
```

## Run

```bash
# Interactive REPL
claude-code-py

# Single prompt (headless)
claude-code-py -p "explain this codebase"

# Pipe input
echo "refactor utils/validation.py" | claude-code-py -p

# With specific model
claude-code-py --model claude-3-5-haiku-20241022

# Debug mode (verbose tool calls)
claude-code-py --debug
```

## How the loop works

```
1. user prompt → context.add_user_message()
2. loop:
   a. context.token_count()          # track usage
   b. api.create_message(streaming)  # LLM think
   c. if no tool_use → return text
   d. for each tool_use: registry.execute()  # parallel if read-only
   e. context.add_tool_results()     # observe
   f. if turn > MAX_AGENT_TURNS → stop
3. print final response
```

## Adding a tool

```python
from claude_code.tools.base import Tool, register_tool

@register_tool
class GrepTool(Tool):
    name = "Grep"
    description = "Search for a pattern in files"
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["pattern"],
    }

    async def execute(self, pattern: str, path: str = ".") -> str:
        # ... your impl
        return result
```

That's it — the registry picks it up on import and the LLM can call it.

## Adding an MCP server

Add to `.env`:

```bash
MCP_SERVERS=[{"name": "github", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]}]
```

On startup, the MCP client spawns the server, lists its tools, and registers them with the same `Tool` interface — so the LLM sees one unified toolset.

## Adding a skill

Create `.claude/skills/my-skill.md`:

```markdown
---
name: code-review
description: Use when reviewing code for quality, security, or performance issues.
---

When reviewing code, check for:
- Resource leaks
- Error handling
- Input validation
- Test coverage
```

The loader reads frontmatter + body, and the agent injects it into the system prompt.

## Tests

```bash
pytest
```

The test suite mocks the API client, so it runs without an API key.
