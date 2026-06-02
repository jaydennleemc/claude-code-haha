"""claude-code-py: a simple Python implementation of the Claude Code architecture.

Modules:
    core    - agent loop, context, types (no I/O)
    api     - Anthropic SDK wrapper
    tools   - tool registry, base class, built-ins
    mcp     - Model Context Protocol client
    skills  - skill loader (markdown frontmatter)
    ui      - terminal display
"""

__version__ = "0.1.0"
