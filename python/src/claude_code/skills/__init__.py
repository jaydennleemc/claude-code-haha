"""Skill loader.

A "skill" is a markdown file with YAML frontmatter that teaches the agent
how to handle a specific kind of task. We load every ``*.md`` in the
configured skills directory and concatenate them into a section of the
system prompt.

Frontmatter schema::

    ---
    name: code-review
    description: Use when reviewing code for quality issues.
    ---

    Body of the skill — instructions, checklists, examples...
"""

from .loader import Skill, load_skills, format_skills_for_prompt

__all__ = ["Skill", "load_skills", "format_skills_for_prompt"]
