"""Load skill files from a directory and format them for the system prompt.

A skill is a markdown file with YAML frontmatter. The frontmatter describes
when the skill applies; the body is the instruction. We assemble all skills
into a single block the model sees as part of its system prompt.

Example file (``my-skill.md``)::

    ---
    name: code-review
    description: Use when reviewing code for quality, security, or performance.
    ---

    ## Checklist

    - Resource leaks
    - Error handling
    - Input validation
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


@dataclass
class Skill:
    """One loaded skill file."""

    name: str
    description: str
    body: str
    path: Path
    # Optional tag fields the loader understands
    when_to_use: str = ""        # alias for description
    tags: list[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        """Render this skill for inclusion in the system prompt."""
        heading = f"### Skill: {self.name}"
        if self.description:
            heading += f"\n*{self.description}*"
        return f"{heading}\n\n{self.body.strip()}"


# Frontmatter delimiter
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_one(path: Path) -> Skill | None:
    """Parse a single skill file. Returns None on malformed files."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"[skills] skipping {path}: {e}", file=sys.stderr)
        return None

    match = _FRONTMATTER_RE.match(text)
    if not match:
        # No frontmatter: use the filename as the name, no description
        return Skill(
            name=path.stem,
            description="",
            body=text.strip(),
            path=path,
        )

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as e:
        print(f"[skills] {path}: invalid YAML frontmatter, skipping ({e})", file=sys.stderr)
        return None

    if not isinstance(meta, dict):
        print(f"[skills] {path}: frontmatter must be a mapping, skipping", file=sys.stderr)
        return None

    return Skill(
        name=str(meta.get("name", path.stem)),
        description=str(meta.get("description", "")),
        body=match.group(2).strip(),
        path=path,
        when_to_use=str(meta.get("when_to_use", "")),
        tags=list(meta.get("tags", []) or []),
    )


def load_skills(skills_dir: str | Path) -> list[Skill]:
    """Load every ``*.md`` in ``skills_dir``, sorted by filename.

    Missing directory is not an error — it just returns ``[]``. That's the
    right default for fresh checkouts where the user hasn't created skills yet.
    """
    path = Path(skills_dir)
    if not path.exists() or not path.is_dir():
        return []

    skills: list[Skill] = []
    for md in sorted(path.glob("*.md")):
        skill = _parse_one(md)
        if skill is not None:
            skills.append(skill)
    return skills


def format_skills_for_prompt(skills: Iterable[Skill]) -> str:
    """Render all skills as a single system-prompt section.

    Returns an empty string if there are no skills — that way we can
    unconditionally concatenate the result with the base system prompt.
    """
    skills = list(skills)
    if not skills:
        return ""

    parts = ["## Available Skills", ""]
    parts.append(
        "The following skills are loaded for this session. Use them when the "
        "task matches the skill's description."
    )
    parts.append("")
    for skill in skills:
        parts.append(skill.to_prompt_section())
        parts.append("")
    return "\n".join(parts).rstrip()
