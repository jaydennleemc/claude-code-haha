"""Tests for the skill loader."""

from pathlib import Path

import pytest

from claude_code.skills.loader import (
    Skill,
    format_skills_for_prompt,
    load_skills,
)


# ---------- load_skills ----------

class TestLoadSkills:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert load_skills(tmp_path) == []

    def test_missing_dir_returns_empty_list(self, tmp_path):
        # Should not raise — a missing skills dir is a valid empty config
        assert load_skills(tmp_path / "does-not-exist") == []

    def test_loads_markdown_file(self, tmp_path):
        (tmp_path / "foo.md").write_text(
            "---\n"
            "name: foo\n"
            "description: does foo things\n"
            "---\n\n"
            "Do foo like this."
        )
        skills = load_skills(tmp_path)
        assert len(skills) == 1
        s = skills[0]
        assert s.name == "foo"
        assert s.description == "does foo things"
        assert "Do foo like this" in s.body

    def test_filename_used_as_name_when_no_frontmatter(self, tmp_path):
        (tmp_path / "nofront.md").write_text("Just some text")
        skills = load_skills(tmp_path)
        assert skills[0].name == "nofront"
        assert skills[0].description == ""
        assert skills[0].body == "Just some text"

    def test_ignores_non_markdown_files(self, tmp_path):
        (tmp_path / "x.md").write_text("---\nname: x\n---\nbody")
        (tmp_path / "x.txt").write_text("not a skill")
        (tmp_path / "x.json").write_text("{}")
        assert len(load_skills(tmp_path)) == 1

    def test_sorted_by_filename(self, tmp_path):
        for name in ("c", "a", "b"):
            (tmp_path / f"{name}.md").write_text(f"---\nname: {name}\n---\nbody")
        skills = load_skills(tmp_path)
        assert [s.name for s in skills] == ["a", "b", "c"]

    def test_invalid_yaml_is_skipped(self, tmp_path, capsys):
        (tmp_path / "bad.md").write_text(
            "---\n"
            "name: bad\n"
            "description: [unclosed\n"
            "---\n\n"
            "body"
        )
        skills = load_skills(tmp_path)
        assert skills == []
        assert "invalid YAML" in capsys.readouterr().err

    def test_frontmatter_not_a_mapping_is_skipped(self, tmp_path, capsys):
        (tmp_path / "list.md").write_text("---\n- a\n- b\n---\nbody")
        skills = load_skills(tmp_path)
        assert skills == []
        assert "must be a mapping" in capsys.readouterr().err

    def test_optional_fields_default(self, tmp_path):
        (tmp_path / "x.md").write_text(
            "---\n"
            "name: x\n"
            "description: d\n"
            "---\n"
            "body"
        )
        s = load_skills(tmp_path)[0]
        assert s.tags == []
        assert s.when_to_use == ""


# ---------- format_skills_for_prompt ----------

class TestFormatSkills:
    def test_empty_returns_empty_string(self):
        assert format_skills_for_prompt([]) == ""

    def test_single_skill_renders_heading_and_body(self):
        s = Skill(name="foo", description="does X", body="Step 1\nStep 2", path=Path("x.md"))
        out = format_skills_for_prompt([s])
        assert "### Skill: foo" in out
        assert "*does X*" in out
        assert "Step 1" in out
        assert "Step 2" in out

    def test_multiple_skills_all_present(self):
        s1 = Skill(name="a", description="d-a", body="body-a", path=Path("a.md"))
        s2 = Skill(name="b", description="d-b", body="body-b", path=Path("b.md"))
        out = format_skills_for_prompt([s1, s2])
        assert "### Skill: a" in out
        assert "### Skill: b" in out
        assert "body-a" in out
        assert "body-b" in out

    def test_no_description_omits_italic(self):
        s = Skill(name="x", description="", body="b", path=Path("x.md"))
        out = format_skills_for_prompt([s])
        assert "*" not in out  # no italic marker if no description
        assert "### Skill: x" in out
