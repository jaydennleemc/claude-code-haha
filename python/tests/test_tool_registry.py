"""Tests for the tool registry and built-in tools."""

import pytest

from claude_code.tools.base import Tool, ToolRegistry, registry, register_tool
from claude_code.tools.builtin import ReadTool, WriteTool, EditTool, BashTool
from claude_code.tools.permissions import (
    PermissionChecker,
    PermissionDecision,
    PermissionRule,
)


# ---------- Tool registration ----------

class _EchoTool(Tool):
    name = "Echo"
    description = "Returns the input string"
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, text: str) -> str:
        return text


class TestToolRegistration:
    def test_decorator_registers(self):
        reg = ToolRegistry()
        register_tool(_EchoTool)

        # The decorator registers on the global registry. We just check
        # the class is well-formed and the decorator returned the class.
        assert _EchoTool.name == "Echo"
        # Register explicitly to a local registry too
        reg.register(_EchoTool())
        assert "Echo" in reg

    def test_register_returns_tool(self):
        reg = ToolRegistry()
        t = reg.register(_EchoTool())
        assert t is not None
        assert reg.get("Echo") is t

    def test_lookup_missing_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError) as exc:
            reg.get("DoesNotExist")
        assert "DoesNotExist" in str(exc.value)

    def test_spec_format(self):
        spec = _EchoTool().to_spec()
        assert spec.name == "Echo"
        assert "text" in spec.input_schema["properties"]
        assert spec.input_schema["required"] == ["text"]

    def test_run_wraps_exception_in_error_result(self):
        class _Bomb(Tool):
            name = "Bomb"
            description = "always fails"
            input_schema = {"type": "object", "properties": {}}

            async def execute(self):
                raise RuntimeError("boom")

        import asyncio
        result = asyncio.run(_Bomb().run("call-1", {}))
        assert result.is_error is True
        assert "boom" in result.content
        assert result.tool_call_id == "call-1"

    def test_run_returns_success_on_normal_return(self):
        import asyncio
        result = asyncio.run(_EchoTool().run("call-1", {"text": "hi"}))
        assert result.is_error is False
        assert result.content == "hi"


# ---------- PermissionChecker ----------

class TestPermissions:
    def test_default_allow_when_no_rules(self):
        checker = PermissionChecker()
        assert checker.check(_EchoTool(), {}) == PermissionDecision.ALLOW

    def test_default_deny_when_rules_set(self):
        checker = PermissionChecker(default=PermissionDecision.DENY)
        assert checker.check(_EchoTool(), {}) == PermissionDecision.DENY

    def test_rule_match_by_name(self):
        checker = PermissionChecker(
            rules=[PermissionRule("Echo")],
            default=PermissionDecision.DENY,
        )
        assert checker.check(_EchoTool(), {}) == PermissionDecision.ALLOW

    def test_rule_no_match_falls_through(self):
        checker = PermissionChecker(
            rules=[PermissionRule("Other")],
            default=PermissionDecision.DENY,
        )
        assert checker.check(_EchoTool(), {}) == PermissionDecision.DENY

    def test_glob_pattern(self):
        checker = PermissionChecker(
            rules=[PermissionRule("B*")],
            default=PermissionDecision.DENY,
        )
        assert checker.check(BashTool(), {}) == PermissionDecision.ALLOW

    def test_arg_pattern(self):
        checker = PermissionChecker(
            rules=[PermissionRule("Bash", arg_pattern="git *")],
            default=PermissionDecision.DENY,
        )
        # git status → allowed
        assert checker.check(BashTool(), {"command": "git status"}) == PermissionDecision.ALLOW
        # rm -rf → denied (default)
        assert checker.check(BashTool(), {"command": "rm -rf /"}) == PermissionDecision.DENY

    def test_from_env_string(self):
        checker = PermissionChecker.from_env_string("Read,Write,Bash:git *")
        assert checker.check(ReadTool(), {}) == PermissionDecision.ALLOW
        assert checker.check(WriteTool(), {}) == PermissionDecision.ALLOW
        assert checker.check(EditTool(), {}) == PermissionDecision.DENY  # not in list
        assert checker.check(BashTool(), {"command": "git log"}) == PermissionDecision.ALLOW

    def test_from_env_empty(self):
        checker = PermissionChecker.from_env_string("")
        # Empty config: default-allow
        assert checker.check(BashTool(), {}) == PermissionDecision.ALLOW


# ---------- Built-in tools (smoke tests) ----------

class TestBuiltinTools:
    @pytest.mark.asyncio
    async def test_read_returns_numbered_lines(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("alpha\nbeta\ngamma\n")
        result = await ReadTool().execute(file_path=str(f))
        assert "alpha" in result
        assert "1" in result  # line number
        assert "3" in result  # last line number

    @pytest.mark.asyncio
    async def test_read_offset_and_limit(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("\n".join(f"line{i}" for i in range(10)))
        result = await ReadTool().execute(file_path=str(f), offset=3, limit=2)
        assert "line3" in result
        assert "line4" in result
        assert "line0" not in result
        assert "line5" not in result

    @pytest.mark.asyncio
    async def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await ReadTool().execute(file_path=str(tmp_path / "missing.txt"))

    @pytest.mark.asyncio
    async def test_write_creates_file_and_parent(self, tmp_path):
        target = tmp_path / "sub" / "x.txt"
        result = await WriteTool().execute(file_path=str(target), content="hello")
        assert target.exists()
        assert target.read_text() == "hello"
        assert "hello" in result or "Wrote" in result

    @pytest.mark.asyncio
    async def test_edit_replaces_unique_string(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hello world")
        result = await EditTool().execute(
            file_path=str(f),
            old_string="world",
            new_string="python",
        )
        assert f.read_text() == "hello python"

    @pytest.mark.asyncio
    async def test_edit_rejects_non_unique(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("foo foo foo")
        with pytest.raises(ValueError) as exc:
            await EditTool().execute(
                file_path=str(f),
                old_string="foo",
                new_string="bar",
            )
        assert "not unique" in str(exc.value)

    @pytest.mark.asyncio
    async def test_bash_runs_command(self, tmp_path):
        result = await BashTool().execute(command="echo hello", timeout=5000)
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_bash_propagates_exit_code(self):
        result = await BashTool().execute(command="exit 7", timeout=5000)
        assert "7" in result  # exit code surfaces
