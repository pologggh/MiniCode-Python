"""Tests for ToolRegistry security gate, command approval fix, and fail-closed behavior."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from minicode.auto_mode import PermissionMode
from minicode.permissions import PermissionManager
from minicode.security_audit import SecurityAuditLog
from minicode.security_policy import SecurityPolicyEngine
from minicode.tooling import ToolContext, ToolDefinition, ToolRegistry, ToolResult


def test_tool_registry_pre_gate_denial_blocks_execution(tmp_path):
    run_counter = 0

    def mock_run(input_data, context):
        nonlocal run_counter
        run_counter += 1
        return ToolResult(ok=True, output="Executed")

    dangerous_tool = ToolDefinition(
        name="run_command",
        description="Run command",
        input_schema={"type": "object"},
        validator=lambda x: x,
        run=mock_run,
    )

    policy = SecurityPolicyEngine()
    audit_file = tmp_path / "audit.jsonl"
    audit = SecurityAuditLog(audit_file)
    registry = ToolRegistry([dangerous_tool], security_policy=policy, security_audit=audit)

    context = ToolContext(cwd=str(tmp_path), permissions=None)
    # Catastrophic command
    res = registry.execute("run_command", {"command": "git reset --hard"}, context)

    assert res.ok is False
    assert "Security policy denied" in res.output
    # CRITICAL: Tool.run() was NEVER executed!
    assert run_counter == 0

    # Audit event must be recorded
    valid, count, _, _ = audit.verify_chain()
    assert valid is True
    assert count == 1


def test_tool_registry_fail_closed_on_missing_permission_manager(tmp_path):
    tool = ToolDefinition(
        name="batch_delete",
        description="Delete files",
        input_schema={"type": "object"},
        validator=lambda x: x,
        run=lambda inp, ctx: ToolResult(ok=True, output="Deleted"),
    )
    policy = SecurityPolicyEngine()
    registry = ToolRegistry([tool], security_policy=policy)

    # Context without permissions (permissions=None)
    context = ToolContext(cwd=str(tmp_path), permissions=None)
    res = registry.execute("batch_delete", {"path": "temp_file.txt"}, context)

    assert res.ok is False
    assert "permission manager missing" in res.output


def test_command_approval_prompt_in_default_mode(tmp_path):
    """Verify fix for P0 bug: non-dangerous dev commands trigger prompt in DEFAULT mode."""
    prompts_received = []

    def mock_prompt(req):
        prompts_received.append(req)
        return {"decision": "allow_once"}

    perms = PermissionManager(
        workspace_root=str(tmp_path),
        prompt=mock_prompt,
        auto_mode=PermissionMode.DEFAULT,
    )

    # 1. Pure read-only command (ls) does NOT prompt
    perms.ensure_command("ls", ["-la"], str(tmp_path))
    assert len(prompts_received) == 0

    # 2. Development command (pytest) MUST prompt in DEFAULT mode!
    perms.ensure_command("pytest", ["tests/"], str(tmp_path))
    assert len(prompts_received) == 1
    assert "pytest" in prompts_received[0]["scope"]

    # 3. python script.py MUST prompt in DEFAULT mode!
    perms.ensure_command("python", ["app.py"], str(tmp_path))
    assert len(prompts_received) == 2


def test_sensitive_file_read_redaction(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=postgres://user:supersecretpass@localhost/db\nAPI_KEY=sk-test-12345678\n", encoding="utf-8")

    def mock_read(input_data, context):
        return ToolResult(ok=True, output=env_file.read_text(encoding="utf-8"))

    read_tool = ToolDefinition(
        name="read_file",
        description="Read file",
        input_schema={"type": "object"},
        validator=lambda x: x,
        run=mock_read,
    )

    perms = PermissionManager(
        workspace_root=str(tmp_path),
        prompt=lambda req: {"decision": "allow_once"},
        auto_mode=PermissionMode.DEFAULT,
    )
    policy = SecurityPolicyEngine()
    registry = ToolRegistry([read_tool], security_policy=policy)
    context = ToolContext(cwd=str(tmp_path), permissions=perms)

    res = registry.execute("read_file", {"path": str(env_file)}, context)
    assert res.ok is True
    # Secret must be redacted in output!
    assert "sk-test-12345678" not in res.output
    assert "supersecretpass" not in res.output
    assert "[REDACTED]" in res.output
