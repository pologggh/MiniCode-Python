"""Tests for ToolRegistry security gate, command approval fix, and fail-closed behavior."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from minicode.auto_mode import PermissionMode
from minicode.permissions import PermissionManager
from minicode.security_audit import SecurityAuditLog
from minicode.security_policy import (
    SecurityDecision,
    SecurityPolicyEngine,
    SecurityRequest,
    SecurityRisk,
)
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


def test_child_agent_mcp_access_denied(tmp_path):
    mcp_tool = ToolDefinition(
        name="mcp__github__create_repo",
        description="Create github repo",
        input_schema={"type": "object"},
        validator=lambda x: x,
        run=lambda inp, ctx: ToolResult(ok=True, output="Created"),
    )
    policy = SecurityPolicyEngine()
    registry = ToolRegistry([mcp_tool], security_policy=policy)

    child_context = ToolContext(
        cwd=str(tmp_path),
        _runtime={"_security_actor": "CHILD", "_security_role": "coder", "_security_depth": 1},
    )
    res = registry.execute("mcp__github__create_repo", {}, child_context)

    assert res.ok is False
    assert "Security policy denied" in res.output
    assert "child_mcp_denied" in res.metadata.get("rule_ids", [])


def test_child_researcher_write_denied(tmp_path):
    write_tool = ToolDefinition(
        name="write_file",
        description="Write file",
        input_schema={"type": "object"},
        validator=lambda x: x,
        run=lambda inp, ctx: ToolResult(ok=True, output="Written"),
    )
    policy = SecurityPolicyEngine()
    registry = ToolRegistry([write_tool], security_policy=policy)

    child_context = ToolContext(
        cwd=str(tmp_path),
        _runtime={"_security_actor": "CHILD", "_security_role": "researcher", "_security_depth": 1},
    )
    res = registry.execute("write_file", {"path": "test.txt", "content": "hello"}, child_context)

    assert res.ok is False
    assert "Security policy denied" in res.output
    assert "child_role_write_denied" in res.metadata.get("rule_ids", [])


def test_fail_closed_across_all_approval_routes_when_permissions_none(tmp_path):
    """Verify that when context.permissions is None, ALL tools requiring ASK fail closed before tool.run."""
    execution_counts = {"write_file": 0, "edit_file": 0, "run_command": 0, "batch_delete": 0, "mcp": 0}

    def make_runner(key: str):
        def _runner(inp, ctx):
            execution_counts[key] += 1
            return ToolResult(ok=True, output=f"{key} succeeded")
        return _runner

    tools = [
        ToolDefinition("write_file", "Write file", {"type": "object"}, lambda x: x, make_runner("write_file")),
        ToolDefinition("edit_file", "Edit file", {"type": "object"}, lambda x: x, make_runner("edit_file")),
        ToolDefinition("run_command", "Run cmd", {"type": "object"}, lambda x: x, make_runner("run_command")),
        ToolDefinition("batch_delete", "Batch del", {"type": "object"}, lambda x: x, make_runner("batch_delete")),
        ToolDefinition("mcp__server__query", "MCP query", {"type": "object"}, lambda x: x, make_runner("mcp")),
    ]

    policy = SecurityPolicyEngine()
    registry = ToolRegistry(tools, security_policy=policy)
    context = ToolContext(cwd=str(tmp_path), permissions=None)

    # 1. write_file in DEFAULT mode (NATIVE_EDIT route)
    res_w = registry.execute("write_file", {"path": "test.txt", "content": "hello"}, context)
    assert res_w.ok is False
    assert "permission manager missing" in res_w.output
    assert execution_counts["write_file"] == 0

    # 2. edit_file in DEFAULT mode (NATIVE_EDIT route)
    res_e = registry.execute("edit_file", {"path": "test.txt", "content": "hello"}, context)
    assert res_e.ok is False
    assert "permission manager missing" in res_e.output
    assert execution_counts["edit_file"] == 0

    # 3. run_command with non-readonly command in DEFAULT mode (NATIVE_COMMAND route)
    res_c = registry.execute("run_command", {"command": "pytest", "args": ["tests/"]}, context)
    assert res_c.ok is False
    assert "permission manager missing" in res_c.output
    assert execution_counts["run_command"] == 0

    # 4. batch_delete (GENERIC_TOOL route)
    res_d = registry.execute("batch_delete", {"path": "some_file.txt"}, context)
    assert res_d.ok is False
    assert "permission manager missing" in res_d.output
    assert execution_counts["batch_delete"] == 0

    # 5. MCP tool (GENERIC_TOOL route)
    res_m = registry.execute("mcp__server__query", {"query": "SELECT 1"}, context)
    assert res_m.ok is False
    assert "permission manager missing" in res_m.output
    assert execution_counts["mcp"] == 0


def test_native_permission_denial_audited(tmp_path):
    """Verify that when a tool's internal permission check raises denial, it is cleanly caught and audited."""
    def run_denied(inp, ctx):
        raise RuntimeError("Command denied by user permission: pytest tests/")

    cmd_tool = ToolDefinition(
        name="run_command",
        description="Run command",
        input_schema={"type": "object"},
        validator=lambda x: x,
        run=run_denied,
    )

    policy = SecurityPolicyEngine()
    audit_file = tmp_path / "audit.jsonl"
    audit = SecurityAuditLog(audit_file)
    registry = ToolRegistry([cmd_tool], security_policy=policy, security_audit=audit)

    perms = PermissionManager(
        workspace_root=str(tmp_path),
        prompt=lambda req: {"decision": "deny_once"},
        auto_mode=PermissionMode.DEFAULT,
    )
    context = ToolContext(cwd=str(tmp_path), permissions=perms)

    res = registry.execute("run_command", {"command": "pytest", "args": ["tests/"]}, context)
    assert res.ok is False
    assert "denied" in res.output.lower()

    valid, count, _, _ = audit.verify_chain()
    assert valid is True
    assert count == 1
    # Check authorization_outcome recorded in event
    import json
    with open(audit_file, "r", encoding="utf-8") as f:
        event = json.loads(f.readline())
    assert event["authorization_outcome"] == "DENIED"
    assert event["result_ok"] is False


def test_audit_failure_graceful_degradation(tmp_path, monkeypatch):
    """Verify that if the audit logger raises an exception after tool execution, the tool result is preserved."""
    def run_ok(inp, ctx):
        return ToolResult(ok=True, output="Tool work completed successfully")

    ok_tool = ToolDefinition(
        name="write_file",
        description="Write file",
        input_schema={"type": "object"},
        validator=lambda x: x,
        run=run_ok,
    )

    policy = SecurityPolicyEngine()
    audit_file = tmp_path / "audit.jsonl"
    audit = SecurityAuditLog(audit_file)

    # Force record_event to raise an exception
    def failing_record_event(*args, **kwargs):
        raise OSError("Disk full or permission denied writing audit log")

    monkeypatch.setattr(audit, "record_event", failing_record_event)

    perms = PermissionManager(
        workspace_root=str(tmp_path),
        prompt=lambda req: {"decision": "allow_once"},
        auto_mode=PermissionMode.AUTO,
    )
    registry = ToolRegistry([ok_tool], security_policy=policy, security_audit=audit)
    context = ToolContext(cwd=str(tmp_path), permissions=perms)

    res = registry.execute("write_file", {"path": "hello.txt", "content": "world"}, context)
    # The tool execution must still succeed despite audit failure!
    assert res.ok is True
    assert res.output == "Tool work completed successfully"


def test_bypass_sensitive_write_protection(tmp_path: Path):
    """Verify that in BYPASS mode, sensitive files (.env, .pem, .key, credentials.json) require explicit generic approval."""
    write_executed = False

    def run_write(inp, ctx):
        nonlocal write_executed
        write_executed = True
        p = Path(ctx.cwd) / inp["path"]
        p.write_text(inp["content"], encoding="utf-8")
        return ToolResult(ok=True, output=f"Wrote {inp['path']}")

    tool = ToolDefinition("write_file", "Write", {"type": "object"}, lambda x: x, run_write)
    policy = SecurityPolicyEngine()
    registry = ToolRegistry([tool], security_policy=policy)

    # 1. Deny on .env -> file not written, execution count 0
    prompt_calls = []
    perms_deny = PermissionManager(
        workspace_root=str(tmp_path),
        prompt=lambda req: (prompt_calls.append(req), {"decision": "deny_once"})[1],
        auto_mode=PermissionMode.BYPASS,
    )
    context_deny = ToolContext(cwd=str(tmp_path), permissions=perms_deny)
    res_deny = registry.execute("write_file", {"path": ".env", "content": "SECRET=123"}, context_deny)

    assert res_deny.ok is False
    assert write_executed is False
    assert not (tmp_path / ".env").exists()
    assert len(prompt_calls) == 1

    # 2. Allow on .env -> file written, executes
    prompt_calls.clear()
    write_executed = False
    perms_allow = PermissionManager(
        workspace_root=str(tmp_path),
        prompt=lambda req: (prompt_calls.append(req), {"decision": "allow_once"})[1],
        auto_mode=PermissionMode.BYPASS,
    )
    context_allow = ToolContext(cwd=str(tmp_path), permissions=perms_allow)
    res_allow = registry.execute("write_file", {"path": ".env", "content": "SECRET=123"}, context_allow)

    assert res_allow.ok is True
    assert write_executed is True
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "SECRET=123"
    assert len(prompt_calls) == 1

    # 3. Same protection for .pem, .key, credentials.json
    for sens_name in ["server.pem", "id_rsa.key", "credentials.json"]:
        prompt_calls.clear()
        write_executed = False
        res = registry.execute("write_file", {"path": sens_name, "content": "keydata"}, context_deny)
        assert res.ok is False
        assert write_executed is False

    # 4. permissions=None -> fail closed
    context_none = ToolContext(cwd=str(tmp_path), permissions=None)
    res_none = registry.execute("write_file", {"path": ".env", "content": "SECRET=123"}, context_none)
    assert res_none.ok is False
    assert "permission manager missing" in res_none.output

    # 5. .git internal write -> hard DENY even under BYPASS
    res_git = registry.execute("write_file", {"path": ".git/config", "content": "evil"}, context_allow)
    assert res_git.ok is False
    assert "Security policy denied" in res_git.output


def test_batch_copy_workspace_root_hard_deny(tmp_path: Path):
    """Verify that batch_copy to workspace root is hard-denied, and recursive self-copy is denied."""
    copy_tool = ToolDefinition("batch_copy", "Copy", {"type": "object"}, lambda x: x, lambda i, c: ToolResult(ok=True, output="Copied"))
    policy = SecurityPolicyEngine()
    registry = ToolRegistry([copy_tool], security_policy=policy)
    context = ToolContext(cwd=str(tmp_path))

    # All variations of workspace root destination must be DENIED
    for bad_dest in [".", "./", "sub/..", str(tmp_path), "a/../"]:
        req = SecurityRequest(
            tool_name="batch_copy",
            input_data={"source": "src", "destination": bad_dest},
            cwd=str(tmp_path),
        )
        assessment = policy.evaluate(req)
        assert assessment.decision == SecurityDecision.DENY
        assert assessment.risk == SecurityRisk.CRITICAL
        assert assessment.hard_deny is True
        assert "workspace_root_overwrite_denied" in assessment.rule_ids

    # Recursive self-copy: source is directory and destination is inside source
    req_recur = SecurityRequest(
        tool_name="batch_copy",
        input_data={"source": ".", "destination": "backup/nested"},
        cwd=str(tmp_path),
    )
    assessment_recur = policy.evaluate(req_recur)
    assert assessment_recur.decision == SecurityDecision.DENY
    assert assessment_recur.risk == SecurityRisk.CRITICAL
    assert assessment_recur.hard_deny is True
    assert "recursive_copy_denied" in assessment_recur.rule_ids


def test_generic_tool_scope_collision_protection(tmp_path: Path):
    """Verify that two generic tool requests sharing first 200 chars do not collide or bypass prompt."""
    prompt_count = 0

    def prompt(req):
        nonlocal prompt_count
        prompt_count += 1
        return {"decision": "allow_turn"}

    mcp_tool = ToolDefinition("mcp__server__action", "MCP", {"type": "object"}, lambda x: x, lambda i, c: ToolResult(ok=True, output="Done"))
    policy = SecurityPolicyEngine()
    registry = ToolRegistry([mcp_tool], security_policy=policy)

    perms = PermissionManager(workspace_root=str(tmp_path), prompt=prompt)
    perms.begin_turn()
    context = ToolContext(cwd=str(tmp_path), permissions=perms)

    # Payload A and B share identical 200-char prefix, but differ at char 201+
    payload_a = {"prefix": "X" * 200, "target": "database_A"}
    payload_b = {"prefix": "X" * 200, "target": "database_B"}

    # Request A executes and gets allow_turn
    res_a = registry.execute("mcp__server__action", payload_a, context)
    assert res_a.ok is True
    assert prompt_count == 1

    # Request B MUST prompt again despite matching the first 200 chars
    res_b = registry.execute("mcp__server__action", payload_b, context)
    assert res_b.ok is True
    assert prompt_count == 2, "Scope collision allowed unauthorized payload to bypass prompt"


def test_mcp_pre_execution_gate(tmp_path: Path):
    """Verify that MCP tool authorization strictly precedes invocation of the tool runner."""
    call_count = 0

    def fake_mcp_run(inp, ctx):
        nonlocal call_count
        call_count += 1
        return ToolResult(ok=True, output="MCP mutated data")

    mcp_tool = ToolDefinition("mcp__fake__mutate", "Fake MCP mutate", {"type": "object"}, lambda x: x, fake_mcp_run)
    policy = SecurityPolicyEngine()
    registry = ToolRegistry([mcp_tool], security_policy=policy)

    # 1. Deny decision
    perms_deny = PermissionManager(
        workspace_root=str(tmp_path),
        prompt=lambda req: {"decision": "deny_once"},
    )
    context_deny = ToolContext(cwd=str(tmp_path), permissions=perms_deny)
    res_deny = registry.execute("mcp__fake__mutate", {"table": "users", "action": "drop"}, context_deny)

    assert res_deny.ok is False
    assert call_count == 0, "MCP runner was invoked despite denial"

    # 2. Allow decision
    perms_allow = PermissionManager(
        workspace_root=str(tmp_path),
        prompt=lambda req: {"decision": "allow_once"},
    )
    context_allow = ToolContext(cwd=str(tmp_path), permissions=perms_allow)
    res_allow = registry.execute("mcp__fake__mutate", {"table": "users", "action": "insert"}, context_allow)

    assert res_allow.ok is True
    assert call_count == 1, "MCP runner was not invoked after approval"


def test_native_permission_denial_audit_without_prompt(tmp_path: Path):
    """Verify that when PermissionManager has prompt=None, native tools audit DENIED instead of TOOL_EXCEPTION."""
    audit_file = tmp_path / "audit_native.jsonl"
    audit = SecurityAuditLog(audit_file)

    def run_write(inp, ctx):
        ctx.permissions.ensure_edit(str(Path(ctx.cwd) / inp["path"]), "diff")
        return ToolResult(ok=True, output="Written")

    def run_cmd(inp, ctx):
        ctx.permissions.ensure_command(inp["command"], inp.get("args", []), ctx.cwd)
        return ToolResult(ok=True, output="Executed")

    write_tool = ToolDefinition("write_file", "Write", {"type": "object"}, lambda x: x, run_write)
    cmd_tool = ToolDefinition("run_command", "Run", {"type": "object"}, lambda x: x, run_cmd)

    policy = SecurityPolicyEngine()
    registry = ToolRegistry([write_tool, cmd_tool], security_policy=policy, security_audit=audit)

    # PermissionManager with prompt=None
    perms = PermissionManager(workspace_root=str(tmp_path), prompt=None, auto_mode=PermissionMode.DEFAULT)
    context = ToolContext(cwd=str(tmp_path), permissions=perms)

    # 1. write_file in DEFAULT mode raises 'Edit requires approval... Start minicode in TTY mode...'
    res_w = registry.execute("write_file", {"path": "main.py", "content": "code"}, context)
    assert res_w.ok is False
    assert "Tool execution denied" in res_w.output

    # 2. run_command in DEFAULT mode raises 'Command requires approval... Start minicode in TTY mode...'
    res_c = registry.execute("run_command", {"command": "pytest", "args": ["tests/"]}, context)
    assert res_c.ok is False
    assert "Tool execution denied" in res_c.output

    valid, count, _, _ = audit.verify_chain()
    assert valid is True
    assert count == 2

    import json
    with open(audit_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    assert events[0]["tool_name"] == "write_file"
    assert events[0]["authorization_outcome"] == "DENIED"
    assert events[0]["result_ok"] is False

    assert events[1]["tool_name"] == "run_command"
    assert events[1]["authorization_outcome"] == "DENIED"
    assert events[1]["result_ok"] is False


