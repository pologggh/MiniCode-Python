from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from minicode.subagent_runner import (
    SubAgentRunConfig,
    SubAgentResult,
    run_subagent,
    MAX_SUBAGENT_DEPTH,
    FORBIDDEN_CHILD_TOOLS,
)
from minicode.tooling import ToolContext, ToolResult
from minicode.tools.task import task_tool, AGENT_TYPES


def test_subagent_depth_limit_rejected():
    """Depth limit >= MAX_SUBAGENT_DEPTH must reject execution and prevent recursive sub-agents."""
    config = SubAgentRunConfig(
        name="NestedAgent",
        task_prompt="Try to nest",
        system_prompt="System prompt",
        depth=1,  # Exceeds/meets max depth
    )
    result = run_subagent(config)
    assert not result.ok
    assert result.error == "DepthLimitExceeded"
    assert "depth 1" in result.output


def test_subagent_mcp_servers_stripped(tmp_path):
    """Child agent runtime must have mcpServers stripped to prevent MCP inheritance."""
    config = SubAgentRunConfig(
        name="MCPTestAgent",
        task_prompt="Inspect MCP",
        system_prompt="System prompt",
        cwd=str(tmp_path),
        runtime={"model": "test-model", "mcpServers": {"unsafe_server": {"command": "evil"}}},
        depth=0,
    )

    with patch("minicode.tools.create_default_tool_registry") as mock_create_tools, \
         patch("minicode.subagent_runner.create_model_adapter") as mock_model_adapter, \
         patch("minicode.subagent_runner.run_agent_turn") as mock_run_turn:

        mock_registry = MagicMock()
        mock_registry.list.return_value = []
        mock_create_tools.return_value = mock_registry
        mock_run_turn.return_value = [{"role": "assistant", "content": "Done <final>"}]

        res = run_subagent(config)
        assert res.ok
        # Check that create_default_tool_registry was called with stripped mcpServers
        called_runtime = mock_create_tools.call_args[1]["runtime"]
        assert called_runtime["mcpServers"] == {}


def test_subagent_strips_forbidden_tools(tmp_path):
    """Forbidden tools (task, agent_team) must NEVER be exposed to child sub-agents."""
    mock_tool_task = MagicMock()
    mock_tool_task.name = "task"
    mock_tool_team = MagicMock()
    mock_tool_team.name = "agent_team"
    mock_tool_read = MagicMock()
    mock_tool_read.name = "read_file"

    config = SubAgentRunConfig(
        name="ToolStripTest",
        task_prompt="Check tools",
        system_prompt="System prompt",
        cwd=str(tmp_path),
        runtime={"model": "test-model"},
        allowed_tools=None,  # General agent requests all tools
        depth=0,
    )

    with patch("minicode.tools.create_default_tool_registry") as mock_create_tools, \
         patch("minicode.subagent_runner.create_model_adapter") as mock_model_adapter, \
         patch("minicode.subagent_runner.run_agent_turn") as mock_run_turn:

        mock_registry = MagicMock()
        mock_registry.list.return_value = [mock_tool_task, mock_tool_team, mock_tool_read]
        mock_create_tools.return_value = mock_registry
        mock_run_turn.return_value = [{"role": "assistant", "content": "Done <final>"}]

        res = run_subagent(config)
        assert res.ok
        # The tools passed to ToolRegistry must only contain read_file
        child_tools = mock_model_adapter.call_args[1]["tools"]
        assert "task" not in child_tools.list_all()
        assert "agent_team" not in child_tools.list_all()
        assert "read_file" in child_tools.list_all()


def test_subagent_read_only_sandboxing(tmp_path):
    """Read-only agents must get an unprompted PermissionManager (auto-denying writes)."""
    parent_permissions = MagicMock()
    parent_permissions.prompt = MagicMock()

    config = SubAgentRunConfig(
        name="ReadOnlyAgent",
        task_prompt="Explore codebase",
        system_prompt="System prompt",
        cwd=str(tmp_path),
        runtime={"model": "test-model"},
        allowed_tools={"read_file"},
        is_writer=False,
        parent_permissions=parent_permissions,
        depth=0,
    )

    with patch("minicode.tools.create_default_tool_registry") as mock_create_tools, \
         patch("minicode.subagent_runner.create_model_adapter"), \
         patch("minicode.subagent_runner.run_agent_turn") as mock_run_turn:

        mock_registry = MagicMock()
        mock_registry.list.return_value = []
        mock_create_tools.return_value = mock_registry
        mock_run_turn.return_value = [{"role": "assistant", "content": "Exploration finished."}]

        res = run_subagent(config)
        assert res.ok
        # Check permissions passed to run_agent_turn
        passed_perms = mock_run_turn.call_args[1]["permissions"]
        # prompt should be None for read-only sandboxing
        assert passed_perms.prompt is None


def test_subagent_crash_containment(tmp_path):
    """When child agent execution raises an unhandled exception, it must not crash the caller."""
    config = SubAgentRunConfig(
        name="CrashAgent",
        task_prompt="Crash now",
        system_prompt="System prompt",
        cwd=str(tmp_path),
        runtime={"model": "test-model"},
        depth=0,
    )

    with patch("minicode.tools.create_default_tool_registry") as mock_create_tools, \
         patch("minicode.subagent_runner.create_model_adapter"), \
         patch("minicode.subagent_runner.run_agent_turn", side_effect=RuntimeError("Simulated LLM network collapse")):

        mock_registry = MagicMock()
        mock_registry.list.return_value = []
        mock_create_tools.return_value = mock_registry

        res = run_subagent(config)
        assert not res.ok
        assert res.error == "Simulated LLM network collapse"
        assert "Sub-agent (CrashAgent) failed" in res.output


def test_subagent_json_extraction(tmp_path):
    """Extract structured JSON payload from final assistant message."""
    config = SubAgentRunConfig(
        name="ReviewerAgent",
        task_prompt="Review diff",
        system_prompt="System prompt",
        cwd=str(tmp_path),
        runtime={"model": "test-model"},
        depth=0,
    )

    review_json = '```json\n{"verdict": "approve", "issues": []}\n```'

    with patch("minicode.tools.create_default_tool_registry") as mock_create_tools, \
         patch("minicode.subagent_runner.create_model_adapter"), \
         patch("minicode.subagent_runner.run_agent_turn") as mock_run_turn:

        mock_registry = MagicMock()
        mock_registry.list.return_value = []
        mock_create_tools.return_value = mock_registry
        mock_run_turn.return_value = [
            {"role": "assistant", "content": f"Here is the review result:\n{review_json}\n<final>"}
        ]

        res = run_subagent(config)
        assert res.ok
        assert res.structured_data == {"verdict": "approve", "issues": []}


def test_task_tool_backward_compatibility(tmp_path):
    """task_tool must maintain backward compatibility in validation and run output format."""
    parsed = task_tool.validator({
        "description": "explore codebase",
        "agent_type": "explore",
    })
    assert parsed["agent_type"] == "explore"
    assert parsed["description"] == "explore codebase"

    context = ToolContext(cwd=str(tmp_path))
    context._runtime = {"model": "test-model"}

    with patch("minicode.tools.create_default_tool_registry") as mock_create_tools, \
         patch("minicode.subagent_runner.create_model_adapter"), \
         patch("minicode.subagent_runner.run_agent_turn") as mock_run_turn:

        mock_registry = MagicMock()
        mock_registry.list.return_value = []
        mock_create_tools.return_value = mock_registry
        mock_run_turn.return_value = [
            {"role": "user", "content": "explore codebase"},
            {"role": "assistant", "content": "Found 3 relevant files <final>"}
        ]

        res = task_tool.run(parsed, context)
        assert isinstance(res, ToolResult)
        assert res.ok
        assert "[Sub-agent Explore completed]" in res.output
        assert "Type: explore" in res.output
        assert "Found 3 relevant files" in res.output
