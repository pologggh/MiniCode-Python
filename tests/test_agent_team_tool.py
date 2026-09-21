from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from minicode.subagent_runner import FORBIDDEN_CHILD_TOOLS, SubAgentRunConfig, run_subagent
from minicode.team_scheduler import TeamExecutionResult
from minicode.tooling import ToolContext, ToolResult
from minicode.tools import create_default_tool_registry
from minicode.tools.agent_team import agent_team_tool


def test_agent_team_tool_validator():
    """agent_team_tool validator enforces required goal and clamps max_replans."""
    # Clamping max_replans (Phase 4.1 clamps to 0-1)
    res_clamped = agent_team_tool.validator({"goal": "Test clamp", "max_replans": 10})
    assert res_clamped["max_replans"] == 1

    res_negative = agent_team_tool.validator({"goal": "Test clamp", "max_replans": -5})
    assert res_negative["max_replans"] == 0

    # Missing goal
    with pytest.raises(ValueError, match="goal is required"):
        agent_team_tool.validator({"goal": "   "})

    with pytest.raises(ValueError, match="goal is required"):
        agent_team_tool.validator({})


def test_agent_team_tool_execution(tmp_path):
    """agent_team_tool delegates to TeamPlanner and TeamScheduler returning ToolResult."""
    context = ToolContext(cwd=str(tmp_path))
    context._runtime = {"model": "test-model"}

    mock_team_result = TeamExecutionResult(
        success=True,
        goal="Implement data caching",
        completed_tasks=["research_impl", "research_test", "coding", "test", "reviewer"],
        failed_tasks=[],
        skipped_tasks=[],
        replan_count=0,
        elapsed_seconds=1.2,
        summary="### Multi-Agent Team Execution Summary\n- **Status**: Success",
    )

    with patch("minicode.team_scheduler.TeamScheduler.schedule_and_run", return_value=mock_team_result) as mock_run:
        tool_result = agent_team_tool.run(
            {"goal": "Implement data caching", "use_worktree": False, "max_replans": 1},
            context,
        )


        assert isinstance(tool_result, ToolResult)
        assert tool_result.ok
        assert "Multi-Agent Team Execution Summary" in tool_result.output
        assert mock_run.called


def test_agent_team_tool_in_default_registry(tmp_path):
    """agent_team tool is present in default parent tool registry."""
    registry = create_default_tool_registry(str(tmp_path))
    assert registry.find("agent_team") is not None
    assert "agent_team" in registry.list_all()


def test_agent_team_tool_stripped_from_subagents(tmp_path):
    """agent_team tool must be stripped from child agent tool registries."""
    assert "agent_team" in FORBIDDEN_CHILD_TOOLS

    config = SubAgentRunConfig(
        name="ChildAgent",
        task_prompt="Inspect available tools",
        system_prompt="System prompt",
        cwd=str(tmp_path),
        runtime={"model": "test-model"},
        allowed_tools=None,
        depth=0,
    )

    with patch("minicode.subagent_runner.create_model_adapter") as mock_model_adapter, \
         patch("minicode.subagent_runner.run_agent_turn") as mock_run_turn:

        mock_run_turn.return_value = [{"role": "assistant", "content": "Done <final>"}]

        res = run_subagent(config)
        assert res.ok
        tools = mock_model_adapter.call_args[1]["tools"]
        assert "agent_team" not in tools.list_all()
        assert "task" not in tools.list_all()
