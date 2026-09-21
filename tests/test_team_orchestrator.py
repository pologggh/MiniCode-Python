from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pytest

from minicode.subagent_runner import SubAgentResult
from minicode.task_graph import TaskState
from minicode.team_planner import TeamPlanner
from minicode.team_roles import AgentRole, get_role_policy
from minicode.team_scheduler import TeamScheduler
from minicode.tooling import ToolContext


def test_team_planner_dag_structure():
    """TeamPlanner must produce standard 5-node software engineering DAG."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Add authentication feature")

    assert plan.goal == "Add authentication feature"
    defs = plan.graph.definitions

    assert "research_impl" in defs
    assert "research_test" in defs
    assert "coding" in defs
    assert "test" in defs
    assert "reviewer" in defs

    # Verify dependencies
    assert defs["research_impl"].dependencies == []
    assert defs["research_test"].dependencies == []
    assert defs["coding"].dependencies == ["research_impl"]
    assert set(defs["test"].dependencies) == {"coding", "research_test"}
    assert defs["reviewer"].dependencies == ["test"]

    # Verify roles
    assert plan.get_role_for_task("research_impl") == AgentRole.RESEARCH
    assert plan.get_role_for_task("research_test") == AgentRole.RESEARCH
    assert plan.get_role_for_task("coding") == AgentRole.CODING
    assert plan.get_role_for_task("test") == AgentRole.TEST
    assert plan.get_role_for_task("reviewer") == AgentRole.REVIEWER


def test_team_roles_tool_allowlists():
    """Each role must have strictly partitioned tool allowlists."""
    research_policy = get_role_policy(AgentRole.RESEARCH)
    assert not research_policy.is_writer
    assert "read_file" in research_policy.allowed_tools
    assert "edit_file" not in research_policy.allowed_tools
    assert "task" not in research_policy.allowed_tools

    coding_policy = get_role_policy(AgentRole.CODING)
    assert coding_policy.is_writer
    assert "edit_file" in coding_policy.allowed_tools
    assert "write_file" in coding_policy.allowed_tools

    test_policy = get_role_policy(AgentRole.TEST)
    assert test_policy.is_writer
    assert "test_runner" in test_policy.allowed_tools

    reviewer_policy = get_role_policy(AgentRole.REVIEWER)
    assert not reviewer_policy.is_writer
    assert "code_review" in reviewer_policy.allowed_tools
    assert "diff_viewer" in reviewer_policy.allowed_tools
    assert "edit_file" not in reviewer_policy.allowed_tools


def test_team_scheduler_parallel_read_and_writer_serialization(tmp_path):
    """TeamScheduler executes ready read-only siblings in parallel, while writers are strictly serialized."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Refactor data pipeline")

    scheduler = TeamScheduler(max_workers=4)
    context = ToolContext(cwd=str(tmp_path))

    executed_tasks = []

    def mock_run_subagent(config):
        executed_tasks.append(config.name)
        # Sibling research tasks simulate a small delay to verify concurrency
        if "research" in config.name:
            time.sleep(0.05)
        return SubAgentResult(
            ok=True,
            output=f"Output for {config.name}",
            final_message=f"Summary for {config.name}",
        )

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_run_subagent):
        result = scheduler.schedule_and_run(plan, context)

    assert result.success
    assert set(result.completed_tasks) == {"research_impl", "research_test", "coding", "test", "reviewer"}
    assert result.failed_tasks == []
    assert result.skipped_tasks == []
    # Maximum concurrent writers must never exceed 1
    assert scheduler.max_concurrent_writers_observed <= 1


def test_team_scheduler_upstream_failure_containment(tmp_path):
    """If research_impl fails, coding, test, and reviewer must be cascade-skipped without crashing."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Fix broken parser")

    scheduler = TeamScheduler(max_workers=4)
    context = ToolContext(cwd=str(tmp_path))

    def mock_run_subagent(config):
        if "research_impl" in config.name:
            return SubAgentResult(
                ok=False,
                output="Failed to locate parser modules",
                error="FileNotFound",
            )
        return SubAgentResult(
            ok=True,
            output=f"Output for {config.name}",
            final_message=f"Summary for {config.name}",
        )

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_run_subagent):
        result = scheduler.schedule_and_run(plan, context)

    assert not result.success
    assert "research_impl" in result.failed_tasks
    # Downstream tasks depending on research_impl or coding must be skipped
    assert "coding" in result.skipped_tasks
    assert "test" in result.skipped_tasks
    assert "reviewer" in result.skipped_tasks
    # research_test succeeded independently
    assert "research_test" in result.completed_tasks
