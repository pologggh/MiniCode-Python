from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pytest

from minicode.subagent_runner import SubAgentResult
from minicode.task_graph import TaskState
from minicode.team_planner import TeamPlanner
from minicode.team_roles import AgentRole, get_role_policy
from minicode.team_scheduler import TeamScheduler, TestGate, ReviewGate
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


def test_test_gate_evaluation():
    """TestGate must strictly evaluate test runner verification."""
    # Passing test result
    pass_res = SubAgentResult(
        ok=True,
        output="running pytest... 10 passed in 0.5s. All tests pass. ok=true",
        final_message="Verification passed: all tests pass.",
        tool_calls_count=1,
    )
    assert TestGate.evaluate(pass_res).passed

    # Structured pass
    struct_pass = SubAgentResult(
        ok=True,
        output="tests ran",
        structured_data={"passed": True, "total": 5, "failed": 0},
    )
    assert TestGate.evaluate(struct_pass).passed

    # Failing test result
    fail_res = SubAgentResult(
        ok=True,
        output="FAILED tests/test_foo.py::test_bar - AssertionError: expected 1 got 2",
        final_message="Tests failed: 1 failure.",
        tool_calls_count=1,
    )
    assert not TestGate.evaluate(fail_res).passed

    # Hollow claim with 0 tool calls
    hollow_res = SubAgentResult(
        ok=True,
        output="I am confident that all tests will pass.",
        final_message="Tests are probably fine.",
        tool_calls_count=0,
    )
    assert not TestGate.evaluate(hollow_res).passed


def test_review_gate_evaluation():
    """ReviewGate requires valid structured JSON approving the changes."""
    # Approved JSON
    approved = SubAgentResult(
        ok=True,
        output='```json\n{"verdict": "approve", "comments": "Clean implementation", "issues": []}\n```',
        structured_data={"verdict": "approve", "comments": "Clean implementation", "issues": []},
    )
    gate_res = ReviewGate.evaluate(approved)
    assert gate_res.passed
    assert gate_res.verdict == "approved"

    # Rejected JSON
    rejected = SubAgentResult(
        ok=True,
        output='```json\n{"verdict": "reject", "comments": "Missing error handling", "issues": ["L42 crash"]}\n```',
        structured_data={"verdict": "reject", "comments": "Missing error handling", "issues": ["L42 crash"]},
    )
    gate_res = ReviewGate.evaluate(rejected)
    assert not gate_res.passed
    assert gate_res.verdict == "rejected"
    assert "Missing error handling" in gate_res.feedback

    # Missing structured JSON
    no_json = SubAgentResult(
        ok=True,
        output="The code looks pretty good to me. Approved in my heart.",
        final_message="Code is fine.",
    )
    assert not ReviewGate.evaluate(no_json).passed


def test_team_scheduler_parallel_read_and_writer_serialization(tmp_path):
    """TeamScheduler executes ready read-only siblings in parallel, while writers are strictly serialized."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Refactor data pipeline")

    scheduler = TeamScheduler(max_workers=4)
    context = ToolContext(cwd=str(tmp_path))

    executed_tasks = []

    def mock_run_subagent(config):
        executed_tasks.append(config.name)
        if "research" in config.name:
            time.sleep(0.05)
            return SubAgentResult(ok=True, output="Research complete", final_message="Found files")
        if "coding" in config.name:
            return SubAgentResult(ok=True, output="Coding complete", final_message="Modified pipeline.py")
        if "test" in config.name:
            return SubAgentResult(
                ok=True,
                output="pytest tests/ - 10 passed in 0.2s. ok=true",
                final_message="All tests pass.",
                tool_calls_count=1,
            )
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "approve", "comments": "All checks pass", "issues": []}',
                final_message='{"verdict": "approve", "comments": "All checks pass", "issues": []}',
                structured_data={"verdict": "approve", "comments": "All checks pass", "issues": []},
            )
        return SubAgentResult(ok=True, output=f"Output for {config.name}")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_run_subagent):
        result = scheduler.schedule_and_run(plan, context)

    assert result.success
    assert set(result.completed_tasks) == {"research_impl", "research_test", "coding", "test", "reviewer"}
    assert result.failed_tasks == []
    assert result.skipped_tasks == []
    assert scheduler.max_concurrent_writers_observed <= 1


def test_team_scheduler_bounded_replan_flow(tmp_path):
    """Review rejection triggers bounded replan (coding -> test -> reviewer) which succeeds on retry."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Implement user profile API")

    scheduler = TeamScheduler(max_workers=4)
    context = ToolContext(cwd=str(tmp_path))

    review_count = 0

    def mock_run_subagent(config):
        nonlocal review_count
        if "research" in config.name:
            return SubAgentResult(ok=True, output="Research done")
        if "coding" in config.name:
            return SubAgentResult(ok=True, output="Coding done")
        if "test" in config.name:
            return SubAgentResult(ok=True, output="10 passed in 0.1s ok=true", tool_calls_count=1)
        if "reviewer" in config.name:
            review_count += 1
            if review_count == 1:
                # First review rejects!
                return SubAgentResult(
                    ok=True,
                    output='{"verdict": "reject", "comments": "Missing field validation", "issues": ["email check missing"]}',
                    structured_data={"verdict": "reject", "comments": "Missing field validation", "issues": ["email check missing"]},
                )
            else:
                # Replan review approves!
                return SubAgentResult(
                    ok=True,
                    output='{"verdict": "approve", "comments": "Validation added, good to go", "issues": []}',
                    structured_data={"verdict": "approve", "comments": "Validation added, good to go", "issues": []},
                )
        return SubAgentResult(ok=True, output="Done")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_run_subagent):
        result = scheduler.schedule_and_run(plan, context, max_replans=1)

    assert result.success
    assert result.replan_count == 1
    # Check that replan tasks were executed and completed
    assert "coding_replan_1" in result.completed_tasks
    assert "test_replan_1" in result.completed_tasks
    assert "reviewer_replan_1" in result.completed_tasks


def test_team_scheduler_bounded_replan_exceeded(tmp_path):
    """When replan reaches max_replans (1) and still fails, it stops and marks failure."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Strict compliance check")

    scheduler = TeamScheduler(max_workers=4)
    context = ToolContext(cwd=str(tmp_path))

    def mock_run_subagent(config):
        if "research" in config.name:
            return SubAgentResult(ok=True, output="Research done")
        if "coding" in config.name:
            return SubAgentResult(ok=True, output="Coding done")
        if "test" in config.name:
            return SubAgentResult(ok=True, output="10 passed in 0.1s ok=true", tool_calls_count=1)
        if "reviewer" in config.name:
            # Always reject
            return SubAgentResult(
                ok=True,
                output='{"verdict": "reject", "comments": "Persistent security defect", "issues": ["hardcoded token"]}',
                structured_data={"verdict": "reject", "comments": "Persistent security defect", "issues": ["hardcoded token"]},
            )
        return SubAgentResult(ok=True, output="Done")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_run_subagent):
        result = scheduler.schedule_and_run(plan, context, max_replans=1)

    assert not result.success
    assert result.replan_count == 1
    # Bounded stop: stops at replan 1
    assert "reviewer_replan_1" in result.failed_tasks


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
    assert "coding" in result.skipped_tasks
    assert "test" in result.skipped_tasks
    assert "reviewer" in result.skipped_tasks
    assert "research_test" in result.completed_tasks
