from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pytest

from minicode.subagent_runner import (
    SubAgentResult,
    SubAgentToolEvent,
    VerificationStatus,
)
from minicode.task_graph import TaskDefinition, TaskGraph, TaskPriority, TaskState
from minicode.team_planner import TeamPlan, TeamPlanner
from minicode.team_roles import AgentRole, get_role_policy
from minicode.team_scheduler import ReviewGate, TeamScheduler, TestGate
from minicode.tooling import ToolContext
from minicode.tools.agent_team import agent_team_tool


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

    # Verify dependencies (Phase 4.1 hardened DAG)
    assert defs["research_impl"].dependencies == []
    assert defs["research_test"].dependencies == []
    assert set(defs["coding"].dependencies) == {"research_impl", "research_test"}
    assert defs["test"].dependencies == ["coding"]
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
    assert "task" not in coding_policy.allowed_tools

    # Test role in Phase 4.1 is strictly read-only
    test_policy = get_role_policy(AgentRole.TEST)
    assert not test_policy.is_writer
    assert "test_runner" in test_policy.allowed_tools
    assert "read_file" in test_policy.allowed_tools
    assert "edit_file" not in test_policy.allowed_tools
    assert "write_file" not in test_policy.allowed_tools
    assert "run_command" not in test_policy.allowed_tools

    reviewer_policy = get_role_policy(AgentRole.REVIEWER)
    assert not reviewer_policy.is_writer
    assert "code_review" in reviewer_policy.allowed_tools
    assert "diff_viewer" in reviewer_policy.allowed_tools
    assert "edit_file" not in reviewer_policy.allowed_tools


def test_test_gate_adversarial_matrix():
    """TestGate must strictly evaluate test runner verification and reject hollow claims."""
    # Case A: Real test_runner tool execution with ok=True -> PASS
    pass_ev = SubAgentToolEvent(
        tool_name="test_runner",
        ok=True,
        output_summary="10 passed in 0.5s",
        tool_use_id="call_1",
    )
    res_a = SubAgentResult(ok=True, output="Tests passed", tool_events=[pass_ev])
    gate_a = TestGate.evaluate(res_a)
    assert gate_a.passed
    assert gate_a.verdict == "PASS"

    # Case B: Real test_runner tool execution with ok=False -> FAIL
    fail_ev = SubAgentToolEvent(
        tool_name="test_runner",
        ok=False,
        output_summary="1 failed in 0.5s",
        tool_use_id="call_2",
    )
    res_b = SubAgentResult(ok=True, output="Tests failed", tool_events=[fail_ev])
    gate_b = TestGate.evaluate(res_b)
    assert not gate_b.passed
    assert gate_b.verdict == "FAIL"

    # Case C: Hollow text claim "10 passed ok=true" without tool event -> UNVERIFIED
    res_c = SubAgentResult(
        ok=True,
        output="running pytest... 10 passed in 0.5s. All tests pass. ok=true",
        tool_calls_count=0,
        tool_events=[],
    )
    gate_c = TestGate.evaluate(res_c)
    assert not gate_c.passed
    assert gate_c.verdict == "UNVERIFIED"

    # Case D: Structured JSON claims passed=True without tool event -> UNVERIFIED
    res_d = SubAgentResult(
        ok=True,
        output="I ran tests",
        structured_data={"passed": True, "total": 5, "failed": 0},
        tool_events=[],
    )
    gate_d = TestGate.evaluate(res_d)
    assert not gate_d.passed
    assert gate_d.verdict == "UNVERIFIED"

    # Case E: Non-test tool event (e.g. read_file ok=True) without test_runner -> UNVERIFIED
    read_ev = SubAgentToolEvent(
        tool_name="read_file",
        ok=True,
        output_summary="file contents",
        tool_use_id="call_3",
    )
    res_e = SubAgentResult(ok=True, output="Read test file", tool_events=[read_ev])
    gate_e = TestGate.evaluate(res_e)
    assert not gate_e.passed
    assert gate_e.verdict == "UNVERIFIED"

    # Case F: Multiple events ending with test_runner ok=False -> FAIL
    res_f = SubAgentResult(
        ok=True,
        output="Mixed runs",
        tool_events=[read_ev, fail_ev],
    )
    gate_f = TestGate.evaluate(res_f)
    assert not gate_f.passed
    assert gate_f.verdict == "FAIL"

    # Case G: Multiple events ending with test_runner ok=True (after earlier fail) -> PASS
    res_g = SubAgentResult(
        ok=True,
        output="Fixed test run",
        tool_events=[fail_ev, pass_ev],
    )
    gate_g = TestGate.evaluate(res_g)
    assert gate_g.passed
    assert gate_g.verdict == "PASS"


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
    gate_invalid = ReviewGate.evaluate(no_json)
    assert not gate_invalid.passed
    assert gate_invalid.verdict == "rejected"


def test_team_planner_validation():
    """TeamPlanner.validate_plan must catch malformed plans."""
    planner = TeamPlanner()

    # 1. Valid standard plan
    valid_plan = planner.plan(goal="Build feature")
    ok, msg = planner.validate_plan(valid_plan)
    assert ok
    assert "Plan is valid" in msg

    # 2. Empty graph
    empty_plan = TeamPlan(goal="Empty", graph=TaskGraph(), task_roles={})
    ok, msg = planner.validate_plan(empty_plan)
    assert not ok
    assert "no tasks" in msg

    # 3. Too many tasks (> 5 initial)
    too_many_graph = TaskGraph()
    for i in range(6):
        too_many_graph.definitions[f"t{i}"] = TaskDefinition(
            id=f"t{i}", name=f"T{i}", description="desc", dependencies=[]
        )
    roles = {f"t{i}": AgentRole.RESEARCH for i in range(6)}
    roles["t0"] = AgentRole.CODING
    ok, msg = planner.validate_plan(TeamPlan(goal="Too many", graph=too_many_graph, task_roles=roles))
    assert not ok
    assert "Too many initial tasks" in msg

    # 4. Self dependency
    self_graph = TaskGraph()
    self_graph.definitions["t1"] = TaskDefinition(
        id="t1", name="T1", description="desc", dependencies=["t1"]
    )
    ok, msg = planner.validate_plan(
        TeamPlan(goal="Self", graph=self_graph, task_roles={"t1": AgentRole.CODING})
    )
    assert not ok
    assert "Self-dependency" in msg

    # 5. Missing dependency
    missing_graph = TaskGraph()
    missing_graph.definitions["t1"] = TaskDefinition(
        id="t1", name="T1", description="desc", dependencies=["t_nonexistent"]
    )
    ok, msg = planner.validate_plan(
        TeamPlan(goal="Missing", graph=missing_graph, task_roles={"t1": AgentRole.CODING})
    )
    assert not ok
    assert "Missing dependency" in msg

    # 6. Cycle detection
    cycle_graph = TaskGraph()
    cycle_graph.definitions["a"] = TaskDefinition(id="a", name="A", description="desc", dependencies=["b"])
    cycle_graph.definitions["b"] = TaskDefinition(id="b", name="B", description="desc", dependencies=["a"])
    ok, msg = planner.validate_plan(
        TeamPlan(goal="Cycle", graph=cycle_graph, task_roles={"a": AgentRole.CODING, "b": AgentRole.RESEARCH})
    )
    assert not ok
    assert "No root tasks found" in msg or "Cyclic dependency" in msg

    # 7. Writer count checks
    no_writer_graph = TaskGraph()
    no_writer_graph.definitions["r1"] = TaskDefinition(id="r1", name="R1", description="desc", dependencies=[])
    ok, msg = planner.validate_plan(
        TeamPlan(goal="NoWriter", graph=no_writer_graph, task_roles={"r1": AgentRole.RESEARCH})
    )
    assert not ok
    assert "no writer tasks" in msg


def test_agent_team_tool_clamping_and_validation(tmp_path):
    """agent_team tool must validate goal and clamp max_replans to 0-1."""
    # 1. Invalid goal
    with pytest.raises(ValueError, match="goal is required"):
        agent_team_tool.validator({"goal": ""})

    # 2. Clamping max_replans
    parsed_neg = agent_team_tool.validator({"goal": "Test", "max_replans": -2})
    assert parsed_neg["max_replans"] == 0

    parsed_over = agent_team_tool.validator({"goal": "Test", "max_replans": 5})
    assert parsed_over["max_replans"] == 1

    parsed_normal = agent_team_tool.validator({"goal": "Test", "max_replans": 1})
    assert parsed_normal["max_replans"] == 1


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
            ev = SubAgentToolEvent(
                tool_name="test_runner",
                ok=True,
                output_summary="10 passed in 0.2s",
                tool_use_id="call_t1",
            )
            return SubAgentResult(
                ok=True,
                output="pytest tests/ - 10 passed in 0.2s",
                final_message="All tests pass.",
                tool_events=[ev],
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
    assert not scheduler.reader_writer_overlap_observed


def test_team_scheduler_test_failure_replan_skipping_reviewer(tmp_path):
    """When test gate fails, original reviewer must be SKIPPED (superseded_by_replan) and replan triggered."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Harden payment handler")

    scheduler = TeamScheduler(max_workers=4)
    context = ToolContext(cwd=str(tmp_path))

    test_attempt = 0

    def mock_run_subagent(config):
        nonlocal test_attempt
        if "research" in config.name:
            return SubAgentResult(ok=True, output="Research done")
        if "coding" in config.name:
            return SubAgentResult(ok=True, output="Coding done")
        if "test" in config.name:
            test_attempt += 1
            if test_attempt == 1:
                # First test fails
                ev_fail = SubAgentToolEvent(
                    tool_name="test_runner",
                    ok=False,
                    output_summary="AssertionError in payment.py",
                    tool_use_id="call_f",
                )
                return SubAgentResult(ok=True, output="Test failed", tool_events=[ev_fail])
            else:
                # Replan test passes
                ev_pass = SubAgentToolEvent(
                    tool_name="test_runner",
                    ok=True,
                    output_summary="All payments verified",
                    tool_use_id="call_p",
                )
                return SubAgentResult(ok=True, output="Test passed", tool_events=[ev_pass])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "approve", "comments": "Good fix", "issues": []}',
                structured_data={"verdict": "approve", "comments": "Good fix", "issues": []},
            )
        return SubAgentResult(ok=True, output="Done")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_run_subagent):
        result = scheduler.schedule_and_run(plan, context, max_replans=1)

    assert result.success
    assert result.replan_count == 1
    # Original reviewer was skipped!
    assert "reviewer" in result.skipped_tasks
    slot = plan.graph.slots.get("default:reviewer")
    assert slot is not None
    assert slot.state == TaskState.SKIPPED
    # Replan tasks completed
    assert "coding_replan_1" in result.completed_tasks
    assert "test_replan_1" in result.completed_tasks
    assert "reviewer_replan_1" in result.completed_tasks


def test_team_scheduler_reviewer_rejection_bounded_replan(tmp_path):
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
            ev = SubAgentToolEvent(
                tool_name="test_runner",
                ok=True,
                output_summary="10 passed",
                tool_use_id="call_t",
            )
            return SubAgentResult(ok=True, output="10 passed in 0.1s ok=true", tool_events=[ev])
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
            ev = SubAgentToolEvent(
                tool_name="test_runner",
                ok=True,
                output_summary="10 passed",
                tool_use_id="call_t",
            )
            return SubAgentResult(ok=True, output="10 passed in 0.1s ok=true", tool_events=[ev])
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
