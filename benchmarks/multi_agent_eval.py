"""Deterministic Benchmark and Evaluation Suite for Phase 4 Centralized Multi-Agent Orchestration.

Evaluates 12 deterministic test cases covering:
CASE 1: Parent Agent Turn & Context Isolation (Real run_agent_turn + Tool Dispatch)
CASE 2: Multi-Agent DAG Planning & Topology
CASE 3: Multi-Agent Plan Validation Engine
CASE 4: Role Tool Conformance Audit
CASE 5: Parallel Read Sibling Concurrency & Speedup
CASE 6: Shared Workspace Writer Serialization & Zero Overlap
CASE 7: TestGate Real Evidence & Adversarial Matrix
CASE 8: ReviewGate Structured JSON Verification
CASE 9: Review Rejection and Corrective Replan Flow
CASE 10: Test Failure Replan Flow & Reviewer Skipping
CASE 11: Replan Boundedness & Upstream Failure Containment
CASE 12: Worktree Lifecycle, Dirty Rejection & Fail-Closed Protection
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

from minicode.agent_loop import run_agent_turn
from minicode.permissions import PermissionManager
from minicode.subagent_runner import (
    FORBIDDEN_CHILD_TOOLS,
    MAX_SUBAGENT_DEPTH,
    SubAgentResult,
    SubAgentRunConfig,
    SubAgentToolEvent,
    VerificationStatus,
    run_subagent,
)
from minicode.task_graph import TaskDefinition, TaskGraph, TaskPriority, TaskState, WorktreeIsolator
from minicode.team_planner import TeamPlan, TeamPlanner
from minicode.team_roles import ROLE_POLICIES, AgentRole, get_role_policy
from minicode.team_scheduler import ReviewGate, TeamExecutionResult, TeamScheduler, TestGate
from minicode.tooling import ToolContext, ToolRegistry, ToolResult
from minicode.tools import create_default_tool_registry
from minicode.tools.agent_team import agent_team_tool
from minicode.tools.task import task_tool
from minicode.types import AgentStep, ChatMessage, ModelAdapter


@dataclass
class MetricRecord:
    numerator: int
    denominator: int
    rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": round(self.rate, 4),
        }


@dataclass
class CaseResult:
    case_id: str
    name: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    metric_records: dict[str, MetricRecord] = field(default_factory=dict)
    details: str = ""


class ScriptedParentModel(ModelAdapter):
    """Deterministic model adapter for parent turn execution."""

    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.call_idx = 0

    def next(self, messages: list[ChatMessage], on_stream_chunk=None, **_kwargs) -> AgentStep:
        if self.call_idx < len(self._steps):
            step = self._steps[self.call_idx]
            self.call_idx += 1
            return step
        return AgentStep(type="assistant", content="Complete <final>")


# ============================================================================
# Case Implementations
# ============================================================================

def run_case_1(tmp_dir: Path) -> CaseResult:
    """CASE 1: Parent Agent Turn & Context Isolation."""
    # Build tool registry with agent_team tool
    registry = ToolRegistry([agent_team_tool])

    # Model calls agent_team tool, then delivers final message
    parent_model = ScriptedParentModel(
        [
            AgentStep(
                type="tool_calls",
                calls=[
                    {
                        "id": "team_call_001",
                        "toolName": "agent_team",
                        "input": {"goal": "Implement secure authentication", "max_replans": 1},
                    }
                ],
            ),
            AgentStep(type="assistant", content="The team has completed successfully. <final>"),
        ]
    )

    child_log_chunks = [
        "Child internal step analysis: examining tokens and sessions. " * 50,  # ~3000 chars
        "Child internal step analysis: inspecting test runners and suites. " * 50,
        "Child internal step analysis: editing crypto hashing and salt logic. " * 50,
        "Child internal step analysis: running unit test verification passes. " * 50,
        "Child internal step analysis: performing structured code review. " * 50,
    ]
    total_child_chars = sum(len(c) for c in child_log_chunks)  # > 15,000 chars

    subagent_index = 0

    def mock_subagent(config):
        nonlocal subagent_index
        idx = min(subagent_index, len(child_log_chunks) - 1)
        subagent_index += 1
        verbose_content = child_log_chunks[idx]

        if "research" in config.name:
            return SubAgentResult(ok=True, output=verbose_content, final_message="Research done")
        if "coding" in config.name:
            return SubAgentResult(ok=True, output=verbose_content, final_message="Code done")
        if "test" in config.name:
            ev = SubAgentToolEvent(
                tool_name="test_runner",
                ok=True,
                output_summary="10 passed",
                tool_use_id="call_t",
            )
            return SubAgentResult(ok=True, output=verbose_content, tool_events=[ev])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output=verbose_content,
                structured_data={"verdict": "approve", "comments": "Approved", "issues": []},
            )
        return SubAgentResult(ok=True, output=verbose_content)

    perms = PermissionManager(workspace_root=str(tmp_dir), prompt=None)
    init_messages = [{"role": "user", "content": "Implement secure authentication"}]

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_subagent):
        final_messages = run_agent_turn(
            model=parent_model,
            tools=registry,
            messages=init_messages,
            cwd=str(tmp_dir),
            permissions=perms,
        )

    # 1. Verify Tool Dispatch
    tool_calls = [m for m in final_messages if m.get("role") == "assistant_tool_call"]
    tool_results = [m for m in final_messages if m.get("role") == "tool_result"]

    dispatched = (
        len(tool_calls) == 1
        and tool_calls[0].get("toolName") == "agent_team"
        and len(tool_results) == 1
        and tool_results[0].get("toolName") == "agent_team"
        and not tool_results[0].get("isError", False)
    )

    # 2. Verify Context Isolation
    result_content = tool_results[0].get("content", "") if tool_results else ""
    tool_result_bounded = len(result_content) <= 8000

    # Ensure no child tool calls or raw child logs leaked into parent messages
    leaked_chars = 0
    for m in final_messages:
        role = m.get("role")
        if role == "assistant_tool_call" and m.get("toolName") != "agent_team":
            leaked_chars += len(str(m))
        elif role == "tool_result" and m.get("toolName") != "agent_team":
            leaked_chars += len(str(m))
        elif "Child internal step analysis:" in str(m.get("content", "")):
            leaked_chars += len(str(m.get("content", "")))

    isolation_ratio = (total_child_chars - leaked_chars) / total_child_chars if total_child_chars > 0 else 1.0
    isolated = total_child_chars >= 10000 and tool_result_bounded and leaked_chars == 0

    passed = dispatched and isolated

    rec_dispatch = MetricRecord(numerator=1 if dispatched else 0, denominator=1, rate=1.0 if dispatched else 0.0)
    rec_isolation = MetricRecord(
        numerator=total_child_chars - leaked_chars,
        denominator=total_child_chars,
        rate=isolation_ratio,
    )

    return CaseResult(
        case_id="case_1",
        name="Parent Agent Turn & Context Isolation",
        passed=passed,
        metrics={
            "total_child_chars": total_child_chars,
            "parent_tool_result_chars": len(result_content),
            "leaked_child_chars": leaked_chars,
            "dispatched": dispatched,
            "isolated": isolated,
        },
        metric_records={
            "parent_agent_team_tool_dispatch": rec_dispatch,
            "parent_context_isolation_ratio": rec_isolation,
        },
        details=f"Parent executed agent_team; {total_child_chars} child chars isolated into 1 tool_result ({len(result_content)} chars).",
    )


def run_case_2(tmp_dir: Path) -> CaseResult:
    """CASE 2: Multi-Agent DAG Planning & Topology."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Add REST API endpoints")

    defs = plan.graph.definitions
    has_nodes = all(
        k in defs for k in ["research_impl", "research_test", "coding", "test", "reviewer"]
    )
    deps_correct = (
        defs["research_impl"].dependencies == []
        and defs["research_test"].dependencies == []
        and set(defs["coding"].dependencies) == {"research_impl", "research_test"}
        and defs["test"].dependencies == ["coding"]
        and defs["reviewer"].dependencies == ["test"]
    )
    roles_correct = (
        plan.get_role_for_task("research_impl") == AgentRole.RESEARCH
        and plan.get_role_for_task("research_test") == AgentRole.RESEARCH
        and plan.get_role_for_task("coding") == AgentRole.CODING
        and plan.get_role_for_task("test") == AgentRole.TEST
        and plan.get_role_for_task("reviewer") == AgentRole.REVIEWER
    )
    passed = has_nodes and deps_correct and roles_correct
    return CaseResult(
        case_id="case_2",
        name="Multi-Agent DAG Planning & Topology",
        passed=passed,
        metrics={
            "node_count": len(defs),
            "dag_valid": passed,
            "root_parallel_tasks": 2,
        },
        details="Canonical 5-node software engineering DAG verified.",
    )


def run_case_3(tmp_dir: Path) -> CaseResult:
    """CASE 3: Multi-Agent Plan Validation Engine."""
    planner = TeamPlanner()
    checks_passed = 0
    total_checks = 7

    # Check 1: Valid standard plan
    valid_plan = planner.plan(goal="Valid standard feature")
    ok1, _ = planner.validate_plan(valid_plan)
    if ok1:
        checks_passed += 1

    # Check 2: Empty plan
    ok2, _ = planner.validate_plan(TeamPlan(goal="Empty", graph=TaskGraph(), task_roles={}))
    if not ok2:
        checks_passed += 1

    # Check 3: Too many tasks (>5)
    g3 = TaskGraph()
    for i in range(6):
        g3.definitions[f"t{i}"] = TaskDefinition(id=f"t{i}", name=f"T{i}", dependencies=[])
    roles3 = {f"t{i}": AgentRole.RESEARCH for i in range(6)}
    roles3["t0"] = AgentRole.CODING
    ok3, _ = planner.validate_plan(TeamPlan(goal="TooMany", graph=g3, task_roles=roles3))
    if not ok3:
        checks_passed += 1

    # Check 4: Self dependency
    g4 = TaskGraph()
    g4.definitions["t1"] = TaskDefinition(id="t1", name="T1", dependencies=["t1"])
    ok4, _ = planner.validate_plan(TeamPlan(goal="Self", graph=g4, task_roles={"t1": AgentRole.CODING}))
    if not ok4:
        checks_passed += 1

    # Check 5: Missing dependency
    g5 = TaskGraph()
    g5.definitions["t1"] = TaskDefinition(id="t1", name="T1", dependencies=["missing_dep"])
    ok5, _ = planner.validate_plan(TeamPlan(goal="Missing", graph=g5, task_roles={"t1": AgentRole.CODING}))
    if not ok5:
        checks_passed += 1

    # Check 6: Cyclic dependency
    g6 = TaskGraph()
    g6.definitions["a"] = TaskDefinition(id="a", name="A", dependencies=["b"])
    g6.definitions["b"] = TaskDefinition(id="b", name="B", dependencies=["a"])
    ok6, _ = planner.validate_plan(
        TeamPlan(goal="Cycle", graph=g6, task_roles={"a": AgentRole.CODING, "b": AgentRole.RESEARCH})
    )
    if not ok6:
        checks_passed += 1

    # Check 7: No writers
    g7 = TaskGraph()
    g7.definitions["r1"] = TaskDefinition(id="r1", name="R1", dependencies=[])
    ok7, _ = planner.validate_plan(
        TeamPlan(goal="NoWriter", graph=g7, task_roles={"r1": AgentRole.RESEARCH})
    )
    if not ok7:
        checks_passed += 1

    passed = checks_passed == total_checks
    rec_plan_valid = MetricRecord(
        numerator=checks_passed, denominator=total_checks, rate=checks_passed / total_checks
    )

    return CaseResult(
        case_id="case_3",
        name="Multi-Agent Plan Validation Engine",
        passed=passed,
        metrics={"checks_passed": checks_passed, "total_checks": total_checks},
        metric_records={"multi_agent_team_plan_validity": rec_plan_valid},
        details=f"Validated plan integrity: {checks_passed}/{total_checks} adversarial checks passed.",
    )


def run_case_4(tmp_dir: Path) -> CaseResult:
    """CASE 4: Role Tool Conformance Audit."""
    conforming_roles = 0
    total_roles = len(AgentRole)

    for role in AgentRole:
        policy = get_role_policy(role)
        is_conforming = False
        if role == AgentRole.RESEARCH:
            is_conforming = (
                not policy.is_writer
                and "edit_file" not in policy.allowed_tools
                and "write_file" not in policy.allowed_tools
                and "task" not in policy.allowed_tools
            )
        elif role == AgentRole.TEST:
            is_conforming = (
                not policy.is_writer
                and "test_runner" in policy.allowed_tools
                and "edit_file" not in policy.allowed_tools
                and "write_file" not in policy.allowed_tools
                and "run_command" not in policy.allowed_tools
            )
        elif role == AgentRole.REVIEWER:
            is_conforming = (
                not policy.is_writer
                and "code_review" in policy.allowed_tools
                and "edit_file" not in policy.allowed_tools
                and "write_file" not in policy.allowed_tools
            )
        elif role == AgentRole.CODING:
            is_conforming = (
                policy.is_writer
                and "edit_file" in policy.allowed_tools
                and "write_file" in policy.allowed_tools
            )
        if is_conforming:
            conforming_roles += 1

    passed = conforming_roles == total_roles
    rec_conformance = MetricRecord(
        numerator=conforming_roles, denominator=total_roles, rate=conforming_roles / total_roles
    )

    return CaseResult(
        case_id="case_4",
        name="Role Tool Conformance Audit",
        passed=passed,
        metrics={"conforming_roles": conforming_roles, "total_roles": total_roles},
        metric_records={"role_tool_conformance_rate": rec_conformance},
        details=f"Audited all role policies: {conforming_roles}/{total_roles} roles strictly conform.",
    )


def run_case_5(tmp_dir: Path) -> CaseResult:
    """CASE 5: Parallel Read Sibling Concurrency & Speedup."""
    planner = TeamPlanner()
    custom_tasks = [
        {"id": "r1", "name": "R1", "role": "research", "dependencies": []},
        {"id": "r2", "name": "R2", "role": "research", "dependencies": []},
    ]
    plan = planner.plan(goal="Parallel research benchmark", custom_tasks=custom_tasks)

    # Sequential baseline
    t0_seq = time.time()
    time.sleep(0.06)
    time.sleep(0.06)
    seq_time = time.time() - t0_seq

    # Parallel execution
    scheduler = TeamScheduler(max_workers=2)
    context = ToolContext(cwd=str(tmp_dir))

    def mock_agent(config):
        time.sleep(0.06)
        return SubAgentResult(ok=True, output=f"Done {config.name}", final_message="Done")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_agent):
        t0_par = time.time()
        res = scheduler.schedule_and_run(plan, context)
        par_time = time.time() - t0_par

    speedup = seq_time / max(0.001, par_time)
    active = res.success and scheduler.max_concurrent_readers_observed >= 2 and speedup >= 1.2

    rec_parallel = MetricRecord(
        numerator=1 if active else 0,
        denominator=1,
        rate=1.0 if active else 0.0,
    )

    return CaseResult(
        case_id="case_5",
        name="Parallel Read Concurrency",
        passed=active,
        metrics={
            "max_concurrent_readers": scheduler.max_concurrent_readers_observed,
            "speedup_factor": round(speedup, 2),
        },
        metric_records={"reader_parallelism_active": rec_parallel},
        details=f"Max concurrent readers: {scheduler.max_concurrent_readers_observed}, speedup: {speedup:.2f}x.",
    )


def run_case_6(tmp_dir: Path) -> CaseResult:
    """CASE 6: Shared Workspace Writer Serialization & Zero Overlap."""
    planner = TeamPlanner()
    custom_tasks = [
        {"id": "w1", "name": "Writer 1", "role": "coding", "dependencies": []},
        {"id": "w2", "name": "Writer 2", "role": "coding", "dependencies": []},
        {"id": "w3", "name": "Writer 3", "role": "coding", "dependencies": []},
    ]
    plan = planner.plan(goal="Writer concurrency test", custom_tasks=custom_tasks)

    scheduler = TeamScheduler(max_workers=3)
    context = ToolContext(cwd=str(tmp_dir))

    def mock_writer(config):
        time.sleep(0.02)
        return SubAgentResult(ok=True, output="Write done", final_message="Done")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_writer):
        res = scheduler.schedule_and_run(plan, context)

    max_writers = scheduler.max_concurrent_writers_observed
    no_overlap = not scheduler.reader_writer_overlap_observed
    serialized = res.success and max_writers <= 1

    rec_ser = MetricRecord(numerator=1 if serialized else 0, denominator=1, rate=1.0 if serialized else 0.0)
    rec_overlap = MetricRecord(numerator=1 if no_overlap else 0, denominator=1, rate=1.0 if no_overlap else 0.0)

    return CaseResult(
        case_id="case_6",
        name="Writer Serialization & Zero Overlap",
        passed=serialized and no_overlap,
        metrics={
            "max_concurrent_writers": max_writers,
            "reader_writer_overlap_observed": scheduler.reader_writer_overlap_observed,
        },
        metric_records={
            "writer_serialization_enforcement": rec_ser,
            "concurrent_writer_zero_overlap": rec_overlap,
        },
        details=f"Max concurrent writers: {max_writers}, overlap observed: {scheduler.reader_writer_overlap_observed}.",
    )


def run_case_7(tmp_dir: Path) -> CaseResult:
    """CASE 7: TestGate Real Evidence & Adversarial Matrix."""
    # Real evidence checks
    real_checks_passed = 0
    total_real = 3

    pass_ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="All 10 passed")
    fail_ev = SubAgentToolEvent(tool_name="test_runner", ok=False, output_summary="1 failed")

    res_pass = SubAgentResult(ok=True, output="Tests ran", tool_events=[pass_ev])
    if TestGate.evaluate(res_pass).passed:
        real_checks_passed += 1

    res_fail = SubAgentResult(ok=True, output="Tests ran", tool_events=[fail_ev])
    if not TestGate.evaluate(res_fail).passed:
        real_checks_passed += 1

    res_seq = SubAgentResult(ok=True, output="Retry pass", tool_events=[fail_ev, pass_ev])
    if TestGate.evaluate(res_seq).passed:
        real_checks_passed += 1

    # Adversarial checks
    adv_checks_passed = 0
    total_adv = 4

    # 1. Hollow text claim
    res_adv1 = SubAgentResult(ok=True, output="10 passed in 0.5s ok=true", tool_events=[])
    if not TestGate.evaluate(res_adv1).passed:
        adv_checks_passed += 1

    # 2. Hollow structured JSON
    res_adv2 = SubAgentResult(ok=True, output="Done", structured_data={"passed": True}, tool_events=[])
    if not TestGate.evaluate(res_adv2).passed:
        adv_checks_passed += 1

    # 3. Non-test tool event (read_file ok=True)
    read_ev = SubAgentToolEvent(tool_name="read_file", ok=True, output_summary="code")
    res_adv3 = SubAgentResult(ok=True, output="Read", tool_events=[read_ev])
    if not TestGate.evaluate(res_adv3).passed:
        adv_checks_passed += 1

    # 4. Non-test tool event (code_review ok=True)
    rev_ev = SubAgentToolEvent(tool_name="code_review", ok=True, output_summary="approved")
    res_adv4 = SubAgentResult(ok=True, output="Review", tool_events=[rev_ev])
    if not TestGate.evaluate(res_adv4).passed:
        adv_checks_passed += 1

    passed = (real_checks_passed == total_real) and (adv_checks_passed == total_adv)

    rec_real = MetricRecord(
        numerator=real_checks_passed, denominator=total_real, rate=real_checks_passed / total_real
    )
    rec_adv = MetricRecord(
        numerator=adv_checks_passed, denominator=total_adv, rate=adv_checks_passed / total_adv
    )

    return CaseResult(
        case_id="case_7",
        name="TestGate Evidence & Adversarial Matrix",
        passed=passed,
        metrics={
            "real_checks_passed": real_checks_passed,
            "total_real": total_real,
            "adv_checks_passed": adv_checks_passed,
            "total_adv": total_adv,
        },
        metric_records={
            "test_gate_real_evidence_accuracy": rec_real,
            "test_gate_adversarial_rejection_rate": rec_adv,
        },
        details=f"Real evidence checks: {real_checks_passed}/{total_real}; Adversarial rejections: {adv_checks_passed}/{total_adv}.",
    )


def run_case_8(tmp_dir: Path) -> CaseResult:
    """CASE 8: ReviewGate Structured JSON Verification."""
    checks_passed = 0
    total_checks = 4

    # 1. Valid approve JSON
    res1 = SubAgentResult(
        ok=True,
        output='```json\n{"verdict": "approve", "comments": "Clean"}\n```',
        structured_data={"verdict": "approve", "comments": "Clean"},
    )
    if ReviewGate.evaluate(res1).passed:
        checks_passed += 1

    # 2. Valid reject JSON
    res2 = SubAgentResult(
        ok=True,
        output='```json\n{"verdict": "reject", "comments": "Defect"}\n```',
        structured_data={"verdict": "reject", "comments": "Defect"},
    )
    g2 = ReviewGate.evaluate(res2)
    if (not g2.passed) and g2.verdict == "rejected":
        checks_passed += 1

    # 3. Missing JSON
    res3 = SubAgentResult(ok=True, output="I approve the code in plain text.")
    g3 = ReviewGate.evaluate(res3)
    if (not g3.passed) and g3.verdict == "rejected":
        checks_passed += 1

    # 4. Subagent failed
    res4 = SubAgentResult(ok=False, output="Crashed", error="Timeout")
    if not ReviewGate.evaluate(res4).passed:
        checks_passed += 1

    passed = checks_passed == total_checks
    rec_rev = MetricRecord(
        numerator=checks_passed, denominator=total_checks, rate=checks_passed / total_checks
    )

    return CaseResult(
        case_id="case_8",
        name="ReviewGate Structured Verification",
        passed=passed,
        metrics={"checks_passed": checks_passed, "total_checks": total_checks},
        metric_records={"review_gate_structured_accuracy": rec_rev},
        details=f"Structured review gate parsing accuracy: {checks_passed}/{total_checks}.",
    )


def run_case_9(tmp_dir: Path) -> CaseResult:
    """CASE 9: Review Rejection and Corrective Replan Flow."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Refactor auth handler")

    scheduler = TeamScheduler(max_workers=4)
    context = ToolContext(cwd=str(tmp_dir))

    review_turns = 0

    def mock_agent(config):
        nonlocal review_turns
        if "research" in config.name:
            return SubAgentResult(ok=True, output="Research done")
        if "coding" in config.name:
            return SubAgentResult(ok=True, output="Coding done")
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="Tests passed")
            return SubAgentResult(ok=True, output="tests passed", tool_events=[ev])
        if "reviewer" in config.name:
            review_turns += 1
            if review_turns == 1:
                return SubAgentResult(
                    ok=True,
                    output='{"verdict": "reject", "comments": "Missing salt"}',
                    structured_data={"verdict": "reject", "comments": "Missing salt"},
                )
            else:
                return SubAgentResult(
                    ok=True,
                    output='{"verdict": "approve", "comments": "Salt added"}',
                    structured_data={"verdict": "approve", "comments": "Salt added"},
                )
        return SubAgentResult(ok=True, output="Done")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_agent):
        res = scheduler.schedule_and_run(plan, context, max_replans=1)

    passed = res.success and res.replan_count == 1 and "reviewer_replan_1" in res.completed_tasks
    rec_replan = MetricRecord(numerator=1 if passed else 0, denominator=1, rate=1.0 if passed else 0.0)

    return CaseResult(
        case_id="case_9",
        name="Review Rejection Corrective Replan",
        passed=passed,
        metrics={"replan_count": res.replan_count, "rejection_handled": passed},
        metric_records={"review_rejection_replan_rate": rec_replan},
        details="Review rejection initiated replan 1; reviewer approved on second attempt.",
    )


def run_case_10(tmp_dir: Path) -> CaseResult:
    """CASE 10: Test Failure Replan Flow & Reviewer Skipping."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Fix memory leak")

    scheduler = TeamScheduler(max_workers=4)
    context = ToolContext(cwd=str(tmp_dir))

    test_turns = 0

    def mock_agent(config):
        nonlocal test_turns
        if "research" in config.name:
            return SubAgentResult(ok=True, output="Research done")
        if "coding" in config.name:
            return SubAgentResult(ok=True, output="Coding done")
        if "test" in config.name:
            test_turns += 1
            if test_turns == 1:
                ev_f = SubAgentToolEvent(tool_name="test_runner", ok=False, output_summary="Leak detected")
                return SubAgentResult(ok=True, output="Failed", tool_events=[ev_f])
            else:
                ev_p = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="No leak")
                return SubAgentResult(ok=True, output="Passed", tool_events=[ev_p])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "approve", "comments": "Good fix"}',
                structured_data={"verdict": "approve", "comments": "Good fix"},
            )
        return SubAgentResult(ok=True, output="Done")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_agent):
        res = scheduler.schedule_and_run(plan, context, max_replans=1)

    # Original reviewer must have been skipped
    orig_reviewer_skipped = "reviewer" in res.skipped_tasks
    passed = res.success and res.replan_count == 1 and orig_reviewer_skipped and "test_replan_1" in res.completed_tasks

    rec_test_replan = MetricRecord(numerator=1 if passed else 0, denominator=1, rate=1.0 if passed else 0.0)

    return CaseResult(
        case_id="case_10",
        name="Test Failure Replan & Reviewer Skip",
        passed=passed,
        metrics={"replan_count": res.replan_count, "original_reviewer_skipped": orig_reviewer_skipped},
        metric_records={"test_failure_replan_rate": rec_test_replan},
        details="Test failure skipped original reviewer and triggered replan which completed successfully.",
    )


def run_case_11(tmp_dir: Path) -> CaseResult:
    """CASE 11: Replan Boundedness & Upstream Failure Containment."""
    # Part A: Replan Bounded Halt (persistent failure halts at max_replans)
    planner = TeamPlanner()
    plan_a = planner.plan(goal="Strict security validation")
    scheduler_a = TeamScheduler(max_workers=4)
    context_a = ToolContext(cwd=str(tmp_dir))

    def mock_persistent_reject(config):
        if "research" in config.name or "coding" in config.name:
            return SubAgentResult(ok=True, output="ok")
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="pass")
            return SubAgentResult(ok=True, output="ok", tool_events=[ev])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "reject", "comments": "Persistent security vulnerability"}',
                structured_data={"verdict": "reject", "comments": "Persistent security vulnerability"},
            )
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_persistent_reject):
        res_a = scheduler_a.schedule_and_run(plan_a, context_a, max_replans=1)

    bounded_halt = (not res_a.success) and res_a.replan_count == 1 and "reviewer_replan_1" in res_a.failed_tasks

    # Part B: Upstream Failure Containment (cascade skipping downstream tasks)
    plan_b = planner.plan(goal="Upstream crash containment")
    scheduler_b = TeamScheduler(max_workers=4)
    context_b = ToolContext(cwd=str(tmp_dir))

    def mock_upstream_fail(config):
        if "research_impl" in config.name:
            return SubAgentResult(ok=False, output="Repo read failed", error="ReadError")
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_upstream_fail):
        res_b = scheduler_b.schedule_and_run(plan_b, context_b)

    cascade_contained = (
        (not res_b.success)
        and "research_impl" in res_b.failed_tasks
        and "coding" in res_b.skipped_tasks
        and "test" in res_b.skipped_tasks
        and "reviewer" in res_b.skipped_tasks
    )

    passed = bounded_halt and cascade_contained

    rec_halt = MetricRecord(numerator=1 if bounded_halt else 0, denominator=1, rate=1.0 if bounded_halt else 0.0)
    rec_contain = MetricRecord(
        numerator=1 if cascade_contained else 0, denominator=1, rate=1.0 if cascade_contained else 0.0
    )

    return CaseResult(
        case_id="case_11",
        name="Replan Boundedness & Upstream Containment",
        passed=passed,
        metrics={"bounded_halt": bounded_halt, "cascade_contained": cascade_contained},
        metric_records={
            "replan_bounded_halt_rate": rec_halt,
            "upstream_failure_containment_rate": rec_contain,
        },
        details=f"Bounded halt at replan 1: {bounded_halt}; Cascade containment: {cascade_contained}.",
    )


def run_case_12(tmp_dir: Path) -> CaseResult:
    """CASE 12: Worktree Lifecycle, Dirty Rejection & Fail-Closed Protection."""
    repo_dir = tmp_dir / "eval_git_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo_dir), check=True, capture_output=True)

    init_file = repo_dir / "index.py"
    init_file.write_text("print('v1')\n", encoding="utf-8")
    subprocess.run(["git", "add", "index.py"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_dir), check=True, capture_output=True)

    isolator = WorktreeIsolator(base_path=repo_dir)

    # 1. Dirty Parent Rejection Check
    dirty_file = repo_dir / "uncommitted.txt"
    dirty_file.write_text("dirty\n", encoding="utf-8")
    is_clean, dirty_msg = isolator.check_parent_workspace_clean()
    dirty_rejected = (not is_clean) and "dirty" in dirty_msg.lower()

    # Clean up dirty file
    dirty_file.unlink()
    is_clean_now, _ = isolator.check_parent_workspace_clean()
    assert is_clean_now

    # 2. Detached Worktree Lifecycle Check
    wt = isolator.create_worktree("eval_detached")
    assert wt is not None and wt.exists()

    # Verify detached HEAD
    stat_res = subprocess.run(["git", "status"], cwd=str(wt), capture_output=True, text=True, check=True)
    detached = "HEAD detached" in stat_res.stdout or "Not currently on any branch" in stat_res.stdout

    # Modify in worktree
    (wt / "index.py").write_text("print('v2_enhanced')\n", encoding="utf-8")
    patch_str = isolator.generate_patch(wt)
    assert "index.py" in patch_str

    # 3. Parent Fingerprint Protection Check
    # Simulate parent modified externally during execution
    init_file.write_text("print('externally_corrupted')\n", encoding="utf-8")
    fp_ok, fp_msg = isolator.verify_parent_fingerprint()
    fingerprint_protected = (not fp_ok) and "changed" in fp_msg.lower()

    # Restore parent to match initial commit
    init_file.write_text("print('v1')\n", encoding="utf-8")
    fp_restored, _ = isolator.verify_parent_fingerprint()
    assert fp_restored

    # 4. Patch Verify Fail-Closed Check
    bad_patch = (
        "diff --git a/index.py b/index.py\n"
        "--- a/index.py\n"
        "+++ b/index.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-nonexistent conflicting line\n"
        "+bad replacement\n"
    )
    verify_failed = not isolator.verify_patch(bad_patch)

    # Clean up worktree
    isolator.cleanup_all()
    cleaned_up = (not wt.exists()) and len(isolator.active_worktrees) == 0

    passed = dirty_rejected and detached and fingerprint_protected and verify_failed and cleaned_up

    rec_dirty = MetricRecord(
        numerator=1 if dirty_rejected else 0, denominator=1, rate=1.0 if dirty_rejected else 0.0
    )
    rec_fp = MetricRecord(
        numerator=1 if fingerprint_protected else 0, denominator=1, rate=1.0 if fingerprint_protected else 0.0
    )
    rec_verify = MetricRecord(
        numerator=1 if verify_failed else 0, denominator=1, rate=1.0 if verify_failed else 0.0
    )

    return CaseResult(
        case_id="case_12",
        name="Worktree Dirty, Fingerprint & Verify Gates",
        passed=passed,
        metrics={
            "dirty_rejected": dirty_rejected,
            "detached_head": detached,
            "fingerprint_protected": fingerprint_protected,
            "patch_verify_failed_closed": verify_failed,
            "cleaned_up": cleaned_up,
        },
        metric_records={
            "worktree_parent_dirty_rejection_rate": rec_dirty,
            "worktree_fingerprint_protection_rate": rec_fp,
            "worktree_patch_verify_fail_closed_rate": rec_verify,
        },
        details=f"Dirty rejection: {dirty_rejected}; Detached: {detached}; Fingerprint protection: {fingerprint_protected}; Verify fail-closed: {verify_failed}.",
    )


# ============================================================================
# Main Evaluation Runner
# ============================================================================

def run_all_evaluations() -> dict[str, Any]:
    print("=" * 65)
    print("Running Phase 4.1 Multi-Agent Orchestration Benchmark (12 Cases)")
    print("=" * 65)

    with tempfile.TemporaryDirectory(prefix="minicode_multi_eval_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        cases = [
            run_case_1(tmp_dir),
            run_case_2(tmp_dir),
            run_case_3(tmp_dir),
            run_case_4(tmp_dir),
            run_case_5(tmp_dir),
            run_case_6(tmp_dir),
            run_case_7(tmp_dir),
            run_case_8(tmp_dir),
            run_case_9(tmp_dir),
            run_case_10(tmp_dir),
            run_case_11(tmp_dir),
            run_case_12(tmp_dir),
        ]

    for c in cases:
        status = "[PASS]" if c.passed else "[FAIL]"
        print(f"  {status} {c.case_id.upper()}: {c.name}")

    # Consolidate all 17 Quantitative Metric Records
    metric_records: dict[str, MetricRecord] = {}
    for c in cases:
        for mname, mrec in c.metric_records.items():
            metric_records[mname] = mrec

    all_passed = all(c.passed for c in cases)

    results = {
        "benchmark": "Phase 4.1 Centralized Multi-Agent Orchestration",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "all_cases_passed": all_passed,
        "total_cases": len(cases),
        "passed_cases": sum(1 for c in cases if c.passed),
        "metrics": {k: v.to_dict() for k, v in metric_records.items()},
        "cases": [
            {
                "case_id": c.case_id,
                "name": c.name,
                "passed": c.passed,
                "metrics": c.metrics,
                "details": c.details,
            }
            for c in cases
        ],
    }

    # Save JSON and Markdown artifacts
    bench_dir = Path("benchmarks")
    bench_dir.mkdir(exist_ok=True)

    json_path = bench_dir / "multi_agent_eval_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Phase 4.1 Multi-Agent Orchestration Benchmark Results",
        "",
        f"- **Timestamp**: {results['timestamp']}",
        f"- **Overall Status**: {'PASS' if all_passed else 'FAIL'} ({results['passed_cases']}/{results['total_cases']})",
        "",
        "## 17 Quantitative Metrics (Computed Dynamically)",
        "",
        "| Metric | Target | Measured Ratio | Rate | Status |",
        "|---|---|---|---|---|",
    ]

    for mname, mrec in sorted(metric_records.items()):
        ratio_str = f"{mrec.numerator} / {mrec.denominator}"
        pct_str = f"{mrec.rate * 100:.1f}%"
        status_str = "PASS" if mrec.rate >= 1.0 else ("FAIL" if mrec.denominator > 0 else "N/A")
        md_lines.append(f"| {mname} | 100% | {ratio_str} | {pct_str} | {status_str} |")

    md_lines.extend(
        [
            "",
            "## Evaluated Test Cases",
            "",
            "| Case ID | Name | Status | Details |",
            "|---|---|---|---|",
        ]
    )
    for c in cases:
        status_badge = "PASS" if c.passed else "FAIL"
        md_lines.append(f"| {c.case_id} | {c.name} | {status_badge} | {c.details} |")

    md_path = bench_dir / "multi_agent_eval_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nArtifacts generated:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    print("=" * 65)
    return results


if __name__ == "__main__":
    res = run_all_evaluations()
    if not res["all_cases_passed"]:
        sys.exit(1)
