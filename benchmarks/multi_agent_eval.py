"""Deterministic Benchmark and Evaluation Suite for Phase 4.2 Centralized Multi-Agent Orchestration.

Evaluates 12 deterministic test cases covering:
CASE 1: Parent Agent Turn & Context Isolation (Real run_agent_turn + Tool Dispatch)
CASE 2: Multi-Agent DAG Planning, Topology & Dependency Satisfaction
CASE 3: Multi-Agent Plan Validation Engine
CASE 4: Role Tool Conformance Audit
CASE 5: Nested Agent Exposure & MCP Server Isolation
CASE 6: Read-Only Sibling Concurrency & Speedup
CASE 7: Shared Workspace Writer Serialization & Zero Overlap
CASE 8: TestGate Real Evidence & Adversarial Matrix
CASE 9: ReviewGate Structured Verification & Replan Boundedness
CASE 10: Child Failure Containment & Upstream Cascade Skip
CASE 11: Worktree Pre-Execution & Race Protection
CASE 12: Permission-Gated Writeback, Leakage Prevention & Verified Patch Apply
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
    target: float = 1.0
    comparison: str = "gte"  # "gte" (higher is better) or "lte" (lower is better)

    @property
    def passed(self) -> bool:
        if self.comparison == "lte":
            return self.rate <= self.target
        return self.rate >= self.target

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": round(self.rate, 4),
            "target": self.target,
            "comparison": self.comparison,
            "passed": self.passed,
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
    registry = ToolRegistry([agent_team_tool])

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
        "Child internal step analysis: examining tokens and sessions. " * 50,
        "Child internal step analysis: inspecting test runners and suites. " * 50,
        "Child internal step analysis: editing crypto hashing and salt logic. " * 50,
        "Child internal step analysis: running unit test verification passes. " * 50,
        "Child internal step analysis: performing structured code review. " * 50,
    ]
    total_child_bytes = sum(len(c.encode("utf-8")) for c in child_log_chunks)  # > 15,000 bytes

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
    tool_result_bytes = len(result_content.encode("utf-8"))
    tool_result_bounded = tool_result_bytes <= 8000

    raw_history_leak_count = 0
    for m in final_messages:
        role = m.get("role")
        if role == "assistant_tool_call" and m.get("toolName") != "agent_team":
            raw_history_leak_count += 1
        elif role == "tool_result" and m.get("toolName") != "agent_team":
            raw_history_leak_count += 1
        elif "Child internal step analysis:" in str(m.get("content", "")):
            raw_history_leak_count += 1

    isolated = (total_child_bytes >= 10000) and tool_result_bounded and (raw_history_leak_count == 0)
    passed = dispatched and isolated

    rec_iso = MetricRecord(
        numerator=1 if isolated else 0,
        denominator=1,
        rate=1.0 if isolated else 0.0,
        target=1.0,
        comparison="gte",
    )

    return CaseResult(
        case_id="case_1",
        name="Parent Agent Turn & Context Isolation",
        passed=passed,
        metrics={
            "raw_child_context_bytes": total_child_bytes,
            "parent_tool_result_bytes": tool_result_bytes,
            "raw_history_leak_count": raw_history_leak_count,
            "dispatched": dispatched,
            "isolated": isolated,
        },
        metric_records={"parent_context_isolation_pass_rate": rec_iso},
        details=(
            f"Child context: {total_child_bytes} bytes; Parent tool result: {tool_result_bytes} bytes; "
            f"Raw history leaks: {raw_history_leak_count}."
        ),
    )


def run_case_2(tmp_dir: Path) -> CaseResult:
    """CASE 2: Multi-Agent DAG Planning, Topology & Dependency Satisfaction."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Add REST API endpoints")

    defs = plan.graph.definitions
    has_nodes = all(
        k in defs for k in ["research_impl", "research_test", "coding", "test", "reviewer"]
    )

    dep_checks = [
        defs["research_impl"].dependencies == [],
        defs["research_test"].dependencies == [],
        set(defs["coding"].dependencies) == {"research_impl", "research_test"},
        defs["test"].dependencies == ["coding"],
        defs["reviewer"].dependencies == ["test"],
    ]
    satisfied_deps = sum(1 for c in dep_checks if c)
    total_deps = len(dep_checks)

    passed = has_nodes and (satisfied_deps == total_deps)
    rec_dep = MetricRecord(
        numerator=satisfied_deps,
        denominator=total_deps,
        rate=satisfied_deps / total_deps,
        target=1.0,
        comparison="gte",
    )

    return CaseResult(
        case_id="case_2",
        name="DAG Planning & Dependency Satisfaction",
        passed=passed,
        metrics={
            "node_count": len(defs),
            "satisfied_deps": satisfied_deps,
            "total_deps": total_deps,
        },
        metric_records={"dependency_satisfaction_rate": rec_dep},
        details=f"Dependencies satisfied: {satisfied_deps}/{total_deps} across canonical 5-node DAG.",
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
        numerator=checks_passed,
        denominator=total_checks,
        rate=checks_passed / total_checks,
        target=1.0,
        comparison="gte",
    )

    return CaseResult(
        case_id="case_3",
        name="Plan Validation Engine",
        passed=passed,
        metrics={"checks_passed": checks_passed, "total_checks": total_checks},
        metric_records={"plan_validity_rate": rec_plan_valid},
        details=f"Plan validation checks: {checks_passed}/{total_checks} passed.",
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
        numerator=conforming_roles,
        denominator=total_roles,
        rate=conforming_roles / total_roles,
        target=1.0,
        comparison="gte",
    )

    return CaseResult(
        case_id="case_4",
        name="Role Tool Conformance Audit",
        passed=passed,
        metrics={"conforming_roles": conforming_roles, "total_roles": total_roles},
        metric_records={"role_tool_conformance_rate": rec_conformance},
        details=f"Audited all role policies: {conforming_roles}/{total_roles} roles conform.",
    )


def run_case_5(tmp_dir: Path) -> CaseResult:
    """CASE 5: Nested Agent Exposure & MCP Server Isolation."""
    parent_runtime = {
        "model": "test-model",
        "mcpServers": {
            "dangerous_mcp": {"command": "external_exec", "args": ["--root"]}
        },
    }

    inspected_registries = 0
    nested_tool_exposures = 0
    mcp_leakages = 0

    # Test all 4 roles
    for role in AgentRole:
        inspected_registries += 1
        policy = get_role_policy(role)
        config = SubAgentRunConfig(
            name=f"Inspect_{role.value}",
            task_prompt="Audit environment",
            system_prompt="System",
            cwd=str(tmp_dir),
            runtime=parent_runtime,
            allowed_tools=policy.allowed_tools,
            is_writer=policy.is_writer,
            depth=0,
        )

        with patch("minicode.tools.create_default_tool_registry") as mock_tools, \
             patch("minicode.subagent_runner.create_model_adapter") as mock_adapter, \
             patch("minicode.subagent_runner.run_agent_turn") as mock_turn:

            mock_reg = MagicMock()
            mock_reg.list_all.return_value = list(policy.allowed_tools)
            mock_tools.return_value = mock_reg
            mock_turn.return_value = [{"role": "assistant", "content": "Audited <final>"}]

            run_subagent(config)

            # Check MCP leakage
            called_runtime = mock_tools.call_args[1].get("runtime", {})
            if called_runtime.get("mcpServers") != {}:
                mcp_leakages += 1

            # Check Nested Tool Exposure
            child_tools_registry = mock_adapter.call_args[1]["tools"]
            exposed = child_tools_registry.list_all()
            if "task" in exposed or "agent_team" in exposed:
                nested_tool_exposures += 1

    rec_nested = MetricRecord(
        numerator=nested_tool_exposures,
        denominator=inspected_registries,
        rate=nested_tool_exposures / inspected_registries if inspected_registries > 0 else 0.0,
        target=0.0,
        comparison="lte",
    )
    rec_mcp = MetricRecord(
        numerator=mcp_leakages,
        denominator=inspected_registries,
        rate=mcp_leakages / inspected_registries if inspected_registries > 0 else 0.0,
        target=0.0,
        comparison="lte",
    )

    passed = (nested_tool_exposures == 0) and (mcp_leakages == 0)
    return CaseResult(
        case_id="case_5",
        name="Nested Agent Exposure & MCP Isolation",
        passed=passed,
        metrics={
            "nested_tool_exposures": nested_tool_exposures,
            "mcp_leakages": mcp_leakages,
            "inspected_registries": inspected_registries,
        },
        metric_records={
            "nested_agent_exposure_rate": rec_nested,
            "mcp_leakage_rate": rec_mcp,
        },
        details=f"Nested tool exposures: {nested_tool_exposures}/{inspected_registries}; MCP leakages: {mcp_leakages}/{inspected_registries}.",
    )


def run_case_6(tmp_dir: Path) -> CaseResult:
    """CASE 6: Read-Only Sibling Concurrency & Speedup."""
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
    active = res.success and (scheduler.max_concurrent_readers_observed >= 2) and (speedup >= 1.2)

    rec_parallel = MetricRecord(
        numerator=1 if active else 0,
        denominator=1,
        rate=1.0 if active else 0.0,
        target=1.0,
        comparison="gte",
    )

    return CaseResult(
        case_id="case_6",
        name="Read-Only Parallelization",
        passed=active,
        metrics={
            "max_concurrent_readers": scheduler.max_concurrent_readers_observed,
            "speedup_factor": round(speedup, 2),
        },
        metric_records={"read_only_parallelization_rate": rec_parallel},
        details=f"Max concurrent readers: {scheduler.max_concurrent_readers_observed}, speedup: {speedup:.2f}x.",
    )


def run_case_7(tmp_dir: Path) -> CaseResult:
    """CASE 7: Shared Workspace Writer Serialization & Zero Overlap."""
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
    overlap_violations = 1 if scheduler.reader_writer_overlap_observed else 0
    serialized = res.success and (max_writers <= 1)

    rec_ser = MetricRecord(
        numerator=1 if serialized else 0,
        denominator=1,
        rate=1.0 if serialized else 0.0,
        target=1.0,
        comparison="gte",
    )
    rec_overlap = MetricRecord(
        numerator=overlap_violations,
        denominator=1,
        rate=float(overlap_violations),
        target=0.0,
        comparison="lte",
    )

    passed = serialized and (overlap_violations == 0)
    return CaseResult(
        case_id="case_7",
        name="Writer Serialization & Zero Overlap",
        passed=passed,
        metrics={
            "max_concurrent_writers": max_writers,
            "overlap_violations": overlap_violations,
        },
        metric_records={
            "writer_serialization_rate": rec_ser,
            "writer_reader_overlap_violation_rate": rec_overlap,
        },
        details=f"Max concurrent writers: {max_writers}, overlap violations: {overlap_violations}.",
    )


def run_case_8(tmp_dir: Path) -> CaseResult:
    """CASE 8: TestGate Real Evidence & Adversarial Matrix."""
    checks_passed = 0
    total_checks = 7

    pass_ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="All 10 passed")
    fail_ev = SubAgentToolEvent(tool_name="test_runner", ok=False, output_summary="1 failed")

    # Real evidence checks (3 checks)
    res_pass = SubAgentResult(ok=True, output="Tests ran", tool_events=[pass_ev])
    if TestGate.evaluate(res_pass).passed:
        checks_passed += 1

    res_fail = SubAgentResult(ok=True, output="Tests ran", tool_events=[fail_ev])
    if not TestGate.evaluate(res_fail).passed:
        checks_passed += 1

    res_seq = SubAgentResult(ok=True, output="Retry pass", tool_events=[fail_ev, pass_ev])
    if TestGate.evaluate(res_seq).passed:
        checks_passed += 1

    # Adversarial checks (4 checks)
    res_adv1 = SubAgentResult(ok=True, output="10 passed in 0.5s ok=true", tool_events=[])
    if not TestGate.evaluate(res_adv1).passed:
        checks_passed += 1

    res_adv2 = SubAgentResult(ok=True, output="Done", structured_data={"passed": True}, tool_events=[])
    if not TestGate.evaluate(res_adv2).passed:
        checks_passed += 1

    read_ev = SubAgentToolEvent(tool_name="read_file", ok=True, output_summary="code")
    res_adv3 = SubAgentResult(ok=True, output="Read", tool_events=[read_ev])
    if not TestGate.evaluate(res_adv3).passed:
        checks_passed += 1

    rev_ev = SubAgentToolEvent(tool_name="code_review", ok=True, output_summary="approved")
    res_adv4 = SubAgentResult(ok=True, output="Review", tool_events=[rev_ev])
    if not TestGate.evaluate(res_adv4).passed:
        checks_passed += 1

    passed = checks_passed == total_checks
    rec_test = MetricRecord(
        numerator=checks_passed,
        denominator=total_checks,
        rate=checks_passed / total_checks,
        target=1.0,
        comparison="gte",
    )

    return CaseResult(
        case_id="case_8",
        name="TestGate Accuracy & Adversarial Matrix",
        passed=passed,
        metrics={"checks_passed": checks_passed, "total_checks": total_checks},
        metric_records={"test_gate_accuracy": rec_test},
        details=f"Test gate checks: {checks_passed}/{total_checks} passed.",
    )


def run_case_9(tmp_dir: Path) -> CaseResult:
    """CASE 9: ReviewGate Structured Verification & Replan Boundedness."""
    # Part 1: ReviewGate Accuracy (4 checks)
    review_checks = 0
    total_review_checks = 4

    res1 = SubAgentResult(
        ok=True,
        output='```json\n{"verdict": "approve", "comments": "Clean"}\n```',
        structured_data={"verdict": "approve", "comments": "Clean"},
    )
    if ReviewGate.evaluate(res1).passed:
        review_checks += 1

    res2 = SubAgentResult(
        ok=True,
        output='```json\n{"verdict": "reject", "comments": "Defect"}\n```',
        structured_data={"verdict": "reject", "comments": "Defect"},
    )
    if not ReviewGate.evaluate(res2).passed:
        review_checks += 1

    res3 = SubAgentResult(ok=True, output="I approve in plain text.")
    if not ReviewGate.evaluate(res3).passed:
        review_checks += 1

    res4 = SubAgentResult(ok=False, output="Crash", error="Error")
    if not ReviewGate.evaluate(res4).passed:
        review_checks += 1

    rec_rev = MetricRecord(
        numerator=review_checks,
        denominator=total_review_checks,
        rate=review_checks / total_review_checks,
        target=1.0,
        comparison="gte",
    )

    # Part 2: Replan Boundedness Flow
    planner = TeamPlanner()
    plan = planner.plan(goal="Refactor auth")
    scheduler = TeamScheduler(max_workers=4)
    context = ToolContext(cwd=str(tmp_dir))

    review_turns = 0

    def mock_agent(config):
        nonlocal review_turns
        if "research" in config.name or "coding" in config.name:
            return SubAgentResult(ok=True, output="ok")
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="pass")
            return SubAgentResult(ok=True, output="pass", tool_events=[ev])
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
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_agent):
        res = scheduler.schedule_and_run(plan, context, max_replans=1)

    replan_bounded = res.success and (res.replan_count == 1) and ("reviewer_replan_1" in res.completed_tasks)
    rec_replan = MetricRecord(
        numerator=1 if replan_bounded else 0,
        denominator=1,
        rate=1.0 if replan_bounded else 0.0,
        target=1.0,
        comparison="gte",
    )

    passed = (review_checks == total_review_checks) and replan_bounded
    return CaseResult(
        case_id="case_9",
        name="ReviewGate Accuracy & Replan Boundedness",
        passed=passed,
        metrics={
            "review_checks": review_checks,
            "total_review_checks": total_review_checks,
            "replan_bounded": replan_bounded,
        },
        metric_records={
            "review_gate_accuracy": rec_rev,
            "replan_boundedness_rate": rec_replan,
        },
        details=f"Review checks: {review_checks}/{total_review_checks}; Replan bounded to 1: {replan_bounded}.",
    )


def run_case_10(tmp_dir: Path) -> CaseResult:
    """CASE 10: Child Failure Containment & Upstream Cascade Skip."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Upstream crash containment")
    scheduler = TeamScheduler(max_workers=4)
    context = ToolContext(cwd=str(tmp_dir))

    def mock_upstream_fail(config):
        if "research_impl" in config.name:
            return SubAgentResult(ok=False, output="Repo read failed", error="ReadError")
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_upstream_fail):
        res = scheduler.schedule_and_run(plan, context)

    cascade_contained = (
        (not res.success)
        and "research_impl" in res.failed_tasks
        and "coding" in res.skipped_tasks
        and "test" in res.skipped_tasks
        and "reviewer" in res.skipped_tasks
    )

    rec_contain = MetricRecord(
        numerator=1 if cascade_contained else 0,
        denominator=1,
        rate=1.0 if cascade_contained else 0.0,
        target=1.0,
        comparison="gte",
    )

    return CaseResult(
        case_id="case_10",
        name="Child Failure Containment",
        passed=cascade_contained,
        metrics={"cascade_contained": cascade_contained},
        metric_records={"child_failure_containment_rate": rec_contain},
        details=f"Upstream failure cascade skipped dependents cleanly: {cascade_contained}.",
    )


def run_case_11(tmp_dir: Path) -> CaseResult:
    """CASE 11: Worktree Pre-Execution & Race Protection."""
    repo_dir = tmp_dir / "eval_git_repo_11"
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

    # 2. Detached Worktree Creation
    wt = isolator.create_worktree("eval_detached")
    assert wt is not None and wt.exists()

    # Modify in worktree
    (wt / "index.py").write_text("print('v2_enhanced')\n", encoding="utf-8")

    # 3. Parent Workspace Race Protection Check
    init_file.write_text("print('externally_corrupted')\n", encoding="utf-8")
    fp_ok, fp_msg = isolator.verify_parent_fingerprint()
    fingerprint_protected = (not fp_ok) and "changed" in fp_msg.lower()

    # Restore parent to clean commit state
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

    isolator.cleanup_all()

    passed = dirty_rejected and fingerprint_protected and verify_failed

    rec_dirty = MetricRecord(
        numerator=1 if dirty_rejected else 0,
        denominator=1,
        rate=1.0 if dirty_rejected else 0.0,
        target=1.0,
        comparison="gte",
    )
    rec_fp = MetricRecord(
        numerator=1 if fingerprint_protected else 0,
        denominator=1,
        rate=1.0 if fingerprint_protected else 0.0,
        target=1.0,
        comparison="gte",
    )
    rec_verify = MetricRecord(
        numerator=1 if verify_failed else 0,
        denominator=1,
        rate=1.0 if verify_failed else 0.0,
        target=1.0,
        comparison="gte",
    )

    return CaseResult(
        case_id="case_11",
        name="Worktree Pre-Execution & Race Protection",
        passed=passed,
        metrics={
            "dirty_rejected": dirty_rejected,
            "fingerprint_protected": fingerprint_protected,
            "verify_failed": verify_failed,
        },
        metric_records={
            "worktree_parent_dirty_rejection_rate": rec_dirty,
            "parent_workspace_race_protection_rate": rec_fp,
            "patch_verify_fail_closed_rate": rec_verify,
        },
        details=f"Dirty rejection: {dirty_rejected}; Race protection: {fingerprint_protected}; Verify fail-closed: {verify_failed}.",
    )


def run_case_12(tmp_dir: Path) -> CaseResult:
    """CASE 12: Permission-Gated Writeback, Leakage Prevention & Verified Patch Apply."""
    repo_dir = tmp_dir / "eval_git_repo_12"
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo_dir), check=True, capture_output=True)

    init_file = repo_dir / "main.py"
    init_file.write_text("print('hello baseline')\n", encoding="utf-8")
    subprocess.run(["git", "add", "main.py"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_dir), check=True, capture_output=True)

    planner = TeamPlanner()
    scheduler = TeamScheduler(max_workers=2)

    # 1. Check Permission Denial Protection
    denying_perms = PermissionManager(
        workspace_root=str(repo_dir),
        prompt=lambda req: {"decision": "deny_once"},
    )
    context_deny = ToolContext(cwd=str(repo_dir), permissions=denying_perms)

    def mock_agent_success(config):
        if "research" in config.name:
            return SubAgentResult(ok=True, output="ok")
        if "coding" in config.name:
            wt_cwd = Path(config.cwd)
            (wt_cwd / "main.py").write_text("print('attempted edit')\n", encoding="utf-8")
            return SubAgentResult(ok=True, output="ok", changed_files=["main.py"])
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="pass")
            return SubAgentResult(ok=True, output="pass", tool_events=[ev])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "approve", "comments": "ok"}',
                structured_data={"verdict": "approve", "comments": "ok"},
            )
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_agent_success):
        res_deny = scheduler.schedule_and_run(planner.plan("Deny test"), context_deny, use_worktree=True)

    perm_denial_protected = (
        (not res_deny.success)
        and (res_deny.status == "permission_denied")
        and (init_file.read_text(encoding="utf-8") == "print('hello baseline')\n")
    )

    # 2. Check Missing Permission Fail-Closed
    context_missing = ToolContext(cwd=str(repo_dir), permissions=None)
    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_agent_success):
        res_missing = scheduler.schedule_and_run(planner.plan("Missing test"), context_missing, use_worktree=True)

    missing_perm_protected = (
        (not res_missing.success)
        and (res_missing.status == "permission_manager_missing")
        and (init_file.read_text(encoding="utf-8") == "print('hello baseline')\n")
    )

    # 3. Check Failed Patch Leakage across 3 failure modes
    failed_worktree_runs = 3
    leaked_runs = 0

    # Mode A: Test Failure
    def mock_test_fail(config):
        if "coding" in config.name:
            (Path(config.cwd) / "main.py").write_text("print('test fail')\n", encoding="utf-8")
            return SubAgentResult(ok=True, output="ok", changed_files=["main.py"])
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=False, output_summary="fail")
            return SubAgentResult(ok=True, output="fail", tool_events=[ev])
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_test_fail):
        scheduler.schedule_and_run(planner.plan("Fail A"), context_deny, use_worktree=True, max_replans=0)
    if init_file.read_text(encoding="utf-8") != "print('hello baseline')\n":
        leaked_runs += 1

    # Mode B: Review Reject
    def mock_rev_reject(config):
        if "coding" in config.name:
            (Path(config.cwd) / "main.py").write_text("print('rev fail')\n", encoding="utf-8")
            return SubAgentResult(ok=True, output="ok", changed_files=["main.py"])
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="pass")
            return SubAgentResult(ok=True, output="pass", tool_events=[ev])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "reject", "comments": "bad"}',
                structured_data={"verdict": "reject", "comments": "bad"},
            )
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_rev_reject):
        scheduler.schedule_and_run(planner.plan("Fail B"), context_deny, use_worktree=True, max_replans=0)
    if init_file.read_text(encoding="utf-8") != "print('hello baseline')\n":
        leaked_runs += 1

    # Mode C: Perm Deny (already checked)
    if init_file.read_text(encoding="utf-8") != "print('hello baseline')\n":
        leaked_runs += 1

    # 4. Check Verified Patch Apply (Parent file actually changes)
    allowing_perms = PermissionManager(
        workspace_root=str(repo_dir),
        prompt=lambda req: {"decision": "allow_once"},
    )
    context_allow = ToolContext(cwd=str(repo_dir), permissions=allowing_perms)

    def mock_verified_success(config):
        if "research" in config.name:
            return SubAgentResult(ok=True, output="ok")
        if "coding" in config.name:
            wt_cwd = Path(config.cwd)
            (wt_cwd / "main.py").write_text("print('verified applied')\n", encoding="utf-8")
            return SubAgentResult(ok=True, output="ok", changed_files=["main.py"])
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="pass")
            return SubAgentResult(ok=True, output="pass", tool_events=[ev])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "approve", "comments": "clean"}',
                structured_data={"verdict": "approve", "comments": "clean"},
            )
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_verified_success):
        res_allow = scheduler.schedule_and_run(planner.plan("Allow test"), context_allow, use_worktree=True)

    parent_changed = init_file.read_text(encoding="utf-8") == "print('verified applied')\n"
    verified_patch_applied = res_allow.success and (res_allow.status == "completed") and parent_changed

    rec_perm_deny = MetricRecord(
        numerator=1 if perm_denial_protected else 0,
        denominator=1,
        rate=1.0 if perm_denial_protected else 0.0,
        target=1.0,
        comparison="gte",
    )
    rec_missing = MetricRecord(
        numerator=1 if missing_perm_protected else 0,
        denominator=1,
        rate=1.0 if missing_perm_protected else 0.0,
        target=1.0,
        comparison="gte",
    )
    rec_leakage = MetricRecord(
        numerator=leaked_runs,
        denominator=failed_worktree_runs,
        rate=leaked_runs / failed_worktree_runs if failed_worktree_runs > 0 else 0.0,
        target=0.0,
        comparison="lte",
    )
    rec_verified_apply = MetricRecord(
        numerator=1 if verified_patch_applied else 0,
        denominator=1,
        rate=1.0 if verified_patch_applied else 0.0,
        target=1.0,
        comparison="gte",
    )

    passed = perm_denial_protected and missing_perm_protected and (leaked_runs == 0) and verified_patch_applied

    return CaseResult(
        case_id="case_12",
        name="Permission Gates, Leakage & Verified Patch Apply",
        passed=passed,
        metrics={
            "perm_denial_protected": perm_denial_protected,
            "missing_perm_protected": missing_perm_protected,
            "failed_patch_leaks": leaked_runs,
            "verified_patch_applied": verified_patch_applied,
        },
        metric_records={
            "permission_denial_protection_rate": rec_perm_deny,
            "missing_permission_fail_closed_rate": rec_missing,
            "failed_patch_leakage_rate": rec_leakage,
            "verified_patch_apply_rate": rec_verified_apply,
        },
        details=(
            f"Perm denial protected: {perm_denial_protected}; Missing perm protected: {missing_perm_protected}; "
            f"Failed patch leaks: {leaked_runs}/{failed_worktree_runs}; Verified patch applied: {verified_patch_applied}."
        ),
    )


# ============================================================================
# Main Evaluation Runner
# ============================================================================

def run_all_evaluations() -> dict[str, Any]:
    print("=" * 65)
    print("Running Phase 4.2 Multi-Agent Orchestration Benchmark (12 Cases)")
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

    # Consolidate all 20 Quantitative Metric Records
    metric_records: dict[str, MetricRecord] = {}
    for c in cases:
        for mname, mrec in c.metric_records.items():
            metric_records[mname] = mrec

    all_passed = all(c.passed for c in cases) and all(m.passed for m in metric_records.values())

    results = {
        "benchmark": "Phase 4.2 Centralized Multi-Agent Orchestration",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "all_cases_passed": all_passed,
        "total_cases": len(cases),
        "passed_cases": sum(1 for c in cases if c.passed),
        "total_metrics": len(metric_records),
        "passed_metrics": sum(1 for m in metric_records.values() if m.passed),
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
        "# Phase 4.2 Multi-Agent Orchestration Benchmark Results",
        "",
        f"- **Timestamp**: {results['timestamp']}",
        f"- **Overall Status**: {'PASS' if all_passed else 'FAIL'} ({results['passed_cases']}/{results['total_cases']} cases, {results['passed_metrics']}/{results['total_metrics']} metrics)",
        "",
        "## 20 Quantitative Metrics (Dynamically Computed)",
        "",
        "| Metric | Target | Measured Ratio | Rate | Status |",
        "|---|---|---|---|---|",
    ]

    for mname, mrec in sorted(metric_records.items()):
        ratio_str = f"{mrec.numerator} / {mrec.denominator}"
        pct_str = f"{mrec.rate * 100:.1f}%"
        target_str = f"<= {mrec.target * 100:.0f}%" if mrec.comparison == "lte" else f">= {mrec.target * 100:.0f}%"
        status_str = "PASS" if mrec.passed else "FAIL"
        md_lines.append(f"| {mname} | {target_str} | {ratio_str} | {pct_str} | {status_str} |")

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
