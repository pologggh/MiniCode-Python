"""Centralized Multi-Agent Team Scheduler.

Coordinates parallel execution of sibling sub-agents using ThreadPoolExecutor,
enforces writer serialization, executes quality gates (Test Gate & Review Gate),
performs bounded replanning (max 1 replan), and safely contains failures.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from minicode.logging_config import get_logger
from minicode.subagent_runner import (
    SubAgentRunConfig,
    SubAgentResult,
    _extract_json_payload,
    run_subagent,
)
from minicode.task_graph import TaskDefinition, TaskGraph, TaskPriority, TaskSlot, TaskState
from minicode.team_planner import TeamPlan
from minicode.team_roles import AgentRole, AgentRolePolicy, get_role_policy
from minicode.tooling import ToolContext

logger = get_logger("team_scheduler")


@dataclass(slots=True)
class QualityGateResult:
    """Outcome of a quality gate evaluation."""
    passed: bool
    gate_name: str
    verdict: str
    feedback: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class TestGate:
    """Verifies that the Test sub-agent executed tests and verified pass status."""

    @staticmethod
    def evaluate(result: SubAgentResult) -> QualityGateResult:
        if not result.ok:
            return QualityGateResult(
                passed=False,
                gate_name="TestGate",
                verdict="failed",
                feedback=f"Test sub-agent execution failed: {result.error or result.output}",
            )

        # 1. Inspect structured data if available
        if result.structured_data:
            data = result.structured_data
            status = str(data.get("status", "")).lower()
            verdict = str(data.get("verdict", "")).lower()
            ok_val = data.get("ok")
            passed_val = data.get("passed")

            if passed_val is False or ok_val is False or verdict in ("fail", "failed", "reject") or status in ("failed", "fail"):
                return QualityGateResult(
                    passed=False,
                    gate_name="TestGate",
                    verdict="failed",
                    feedback=str(data.get("feedback") or data.get("comments") or "Test gate rejected by structured verdict"),
                    details=data,
                )
            if passed_val is True or ok_val is True or verdict in ("pass", "passed", "approve") or status in ("passed", "pass"):
                return QualityGateResult(
                    passed=True,
                    gate_name="TestGate",
                    verdict="passed",
                    feedback="Test gate passed via structured test result",
                    details=data,
                )

        combined_text = (result.final_message + "\n" + result.output).lower()

        # 2. Check for explicit failure markers
        failure_signals = [
            "test failed",
            "tests failed",
            "failures=",
            "errors=",
            "assertionerror",
            "fail:",
            "exit code 1",
            "status: failed",
            "tests: failed",
            "failing tests",
        ]
        has_failure = any(sig in combined_text for sig in failure_signals)

        # 3. Check for passing signals
        success_signals = [
            "all tests pass",
            "tests passed",
            "passed in",
            "100% passed",
            "ok=true",
            "status: passed",
            "test passed",
            "all passing",
        ]
        has_success = any(sig in combined_text for sig in success_signals)

        if has_failure and not has_success:
            return QualityGateResult(
                passed=False,
                gate_name="TestGate",
                verdict="failed",
                feedback="Test gate failed: test execution output indicates failures.",
            )

        if not has_success and result.tool_calls_count == 0:
            return QualityGateResult(
                passed=False,
                gate_name="TestGate",
                verdict="failed",
                feedback="Test gate failed: test subagent produced no test verification tool calls.",
            )

        return QualityGateResult(
            passed=True,
            gate_name="TestGate",
            verdict="passed",
            feedback="Test gate passed: test execution verified.",
        )


class ReviewGate:
    """Verifies that the Reviewer sub-agent approved changes with structured JSON."""

    @staticmethod
    def evaluate(result: SubAgentResult) -> QualityGateResult:
        if not result.ok:
            return QualityGateResult(
                passed=False,
                gate_name="ReviewGate",
                verdict="failed",
                feedback=f"Reviewer subagent failed: {result.error or result.output}",
            )

        data = result.structured_data
        if not data:
            data = _extract_json_payload(result.final_message or result.output)

        if not data or not isinstance(data, dict):
            return QualityGateResult(
                passed=False,
                gate_name="ReviewGate",
                verdict="rejected",
                feedback="Review gate rejected: reviewer did not provide required structured JSON verdict.",
            )

        verdict = str(data.get("verdict", "")).strip().lower()
        status = str(data.get("status", "")).strip().lower()
        comments = str(data.get("comments", ""))
        issues = data.get("issues", [])

        is_approved = verdict in ("approve", "approved", "pass", "passed") or status in ("approve", "approved", "pass", "passed")

        if is_approved:
            return QualityGateResult(
                passed=True,
                gate_name="ReviewGate",
                verdict="approved",
                feedback=comments or "Reviewer approved changes.",
                details=data,
            )

        return QualityGateResult(
            passed=False,
            gate_name="ReviewGate",
            verdict="rejected",
            feedback=comments or f"Reviewer rejected changes. Issues: {issues}",
            details=data,
        )


@dataclass(slots=True)
class TeamExecutionResult:
    """Consolidated outcome of an orchestrated team execution."""
    success: bool
    goal: str
    task_results: dict[str, SubAgentResult] = field(default_factory=dict)
    completed_tasks: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    skipped_tasks: list[str] = field(default_factory=list)
    replan_count: int = 0
    elapsed_seconds: float = 0.0
    summary: str = ""
    error: str | None = None
    gate_results: dict[str, QualityGateResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "goal": self.goal,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "skipped_tasks": self.skipped_tasks,
            "replan_count": self.replan_count,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "summary": self.summary,
            "error": self.error,
        }


class TeamScheduler:
    """Centralized orchestrator managing multi-agent task execution."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._writer_lock = threading.Lock()
        self.concurrent_writer_count = 0
        self.max_concurrent_writers_observed = 0

    def _cascade_skip_unreachable_tasks(self, graph: TaskGraph) -> list[str]:
        """Identify tasks whose dependencies have failed or skipped, and mark them skipped (transitive)."""
        all_newly_skipped: list[str] = []
        while True:
            failed_or_skipped = {
                slot.task_id
                for slot in graph.slots.values()
                if slot.state in (TaskState.FAILED, TaskState.SKIPPED)
            }
            round_skipped: list[str] = []
            for task_def in graph.definitions.values():
                slot_key = f"default:{task_def.id}"
                slot = graph.slots.get(slot_key)
                if not slot or slot.state != TaskState.PENDING:
                    continue

                if any(dep in failed_or_skipped for dep in task_def.dependencies):
                    graph.skip_task(slot_key, reason="Upstream dependency failed or was skipped")
                    round_skipped.append(task_def.id)

            if not round_skipped:
                break
            all_newly_skipped.extend(round_skipped)

        return all_newly_skipped

    def _trigger_replan(
        self,
        plan: TeamPlan,
        source_task_id: str,
        feedback: str,
        replan_index: int,
    ) -> None:
        """Dynamically append bounded corrective tasks (coding -> test -> reviewer) to the graph."""
        graph = plan.graph
        coding_id = f"coding_replan_{replan_index}"
        test_id = f"test_replan_{replan_index}"
        reviewer_id = f"reviewer_replan_{replan_index}"

        plan.task_roles[coding_id] = AgentRole.CODING
        plan.task_roles[test_id] = AgentRole.TEST
        plan.task_roles[reviewer_id] = AgentRole.REVIEWER

        def_coding = TaskDefinition(
            id=coding_id,
            name=f"Fix Rejection Issues (Replan {replan_index})",
            description=f"Implement fixes for quality gate issues: {feedback}",
            dependencies=[source_task_id],
            priority=TaskPriority.CRITICAL,
            metadata={"role": AgentRole.CODING.value, "replan": replan_index},
        )
        graph.definitions[coding_id] = def_coding
        graph.assign_slot(coding_id, slot_name="default")

        def_test = TaskDefinition(
            id=test_id,
            name=f"Verify Replan Fix (Replan {replan_index})",
            description=f"Execute tests to verify fix for: {feedback}",
            dependencies=[coding_id],
            priority=TaskPriority.CRITICAL,
            metadata={"role": AgentRole.TEST.value, "replan": replan_index},
        )
        graph.definitions[test_id] = def_test
        graph.assign_slot(test_id, slot_name="default")

        def_reviewer = TaskDefinition(
            id=reviewer_id,
            name=f"Re-review Fixed Solution (Replan {replan_index})",
            description=f"Inspect fixed code and tests; output structured JSON verdict",
            dependencies=[test_id],
            priority=TaskPriority.NORMAL,
            metadata={"role": AgentRole.REVIEWER.value, "replan": replan_index},
        )
        graph.definitions[reviewer_id] = def_reviewer
        graph.assign_slot(reviewer_id, slot_name="default")

    def _execute_single_task(
        self,
        task_def: TaskDefinition,
        plan: TeamPlan,
        context: ToolContext,
        dependency_outputs: dict[str, str],
    ) -> SubAgentResult:
        """Execute a single task slot within the sub-agent harness."""
        role = plan.get_role_for_task(task_def.id)
        policy = get_role_policy(role)

        prompt_parts = [
            f"Goal: {plan.goal}",
            f"Task: {task_def.name}",
            f"Description: {task_def.description}",
        ]
        if task_def.dependencies:
            prompt_parts.append("\n=== Context from Completed Dependencies ===")
            for dep_id in task_def.dependencies:
                dep_output = dependency_outputs.get(dep_id, "")
                if dep_output:
                    excerpt = dep_output[:2500]
                    prompt_parts.append(f"--- Output from {dep_id} ---\n{excerpt}")

        full_prompt = "\n\n".join(prompt_parts)

        runtime = getattr(context, "_runtime", None)
        parent_perms = getattr(context, "permissions", None)
        cwd = context.cwd

        config = SubAgentRunConfig(
            name=f"{task_def.id}_{role.value}",
            role=role.value,
            task_prompt=full_prompt,
            system_prompt=policy.system_prompt,
            allowed_tools=policy.allowed_tools,
            max_turns=policy.max_turns,
            cwd=cwd,
            runtime=runtime,
            parent_permissions=parent_perms,
            depth=0,
            is_writer=policy.is_writer,
        )

        if policy.is_writer:
            with self._writer_lock:
                self.concurrent_writer_count += 1
                if self.concurrent_writer_count > self.max_concurrent_writers_observed:
                    self.max_concurrent_writers_observed = self.concurrent_writer_count
                try:
                    return run_subagent(config)
                finally:
                    self.concurrent_writer_count -= 1
        else:
            return run_subagent(config)

    def schedule_and_run(
        self,
        plan: TeamPlan,
        context: ToolContext,
        max_replans: int = 1,
    ) -> TeamExecutionResult:
        """Execute the team plan to completion with quality gates and bounded replan."""
        start_time = time.time()
        graph = plan.graph
        task_results: dict[str, SubAgentResult] = {}
        dependency_outputs: dict[str, str] = {}
        gate_results: dict[str, QualityGateResult] = {}
        replan_count = 0
        handled_replan_tasks: set[str] = set()

        while True:
            self._cascade_skip_unreachable_tasks(graph)

            ready_tasks = graph.get_ready_tasks()
            if not ready_tasks:
                break

            for task_def in ready_tasks:
                slot_key = f"default:{task_def.id}"
                graph.start_task(slot_key)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._execute_single_task,
                        task_def,
                        plan,
                        context,
                        dependency_outputs,
                    ): task_def
                    for task_def in ready_tasks
                }

                for future in as_completed(futures):
                    task_def = futures[future]
                    slot_key = f"default:{task_def.id}"
                    role = plan.get_role_for_task(task_def.id)

                    try:
                        res = future.result()
                    except Exception as e:
                        logger.exception("Unexpected error executing task %s", task_def.id)
                        res = SubAgentResult(
                            ok=False,
                            output=f"Task {task_def.id} execution failed: {e}",
                            error=str(e),
                        )

                    task_results[task_def.id] = res

                    # Evaluate Quality Gates and handle outcomes
                    if not res.ok:
                        graph.fail_task(slot_key, error=res.error or res.output)
                        continue

                    # Task execution was technically ok; check role gates
                    if role == AgentRole.TEST:
                        gate_res = TestGate.evaluate(res)
                        gate_results[task_def.id] = gate_res
                        if gate_res.passed:
                            graph.complete_task(slot_key, result=res.output)
                            dependency_outputs[task_def.id] = res.final_message or res.output
                        else:
                            # Test gate failed!
                            if replan_count < max_replans and task_def.id not in handled_replan_tasks:
                                handled_replan_tasks.add(task_def.id)
                                replan_count += 1
                                graph.complete_task(slot_key, result=f"Completed with gate failure: {gate_res.feedback}")
                                dependency_outputs[task_def.id] = res.final_message or res.output
                                self._trigger_replan(plan, task_def.id, gate_res.feedback, replan_count)
                            else:
                                graph.fail_task(slot_key, error=f"TestGate failed: {gate_res.feedback}")

                    elif role == AgentRole.REVIEWER:
                        gate_res = ReviewGate.evaluate(res)
                        gate_results[task_def.id] = gate_res
                        if gate_res.passed:
                            graph.complete_task(slot_key, result=res.output)
                            dependency_outputs[task_def.id] = res.final_message or res.output
                        else:
                            # Review gate rejected!
                            if replan_count < max_replans and task_def.id not in handled_replan_tasks:
                                handled_replan_tasks.add(task_def.id)
                                replan_count += 1
                                graph.complete_task(slot_key, result=f"Completed with rejection: {gate_res.feedback}")
                                dependency_outputs[task_def.id] = res.final_message or res.output
                                self._trigger_replan(plan, task_def.id, gate_res.feedback, replan_count)
                            else:
                                graph.fail_task(slot_key, error=f"ReviewGate rejected: {gate_res.feedback}")

                    else:
                        graph.complete_task(slot_key, result=res.output)
                        dependency_outputs[task_def.id] = res.final_message or res.output

        elapsed = time.time() - start_time

        completed = [
            slot.task_id
            for slot in graph.slots.values()
            if slot.state == TaskState.COMPLETED
        ]
        failed = [
            slot.task_id
            for slot in graph.slots.values()
            if slot.state == TaskState.FAILED
        ]
        skipped = [
            slot.task_id
            for slot in graph.slots.values()
            if slot.state == TaskState.SKIPPED
        ]

        # Verify final reviewer / test gates
        final_gates_pass = True
        for gid, gres in gate_results.items():
            # If a gate failed and wasn't followed by a successful replan, team fails
            if not gres.passed:
                # Check if there is a later gate of the same type that passed
                replan_gates = [g for tid, g in gate_results.items() if g.gate_name == gres.gate_name and g.passed]
                if not replan_gates:
                    final_gates_pass = False

        overall_success = len(failed) == 0 and len(completed) > 0 and final_gates_pass

        summary_lines = [
            f"### Multi-Agent Team Execution Summary",
            f"- **Goal**: {plan.goal}",
            f"- **Status**: {'Success' if overall_success else 'Failed'}",
            f"- **Completed**: {len(completed)} / {len(graph.definitions)} ({', '.join(completed) if completed else 'None'})",
            f"- **Failed**: {len(failed)} ({', '.join(failed) if failed else 'None'})",
            f"- **Skipped**: {len(skipped)} ({', '.join(skipped) if skipped else 'None'})",
            f"- **Replans**: {replan_count} / {max_replans}",
            f"- **Duration**: {elapsed:.1f}s",
            f"- **Max Concurrent Writers**: {self.max_concurrent_writers_observed}",
        ]
        if gate_results:
            summary_lines.append("- **Quality Gates**:")
            for tid, gres in gate_results.items():
                summary_lines.append(f"  - {tid} ({gres.gate_name}): {gres.verdict.upper()} ({gres.feedback})")

        return TeamExecutionResult(
            success=overall_success,
            goal=plan.goal,
            task_results=task_results,
            completed_tasks=completed,
            failed_tasks=failed,
            skipped_tasks=skipped,
            replan_count=replan_count,
            elapsed_seconds=elapsed,
            summary="\n".join(summary_lines),
            error=failed[0] if failed else None,
            gate_results=gate_results,
        )
