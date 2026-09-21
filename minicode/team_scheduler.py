"""Centralized Multi-Agent Team Scheduler.

Coordinates parallel execution of sibling read-only sub-agents,
enforces writer/reader mutual exclusion and writer serialization,
executes strict quality gates (Test Gate & Review Gate),
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
    VerificationStatus,
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
    """Verifies that the Test sub-agent executed tests and verified pass status strictly from test_runner evidence."""

    @staticmethod
    def evaluate(result: SubAgentResult) -> QualityGateResult:
        # Check tool_events for test_runner
        test_events = [
            e for e in getattr(result, "tool_events", [])
            if e.tool_name == "test_runner"
        ]

        if not test_events:
            return QualityGateResult(
                passed=False,
                gate_name="TestGate",
                verdict=VerificationStatus.UNVERIFIED.value,
                feedback="Test gate unverified: no test_runner tool execution found in child events.",
            )

        # Look strictly at the LAST test_runner execution result
        last_test = test_events[-1]
        if last_test.ok and not last_test.is_error:
            return QualityGateResult(
                passed=True,
                gate_name="TestGate",
                verdict=VerificationStatus.PASS.value,
                feedback="Test gate passed: verified by test_runner tool result (ok=True).",
                details=last_test.to_dict(),
            )
        else:
            return QualityGateResult(
                passed=False,
                gate_name="TestGate",
                verdict=VerificationStatus.FAIL.value,
                feedback=f"Test gate failed: test_runner execution returned failure ({last_test.output_summary[:200]}).",
                details=last_test.to_dict(),
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

        is_approved = (
            verdict in ("approve", "approved", "pass", "passed")
            or status in ("approve", "approved", "pass", "passed")
        )

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
    status: str = "completed"
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
            "status": self.status,
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

    def __init__(self, max_workers: int = 4, max_parallel_readers: int = 2):
        self.max_workers = max_workers
        self.max_parallel_readers = max_parallel_readers
        self._writer_lock = threading.Lock()
        self._reader_lock = threading.Lock()
        self.active_writers = 0
        self.active_readers = 0
        self.max_concurrent_writers_observed = 0
        self.max_concurrent_readers_observed = 0
        self.reader_writer_overlap_observed = False

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

        # If source was test, explicitly mark original reviewer as SKIPPED so it never runs!
        for tid, tdef in list(graph.definitions.items()):
            if source_task_id in tdef.dependencies:
                slot_key = f"default:{tid}"
                if slot_key in graph.slots and graph.slots[slot_key].state == TaskState.PENDING:
                    graph.skip_task(slot_key, reason="superseded_by_replan")

        coding_id = f"coding_replan_{replan_index}"
        test_id = f"test_replan_{replan_index}"
        reviewer_id = f"reviewer_replan_{replan_index}"

        plan.task_roles[coding_id] = AgentRole.CODING
        plan.task_roles[test_id] = AgentRole.TEST
        plan.task_roles[reviewer_id] = AgentRole.REVIEWER

        # coding_replan depends on completed upstream coding task, not the failed test gate
        upstream_coding = "coding"
        if replan_index > 1:
            upstream_coding = f"coding_replan_{replan_index - 1}"

        def_coding = TaskDefinition(
            id=coding_id,
            name=f"Fix Quality Gate Issues (Replan {replan_index})",
            description=f"Implement fixes for quality gate issues: {feedback}",
            dependencies=[upstream_coding],
            priority=TaskPriority.CRITICAL,
            metadata={"role": AgentRole.CODING.value, "replan": replan_index},
        )
        graph.definitions[coding_id] = def_coding
        graph.assign_slot(coding_id, slot_name="default")

        def_test = TaskDefinition(
            id=test_id,
            name=f"Verify Replan Fix (Replan {replan_index})",
            description=f"Execute test_runner to verify fix for: {feedback}",
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

        return run_subagent(config)

    def _handle_task_result(
        self,
        task_def: TaskDefinition,
        role: AgentRole,
        res: SubAgentResult,
        slot_key: str,
        plan: TeamPlan,
        graph: TaskGraph,
        task_results: dict[str, SubAgentResult],
        dependency_outputs: dict[str, str],
        gate_results: dict[str, QualityGateResult],
        max_replans: int,
        replan_state: dict[str, Any],
    ) -> None:
        """Process execution result and quality gate logic for a task."""
        task_results[task_def.id] = res

        if not res.ok:
            graph.fail_task(slot_key, error=res.error or res.output)
            return

        if role == AgentRole.TEST:
            gate_res = TestGate.evaluate(res)
            gate_results[task_def.id] = gate_res
            if gate_res.passed:
                graph.complete_task(slot_key, result=res.output)
                dependency_outputs[task_def.id] = res.final_message or res.output
            else:
                graph.fail_task(slot_key, error=f"TestGate failed: {gate_res.feedback}")
                if replan_state["count"] < max_replans and task_def.id not in replan_state["handled"]:
                    replan_state["handled"].add(task_def.id)
                    replan_state["count"] += 1
                    dependency_outputs[task_def.id] = res.final_message or res.output
                    self._trigger_replan(plan, task_def.id, gate_res.feedback, replan_state["count"])

        elif role == AgentRole.REVIEWER:
            gate_res = ReviewGate.evaluate(res)
            gate_results[task_def.id] = gate_res
            if gate_res.passed:
                graph.complete_task(slot_key, result=res.output)
                dependency_outputs[task_def.id] = res.final_message or res.output
            else:
                graph.fail_task(slot_key, error=f"ReviewGate rejected: {gate_res.feedback}")
                if replan_state["count"] < max_replans and task_def.id not in replan_state["handled"]:
                    replan_state["handled"].add(task_def.id)
                    replan_state["count"] += 1
                    dependency_outputs[task_def.id] = res.final_message or res.output
                    self._trigger_replan(plan, task_def.id, gate_res.feedback, replan_state["count"])

        else:
            graph.complete_task(slot_key, result=res.output)
            dependency_outputs[task_def.id] = res.final_message or res.output

    def schedule_and_run(
        self,
        plan: TeamPlan,
        context: ToolContext,
        max_replans: int = 1,
        use_worktree: bool = False,
    ) -> TeamExecutionResult:
        """Execute the team plan to completion with quality gates, bounded replan, and optional worktree isolation."""
        start_time = time.time()
        graph = plan.graph
        task_results: dict[str, SubAgentResult] = {}
        dependency_outputs: dict[str, str] = {}
        gate_results: dict[str, QualityGateResult] = {}
        replan_state = {"count": 0, "handled": set()}

        # Clamp max_replans strictly to 0..1 in Phase 4
        max_replans = max(0, min(1, int(max_replans)))

        isolator = None
        worktree_path = None
        target_context = context

        if use_worktree:
            from pathlib import Path
            from minicode.task_graph import WorktreeIsolator
            isolator = WorktreeIsolator(base_path=Path(context.cwd))

            # 1. Check parent clean
            is_clean, clean_msg = isolator.check_parent_workspace_clean()
            if not is_clean:
                return TeamExecutionResult(
                    success=False,
                    goal=plan.goal,
                    status="parent_workspace_dirty",
                    error=f"Cannot execute worktree: {clean_msg}",
                    summary=f"### Multi-Agent Team Execution Summary\n- **Status**: Failed (parent_workspace_dirty)\n- **Reason**: {clean_msg}",
                )

            # 2. Check git repo and create worktree in detached mode
            worktree_path = isolator.create_worktree(plan.graph.name)
            if not worktree_path:
                # Fail closed! Never fall back to parent workspace
                return TeamExecutionResult(
                    success=False,
                    goal=plan.goal,
                    status="worktree_setup_failed",
                    error="Worktree setup failed (fail-closed enforced).",
                    summary="### Multi-Agent Team Execution Summary\n- **Status**: Failed (worktree_setup_failed)",
                )

            # Scoped approval handler for worktree cwd
            from minicode.permissions import PermissionManager
            worktree_permissions = isolator.create_isolated_permission_manager(
                worktree_cwd=str(worktree_path),
                parent_permissions=context.permissions,
            )

            target_context = ToolContext(
                cwd=str(worktree_path),
                permissions=worktree_permissions,
                session=context.session,
                _runtime=context._runtime,
            )

        while True:
            self._cascade_skip_unreachable_tasks(graph)

            ready_tasks = graph.get_ready_tasks()
            if not ready_tasks:
                break

            # Partition ready tasks into writers vs readers
            writer_tasks = [
                t for t in ready_tasks
                if get_role_policy(plan.get_role_for_task(t.id)).is_writer
            ]
            reader_tasks = [
                t for t in ready_tasks
                if not get_role_policy(plan.get_role_for_task(t.id)).is_writer
            ]

            if writer_tasks:
                # WRITER WAVE: At most 1 writer, ZERO concurrent readers
                task_def = writer_tasks[0]
                slot_key = f"default:{task_def.id}"
                graph.start_task(slot_key)
                role = plan.get_role_for_task(task_def.id)

                with self._writer_lock:
                    self.active_writers += 1
                    if self.active_readers > 0:
                        self.reader_writer_overlap_observed = True
                    if self.active_writers > self.max_concurrent_writers_observed:
                        self.max_concurrent_writers_observed = self.active_writers
                    try:
                        res = self._execute_single_task(
                            task_def, plan, target_context, dependency_outputs
                        )
                    except Exception as e:
                        logger.exception("Unexpected error executing writer task %s", task_def.id)
                        res = SubAgentResult(
                            ok=False,
                            output=f"Task {task_def.id} execution failed: {e}",
                            error=str(e),
                        )
                    finally:
                        self.active_writers -= 1

                self._handle_task_result(
                    task_def, role, res, slot_key, plan, graph,
                    task_results, dependency_outputs, gate_results,
                    max_replans, replan_state
                )

            else:
                # READER WAVE: Read-only siblings can run in parallel
                batch = reader_tasks[:self.max_parallel_readers]
                for tdef in batch:
                    graph.start_task(f"default:{tdef.id}")

                def _run_reader(tdef):
                    with self._reader_lock:
                        self.active_readers += 1
                        if self.active_writers > 0:
                            self.reader_writer_overlap_observed = True
                        if self.active_readers > self.max_concurrent_readers_observed:
                            self.max_concurrent_readers_observed = self.active_readers
                    try:
                        return self._execute_single_task(
                            tdef, plan, target_context, dependency_outputs
                        )
                    finally:
                        with self._reader_lock:
                            self.active_readers -= 1

                with ThreadPoolExecutor(max_workers=min(len(batch), self.max_parallel_readers)) as executor:
                    future_to_task = {
                        executor.submit(_run_reader, tdef): tdef
                        for tdef in batch
                    }

                    for future in as_completed(future_to_task):
                        tdef = future_to_task[future]
                        slot_key = f"default:{tdef.id}"
                        role = plan.get_role_for_task(tdef.id)
                        try:
                            res = future.result()
                        except Exception as e:
                            logger.exception("Unexpected error executing reader task %s", tdef.id)
                            res = SubAgentResult(
                                ok=False,
                                output=f"Task {tdef.id} execution failed: {e}",
                                error=str(e),
                            )

                        self._handle_task_result(
                            tdef, role, res, slot_key, plan, graph,
                            task_results, dependency_outputs, gate_results,
                            max_replans, replan_state
                        )

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

        # Determine terminal gate status
        replan_count = replan_state["count"]
        terminal_test_id = f"test_replan_{replan_count}" if replan_count > 0 else "test"
        terminal_reviewer_id = f"reviewer_replan_{replan_count}" if replan_count > 0 else "reviewer"

        test_gate_res = gate_results.get(terminal_test_id)
        review_gate_res = gate_results.get(terminal_reviewer_id)

        terminal_test_pass = test_gate_res is not None and test_gate_res.passed
        terminal_review_pass = review_gate_res is not None and review_gate_res.passed

        # Unhandled failures check
        unhandled_failures = [f for f in failed if f not in replan_state["handled"]]
        team_workflow_success = len(unhandled_failures) == 0 and terminal_test_pass and terminal_review_pass

        final_status = "completed" if team_workflow_success else "failed"
        patch_applied = False

        if isolator and worktree_path:
            try:
                if team_workflow_success:
                    # 1. Check workspace fingerprint
                    fp_ok, fp_msg = isolator.verify_parent_fingerprint()
                    if not fp_ok:
                        team_workflow_success = False
                        final_status = "parent_workspace_changed"
                    else:
                        # 2. Generate binary patch
                        patch_content = isolator.generate_patch(worktree_path)
                        if patch_content.strip():
                            # 3. Dry-run verify patch
                            if not isolator.verify_patch(patch_content):
                                team_workflow_success = False
                                final_status = "patch_verify_failed"
                            else:
                                # 4. Permission check & apply patch
                                apply_ok, apply_status = isolator.apply_patch(
                                    patch_content, permissions=context.permissions
                                )
                                if apply_ok:
                                    patch_applied = True
                                else:
                                    team_workflow_success = False
                                    final_status = apply_status
                        else:
                            # No changes produced
                            patch_applied = True
                else:
                    final_status = "execution_failed"
            finally:
                isolator.cleanup_all()

        summary_lines = [
            f"### Multi-Agent Team Execution Summary",
            f"- **Goal**: {plan.goal}",
            f"- **Status**: {'Success' if team_workflow_success else f'Failed ({final_status})'}",
            f"- **Completed**: {len(completed)} / {len(graph.definitions)} ({', '.join(completed) if completed else 'None'})",
            f"- **Failed**: {len(failed)} ({', '.join(failed) if failed else 'None'})",
            f"- **Skipped**: {len(skipped)} ({', '.join(skipped) if skipped else 'None'})",
            f"- **Replans**: {replan_count} / {max_replans}",
            f"- **Duration**: {elapsed:.1f}s",
            f"- **Max Concurrent Writers**: {self.max_concurrent_writers_observed}",
            f"- **Max Concurrent Readers**: {self.max_concurrent_readers_observed}",
            f"- **Reader/Writer Overlap**: {self.reader_writer_overlap_observed}",
        ]
        if use_worktree:
            summary_lines.append(f"- **Worktree Isolation**: Active (Status: {final_status}, Patch applied: {patch_applied})")
        if gate_results:
            summary_lines.append("- **Quality Gates**:")
            for tid, gres in gate_results.items():
                summary_lines.append(f"  - {tid} ({gres.gate_name}): {gres.verdict.upper()} ({gres.feedback})")

        return TeamExecutionResult(
            success=team_workflow_success,
            goal=plan.goal,
            status=final_status,
            task_results=task_results,
            completed_tasks=completed,
            failed_tasks=failed,
            skipped_tasks=skipped,
            replan_count=replan_count,
            elapsed_seconds=elapsed,
            summary="\n".join(summary_lines),
            error=unhandled_failures[0] if unhandled_failures else (None if team_workflow_success else final_status),
            gate_results=gate_results,
        )
