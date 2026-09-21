"""Centralized Multi-Agent Team Scheduler.

Coordinates parallel execution of sibling sub-agents using ThreadPoolExecutor,
enforces writer serialization, executes quality gates, and safely contains failures.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from minicode.logging_config import get_logger
from minicode.subagent_runner import SubAgentRunConfig, SubAgentResult, run_subagent
from minicode.task_graph import TaskDefinition, TaskGraph, TaskPriority, TaskSlot, TaskState
from minicode.team_planner import TeamPlan
from minicode.team_roles import AgentRole, AgentRolePolicy, get_role_policy
from minicode.tooling import ToolContext

logger = get_logger("team_scheduler")


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

                # If any dependency is failed/skipped, this task can never succeed
                if any(dep in failed_or_skipped for dep in task_def.dependencies):
                    graph.skip_task(slot_key, reason="Upstream dependency failed or was skipped")
                    round_skipped.append(task_def.id)

            if not round_skipped:
                break
            all_newly_skipped.extend(round_skipped)

        return all_newly_skipped


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

        # Build context-rich prompt incorporating upstream outputs
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
                    # Truncate upstream context to protect context window
                    excerpt = dep_output[:2500]
                    prompt_parts.append(f"--- Output from {dep_id} ---\n{excerpt}")

        full_prompt = "\n\n".join(prompt_parts)

        # Build SubAgentRunConfig
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

        # Enforce writer serialization
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
        """Execute the team plan to completion."""
        start_time = time.time()
        graph = plan.graph
        task_results: dict[str, SubAgentResult] = {}
        dependency_outputs: dict[str, str] = {}

        while True:
            # First, cascade skips for any task whose dependencies failed/skipped
            self._cascade_skip_unreachable_tasks(graph)

            ready_tasks = graph.get_ready_tasks()
            if not ready_tasks:
                break

            # Mark ready tasks as running in graph
            for task_def in ready_tasks:
                slot_key = f"default:{task_def.id}"
                graph.start_task(slot_key)

            # Execute ready tasks in parallel
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
                    if res.ok:
                        graph.complete_task(slot_key, result=res.output)
                        dependency_outputs[task_def.id] = res.final_message or res.output
                    else:
                        graph.fail_task(slot_key, error=res.error or res.output)

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

        overall_success = len(failed) == 0 and len(completed) > 0

        # Build summary text
        summary_lines = [
            f"### Multi-Agent Team Execution Summary",
            f"- **Goal**: {plan.goal}",
            f"- **Status**: {'Success' if overall_success else 'Failed'}",
            f"- **Completed**: {len(completed)} / {len(graph.definitions)} ({', '.join(completed) if completed else 'None'})",
            f"- **Failed**: {len(failed)} ({', '.join(failed) if failed else 'None'})",
            f"- **Skipped**: {len(skipped)} ({', '.join(skipped) if skipped else 'None'})",
            f"- **Duration**: {elapsed:.1f}s",
            f"- **Max Concurrent Writers**: {self.max_concurrent_writers_observed}",
        ]

        return TeamExecutionResult(
            success=overall_success,
            goal=plan.goal,
            task_results=task_results,
            completed_tasks=completed,
            failed_tasks=failed,
            skipped_tasks=skipped,
            replan_count=0,
            elapsed_seconds=elapsed,
            summary="\n".join(summary_lines),
            error=failed[0] if failed else None,
        )
