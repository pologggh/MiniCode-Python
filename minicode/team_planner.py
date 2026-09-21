"""Deterministic Team Planner for Centralized Multi-Agent Orchestration.

Builds a structured TaskGraph DAG establishing the standard multi-agent topology:
  parallel(research_impl, research_test) -> coding -> test -> reviewer
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from minicode.task_graph import TaskDefinition, TaskGraph, TaskPriority
from minicode.team_roles import AgentRole, get_role_policy


@dataclass(slots=True)
class TeamPlan:
    """Execution plan containing the configured TaskGraph and task role metadata."""
    goal: str
    graph: TaskGraph
    task_roles: dict[str, AgentRole] = field(default_factory=dict)

    def get_role_for_task(self, task_id: str) -> AgentRole:
        return self.task_roles.get(task_id, AgentRole.RESEARCH)


class TeamPlanner:
    """Generates deterministic execution plans and DAGs for multi-agent teams."""

    def plan(self, goal: str, custom_tasks: list[dict[str, Any]] | None = None) -> TeamPlan:
        """Construct a TaskGraph DAG for the specified goal.
        
        If custom_tasks is provided, constructs DAG from custom specifications.
        Otherwise, establishes the canonical 5-node software engineering DAG:
        1. research_impl (RESEARCH) - parallel
        2. research_test (RESEARCH) - parallel
        3. coding (CODING) - depends on research_impl
        4. test (TEST) - depends on coding, research_test
        5. reviewer (REVIEWER) - depends on test
        """
        graph = TaskGraph(name=f"team_{int(time.time())}")
        task_roles: dict[str, AgentRole] = {}

        if custom_tasks:
            for item in custom_tasks:
                task_id = item["id"]
                role_str = item.get("role", "research")
                try:
                    role = AgentRole(role_str)
                except ValueError:
                    role = AgentRole.RESEARCH
                task_roles[task_id] = role

                task_def = TaskDefinition(
                    id=task_id,
                    name=item.get("name", task_id),
                    description=item.get("description", f"{task_id} for {goal}"),
                    dependencies=item.get("dependencies", []),
                    priority=TaskPriority(item.get("priority", "normal")),
                    timeout_seconds=item.get("timeout_seconds", 300),
                    metadata={"role": role.value, **item.get("metadata", {})},
                )
                graph.definitions[task_id] = task_def
                graph.assign_slot(task_id, slot_name="default")
            return TeamPlan(goal=goal, graph=graph, task_roles=task_roles)

        # Standard canonical workflow
        # 1. Research Implementation
        task_roles["research_impl"] = AgentRole.RESEARCH
        def_r_impl = TaskDefinition(
            id="research_impl",
            name="Research Implementation Architecture",
            description=f"Explore files, trace interfaces, and identify code changes for: {goal}",
            dependencies=[],
            priority=TaskPriority.HIGH,
            metadata={"role": AgentRole.RESEARCH.value},
        )
        graph.definitions[def_r_impl.id] = def_r_impl
        graph.assign_slot(def_r_impl.id, slot_name="default")

        # 2. Research Testing
        task_roles["research_test"] = AgentRole.RESEARCH
        def_r_test = TaskDefinition(
            id="research_test",
            name="Research Verification & Test Strategy",
            description=f"Inspect existing test suites, runners, and validation requirements for: {goal}",
            dependencies=[],
            priority=TaskPriority.HIGH,
            metadata={"role": AgentRole.RESEARCH.value},
        )
        graph.definitions[def_r_test.id] = def_r_test
        graph.assign_slot(def_r_test.id, slot_name="default")

        # 3. Coding (Writer)
        task_roles["coding"] = AgentRole.CODING
        def_coding = TaskDefinition(
            id="coding",
            name="Implement Code Changes",
            description=f"Implement changes and solutions according to research for: {goal}",
            dependencies=["research_impl"],
            priority=TaskPriority.HIGH,
            metadata={"role": AgentRole.CODING.value},
        )
        graph.definitions[def_coding.id] = def_coding
        graph.assign_slot(def_coding.id, slot_name="default")

        # 4. Testing (Test Gate)
        task_roles["test"] = AgentRole.TEST
        def_test = TaskDefinition(
            id="test",
            name="Execute Verification Tests",
            description=f"Execute tests using test_runner to verify implementation for: {goal}",
            dependencies=["coding", "research_test"],
            priority=TaskPriority.CRITICAL,
            metadata={"role": AgentRole.TEST.value},
        )
        graph.definitions[def_test.id] = def_test
        graph.assign_slot(def_test.id, slot_name="default")

        # 5. Reviewer (Review Gate)
        task_roles["reviewer"] = AgentRole.REVIEWER
        def_reviewer = TaskDefinition(
            id="reviewer",
            name="Structured Code Review",
            description=f"Review code modifications and verification evidence; provide structured JSON verdict for: {goal}",
            dependencies=["test"],
            priority=TaskPriority.NORMAL,
            metadata={"role": AgentRole.REVIEWER.value},
        )
        graph.definitions[def_reviewer.id] = def_reviewer
        graph.assign_slot(def_reviewer.id, slot_name="default")

        return TeamPlan(goal=goal, graph=graph, task_roles=task_roles)
