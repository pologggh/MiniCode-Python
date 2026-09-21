"""Agent Team tool - Orchestrate a centralized team of sub-agents.

Allows the parent agent to delegate complex, multi-faceted engineering tasks
to a coordinated team consisting of:
- Parallel Research Sub-agents (Implementation & Testing)
- Coding Sub-agent (Writer)
- Test Sub-agent (Verification Gate)
- Code Review Sub-agent (Structured Review Gate)

Crucially, this tool operates within the parent tool execution pipeline and
never hijacks or overrides the parent's turn kernel.
"""

from __future__ import annotations

from typing import Any

from minicode.tooling import ToolContext, ToolDefinition, ToolResult


def _validate(input_data: dict[str, Any]) -> dict[str, Any]:
    goal = input_data.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal is required and must be a non-empty string")

    use_worktree = bool(input_data.get("use_worktree", False))
    max_replans = int(input_data.get("max_replans", 1))
    if max_replans < 0:
        max_replans = 0
    elif max_replans > 3:
        max_replans = 3

    return {
        "goal": goal.strip(),
        "use_worktree": use_worktree,
        "max_replans": max_replans,
    }


def _run(input_data: dict[str, Any], context: ToolContext) -> ToolResult:
    """Execute team planning and scheduling."""
    from minicode.team_planner import TeamPlanner
    from minicode.team_scheduler import TeamScheduler

    goal = input_data["goal"]
    use_worktree = input_data["use_worktree"]
    max_replans = input_data["max_replans"]


    # Verify model configuration
    runtime = getattr(context, "_runtime", None)
    if not runtime:
        try:
            from minicode.config import load_runtime_config
            runtime = load_runtime_config(context.cwd)
        except Exception:
            pass

    if not runtime:
        return ToolResult(
            ok=False,
            output="Cannot execute agent team: no model configuration available.",
        )

    planner = TeamPlanner()
    plan = planner.plan(goal=goal)

    scheduler = TeamScheduler(max_workers=4)
    result = scheduler.schedule_and_run(
        plan=plan,
        context=context,
        max_replans=max_replans,
        use_worktree=use_worktree,
    )

    return ToolResult(
        ok=result.success,
        output=result.summary,
    )


agent_team_tool = ToolDefinition(
    name="agent_team",
    description=(
        "Orchestrate a centralized multi-agent team (researchers, coder, test runner, reviewer) "
        "to autonomously plan and execute a complex software engineering task. "
        "The team coordinates parallel research, serialized coding, test verification, "
        "and structured code review."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "High-level goal or task description for the multi-agent team to accomplish.",
            },
            "use_worktree": {
                "type": "boolean",
                "description": "Whether to isolate file changes inside a temporary git worktree (default: false).",
                "default": False,
            },
            "max_replans": {
                "type": "integer",
                "description": "Maximum allowed replanning cycles if quality gates fail (default: 1, range: 0-3).",
                "default": 1,
            },
        },
        "required": ["goal"],
    },
    validator=_validate,
    run=_run,
)
