"""Task tool — spawn a sub-agent to handle complex multi-step tasks.

Inspired by Claude Code's Task tool which launches an independent agent loop
with its own context window, isolated from the main conversation.

The sub-agent runs a full agent loop (model + tools) with:
- Its own system prompt tailored to the task type
- A filtered tool set based on the agent type
- A turn limit to prevent runaway execution
- Result summarized back into the parent context
"""
from __future__ import annotations

import time
from typing import TypedDict, cast

from minicode.agent_loop import run_agent_turn
from minicode.tooling import ToolDefinition, ToolResult
from minicode.types import ChatMessage


# ---------------------------------------------------------------------------
# Agent type definitions
# ---------------------------------------------------------------------------

class AgentDef(TypedDict):
    name: str
    description: str
    system_prompt: str
    allowed_tools: set[str] | None
    max_turns: int


AGENT_TYPES: dict[str, AgentDef] = {
    "explore": {
        "name": "Explore",
        "description": "Fast, read-only agent for codebase exploration and search",
        "system_prompt": (
            "You are an exploration agent. Your job is to quickly search and "
            "understand codebases. You should be fast and focused on finding "
            "relevant files and understanding structure. "
            "You can only use read-only tools. "
            "When done, provide a concise summary of your findings."
        ),
        "allowed_tools": {"read_file", "list_files", "grep_files", "file_tree", "find_symbols", "find_references", "get_ast_info"},
        "max_turns": 5,
    },
    "plan": {
        "name": "Plan",
        "description": "Thorough agent for gathering context and understanding code",
        "system_prompt": (
            "You are a planning agent. Your job is to thoroughly understand "
            "the codebase and task before acting. Read multiple files, trace "
            "code paths, and build a complete mental model. "
            "You can only use read-only tools. "
            "When done, provide a detailed analysis with actionable recommendations."
        ),
        "allowed_tools": {"read_file", "list_files", "grep_files", "file_tree", "find_symbols", "find_references", "get_ast_info", "code_review"},
        "max_turns": 8,
    },
    "general": {
        "name": "General",
        "description": "Full-featured agent for complex multi-step tasks",
        "system_prompt": (
            "You are a general-purpose coding agent. You can read, write, "
            "and modify code. Follow best practices and explain your changes. "
            "Break complex tasks into smaller steps. "
            "When done, provide a summary of what you did and any important findings."
        ),
        "allowed_tools": None,  # None = all tools allowed
        "max_turns": 15,
    },
}


def _validate(input_data: dict) -> dict:
    description = input_data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required")
    
    agent_type = input_data.get("agent_type", "general")
    if agent_type not in AGENT_TYPES:
        valid = ", ".join(AGENT_TYPES.keys())
        raise ValueError(f"agent_type must be one of: {valid}. Got: {agent_type}")
    
    return {
        "description": description.strip(),
        "agent_type": agent_type,
        "prompt": input_data.get("prompt", description.strip()),
    }


def _run(input_data: dict, context) -> ToolResult:
    """Execute a sub-agent task via SubAgentRunner."""
    from minicode.subagent_runner import SubAgentRunConfig, run_subagent

    agent_type = input_data["agent_type"]
    agent_def = AGENT_TYPES[agent_type]
    task_prompt = input_data["prompt"]

    runtime = None
    if hasattr(context, "_runtime") and context._runtime:
        runtime = context._runtime

    if not runtime:
        try:
            from minicode.config import load_runtime_config
            runtime = load_runtime_config(context.cwd)
        except Exception:
            pass

    if not runtime:
        return ToolResult(
            ok=False,
            output="Cannot run sub-agent: no model configuration available. Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL.",
        )

    is_writer = agent_def["allowed_tools"] is None
    parent_depth = getattr(context, "depth", 0) if hasattr(context, "depth") else 0

    config = SubAgentRunConfig(
        name=agent_def["name"],
        role=agent_type,
        task_prompt=task_prompt,
        system_prompt=agent_def["system_prompt"],
        allowed_tools=agent_def["allowed_tools"],
        max_turns=agent_def["max_turns"],
        cwd=context.cwd,
        runtime=runtime,
        parent_permissions=getattr(context, "permissions", None),
        depth=parent_depth,
        is_writer=is_writer,
    )

    sub_res = run_subagent(config)

    if not sub_res.ok:
        return ToolResult(ok=False, output=sub_res.output)

    header = (
        f"[Sub-agent {agent_def['name']} completed]\n"
        f"  Type: {agent_type}\n"
        f"  Turns: {sub_res.turn_count} (tool calls: {sub_res.tool_calls_count})\n"
        f"  Duration: {sub_res.elapsed_seconds:.1f}s\n"
        f"  Max turns: {agent_def['max_turns']}\n"
    )

    result_text = sub_res.final_message
    MAX_RESULT_LEN = 8000
    if len(result_text) > MAX_RESULT_LEN:
        result_text = result_text[:MAX_RESULT_LEN] + f"\n\n... (truncated, {len(sub_res.final_message)} chars total)"

    return ToolResult(ok=True, output=header + "\n" + result_text)



task_tool = ToolDefinition(
    name="task",
    description=(
        "Launch a sub-agent to handle a complex task autonomously. "
        "The sub-agent runs in its own isolated context with a turn limit. "
        "Use 'explore' for fast read-only codebase exploration, "
        "'plan' for thorough analysis, or 'general' for full-featured multi-step work. "
        "The sub-agent's final result is returned to you."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short 3-5 word description of the task",
            },
            "prompt": {
                "type": "string",
                "description": "Full task description for the sub-agent. If not provided, uses 'description'.",
            },
            "agent_type": {
                "type": "string",
                "enum": ["explore", "plan", "general"],
                "description": "Type of sub-agent: 'explore' (fast, read-only), 'plan' (thorough, read-only), 'general' (full tools, default)",
            },
        },
        "required": ["description"],
    },
    validator=_validate,
    run=_run,
)
