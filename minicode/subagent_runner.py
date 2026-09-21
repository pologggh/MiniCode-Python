"""Reusable SubAgentRunner for isolated sub-agent execution.

Extracted from minicode/tools/task.py to provide a unified, secure foundation
for both single-agent tasks and centralized multi-agent orchestration.

Key Isolation Guarantees:
1. Depth Enforcement: Depth limit <= 1. Child agents cannot spawn sub-agents.
   Child tool registries explicitly strip 'task' and 'agent_team'.
2. MCP Server Isolation: Child runtime never inherits parent MCP servers (mcpServers = {}).
3. Privilege Containment: Read-only roles use unprompted PermissionManager (auto-denying writes).
4. Failure Containment: Child agent crashes are safely caught and converted to SubAgentResult.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, cast

from minicode.agent_loop import run_agent_turn
from minicode.logging_config import get_logger
from minicode.model_registry import create_model_adapter
from minicode.permissions import PermissionManager
from minicode.tooling import ToolRegistry
from minicode.types import ChatMessage

logger = get_logger("subagent_runner")

# Tools that child agents must never be permitted to invoke
FORBIDDEN_CHILD_TOOLS: frozenset[str] = frozenset({
    "task",
    "agent_team",
})

MAX_SUBAGENT_DEPTH: int = 1


class SubAgentDepthError(RuntimeError):
    """Raised when recursive sub-agent nesting exceeds allowed depth."""
    pass


@dataclass(slots=True)
class SubAgentRunConfig:
    """Configuration for running an isolated sub-agent."""
    name: str
    task_prompt: str
    system_prompt: str
    allowed_tools: set[str] | list[str] | None = None
    max_turns: int = 10
    cwd: str = ""
    runtime: dict[str, Any] | None = None
    parent_permissions: Any | None = None
    depth: int = 0
    is_writer: bool = False
    model_name: str | None = None
    role: str = "general"
    timeout_seconds: float = 300.0


@dataclass(slots=True)
class SubAgentResult:
    """Structured result returned by a sub-agent execution."""
    ok: bool
    output: str
    final_message: str = ""
    tool_calls_count: int = 0
    turn_count: int = 0
    elapsed_seconds: float = 0.0
    structured_data: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "final_message": self.final_message,
            "tool_calls_count": self.tool_calls_count,
            "turn_count": self.turn_count,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "structured_data": self.structured_data,
            "error": self.error,
        }


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    """Attempt to extract structured JSON object from sub-agent final text."""
    if not text or not text.strip():
        return None
    # 1. Direct parse
    text_stripped = text.strip()
    if text_stripped.startswith("{") and text_stripped.endswith("}"):
        try:
            parsed = json.loads(text_stripped)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 2. Markdown code block
    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block_match:
        try:
            parsed = json.loads(json_block_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 3. First balanced curly braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def run_subagent(config: SubAgentRunConfig) -> SubAgentResult:
    """Execute an isolated sub-agent with strict boundaries.
    
    Guarantees:
    - Rejects recursive nesting if depth >= MAX_SUBAGENT_DEPTH (depth limit = 1).
    - Strips 'task' and 'agent_team' from child tool registry.
    - Strips mcpServers from child runtime to prevent MCP inheritance.
    - Enforces read-only permissions when is_writer is False.
    - Catches all exceptions and returns structured SubAgentResult without crashing parent.
    """
    start_time = time.time()

    # 1. Depth Limit Enforcement
    if config.depth >= MAX_SUBAGENT_DEPTH:
        msg = (
            f"Recursive sub-agent execution rejected: depth {config.depth} "
            f"meets or exceeds limit {MAX_SUBAGENT_DEPTH}."
        )
        logger.warning(msg)
        return SubAgentResult(
            ok=False,
            output=msg,
            elapsed_seconds=0.0,
            error="DepthLimitExceeded",
        )

    # 2. Runtime & MCP Isolation
    child_runtime: dict[str, Any] = dict(config.runtime or {})
    # Strictly strip MCP servers to prevent child agent inheritance
    child_runtime["mcpServers"] = {}

    cwd = config.cwd or "."

    # 3. Tool Filtering & Stripping
    try:
        from minicode.tools import create_default_tool_registry
        full_tools = create_default_tool_registry(cwd, runtime=child_runtime)
    except Exception as e:
        logger.error("Failed to create tool registry for sub-agent: %s", e)
        return SubAgentResult(
            ok=False,
            output=f"Sub-agent tool registry initialization failed: {e}",
            elapsed_seconds=time.time() - start_time,
            error=str(e),
        )

    # Always remove forbidden tools (task, agent_team)
    if config.allowed_tools is not None:
        allowed_set = set(config.allowed_tools) - FORBIDDEN_CHILD_TOOLS
        filtered_tools = [t for t in full_tools.list() if t.name in allowed_set]
    else:
        filtered_tools = [t for t in full_tools.list() if t.name not in FORBIDDEN_CHILD_TOOLS]

    tools = ToolRegistry(filtered_tools)

    # 4. Model Adapter
    model_identifier = config.model_name or child_runtime.get("model", "")
    if not model_identifier and not child_runtime:
        # Fallback check
        try:
            from minicode.config import load_runtime_config
            loaded_runtime = load_runtime_config(cwd)
            child_runtime.update(loaded_runtime)
            child_runtime["mcpServers"] = {}
            model_identifier = config.model_name or child_runtime.get("model", "")
        except Exception:
            pass

    if not model_identifier and not child_runtime.get("model"):
        return SubAgentResult(
            ok=False,
            output="Cannot run sub-agent: no model configuration available.",
            elapsed_seconds=time.time() - start_time,
            error="MissingModelConfiguration",
        )

    try:
        model = create_model_adapter(
            model=model_identifier,
            tools=tools,
            runtime=child_runtime,
        )
    except Exception as e:
        logger.error("Failed to create model adapter for sub-agent: %s", e)
        return SubAgentResult(
            ok=False,
            output=f"Model adapter initialization failed: {e}",
            elapsed_seconds=time.time() - start_time,
            error=str(e),
        )

    # 5. Permission Sandboxing
    if not config.is_writer or config.allowed_tools is not None:
        # Read-only or restricted agent: prompt=None auto-denies writes outside cwd
        sub_permissions = PermissionManager(cwd, prompt=None)
    else:
        # Writer agent: inherit parent's permission prompt handler
        sub_permissions = PermissionManager(
            cwd,
            prompt=getattr(config.parent_permissions, "prompt", None),
        )

    # 6. Messages Setup
    sub_messages: list[ChatMessage] = cast(
        list[ChatMessage],
        [
            {
                "role": "system",
                "content": (
                    config.system_prompt
                    + f"\n\nCurrent cwd: {cwd}"
                    + "\n\nIMPORTANT: When you have completed your task, end with <final> and provide your findings."
                    + " Do not ask the user questions — work autonomously with the tools available."
                    + " Be concise and focused."
                ),
            },
            {
                "role": "user",
                "content": config.task_prompt,
            },
        ],
    )

    # 7. Execution Loop with Crash Containment
    try:
        result_messages = run_agent_turn(
            model=model,
            tools=tools,
            messages=sub_messages,
            cwd=cwd,
            permissions=sub_permissions,
            max_steps=config.max_turns,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception("Sub-agent %s crashed during execution", config.name)
        return SubAgentResult(
            ok=False,
            output=f"Sub-agent ({config.name}) failed: {type(e).__name__}: {e}",
            elapsed_seconds=elapsed,
            error=str(e),
        )

    elapsed = time.time() - start_time

    # 8. Extract Outcome
    final_message = ""
    for msg in reversed(result_messages):
        if msg.get("role") == "assistant" and msg.get("content", "").strip():
            final_message = msg["content"]
            break

    if not final_message:
        final_message = "(sub-agent completed without a final message)"

    tool_calls_count = sum(1 for m in result_messages if m.get("role") == "assistant_tool_call")
    user_messages_count = sum(1 for m in result_messages if m.get("role") == "user")

    # Extract structured JSON if available (e.g. for Reviewer verdict)
    structured_data = _extract_json_payload(final_message)

    return SubAgentResult(
        ok=True,
        output=final_message,
        final_message=final_message,
        tool_calls_count=tool_calls_count,
        turn_count=user_messages_count,
        elapsed_seconds=elapsed,
        structured_data=structured_data,
        error=None,
    )
