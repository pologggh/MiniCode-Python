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
from enum import Enum
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


class VerificationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"


@dataclass(slots=True)
class SubAgentToolEvent:
    """Bounded event record of a tool execution within a sub-agent turn."""
    tool_name: str
    ok: bool
    is_error: bool = False
    output_summary: str = ""  # bounded: capped at 1500 chars
    tool_use_id: str = ""
    input_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "is_error": self.is_error,
            "output_summary": self.output_summary,
            "tool_use_id": self.tool_use_id,
            "input_args": self.input_args,
        }


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
    tool_events: list[SubAgentToolEvent] = field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    changed_files: list[str] = field(default_factory=list)

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
            "tool_events": [e.to_dict() for e in self.tool_events],
            "verification_status": self.verification_status.value,
            "changed_files": self.changed_files,
        }


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    """Attempt to extract structured JSON object from sub-agent final text."""
    if not text or not text.strip():
        return None
    text_stripped = text.strip()
    if text_stripped.startswith("{") and text_stripped.endswith("}"):
        try:
            parsed = json.loads(text_stripped)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block_match:
        try:
            parsed = json.loads(json_block_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

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
    """Execute an isolated sub-agent with strict boundaries."""
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
            verification_status=VerificationStatus.UNVERIFIED,
        )

    # 2. Runtime & MCP Isolation
    child_runtime: dict[str, Any] = dict(config.runtime or {})
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
            verification_status=VerificationStatus.UNVERIFIED,
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
            verification_status=VerificationStatus.UNVERIFIED,
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
            verification_status=VerificationStatus.UNVERIFIED,
        )

    # 5. Permission Sandboxing: based on is_writer, NOT allowed_tools is not None
    if config.is_writer:
        # Writer agent: inherit parent's permission prompt handler
        sub_permissions = PermissionManager(
            cwd,
            prompt=getattr(config.parent_permissions, "prompt", None),
        )
    else:
        # Read-only agent: prompt=None auto-denies writes
        sub_permissions = PermissionManager(cwd, prompt=None)

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
            verification_status=VerificationStatus.UNVERIFIED,
        )

    elapsed = time.time() - start_time

    # 8. Extract Tool Events and Changed Files from Result Messages
    tool_events: list[SubAgentToolEvent] = []
    tool_calls_map: dict[str, dict[str, Any]] = {}
    changed_files: list[str] = []

    for msg in result_messages:
        role = msg.get("role")
        if role == "assistant_tool_call":
            call_id = msg.get("toolUseId") or msg.get("tool_use_id") or msg.get("id") or ""
            tname = msg.get("toolName") or msg.get("tool_name") or msg.get("name") or ""
            tinput = msg.get("input") or msg.get("arguments") or {}
            tool_calls_map[call_id] = {
                "tool_name": tname,
                "input": tinput if isinstance(tinput, dict) else {},
            }
        elif role == "assistant" and "assistant_tool_call" in msg:
            atc = msg["assistant_tool_call"]
            call_id = atc.get("toolUseId") or atc.get("tool_use_id") or atc.get("id") or ""
            tname = atc.get("toolName") or atc.get("tool_name") or atc.get("name") or ""
            tinput = atc.get("input") or atc.get("arguments") or {}
            tool_calls_map[call_id] = {
                "tool_name": tname,
                "input": tinput if isinstance(tinput, dict) else {},
            }
        elif role in {"tool_result", "tool"}:
            call_id = msg.get("toolUseId") or msg.get("tool_use_id") or ""
            call_info = tool_calls_map.get(call_id, {})
            tname = msg.get("toolName") or msg.get("tool_name") or call_info.get("tool_name", "")
            tinput = call_info.get("input", {})
            if "content" in msg and msg["content"] is not None:
                content = str(msg["content"])
                is_err = bool(msg.get("isError", False) or msg.get("is_error", False))
            elif "tool_result" in msg:
                tr = msg["tool_result"]
                if isinstance(tr, dict):
                    content = str(tr.get("output", ""))
                    is_err = not tr.get("ok", True)
                else:
                    content = str(tr)
                    is_err = bool(msg.get("isError", False) or msg.get("is_error", False))
            else:
                content = ""
                is_err = bool(msg.get("isError", False) or msg.get("is_error", False))

            if len(content) > 1500:
                suffix = "... [truncated]"
                summary = content[: 1500 - len(suffix)] + suffix
            else:
                summary = content

            event = SubAgentToolEvent(
                tool_name=tname,
                ok=not is_err,
                is_error=is_err,
                output_summary=summary,
                tool_use_id=call_id,
                input_args=tinput,
            )
            tool_events.append(event)

            if tname in {"write_file", "edit_file", "patch_file"} and not is_err:
                fpath = tinput.get("path") or tinput.get("file_path") or tinput.get("target_file")
                if fpath and str(fpath) not in changed_files:
                    changed_files.append(str(fpath))

    # Evaluate verification status strictly from test_runner tool events
    test_runner_events = [e for e in tool_events if e.tool_name == "test_runner"]
    if not test_runner_events:
        verification_status = VerificationStatus.UNVERIFIED
    else:
        last_test = test_runner_events[-1]
        if last_test.ok and not last_test.is_error:
            verification_status = VerificationStatus.PASS
        else:
            verification_status = VerificationStatus.FAIL

    # 9. Extract Final Message
    final_message = ""
    for msg in reversed(result_messages):
        if msg.get("role") == "assistant" and msg.get("content", "").strip():
            final_message = msg["content"]
            break

    if not final_message:
        final_message = "(sub-agent completed without a final message)"

    tool_calls_count = sum(1 for m in result_messages if m.get("role") == "assistant_tool_call")
    user_messages_count = sum(1 for m in result_messages if m.get("role") == "user")
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
        tool_events=tool_events,
        verification_status=verification_status,
        changed_files=changed_files,
    )
