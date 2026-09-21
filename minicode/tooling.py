from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from minicode.logging_config import get_logger, log_tool_execution


# ---------------------------------------------------------------------------
# Constants for smart truncation
# ---------------------------------------------------------------------------

# Default max output size (characters) — per tool type
_DEFAULT_MAX_OUTPUT = 30_000       # ~8K tokens, safe for context
_LARGE_OUTPUT_THRESHOLD = 50_000   # Trigger smart truncation above this

# Tool-specific output limits (characters)
_TOOL_OUTPUT_LIMITS: dict[str, int] = {
    "read_file": 40_000,
    "grep_files": 20_000,
    "run_command": 30_000,
    "web_fetch": 20_000,
    "web_search": 15_000,
    "list_files": 15_000,
    "file_tree": 15_000,
    "code_review": 20_000,
    "diff_viewer": 20_000,
    "test_runner": 25_000,
}


def _smart_truncate_output(output: str, tool_name: str, max_chars: int | None = None) -> str:
    """Intelligently truncate large tool output to preserve context window.
    
    Strategy:
    1. If output fits within limit, return as-is
    2. For file reads: keep head + tail (beginning and end of file)
    3. For command output: keep head + tail + error lines
    4. For grep/search: keep first N matches + summary
    5. Generic: keep head + tail with line count summary
    """
    if not output:
        return output
    
    limit = max_chars or _TOOL_OUTPUT_LIMITS.get(tool_name, _DEFAULT_MAX_OUTPUT)
    
    if len(output) <= limit:
        return output
    
    lines = output.split("\n")
    total_lines = len(lines)
    total_chars = len(output)
    
    # Calculate how many lines we can keep (rough estimate)
    avg_line_len = total_chars / max(1, total_lines)
    max_lines = int(limit / max(40, avg_line_len))
    
    if tool_name == "read_file":
        # Keep head + tail — most important for understanding file structure
        head_lines = max(1, int(max_lines * 0.6))
        tail_lines = max(1, max_lines - head_lines)
        head = "\n".join(lines[:head_lines])
        tail = "\n".join(lines[-tail_lines:])
        omitted = total_lines - head_lines - tail_lines
        return (
            f"{head}\n"
            f"\n... [{omitted} lines omitted (output too large: {total_chars:,} chars)] ...\n\n"
            f"{tail}"
        )
    
    if tool_name == "run_command":
        # Keep head + error lines + tail
        head_lines = max(1, int(max_lines * 0.4))
        tail_lines = max(1, int(max_lines * 0.4))
        
        # Also extract error/warning lines
        error_pattern = re.compile(r'(?i)(error|fail|exception|traceback|warning)', re.IGNORECASE)
        error_lines = [
            (i, line) for i, line in enumerate(lines)
            if error_pattern.search(line) and head_lines <= i < total_lines - tail_lines
        ]
        error_text = ""
        if error_lines:
            error_text = "\n\n[Key errors/warnings from omitted section:]\n" + "\n".join(
                f"L{i+1}: {line[:200]}" for i, line in error_lines[:20]
            )
        
        head = "\n".join(lines[:head_lines])
        tail = "\n".join(lines[-tail_lines:])
        omitted = total_lines - head_lines - tail_lines
        return (
            f"{head}\n"
            f"\n... [{omitted} lines omitted (output too large: {total_chars:,} chars)] ...{error_text}\n\n"
            f"{tail}"
        )
    
    if tool_name in ("grep_files", "web_search"):
        # Keep first N matches + summary
        head = "\n".join(lines[:max_lines])
        omitted = total_lines - max_lines
        return (
            f"{head}\n"
            f"\n... [{omitted} more lines omitted (output too large: {total_chars:,} chars, {total_lines} total lines)] ..."
        )
    
    # Generic: head + tail
    head_lines = max(1, int(max_lines * 0.5))
    tail_lines = max(1, max_lines - head_lines)
    head = "\n".join(lines[:head_lines])
    tail = "\n".join(lines[-tail_lines:])
    omitted = total_lines - head_lines - tail_lines
    return (
        f"{head}\n"
        f"\n... [{omitted} lines omitted (output too large: {total_chars:,} chars)] ...\n\n"
        f"{tail}"
    )


# ---------------------------------------------------------------------------
# Tool metadata (inspired by Claude Code's Tool type)
# ---------------------------------------------------------------------------

class ToolCapability(str, Enum):
    """Tool capability flags."""
    READ_ONLY = "read_only"
    DESTRUCTIVE = "destructive"
    CONCURRENCY_SAFE = "concurrency_safe"
    REQUIRES_PERMISSION = "requires_permission"


@dataclass
class ToolMetadata:
    """Tool metadata for classification and discovery.
    
    Inspired by Claude Code's Tool type definition.
    """
    name: str
    description: str
    capabilities: set[ToolCapability] = field(default_factory=set)
    input_schema: dict[str, Any] = field(default_factory=dict)
    is_enabled: bool = True
    max_result_size_chars: int = 10_000
    tags: list[str] = field(default_factory=list)
    
    @property
    def is_read_only(self) -> bool:
        """Check if tool is read-only."""
        return ToolCapability.READ_ONLY in self.capabilities
    
    @property
    def is_destructive(self) -> bool:
        """Check if tool can modify/delete data."""
        return ToolCapability.DESTRUCTIVE in self.capabilities
    
    @property
    def is_concurrency_safe(self) -> bool:
        """Check if tool is safe for concurrent execution."""
        return ToolCapability.CONCURRENCY_SAFE in self.capabilities


# ---------------------------------------------------------------------------
# Tool Protocol (inspired by Claude Code's Tool interface)
# ---------------------------------------------------------------------------

class Tool(Protocol):
    """Tool protocol defining a complete tool lifecycle.
    
    Inspired by Claude Code's Tool type which includes:
    - call: Execution logic
    - description: Dynamic description generation
    - validate_input: Input validation
    - check_permissions: Permission checking
    - Metadata: is_read_only, is_destructive, etc.
    """
    
    @property
    def name(self) -> str: ...
    
    @property
    def description_template(self) -> str: ...
    
    def get_description(self, args: dict[str, Any], options: dict[str, Any] | None = None) -> str: ...
    def validate_input(self, args: dict[str, Any]) -> tuple[bool, str]: ...
    def check_permissions(self, args: dict[str, Any], context: ToolContext) -> tuple[bool, str]: ...
    def call(
        self,
        args: dict[str, Any],
        context: ToolContext,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> ToolResult: ...
    def is_enabled(self) -> bool: ...
    def is_read_only(self, args: dict[str, Any]) -> bool: ...
    def is_destructive(self, args: dict[str, Any]) -> bool: ...


@dataclass(slots=True)
class BackgroundTaskResult:
    taskId: str
    type: str
    command: str
    pid: int
    status: str
    startedAt: int


@dataclass
class ToolResult:
    ok: bool
    output: str
    backgroundTask: BackgroundTaskResult | None = None
    awaitUser: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)



@dataclass(slots=True)
class ToolContext:
    cwd: str
    permissions: Any | None = None
    session: Any | None = None
    _runtime: dict | None = None


Validator = Callable[[Any], Any]
Runner = Callable[[Any, ToolContext], ToolResult]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    validator: Validator
    run: Runner
    metadata: ToolMetadata | None = None
    
    @property
    def is_read_only(self) -> bool:
        """Check if this tool is read-only (safe for concurrent execution)."""
        if self.metadata:
            return self.metadata.is_read_only
        # Fallback: heuristic based on tool name
        return self.name in _READ_ONLY_TOOL_NAMES
    
    @property
    def is_concurrency_safe(self) -> bool:
        """Check if this tool is safe for concurrent execution."""
        if self.metadata:
            return self.metadata.is_concurrency_safe or self.metadata.is_read_only
        return self.is_read_only


# Heuristic: tool names that are known to be read-only
_READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "list_files", "grep_files", "file_tree",
    "find_symbols", "find_references", "get_ast_info",
    "code_review", "diff_viewer",
    "web_fetch", "web_search",
    "ask_user", "todo_write",
})


class ToolRegistry:
    def __init__(
        self,
        tools: list[ToolDefinition],
        skills: list[dict[str, Any]] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        disposer: Callable[[], Any] | None = None,
        security_policy: Any | None = None,
        security_audit: Any | None = None,
    ) -> None:
        self._tools = tools
        self._skills = skills or []
        self._mcp_servers = mcp_servers or []
        self._disposer = disposer
        self.security_policy = security_policy
        self.security_audit = security_audit
        # 工具查找缓存 - O(1) 查找代替 O(n) 遍历
        self._tool_index: dict[str, ToolDefinition] = {t.name: t for t in tools}

    def list(self) -> list[ToolDefinition]:
        return list(self._tools)

    def list_all(self) -> list[str]:
        return list(self._tool_index.keys())

    def get_skills(self) -> list[dict[str, Any]]:
        return list(self._skills)

    def get_mcp_servers(self) -> list[dict[str, Any]]:
        return list(self._mcp_servers)

    def find(self, name: str) -> ToolDefinition | None:
        # O(1) lookup via cached index
        return self._tool_index.get(name)

    def execute(self, tool_name: str, input_data: Any, context: ToolContext) -> ToolResult:
        """Execute a tool with comprehensive error protection and centralized security policy."""
        tool = self.find(tool_name)
        if tool is None:
            return ToolResult(ok=False, output=f"Unknown tool: {tool_name}")

        _logger = get_logger("tools")
        _start = time.monotonic()
        try:
            # Phase 1: Input validation (with error context)
            try:
                parsed = tool.validator(input_data)
            except (ValueError, TypeError, KeyError) as ve:
                log_tool_execution(
                    tool_name, False, (time.monotonic() - _start) * 1000,
                    error=f"input validation: {ve}",
                )
                return ToolResult(
                    ok=False,
                    output=f"Input validation error in {tool_name}: {ve}\n"
                           f"Input was: {str(input_data)[:200]}"
                )

            # Phase 1.5: Centralized Security Pre-Check
            assessment = None
            actor_str = "PARENT"
            role = "parent"
            if self.security_policy is not None:
                from minicode.security_policy import SecurityActor, SecurityDecision, SecurityRequest, ApprovalRoute
                from minicode.auto_mode import PermissionMode

                runtime = getattr(context, "_runtime", None) or {}
                actor_str = str(runtime.get("_security_actor", "parent")).upper()
                actor = SecurityActor.CHILD if actor_str == "CHILD" else SecurityActor.PARENT
                role = str(runtime.get("_security_role", "parent"))
                depth = int(runtime.get("_security_depth", 0))
                untrusted_seen = bool(runtime.get("_security_untrusted_seen", False))

                perm_mode = getattr(context.permissions, "auto_checker", None)
                mode = getattr(perm_mode, "mode", PermissionMode.DEFAULT)

                sec_req = SecurityRequest(
                    tool_name=tool_name,
                    input_data=parsed,
                    cwd=context.cwd,
                    actor=actor,
                    agent_role=role,
                    agent_depth=depth,
                    permission_mode=mode,
                    is_mcp=tool_name.startswith("mcp__"),
                    untrusted_context_seen=untrusted_seen,
                )
                assessment = self.security_policy.evaluate(sec_req)

                if assessment.decision == SecurityDecision.DENY:
                    reason_msg = "; ".join(assessment.reasons) or "Action blocked by security policy"
                    denial_res = ToolResult(
                        ok=False,
                        output=f"Security policy denied tool '{tool_name}': {reason_msg}",
                        metadata={"security_decision": "DENY", "reasons": assessment.reasons, "rule_ids": assessment.rule_ids},
                    )
                    if self.security_audit:
                        self.security_audit.record_event(
                            session_id=str(getattr(context, "session", "") or ""),
                            actor=actor_str,
                            agent_role=role,
                            tool_name=tool_name,
                            decision=assessment.decision.value,
                            risk=assessment.risk.value,
                            rule_ids=assessment.rule_ids,
                            reasons=assessment.reasons,
                            input_data=parsed,
                            result_ok=False,
                            output=denial_res.output,
                        )
                    return denial_res

                if assessment.decision == SecurityDecision.ASK and assessment.approval_route == ApprovalRoute.GENERIC_TOOL:
                    if context.permissions is None:
                        fail_closed_res = ToolResult(
                            ok=False,
                            output=f"Security approval unavailable for '{tool_name}': permission manager missing",
                            metadata={"security_decision": "DENY", "reasons": ["permission_manager_missing"]},
                        )
                        if self.security_audit:
                            self.security_audit.record_event(
                                session_id=str(getattr(context, "session", "") or ""),
                                actor=actor_str,
                                agent_role=role,
                                tool_name=tool_name,
                                decision="DENY",
                                risk=assessment.risk.value,
                                rule_ids=["missing_permission_fail_closed"],
                                reasons=["permission manager missing"],
                                input_data=parsed,
                                result_ok=False,
                                output=fail_closed_res.output,
                            )
                        return fail_closed_res

                    try:
                        context.permissions.ensure_tool_action(
                            tool_name=tool_name,
                            scope=str(parsed)[:200],
                            summary=f"Approval requested for tool '{tool_name}'",
                            details=[f"tool: {tool_name}", f"reasons: {'; '.join(assessment.reasons)}"],
                        )
                    except RuntimeError as perm_err:
                        perm_denied_res = ToolResult(
                            ok=False,
                            output=f"Tool execution denied: {perm_err}",
                            metadata={"security_decision": "DENIED", "reasons": [str(perm_err)]},
                        )
                        if self.security_audit:
                            self.security_audit.record_event(
                                session_id=str(getattr(context, "session", "") or ""),
                                actor=actor_str,
                                agent_role=role,
                                tool_name=tool_name,
                                decision="ASK",
                                risk=assessment.risk.value,
                                rule_ids=assessment.rule_ids,
                                reasons=[str(perm_err)],
                                input_data=parsed,
                                result_ok=False,
                                output=perm_denied_res.output,
                            )
                        return perm_denied_res

            # Phase 2: Execution (with crash protection)
            result = tool.run(parsed, context)

            # Phase 3: Output sanitization & post-execution security inspection
            if result.output is None:
                result.output = ""

            # Check sensitive path reads -> redact secrets
            if assessment and assessment.sensitive_paths:
                from minicode.redaction import redact_text
                result.output = redact_text(result.output)
                if self.security_policy:
                    self.security_policy.metrics.sensitive_output_redactions += 1

            # Check external output -> scan for injection, wrap content, set taint
            if assessment and (assessment.is_external or assessment.output_trust.value == "UNTRUSTED_EXTERNAL"):
                from minicode.untrusted_content import scan_and_wrap_external_content
                runtime = getattr(context, "_runtime", None)
                wrapped_out, scan_res = scan_and_wrap_external_content(result.output, source=tool_name)
                result.output = wrapped_out
                result.metadata.update({
                    "trust_level": "untrusted_external",
                    "source": tool_name,
                    "injection_detected": scan_res.detected,
                })
                if scan_res.detected and runtime is not None:
                    runtime["_security_untrusted_seen"] = True
                    if self.security_policy:
                        self.security_policy.metrics.injection_detections += 1

            # Record audit event
            if self.security_audit:
                self.security_audit.record_event(
                    session_id=str(getattr(context, "session", "") or ""),
                    actor=actor_str,
                    agent_role=role,
                    tool_name=tool_name,
                    decision=assessment.decision.value if assessment else "ALLOW",
                    risk=assessment.risk.value if assessment else "SAFE",
                    rule_ids=assessment.rule_ids if assessment else [],
                    reasons=assessment.reasons if assessment else [],
                    input_data=parsed,
                    result_ok=bool(result.ok),
                    output=result.output,
                    untrusted_output=bool(assessment.is_external) if assessment else False,
                    injection_detected=bool(result.metadata.get("injection_detected", False)),
                )

            # Smart truncation for large outputs
            if result.output and len(result.output) > _LARGE_OUTPUT_THRESHOLD:
                result.output = _smart_truncate_output(result.output, tool_name)

            log_tool_execution(
                tool_name, bool(result.ok), (time.monotonic() - _start) * 1000,
                error=None if result.ok else (result.output or "")[:200],
            )
            return result


        except (KeyboardInterrupt, SystemExit):
            # These should always propagate upward
            raise
        except Exception as error:  # noqa: BLE001
            # Global safety net: convert any unhandled exception to error result
            # This prevents a single buggy tool from crashing the entire session
            duration_ms = (time.monotonic() - _start) * 1000
            # Persist the crash to the log file (searchable) while still
            # returning a ToolResult to the caller (issue #5).
            _logger.exception("Tool %s crashed", tool_name)
            log_tool_execution(tool_name, False, duration_ms, error=str(error))
            import traceback
            tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
            # Include last 5 lines of traceback for debugging
            tb_excerpt = "".join(tb_lines[-5:]).strip()
            error_type = type(error).__name__

            return ToolResult(
                ok=False,
                output=f"[{error_type}] Tool {tool_name} crashed: {error}\n"
                       f"Traceback (most recent):\n{tb_excerpt}"
            )

    def dispose(self) -> None:
        if self._disposer is not None:
            self._disposer()
