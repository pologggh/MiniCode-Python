"""Execution trace capture and sanitization for structured experience memory.

Captures tool calls, verification results, assistant thoughts, and lifecycle
events into bounded, secret-redacted execution traces suitable for deterministic
experience extraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
import time
from typing import Any

from minicode.redaction import redact_payload, redact_text
from minicode.release_readiness import redact_sensitive_payload, redact_sensitive_text

# Upper bound on strings stored in traces to avoid memory bloat
MAX_SUMMARY_LENGTH = 1500

_EXTRA_SECRET_PATTERNS = (
    re.compile(
        r"-----BEGIN [A-Z0-9_-]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9_-]+ PRIVATE KEY-----",
        re.MULTILINE,
    ),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(
        r"(?P<key>(?:password|secret|token|api_?key|auth|bearer)[\s:=]+)(?P<value>[^\s\"',;]{8,})",
        re.IGNORECASE,
    ),
)


def sanitize_text(text: str, max_length: int = MAX_SUMMARY_LENGTH) -> str:
    """Sanitize secrets and truncate text to maximum allowed length."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    cleaned = redact_sensitive_text(text)
    for pattern in _EXTRA_SECRET_PATTERNS:
        if "value" in pattern.groupindex:
            cleaned = pattern.sub(r"\g<key>[REDACTED]", cleaned)
        else:
            cleaned = pattern.sub("[REDACTED]", cleaned)

    if len(cleaned) > max_length:
        cleaned = cleaned[: max_length - 17] + "... [truncated]"

    return cleaned


def sanitize_payload(value: Any, max_length: int = MAX_SUMMARY_LENGTH) -> Any:
    """Recursively redact secrets and truncate string values in payloads."""
    redacted = redact_sensitive_payload(value)
    if isinstance(redacted, dict):
        return {
            str(k): sanitize_payload(v, max_length=max_length)
            for k, v in redacted.items()
        }
    elif isinstance(redacted, (list, tuple)):
        return [sanitize_payload(item, max_length=max_length) for item in redacted]
    elif isinstance(redacted, str):
        return sanitize_text(redacted, max_length=max_length)
    return redacted



class TraceEventType(str, Enum):
    """Types of events recorded in an execution trace."""
    TASK_START = "task_start"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    VERIFICATION = "verification"
    TASK_END = "task_end"


@dataclass
class TraceEvent:
    """A single discrete event within an execution trace."""
    timestamp: float
    event_type: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceEvent":
        return cls(
            timestamp=data.get("timestamp", time.time()),
            event_type=data.get("event_type", "unknown"),
            details=data.get("details", {}) if isinstance(data.get("details"), dict) else {},
        )


@dataclass
class ExecutionTrace:
    """Execution trace of an agent turn or complete task execution."""
    task_id: str = ""
    task_description: str = ""
    workspace: str = ""
    events: list[TraceEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: str = "running"
    verification_results: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def record_task_start(
        self,
        task_id: str,
        description: str,
        workspace: str = "",
    ) -> None:
        """Record the initiation of a task."""
        self.task_id = task_id
        self.task_description = sanitize_text(description)
        self.workspace = workspace
        self.start_time = time.time()
        self.status = "running"

        self.events.append(
            TraceEvent(
                timestamp=self.start_time,
                event_type=TraceEventType.TASK_START.value,
                details={
                    "task_id": self.task_id,
                    "description": self.task_description,
                    "workspace": self.workspace,
                },
            )
        )

    def record_assistant(self, content: str) -> None:
        """Record assistant message or thinking snippet (deduplicating adjacent identical messages)."""
        sanitized = sanitize_text(content)
        if not sanitized:
            return
        if self.events and self.events[-1].event_type == TraceEventType.ASSISTANT.value:
            if self.events[-1].details.get("content") == sanitized:
                return  # Skip duplicate adjacent assistant message
        self.events.append(
            TraceEvent(
                timestamp=time.time(),
                event_type=TraceEventType.ASSISTANT.value,
                details={"content": sanitized},
            )
        )

    def record_tool_call(
        self,
        call_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> None:
        """Record an attempted tool invocation."""
        sanitized_args = sanitize_payload(args)
        self.events.append(
            TraceEvent(
                timestamp=time.time(),
                event_type=TraceEventType.TOOL_CALL.value,
                details={
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "args": sanitized_args,
                },
            )
        )

    def record_tool_result(
        self,
        call_id: str,
        tool_name: str,
        ok: bool,
        output: str,
        error: str = "",
        exit_code: int | None = None,
    ) -> None:
        """Record the outcome of a tool execution."""
        sanitized_output = sanitize_text(output)
        sanitized_error = sanitize_text(error)
        self.events.append(
            TraceEvent(
                timestamp=time.time(),
                event_type=TraceEventType.TOOL_RESULT.value,
                details={
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "ok": ok,
                    "output": sanitized_output,
                    "error": sanitized_error,
                    "exit_code": exit_code,
                },
            )
        )

    def record_verification(
        self,
        command: str,
        exit_code: int,
        passed: bool,
        output: str = "",
    ) -> None:
        """Record an explicit verification gate check (e.g. pytest or bash test run)."""
        sanitized_output = sanitize_text(output)
        res = {
            "command": sanitize_text(command, max_length=200),
            "exit_code": exit_code,
            "passed": passed,
            "output": sanitized_output,
            "timestamp": time.time(),
        }
        self.verification_results.append(res)
        self.events.append(
            TraceEvent(
                timestamp=time.time(),
                event_type=TraceEventType.VERIFICATION.value,
                details=res,
            )
        )

    def record_task_end(
        self,
        status: str,
        summary: str = "",
    ) -> None:
        """Record task finalization."""
        self.end_time = time.time()
        self.status = status
        sanitized_summary = sanitize_text(summary)
        self.events.append(
            TraceEvent(
                timestamp=self.end_time,
                event_type=TraceEventType.TASK_END.value,
                details={
                    "status": status,
                    "summary": sanitized_summary,
                    "duration": self.end_time - self.start_time,
                },
            )
        )

    def get_tool_sequence(self) -> list[str]:
        """Return the sequence of tool names called during this trace."""
        seq: list[str] = []
        for ev in self.events:
            if ev.event_type == TraceEventType.TOOL_CALL.value:
                tool_name = ev.details.get("tool_name")
                if tool_name and isinstance(tool_name, str):
                    seq.append(tool_name)
        return seq

    def get_errors(self) -> list[str]:
        """Collect error snippets from failed tool executions and verifications."""
        errors: list[str] = []
        for ev in self.events:
            if ev.event_type == TraceEventType.TOOL_RESULT.value:
                if not ev.details.get("ok", True):
                    err = ev.details.get("error") or ev.details.get("output", "")
                    if err:
                        errors.append(str(err))
            elif ev.event_type == TraceEventType.VERIFICATION.value:
                if not ev.details.get("passed", True):
                    output = ev.details.get("output", "")
                    if output:
                        errors.append(str(output))
        return errors

    def has_successful_verification(self) -> bool:
        """Check if any verification gate explicitly passed."""
        return any(v.get("passed") is True for v in self.verification_results)

    def to_reflection_trace(self) -> list[dict[str, Any]]:
        """Convert trace to legacy ReflectionEngine compatible trace format."""
        legacy: list[dict[str, Any]] = []
        for ev in self.events:
            if ev.event_type == TraceEventType.TOOL_CALL.value:
                tool_name = str(ev.details.get("tool_name", ""))
                args = ev.details.get("args", {})
                legacy.append({
                    "type": "tool_call",
                    "name": tool_name,
                    "toolName": tool_name,
                    "input": args if isinstance(args, dict) else {},
                    "call_id": ev.details.get("call_id", ""),
                })
            elif ev.event_type == TraceEventType.TOOL_RESULT.value:
                ok = ev.details.get("ok", True)
                err = ev.details.get("error") or ""
                out = ev.details.get("output") or ""
                if not ok or err:
                    legacy.append({
                        "type": "error",
                        "content": err or out,
                        "tool_name": ev.details.get("tool_name", ""),
                        "call_id": ev.details.get("call_id", ""),
                    })
            elif ev.event_type == TraceEventType.ASSISTANT.value:
                content = ev.details.get("content", "")
                if content:
                    legacy.append({
                        "type": "assistant",
                        "content": content,
                    })
            elif ev.event_type == TraceEventType.VERIFICATION.value:
                passed = ev.details.get("passed", True)
                if not passed:
                    out = ev.details.get("output", "")
                    cmd = ev.details.get("command", "")
                    legacy.append({
                        "type": "error",
                        "content": out or f"Verification failed: {cmd}",
                        "tool_name": "verification",
                        "command": cmd,
                    })
        return legacy

    def get_files_read(self) -> list[str]:
        """Extract sanitized list of unique file paths read or searched."""
        read_tools = {
            "view_file", "read_file", "grep_search", "find_by_name",
            "list_dir", "read_url_content",
        }
        files: set[str] = set()
        for ev in self.events:
            if ev.event_type == TraceEventType.TOOL_CALL.value:
                tool_name = ev.details.get("tool_name", "")
                if tool_name in read_tools:
                    args = ev.details.get("args", {})
                    extracted = _extract_file_paths_from_args(args)
                    for f in extracted:
                        if _is_safe_path(f):
                            files.add(f)
        return sorted(files)

    def get_files_touched(self) -> list[str]:
        """Extract sanitized list of unique file paths written, edited, or modified."""
        write_tools = {
            "write_to_file", "replace_file_content", "edit_file",
            "apply_patch", "patch_file",
        }
        files: set[str] = set()
        for ev in self.events:
            if ev.event_type == TraceEventType.TOOL_CALL.value:
                tool_name = ev.details.get("tool_name", "")
                if tool_name in write_tools:
                    args = ev.details.get("args", {})
                    extracted = _extract_file_paths_from_args(args)
                    for f in extracted:
                        if _is_safe_path(f):
                            files.add(f)
        return sorted(files)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Return events as a list of raw dictionaries."""
        return [ev.to_dict() for ev in self.events]

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete execution trace to dictionary."""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "workspace": self.workspace,
            "events": self.to_dict_list(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "verification_results": self.verification_results,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionTrace":
        """Deserialize execution trace from dictionary."""
        events_data = data.get("events", [])
        events = [
            TraceEvent.from_dict(ev)
            for ev in events_data
            if isinstance(ev, dict)
        ]
        return cls(
            task_id=data.get("task_id", ""),
            task_description=data.get("task_description", ""),
            workspace=data.get("workspace", ""),
            events=events,
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time"),
            status=data.get("status", "unknown"),
            verification_results=data.get("verification_results", []),
            metrics=data.get("metrics", {}),
        )


def _extract_file_paths_from_args(args: Any) -> list[str]:
    """Helper to extract file path strings from tool arguments."""
    paths: list[str] = []
    if not isinstance(args, dict):
        return paths

    candidate_keys = (
        "path", "TargetFile", "AbsolutePath", "SearchPath",
        "SearchDirectory", "DirectoryPath", "file_path", "filePath",
    )
    for key in candidate_keys:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            paths.append(val.strip())

    return paths


_SENSITIVE_PATH_PATTERNS = re.compile(
    r"(?:\.env|\.git|id_rsa|id_ed25519|credentials|\.pem|\.key|\.p12|\.pfx)\b",
    re.I,
)


def _is_safe_path(path: str) -> bool:
    """Validate path does not point to secret files or credentials."""
    if not path or not isinstance(path, str):
        return False
    # Reject secret patterns
    if _SENSITIVE_PATH_PATTERNS.search(path):
        return False
    return True


_VERIFICATION_COMMAND_PATTERNS = (
    re.compile(r"^\s*(?:python\d?(?:\.\d+)?\s+-m\s+)?pytest\b", re.I),
    re.compile(r"^\s*python\d?(?:\.\d+)?\s+-m\s+unittest\b", re.I),
    re.compile(r"^\s*(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b", re.I),
    re.compile(r"^\s*cargo\s+test\b", re.I),
    re.compile(r"^\s*go\s+test\b", re.I),
    re.compile(r"^\s*ctest\b", re.I),
    re.compile(r"^\s*(?:mvn|gradle|gradlew)\s+test\b", re.I),
)

_NON_VERIFICATION_PREFIXES = re.compile(
    r"^\s*(?:grep|cat|echo|git|find|ls|dir|head|tail|sed|awk|which|where|touch)\b",
    re.I,
)


def is_explicit_verification_command(cmd: str) -> bool:
    """Check if command is an explicit test or verification command.

    Rejects commands that merely contain 'test' as substring or argument (e.g. grep test, echo test, git checkout test).
    """
    if not isinstance(cmd, str):
        return False
    cmd_clean = cmd.strip()
    if not cmd_clean:
        return False
    if _NON_VERIFICATION_PREFIXES.match(cmd_clean):
        return False
    return any(p.search(cmd_clean) for p in _VERIFICATION_COMMAND_PATTERNS)

