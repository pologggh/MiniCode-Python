"""Unit tests for execution trace capture and sanitization."""
from __future__ import annotations

import pytest

from minicode.execution_trace import (
    ExecutionTrace,
    MAX_SUMMARY_LENGTH,
    TraceEvent,
    TraceEventType,
    sanitize_payload,
    sanitize_text,
)


def test_sanitize_text_redacts_secrets():
    """Verify secrets like API keys, bearer tokens, and private keys are redacted."""
    raw_api_key = "export OPENAI_API_KEY=sk-1234567890abcdef12345678"
    assert "sk-1234567890abcdef12345678" not in sanitize_text(raw_api_key)
    assert "[REDACTED]" in sanitize_text(raw_api_key)

    raw_bearer = "Authorization: Bearer secret_token_1234567890"
    assert "secret_token_1234567890" not in sanitize_text(raw_bearer)
    assert "[REDACTED]" in sanitize_text(raw_bearer)

    raw_ssh_key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0m4wz...\n"
        "-----END RSA PRIVATE KEY-----"
    )
    assert "MIIEowIBAAKCAQEA0m4wz" not in sanitize_text(raw_ssh_key)
    assert "[REDACTED]" in sanitize_text(raw_ssh_key)


def test_sanitize_text_bounds_length():
    """Verify text exceeding MAX_SUMMARY_LENGTH is bounded."""
    huge_text = "x" * (MAX_SUMMARY_LENGTH + 500)
    sanitized = sanitize_text(huge_text)
    assert len(sanitized) <= MAX_SUMMARY_LENGTH
    assert sanitized.endswith("... [truncated]")


def test_sanitize_payload():
    """Verify recursive payload sanitization for dictionaries and lists."""
    payload = {
        "api_key": "sk-1234567890abcdef",
        "nested": {
            "token": "secret_abc1234567",
            "safe": "normal value",
        },
        "items": [
            "password: mysecretpassword123",
            "normal string",
        ],
    }
    sanitized = sanitize_payload(payload)
    assert "sk-1234567890abcdef" not in str(sanitized)
    assert "secret_abc1234567" not in str(sanitized)
    assert "mysecretpassword123" not in str(sanitized)
    assert sanitized["nested"]["safe"] == "normal value"
    assert sanitized["items"][1] == "normal string"


def test_execution_trace_lifecycle_and_helpers():
    """Test full trace recording workflow and helper queries."""
    trace = ExecutionTrace()
    trace.record_task_start("task-101", "Fix test_runner bug", workspace="/tmp/ws")
    trace.record_assistant("I will view the file and run tests.")
    trace.record_tool_call("call-1", "view_file", {"path": "test_runner.py"})
    trace.record_tool_result("call-1", "view_file", ok=True, output="def run(): pass")
    trace.record_tool_call("call-2", "run_command", {"CommandLine": "pytest"})
    trace.record_tool_result("call-2", "run_command", ok=False, output="", error="AssertionError: 1 != 2")
    trace.record_verification("pytest", exit_code=1, passed=False, output="FAILED test_runner")
    trace.record_verification("pytest", exit_code=0, passed=True, output="PASSED test_runner")
    trace.record_task_end("completed", summary="Fixed bug and verified tests")

    assert trace.task_id == "task-101"
    assert trace.status == "completed"
    assert trace.get_tool_sequence() == ["view_file", "run_command"]
    errors = trace.get_errors()
    assert len(errors) == 2
    assert "AssertionError" in errors[0]
    assert "FAILED test_runner" in errors[1]
    assert trace.has_successful_verification() is True


def test_execution_trace_serialization_roundtrip():
    """Test serializing to dict and restoring from dict."""
    trace = ExecutionTrace()
    trace.record_task_start("task-serialization", "Testing roundtrip")
    trace.record_tool_call("call-1", "grep_search", {"Query": "test"})
    trace.record_tool_result("call-1", "grep_search", ok=True, output="found 1")
    trace.record_task_end("completed", summary="Finished successfully")

    d = trace.to_dict()
    restored = ExecutionTrace.from_dict(d)

    assert restored.task_id == "task-serialization"
    assert restored.task_description == "Testing roundtrip"
    assert restored.status == "completed"
    assert len(restored.events) == 4
    assert restored.get_tool_sequence() == ["grep_search"]
