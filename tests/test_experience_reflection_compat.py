"""Compatibility tests for ReflectionEngine with ExecutionTrace and legacy trace formats."""
from __future__ import annotations

from pathlib import Path
import pytest

from minicode.agent_reflection import ReflectionEngine, ReflectionResult
from minicode.execution_trace import ExecutionTrace, TraceEventType
from minicode.memory import MemoryManager, MemoryScope


def test_reflection_engine_with_execution_trace(tmp_path: Path):
    """Test ReflectionEngine accepts ExecutionTrace directly and extracts context properly."""
    mem_mgr = MemoryManager(workspace=tmp_path)
    engine = ReflectionEngine(memory_manager=mem_mgr, min_confidence_threshold=0.5)

    trace = ExecutionTrace()
    trace.record_task_start("task-001", "Fix database connection leak in auth service", workspace=str(tmp_path))
    trace.record_tool_call("call-1", "view_file", {"path": "auth/db.py"})
    trace.record_tool_result("call-1", "view_file", ok=True, output="class ConnectionPool: ...")
    trace.record_assistant("I will fix the connection leak by adding pool.close() on timeout.")
    trace.record_tool_call("call-2", "replace_file_content", {"TargetFile": "auth/db.py", "Replacement": "pool.close()"})
    trace.record_tool_result("call-2", "replace_file_content", ok=True, output="replaced")
    trace.record_verification("pytest tests/test_db.py", exit_code=0, passed=True, output="1 passed in 0.05s")
    trace.record_task_end(status="completed", summary="Fixed connection leak")

    result = engine.reflect(
        task_description="Fix database connection leak in auth service",
        execution_trace=trace,
        persist=True,
    )

    assert isinstance(result, ReflectionResult)
    assert result.success is True
    assert len(result.errors_encountered) == 0
    assert result.task_context is not None
    # Files touched/read should be captured
    assert "auth/db.py" in result.task_context.get("files", [])
    assert "view_file" in result.task_context.get("tools", [])
    assert "replace_file_content" in result.task_context.get("tools", [])


def test_reflection_engine_with_trace_errors(tmp_path: Path):
    """Test ReflectionEngine identifies errors from ExecutionTrace tool results and verifications."""
    engine = ReflectionEngine(memory_manager=None)

    trace = ExecutionTrace()
    trace.record_task_start("task-002", "Run test suite", workspace=str(tmp_path))
    trace.record_tool_call("call-1", "run_command", {"CommandLine": "pytest tests/"})
    trace.record_tool_result("call-1", "run_command", ok=False, output="AssertionError: 1 != 2", error="Test assertion failed")
    trace.record_verification("pytest tests/", exit_code=1, passed=False, output="FAILED tests/test_foo.py")
    trace.record_assistant("The tests failed with assertion error.")
    trace.record_task_end(status="failed", summary="Tests failed")

    result = engine.reflect(
        task_description="Run test suite",
        execution_trace=trace,
        persist=False,
    )

    assert result.success is False
    assert len(result.errors_encountered) >= 1
    error_text = " ".join(result.errors_encountered)
    assert "assertion failed" in error_text.lower() or "failed" in error_text.lower()


def test_reflection_engine_legacy_and_dict_compatibility():
    """Test ReflectionEngine handles legacy type dicts and modern event_type dicts."""
    engine = ReflectionEngine(memory_manager=None)

    # Legacy format
    legacy_trace = [
        {"type": "tool_call", "name": "read_file", "input": {"path": "server.py"}},
        {"type": "assistant", "content": "Server file inspected"},
    ]
    res_legacy = engine.reflect("Legacy test", legacy_trace, persist=False)
    assert res_legacy.success is True
    assert "server.py" in res_legacy.task_context.get("files", [])

    # Modern TraceEvent dict format
    modern_trace_dicts = [
        {"event_type": "tool_call", "details": {"tool_name": "grep_search", "args": {"SearchPath": "api/router.py"}}},
        {"event_type": "assistant", "details": {"content": "Found routes"}},
    ]
    res_modern = engine.reflect("Modern dict test", modern_trace_dicts, persist=False)
    assert res_modern.success is True
    assert "grep_search" in res_modern.task_context.get("tools", [])
