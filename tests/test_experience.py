"""Unit tests for experience memory extraction, quality gate, and conversion."""
from __future__ import annotations

import pytest

from minicode.execution_trace import ExecutionTrace
from minicode.experience import (
    ExperienceExtractor,
    ExperienceOutcome,
    ExperienceQualityGate,
    ExperienceRecord,
    compute_experience_fingerprint,
    experience_to_memory_entry,
    format_experience_content,
)
from minicode.memory import MemoryScope


def test_experience_extractor_success_verified():
    """Verify deterministic extraction of a successfully verified bug fix."""
    trace = ExecutionTrace()
    trace.record_task_start("t-1", "Fix AttributeError in parser")
    trace.record_tool_call("c-1", "view_file", {"path": "parser.py"})
    trace.record_tool_result("c-1", "view_file", ok=True, output="code")
    trace.record_tool_call("c-2", "replace_file_content", {"TargetFile": "parser.py"})
    trace.record_tool_result("c-2", "replace_file_content", ok=True, output="replaced")
    trace.record_verification("pytest tests/test_parser.py", exit_code=0, passed=True, output="1 passed")
    trace.record_task_end("completed", summary="Fixed AttributeError")

    extractor = ExperienceExtractor()
    record = extractor.extract(trace)

    assert record is not None
    assert record.task_type == "bug_fix"
    assert record.outcome == ExperienceOutcome.SUCCESS_VERIFIED
    assert record.strategy == ["view_file", "replace_file_content"]
    assert record.verification is not None
    assert record.verification.passed is True
    assert record.confidence >= 0.85
    assert len(record.fingerprint) == 16


def test_experience_extractor_factual_root_cause_and_symptom():
    """Verify root cause extraction only captures factual errors without speculation."""
    trace = ExecutionTrace()
    trace.record_task_start("t-2", "Fix failing test in db module")
    trace.record_tool_call("c-1", "run_command", {"CommandLine": "pytest"})
    trace.record_tool_result(
        "c-1",
        "run_command",
        ok=False,
        output="",
        error="Traceback:\n  File 'db.py', line 10\nZeroDivisionError: division by zero",
    )
    trace.record_task_end("failed", summary="Failed to fix")

    extractor = ExperienceExtractor()
    record = extractor.extract(trace)

    assert record is not None
    assert record.outcome == ExperienceOutcome.FAILED_TOOL
    assert "division by zero" in record.symptom
    assert record.root_cause == "ZeroDivisionError: division by zero"


def test_experience_quality_gate():
    """Test quality gate criteria for persistence."""
    gate = ExperienceQualityGate(min_confidence=0.60)
    extractor = ExperienceExtractor()

    # Case 1: Empty trace
    empty_trace = ExecutionTrace()
    rec_empty = extractor.extract(empty_trace)
    assert rec_empty is not None
    can_persist, reason = gate.should_persist(rec_empty, empty_trace)
    assert not can_persist
    assert "no tool executions" in reason

    # Case 2: Aborted task
    abort_trace = ExecutionTrace()
    abort_trace.record_task_start("t-abort", "Some task")
    abort_trace.record_tool_call("c-1", "view_file", {})
    abort_trace.record_tool_result("c-1", "view_file", ok=True, output="ok")
    abort_trace.record_task_end("aborted")
    rec_abort = extractor.extract(abort_trace)
    assert rec_abort is not None
    can_persist, reason = gate.should_persist(rec_abort, abort_trace)
    assert not can_persist
    assert "Aborted" in reason

    # Case 3: Verified success passes gate
    valid_trace = ExecutionTrace()
    valid_trace.record_task_start("t-valid", "Fix issue")
    valid_trace.record_tool_call("c-1", "view_file", {})
    valid_trace.record_tool_result("c-1", "view_file", ok=True, output="ok")
    valid_trace.record_verification("pytest", 0, True, "passed")
    valid_trace.record_task_end("completed")
    rec_valid = extractor.extract(valid_trace)
    assert rec_valid is not None
    can_persist, reason = gate.should_persist(rec_valid, valid_trace)
    assert can_persist
    assert reason == ""


def test_experience_fingerprint_dedup():
    """Verify fingerprints match for normalized equivalent inputs."""
    fp1 = compute_experience_fingerprint(
        task_type="bug_fix",
        symptom="IndexError: list index out of range",
        root_cause="IndexError: list index out of range",
        outcome="success_verified",
    )
    fp2 = compute_experience_fingerprint(
        task_type="bug_fix  ",
        symptom="IndexError:   list index out of range",
        root_cause="indexerror: list index out of range",
        outcome="SUCCESS_VERIFIED",
    )
    assert fp1 == fp2

    # Different outcome results in different fingerprint
    fp3 = compute_experience_fingerprint(
        task_type="bug_fix",
        symptom="IndexError: list index out of range",
        root_cause="IndexError: list index out of range",
        outcome="failed_tool",
    )
    assert fp1 != fp3


def test_experience_to_memory_entry_conversion():
    """Verify conversion from ExperienceRecord to MemoryEntry."""
    record = ExperienceRecord(
        task_id="t-conv",
        task_type="bug_fix",
        outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        symptom="KeyError: 'user'",
        root_cause="KeyError: 'user'",
        strategy=["grep_search", "replace_file_content"],
        confidence=0.92,
        fingerprint="a1b2c3d4e5f60718",
    )
    entry = experience_to_memory_entry(record, scope=MemoryScope.PROJECT)

    assert entry.id == "project-exp-a1b2c3d4e5f60718"
    assert entry.scope == MemoryScope.PROJECT
    assert entry.category == "experience"
    assert "experience" in entry.tags
    assert "bug_fix" in entry.tags
    assert "success_verified" in entry.tags
    assert "metadata" in entry.to_dict()
    assert entry.metadata["fingerprint"] == "a1b2c3d4e5f60718"
    assert entry.metadata["experience"]["outcome"] == "success_verified"
    assert "KeyError" in entry.content
