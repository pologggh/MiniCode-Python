"""Integration test running the structured experience memory benchmark."""
from __future__ import annotations

from pathlib import Path
import pytest

from benchmarks.experience_memory_eval import run_experience_benchmark


def test_structured_experience_memory_benchmark(tmp_path: Path):
    """Run full experience evaluation and assert strict requirements pass."""
    results = run_experience_benchmark(workspace_path=tmp_path)

    assert results["extraction_rate"] == 1.0
    assert results["secret_leaks"] == 0
    assert results["dedup_count"] >= 1
    assert results["rejected_count"] >= 2  # empty and aborted tasks
    assert results["recall_verified_experience"] is True
    assert results["recall_failure_pattern"] is True
    assert results["feedback_loop_accurate"] is True
    assert results["quality_gate_accuracy"] == 1.0
