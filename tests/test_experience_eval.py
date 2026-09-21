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
    assert results["secret_leak_rate"] == 0.0
    assert results["dedup_count"] >= 1
    assert results["rejected_count"] >= 2
    assert results["task_type_accuracy"] == 1.0
    assert results["outcome_accuracy"] == 1.0
    assert results["quality_gate_accuracy"] == 1.0
    assert results["recall_at_1"] == 1.0
    assert results["recall_at_3"] == 1.0
    assert results["failure_recall_at_k"] == 1.0
    assert results["normal_failure_leakage_rate"] == 0.0
    assert results["negative_transfer_rate"] == 0.0
    assert results["verified_precision_at_3"] == 1.0
    assert results["verified_precision_at_k"] >= 0.7
    assert results["dedup_precision"] == 1.0
    assert results["metadata_preservation_rate"] == 1.0
    assert results["feedback_accuracy"] == 1.0
    assert results["recall_verified_experience"] is True
    assert results["recall_failure_pattern"] is True
    assert results["feedback_loop_accurate"] is True
