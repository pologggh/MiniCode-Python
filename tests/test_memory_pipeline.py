"""Unit tests for MemoryPipeline structured experience writing, deduplication, and feedback."""
from __future__ import annotations

from pathlib import Path
import pytest

from minicode.experience import (
    ExperienceOutcome,
    ExperienceRecord,
    VerificationEvidence,
)
from minicode.memory import MemoryManager, MemoryScope
from minicode.memory_pipeline import MemoryPipeline


def test_memory_pipeline_write_experience_and_deduplication(tmp_path: Path):
    """Test MemoryPipeline persists experiences and deduplicates identical fingerprints."""
    mgr = MemoryManager(workspace=tmp_path)
    pipeline = MemoryPipeline(memory_manager=mgr)
    pipeline.initialize(workspace_path=str(tmp_path), enable_reranker=False)

    rec = ExperienceRecord(
        task_id="t-100",
        task_type="bug_fix",
        outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        symptom="TypeError: 'NoneType' object is not callable",
        root_cause="TypeError: 'NoneType' object is not callable",
        strategy=["view_file", "replace_file_content"],
        verification=VerificationEvidence("pytest tests/test_core.py", 0, True, "passed"),
        confidence=0.92,
        fingerprint="dedup_hash_12345",
    )

    # First write: creates new entry
    mem_id = pipeline.write_experience(rec)
    assert mem_id is not None
    assert pipeline.metrics.persisted_count == 1
    assert pipeline.metrics.dedup_count == 0

    entries = mgr.memories[MemoryScope.PROJECT].entries
    assert len(entries) == 1
    first_entry = entries[0]
    assert first_entry.metadata["fingerprint"] == "dedup_hash_12345"
    assert first_entry.usage_count == 0

    # Second write: identical fingerprint should deduplicate
    mem_id2 = pipeline.write_experience(rec)
    assert mem_id2 == mem_id
    assert pipeline.metrics.persisted_count == 1
    assert pipeline.metrics.dedup_count == 1
    assert len(mgr.memories[MemoryScope.PROJECT].entries) == 1
    assert mgr.memories[MemoryScope.PROJECT].entries[0].usage_count == 1


def test_memory_pipeline_feedback_loop(tmp_path: Path):
    """Test MemoryPipeline.feedback reinforces useful memories and penalizes failing ones."""
    mgr = MemoryManager(workspace=tmp_path)
    entry = mgr.add_entry(
        scope=MemoryScope.PROJECT,
        category="experience",
        content="Fixing recursion error",
        tags=["fix"],
    )
    assert entry.usage_count == 0

    pipeline = MemoryPipeline(memory_manager=mgr)
    pipeline.initialize(workspace_path=str(tmp_path), enable_reranker=False)

    # Positive feedback
    pipeline.feedback(task_success=True, injected_memory_ids=[entry.id])
    assert entry.usage_count == 2

    # Negative feedback
    pipeline.feedback(task_success=False, injected_memory_ids=[entry.id])
    assert entry.usage_count == 1
