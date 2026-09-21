"""Unit tests for MemoryInjector experience retrieval and formatting."""
from __future__ import annotations

from pathlib import Path
import pytest

from minicode.experience import (
    ExperienceOutcome,
    ExperienceRecord,
    experience_to_memory_entry,
)
from minicode.memory import MemoryManager, MemoryScope
from minicode.memory_injector import InjectedMemory, MemoryInjector


def test_injected_memory_memory_id_and_formatting(tmp_path: Path):
    """Test that InjectedMemory preserves memory_id and formats experience categories."""
    mgr = MemoryManager(workspace=tmp_path)

    # 1. Add verified experience entry
    rec_success = ExperienceRecord(
        task_id="t-1",
        task_type="bug_fix",
        outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        symptom="ValueError: invalid literal",
        root_cause="ValueError: invalid literal",
        strategy=["view_file", "replace_file_content"],
        confidence=0.9,
        fingerprint="fp11111111111111",
    )
    entry1 = experience_to_memory_entry(rec_success, scope=MemoryScope.PROJECT)
    mgr.memories[MemoryScope.PROJECT].add_entry(entry1)
    mgr._save_scope(MemoryScope.PROJECT)

    # 2. Add failed experience entry
    rec_fail = ExperienceRecord(
        task_id="t-2",
        task_type="bug_fix",
        outcome=ExperienceOutcome.FAILED_TOOL,
        symptom="PermissionError: access denied",
        root_cause="PermissionError: access denied",
        strategy=["run_command"],
        confidence=0.8,
        fingerprint="fp22222222222222",
    )
    entry2 = experience_to_memory_entry(rec_fail, scope=MemoryScope.PROJECT)
    mgr.memories[MemoryScope.PROJECT].add_entry(entry2)
    mgr._save_scope(MemoryScope.PROJECT)

    injector = MemoryInjector(memory_manager=mgr, min_relevance=0.1)

    # Test inject_for_task preserves memory_id
    injected = injector.inject_for_task("Fix ValueError bug in parser")
    assert len(injected) > 0
    match_entry1 = next((m for m in injected if m.memory_id == entry1.id), None)
    assert match_entry1 is not None
    assert match_entry1.memory_id == entry1.id

    # Test format_for_prompt produces [Verified Experience]
    formatted = injector.format_for_prompt([match_entry1])
    assert "[Verified Experience]" in formatted

    # Test inject_on_failure preserves memory_id and format produces [Past Failure Pattern]
    fail_injected = injector.inject_on_failure(
        error_message="PermissionError: access denied",
        tool_name="run_command",
    )
    assert len(fail_injected) > 0
    match_fail = next((m for m in fail_injected if m.memory_id == entry2.id), None)
    assert match_fail is not None
    formatted_fail = injector.format_for_prompt([match_fail])
    assert "[Past Failure Pattern]" in formatted_fail


def test_tag_bypass_prevention_for_failure_experience(tmp_path: Path):
    """Verify that tag-based injection does not leak failure experiences into normal tasks."""
    mgr = MemoryManager(workspace=tmp_path)
    rec_fail = ExperienceRecord(
        task_id="t-fail-api",
        task_type="bug_fix",
        outcome=ExperienceOutcome.FAILED_VERIFICATION,
        symptom="api validation failed",
        root_cause="invalid schema in endpoint",
        strategy=["run_command"],
        confidence=0.8,
        fingerprint="fp_fail_api_12345",
    )
    entry_fail = experience_to_memory_entry(rec_fail, scope=MemoryScope.PROJECT)
    entry_fail.tags.append("api")
    mgr.memories[MemoryScope.PROJECT].add_entry(entry_fail)
    mgr._save_scope(MemoryScope.PROJECT)

    injector = MemoryInjector(memory_manager=mgr, min_relevance=0.1)

    # 1. Normal task injection: failure experience must NOT enter
    injected = injector.inject_for_task("implement api endpoint")
    injected_ids = [m.memory_id for m in injected]
    assert entry_fail.id not in injected_ids
    formatted = injector.format_for_prompt(injected)
    assert "[Past Failure Pattern]" not in formatted

    # 2. Failure recovery: failure experience CAN be retrieved
    recovered = injector.inject_on_failure(
        error_message="api validation failed",
        tool_name="run_command",
    )
    recovered_ids = [m.memory_id for m in recovered]
    assert entry_fail.id in recovered_ids
    formatted_recovered = injector.format_for_prompt(recovered)
    assert "[Past Failure Pattern]" in formatted_recovered


def test_verified_vs_failure_ranking_and_prompt_labels(tmp_path: Path):
    """Verify ranking and prompt labels when both verified and failure memories match."""
    mgr = MemoryManager(workspace=tmp_path)

    # Memory A: SUCCESS_VERIFIED
    rec_a = ExperienceRecord(
        task_id="t-fastapi-verified",
        task_type="bug_fix",
        outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        task_description="FastAPI validation fix",
        symptom="RequestValidationError 422",
        root_cause="missing pydantic validator",
        strategy=["replace_file_content"],
        confidence=0.95,
        fingerprint="fp_fastapi_verified_111",
    )
    entry_a = experience_to_memory_entry(rec_a, scope=MemoryScope.PROJECT)
    mgr.memories[MemoryScope.PROJECT].add_entry(entry_a)

    # Memory B: FAILED_VERIFICATION
    rec_b = ExperienceRecord(
        task_id="t-fastapi-failed",
        task_type="bug_fix",
        outcome=ExperienceOutcome.FAILED_VERIFICATION,
        task_description="FastAPI validation fix",
        symptom="RequestValidationError 422",
        root_cause="wrong validator signature",
        strategy=["replace_file_content"],
        confidence=0.8,
        fingerprint="fp_fastapi_failed_222",
    )
    entry_b = experience_to_memory_entry(rec_b, scope=MemoryScope.PROJECT)
    mgr.memories[MemoryScope.PROJECT].add_entry(entry_b)
    mgr._save_scope(MemoryScope.PROJECT)

    injector = MemoryInjector(memory_manager=mgr, min_relevance=0.1)

    # Normal task: "fix fastapi validation error"
    normal_injected = injector.inject_for_task("fix fastapi validation error")
    normal_ids = [m.memory_id for m in normal_injected]
    assert entry_a.id in normal_ids
    assert entry_b.id not in normal_ids
    normal_prompt = injector.format_for_prompt(normal_injected)
    assert "[Verified Experience]" in normal_prompt
    assert "[Past Failure Pattern]" not in normal_prompt

    # Error recovery: both can enter as distinct semantic roles
    fail_injected = injector.inject_on_failure(
        error_message="RequestValidationError: validation error",
        tool_name="pytest",
    )
    fail_ids = [m.memory_id for m in fail_injected]
    assert entry_a.id in fail_ids
    assert entry_b.id in fail_ids

    prompt_recovery = injector.format_for_prompt(fail_injected)
    assert "[Verified Experience]" in prompt_recovery
    assert "[Past Failure Pattern]" in prompt_recovery


def test_success_unverified_ranking_and_label(tmp_path: Path):
    """Verify that SUCCESS_UNVERIFIED ranks below verified and receives [Unverified Experience]."""
    mgr = MemoryManager(workspace=tmp_path)

    # Memory A: SUCCESS_VERIFIED
    rec_verified = ExperienceRecord(
        task_id="t-django-v",
        task_type="feature_impl",
        outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        task_description="Django auth middleware update",
        symptom="",
        root_cause="",
        strategy=["replace_file_content"],
        confidence=0.9,
        fingerprint="fp_django_v_111",
    )
    entry_v = experience_to_memory_entry(rec_verified, scope=MemoryScope.PROJECT)
    mgr.memories[MemoryScope.PROJECT].add_entry(entry_v)

    # Memory B: SUCCESS_UNVERIFIED
    rec_unverified = ExperienceRecord(
        task_id="t-django-u",
        task_type="feature_impl",
        outcome=ExperienceOutcome.SUCCESS_UNVERIFIED,
        task_description="Django auth middleware update",
        symptom="",
        root_cause="",
        strategy=["replace_file_content"],
        confidence=0.7,
        fingerprint="fp_django_u_222",
    )
    entry_u = experience_to_memory_entry(rec_unverified, scope=MemoryScope.PROJECT)
    mgr.memories[MemoryScope.PROJECT].add_entry(entry_u)
    mgr._save_scope(MemoryScope.PROJECT)

    injector = MemoryInjector(memory_manager=mgr, min_relevance=0.1)

    # Check relevance scores
    score_v = injector._calculate_relevance(entry_v, "Django auth middleware update", None)
    score_u = injector._calculate_relevance(entry_u, "Django auth middleware update", None)
    assert score_v > score_u

    # Normal injection
    injected = injector.inject_for_task("Django auth middleware update")
    assert len(injected) >= 2
    # First candidate should be the verified one
    assert injected[0].memory_id == entry_v.id
    assert injected[1].memory_id == entry_u.u if hasattr(entry_u, "u") else entry_u.id

    formatted = injector.format_for_prompt(injected)
    assert "[Verified Experience]" in formatted
    assert "[Unverified Experience]" in formatted


def test_retrieval_and_injection_metrics_increment(tmp_path: Path):
    """Verify that experience_retrieval_count and experience_injection_count increment."""
    from minicode.experience import ExperienceMemoryMetrics

    mgr = MemoryManager(workspace=tmp_path)
    rec = ExperienceRecord(
        task_id="t-metrics",
        task_type="bug_fix",
        outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        task_description="Fix index out of range in tokenizer",
        symptom="IndexError: string index out of range",
        root_cause="bounds check missing",
        strategy=["replace_file_content"],
        confidence=0.9,
        fingerprint="fp_metrics_333",
    )
    entry = experience_to_memory_entry(rec, scope=MemoryScope.PROJECT)
    mgr.memories[MemoryScope.PROJECT].add_entry(entry)
    mgr._save_scope(MemoryScope.PROJECT)

    metrics = ExperienceMemoryMetrics()
    injector = MemoryInjector(memory_manager=mgr, min_relevance=0.1, metrics=metrics)

    assert metrics.experience_retrieval_count == 0
    assert metrics.experience_injection_count == 0

    injected = injector.inject_for_task("Fix index out of range in tokenizer")
    assert len(injected) > 0
    assert metrics.experience_retrieval_count >= 1
    assert metrics.experience_injection_count >= 1
