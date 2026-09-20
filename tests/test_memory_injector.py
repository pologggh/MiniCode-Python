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
