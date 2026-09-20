"""Unit tests for MemoryEntry metadata persistence and validation."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from minicode.memory import (
    MemoryEntry,
    MemoryManager,
    MemoryPaths,
    MemoryScope,
    _validate_entry,
    _validate_memory_data,
)


def test_memory_entry_metadata_defaults_and_post_init():
    """Test default metadata and non-dict coercion."""
    entry = MemoryEntry(
        id="test-1",
        scope=MemoryScope.USER,
        category="experience",
        content="Test content",
    )
    assert entry.metadata == {}

    entry_none = MemoryEntry(
        id="test-2",
        scope=MemoryScope.USER,
        category="experience",
        content="Test content",
        metadata=None,  # type: ignore
    )
    assert entry_none.metadata == {}


def test_memory_entry_to_dict_and_from_dict_roundtrip():
    """Test metadata serialization and deserialization roundtrip."""
    meta = {
        "experience": {
            "task_type": "debugging",
            "symptom": "IndexError",
            "root_cause": "Off-by-one error",
            "confidence": 0.95,
        }
    }
    entry = MemoryEntry(
        id="test-roundtrip",
        scope=MemoryScope.PROJECT,
        category="experience",
        content="Fix off-by-one error in parser",
        tags=["debugging", "parser"],
        metadata=meta,
    )
    d = entry.to_dict()
    assert "metadata" in d
    assert d["metadata"] == meta

    restored = MemoryEntry.from_dict(d)
    assert restored.id == entry.id
    assert restored.metadata == meta
    assert restored.metadata["experience"]["confidence"] == 0.95


def test_memory_entry_from_dict_legacy_compatibility():
    """Legacy entries without 'metadata' field should deserialize cleanly with empty dict."""
    legacy_data = {
        "id": "legacy-1",
        "scope": "project",
        "category": "architecture",
        "content": "Use layered architecture",
        "tags": ["arch"],
    }
    entry = MemoryEntry.from_dict(legacy_data)
    assert entry.metadata == {}

    # If metadata is malformed in dict (not a dict), it safely becomes {}
    malformed_data = {
        "id": "legacy-2",
        "scope": "project",
        "category": "architecture",
        "content": "Use layered architecture",
        "metadata": "not-a-dict",
    }
    entry2 = MemoryEntry.from_dict(malformed_data)
    assert entry2.metadata == {}


def test_validate_entry_metadata():
    """Test schema validation for entry metadata."""
    valid_entry = {
        "id": "v-1",
        "content": "Valid",
        "metadata": {"key": "value"},
    }
    is_valid, errors = _validate_entry(valid_entry, 0)
    assert is_valid
    assert len(errors) == 0

    invalid_entry = {
        "id": "v-2",
        "content": "Invalid",
        "metadata": "invalid_type",
    }
    is_valid, errors = _validate_entry(invalid_entry, 0)
    assert not is_valid
    assert any("metadata" in err for err in errors)


def test_memory_manager_add_entry_persists_metadata(tmp_path: Path):
    """Test MemoryManager saves and reloads entries with metadata."""
    mgr = MemoryManager(workspace=tmp_path)

    meta = {"source": "execution_trace", "status": "verified"}
    entry = mgr.add_entry(
        scope=MemoryScope.PROJECT,
        category="experience",
        content="Verified debugging strategy",
        tags=["fix", "debug"],
        metadata=meta,
    )
    assert entry.metadata == meta

    # Reload from disk
    mgr2 = MemoryManager(workspace=tmp_path)
    loaded_entries = mgr2.memories[MemoryScope.PROJECT].entries
    assert len(loaded_entries) == 1
    assert loaded_entries[0].id == entry.id
    assert loaded_entries[0].metadata == meta
