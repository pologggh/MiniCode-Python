"""Tests for ContextArtifactStore and load_context_artifact tool."""
from __future__ import annotations

from pathlib import Path
import pytest

from minicode.context_artifacts import ContextArtifactStore
from minicode.tools.load_context_artifact import create_load_context_artifact_tool


def test_artifact_persist_and_read(tmp_path: Path):
    store = ContextArtifactStore(workspace=tmp_path)
    content = "Hello world, this is a test artifact content."
    meta = store.persist(content=content, tool_name="test_tool")

    assert meta.artifact_id.startswith("ctx_")
    assert len(meta.artifact_id) == 20  # ctx_ + 16 chars
    assert meta.tool_name == "test_tool"
    assert meta.original_chars == len(content)
    assert meta.estimated_tokens > 0
    assert store.exists(meta.artifact_id)

    # Read full
    full = store.read(meta.artifact_id)
    assert full == content

    # Read metadata
    saved_meta = store.get_metadata(meta.artifact_id)
    assert saved_meta is not None
    assert saved_meta.sha256 == meta.sha256


def test_artifact_secret_redaction_in_preview(tmp_path: Path):
    store = ContextArtifactStore(workspace=tmp_path)
    sensitive_content = (
        "Authorization: Bearer my-secret-token-1234567890\n"
        "AWS_KEY=AKIAIOSFODNN7EXAMPLE\n"
        "Normal log output follows here."
    )
    meta = store.persist(content=sensitive_content, tool_name="curl")

    # Full content is intact for exact tool recovery
    assert store.read(meta.artifact_id) == sensitive_content

    # Preview must be redacted
    assert "my-secret-token" not in meta.preview
    assert "[REDACTED" in meta.preview or "REDACTED" in meta.preview


def test_artifact_bounded_range_read(tmp_path: Path):
    store = ContextArtifactStore(workspace=tmp_path)
    huge_content = "0123456789" * 1500  # 15,000 chars
    meta = store.persist(huge_content, tool_name="cat")

    # Read slice with offset
    slice_content, slice_meta = store.read_range(meta.artifact_id, offset=10, limit=20)
    assert slice_content == huge_content[10:30]
    assert slice_meta.artifact_id == meta.artifact_id

    # Recovery boundedness: limit > 8000 must be clamped to 8000
    huge_slice, _ = store.read_range(meta.artifact_id, offset=0, limit=20000)
    assert len(huge_slice) == 8000


def test_artifact_path_traversal_rejection(tmp_path: Path):
    store = ContextArtifactStore(workspace=tmp_path)

    bad_ids = [
        "../etc/passwd",
        "ctx_../../secret",
        "ctx_1234/5678",
        "ctx_1234\\5678",
        "ctx_not_hex_chars!",
        "ctx_short",
    ]

    for bad in bad_ids:
        with pytest.raises(ValueError):
            store.read(bad)

        with pytest.raises(ValueError):
            store.get_metadata(bad)

        with pytest.raises(ValueError):
            store.read_range(bad)


def test_artifact_cleanup_respects_active_references(tmp_path: Path):
    store = ContextArtifactStore(workspace=tmp_path)

    m1 = store.persist("Content 1", tool_name="t1")
    m2 = store.persist("Content 2", tool_name="t2")

    # Cleanup with max_artifacts=1, protecting m1
    deleted = store.cleanup(
        retention_days=0,
        max_artifacts=1,
        active_artifact_ids={m1.artifact_id},
    )

    # m2 was deleted, m1 was protected
    assert store.exists(m1.artifact_id)
    assert not store.exists(m2.artifact_id)
    assert deleted == 1


def test_load_context_artifact_tool(tmp_path: Path):
    store = ContextArtifactStore(workspace=tmp_path)
    meta = store.persist("Important error trace evidence line 100", tool_name="pytest")

    tool = create_load_context_artifact_tool(str(tmp_path), store=store)

    # Valid run
    res = tool.run({"artifact_id": meta.artifact_id, "offset": 0, "limit": 1000}, None)
    assert res.ok is True
    assert meta.artifact_id in res.output
    assert "Important error trace" in res.output

    # Missing artifact run
    missing_res = tool.run({"artifact_id": "ctx_0123456789abcdef"}, None)
    assert missing_res.ok is False
    assert "not found" in missing_res.output.lower()

    # Tool validator checks
    with pytest.raises(ValueError):
        tool.validator({"artifact_id": "../traversal"})


def test_context_budget_apply_with_artifact_store(tmp_path: Path):
    from minicode.context_budget import ContextBudgetConfig, ContextBudgetManager

    store = ContextArtifactStore(workspace=tmp_path)
    mgr = ContextBudgetManager(config=ContextBudgetConfig(offload_threshold_tokens=50))

    big_output = "Log output line from test runner\n" * 200
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "tool_result", "toolName": "pytest", "content": big_output},
        {"role": "user", "content": "fix errors"},
    ]

    modified, plan = mgr.plan_and_apply(
        messages=messages,
        artifact_store=store,
        available_budget=100,
    )

    assert plan.offload_count == 1
    # Check that tool_result was offloaded to artifact store
    tool_msg = modified[1]
    assert tool_msg.get("_context_action") == "offload"
    artifact_id = tool_msg.get("_context_artifact_id")
    assert artifact_id is not None
    assert artifact_id.startswith("ctx_")
    assert "[Context Artifact]" in tool_msg["content"]
    assert "Use load_context_artifact" in tool_msg["content"]

    # Recovery via tool
    tool = create_load_context_artifact_tool(str(tmp_path), store=store)
    recovered = tool.run({"artifact_id": artifact_id}, None)
    assert recovered.ok is True
    assert "Log output line from test runner" in recovered.output
