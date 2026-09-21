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


def test_artifact_symlink_escape_rejection(tmp_path: Path):
    store = ContextArtifactStore(workspace=tmp_path)
    store_dir = tmp_path / ".mini-code-tool-results"
    store_dir.mkdir(parents=True, exist_ok=True)

    # Sibling file outside store dir
    sibling_file = tmp_path / "sibling_secret.txt"
    sibling_file.write_text("SUPER_SECRET_EXTERNAL_CONTENT", encoding="utf-8")

    symlink_target_id = "ctx_1111222233334444"
    symlink_path = store_dir / f"{symlink_target_id}.txt"

    try:
        symlink_path.symlink_to(sibling_file)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks not supported in this test environment: {exc}")

    # Access must be rejected by Path containment check
    with pytest.raises(ValueError, match="Symlink escape attempt detected"):
        store.read(symlink_target_id)


def test_load_context_artifact_metrics_wiring(tmp_path: Path):
    from minicode.context_budget import ContextBudgetMetrics

    store = ContextArtifactStore(workspace=tmp_path)
    meta = store.persist("Traced exception line", tool_name="pytest")
    metrics = ContextBudgetMetrics()

    tool = create_load_context_artifact_tool(str(tmp_path), store=store, metrics=metrics)

    # Success increments artifact_recovery_count
    res1 = tool.run({"artifact_id": meta.artifact_id}, None)
    assert res1.ok is True
    assert metrics.artifact_recovery_count == 1
    assert metrics.recovery_failures == 0

    # Failure increments recovery_failures
    res2 = tool.run({"artifact_id": "ctx_nonexistent12345"}, None)
    assert res2.ok is False
    assert metrics.artifact_recovery_count == 1
    assert metrics.recovery_failures == 1


def test_artifact_store_root_symlink_escape_rejected(tmp_path: Path):
    """Verify that if .mini-code-tool-results is a symlink pointing outside workspace, it is rejected."""
    ws = tmp_path / "workspace"
    outside = tmp_path / "sibling_outside_dir"
    ws.mkdir()
    outside.mkdir()

    symlink_store = ws / ".mini-code-tool-results"
    try:
        symlink_store.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks not supported in this test environment: {exc}")

    # Initialization must reject the escaped store root
    with pytest.raises(ValueError, match="points outside workspace"):
        ContextArtifactStore(workspace=ws)

    # If store was created before symlink was placed:
    ws2 = tmp_path / "workspace2"
    ws2.mkdir()
    store2 = ContextArtifactStore(workspace=ws2)

    symlink_store2 = ws2 / ".mini-code-tool-results"
    try:
        symlink_store2.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks not supported in this test environment: {exc}")

    # persist, read, read_range must all reject
    with pytest.raises(ValueError, match="points outside workspace"):
        store2.persist("Attempt write outside workspace")

    with pytest.raises(ValueError, match="points outside workspace"):
        store2.read("ctx_0123456789abcdef")

    with pytest.raises(ValueError, match="points outside workspace"):
        store2.read_range("ctx_0123456789abcdef")


def test_active_artifact_cleanup_protection(tmp_path: Path):
    """Verify cleanup(retention_days=0) protects active artifacts while deleting unreferenced ones."""
    import os
    import time

    store = ContextArtifactStore(workspace=tmp_path)
    meta1 = store.persist("Active artifact content that must be preserved", tool_name="grep")
    meta2 = store.persist("Unreferenced old artifact content that should be pruned", tool_name="read_file")

    # Set both mtimes back in the past
    old_time = time.time() - 100_000
    for aid in (meta1.artifact_id, meta2.artifact_id):
        c_p, m_p = store._get_paths(aid)
        os.utime(c_p, (old_time, old_time))
        os.utime(m_p, (old_time, old_time))

    current_messages = [
        {"role": "system", "content": "You are assistant."},
        {
            "role": "tool_result",
            "content": f"[Context Artifact]\nid: {meta1.artifact_id}\npreview...",
            "_context_artifact_id": meta1.artifact_id,
        },
    ]

    active_ids = {
        str(m.get("_context_artifact_id"))
        for m in current_messages
        if m.get("_context_artifact_id")
    }
    assert meta1.artifact_id in active_ids

    deleted = store.cleanup(retention_days=0, active_artifact_ids=active_ids)
    assert deleted == 1

    # Active artifact must remain intact and readable
    assert store.exists(meta1.artifact_id) is True
    assert store.read(meta1.artifact_id) == "Active artifact content that must be preserved"

    # Unreferenced artifact must be deleted
    assert store.exists(meta2.artifact_id) is False

