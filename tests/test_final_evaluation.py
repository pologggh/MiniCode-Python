from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import pytest

from benchmarks.final_eval.fixtures import (
    COMMON_RUNTIME_TASKS,
    EXPERIENCE_FIXTURES,
    SECURITY_EVAL_FIXTURES,
    SKILL_EVAL_QUERIES,
    generate_context_message_stream,
    generate_skill_catalogs,
)
from benchmarks.final_eval.metrics import (
    Comparability,
    MetricDirection,
    MetricRecord,
    calculate_median,
    compute_metric,
)
from benchmarks.final_eval.worker import detect_capabilities
from benchmarks.final_evaluation import (
    ADAPTIVE_COMMIT,
    BASELINE_COMMIT,
    create_worktree,
    remove_worktree,
    verify_git_provenance,
)


def test_git_provenance():
    """Verify baseline and adaptive Git commits and merge parentage."""
    repo_dir = str(Path.cwd())
    provenance = verify_git_provenance(repo_dir)

    assert provenance["baseline_commit"] == BASELINE_COMMIT
    assert provenance["adaptive_commit"] == ADAPTIVE_COMMIT
    assert provenance["provenance_verified"] is True
    assert provenance["phase1_parent_baseline"].startswith(BASELINE_COMMIT)


def test_adaptive_capability_detection():
    """Verify that current adaptive branch detects all 5 subsystems."""
    caps = detect_capabilities()
    assert caps["skill_router"] is True
    assert caps["experience_memory"] is True
    assert caps["context_budget"] is True
    assert caps["agent_team"] is True
    assert caps["security_policy"] is True


def test_metric_direction_and_deltas():
    """Verify metric delta calculations, directionality, and zero-division safety."""
    # 1. Higher is better: improvement when adaptive > baseline
    rec_higher = compute_metric(
        name="precision",
        category="Test",
        description="Precision metric",
        comparability=Comparability.DIRECT,
        direction=MetricDirection.HIGHER_IS_BETTER,
        unit="rate",
        baseline_val=0.50,
        adaptive_val=0.80,
    )
    assert rec_higher.absolute_delta == 0.3
    assert rec_higher.relative_delta == 60.0
    assert rec_higher.improved is True

    # 2. Lower is better: improvement when adaptive < baseline (e.g. token drop)
    rec_lower = compute_metric(
        name="tokens",
        category="Test",
        description="Prompt tokens",
        comparability=Comparability.DIRECT,
        direction=MetricDirection.LOWER_IS_BETTER,
        unit="tokens",
        baseline_val=1000,
        adaptive_val=200,
    )
    assert rec_lower.absolute_delta == -800.0
    assert rec_lower.relative_delta == -80.0
    assert rec_lower.improved is True

    # 3. Zero-division safety: when baseline is 0, relative_delta must be None (N/A)
    rec_zero = compute_metric(
        name="zero_base",
        category="Test",
        description="Zero baseline",
        comparability=Comparability.DIRECT,
        direction=MetricDirection.HIGHER_IS_BETTER,
        unit="count",
        baseline_val=0,
        adaptive_val=5,
    )
    assert rec_zero.absolute_delta == 5.0
    assert rec_zero.relative_delta is None  # Safe divide-by-zero handling

    # 4. Adaptive-only comparability
    rec_adapt_only = compute_metric(
        name="feature_x",
        category="Test",
        description="Adaptive only capability",
        comparability=Comparability.ADAPTIVE_ONLY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        unit="boolean",
        baseline_val="UNSUPPORTED",
        adaptive_val=True,
    )
    assert rec_adapt_only.comparability == "ADAPTIVE_ONLY"
    assert rec_adapt_only.absolute_delta is None
    assert rec_adapt_only.relative_delta is None

    # 5. Median calculation
    assert calculate_median([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0
    assert calculate_median([10.0, 20.0]) == 15.0


def test_deterministic_fixtures_repeatability():
    """Verify that fixture generation is deterministic and reproducible."""
    cat1 = generate_skill_catalogs(seed=42)
    cat2 = generate_skill_catalogs(seed=42)

    assert len(cat1[10]) == 10
    assert len(cat1[100]) == 100
    assert len(cat1[500]) == 500

    for sz in [10, 100, 500]:
        for s1, s2 in zip(cat1[sz], cat2[sz]):
            assert s1.name == s2.name
            assert s1.description == s2.description

    stream1 = generate_context_message_stream()
    stream2 = generate_context_message_stream()
    assert len(stream1) == len(stream2)
    assert stream1[1]["content"] == stream2[1]["content"]


def test_worktree_lifecycle_and_cleanup():
    """Verify that temporary worktrees are created and cleanly destroyed."""
    repo_dir = str(Path.cwd())
    wt = create_worktree(repo_dir, ADAPTIVE_COMMIT, prefix="test_wt_cleanup_")
    try:
        assert wt.exists()
        # Verify it shows in git worktree list
        res = subprocess.run(["git", "worktree", "list"], cwd=repo_dir, capture_output=True, text=True)
        assert str(wt).replace("\\", "/") in res.stdout.replace("\\", "/")
    finally:
        remove_worktree(repo_dir, wt)

    assert not wt.exists()
    # Confirm it was pruned from git worktree list
    res_after = subprocess.run(["git", "worktree", "list"], cwd=repo_dir, capture_output=True, text=True)
    assert str(wt).replace("\\", "/") not in res_after.stdout.replace("\\", "/")


def test_worker_output_schema():
    """Verify that worker process executes and outputs expected JSON schema."""
    import time
    repo_dir = str(Path.cwd())
    tmp_out = Path(tempfile.gettempdir()) / f"schema_test_{int(time.time() * 1000)}.json"
    worker_script = Path(repo_dir) / "benchmarks" / "final_eval_worker.py"

    cmd = [
        sys.executable,
        str(worker_script),
        "--repo-dir", repo_dir,
        "--output", str(tmp_out),
        "--category", "skill",
    ]
    res = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    assert res.returncode == 0

    try:
        assert tmp_out.exists()
        with open(tmp_out, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "commit_sha" in data
        assert "python_version" in data
        assert "platform" in data
        assert "timestamp" in data
        assert "capabilities" in data
        assert "categories" in data
        assert "skill_routing" in data["categories"]
    finally:
        if tmp_out.exists():
            tmp_out.unlink()


def test_generated_artifact_integrity():
    """Verify that generated evaluation files exist and have non-empty valid content."""
    json_path = Path("benchmarks/final_evaluation_results.json")
    md_path = Path("benchmarks/final_evaluation_results.md")
    resume_path = Path("benchmarks/resume_metrics_candidates.md")

    assert json_path.exists()
    assert md_path.exists()
    assert resume_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    assert "provenance" in json_data
    assert "metrics" in json_data
    assert len(json_data["metrics"]) >= 20

    md_content = md_path.read_text(encoding="utf-8")
    assert "# Adaptive MiniCode Final Evaluation" in md_content
    assert "fd9bf63" in md_content
    assert "1a07358" in md_content
    assert "Category Matrix" in md_content
    assert "LIVE_EVAL_NOT_RUN" in md_content

    resume_content = resume_path.read_text(encoding="utf-8")
    assert "# Verifiable Resume Metrics Candidates" in resume_content
    assert "Skill Prompt Token Reduction" in resume_content
    assert "Negative Transfer" in resume_content
