from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import pytest

from benchmarks.final_eval.fixtures import (
    COMMON_RUNTIME_TASKS,
    EXPERIENCE_FIXTURES,
    MEMORY_EVAL_QUERIES,
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
from benchmarks.final_eval.worker import (
    detect_capabilities,
    execute_tool_compat,
    run_common_runtime_tasks,
    run_context_runtime_benchmark,
    run_experience_memory_benchmark,
    run_multi_agent_benchmark,
    run_security_benchmark,
    run_skill_routing_benchmark,
    setup_worker_environment,
)
from benchmarks.final_evaluation import (
    ADAPTIVE_COMMIT,
    BASELINE_COMMIT,
    build_comparison_matrix,
    create_worktree,
    generate_markdown_report,
    generate_resume_metrics,
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
        res = subprocess.run(["git", "worktree", "list"], cwd=repo_dir, capture_output=True, text=True)
        assert str(wt).replace("\\", "/") in res.stdout.replace("\\", "/")
    finally:
        remove_worktree(repo_dir, wt)

    assert not wt.exists()
    res_after = subprocess.run(["git", "worktree", "list"], cwd=repo_dir, capture_output=True, text=True)
    assert str(wt).replace("\\", "/") not in res_after.stdout.replace("\\", "/")


def test_worker_output_schema_and_path_assertion():
    """Verify that worker process executes, records loaded_minicode_path, and outputs valid schema."""
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
        assert "loaded_minicode_path" in data
        assert Path(data["loaded_minicode_path"]).resolve().is_relative_to(Path(repo_dir).resolve())
        assert "python_version" in data
        assert "platform" in data
        assert "timestamp" in data
        assert "capabilities" in data
        assert "categories" in data
        assert "skill_routing" in data["categories"]
    finally:
        if tmp_out.exists():
            tmp_out.unlink()


def test_execute_tool_compat():
    """Verify execute_tool_compat handles both .ok and .success ToolResults."""
    class MockResultOk:
        def __init__(self, ok: bool, output: str):
            self.ok = ok
            self.output = output

    class MockResultSuccess:
        def __init__(self, success: bool, output: str):
            self.success = success
            self.output = output

    class MockTools:
        def execute(self, tool_name, input_data, context):
            if tool_name == "tool_ok":
                return MockResultOk(True, "output_ok")
            return MockResultSuccess(False, "output_err")

    tools = MockTools()
    ok1, out1 = execute_tool_compat(tools, "tool_ok", {}, None)
    assert ok1 is True
    assert out1 == "output_ok"

    ok2, out2 = execute_tool_compat(tools, "tool_fail", {}, None)
    assert ok2 is False
    assert out2 == "output_err"


def test_runtime_tasks_execution_and_oracles():
    """Verify that run_common_runtime_tasks executes real agent turns and validates oracles."""
    caps = detect_capabilities()
    res = run_common_runtime_tasks(caps, str(Path.cwd()))

    assert res["all_tasks_completed"] is True
    assert res["total_tool_calls"] == 5
    assert len(res["task_details"]) == 5

    for td in res["task_details"]:
        assert td["completed"] is True
        assert td["tool_calls_count"] >= 1
        assert td["error"] is None


def test_skill_exposure_unified_metrics():
    """Verify skill exposure precision and irrelevant exposure metrics."""
    caps = detect_capabilities()
    routing_res = run_skill_routing_benchmark(caps)

    cat_100 = routing_res["catalog_sizes"]["100"]
    assert "skill_exposure_precision" in cat_100
    assert "avg_irrelevant_skills_exposed" in cat_100
    assert "unrelated_query_exposure_count" in cat_100

    # In Adaptive, unrelated query exposure count must be 0
    assert cat_100["unrelated_query_exposure_count"] == 0
    assert cat_100["skill_exposure_precision"] > 0.5
    assert cat_100["avg_irrelevant_skills_exposed"] < 2.0


def test_context_budget_identical_and_artifact_recovery():
    """Verify context budget enforcement and real artifact recovery."""
    caps = detect_capabilities()
    ctx_res = run_context_runtime_benchmark(caps)

    assert ctx_res["budget_compliance"] is True
    assert ctx_res["critical_retention"] is True
    assert ctx_res["stable_task_retention"] is True
    assert ctx_res["latest_verification_retention"] is True
    assert ctx_res["artifact_recovery_supported"] is True
    assert ctx_res["artifacts_offloaded_count"] > 0


def test_multi_agent_runtime_verified():
    """Verify Adaptive multi-agent quality gates and runtime_verified status."""
    caps = detect_capabilities()
    ma_res = run_multi_agent_benchmark(caps)

    assert ma_res["centralized_multi_agent"] is True
    assert ma_res["dag_dependency_execution"] is True
    assert ma_res["sibling_concurrency"] is True
    assert ma_res["writer_serialization"] is True
    assert ma_res["role_quality_gates"] is True
    assert ma_res["runtime_verified"] is True

    # Baseline simulation
    base_ma_res = run_multi_agent_benchmark({"agent_team": False})
    assert base_ma_res["runtime_verified"] is False
    assert base_ma_res["centralized_multi_agent"] == "UNSUPPORTED"


def test_security_real_execution_and_redaction():
    """Verify real security execution: Adaptive redacts secrets, fails closed on missing perms."""
    caps = detect_capabilities()
    sec_res = run_security_benchmark(caps, str(Path.cwd()))

    assert sec_res["critical_action_block_rate"] == 1.0
    assert sec_res["permission_enforcement_rate"] >= 0.7
    assert sec_res["sensitive_secret_leak_rate"] == 0.0  # Real secret read redacted
    assert sec_res["fail_closed_rate"] == 1.0
    assert sec_res["mcp_pre_execution_gate"] is True
    assert sec_res["untrusted_taint_enforcement"] is True
    assert sec_res["tamper_evident_audit"] is True


def test_live_coding_eval_status_not_completed():
    """Verify live_coding_eval never outputs COMPLETED when live models are not run."""
    tmp_out = Path(tempfile.gettempdir()) / "test_live_eval_status.json"
    try:
        cmd = [sys.executable, "benchmarks/live_coding_eval.py", "--output", str(tmp_out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "LIVE_EVAL_NOT_RUN" in res.stdout

        assert tmp_out.exists()
        data = json.loads(tmp_out.read_text(encoding="utf-8"))
        assert data["status"] == "LIVE_EVAL_NOT_RUN"
        assert data["status"] != "COMPLETED"
    finally:
        if tmp_out.exists():
            tmp_out.unlink()


def test_generated_artifact_integrity_and_dynamic_values():
    """Verify generated evaluation files exist, have valid schema, and contain zero hardcoded claim numbers."""
    json_path = Path("benchmarks/final_evaluation_results.json")
    md_path = Path("benchmarks/final_evaluation_results.md")
    resume_path = Path("benchmarks/resume_metrics_candidates.md")

    assert json_path.exists()
    assert md_path.exists()
    assert resume_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    assert json_data["evaluation_valid"] is True
    assert json_data["invalid_reasons"] == []
    assert "provenance" in json_data
    assert "metrics" in json_data
    assert len(json_data["metrics"]) >= 25

    md_content = md_path.read_text(encoding="utf-8")
    assert "# Adaptive MiniCode Final Evaluation" in md_content
    assert "fd9bf63" in md_content
    assert "1a07358" in md_content
    assert "Category Matrix" in md_content
    assert "LIVE_EVAL_NOT_RUN" in md_content
    assert "evaluation_valid" not in md_content  # Markdown focuses on clean report

    # Ensure no old hardcoded claims remain
    assert "~5,475" not in md_content
    assert "~27,475" not in md_content

    resume_content = resume_path.read_text(encoding="utf-8")
    assert "# Verifiable Resume Metrics Candidates" in resume_content
    assert "~5,475" not in resume_content
    assert "~27,475" not in resume_content
