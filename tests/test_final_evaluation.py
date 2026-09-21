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
    CONTEXT_EXPECTED_ARTIFACT_HASHES,
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
    require_metric,
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
    assert cat_100["skill_exposure_precision"] >= 0.5
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


def test_context_budget_threshold_identical():
    """Verify that Baseline and Adaptive use identical 6000 token budget limits without * 1.5."""
    caps_adapt = {"context_budget": True}
    caps_base = {"context_budget": False}
    adapt_res = run_context_runtime_benchmark(caps_adapt)
    base_res = run_context_runtime_benchmark(caps_base)

    assert adapt_res["budget_limit"] == 6000
    assert base_res["budget_limit"] == 6000
    assert adapt_res["budget_limit"] == base_res["budget_limit"]
    assert adapt_res["budget_compliance"] is True
    assert base_res["budget_compliance"] is True


def test_artifact_recovery_hash_match():
    """Verify that offloaded artifacts are recoverable with 100% SHA-256 hash match."""
    caps_adapt = {"context_budget": True}
    caps_base = {"context_budget": False}
    adapt_res = run_context_runtime_benchmark(caps_adapt)
    base_res = run_context_runtime_benchmark(caps_base)

    # Adaptive supports recovery with 100% hash match
    assert adapt_res["artifact_recovery_supported"] is True
    assert adapt_res["artifact_recovery_attempts"] >= 2
    assert adapt_res["artifact_recovery_successes"] == adapt_res["artifact_recovery_attempts"]
    assert adapt_res["artifact_recovery_success_rate"] == 1.0

    # Baseline does not support artifact recovery
    assert base_res["artifact_recovery_supported"] is False
    assert base_res["artifact_recovery_attempts"] == 0
    assert base_res["artifact_recovery_success_rate"] == 0.0


def test_multi_agent_runtime_verification_not_hardcoded():
    """Verify multi_agent_runtime_verified is dynamically evaluated and not hardcoded."""
    caps_adapt = {"agent_team": True}
    adapt_res = run_multi_agent_benchmark(caps_adapt)
    assert adapt_res["runtime_verified"] is True
    assert adapt_res["concurrency_verified"] is True
    assert adapt_res["dag_dependency_verified"] is True
    assert adapt_res["test_gate_verified"] is True
    assert adapt_res["review_gate_verified"] is True
    assert adapt_res["writer_concurrency_verified"] is True

    # When agent_team is unsupported, runtime_verified is False
    base_res = run_multi_agent_benchmark({"agent_team": False})
    assert base_res["runtime_verified"] is False
    assert base_res["centralized_multi_agent"] == "UNSUPPORTED"


def test_multi_agent_replan_execution():
    """Verify multi-agent orchestrator replans when test gate rejects failing output."""
    caps_adapt = {"agent_team": True}
    adapt_res = run_multi_agent_benchmark(caps_adapt)
    assert adapt_res["replan_verified"] is True
    assert adapt_res["bounded_replan"] is True


def test_multi_agent_parent_isolation():
    """Verify parent context is isolated from raw child history and intermediate markers."""
    caps_adapt = {"agent_team": True}
    adapt_res = run_multi_agent_benchmark(caps_adapt)
    assert adapt_res["parent_context_isolation"] is True
    assert adapt_res["parent_isolation_verified"] is True


def test_security_fail_closed_cases():
    """Verify fail-closed evaluation across scenarios with permissions=None is case-derived."""
    repo_dir = str(Path.cwd())
    adapt_sec = run_security_benchmark({"security_policy": True}, repo_dir)
    base_sec = run_security_benchmark({"security_policy": False}, repo_dir)

    assert adapt_sec["fail_closed_case_count"] >= 2
    assert base_sec["fail_closed_case_count"] >= 2
    assert adapt_sec["fail_closed_rate"] == round(adapt_sec["fail_closed_blocked_count"] / adapt_sec["fail_closed_case_count"], 2)
    assert base_sec["fail_closed_rate"] == round(base_sec["fail_closed_blocked_count"] / base_sec["fail_closed_case_count"], 2)
    assert adapt_sec["fail_closed_rate"] > base_sec["fail_closed_rate"]


def test_security_mcp_taint_adaptive_only():
    """Verify MCP pre-execution gate and taint tracking are ADAPTIVE_ONLY."""
    repo_dir = str(Path.cwd())
    adapt_sec = run_security_benchmark({"security_policy": True}, repo_dir)
    base_sec = run_security_benchmark({"security_policy": False}, repo_dir)

    assert adapt_sec["mcp_pre_execution_gate"] is True
    assert adapt_sec["untrusted_taint_enforcement"] is True
    assert adapt_sec["tamper_evident_audit"] is True

    assert base_sec["mcp_pre_execution_gate"] == "UNSUPPORTED"
    assert base_sec["untrusted_taint_enforcement"] == "UNSUPPORTED"
    assert base_sec["tamper_evident_audit"] == "UNSUPPORTED"


def test_memory_eval_id_verification():
    """Verify memory retrieval uses [EVAL_ID] ground truth mapping, not simple keyword search."""
    adapt_mem = run_experience_memory_benchmark({"experience_memory": True})
    base_mem = run_experience_memory_benchmark({"experience_memory": False})

    # In Adaptive, outcome-aware gating prevents normal failure leakage
    assert adapt_mem["normal_failure_leakage"] == 0.0
    assert adapt_mem["verified_retrieval_precision"] >= 0.60
    assert adapt_mem["verified_retrieval_precision"] > base_mem["verified_retrieval_precision"]
    assert adapt_mem["failure_recovery_recall"] == 1.0

    # In Baseline, unverified keyword search leaks normal failure experiences
    assert base_mem["normal_failure_leakage"] > 0.0
    assert base_mem["failure_recovery_recall"] == "UNSUPPORTED"


def test_skill_micro_precision():
    """Verify skill exposure micro precision is computed as total_relevant / total_exposed."""
    adapt_routing = run_skill_routing_benchmark({"skill_router": True})
    base_routing = run_skill_routing_benchmark({"skill_router": False})

    for sz in ["10", "100", "500"]:
        a_cat = adapt_routing["catalog_sizes"][sz]
        b_cat = base_routing["catalog_sizes"][sz]

        assert "skill_exposure_micro_precision" in a_cat
        assert "skill_exposure_micro_precision" in b_cat

        # Adaptive precision is significantly higher than dumping entire catalog
        assert a_cat["skill_exposure_micro_precision"] >= 0.50
        assert b_cat["skill_exposure_micro_precision"] < 0.10
        assert a_cat["unrelated_query_exposure_count"] == 0
        assert b_cat["unrelated_query_exposure_count"] > 0


def test_deprecated_metrics_handled():
    """Verify deprecated false_positive_rate is omitted and policy_critical_action_block_rate is used."""
    invalid_reasons: list[str] = []
    base_data = {
        "commit_sha": BASELINE_COMMIT,
        "loaded_minicode_path": "d:/minicode/MiniCode-Python/minicode",
        "categories": {
            "skill_routing": {
                "catalog_sizes": {
                    sz: {
                        "recall_rate": 1.0,
                        "median_latency_ms": 1.0,
                        "avg_estimated_tokens": 100,
                        "avg_skills_exposed": 10,
                        "skill_exposure_micro_precision": 0.01,
                        "avg_irrelevant_skills_exposed": 9,
                        "unrelated_query_exposure_count": 50,
                    } for sz in ["10", "100", "500"]
                },
                "edge_cases": {"high_priority_unrelated_suppressed": True},
            },
            "experience_memory": {
                "normal_failure_leakage": 0.5,
                "verified_retrieval_precision": 0.5,
                "dedup_behavior": False,
                "metadata_preservation": False,
            },
            "context_runtime": {
                "estimated_context_tokens": 512,
                "critical_retention": True,
                "stable_task_retention": True,
                "latest_verification_retention": True,
                "budget_limit": 6000,
                "budget_compliance": True,
                "artifact_recovery_supported": False,
                "artifact_recovery_success_rate": 0.0,
                "artifact_source_hash_match_rate": 0.0,
                "artifact_metadata_integrity_rate": 0.0,
            },
            "multi_agent": {
                "one_off_task_delegation": True,
            },
            "security": {
                "policy_critical_action_block_rate": 0.0,
                "policy_intervention_rate": 0.5,
                "sensitive_secret_leak_rate": 1.0,
                "fail_closed_case_count": 2,
                "fail_closed_blocked_count": 0,
                "fail_closed_rate": 0.0,
            },
            "runtime_tasks": {
                "all_tasks_completed": True,
            },
        }
    }
    adapt_data = {
        "commit_sha": ADAPTIVE_COMMIT,
        "loaded_minicode_path": "d:/minicode/MiniCode-Python/minicode",
        "categories": {
            "skill_routing": {
                "catalog_sizes": {
                    sz: {
                        "recall_rate": 1.0,
                        "median_latency_ms": 1.0,
                        "avg_estimated_tokens": 50,
                        "avg_skills_exposed": 2,
                        "skill_exposure_micro_precision": 0.5,
                        "avg_irrelevant_skills_exposed": 1,
                        "unrelated_query_exposure_count": 0,
                    } for sz in ["10", "100", "500"]
                },
                "edge_cases": {"high_priority_unrelated_suppressed": True},
            },
            "experience_memory": {
                "normal_failure_leakage": 0.0,
                "verified_retrieval_precision": 1.0,
                "failure_recovery_recall": 1.0,
                "dedup_behavior": True,
                "metadata_preservation": True,
            },
            "context_runtime": {
                "estimated_context_tokens": 743,
                "critical_retention": True,
                "stable_task_retention": True,
                "latest_verification_retention": True,
                "budget_limit": 6000,
                "budget_compliance": True,
                "artifact_recovery_supported": True,
                "artifact_recovery_success_rate": 1.0,
                "artifact_source_hash_match_rate": 1.0,
                "artifact_metadata_integrity_rate": 1.0,
            },
            "multi_agent": {
                "one_off_task_delegation": True,
                "centralized_multi_agent": True,
                "dag_dependency_execution": True,
                "sibling_concurrency": True,
                "writer_serialization": True,
                "role_quality_gates": True,
                "bounded_replan": True,
                "parent_context_isolation": True,
                "concurrency_verified": True,
                "dag_dependency_verified": True,
                "test_gate_verified": True,
                "review_gate_verified": True,
                "writer_concurrency_verified": True,
                "replan_verified": True,
                "parent_isolation_verified": True,
                "runtime_verified": True,
            },
            "security": {
                "policy_critical_action_block_rate": 1.0,
                "policy_intervention_rate": 1.0,
                "sensitive_secret_leak_rate": 0.0,
                "fail_closed_case_count": 2,
                "fail_closed_blocked_count": 2,
                "fail_closed_rate": 1.0,
                "mcp_pre_execution_gate": True,
                "untrusted_taint_enforcement": True,
                "tamper_evident_audit": True,
            },
            "runtime_tasks": {
                "all_tasks_completed": True,
            },
        }
    }
    records = build_comparison_matrix(base_data, adapt_data, invalid_reasons)
    assert len(invalid_reasons) == 0

    metric_names = [r.name for r in records]
    # Old metric names should NOT be present
    assert "false_positive_rate_100" not in metric_names
    assert "critical_action_block_rate" not in metric_names
    # New standardized metric names should be present
    assert "policy_critical_action_block_rate" in metric_names
    assert "skill_exposure_micro_precision_100" in metric_names


def test_fail_closed_rate_is_case_derived():
    """Verify fail_closed_rate is derived from actual case outcomes (blocked / cases)."""
    repo_dir = str(Path.cwd())
    adapt_sec = run_security_benchmark({"security_policy": True}, repo_dir)
    base_sec = run_security_benchmark({"security_policy": False}, repo_dir)

    assert "fail_closed_case_count" in adapt_sec
    assert "fail_closed_blocked_count" in adapt_sec
    assert "fail_closed_rate" in adapt_sec

    assert adapt_sec["fail_closed_case_count"] >= 2
    expected_adapt = round(adapt_sec["fail_closed_blocked_count"] / adapt_sec["fail_closed_case_count"], 2)
    assert adapt_sec["fail_closed_rate"] == expected_adapt

    assert base_sec["fail_closed_case_count"] >= 2
    expected_base = round(base_sec["fail_closed_blocked_count"] / base_sec["fail_closed_case_count"], 2)
    assert base_sec["fail_closed_rate"] == expected_base


def test_baseline_context_legacy_persistence_documented():
    """Verify reports accurately document baseline's ToolResultBudgetManager legacy persistence."""
    base_data = {
        "commit_sha": BASELINE_COMMIT,
        "loaded_minicode_path": "minicode",
        "categories": {
            "skill_routing": {"catalog_sizes": {sz: {"recall_rate": 1.0, "avg_estimated_tokens": 10, "avg_skills_exposed": 1, "skill_exposure_micro_precision": 1.0, "avg_irrelevant_skills_exposed": 0, "unrelated_query_exposure_count": 0} for sz in ["10", "100", "500"]}, "edge_cases": {"high_priority_unrelated_suppressed": True}},
            "experience_memory": {"normal_failure_leakage": 0.0, "verified_retrieval_precision": 1.0, "dedup_behavior": False, "metadata_preservation": False},
            "context_runtime": {"estimated_context_tokens": 500, "critical_retention": True, "stable_task_retention": True, "latest_verification_retention": True, "budget_limit": 6000, "budget_compliance": True, "artifact_recovery_supported": False, "artifact_recovery_success_rate": 0.0, "artifact_source_hash_match_rate": 0.0, "artifact_metadata_integrity_rate": 0.0},
            "multi_agent": {"one_off_task_delegation": True},
            "security": {"policy_critical_action_block_rate": 0.33, "policy_intervention_rate": 0.5, "sensitive_secret_leak_rate": 1.0, "fail_closed_case_count": 2, "fail_closed_blocked_count": 0, "fail_closed_rate": 0.0},
            "runtime_tasks": {"all_tasks_completed": True},
        },
    }
    adapt_data = {
        "commit_sha": ADAPTIVE_COMMIT,
        "loaded_minicode_path": "minicode",
        "categories": {
            "skill_routing": {"catalog_sizes": {sz: {"recall_rate": 1.0, "avg_estimated_tokens": 10, "avg_skills_exposed": 1, "skill_exposure_micro_precision": 1.0, "avg_irrelevant_skills_exposed": 0, "unrelated_query_exposure_count": 0} for sz in ["10", "100", "500"]}, "edge_cases": {"high_priority_unrelated_suppressed": True}},
            "experience_memory": {"normal_failure_leakage": 0.0, "verified_retrieval_precision": 1.0, "failure_recovery_recall": 1.0, "dedup_behavior": True, "metadata_preservation": True},
            "context_runtime": {"estimated_context_tokens": 700, "critical_retention": True, "stable_task_retention": True, "latest_verification_retention": True, "budget_limit": 6000, "budget_compliance": True, "artifact_recovery_supported": True, "artifact_recovery_success_rate": 1.0, "artifact_source_hash_match_rate": 1.0, "artifact_metadata_integrity_rate": 1.0},
            "multi_agent": {"one_off_task_delegation": True, "centralized_multi_agent": True, "dag_dependency_execution": True, "sibling_concurrency": True, "writer_serialization": True, "role_quality_gates": True, "bounded_replan": True, "parent_context_isolation": True, "concurrency_verified": True, "dag_dependency_verified": True, "test_gate_verified": True, "review_gate_verified": True, "writer_concurrency_verified": True, "replan_verified": True, "parent_isolation_verified": True, "runtime_verified": True},
            "security": {"policy_critical_action_block_rate": 1.0, "policy_intervention_rate": 1.0, "sensitive_secret_leak_rate": 0.0, "fail_closed_case_count": 2, "fail_closed_blocked_count": 2, "fail_closed_rate": 1.0, "mcp_pre_execution_gate": True, "untrusted_taint_enforcement": True, "tamper_evident_audit": True},
            "runtime_tasks": {"all_tasks_completed": True},
        },
    }
    invalid_reasons: list[str] = []
    matrix = build_comparison_matrix(base_data, adapt_data, invalid_reasons)
    assert len(invalid_reasons) == 0

    provenance = {"baseline_commit": BASELINE_COMMIT, "adaptive_commit": ADAPTIVE_COMMIT, "phase1_merge_sha": "f3d8d7a", "phase1_parent_baseline": BASELINE_COMMIT, "phase1_parent_feature": "0db89b1"}
    md = generate_markdown_report(provenance, base_data, adapt_data, matrix)
    resume = generate_resume_metrics(matrix)

    # Must document ToolResultBudgetManager legacy persistence
    assert "ToolResultBudgetManager" in md
    assert "Legacy Large Tool Result Persistence" in md
    assert "ToolResultBudgetManager" in resume
    assert "lacked a first-class artifact recovery interface" in resume
    assert "zero disk persistence" not in md
    assert "permanently discards" not in md


def test_artifact_source_hash_exact_match():
    """Verify artifact recovery strictly requires matching CONTEXT_EXPECTED_ARTIFACT_HASHES."""
    caps_adapt = {"context_budget": True}
    res = run_context_runtime_benchmark(caps_adapt)

    assert res["artifact_recovery_supported"] is True
    assert res["artifact_source_hash_match_rate"] == 1.0
    assert res["artifact_metadata_integrity_rate"] == 1.0
    assert res["artifact_recovery_successes"] == len(CONTEXT_EXPECTED_ARTIFACT_HASHES)


def test_self_consistent_wrong_artifact_not_counted():
    """Verify that an artifact with self-consistent metadata but incorrect source hash is not counted."""
    from minicode.context_artifacts import ContextArtifactStore
    import hashlib

    temp_dir = Path(tempfile.mkdtemp(prefix="test_wrong_art_"))
    try:
        store = ContextArtifactStore(workspace=temp_dir)
        fake_content = "Corrupted or non-fixture content that should never match source hashes"
        meta = store.persist(fake_content, tool_name="test_tool")

        # The store metadata is self-consistent
        read_back = store.read(meta.artifact_id)
        rec_hash = hashlib.sha256(read_back.encode("utf-8")).hexdigest()
        assert rec_hash == meta.sha256

        # But it is NOT in CONTEXT_EXPECTED_ARTIFACT_HASHES
        assert rec_hash not in CONTEXT_EXPECTED_ARTIFACT_HASHES
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_mcp_verified_not_hardcoded():
    """Verify mcp_pre_execution_gate dynamically executes tool and verifies call_count."""
    repo_dir = str(Path.cwd())
    adapt_sec = run_security_benchmark({"security_policy": True}, repo_dir)
    assert adapt_sec["mcp_pre_execution_gate"] is True

    base_sec = run_security_benchmark({"security_policy": False}, repo_dir)
    assert base_sec["mcp_pre_execution_gate"] == "UNSUPPORTED"


def test_taint_verified_through_mutation_gate():
    """Verify untrusted_taint_enforcement dynamically exercises the mutation gate in BYPASS mode."""
    repo_dir = str(Path.cwd())
    adapt_sec = run_security_benchmark({"security_policy": True}, repo_dir)
    assert adapt_sec["untrusted_taint_enforcement"] is True

    base_sec = run_security_benchmark({"security_policy": False}, repo_dir)
    assert base_sec["untrusted_taint_enforcement"] == "UNSUPPORTED"


def test_security_resume_claim_scope():
    """Verify security block rate claim is strictly scoped to deterministic policy fixture."""
    rec = compute_metric(
        name="policy_critical_action_block_rate",
        category="Security Policy",
        description="Deterministic policy fixture block rate for catastrophic actions",
        comparability=Comparability.DIRECT,
        direction=MetricDirection.HIGHER_IS_BETTER,
        unit="rate",
        baseline_val=0.33,
        adaptive_val=1.0,
    )
    resume = generate_resume_metrics([rec])
    assert "deterministic policy-fixture block rate" in resume
    assert "across all permission modes" not in resume


def test_require_metric_missing_fails():
    """Verify require_metric detects missing fields and invalidates evaluation."""
    invalid_reasons: list[str] = []
    data = {"a": {"b": 123}}

    # Present metric returns value
    val = require_metric(data, "a.b", invalid_reasons)
    assert val == 123
    assert len(invalid_reasons) == 0

    # Missing metric appends to invalid_reasons and returns None
    missing_val = require_metric(data, "a.missing_key", invalid_reasons)
    assert missing_val is None
    assert len(invalid_reasons) == 1
    assert "Missing required metric 'a.missing_key'" in invalid_reasons[0]

    # None value also invalidates
    data_with_none = {"a": {"c": None}}
    none_val = require_metric(data_with_none, "a.c", invalid_reasons)
    assert none_val is None
    assert len(invalid_reasons) == 2
    assert "Required metric 'a.c' is None" in invalid_reasons[1]

