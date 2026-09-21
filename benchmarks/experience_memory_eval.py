"""Structured Experience Memory Benchmark and Evaluation.

Evaluates:
1. Phase A (Extraction & Persistence):
   - Deterministic extraction accuracy (Task Type & Outcome)
   - Quality gate filtering (rejecting empty/aborted/blocked tasks)
   - Secret redaction verification (zero secret leaks)
   - SHA-256 fingerprint deduplication
   - Structured metadata preservation
2. Phase B (Recall & Reuse):
   - Outcome-aware retrieval and category formatting
   - Prevention of negative transfer (avoiding past failure patterns for normal tasks)
   - Closed-loop feedback (usage count reinforcement and decay)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from minicode.execution_trace import ExecutionTrace
from minicode.experience import (
    ExperienceExtractor,
    ExperienceOutcome,
    ExperienceQualityGate,
    ExperienceRecord,
    compute_experience_fingerprint,
    experience_to_memory_entry,
)
from minicode.memory import MemoryManager, MemoryScope
from minicode.memory_injector import MemoryInjector
from minicode.memory_pipeline import MemoryPipeline


@dataclass
class EvalTraceCase:
    case_id: str
    description: str
    tools: list[tuple[str, dict[str, Any], bool, str]]  # tool_name, args, ok, output
    verification: tuple[str, int, bool, str] | None     # cmd, exit_code, passed, output
    expected_task_type: str
    expected_outcome: ExperienceOutcome
    expect_persist: bool
    contains_secret: bool = False
    stop_status: str = "completed"


BENCHMARK_CASES: list[EvalTraceCase] = [
    EvalTraceCase(
        case_id="case-1-pytest-fix",
        description="Fix failing pytest assertion in user authentication endpoint",
        tools=[
            ("view_file", {"path": "auth.py"}, True, "def verify_token(): ..."),
            ("replace_file_content", {"TargetFile": "auth.py"}, True, "replaced"),
        ],
        verification=("pytest tests/test_auth.py", 0, True, "1 passed in 0.12s"),
        expected_task_type="bug_fix",
        expected_outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        expect_persist=True,
    ),
    EvalTraceCase(
        case_id="case-2-module-not-found",
        description="Fix ModuleNotFoundError: No module named 'yaml' in config loader",
        tools=[
            ("run_command", {"CommandLine": "python loader.py"}, False, "ModuleNotFoundError: No module named 'yaml'"),
            ("run_command", {"CommandLine": "pip install pyyaml"}, True, "Successfully installed pyyaml"),
        ],
        verification=("pytest tests/test_loader.py", 0, True, "passed"),
        expected_task_type="dependency_issue",
        expected_outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        expect_persist=True,
    ),
    EvalTraceCase(
        case_id="case-3-duplicate-module-not-found",
        description="Fix ModuleNotFoundError: No module named 'yaml' in config loader",
        tools=[
            ("run_command", {"CommandLine": "python loader.py"}, False, "ModuleNotFoundError: No module named 'yaml'"),
            ("run_command", {"CommandLine": "pip install pyyaml"}, True, "Successfully installed pyyaml"),
        ],
        verification=("pytest tests/test_loader.py", 0, True, "passed"),
        expected_task_type="dependency_issue",
        expected_outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        expect_persist=True,  # Will be caught by deduplication
    ),
    EvalTraceCase(
        case_id="case-4-permission-error-failure",
        description="Fix permission error when executing deployment script",
        tools=[
            ("run_command", {"CommandLine": "./deploy.sh"}, False, "PermissionError: [Errno 13] Permission denied"),
        ],
        verification=("bash ./deploy.sh", 1, False, "Permission denied"),
        expected_task_type="bug_fix",
        expected_outcome=ExperienceOutcome.FAILED_VERIFICATION,
        expect_persist=True,
        stop_status="failed",
    ),
    EvalTraceCase(
        case_id="case-5-secret-sanitization",
        description="Fix authentication error configuring credentials with OpenAI and GitHub",
        tools=[
            (
                "run_command",
                {"CommandLine": "export OPENAI_API_KEY=sk-live1234567890abcdef12345678"},
                False,
                "AuthenticationError: sk-live1234567890abcdef12345678 failed with Bearer secret_bearer_token_9999",
            ),
        ],
        verification=None,
        expected_task_type="bug_fix",
        expected_outcome=ExperienceOutcome.FAILED_TOOL,
        expect_persist=True,
        contains_secret=True,
        stop_status="failed",
    ),
    EvalTraceCase(
        case_id="case-6-aborted-task",
        description="Interactive session cancelled by user midway",
        tools=[
            ("view_file", {"path": "main.py"}, True, "def main(): pass"),
        ],
        verification=None,
        expected_task_type="exploration",
        expected_outcome=ExperienceOutcome.ABORTED,
        expect_persist=False,
        stop_status="aborted",
    ),
    EvalTraceCase(
        case_id="case-7-empty-trace",
        description="Empty task with no tool actions",
        tools=[],
        verification=None,
        expected_task_type="general",
        expected_outcome=ExperienceOutcome.SUCCESS_UNVERIFIED,
        expect_persist=False,
    ),
    EvalTraceCase(
        case_id="case-8-negative-transfer-test",
        description="Implement user avatar upload endpoint with s3 storage",
        tools=[
            ("write_to_file", {"TargetFile": "avatar.py"}, True, "created"),
        ],
        verification=("pytest tests/test_avatar.py", 0, True, "1 passed"),
        expected_task_type="feature_impl",
        expected_outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        expect_persist=True,
    ),
    EvalTraceCase(
        case_id="case-9-metadata-preservation",
        description="Refactor database connection pool metrics",
        tools=[
            ("view_file", {"path": "db/pool.py"}, True, "class Pool: ..."),
            ("replace_file_content", {"TargetFile": "db/pool.py"}, True, "modified"),
        ],
        verification=("pytest tests/test_pool.py", 0, True, "passed"),
        expected_task_type="refactoring",
        expected_outcome=ExperienceOutcome.SUCCESS_VERIFIED,
        expect_persist=True,
    ),
    EvalTraceCase(
        case_id="case-10-blocked-task",
        description="Explore external production database without credentials",
        tools=[
            ("run_command", {"CommandLine": "connect_prod_db"}, False, "PermissionError: Blocked"),
        ],
        verification=None,
        expected_task_type="exploration",
        expected_outcome=ExperienceOutcome.BLOCKED,
        expect_persist=False,
        stop_status="blocked",
    ),
    EvalTraceCase(
        case_id="case-11-unverified-doc-formatting",
        description="Fix markdown documentation formatting and headers",
        tools=[
            ("view_file", {"path": "docs/api.md"}, True, "# API Docs\n..."),
            ("replace_file_content", {"TargetFile": "docs/api.md"}, True, "formatted"),
        ],
        verification=None,
        expected_task_type="bug_fix",
        expected_outcome=ExperienceOutcome.SUCCESS_UNVERIFIED,
        expect_persist=True,
    ),
]


def run_experience_benchmark(workspace_path: Path | None = None) -> dict[str, Any]:
    """Execute complete experience memory evaluation suite with genuine metrics."""
    temp_dir = None
    if workspace_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        ws = Path(temp_dir.name)
    else:
        ws = workspace_path

    mgr = MemoryManager(workspace=ws)
    pipeline = MemoryPipeline(memory_manager=mgr)
    pipeline.initialize(workspace_path=str(ws), enable_reranker=False)

    extractor = ExperienceExtractor()
    gate = ExperienceQualityGate()

    # --- Phase A: Extraction, Validation, Deduplication, Redaction ---
    extracted_records: list[ExperienceRecord] = []
    persisted_ids: list[str] = []
    rejected_cases: list[str] = []
    persist_decisions: list[bool] = []
    dedup_hits = 0
    secret_leaks = 0

    for case in BENCHMARK_CASES:
        trace = ExecutionTrace()
        trace.record_task_start(case.case_id, case.description, workspace=str(ws))

        for t_name, t_args, t_ok, t_out in case.tools:
            trace.record_tool_call(f"call-{case.case_id}", t_name, t_args)
            trace.record_tool_result(
                f"call-{case.case_id}",
                t_name,
                ok=t_ok,
                output=t_out,
                error="" if t_ok else t_out,
            )

        if case.verification:
            cmd, code, passed, out = case.verification
            trace.record_verification(cmd, code, passed, out)

        trace.record_task_end(status=case.stop_status, summary=case.description)

        rec = extractor.extract(trace)
        assert rec is not None, f"Extraction failed for {case.case_id}"
        extracted_records.append(rec)

        should_persist, reason = gate.should_persist(rec, trace)
        persist_decisions.append(should_persist)

        if should_persist:
            # Check if this fingerprint already exists
            fp = rec.fingerprint
            is_dup = any(
                entry.metadata.get("fingerprint") == fp
                for entry in mgr.memories[MemoryScope.PROJECT].entries
                if entry.metadata
            )
            mem_id = pipeline.write_experience(rec)
            if is_dup:
                dedup_hits += 1
            else:
                persisted_ids.append(mem_id)
        else:
            rejected_cases.append(case.case_id)

        # Check for secret leakage
        if case.contains_secret:
            rec_json = json.dumps(rec.to_dict())
            if "sk-live1234567890abcdef" in rec_json or "secret_bearer_token" in rec_json:
                secret_leaks += 1

    # Genuine Extraction & Quality Metrics
    task_type_correct = sum(1 for rec, case in zip(extracted_records, BENCHMARK_CASES) if rec.task_type == case.expected_task_type)
    task_type_accuracy = round(task_type_correct / len(BENCHMARK_CASES), 3)

    outcome_correct = sum(1 for rec, case in zip(extracted_records, BENCHMARK_CASES) if rec.outcome == case.expected_outcome)
    outcome_accuracy = round(outcome_correct / len(BENCHMARK_CASES), 3)

    quality_gate_correct = sum(1 for p, case in zip(persist_decisions, BENCHMARK_CASES) if p == case.expect_persist)
    quality_gate_accuracy = round(quality_gate_correct / len(BENCHMARK_CASES), 3)

    # --- Phase B: Retrieval, Reuse, and Feedback Loop ---
    injector = MemoryInjector(memory_manager=mgr, min_relevance=0.1, metrics=pipeline.metrics)

    # 1. Query A: Relevant SUCCESS_VERIFIED (Auth bug fix)
    q_a_results = injector.inject_for_task("AssertionError in authentication endpoint")
    q_a_top = q_a_results[0] if q_a_results else None
    q_a_formatted = injector.format_for_prompt(q_a_results) if q_a_results else ""
    q_a_match = (
        q_a_top is not None
        and "[Verified Experience]" in q_a_formatted
        and "auth" in q_a_top.content.lower()
    )

    # 2. Query A2: Relevant SUCCESS_VERIFIED (Dependency resolution)
    q_a2_results = injector.inject_for_task("ModuleNotFoundError pyyaml in config loader")
    q_a2_top = q_a2_results[0] if q_a2_results else None
    q_a2_match = (
        q_a2_top is not None
        and "yaml" in q_a2_top.content.lower()
    )

    # 3. Query B: Relevant SUCCESS_UNVERIFIED (Doc formatting)
    q_b_results = injector.inject_for_task("Fix markdown documentation formatting")
    q_b_top = q_b_results[0] if q_b_results else None
    q_b_formatted = injector.format_for_prompt(q_b_results) if q_b_results else ""
    q_b_match = (
        q_b_top is not None
        and "[Unverified Experience]" in q_b_formatted
        and "docs/api.md" in q_b_top.content.lower()
    )

    # 4. Query C: Surface/lexically highly relevant to FAILED_VERIFICATION (Permission error)
    # Must NOT leak failure pattern into normal task prompt
    q_c_results = injector.inject_for_task("Fix permission error when executing deployment script")

    # 5. Query D: Surface/lexically highly relevant to FAILED_TOOL (Secret credential error)
    # Must NOT leak failure pattern into normal task prompt
    q_d_results = injector.inject_for_task("Fix authentication error configuring credentials with OpenAI and GitHub")

    # 6. Query E: Completely irrelevant query (No relevant memories should match)
    q_e_results = injector.inject_for_task("Implement quantum encryption algorithm from scratch")

    # 7. Query F: Failure recovery query (inject_on_failure)
    q_fail_results = injector.inject_on_failure(
        error_message="PermissionError: [Errno 13] Permission denied",
        tool_name="run_command",
    )
    q_fail_top = q_fail_results[0] if q_fail_results else None
    q_fail_formatted = injector.format_for_prompt(q_fail_results) if q_fail_results else ""
    q_fail_match = (
        q_fail_top is not None
        and "[Past Failure Pattern]" in q_fail_formatted
        and "permission" in q_fail_top.content.lower()
    )

    # Normal queries collection (A, A2, B, C, D, E)
    all_normal_injected = q_a_results + q_a2_results + q_b_results + q_c_results + q_d_results + q_e_results
    normal_exp_injected = [m for m in all_normal_injected if m.category == "experience"]

    # Normal failure leakage rate: proportion of failure experiences among injected normal experiences (Target: 0.0)
    failed_injected_count = sum(
        1 for m in normal_exp_injected
        if getattr(m, "outcome", None) in ("failed_tool", "failed_verification")
        or "Past Failure Pattern" in m.content
    )
    normal_failure_leakage_rate = round(failed_injected_count / max(len(normal_exp_injected), 1), 3)
    negative_transfer_rate = normal_failure_leakage_rate

    # Verified Experience Precision@3 across queries targeting verified experience (q_a, q_a2)
    verified_queries_injected = q_a_results[:3] + q_a2_results[:3]
    verified_at_3_count = sum(1 for m in verified_queries_injected if getattr(m, "outcome", None) == "success_verified")
    verified_precision_at_3 = round(verified_at_3_count / max(len(verified_queries_injected), 1), 3)

    # Verified precision across all normal injected experiences
    verified_total_count = sum(1 for m in normal_exp_injected if getattr(m, "outcome", None) == "success_verified")
    verified_precision_at_k = round(verified_total_count / max(len(normal_exp_injected), 1), 3)

    # Recall metrics
    recall_queries = [
        (q_a_results, "auth"),
        (q_a2_results, "yaml"),
        (q_fail_results, "permission"),
    ]
    r_at_1_hits = sum(1 for res, kw in recall_queries if res and kw in res[0].content.lower())
    recall_at_1 = round(r_at_1_hits / len(recall_queries), 3)

    r_at_3_hits = sum(1 for res, kw in recall_queries if any(kw in m.content.lower() for m in res[:3]))
    recall_at_3 = round(r_at_3_hits / len(recall_queries), 3)

    failure_recall_at_k = 1.0 if q_fail_match else 0.0
    reuse_hit_rate = round(sum(1 for res, _ in recall_queries if len(res) > 0) / len(recall_queries), 3)

    # Metadata Preservation Verification (Case 9)
    metadata_preservation_score = 0.0
    case9_entry = None
    for entry in mgr.memories[MemoryScope.PROJECT].entries:
        if "pool" in entry.content.lower():
            case9_entry = entry
            break

    if case9_entry and isinstance(case9_entry.metadata, dict):
        exp_dict = case9_entry.metadata.get("experience", {})
        checks = [
            exp_dict.get("schema_version") == "1.0",
            "db/pool.py" in exp_dict.get("files_read", []),
            "db/pool.py" in exp_dict.get("files_touched", []),
            "view_file" in exp_dict.get("tools_used", []),
            "replace_file_content" in exp_dict.get("tools_used", []),
            exp_dict.get("verification", {}).get("passed") is True,
            isinstance(exp_dict.get("provenance"), dict) and len(exp_dict.get("provenance")) > 0,
        ]
        metadata_preservation_score = round(sum(1 for c in checks if c) / len(checks), 3)

    # Dedup precision: verify deduplicated entry tracked observation_count without inflating usage_count during write
    dedup_precision = 0.0
    for entry in mgr.memories[MemoryScope.PROJECT].entries:
        if "yaml" in entry.content.lower():
            if dedup_hits >= 1 and entry.metadata.get("observation_count", 0) >= 2:
                dedup_precision = 1.0
            break

    # Closed-loop feedback accuracy
    initial_usage = 0
    final_usage_positive = 0
    final_usage_negative = 0
    if q_a_top and q_a_top.memory_id:
        target_entry = mgr.memories[MemoryScope.PROJECT]._id_index.get(q_a_top.memory_id)
        if target_entry:
            initial_usage = target_entry.usage_count
            pipeline.feedback(task_success=True, injected_memory_ids=[q_a_top.memory_id])
            final_usage_positive = target_entry.usage_count

            pipeline.feedback(task_success=False, injected_memory_ids=[q_a_top.memory_id])
            final_usage_negative = target_entry.usage_count

    feedback_correct = (
        final_usage_positive == initial_usage + 2
        and final_usage_negative == final_usage_positive - 1
    )
    feedback_accuracy = 1.0 if feedback_correct else 0.0
    secret_leak_rate = round(secret_leaks / len(BENCHMARK_CASES), 3)

    results = {
        "total_test_cases": len(BENCHMARK_CASES),
        "extracted_count": len(extracted_records),
        "extraction_rate": round(len(extracted_records) / len(BENCHMARK_CASES), 2),
        "persisted_count": len(persisted_ids),
        "rejected_count": len(rejected_cases),
        "dedup_count": dedup_hits,
        "secret_leaks": secret_leaks,
        "secret_leak_rate": secret_leak_rate,
        "task_type_accuracy": task_type_accuracy,
        "outcome_accuracy": outcome_accuracy,
        "quality_gate_accuracy": quality_gate_accuracy,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "verified_precision_at_3": verified_precision_at_3,
        "verified_precision_at_k": verified_precision_at_k,
        "normal_failure_leakage_rate": normal_failure_leakage_rate,
        "failure_recall_at_k": failure_recall_at_k,
        "reuse_hit_rate": reuse_hit_rate,
        "negative_transfer_rate": negative_transfer_rate,
        "dedup_precision": dedup_precision,
        "metadata_preservation_rate": metadata_preservation_score,
        "feedback_accuracy": feedback_accuracy,
        # Legacy boolean flags for backwards compatibility
        "recall_verified_experience": q_a_match,
        "recall_failure_pattern": q_fail_match,
        "feedback_loop_accurate": feedback_correct,
    }

    if temp_dir:
        temp_dir.cleanup()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured Experience Memory Evaluation")
    parser.add_argument("--json-out", help="Optional path to output evaluation JSON")
    args = parser.parse_args()

    results = run_experience_benchmark()
    print("=" * 65)
    print("STRUCTURED EXPERIENCE MEMORY HARDENED EVALUATION RESULTS")
    print("=" * 65)
    for k, v in results.items():
        print(f"  {k:30}: {v}")
    print("=" * 65)

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results saved to {args.json_out}")


if __name__ == "__main__":
    main()
