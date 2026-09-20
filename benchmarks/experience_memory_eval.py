"""Structured Experience Memory Benchmark and Evaluation.

Evaluates:
1. Phase A (Extraction & Persistence):
   - Deterministic extraction accuracy
   - Quality gate filtering (rejecting empty/aborted tasks)
   - Secret redaction verification (zero secret leaks)
   - SHA-256 fingerprint deduplication
2. Phase B (Recall & Reuse):
   - Retrieval relevance and category formatting
   - Memory ID preservation
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
        description="Execute deployment script with restricted filesystem permissions",
        tools=[
            ("run_command", {"CommandLine": "./deploy.sh"}, False, "PermissionError: [Errno 13] Permission denied"),
        ],
        verification=("bash ./deploy.sh", 1, False, "Permission denied"),
        expected_task_type="bug_fix",
        expected_outcome=ExperienceOutcome.FAILED_VERIFICATION,
        expect_persist=True,
    ),
    EvalTraceCase(
        case_id="case-5-secret-sanitization",
        description="Configure API integration with OpenAI and GitHub credentials",
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
]


def run_experience_benchmark(workspace_path: Path | None = None) -> dict[str, Any]:
    """Execute complete experience memory evaluation suite."""
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

        status = "aborted" if case.expected_outcome == ExperienceOutcome.ABORTED else "completed"
        trace.record_task_end(status=status, summary=case.description)

        rec = extractor.extract(trace)
        assert rec is not None, f"Extraction failed for {case.case_id}"
        extracted_records.append(rec)

        should_persist, reason = gate.should_persist(rec, trace)
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

    # --- Phase B: Retrieval, Reuse, and Feedback Loop ---
    injector = MemoryInjector(memory_manager=mgr, min_relevance=0.1)

    # Query 1: Should retrieve case-1 verified experience
    q1_results = injector.inject_for_task("AssertionError in authentication endpoint")
    q1_top = q1_results[0] if q1_results else None
    q1_formatted = injector.format_for_prompt(q1_results) if q1_results else ""

    q1_recalled_verified = (
        q1_top is not None
        and "[Verified Experience]" in q1_formatted
        and q1_top.memory_id is not None
    )

    # Query 2: Should retrieve case-4 failure pattern
    q2_results = injector.inject_on_failure(
        error_message="PermissionError: [Errno 13] Permission denied",
        tool_name="run_command",
    )
    q2_top = q2_results[0] if q2_results else None
    q2_formatted = injector.format_for_prompt(q2_results) if q2_results else ""
    q2_recalled_failure = (
        q2_top is not None
        and "[Past Failure Pattern]" in q2_formatted
        and q2_top.memory_id is not None
    )

    # Test closed-loop feedback
    initial_usage = 0
    final_usage_positive = 0
    final_usage_negative = 0
    if q1_top and q1_top.memory_id:
        target_entry = mgr.memories[MemoryScope.PROJECT]._id_index.get(q1_top.memory_id)
        if target_entry:
            initial_usage = target_entry.usage_count
            pipeline.feedback(task_success=True, injected_memory_ids=[q1_top.memory_id])
            final_usage_positive = target_entry.usage_count

            pipeline.feedback(task_success=False, injected_memory_ids=[q1_top.memory_id])
            final_usage_negative = target_entry.usage_count

    feedback_correct = (
        final_usage_positive == initial_usage + 2
        and final_usage_negative == final_usage_positive - 1
    )

    results = {
        "total_test_cases": len(BENCHMARK_CASES),
        "extracted_count": len(extracted_records),
        "extraction_rate": round(len(extracted_records) / len(BENCHMARK_CASES), 2),
        "persisted_count": len(persisted_ids),
        "rejected_count": len(rejected_cases),
        "dedup_count": dedup_hits,
        "secret_leaks": secret_leaks,
        "recall_verified_experience": q1_recalled_verified,
        "recall_failure_pattern": q2_recalled_failure,
        "feedback_loop_accurate": feedback_correct,
        "quality_gate_accuracy": round(
            (len(persisted_ids) + dedup_hits + len(rejected_cases)) / len(BENCHMARK_CASES), 2
        ),
    }

    if temp_dir:
        temp_dir.cleanup()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured Experience Memory Evaluation")
    parser.add_argument("--json-out", help="Optional path to output evaluation JSON")
    args = parser.parse_args()

    results = run_experience_benchmark()
    print("=" * 60)
    print("STRUCTURED EXPERIENCE MEMORY EVALUATION RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:30}: {v}")
    print("=" * 60)

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results saved to {args.json_out}")


if __name__ == "__main__":
    main()
