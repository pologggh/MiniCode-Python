# Phase 4.2 Multi-Agent Orchestration Benchmark Results

- **Timestamp**: 2026-09-21 10:49:58
- **Overall Status**: PASS (12/12 cases, 20/20 metrics)

## 20 Quantitative Metrics (Dynamically Computed)

| Metric | Target | Measured Ratio | Rate | Status |
|---|---|---|---|---|
| child_failure_containment_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| dependency_satisfaction_rate | >= 100% | 5 / 5 | 100.0% | PASS |
| failed_patch_leakage_rate | <= 0% | 0 / 3 | 0.0% | PASS |
| mcp_leakage_rate | <= 0% | 0 / 4 | 0.0% | PASS |
| missing_permission_fail_closed_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| nested_agent_exposure_rate | <= 0% | 0 / 4 | 0.0% | PASS |
| parent_context_isolation_pass_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| parent_workspace_race_protection_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| patch_verify_fail_closed_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| permission_denial_protection_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| plan_validity_rate | >= 100% | 7 / 7 | 100.0% | PASS |
| read_only_parallelization_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| replan_boundedness_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| review_gate_accuracy | >= 100% | 4 / 4 | 100.0% | PASS |
| role_tool_conformance_rate | >= 100% | 4 / 4 | 100.0% | PASS |
| test_gate_accuracy | >= 100% | 7 / 7 | 100.0% | PASS |
| verified_patch_apply_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| worktree_parent_dirty_rejection_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| writer_reader_overlap_violation_rate | <= 0% | 0 / 1 | 0.0% | PASS |
| writer_serialization_rate | >= 100% | 1 / 1 | 100.0% | PASS |

## Evaluated Test Cases

| Case ID | Name | Status | Details |
|---|---|---|---|
| case_1 | Parent Agent Turn & Context Isolation | PASS | Child context: 16500 bytes; Parent tool result: 531 bytes; Raw history leaks: 0. |
| case_2 | DAG Planning & Dependency Satisfaction | PASS | Dependencies satisfied: 5/5 across canonical 5-node DAG. |
| case_3 | Plan Validation Engine | PASS | Plan validation checks: 7/7 passed. |
| case_4 | Role Tool Conformance Audit | PASS | Audited all role policies: 4/4 roles conform. |
| case_5 | Nested Agent Exposure & MCP Isolation | PASS | Nested tool exposures: 0/4; MCP leakages: 0/4. |
| case_6 | Read-Only Parallelization | PASS | Max concurrent readers: 2, speedup: 1.97x. |
| case_7 | Writer Serialization & Zero Overlap | PASS | Max concurrent writers: 1, overlap violations: 0. |
| case_8 | TestGate Accuracy & Adversarial Matrix | PASS | Test gate checks: 7/7 passed. |
| case_9 | ReviewGate Accuracy & Replan Boundedness | PASS | Review checks: 4/4; Replan bounded to 1: True. |
| case_10 | Child Failure Containment | PASS | Upstream failure cascade skipped dependents cleanly: True. |
| case_11 | Worktree Pre-Execution & Race Protection | PASS | Dirty rejection: True; Race protection: True; Verify fail-closed: True. |
| case_12 | Permission Gates, Leakage & Verified Patch Apply | PASS | Perm denial protected: True; Missing perm protected: True; Failed patch leaks: 0/3; Verified patch applied: True. |