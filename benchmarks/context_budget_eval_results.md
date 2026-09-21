# Context Budget Manager Evaluation Results

**Timestamp**: 2026-09-21 09:10:28
**Scope**: Phase 3 Context Budget Manager / Layered Context Engineering

## Summary Metrics

| Metric | Phase 3 Value | Baseline (Compactor Only) | Target |
|---|---|---|---|
| **Critical Information Retention Rate** | **100.0%** | 70.0% | 100.0% |
| **Latest Verification Retention Rate** | **100.0%** | 60.0% | 100.0% |
| **Stable Task Retention Rate** | **100.0%** | 80.0% | 100.0% |
| **Failure Evidence Retention Rate** | **100.0%** | 50.0% | 100.0% |
| **Estimated Context Token Reduction** | **83.7%** | 25.4% | > 40.0% |
| **Budget Compliance Rate** | **100.0%** | 60.0% | 100.0% |
| **Artifact Offload Success Rate** | **100.0%** | N/A (Plain preview) | 100.0% |
| **Artifact Recovery Success Rate** | **100.0%** | 0.0% (No tool) | 100.0% |
| **Recovery Boundedness Rate** | **100.0%** | 0.0% | 100.0% |
| **Tool Pair Integrity Rate** | **100.0%** | 100.0% | 100.0% |
| **False Eviction Rate** | **0.0%** | 30.0% | 0.0% |

## Detailed Test Case Results

| Case | Scenario | Tokens Before | Phase 3 Tokens | Critical Retained | Offloaded | Recoverable |
|---|---|---|---|---|---|---|
| `case_1` | Early Critical Constraint Retention | 899 | 664 | ✅ | — | ✅ |
| `case_2` | Huge Pytest Offload with Failure Evidence | 3780 | 263 | ✅ | ✅ | ✅ |
| `case_3` | Huge read_file Offload and Recovery | 3722 | 217 | ✅ | ✅ | ✅ |
| `case_4` | Old Irrelevant Tool Result Compression | 460 | 48 | ✅ | — | ✅ |
| `case_5` | Latest Verification Pass Retention | 29 | 29 | ✅ | — | ✅ |
| `case_6` | Earlier Pass vs Final Fail Semantics | 63 | 63 | ✅ | — | ✅ |
| `case_7` | Stable Task State Protection | 71 | 62 | ✅ | — | ✅ |
| `case_8` | Repeated Read Content Offload and Dedup | 1232 | 508 | ✅ | ✅ | ✅ |
| `case_9` | Artifact Recovery Boundedness | 5250 | 125 | ✅ | ✅ | ✅ |
| `case_10` | Extreme Pressure Compactor Fallback & Invariant Integrity | 3387 | 1107 | ✅ | ✅ | ✅ |

## Architectural Verification Notes
- **Policy vs Actuator Separation**: `ContextBudgetManager` acts as the planning policy deciding KEEP, COMPRESS, OFFLOAD, EVICT; existing `ContextCompactor` and `ToolResultBudgetManager` remain the actuators.
- **Strict Invariant Maintenance**: Tool result pairs are strictly preserved with minimal placeholder tombstones when evicted, ensuring model providers never reject broken message sequences.
- **Recoverable Evidence**: Oversized logs and file reads are offloaded to `.mini-code-tool-results/` with stable `ctx_<hash>` IDs and secret-redacted previews. The model recovers bounded slices on demand via `load_context_artifact`.
