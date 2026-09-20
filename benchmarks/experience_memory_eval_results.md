# Structured Experience Memory Hardened Benchmark Results

- **Evaluation Date**: 2026-09-20
- **Branch**: `feat/experience-memory`
- **Total Test Cases**: 10

## Summary Metrics

| Metric | Target | Result | Status |
|---|---|---|---|
| Extraction Rate | 100% | 1.0 (10/10) | PASSED |
| Task Type Classification Accuracy | 100% | 1.0 (10/10) | PASSED |
| Outcome Determination Accuracy | 100% | 1.0 (10/10) | PASSED |
| Quality Gate Accuracy | 100% | 1.0 (10/10) | PASSED |
| Persisted Experience Entries | - | 6 | PASSED |
| Rejected Cases (Empty / Aborted / Blocked) | - | 3 | PASSED |
| Deduplication Precision (SHA-256) | 100% | 1.0 (1 hit, observation_count tracked) | PASSED |
| Secret Leak Rate | 0.0% | 0.0 (0 leaks detected) | PASSED |
| Negative Transfer Rate | 0.0% | 0.0 (failures blocked in normal tasks) | PASSED |
| Structured Metadata Preservation | 100% | 1.0 (schema v1.0, provenance, touched files) | PASSED |
| Recall@1 | >= 90% | 1.0 | PASSED |
| Recall@3 | >= 95% | 1.0 | PASSED |
| Failure Recall@K (Error Recovery) | 100% | 1.0 (boosted on tool failure) | PASSED |
| Closed-Loop Feedback Accuracy | 100% | 1.0 (Reinforce +2 / Decay -1) | PASSED |

## Benchmark Cases (10 Scenarios)

1. `case-1-pytest-fix`: Bug fix with verified pytest run (`SUCCESS_VERIFIED` -> Persisted)
2. `case-2-module-not-found`: Dependency resolution with pip and pytest verification (`SUCCESS_VERIFIED` -> Persisted)
3. `case-3-duplicate-module-not-found`: Duplicate of case-2 (`SUCCESS_VERIFIED` -> Deduplicated, increments `observation_count`)
4. `case-4-permission-error-failure`: Bug fix with terminal verification failure (`FAILED_VERIFICATION` -> Persisted for failure avoidance)
5. `case-5-secret-sanitization`: API key error with sanitized bearer credentials (`FAILED_TOOL` -> Persisted, secrets scrubbed)
6. `case-6-aborted-task`: User-interrupted task without work (`ABORTED` -> Rejected by Quality Gate)
7. `case-7-empty-trace`: No tool actions executed (`SUCCESS_UNVERIFIED` -> Rejected by Quality Gate)
8. `case-8-negative-transfer-test`: Feature implementation verifying negative transfer avoidance (`SUCCESS_VERIFIED` -> Persisted)
9. `case-9-metadata-preservation`: Refactoring task testing schema 1.0 and provenance capture (`SUCCESS_VERIFIED` -> Persisted)
10. `case-10-blocked-task`: Security/permission blocked task (`BLOCKED` -> Rejected by Quality Gate)

## Verification Coverage

1. `tests/test_memory.py`: Metadata roundtrip, schema validation, backwards compatibility.
2. `tests/test_execution_trace.py`: Execution trace capture, 1500 char bounding, secret scrubbing, reflection trace conversion.
3. `tests/test_experience.py`: Rule-based extraction, quality gate, SHA-256 fingerprinting, outcome determination.
4. `tests/test_experience_reflection_compat.py`: Agent reflection schema compatibility across legacy and modern trace formats.
5. `tests/test_experience_agent_loop_integration.py`: End-to-end agent loop execution, outcome-driven feedback, verification status.
6. `tests/test_memory_injector.py`: Memory ID propagation, outcome-aware prompt formatting, failure avoidance.
7. `tests/test_memory_pipeline.py`: Pipeline persistence, deduplication without usage inflation, feedback loop.
8. `tests/test_experience_eval.py`: End-to-end benchmark integration asserting all hardened evaluation metrics.
