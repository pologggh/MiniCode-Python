# Structured Experience Memory Benchmark Results

- **Evaluation Date**: 2026-09-20
- **Branch**: `feat/experience-memory`
- **Total Test Cases**: 7

## Summary Metrics

| Metric | Target | Result | Status |
|---|---|---|---|
| Extraction Rate | 100% | 100% (7/7) | PASSED |
| Quality Gate Accuracy | >= 95% | 100% | PASSED |
| Persisted Experience Entries | - | 4 | PASSED |
| Rejected Cases (Trivial / Empty) | - | 2 | PASSED |
| Deduplication Precision (SHA-256) | 100% | 1 hit (100%) | PASSED |
| Secret Leak Rate | 0.0% | 0 leaks detected | PASSED |
| Verified Experience Recall | True | True | PASSED |
| Failure Pattern Recall | True | True | PASSED |
| Closed-Loop Feedback Accuracy | 100% | True (Reinforce +2 / Decay -1) | PASSED |

## Verification Coverage

1. `tests/test_memory.py`: Metadata roundtrip, schema validation, backwards compatibility.
2. `tests/test_execution_trace.py`: Execution trace capture, 1500 char bounding, secret scrubbing.
3. `tests/test_experience.py`: Rule-based extraction, quality gate, SHA-256 fingerprinting.
4. `tests/test_memory_injector.py`: Memory ID propagation, experience prompt formatting.
5. `tests/test_memory_pipeline.py`: Pipeline persistence, deduplication, feedback loop.
6. `tests/test_experience_eval.py`: End-to-end benchmark integration.
