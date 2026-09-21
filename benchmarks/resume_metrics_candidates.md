# Verifiable Resume Metrics Candidates

The following candidate metrics are strictly derived from the reproducible Final Evaluation benchmark. Every metric includes its verified baseline value, adaptive value, and exact source in the codebase.

---

### 1. Context & Prompt Efficiency

- **Skill Prompt Token Reduction**:
  - *Baseline*: ~3,378 estimated tokens per task (100-skill catalog) | ~17,303 estimated tokens (500-skill catalog).
  - *Adaptive*: ~33.0 estimated tokens per task.
  - *Impact*: **~99.0% reduction** in prompt tokens exposed to model context while maintaining **100% relevant skill recall** on deterministic synthetic fixtures.
  - *Source*: `benchmarks/final_eval/worker.py:run_skill_routing_benchmark` & `minicode/skill_router.py`.

- **Context Budget & Recoverable Artifact Offloading**:
  - *Baseline*: Baseline persisted oversized tool results through the existing ToolResultBudgetManager but lacked a first-class artifact recovery interface. Adaptive added stable Artifact IDs and explicit bounded recovery.
  - *Adaptive*: Adaptive retained a larger prepared context on this deterministic fixture (743 vs 512 estimated tokens, +45.1%), while remaining within the same 6000-token budget and supporting first-class artifact recovery. In the deterministic synthetic context fixture, Adaptive remained within the configured 6000-token budget and achieved 100% source-hash-verified artifact recovery.
  - *Impact*: Protected early critical architectural constraints and latest verification evidence under extreme context pressure without unrecoverable data loss in deterministic synthetic benchmarks.
  - *Source*: `minicode/context_budget.py` and `minicode/context_artifacts.py`.

---

### 2. Experience Memory & Knowledge Transfer

- **Negative Transfer / Failure Leakage Elimination**:
  - *Baseline*: Keyword-based memory search leaked past failure records into ~38.0% of normal coding queries on deterministic memory fixtures.
  - *Adaptive*: Outcome-aware memory gating achieved **0.0% failure leakage** and **67.0% verified experience precision** for standard task retrieval on deterministic evaluation fixtures.
  - *Source*: `minicode/memory_injector.py` and `minicode/experience.py`.

- **Experience Deduplication**:
  - *Baseline*: Stored duplicate workflows without fingerprinting.
  - *Adaptive*: Deterministic SHA-256 fingerprinting successfully deduplicated 100% of redundant task resolutions on local benchmark fixtures.
  - *Source*: `minicode/experience.py:compute_experience_fingerprint`.

---

### 3. Multi-Agent Orchestration & Concurrency

- **Topological DAG Multi-Agent Scheduling**:
  - *Baseline*: Limited to single one-off `task` delegation.
  - *Adaptive*: Orchestrated 5-node subagent teams (Researcher, Coder, Tester, Reviewer) with parallel sibling research concurrency, workspace writer serialization locks, and automated quality gates under deterministic scheduler runtime verification. TestGate and ReviewGate enforce team-level quality acceptance; when optional worktree isolation is enabled, only verified and approved patches are written back to the parent workspace.
  - *Source*: `minicode/team_planner.py`, `minicode/team_scheduler.py`, `minicode/task_graph.py`.

---

### 4. Security Policy & Tamper-Evident Auditing

- **Hard-Denial of Catastrophic Operations**:
  - *Baseline*: 33% block rate on deterministic policy fixture; destructive commands like `git reset --hard` were permitted in auto/bypass modes.
  - *Adaptive*: Achieved **100% deterministic policy-fixture block rate** for catastrophic commands and directory traversal attacks (evaluating hard-denial rules on destructive operations).
  - *Source*: `minicode/security_policy.py` and `minicode/security_rules.py`.

- **Sensitive Data Redaction & Tamper-Evident Audit**:
  - *Baseline*: 100% secret leakage on `.env` file reads; zero audit chain.
  - *Adaptive*: **0% secret leakage** via automated API key masking on deterministic test fixtures, and **100% audit log verification** via append-only SHA-256 cryptographic hash chaining in local benchmarks.
  - *Source*: `minicode/redaction.py` and `minicode/security_audit.py`.