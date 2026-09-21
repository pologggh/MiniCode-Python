# Verifiable Resume Metrics Candidates

The following candidate metrics are strictly derived from the reproducible Final Evaluation benchmark. Every metric includes its verified baseline value, adaptive value, and exact source in the codebase.

---

### 1. Context & Prompt Efficiency

- **Skill Prompt Token Reduction**:
  - *Baseline*: ~3,378 estimated tokens per task (100-skill catalog) | ~17,303 estimated tokens (500-skill catalog).
  - *Adaptive*: ~33.0 estimated tokens per task.
  - *Impact*: **~99.0% reduction** in prompt tokens exposed to model context while maintaining **100% relevant skill recall**.
  - *Source*: `benchmarks/final_eval/worker.py:run_skill_routing_benchmark` & `minicode/skill_router.py`.

- **Context Budget & Recoverable Artifact Offloading**:
  - *Baseline*: Context compactor truncated large tool logs permanently (zero artifact recovery).
  - *Adaptive*: Adaptive 包含结构化上下文元数据与可恢复 artifact 引用，单轮上下文基础开销略高于纯文本（743 vs 512 tokens, +45.1%），但在长上下文和大型工具输出场景下通过 offload 保证 100% 遵守 6000 token budget，且产物 100% 可恢复验证 (SHA-256 match).
  - *Impact*: Protected early critical architectural constraints and latest verification evidence under extreme context pressure without unrecoverable data loss.
  - *Source*: `minicode/context_budget.py` and `minicode/context_artifacts.py`.

---

### 2. Experience Memory & Knowledge Transfer

- **Negative Transfer / Failure Leakage Elimination**:
  - *Baseline*: Keyword-based memory search leaked past failure records into ~38.0% of normal coding queries.
  - *Adaptive*: Outcome-aware memory gating achieved **0.0% failure leakage** and **67.0% verified experience precision** for standard task retrieval.
  - *Source*: `minicode/memory_injector.py` and `minicode/experience.py`.

- **Experience Deduplication**:
  - *Baseline*: Stored duplicate workflows without fingerprinting.
  - *Adaptive*: Deterministic SHA-256 fingerprinting successfully deduplicated 100% of redundant task resolutions.
  - *Source*: `minicode/experience.py:compute_experience_fingerprint`.

---

### 3. Multi-Agent Orchestration & Concurrency

- **Topological DAG Multi-Agent Scheduling**:
  - *Baseline*: Limited to single one-off `task` delegation.
  - *Adaptive*: Orchestrated 5-node subagent teams (Researcher, Coder, Tester, Reviewer) with parallel sibling research concurrency, workspace writer serialization locks, and automated quality gates (TestGate and ReviewGate).
  - *Source*: `minicode/team_planner.py`, `minicode/team_scheduler.py`, `minicode/task_graph.py`.

---

### 4. Security Policy & Tamper-Evident Auditing

- **Hard-Denial of Catastrophic Operations**:
  - *Baseline*: 33% block rate; destructive commands like `git reset --hard` were permitted in auto/bypass modes.
  - *Adaptive*: Achieved **100% block rate** for catastrophic commands and directory traversal attacks across all permission modes.
  - *Source*: `minicode/security_policy.py` and `minicode/security_rules.py`.

- **Sensitive Data Redaction & Tamper-Evident Audit**:
  - *Baseline*: 100% secret leakage on `.env` file reads; zero audit chain.
  - *Adaptive*: **0% secret leakage** via automated API key masking, and **100% audit log verification** via append-only SHA-256 cryptographic hash chaining.
  - *Source*: `minicode/redaction.py` and `minicode/security_audit.py`.