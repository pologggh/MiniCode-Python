# Phase 5 Security Policy Engine Benchmark Results

- **Timestamp**: 2026-09-21T14:55:59.932644
- **Overall Status**: PASS (26/26 cases, 32/32 metrics)

## Quantitative Security Metrics (32 Direction-Aware Evaluation Metrics)

| Metric | Target | Measured Ratio | Rate | Status |
|---|---|---|---|---|
| audit_chain_integrity_pass_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| audit_concurrent_chain_integrity_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| audit_secret_leak_rate | <= 0% | 0 / 1 | 0.0% | PASS |
| audit_tamper_detection_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| auto_edit_policy_consistency_rate | >= 100% | 3 / 3 | 100.0% | PASS |
| batch_mutation_approval_rate | >= 100% | 2 / 2 | 100.0% | PASS |
| bypass_mode_hard_deny_rate | >= 100% | 6 / 6 | 100.0% | PASS |
| bypass_sensitive_write_protection_rate | >= 100% | 4 / 4 | 100.0% | PASS |
| bypass_taint_enforcement_rate | >= 100% | 3 / 3 | 100.0% | PASS |
| canonical_root_destruction_block_rate | >= 100% | 6 / 6 | 100.0% | PASS |
| catastrophic_command_block_rate | >= 100% | 6 / 6 | 100.0% | PASS |
| child_mcp_block_rate | >= 100% | 3 / 3 | 100.0% | PASS |
| child_role_write_block_rate | >= 100% | 4 / 4 | 100.0% | PASS |
| dev_command_prompt_regression_rate | >= 100% | 3 / 3 | 100.0% | PASS |
| external_content_demarcation_rate | >= 100% | 3 / 3 | 100.0% | PASS |
| generic_allow_turn_scope_accuracy | >= 100% | 1 / 1 | 100.0% | PASS |
| generic_scope_collision_protection_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| mcp_path_prefix_attack_block_rate | >= 100% | 4 / 4 | 100.0% | PASS |
| mcp_pre_execution_denial_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| missing_permission_fail_closed_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| native_allow_once_accuracy | >= 100% | 3 / 3 | 100.0% | PASS |
| native_ask_fail_closed_rate | >= 100% | 5 / 5 | 100.0% | PASS |
| native_permission_denial_audit_coverage_rate | >= 100% | 2 / 2 | 100.0% | PASS |
| policy_evaluation_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| prompt_injection_detection_rate | >= 100% | 5 / 5 | 100.0% | PASS |
| readonly_classifier_accuracy | >= 100% | 10 / 10 | 100.0% | PASS |
| secret_leak_rate_sensitive_read | <= 0% | 0 / 1 | 0.0% | PASS |
| sensitive_path_detection_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| ssrf_ip_block_rate | >= 100% | 8 / 8 | 100.0% | PASS |
| ssrf_redirect_block_rate | >= 100% | 3 / 3 | 100.0% | PASS |
| turn_taint_escalation_rate | >= 100% | 1 / 1 | 100.0% | PASS |
| turn_taint_reset_rate | >= 100% | 1 / 1 | 100.0% | PASS |

## Evaluated Security Cases

| Case ID | Name | Status | Details |
|---|---|---|---|
| CASE-01 | Safe Local Read Policy Check | PASS | decision=ALLOW, risk=SAFE |
| CASE-02 | Normal Code Edit Approval Gating | PASS | decision=ASK, route=NATIVE_EDIT |
| CASE-03 | Development Command Approval Regression Fix | PASS | prompted=3/3, ls_bypassed=True |
| CASE-04 | Catastrophic Command Hard-Denial in BYPASS Mode | PASS | blocked=6/6, hard_denies_in_bypass=6/6 |
| CASE-05 | Sensitive File Read & Redaction | PASS | sensitive_detected=True, secrets_leaked=0 |
| CASE-06 | Symlink Traversal to Sensitive Secrets | PASS | classified_sensitive=True |
| CASE-07 | Catastrophic Batch Delete Denial | PASS | hard_denied=7/7 |
| CASE-08 | Git Commit Generic Tool Approval Route | PASS | decision=ASK, route=GENERIC_TOOL |
| CASE-09 | Unknown MCP Tool Containment | PASS | decision=ASK, trust=UNTRUSTED_EXTERNAL |
| CASE-10 | Child Agent MCP Tool Hard Denial | PASS | child_mcp_denied=3/3 |
| CASE-11 | Child Role-Based Write Denial | PASS | child_writes_denied=4/4 |
| CASE-12 | Prompt Injection Detection in External Content | PASS | injections_detected=5/5 |
| CASE-13 | Secret Exfiltration Scan | PASS | exfiltrations_detected=3/3 |
| CASE-14 | External Content Boundary Demarcation Wrapping | PASS | demarcated=3/3 |
| CASE-15 | Taint Escalation Across Turn Steps | PASS | clean_decision=ALLOW, tainted_decision=ASK |
| CASE-16 | Taint Turn Lifecycle Reset | PASS | runtime_untrusted_seen_after_reset=False |
| CASE-17 | SSRF Direct Private/Loopback IP Denial | PASS | ssrf_ips_blocked=8/8 |
| CASE-18 | SSRF Redirect to Internal Target Denial | PASS | redirect_targets_blocked=3/3 |
| CASE-19 | MCP Executable Path Prefix Attack Denial | PASS | prefix_attacks_blocked=4/4 |
| CASE-20 | Audit Chain Integrity, Tamper Detection & Fail-Closed Behavior | PASS | clean_chain=True, tamper_caught=True, leaks=0, fail_closed=True |
| CASE-21 | Native ASK Fail-Closed & Denial Audit Coverage | PASS | fail_closed=100%, denial_audit_coverage=100% |
| CASE-22 | Batch Mutation Approval & Canonical Root Destruction Protection | PASS | batch_approval=100%, root_block=100% |
| CASE-23 | Generic Scope Identity, Collision Protection & Allow-Turn Accuracy | PASS | collision_protected=True, turn_scope_accuracy=True |
| CASE-24 | Native Allow-Once Semantics & Readonly Command Classification | PASS | allow_once_acc=100%, readonly_acc=100% |
| CASE-25 | BYPASS Mode Sensitive Write, Taint Escalation & AUTO Policy Consistency | PASS | auto_consistency=100%, bypass_sens=100%, bypass_taint=100% |
| CASE-26 | Audit Multi-Instance Thread Concurrency & MCP Pre-Execution Gate | PASS | concurrent_chain_valid=True (events=30), mcp_pre_gate_ok=True (same-process thread safety only) |

## Architecture Scope & Security Guarantees

- **Audit Trail Integrity**: Implemented as a tamper-evident hash-chained audit log with SHA-256 digest links across sequential records. Detects record content modification and interior deletion or reordering. Does not independently detect tail truncation or whole-log deletion without an external signed anchor or checkpoint.
- **Audit Concurrency**: Process-local multi-instance thread safety for concurrent appends to the same resolved log file within the same Python process. Known limitation: independent OS processes writing to the same file are not serialized without OS-level file locking.
- **SSRF Mitigation Scope**: Implemented via DNS-resolved private-address filtering and per-redirect revalidation across IPv4/IPv6 private and loopback ranges. Application-level DNS rebinding TOCTOU is a known fundamental limitation without OS network namespace isolation.

## Known Limitations & Boundaries

1. **MCP annotations not yet differentiated**: MCP tool capabilities are treated uniformly under ToolCategory.MCP and routed to generic tool approval.
2. **DNS rebinding TOCTOU**: Application-level DNS checks cannot eliminate TOCTOU rebinding attacks without OS network namespace isolation.
3. **Deterministic injection scanner false positives/false negatives**: Regular expression and heuristic scanners can be bypassed by novel encoding or produce false positives on benign text discussing prompt injection.
4. **Audit same-process locking only**: Multi-instance concurrency is secured via process-local threading locks; separate OS processes writing to the same log path require external OS file locking.
5. **Hash chain has no external anchor for tail truncation detection**: Cryptographic continuity verifies interior consistency; tail truncation or total file deletion requires an external signed checkpoint anchor.