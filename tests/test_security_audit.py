"""Tests for tamper-evident hash-chained security audit log."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from minicode.security_audit import SecurityAuditLog


def test_audit_log_hash_chain_and_verify(tmp_path):
    log_file = tmp_path / "security_audit.jsonl"
    audit = SecurityAuditLog(log_file)

    # 1. Record series of events
    ev1 = audit.record_event(
        session_id="s1",
        tool_name="read_file",
        decision="ALLOW",
        risk="SAFE",
        input_data={"path": "main.py"},
        output="code contents",
    )
    assert ev1.prev_hash == SecurityAuditLog.GENESIS_HASH
    assert len(ev1.event_hash) == 64

    ev2 = audit.record_event(
        session_id="s1",
        tool_name="edit_file",
        decision="ASK",
        risk="MEDIUM",
        input_data={"path": "main.py", "content": "print(1)"},
        output="file updated",
    )
    assert ev2.prev_hash == ev1.event_hash

    ev3 = audit.record_event(
        session_id="s1",
        tool_name="run_command",
        decision="DENY",
        risk="CRITICAL",
        rule_ids=["catastrophic_command"],
        input_data={"command": "git reset --hard"},
        output="Security policy denied tool 'run_command'",
    )
    assert ev3.prev_hash == ev2.event_hash

    # 2. Verify intact chain
    valid, count, invalid_idx, reason = audit.verify_chain()
    assert valid is True
    assert count == 3
    assert invalid_idx == -1
    assert reason == "Chain valid"


def test_audit_log_detects_content_tampering(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    audit = SecurityAuditLog(log_file)

    audit.record_event(tool_name="read_file", decision="ALLOW", input_data={"path": "a.txt"})
    audit.record_event(tool_name="edit_file", decision="ASK", input_data={"path": "b.txt"})
    audit.record_event(tool_name="git", decision="ALLOW", input_data={"action": "status"})

    # Tamper with the middle event (change decision from ASK to ALLOW)
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    middle_record = json.loads(lines[1])
    middle_record["decision"] = "ALLOW"  # unauthorized modification!
    lines[1] = json.dumps(middle_record)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Verify must detect tampering!
    valid, count, invalid_idx, reason = audit.verify_chain()
    assert valid is False
    assert invalid_idx == 1
    assert "tampering detected" in reason


def test_audit_log_detects_event_deletion(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    audit = SecurityAuditLog(log_file)

    audit.record_event(tool_name="tool_1", decision="ALLOW")
    audit.record_event(tool_name="tool_2", decision="ALLOW")
    audit.record_event(tool_name="tool_3", decision="ALLOW")

    # Delete the middle event
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    tampered_lines = [lines[0], lines[2]]
    log_file.write_text("\n".join(tampered_lines) + "\n", encoding="utf-8")

    # Verification must fail because prev_hash of line 2 won't match line 0
    valid, count, invalid_idx, reason = audit.verify_chain()
    assert valid is False
    assert invalid_idx == 1
    assert "Hash chain broken" in reason


def test_audit_log_redacts_secrets_in_summary(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    audit = SecurityAuditLog(log_file)

    secret_key = "sk-proj-supersecretkey12345678"
    audit.record_event(
        tool_name="write_file",
        input_data={"path": ".env", "content": f"API_KEY={secret_key}"},
        output=f"Stored key {secret_key}",
    )

    raw_file_content = log_file.read_text(encoding="utf-8")
    assert secret_key not in raw_file_content
    assert "[REDACTED]" in raw_file_content


def test_concurrent_audit_log_from_multiple_instances(tmp_path):
    """Verify that multiple SecurityAuditLog instances writing concurrently maintain a valid hash chain."""
    from concurrent.futures import ThreadPoolExecutor

    log_file = tmp_path / "concurrent_audit.jsonl"
    num_threads = 5
    events_per_thread = 10
    total_events = num_threads * events_per_thread

    def worker(worker_id: int):
        # Each worker creates its own SecurityAuditLog instance targeting the same log_path
        local_audit = SecurityAuditLog(log_file)
        for i in range(events_per_thread):
            local_audit.record_event(
                session_id=f"sess_{worker_id}",
                tool_name=f"tool_{worker_id}_{i}",
                decision="ALLOW",
                input_data={"worker": worker_id, "iter": i},
                output=f"result_{worker_id}_{i}",
                authorization_outcome="NOT_REQUIRED",
            )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, w) for w in range(num_threads)]
        for f in futures:
            f.result()

    # Now verify the chain with a new instance
    verifier = SecurityAuditLog(log_file)
    valid, count, invalid_idx, reason = verifier.verify_chain()
    assert valid is True, f"Hash chain broken: {reason} at index {invalid_idx}"
    assert count == total_events
    assert invalid_idx == -1
    assert reason == "Chain valid"
