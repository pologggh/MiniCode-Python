"""Tamper-evident hash-chained security audit logging for MiniCode.

Provides a structured, secret-redacted, SHA-256 hash-chained JSONL audit trail
for all tool evaluations, permission gates, and security decisions.

Guarantees and Boundaries:
- Concurrency: Provides process-local multi-instance thread safety for multiple
  SecurityAuditLog instances writing to the same resolved log path within the same
  Python process.
- Known Limitation: Multiple independent MiniCode OS processes writing to the same
  audit JSONL are not serialized by the current threading lock (cross-process safety
  would require OS file locking).
- Tamper Evidence: Detects content modification and interior record deletion or
  reordering. Does not independently detect tail truncation or whole-log deletion
  without an external signed anchor or checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from minicode.redaction import redact_payload, redact_text


_PATH_LOCKS: dict[Path, threading.Lock] = {}
_MODULE_LOCK = threading.Lock()


def _get_path_lock(path: Path) -> threading.Lock:
    """Return a shared process-local threading lock for the resolved path."""
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    with _MODULE_LOCK:
        if resolved not in _PATH_LOCKS:
            _PATH_LOCKS[resolved] = threading.Lock()
        return _PATH_LOCKS[resolved]


@dataclass(slots=True)
class SecurityAuditEvent:
    event_id: str
    timestamp: float
    session_id: str
    actor: str
    agent_role: str
    tool_name: str
    decision: str
    risk: str
    rule_ids: list[str]
    reasons: list[str]
    input_digest: str
    redacted_input_summary: str
    result_ok: bool
    output_digest: str
    output_length: int
    untrusted_output: bool
    injection_detected: bool
    authorization_outcome: str
    prev_hash: str
    event_hash: str = ""

    def canonical_bytes_for_hash(self) -> bytes:
        """Serialize event fields deterministically, excluding event_hash."""
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "actor": self.actor,
            "agent_role": self.agent_role,
            "tool_name": self.tool_name,
            "decision": self.decision,
            "risk": self.risk,
            "rule_ids": sorted(self.rule_ids),
            "reasons": self.reasons,
            "input_digest": self.input_digest,
            "redacted_input_summary": self.redacted_input_summary,
            "result_ok": self.result_ok,
            "output_digest": self.output_digest,
            "output_length": self.output_length,
            "untrusted_output": self.untrusted_output,
            "injection_detected": self.injection_detected,
            "authorization_outcome": self.authorization_outcome,
            "prev_hash": self.prev_hash,
        }
        return json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of canonical event data."""
        return hashlib.sha256(self.canonical_bytes_for_hash()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SecurityAuditLog:
    """Process-local thread-safe, append-only, tamper-evident hash-chained audit log."""

    GENESIS_HASH = "0" * 64

    def __init__(self, log_path: Path | str | None = None) -> None:
        if log_path:
            self.log_path = Path(log_path)
        else:
            base_dir = Path.home() / ".mini-code" / "security-audit"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = base_dir / "audit.jsonl"

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _get_path_lock(self.log_path)
        self._last_hash = self._get_tail_hash()

    def _get_tail_hash(self) -> str:
        """Read the hash of the last event in the log file, or return GENESIS_HASH."""
        if not self.log_path.exists():
            return self.GENESIS_HASH

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                last_line = ""
                for line in f:
                    if line.strip():
                        last_line = line.strip()
                if last_line:
                    record = json.loads(last_line)
                    return record.get("event_hash", self.GENESIS_HASH)
        except Exception:
            return self.GENESIS_HASH
        return self.GENESIS_HASH

    def record_event(
        self,
        *,
        session_id: str = "",
        actor: str = "PARENT",
        agent_role: str = "parent",
        tool_name: str = "",
        decision: str = "ALLOW",
        risk: str = "SAFE",
        rule_ids: list[str] | None = None,
        reasons: list[str] | None = None,
        input_data: Any = None,
        result_ok: bool = True,
        output: str = "",
        untrusted_output: bool = False,
        injection_detected: bool = False,
        authorization_outcome: str = "NOT_REQUIRED",
    ) -> SecurityAuditEvent:
        """Record an audit event with secret redaction and hash chaining."""
        with self._lock:
            # Re-read tail hash from disk inside lock before computing prev_hash
            prev_hash = self._get_tail_hash()

            # Hash input and output data
            raw_input_str = json.dumps(input_data, default=str, sort_keys=True) if input_data else ""
            input_digest = hashlib.sha256(raw_input_str.encode("utf-8")).hexdigest()

            # Redact input summary so real secrets are never stored in plain text
            redacted_input = redact_payload(input_data, max_length=500) if input_data else {}
            redacted_summary = json.dumps(redacted_input, default=str)[:500]

            output_str = str(output or "")
            output_digest = hashlib.sha256(output_str.encode("utf-8")).hexdigest()

            event = SecurityAuditEvent(
                event_id=str(uuid.uuid4())[:8],
                timestamp=time.time(),
                session_id=session_id,
                actor=actor,
                agent_role=agent_role,
                tool_name=tool_name,
                decision=decision,
                risk=risk,
                rule_ids=rule_ids or [],
                reasons=[redact_text(str(r)) for r in (reasons or [])],
                input_digest=input_digest,
                redacted_input_summary=redacted_summary,
                result_ok=result_ok,
                output_digest=output_digest,
                output_length=len(output_str),
                untrusted_output=untrusted_output,
                injection_detected=injection_detected,
                authorization_outcome=authorization_outcome,
                prev_hash=prev_hash,
            )
            event.event_hash = event.compute_hash()
            self._last_hash = event.event_hash

            # Atomic append and flush
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            return event

    def verify_chain(self, target_path: Path | str | None = None) -> tuple[bool, int, int, str]:
        """Verify the cryptographic hash continuity of the audit log.
        
        Returns:
            (valid, total_events, first_invalid_index, reason)
        """
        path = Path(target_path) if target_path else self.log_path
        if not path.exists():
            return True, 0, -1, "Log file does not exist"

        path_lock = _get_path_lock(path)
        with path_lock:
            events: list[dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        try:
                            events.append(json.loads(line_s))
                        except Exception as e:
                            return False, len(events), len(events), f"Malformed JSON: {e}"

            if not events:
                return True, 0, -1, "Empty log"

            expected_prev = self.GENESIS_HASH
            for idx, ev in enumerate(events):
                # 1. Check prev_hash matches prior event
                if ev.get("prev_hash") != expected_prev:
                    return (
                        False,
                        len(events),
                        idx,
                        f"Hash chain broken at index {idx}: expected prev_hash '{expected_prev}', got '{ev.get('prev_hash')}'",
                    )

                # 2. Recompute event hash
                try:
                    audit_ev = SecurityAuditEvent(
                        event_id=ev["event_id"],
                        timestamp=ev["timestamp"],
                        session_id=ev["session_id"],
                        actor=ev["actor"],
                        agent_role=ev["agent_role"],
                        tool_name=ev["tool_name"],
                        decision=ev["decision"],
                        risk=ev["risk"],
                        rule_ids=ev["rule_ids"],
                        reasons=ev["reasons"],
                        input_digest=ev["input_digest"],
                        redacted_input_summary=ev["redacted_input_summary"],
                        result_ok=ev["result_ok"],
                        output_digest=ev["output_digest"],
                        output_length=ev["output_length"],
                        untrusted_output=ev["untrusted_output"],
                        injection_detected=ev["injection_detected"],
                        authorization_outcome=ev.get("authorization_outcome", "NOT_REQUIRED"),
                        prev_hash=ev["prev_hash"],
                    )
                    recomputed_hash = audit_ev.compute_hash()
                    if recomputed_hash != ev.get("event_hash"):
                        return (
                            False,
                            len(events),
                            idx,
                            f"Content tampering detected at index {idx}: recorded hash '{ev.get('event_hash')}' != recomputed '{recomputed_hash}'",
                        )
                except Exception as e:
                    return False, len(events), idx, f"Deserialization failure at index {idx}: {e}"

                expected_prev = ev["event_hash"]

            return True, len(events), -1, "Chain valid"
