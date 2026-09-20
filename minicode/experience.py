"""Structured experience memory data models, extraction, and quality gate.

Extracts deterministic, rule-based experience records from execution traces,
applies quality validation and deduplication, and formats them for memory storage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import re
import time
from typing import Any

from minicode.execution_trace import ExecutionTrace, TraceEventType, sanitize_text
from minicode.memory import MemoryEntry, MemoryScope


class ExperienceOutcome(str, Enum):
    """Categorized outcome of a completed or failed task turn."""
    SUCCESS_VERIFIED = "success_verified"      # Task succeeded with passing tests/checks
    SUCCESS_UNVERIFIED = "success_unverified"  # Task completed without formal verification proof
    FAILED_TOOL = "failed_tool"                # Failed due to tool error or runtime exception
    FAILED_VERIFICATION = "failed_verification"  # Failed due to failing test assertion or gate
    BLOCKED = "blocked"                        # Agent was blocked (missing credentials, etc.)
    ABORTED = "aborted"                        # User cancelled or session was killed


@dataclass
class VerificationEvidence:
    """Proof of verification from execution checks."""
    command: str
    exit_code: int
    passed: bool
    evidence_snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationEvidence":
        return cls(
            command=data.get("command", ""),
            exit_code=data.get("exit_code", 0),
            passed=data.get("passed", False),
            evidence_snippet=data.get("evidence_snippet", ""),
        )


@dataclass
class ExperienceRecord:
    """Structured experience distilled from an execution trace."""
    task_id: str
    task_type: str
    outcome: ExperienceOutcome
    symptom: str
    root_cause: str
    strategy: list[str]
    verification: VerificationEvidence | None = None
    lessons_learned: list[str] = field(default_factory=list)
    confidence: float = 0.5
    fingerprint: str = ""
    task_description: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "outcome": self.outcome.value,
            "task_description": self.task_description,
            "symptom": self.symptom,
            "root_cause": self.root_cause,
            "strategy": self.strategy,
            "verification": self.verification.to_dict() if self.verification else None,
            "lessons_learned": self.lessons_learned,
            "confidence": round(self.confidence, 3),
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperienceRecord":
        verif_data = data.get("verification")
        verification = VerificationEvidence.from_dict(verif_data) if verif_data else None
        outcome_str = data.get("outcome", ExperienceOutcome.SUCCESS_UNVERIFIED.value)
        try:
            outcome = ExperienceOutcome(outcome_str)
        except ValueError:
            outcome = ExperienceOutcome.SUCCESS_UNVERIFIED

        return cls(
            task_id=data.get("task_id", ""),
            task_type=data.get("task_type", "general"),
            outcome=outcome,
            task_description=data.get("task_description", ""),
            symptom=data.get("symptom", ""),
            root_cause=data.get("root_cause", ""),
            strategy=data.get("strategy", []),
            verification=verification,
            lessons_learned=data.get("lessons_learned", []),
            confidence=data.get("confidence", 0.5),
            fingerprint=data.get("fingerprint", ""),
            created_at=data.get("created_at", time.time()),
        )


def compute_experience_fingerprint(
    task_type: str,
    symptom: str,
    root_cause: str,
    outcome: str,
) -> str:
    """Compute SHA-256 fingerprint for deduplicating identical experiences."""
    norm_type = task_type.strip().lower()
    norm_sym = re.sub(r"\s+", " ", symptom.strip().lower())
    norm_root = re.sub(r"\s+", " ", root_cause.strip().lower())
    norm_out = outcome.strip().lower()

    content = f"{norm_type}|{norm_sym}|{norm_root}|{norm_out}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class ExperienceQualityGate:
    """Quality gate validating whether an experience record qualifies for long-term storage."""

    def __init__(self, min_confidence: float = 0.60):
        self.min_confidence = min_confidence

    def should_persist(
        self,
        record: ExperienceRecord,
        trace: ExecutionTrace,
    ) -> tuple[bool, str]:
        """Evaluate if an experience record meets storage quality criteria.

        Returns:
            Tuple of (passes_gate, rejection_reason)
        """
        # Rule 1: Trace must not be empty
        tool_seq = trace.get_tool_sequence()
        if not tool_seq and not trace.verification_results:
            return False, "Trace contains no tool executions or verification results"

        # Rule 2: Reject trivial aborted tasks
        if record.outcome == ExperienceOutcome.ABORTED:
            return False, "Aborted task without actionable insight"

        # Rule 3: For failed tasks, require a captured symptom
        if record.outcome in (ExperienceOutcome.FAILED_TOOL, ExperienceOutcome.FAILED_VERIFICATION):
            if not record.symptom:
                return False, "Failed task has no identifiable symptom"

        # Rule 4: Confidence threshold
        if record.confidence < self.min_confidence:
            return False, f"Confidence {record.confidence:.2f} below threshold {self.min_confidence:.2f}"

        # Rule 5: SUCCESS_UNVERIFIED requires higher threshold
        if record.outcome == ExperienceOutcome.SUCCESS_UNVERIFIED:
            if record.confidence < 0.75:
                return False, "Unverified success requires confidence >= 0.75"
            if len(record.strategy) < 1:
                return False, "Unverified success requires non-empty strategy"

        # Rule 6: Secret leakage check
        combined_text = f"{record.symptom} {record.root_cause} {' '.join(record.lessons_learned)}"
        if "[REDACTED]" in combined_text and len(combined_text.replace("[REDACTED]", "").strip()) < 10:
            return False, "Record content consists only of redacted secrets"

        return True, ""


class ExperienceExtractor:
    """Deterministic, rule-based experience extractor."""

    _TASK_KEYWORDS = {
        "bug_fix": ("fix", "bug", "error", "issue", "crash", "traceback", "exception", "failed"),
        "test_failure": ("test", "pytest", "assert", "fail", "broken test"),
        "dependency_issue": ("import", "module", "requirement", "pip", "package", "not found"),
        "refactoring": ("refactor", "cleanup", "clean up", "rename", "restructure"),
        "feature_impl": ("add", "implement", "create", "support", "new feature"),
    }

    def classify_task_type(self, description: str, tool_seq: list[str]) -> str:
        """Classify task type based on description keywords and tool usage."""
        desc_lower = description.lower()
        for task_type, keywords in self._TASK_KEYWORDS.items():
            if any(k in desc_lower for k in keywords):
                return task_type

        # Tool-based heuristic fallback
        if "replace_file_content" in tool_seq or "write_to_file" in tool_seq:
            return "implementation"
        if "view_file" in tool_seq or "grep_search" in tool_seq:
            return "exploration"

        return "general"

    def classify_outcome(self, trace: ExecutionTrace) -> ExperienceOutcome:
        """Deterministically determine outcome from trace execution results."""
        if trace.status == "aborted":
            return ExperienceOutcome.ABORTED

        # Check verification results first (most authoritative)
        has_passed_verif = any(v.get("passed") is True for v in trace.verification_results)
        has_failed_verif = any(v.get("passed") is False for v in trace.verification_results)

        if has_passed_verif:
            return ExperienceOutcome.SUCCESS_VERIFIED

        if has_failed_verif:
            return ExperienceOutcome.FAILED_VERIFICATION

        # Check tool execution results
        errors = trace.get_errors()
        if errors and trace.status not in ("completed", "success"):
            return ExperienceOutcome.FAILED_TOOL

        if trace.status in ("completed", "success") or (trace.events and not errors):
            return ExperienceOutcome.SUCCESS_UNVERIFIED

        return ExperienceOutcome.FAILED_TOOL

    def extract_root_cause_and_symptom(
        self,
        trace: ExecutionTrace,
    ) -> tuple[str, str]:
        """Extract proven factual root cause and symptom from trace errors.

        Never speculates: returns empty root cause unless proven by execution evidence.
        """
        errors = trace.get_errors()
        if not errors:
            return "", ""

        # Symptom: first prominent error line or exception
        first_err = errors[0].strip()
        symptom = sanitize_text(first_err.splitlines()[-1] if first_err else "", max_length=200)

        # Root cause: look for explicit Python exception names or exit codes
        root_cause = ""
        exc_match = re.search(r"([A-Za-z0-9_]+Error|Exception|AssertionError):\s*([^\n\r]+)", first_err)
        if exc_match:
            root_cause = sanitize_text(f"{exc_match.group(1)}: {exc_match.group(2).strip()}", max_length=200)
        elif "command not found" in first_err.lower():
            root_cause = "Missing executable dependency"
        elif "permission denied" in first_err.lower():
            root_cause = "Filesystem permission restriction"

        return symptom, root_cause

    def extract_strategy(self, trace: ExecutionTrace) -> list[str]:
        """Extract deduplicated high-level tool and action sequence."""
        raw_seq = trace.get_tool_sequence()
        compact_seq: list[str] = []
        for tool in raw_seq:
            if not compact_seq or compact_seq[-1] != tool:
                compact_seq.append(tool)
        return compact_seq

    def extract(self, trace: ExecutionTrace) -> ExperienceRecord | None:
        """Extract structured experience record from execution trace."""
        tool_seq = trace.get_tool_sequence()
        task_type = self.classify_task_type(trace.task_description, tool_seq)
        outcome = self.classify_outcome(trace)
        symptom, root_cause = self.extract_root_cause_and_symptom(trace)
        strategy = self.extract_strategy(trace)

        # Verification evidence
        verification = None
        for v in reversed(trace.verification_results):
            verification = VerificationEvidence(
                command=v.get("command", ""),
                exit_code=v.get("exit_code", 0),
                passed=v.get("passed", False),
                evidence_snippet=sanitize_text(v.get("output", ""), max_length=300),
            )
            break

        # Confidence calculation
        confidence = 0.50
        if outcome == ExperienceOutcome.SUCCESS_VERIFIED:
            confidence += 0.35
            if root_cause:
                confidence += 0.10
        elif outcome == ExperienceOutcome.FAILED_VERIFICATION:
            confidence += 0.25  # High confidence failure signal
        elif outcome == ExperienceOutcome.FAILED_TOOL:
            confidence += 0.20
        elif outcome == ExperienceOutcome.SUCCESS_UNVERIFIED:
            confidence += 0.25 if strategy else 0.10

        confidence = min(1.0, max(0.1, confidence))

        # Generate rule-based lessons learned
        lessons: list[str] = []
        if outcome == ExperienceOutcome.SUCCESS_VERIFIED:
            lessons.append(
                f"Resolved {task_type} using workflow [{', '.join(strategy)}] confirmed by verification."
            )
        elif outcome in (ExperienceOutcome.FAILED_TOOL, ExperienceOutcome.FAILED_VERIFICATION):
            lessons.append(
                f"Encountered {symptom or 'failure'} when applying [{', '.join(strategy)}]; check root cause: {root_cause or 'unknown'}."
            )
        elif outcome == ExperienceOutcome.SUCCESS_UNVERIFIED:
            lessons.append(f"Completed {task_type} via [{', '.join(strategy)}], pending formal verification.")

        fingerprint = compute_experience_fingerprint(
            task_type=task_type,
            symptom=symptom,
            root_cause=root_cause,
            outcome=outcome.value,
        )

        return ExperienceRecord(
            task_id=trace.task_id,
            task_type=task_type,
            outcome=outcome,
            task_description=trace.task_description,
            symptom=symptom,
            root_cause=root_cause,
            strategy=strategy,
            verification=verification,
            lessons_learned=lessons,
            confidence=confidence,
            fingerprint=fingerprint,
        )


def format_experience_content(record: ExperienceRecord) -> str:
    """Format an experience record into human-readable memory content."""
    lines = [
        f"Experience [{record.task_type}]: outcome={record.outcome.value}",
    ]
    if record.task_description:
        lines.append(f"Task: {record.task_description}")
    lines.append(f"Strategy: {' -> '.join(record.strategy) if record.strategy else 'None'}")
    if record.symptom:
        lines.append(f"Symptom: {record.symptom}")
    if record.root_cause:
        lines.append(f"Root Cause: {record.root_cause}")
    if record.verification:
        status = "PASSED" if record.verification.passed else "FAILED"
        lines.append(f"Verification: {record.verification.command} ({status})")
    if record.lessons_learned:
        lines.append(f"Lesson: {record.lessons_learned[0]}")

    return "\n".join(lines)


def experience_to_memory_entry(
    record: ExperienceRecord,
    scope: MemoryScope = MemoryScope.PROJECT,
) -> MemoryEntry:
    """Convert an ExperienceRecord into a standard MemoryEntry with metadata."""
    content = format_experience_content(record)
    tags = ["experience", record.task_type, record.outcome.value]
    if record.task_description:
        desc_tags = [w.lower() for w in re.findall(r"[A-Za-z0-9_]{3,}", record.task_description)[:5]]
        tags.extend(desc_tags)
    if record.symptom:
        # Add basic normalized words from symptom as tags
        sym_tags = [w.lower() for w in re.findall(r"[A-Za-z0-9_]{3,}", record.symptom)[:3]]
        tags.extend(sym_tags)

    entry_id = f"{scope.value}-exp-{record.fingerprint}"
    return MemoryEntry(
        id=entry_id,
        scope=scope,
        category="experience",
        content=content,
        tags=list(dict.fromkeys(tags)),
        metadata={
            "experience": record.to_dict(),
            "fingerprint": record.fingerprint,
        },
    )


@dataclass
class ExperienceMemoryMetrics:
    """Metrics tracking experience lifecycle."""
    extracted_count: int = 0
    persisted_count: int = 0
    rejected_count: int = 0
    dedup_count: int = 0
    reused_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
