"""Context budget management and classification for MiniCode.

Coordinates context allocation across conversation turns, classifies messages into
zones, computes deterministic importance scores, and formulates budget plans
(KEEP, COMPRESS, OFFLOAD, EVICT) without altering existing compaction primitives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from minicode.context_manager import (
    estimate_message_tokens,
    estimate_tokens,
    get_model_context_window,
)
from minicode.logging_config import get_logger

logger = get_logger("context_budget")


class ContextAction(str, Enum):
    """Actions decided by ContextBudgetManager."""
    KEEP = "keep"
    COMPRESS = "compress"
    OFFLOAD = "offload"
    EVICT = "evict"


class ContextZone(str, Enum):
    """Semantic context classification zones."""
    CRITICAL = "critical"
    ACTIVE_TASK = "active_task"
    VERIFICATION = "verification"
    ERROR_EVIDENCE = "error_evidence"
    MEMORY = "memory"
    TOOL_EVIDENCE = "tool_evidence"
    CONVERSATION = "conversation"
    EPHEMERAL = "ephemeral"


@dataclass
class ContextBudgetConfig:
    """Centralized configuration for context budget planning and scoring."""
    safety_margin: float = 0.05
    offload_threshold_tokens: int = 500  # Tool results >= this are candidates for offload
    compress_threshold_tokens: int = 200
    max_artifact_preview_chars: int = 400
    max_artifact_recovery_chars: int = 8000
    stale_message_threshold: int = 6

    # Zone base weights (0.0 to 1.0)
    zone_weights: dict[str, float] = field(
        default_factory=lambda: {
            ContextZone.CRITICAL.value: 1.00,
            ContextZone.VERIFICATION.value: 0.95,
            ContextZone.ERROR_EVIDENCE.value: 0.95,
            ContextZone.ACTIVE_TASK.value: 0.90,
            ContextZone.MEMORY.value: 0.70,
            ContextZone.TOOL_EVIDENCE.value: 0.55,
            ContextZone.CONVERSATION.value: 0.40,
            ContextZone.EPHEMERAL.value: 0.10,
        }
    )

    # Dynamic modifiers
    recency_weight: float = 0.15
    current_file_boost: float = 0.10
    verified_experience_boost: float = 0.15
    staleness_penalty: float = 0.15
    redundancy_penalty: float = 0.20


@dataclass
class ContextItem:
    """Individual context message representation in budget analysis."""
    item_id: str
    message_index: int
    zone: ContextZone
    role: str
    source: str
    estimated_tokens: int
    importance_score: float = 0.0
    recency_score: float = 0.0
    compressible: bool = True
    externalizable: bool = False
    recoverable: bool = False
    protected: bool = False
    artifact_id: str | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class ContextDecision:
    """Deterministic planning decision for a context item."""
    item_id: str
    message_index: int
    action: ContextAction
    score: float
    original_tokens: int
    target_tokens: int
    reason: str


@dataclass
class ContextBudgetPlan:
    """Budget plan formulated for a model request."""
    budget_tokens: int
    reserved_output_tokens: int
    available_input_tokens: int
    tokens_before: int
    tokens_after_estimate: int
    keep_count: int = 0
    compress_count: int = 0
    offload_count: int = 0
    evict_count: int = 0
    decisions: list[ContextDecision] = field(default_factory=list)
    fallback_attempted: bool = False
    fallback_effective: bool = False
    budget_compliant: bool = True
    budget_violation_reason: str | None = None


@dataclass
class ContextBudgetMetrics:
    """Observable metrics for context budget planning and execution."""
    plans_created: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    kept_tokens: int = 0
    compressed_tokens: int = 0
    offloaded_tokens: int = 0
    evicted_tokens: int = 0
    protected_items: int = 0
    artifact_count: int = 0
    artifact_recovery_count: int = 0
    budget_violations: int = 0
    critical_retention_count: int = 0
    recovery_failures: int = 0
    fallback_count: int = 0
    fallback_success_count: int = 0


# Test framework output patterns
_TEST_FRAMEWORK_PATTERNS = (
    re.compile(r"===+\s*test session starts\s*===+", re.IGNORECASE),
    re.compile(r"\b\d+\s+passed\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+failed\b", re.IGNORECASE),
    re.compile(r"\bpytest\b", re.IGNORECASE),
    re.compile(r"\bunittest\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+test\b", re.IGNORECASE),
)

_ERROR_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"\b(?:Error|Exception|Failed|Failure):", re.IGNORECASE),
    re.compile(r"\bcommand failed with exit code\b", re.IGNORECASE),
    re.compile(r"\bFAILED\s+[^\n]+::", re.IGNORECASE),
)

_TEST_PASS_PATTERNS = (
    re.compile(r"\b\d+\s+passed\b", re.IGNORECASE),
    re.compile(r"\bpassed in\b", re.IGNORECASE),
    re.compile(r"\bPASSED\b"),
)

_TEST_FAIL_PATTERNS = (
    re.compile(r"\b\d+\s+failed\b", re.IGNORECASE),
    re.compile(r"\bFAILED\b"),
    re.compile(r"\b(?:failures?|errors?):\b", re.IGNORECASE),
)

# Natural language and explicit constraint detection patterns
_CONSTRAINT_PATTERNS = (
    re.compile(r"\b(?:critical\s+)?constraints?:", re.IGNORECASE),
    re.compile(r"\b(?:security|safety)\s+(?:rules?|constraints?)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:do\s+not|don't|must\s+not|you\s+must\s+not|cannot|can't|never)\s+"
        r"(?:modify|edit|delete|remove|run|change|touch|alter|commit|execute|import|use|break|drop|mutate)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:use\s+only|strictly\s+use|only\s+use)\b", re.IGNORECASE),
    re.compile(r"\bkeep\s+(?:the\s+)?[^\n\.\,]+backward[s]?\s+compatible\b", re.IGNORECASE),
    re.compile(r"\bbackward[s]?\s+compatibility\s+(?:is\s+required|must\s+be\s+maintained)\b", re.IGNORECASE),
    re.compile(r"\b(?:strictly\s+prohibited|forbidden|mandatory|strictly\s+required)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+(?:break|remove|rename)\b", re.IGNORECASE),
)


def is_constraint_text(text: str) -> bool:
    """Deterministic check for explicit or natural-language task constraints."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _CONSTRAINT_PATTERNS)


_SALIENT_ERROR_LINE = re.compile(
    r"(?:AssertionError|Error|Exception|FAILED|Traceback|failed with exit code)",
    re.IGNORECASE,
)
_SALIENT_VERIF_LINE = re.compile(
    r"(?:PASSED|\bpassed in\b|\b\d+ passed\b|\btest session starts\b)",
    re.IGNORECASE,
)
_SALIENT_FILE_PATH = re.compile(
    r"\b(?:[a-zA-Z0-9_\.\-]+/[a-zA-Z0-9_\.\-]+(?:\.[a-zA-Z0-9]+)?)\b"
)


def build_compact_preview(content: str, max_chars: int = 180) -> str:
    """Build a concise, semantically rich preview of content for compression.
    
    Extracts salient lines (constraints, errors, verifications, file paths, head/tail)
    rather than naive string slicing.
    """
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines:
        return ""

    salient_lines: list[str] = []

    # 1. Constraint lines (never drop constraints)
    for ln in lines:
        if is_constraint_text(ln):
            salient_lines.append(ln)
            break

    # 2. Error line
    for ln in lines:
        if _SALIENT_ERROR_LINE.search(ln) and ln not in salient_lines:
            salient_lines.append(ln)
            break

    # 3. Verification line
    for ln in lines:
        if _SALIENT_VERIF_LINE.search(ln) and ln not in salient_lines:
            salient_lines.append(ln)
            break

    # 4. File path line
    for ln in lines:
        if _SALIENT_FILE_PATH.search(ln) and ln not in salient_lines and len(salient_lines) < 2:
            salient_lines.append(ln)
            break

    # 5. Head and tail lines if salient lines are sparse
    if not salient_lines:
        salient_lines.append(lines[0])
        if len(lines) > 1 and lines[-1] != lines[0]:
            salient_lines.append(lines[-1])
    elif len(salient_lines) == 1 and len(lines) > 1:
        if lines[0] not in salient_lines:
            salient_lines.insert(0, lines[0])

    combined = " | ".join(salient_lines)
    if len(combined) > max_chars:
        combined = combined[: max_chars - 3] + "..."
    return f"[Summary: {combined}]"


def validate_tool_pair_integrity(messages: list[dict[str, Any]]) -> bool:
    """Validate that all tool calls and tool results have intact pairs and matching IDs.
    
    Returns False if:
    - There is a tool_result without a preceding matching assistant tool call
    - There is an assistant tool call without a matching tool_result
    - The tool call ID / toolUseId is missing or mismatched
    """
    call_ids: set[str] = set()
    open_calls: list[str] = []

    for msg in messages:
        role = msg.get("role")
        if role == "assistant_tool_call":
            call_id = msg.get("toolUseId") or msg.get("id") or msg.get("tool_call_id")
            if not call_id:
                return False
            call_ids.add(str(call_id))
            open_calls.append(str(call_id))
        elif role == "assistant" and "tool_calls" in msg:
            for tc in msg.get("tool_calls", []):
                call_id = tc.get("id")
                if not call_id:
                    return False
                call_ids.add(str(call_id))
                open_calls.append(str(call_id))
        elif role in ("tool", "tool_result"):
            res_id = msg.get("toolUseId") or msg.get("tool_call_id") or msg.get("id")
            if not res_id:
                return False
            res_id_str = str(res_id)
            if res_id_str not in call_ids:
                return False
            if res_id_str in open_calls:
                open_calls.remove(res_id_str)

    return len(open_calls) == 0


def sanitize_tool_pair_invariants(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize message sequence to preserve tool call / result pair invariants.
    
    If an external compactor or slicing operation produced orphan tool results (missing their
    preceding tool call) or dangling tool calls (missing their tool result), cleans them up
    so model provider APIs and integrity checks pass.
    """
    call_ids: set[str] = set()
    result_ids: set[str] = set()

    for msg in messages:
        role = str(msg.get("role", ""))
        if role == "assistant_tool_call":
            cid = msg.get("toolUseId") or msg.get("id") or msg.get("tool_call_id")
            if cid:
                call_ids.add(str(cid))
        elif role == "assistant" and "tool_calls" in msg:
            for tc in msg.get("tool_calls", []):
                cid = tc.get("id")
                if cid:
                    call_ids.add(str(cid))
        elif role in ("tool", "tool_result"):
            rid = msg.get("toolUseId") or msg.get("tool_call_id") or msg.get("id")
            if rid:
                result_ids.add(str(rid))

    # Only pairs that have BOTH call and result are valid
    valid_pair_ids = call_ids & result_ids

    sanitized: list[dict[str, Any]] = []
    seen_calls: set[str] = set()

    for msg in messages:
        role = str(msg.get("role", ""))
        if role == "assistant_tool_call":
            cid = msg.get("toolUseId") or msg.get("id") or msg.get("tool_call_id")
            if cid and str(cid) not in valid_pair_ids:
                continue
            if cid:
                seen_calls.add(str(cid))
            sanitized.append(msg)
        elif role == "assistant" and "tool_calls" in msg:
            valid_calls = [tc for tc in msg.get("tool_calls", []) if str(tc.get("id", "")) in valid_pair_ids]
            for tc in valid_calls:
                seen_calls.add(str(tc.get("id", "")))
            if not valid_calls and not msg.get("content"):
                continue
            msg_copy = dict(msg)
            if valid_calls:
                msg_copy["tool_calls"] = valid_calls
            else:
                msg_copy.pop("tool_calls", None)
            sanitized.append(msg_copy)
        elif role in ("tool", "tool_result"):
            rid = msg.get("toolUseId") or msg.get("tool_call_id") or msg.get("id")
            if rid and str(rid) not in valid_pair_ids:
                continue
            if rid and str(rid) not in seen_calls:
                continue
            sanitized.append(msg)
        else:
            sanitized.append(msg)

    return sanitized


class ContextBudgetManager:
    """Policy and planning engine for layered context engineering.

    Determines what to keep, compress, offload, or evict while ensuring
    critical task constraints, latest verification evidence, and stable task
    states are preserved.
    """

    def __init__(
        self,
        config: ContextBudgetConfig | None = None,
        workspace: str | None = None,
    ):
        self.config = config or ContextBudgetConfig()
        self.workspace = workspace
        self.metrics = ContextBudgetMetrics()

    def classify_all(
        self,
        messages: list[dict[str, Any]],
        active_files: set[str] | None = None,
    ) -> list[ContextItem]:
        """Classify and score all messages into context items."""
        total = len(messages)
        if total == 0:
            return []

        active_files = active_files or set()

        # Pre-scan for latest verification and latest unresolved error
        latest_verification_idx = -1
        active_task_idx = -1

        # Track error records: (index, is_test_error, tool_name)
        error_records: list[tuple[int, bool, str]] = []
        # Track test pass indices
        test_pass_indices: list[int] = []
        # Track successful tool executions: (index, tool_name)
        tool_success_indices: list[tuple[int, str]] = []

        for i, msg in enumerate(messages):
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", "") or "")
            tool_name = str(msg.get("toolName", "") or "").lower()
            is_err = bool(msg.get("isError", False))

            if role == "user":
                active_task_idx = i

            # Check verification
            is_verif = (
                role in ("tool", "tool_result")
                or tool_name in ("test_runner", "pytest", "run_command")
            ) and (any(p.search(content) for p in _TEST_FRAMEWORK_PATTERNS) or "pytest" in tool_name)

            if is_verif:
                latest_verification_idx = i
                has_fail = (
                    is_err
                    or any(p.search(content) for p in _TEST_FAIL_PATTERNS)
                    or any(p.search(content) for p in _ERROR_PATTERNS)
                )
                has_pass = any(p.search(content) for p in _TEST_PASS_PATTERNS)
                if has_pass and not has_fail:
                    test_pass_indices.append(i)

            # Check error vs success
            has_error = (
                is_err
                or (role in ("tool", "tool_result") and any(p.search(content) for p in _ERROR_PATTERNS))
                or any(p.search(content) for p in _TEST_FAIL_PATTERNS)
            )
            if has_error:
                error_records.append((i, is_verif, tool_name))
            elif role in ("tool", "tool_result"):
                tool_success_indices.append((i, tool_name))

        # Determine which errors are unresolved
        latest_error_idx = -1
        unresolved_error_indices: list[int] = []

        for err_idx, is_test_err, t_name in error_records:
            if is_test_err:
                # Test error is resolved if a subsequent test verification passed
                if any(pass_idx > err_idx for pass_idx in test_pass_indices):
                    continue
                unresolved_error_indices.append(err_idx)
            else:
                # Generic tool error is resolved if the same tool succeeded later
                if any(succ_idx > err_idx and (not t_name or succ_tool == t_name) for succ_idx, succ_tool in tool_success_indices):
                    continue
                unresolved_error_indices.append(err_idx)

        if unresolved_error_indices:
            latest_error_idx = max(unresolved_error_indices)

        items: list[ContextItem] = []
        for i, msg in enumerate(messages):
            item = self.classify(
                msg,
                index=i,
                total_messages=total,
                active_task_index=active_task_idx,
                latest_verification_index=latest_verification_idx,
                latest_error_index=latest_error_idx,
            )
            item.importance_score = self.score_importance(item, active_files=active_files, total_messages=total)
            items.append(item)

        return items

    def classify(
        self,
        message: dict[str, Any],
        index: int,
        total_messages: int,
        active_task_index: int = -1,
        latest_verification_index: int = -1,
        latest_error_index: int = -1,
    ) -> ContextItem:
        """Deterministically classify a single message into a ContextZone."""
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", "") or "")
        tool_name = str(message.get("toolName", "") or "")
        is_error = bool(message.get("isError", False))

        estimated_tokens = estimate_message_tokens(message)
        item_id = f"item_{index}_{role}"
        reasons: list[str] = []

        # 1. Ephemeral checks
        if role in ("assistant_progress", "progress", "thought", "thinking"):
            return ContextItem(
                item_id=item_id,
                message_index=index,
                zone=ContextZone.EPHEMERAL,
                role=role,
                source=tool_name or "progress",
                estimated_tokens=estimated_tokens,
                compressible=True,
                externalizable=False,
                recoverable=False,
                protected=False,
                reasons=["ephemeral_progress_role"],
            )

        # 2. Critical: System prompt & Stable task state
        if role == "system":
            is_stable_task = "[Stable task state]" in content or "STABLE_TASK_HEADER" in content
            if is_stable_task:
                reasons.append("stable_task_state")
            else:
                reasons.append("system_prompt")
            return ContextItem(
                item_id=item_id,
                message_index=index,
                zone=ContextZone.CRITICAL,
                role=role,
                source="system",
                estimated_tokens=estimated_tokens,
                compressible=False,
                externalizable=False,
                recoverable=False,
                protected=True,
                reasons=reasons,
            )

        if "[Stable task state]" in content:
            return ContextItem(
                item_id=item_id,
                message_index=index,
                zone=ContextZone.CRITICAL,
                role=role,
                source="stable_task",
                estimated_tokens=estimated_tokens,
                compressible=False,
                externalizable=False,
                recoverable=False,
                protected=True,
                reasons=["stable_task_state"],
            )

        if is_constraint_text(content):
            return ContextItem(
                item_id=item_id,
                message_index=index,
                zone=ContextZone.CRITICAL,
                role=role,
                source="constraint",
                estimated_tokens=estimated_tokens,
                compressible=False,
                externalizable=False,
                recoverable=False,
                protected=True,
                reasons=["task_constraint"],
            )

        # 3. Memory: Injected Experience Memory
        if (
            "[Verified Experience]" in content
            or "[Past Failure Pattern]" in content
            or "## Relevant Coding Experience" in content
            or "[Relevant Experience]" in content
        ):
            is_verified = "[Verified Experience]" in content
            reasons.append("verified_experience" if is_verified else "past_experience")
            return ContextItem(
                item_id=item_id,
                message_index=index,
                zone=ContextZone.MEMORY,
                role=role,
                source="memory",
                estimated_tokens=estimated_tokens,
                compressible=True,
                externalizable=False,
                recoverable=False,
                protected=False,
                reasons=reasons,
            )

        # 4. Active Task: Most recent user request
        if role == "user" and (index == active_task_index or index >= total_messages - 2):
            return ContextItem(
                item_id=item_id,
                message_index=index,
                zone=ContextZone.ACTIVE_TASK,
                role=role,
                source="user",
                estimated_tokens=estimated_tokens,
                compressible=False,
                externalizable=False,
                recoverable=False,
                protected=True,
                reasons=["active_user_task"],
            )

        # 5. Verification: Test runner / pytest evidence
        is_test_evidence = (
            tool_name in ("test_runner", "pytest")
            or any(p.search(content) for p in _TEST_FRAMEWORK_PATTERNS)
        )
        if (role in ("tool", "tool_result") and is_test_evidence) or index == latest_verification_index:
            is_latest = (index == latest_verification_index)
            is_large = estimated_tokens >= self.config.offload_threshold_tokens
            if is_latest:
                reasons.append("latest_verification")
                if is_error or any(p.search(content) for p in _ERROR_PATTERNS) or any(p.search(content) for p in _TEST_FAIL_PATTERNS):
                    reasons.append("unresolved_verification_failure")
            else:
                reasons.append("earlier_verification")
            return ContextItem(
                item_id=item_id,
                message_index=index,
                zone=ContextZone.VERIFICATION,
                role=role,
                source=tool_name or "test_runner",
                estimated_tokens=estimated_tokens,
                compressible=not is_latest,
                externalizable=is_large,
                recoverable=True,
                protected=is_latest,  # Latest verification must be protected
                reasons=reasons,
            )

        # 6. Error Evidence: Exceptions / tool execution failures
        has_error_pattern = is_error or any(p.search(content) for p in _ERROR_PATTERNS)
        if (role in ("tool", "tool_result") and has_error_pattern) or index == latest_error_index:
            is_latest = (index == latest_error_index)
            is_large = estimated_tokens >= self.config.offload_threshold_tokens
            reasons.append("latest_error_evidence" if is_latest else "earlier_error_evidence")
            return ContextItem(
                item_id=item_id,
                message_index=index,
                zone=ContextZone.ERROR_EVIDENCE,
                role=role,
                source=tool_name or "error",
                estimated_tokens=estimated_tokens,
                compressible=not is_latest,
                externalizable=is_large,
                recoverable=True,
                protected=is_latest,  # Latest unresolved error is protected
                reasons=reasons,
            )

        # 7. Tool Evidence: Normal tool calls and results (read_file, grep, etc.)
        if role in ("assistant_tool_call", "tool_call"):
            is_associated_protected = (
                (index + 1 == latest_verification_index)
                or (index + 1 == latest_error_index)
            )
            reasons.append("tool_call")
            return ContextItem(
                item_id=item_id,
                message_index=index,
                zone=ContextZone.TOOL_EVIDENCE,
                role=role,
                source=tool_name or "tool_call",
                estimated_tokens=estimated_tokens,
                compressible=False,
                externalizable=False,
                recoverable=False,
                protected=is_associated_protected,
                reasons=reasons,
            )

        if role in ("tool", "tool_result"):
            is_large = estimated_tokens >= self.config.offload_threshold_tokens
            reasons.append("tool_result")
            return ContextItem(
                item_id=item_id,
                message_index=index,
                zone=ContextZone.TOOL_EVIDENCE,
                role=role,
                source=tool_name or "tool",
                estimated_tokens=estimated_tokens,
                compressible=True,
                externalizable=is_large,
                recoverable=True,
                protected=False,
                reasons=reasons,
            )

        # 8. Conversation: User / Assistant dialogue
        is_recent_assistant = role == "assistant" and index >= total_messages - 2
        return ContextItem(
            item_id=item_id,
            message_index=index,
            zone=ContextZone.ACTIVE_TASK if is_recent_assistant else ContextZone.CONVERSATION,
            role=role,
            source=role,
            estimated_tokens=estimated_tokens,
            compressible=True,
            externalizable=False,
            recoverable=False,
            protected=is_recent_assistant,
            reasons=["recent_assistant_response" if is_recent_assistant else "past_conversation"],
        )

    def score_importance(
        self,
        item: ContextItem,
        active_files: set[str] | None = None,
        total_messages: int = 1,
    ) -> float:
        """Compute deterministic importance score in range [0.0, 1.0]."""
        active_files = active_files or set()
        cfg = self.config

        base_weight = cfg.zone_weights.get(item.zone.value, 0.40)
        recency_ratio = item.message_index / max(total_messages - 1, 1)
        recency_boost = recency_ratio * cfg.recency_weight
        item.recency_score = recency_boost

        score = base_weight + recency_boost

        # Current file boost
        if active_files:
            for f in active_files:
                if f and f in item.source:
                    score += cfg.current_file_boost
                    item.reasons.append(f"active_file_boost:{f}")
                    break

        # Verified experience boost
        if item.zone == ContextZone.MEMORY and "verified_experience" in item.reasons:
            score += cfg.verified_experience_boost

        # Staleness penalty for older tool outputs and conversation
        age = total_messages - 1 - item.message_index
        if age > cfg.stale_message_threshold:
            if item.zone in (ContextZone.TOOL_EVIDENCE, ContextZone.CONVERSATION):
                score -= cfg.staleness_penalty
                item.reasons.append("staleness_penalty")

        # Clamp score
        score = max(0.0, min(1.0, score))

        # Critical items maintain priority
        if item.protected:
            score = max(score, 0.90)

        return round(score, 4)

    def plan(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        available_budget: int | None = None,
        active_files: set[str] | None = None,
        metrics: ContextBudgetMetrics | None = None,
    ) -> ContextBudgetPlan:
        """Formulate deterministic budget plan based on model context window."""
        active_metrics = metrics or self.metrics
        active_metrics.plans_created += 1

        mcw = get_model_context_window(model or "default")
        budget = available_budget
        if budget is None:
            budget = int(mcw.effective_input * (1.0 - self.config.safety_margin))

        items = self.classify_all(messages, active_files=active_files)
        tokens_before = sum(item.estimated_tokens for item in items)
        active_metrics.tokens_before += tokens_before

        decisions: list[ContextDecision] = []
        tokens_to_free = max(0, tokens_before - budget)

        # First pass: mark protected and items that fit cleanly
        remaining_budget = budget
        current_estimate = tokens_before

        # Sort items by ascending importance score, with tie-break by ascending index (older first)
        # Protected items are sorted to the very end
        ranked_indices = sorted(
            range(len(items)),
            key=lambda idx: (1 if items[idx].protected else 0, items[idx].importance_score, items[idx].message_index),
        )

        decisions_by_idx: dict[int, ContextDecision] = {}

        # Pass 1: If over budget, evict ephemeral items first
        if tokens_before > budget:
            for idx, item in enumerate(items):
                if item.zone == ContextZone.EPHEMERAL:
                    decisions_by_idx[idx] = ContextDecision(
                        item_id=item.item_id,
                        message_index=idx,
                        action=ContextAction.EVICT,
                        score=item.importance_score,
                        original_tokens=item.estimated_tokens,
                        target_tokens=0,
                        reason="evict_ephemeral",
                    )
                    current_estimate -= item.estimated_tokens

        # Pass 2: Preemptively offload oversized externalizable tool results (>= offload_threshold_tokens)
        for idx, item in enumerate(items):
            if idx in decisions_by_idx:
                continue
            if item.externalizable and item.estimated_tokens >= self.config.offload_threshold_tokens:
                preview_tokens = min(100, item.estimated_tokens // 4)
                decisions_by_idx[idx] = ContextDecision(
                    item_id=item.item_id,
                    message_index=idx,
                    action=ContextAction.OFFLOAD,
                    score=item.importance_score,
                    original_tokens=item.estimated_tokens,
                    target_tokens=preview_tokens,
                    reason=f"offload_oversized_{item.zone.value}",
                )
                current_estimate -= (item.estimated_tokens - preview_tokens)

        # Pass 2: Process remaining unassigned items
        for idx in ranked_indices:
            if idx in decisions_by_idx:
                continue
            item = items[idx]

            # If under budget, keep everything remaining
            if current_estimate <= budget:
                decisions_by_idx[idx] = ContextDecision(
                    item_id=item.item_id,
                    message_index=idx,
                    action=ContextAction.KEEP,
                    score=item.importance_score,
                    original_tokens=item.estimated_tokens,
                    target_tokens=item.estimated_tokens,
                    reason="within_budget" if not item.protected else "protected_item",
                )
                continue

            # Protected items are kept (cannot be evicted)
            if item.protected:
                decisions_by_idx[idx] = ContextDecision(
                    item_id=item.item_id,
                    message_index=idx,
                    action=ContextAction.KEEP,
                    score=item.importance_score,
                    original_tokens=item.estimated_tokens,
                    target_tokens=item.estimated_tokens,
                    reason=f"protected:{','.join(item.reasons)}",
                )
                continue

            # 1. Ephemeral items -> EVICT
            if item.zone == ContextZone.EPHEMERAL:
                freed = item.estimated_tokens
                current_estimate -= freed
                decisions_by_idx[idx] = ContextDecision(
                    item_id=item.item_id,
                    message_index=idx,
                    action=ContextAction.EVICT,
                    score=item.importance_score,
                    original_tokens=item.estimated_tokens,
                    target_tokens=0,
                    reason="evict_ephemeral",
                )
                continue

            # 2. Large tool results that were below preemptive threshold but still externalizable -> OFFLOAD
            if item.externalizable:
                preview_tokens = min(100, item.estimated_tokens // 4)
                freed = max(0, item.estimated_tokens - preview_tokens)
                current_estimate -= freed
                decisions_by_idx[idx] = ContextDecision(
                    item_id=item.item_id,
                    message_index=idx,
                    action=ContextAction.OFFLOAD,
                    score=item.importance_score,
                    original_tokens=item.estimated_tokens,
                    target_tokens=preview_tokens,
                    reason=f"offload_large_{item.zone.value}",
                )
                continue

            # 3. Compressible conversation / tool evidence -> COMPRESS
            if item.compressible and item.estimated_tokens >= 25 and item.zone in (
                ContextZone.CONVERSATION,
                ContextZone.TOOL_EVIDENCE,
                ContextZone.MEMORY,
                ContextZone.VERIFICATION,
                ContextZone.ERROR_EVIDENCE,
            ):
                compressed_tokens = min(item.estimated_tokens, 25)
                freed = max(0, item.estimated_tokens - compressed_tokens)
                current_estimate -= freed
                decisions_by_idx[idx] = ContextDecision(
                    item_id=item.item_id,
                    message_index=idx,
                    action=ContextAction.COMPRESS,
                    score=item.importance_score,
                    original_tokens=item.estimated_tokens,
                    target_tokens=compressed_tokens,
                    reason=f"compress_{item.zone.value}",
                )
                continue

            # 4. If still over budget and item is not protected, evict low-priority conversation/tool results
            if current_estimate > budget and not item.protected and item.zone in (
                ContextZone.CONVERSATION,
                ContextZone.TOOL_EVIDENCE,
                ContextZone.MEMORY,
            ):
                tombstone_tokens = 1 if item.role in ("tool", "tool_result") else 0
                freed = max(0, item.estimated_tokens - tombstone_tokens)
                current_estimate -= freed
                decisions_by_idx[idx] = ContextDecision(
                    item_id=item.item_id,
                    message_index=idx,
                    action=ContextAction.EVICT,
                    score=item.importance_score,
                    original_tokens=item.estimated_tokens,
                    target_tokens=tombstone_tokens,
                    reason=f"evict_over_budget_{item.zone.value}",
                )
                continue

            # Default: KEEP
            decisions_by_idx[idx] = ContextDecision(
                item_id=item.item_id,
                message_index=idx,
                action=ContextAction.KEEP,
                score=item.importance_score,
                original_tokens=item.estimated_tokens,
                target_tokens=item.estimated_tokens,
                reason="preserve_by_importance",
            )

        # Order decisions in original message order
        ordered_decisions = [decisions_by_idx[i] for i in range(len(messages))]

        keep_count = sum(1 for d in ordered_decisions if d.action == ContextAction.KEEP)
        compress_count = sum(1 for d in ordered_decisions if d.action == ContextAction.COMPRESS)
        offload_count = sum(1 for d in ordered_decisions if d.action == ContextAction.OFFLOAD)
        evict_count = sum(1 for d in ordered_decisions if d.action == ContextAction.EVICT)

        plan = ContextBudgetPlan(
            budget_tokens=budget,
            reserved_output_tokens=mcw.output_reserve,
            available_input_tokens=mcw.effective_input,
            tokens_before=tokens_before,
            tokens_after_estimate=max(0, current_estimate),
            keep_count=keep_count,
            compress_count=compress_count,
            offload_count=offload_count,
            evict_count=evict_count,
            decisions=ordered_decisions,
        )

        return plan

    def apply(
        self,
        messages: list[dict[str, Any]],
        plan: ContextBudgetPlan,
        artifact_store: Any = None,
        compactor: Any = None,
        metrics: ContextBudgetMetrics | None = None,
    ) -> tuple[list[dict[str, Any]], ContextBudgetMetrics]:
        """Apply plan decisions to messages, offloading, compressing, or evicting.

        Maintains tool call/result pair invariants so model providers don't reject
        the conversation.
        """
        active_metrics = metrics or self.metrics
        modified: list[dict[str, Any]] = []

        for i, msg in enumerate(messages):
            if i >= len(plan.decisions):
                modified.append(dict(msg))
                continue

            decision = plan.decisions[i]
            action = decision.action
            role = str(msg.get("role", ""))

            if action == ContextAction.KEEP:
                active_metrics.kept_tokens += decision.original_tokens
                if "protected" in decision.reason or decision.reason == "task_constraint":
                    active_metrics.protected_items += 1
                    active_metrics.critical_retention_count += 1
                modified.append(dict(msg))

            elif action == ContextAction.OFFLOAD:
                content = str(msg.get("content", "") or "")
                tool_name = str(msg.get("toolName", "unknown"))

                if artifact_store is not None:
                    metadata = artifact_store.persist(content, tool_name=tool_name)
                    preview_text = (
                        f"[Context Artifact]\n"
                        f"id: {metadata.artifact_id}\n"
                        f"tool: {tool_name}\n"
                        f"original: {metadata.estimated_tokens} tokens ({metadata.original_chars} chars)\n"
                        f"preview:\n{metadata.preview}\n"
                        f"Use load_context_artifact if full evidence is required."
                    )
                    active_metrics.artifact_count += 1
                    active_metrics.offloaded_tokens += (decision.original_tokens - decision.target_tokens)
                    new_msg = dict(msg)
                    new_msg["content"] = preview_text
                    new_msg["_context_artifact_id"] = metadata.artifact_id
                    new_msg["_context_action"] = "offload"
                    modified.append(new_msg)
                else:
                    # Fallback preview if no artifact store
                    snippet = content[:self.config.max_artifact_preview_chars] + "\n... [offloaded]"
                    active_metrics.offloaded_tokens += (decision.original_tokens - decision.target_tokens)
                    new_msg = dict(msg)
                    new_msg["content"] = snippet
                    new_msg["_context_action"] = "offload"
                    modified.append(new_msg)

            elif action == ContextAction.COMPRESS:
                content = str(msg.get("content", "") or "")
                if len(content) <= 60:
                    compressed_content = content
                else:
                    compressed_content = build_compact_preview(content, max_chars=160)
                active_metrics.compressed_tokens += max(0, decision.original_tokens - estimate_tokens(compressed_content))
                new_msg = dict(msg)
                new_msg["content"] = compressed_content
                new_msg["_context_action"] = "compress"
                modified.append(new_msg)

            elif action == ContextAction.EVICT:
                active_metrics.evicted_tokens += decision.original_tokens
                # Invariant preservation: maintain paired tool_call and tool_result tombstones
                if role in ("tool", "tool_result"):
                    new_msg = dict(msg)
                    new_msg["content"] = "[Output cleared]"
                    new_msg["_context_action"] = "evict"
                    modified.append(new_msg)
                elif role in ("assistant_tool_call", "tool_call"):
                    new_msg = dict(msg)
                    new_msg["_context_action"] = "evict"
                    modified.append(new_msg)
                elif role in ("assistant_progress", "thought", "thinking"):
                    # Ephemeral message can be omitted
                    continue
                else:
                    # Non-tool conversational message evicted
                    continue

        # Evaluate token count after applying initial decisions
        current_tokens = sum(estimate_message_tokens(m) for m in modified)
        tokens_before_fallback = current_tokens

        # Track tokens belonging to protected items
        protected_tokens = sum(
            d.original_tokens for d in plan.decisions if d.action == ContextAction.KEEP and "protected" in d.reason
        )

        # Fallback to compactor if still over budget
        if current_tokens > plan.budget_tokens and compactor is not None:
            plan.fallback_attempted = True
            active_metrics.fallback_count += 1
            logger.info(
                "Context budget still exceeded (%d > %d), invoking compactor fallback",
                current_tokens,
                plan.budget_tokens,
            )

            # Step 1: Real ContextCompactor.process_request
            try:
                if hasattr(compactor, "process_request"):
                    compact_res = compactor.process_request(modified)
                    if compact_res and getattr(compact_res, "messages", None):
                        candidate_msgs = sanitize_tool_pair_invariants(compact_res.messages)
                        candidate_tokens = sum(estimate_message_tokens(m) for m in candidate_msgs)
                        if candidate_tokens < current_tokens:
                            modified = candidate_msgs
                            current_tokens = candidate_tokens
                            plan.fallback_effective = True
            except Exception as exc:
                logger.warning("Compactor process_request fallback failed: %s", exc)

            # Step 2: Severe overflow -> ContextCompactor.reactive_recover
            if current_tokens > plan.budget_tokens and hasattr(compactor, "reactive_recover"):
                try:
                    reactive_res = compactor.reactive_recover(
                        modified,
                        error=f"Context overflow: {current_tokens} tokens exceeds budget {plan.budget_tokens}",
                    )
                    if reactive_res and getattr(reactive_res, "messages", None):
                        candidate_msgs = sanitize_tool_pair_invariants(reactive_res.messages)
                        candidate_tokens = sum(estimate_message_tokens(m) for m in candidate_msgs)
                        if candidate_tokens < current_tokens:
                            modified = candidate_msgs
                            current_tokens = candidate_tokens
                            plan.fallback_effective = True
                except Exception as exc:
                    logger.warning("Compactor reactive_recover fallback failed: %s", exc)

            if plan.fallback_effective:
                active_metrics.fallback_success_count += 1

        modified = sanitize_tool_pair_invariants(modified)
        current_tokens = sum(estimate_message_tokens(m) for m in modified)
        plan.tokens_after_estimate = current_tokens
        if current_tokens <= plan.budget_tokens:
            plan.budget_compliant = True
            plan.budget_violation_reason = None
        else:
            plan.budget_compliant = False
            active_metrics.budget_violations += 1
            if protected_tokens > plan.budget_tokens:
                plan.budget_violation_reason = "protected_context_exceeds_budget"
            else:
                plan.budget_violation_reason = "insufficient_compaction"

        active_metrics.tokens_after += current_tokens
        return modified, active_metrics

    def plan_and_apply(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        available_budget: int | None = None,
        artifact_store: Any = None,
        compactor: Any = None,
        metrics: ContextBudgetMetrics | None = None,
        active_files: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], ContextBudgetPlan]:
        """Single entrypoint to plan and apply context budget before a model step."""
        active_metrics = metrics or self.metrics
        plan = self.plan(
            messages,
            model=model,
            available_budget=available_budget,
            active_files=active_files,
            metrics=active_metrics,
        )
        modified, _ = self.apply(
            messages,
            plan=plan,
            artifact_store=artifact_store,
            compactor=compactor,
            metrics=active_metrics,
        )
        return modified, plan
