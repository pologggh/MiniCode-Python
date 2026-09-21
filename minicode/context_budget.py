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

        # Pre-scan for latest verification and latest error
        latest_verification_idx = -1
        latest_error_idx = -1
        active_task_idx = -1

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = str(msg.get("content", "") or "")
            tool_name = str(msg.get("toolName", "") or "")

            if role == "user":
                active_task_idx = i

            # Check verification
            if role in ("tool", "tool_result") or tool_name in ("test_runner", "pytest", "run_command"):
                if any(p.search(content) for p in _TEST_FRAMEWORK_PATTERNS) or "pytest" in tool_name:
                    latest_verification_idx = i

            # Check error
            is_err = msg.get("isError") is True
            if is_err or (role in ("tool", "tool_result") and any(p.search(content) for p in _ERROR_PATTERNS)):
                latest_error_idx = i

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
            reasons.append("latest_verification" if is_latest else "earlier_verification")
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

        # 7. Tool Evidence: Normal tool results (read_file, grep, etc.)
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
    ) -> ContextBudgetPlan:
        """Formulate deterministic budget plan based on model context window."""
        self.metrics.plans_created += 1

        mcw = get_model_context_window(model or "default")
        budget = available_budget
        if budget is None:
            budget = int(mcw.effective_input * (1.0 - self.config.safety_margin))

        items = self.classify_all(messages, active_files=active_files)
        tokens_before = sum(item.estimated_tokens for item in items)
        self.metrics.tokens_before += tokens_before

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

        for idx in ranked_indices:
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

            # Protected items are kept unless catastrophic overflow
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

            # 2. Large tool results / verification -> OFFLOAD
            if item.externalizable or (
                item.estimated_tokens >= self.config.offload_threshold_tokens
                and item.zone in (ContextZone.TOOL_EVIDENCE, ContextZone.VERIFICATION, ContextZone.ERROR_EVIDENCE)
            ):
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
            if item.compressible and item.zone in (
                ContextZone.CONVERSATION,
                ContextZone.TOOL_EVIDENCE,
                ContextZone.MEMORY,
                ContextZone.VERIFICATION,
                ContextZone.ERROR_EVIDENCE,
            ):
                compressed_tokens = min(item.estimated_tokens, max(40, item.estimated_tokens // 3))
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
                if "protected" in decision.reason:
                    active_metrics.protected_items += 1
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
                # Create concise summary stub
                first_line = content.strip().split("\n")[0][:120] if content.strip() else ""
                compressed_content = (
                    f"[Compressed context: {first_line}... "
                    f"({len(content)} chars compressed to retain key semantics)]"
                )
                active_metrics.compressed_tokens += (decision.original_tokens - decision.target_tokens)
                new_msg = dict(msg)
                new_msg["content"] = compressed_content
                new_msg["_context_action"] = "compress"
                modified.append(new_msg)

            elif action == ContextAction.EVICT:
                active_metrics.evicted_tokens += decision.original_tokens
                # Invariant preservation: if role is tool_result, replace with minimal tombstone
                if role in ("tool", "tool_result"):
                    new_msg = dict(msg)
                    new_msg["content"] = "[Output cleared]"
                    new_msg["_context_action"] = "evict"
                    modified.append(new_msg)
                elif role in ("assistant_progress", "thought", "thinking"):
                    # Ephemeral message can be omitted
                    continue
                else:
                    # Non-tool conversational message evicted
                    continue

        # Check if fallback to compactor is needed
        current_tokens = sum(estimate_message_tokens(m) for m in modified)
        if current_tokens > plan.budget_tokens and compactor is not None:
            active_metrics.budget_violations += 1
            logger.info(
                "Context budget still exceeded (%d > %d), invoking compactor fallback",
                current_tokens,
                plan.budget_tokens,
            )
            try:
                if hasattr(compactor, "compact_on_high_water"):
                    compact_res = compactor.compact_on_high_water(modified)
                    if compact_res and compact_res.effective:
                        modified = compact_res.messages
                elif hasattr(compactor, "try_auto_compact"):
                    compact_res = compactor.try_auto_compact(modified)
                    if compact_res and compact_res.effective:
                        modified = compact_res.messages
            except Exception as exc:
                logger.warning("Compactor fallback failed: %s", exc)

        active_metrics.tokens_after += sum(estimate_message_tokens(m) for m in modified)
        return modified, active_metrics

    def plan_and_apply(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        artifact_store: Any = None,
        compactor: Any = None,
        metrics: ContextBudgetMetrics | None = None,
        active_files: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], ContextBudgetPlan]:
        """Single entrypoint to plan and apply context budget before a model step."""
        plan = self.plan(messages, model=model, active_files=active_files)
        modified, _ = self.apply(
            messages,
            plan=plan,
            artifact_store=artifact_store,
            compactor=compactor,
            metrics=metrics or self.metrics,
        )
        return modified, plan
