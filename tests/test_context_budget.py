"""Tests for ContextBudgetManager and context zone classification."""
from __future__ import annotations

import pytest

from minicode.context_budget import (
    ContextAction,
    ContextBudgetConfig,
    ContextBudgetManager,
    ContextItem,
    ContextZone,
)


def test_context_zone_classification_critical():
    mgr = ContextBudgetManager()
    messages = [
        {"role": "system", "content": "You are MiniCode assistant with safety rules."},
        {"role": "user", "content": "Fix the bug in auth.py"},
        {"role": "assistant", "content": "[Stable task state]\nTask: Fix bug\nStatus: IN_PROGRESS"},
    ]
    items = mgr.classify_all(messages)

    assert items[0].zone == ContextZone.CRITICAL
    assert items[0].protected is True
    assert "system_prompt" in items[0].reasons

    assert items[1].zone == ContextZone.ACTIVE_TASK
    assert items[1].protected is True

    assert items[2].zone == ContextZone.CRITICAL
    assert items[2].protected is True
    assert "stable_task_state" in items[2].reasons


def test_context_zone_classification_verification_latest_vs_earlier():
    mgr = ContextBudgetManager()
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Run tests"},
        # Earlier verification failure
        {
            "role": "tool_result",
            "toolName": "pytest",
            "content": "=== test session starts ===\nFAILED tests/test_login.py::test_auth - AssertionError\n1 failed in 0.5s",
        },
        {"role": "assistant", "content": "I fixed the issue. Re-running tests."},
        # Latest verification pass
        {
            "role": "tool_result",
            "toolName": "pytest",
            "content": "=== test session starts ===\n2 passed in 0.2s",
        },
    ]
    items = mgr.classify_all(messages)

    # Item 2 is earlier verification
    assert items[2].zone == ContextZone.VERIFICATION
    assert items[2].protected is False
    assert "earlier_verification" in items[2].reasons

    # Item 4 is latest verification
    assert items[4].zone == ContextZone.VERIFICATION
    assert items[4].protected is True
    assert "latest_verification" in items[4].reasons


def test_context_zone_classification_error_evidence():
    mgr = ContextBudgetManager()
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Do work"},
        {
            "role": "tool_result",
            "toolName": "run_command",
            "content": "Traceback (most recent call last):\n  File 'app.py', line 10\nZeroDivisionError: division by zero",
            "isError": True,
        },
    ]
    items = mgr.classify_all(messages)
    assert items[2].zone == ContextZone.ERROR_EVIDENCE
    assert items[2].protected is True


def test_context_zone_classification_memory_and_ephemeral():
    mgr = ContextBudgetManager()
    messages = [
        {"role": "system", "content": "System"},
        {
            "role": "user",
            "content": "## Relevant Coding Experience\n[Verified Experience]\nFixed authentication token expiry.",
        },
        {"role": "assistant_progress", "content": "Analyzing repository files..."},
    ]
    items = mgr.classify_all(messages)

    assert items[1].zone == ContextZone.MEMORY
    assert "verified_experience" in items[1].reasons
    assert items[1].importance_score >= 0.85  # Boosted memory

    assert items[2].zone == ContextZone.EPHEMERAL
    assert items[2].protected is False


def test_deterministic_importance_scoring():
    cfg = ContextBudgetConfig()
    mgr = ContextBudgetManager(config=cfg)

    item1 = ContextItem(
        item_id="i1",
        message_index=0,
        zone=ContextZone.CRITICAL,
        role="system",
        source="system",
        estimated_tokens=50,
        protected=True,
    )
    score1 = mgr.score_importance(item1, total_messages=10)
    assert score1 >= 0.90

    item2 = ContextItem(
        item_id="i2",
        message_index=2,
        zone=ContextZone.CONVERSATION,
        role="user",
        source="user",
        estimated_tokens=50,
    )
    score2 = mgr.score_importance(item2, total_messages=5)
    # base 0.40 + recency (2/4 * 0.15 = 0.075) = 0.475
    assert 0.45 <= score2 <= 0.50

    # With staleness (age = 20 - 1 - 1 = 18 > 6)
    item_stale = ContextItem(
        item_id="i3",
        message_index=1,
        zone=ContextZone.CONVERSATION,
        role="user",
        source="user",
        estimated_tokens=50,
    )
    score_stale = mgr.score_importance(item_stale, total_messages=20)
    assert score_stale < score2


def test_plan_under_budget_keeps_all():
    mgr = ContextBudgetManager()
    messages = [
        {"role": "system", "content": "Short system prompt"},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am doing well, ready to help."},
    ]
    plan = mgr.plan(messages, available_budget=10000)

    assert plan.keep_count == 3
    assert plan.compress_count == 0
    assert plan.offload_count == 0
    assert plan.evict_count == 0
    for d in plan.decisions:
        assert d.action == ContextAction.KEEP


def test_plan_over_budget_evicts_ephemeral_and_offloads_tool():
    mgr = ContextBudgetManager(
        config=ContextBudgetConfig(offload_threshold_tokens=100)
    )
    # Huge tool result: ~1000 tokens
    large_text = "line of log data\n" * 500
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "assistant_progress", "content": "Searching codebase..."},
        {
            "role": "tool_result",
            "toolName": "read_file",
            "content": large_text,
        },
        {"role": "user", "content": "Show me the logs"},
    ]
    # Small budget forcing action
    plan = mgr.plan(messages, available_budget=200)

    actions = [d.action for d in plan.decisions]
    # Index 0: system -> KEEP
    assert actions[0] == ContextAction.KEEP
    # Index 1: ephemeral -> EVICT
    assert actions[1] == ContextAction.EVICT
    # Index 2: large tool result -> OFFLOAD
    assert actions[2] == ContextAction.OFFLOAD
    # Index 3: active user task -> KEEP (protected)
    assert actions[3] == ContextAction.KEEP


def test_apply_preserves_tool_result_pairs_on_eviction():
    mgr = ContextBudgetManager()
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "assistant_progress", "content": "Progress"},
        {"role": "tool_result", "toolName": "grep", "content": "unimportant output"},
    ]
    # Force eviction on items 1 and 2
    from minicode.context_budget import ContextBudgetPlan, ContextDecision
    plan = ContextBudgetPlan(
        budget_tokens=50,
        reserved_output_tokens=10,
        available_input_tokens=40,
        tokens_before=100,
        tokens_after_estimate=10,
        decisions=[
            ContextDecision("i0", 0, ContextAction.KEEP, 1.0, 10, 10, "keep"),
            ContextDecision("i1", 1, ContextAction.EVICT, 0.1, 20, 0, "evict"),
            ContextDecision("i2", 2, ContextAction.EVICT, 0.2, 50, 0, "evict"),
        ],
    )
    modified, metrics = mgr.apply(messages, plan)

    # assistant_progress can be evicted completely
    # tool_result must retain placeholder so tool pair invariant isn't violated
    assert len(modified) == 2
    assert modified[0]["role"] == "system"
    assert modified[1]["role"] == "tool_result"
    assert modified[1]["content"] == "[Output cleared]"
    assert metrics.evicted_tokens > 0


def test_dynamic_model_window_plan():
    mgr = ContextBudgetManager()
    messages = [{"role": "system", "content": "hello"}]

    # Model with 200k window
    plan_claude = mgr.plan(messages, model="claude-sonnet-4-20250514")
    assert plan_claude.budget_tokens > 150_000

    # Model with 128k window
    plan_gpt = mgr.plan(messages, model="gpt-4o")
    assert plan_gpt.budget_tokens < plan_claude.budget_tokens
    assert plan_gpt.budget_tokens > 100_000


def test_verification_pass_then_fail_keeps_fail():
    mgr = ContextBudgetManager()
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Run tests"},
        # Earlier pass
        {
            "role": "tool_result",
            "toolName": "pytest",
            "content": "=== test session starts ===\n3 passed in 0.1s",
        },
        {"role": "assistant", "content": "Now breaking change"},
        # Final fail
        {
            "role": "tool_result",
            "toolName": "pytest",
            "content": "=== test session starts ===\nFAILED tests/test_core.py::test_fail\n1 failed in 0.1s",
        },
    ]
    items = mgr.classify_all(messages)
    # Earlier pass is not protected
    assert items[2].zone == ContextZone.VERIFICATION
    assert items[2].protected is False
    # Final fail is protected
    assert items[4].zone == ContextZone.VERIFICATION
    assert items[4].protected is True


def test_active_file_boost():
    mgr = ContextBudgetManager()
    item = ContextItem(
        item_id="i1",
        message_index=3,
        zone=ContextZone.TOOL_EVIDENCE,
        role="tool_result",
        source="minicode/agent_loop.py",
        estimated_tokens=100,
    )
    score_normal = mgr.score_importance(item, active_files=set(), total_messages=10)
    score_boosted = mgr.score_importance(item, active_files={"minicode/agent_loop.py"}, total_messages=10)
    assert score_boosted > score_normal
    assert "active_file_boost:minicode/agent_loop.py" in item.reasons
