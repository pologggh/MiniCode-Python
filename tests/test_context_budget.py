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


def test_natural_language_constraints():
    mgr = ContextBudgetManager()
    constraints = [
        "Do not modify database migrations.",
        "Never edit files under auth/.",
        "Use only Python standard library.",
        "You must not run destructive git commands.",
        "Keep the public API backward compatible.",
        "CRITICAL CONSTRAINT: No external requests.",
    ]
    for c in constraints:
        messages = [
            {"role": "system", "content": "You are assistant."},
            {"role": "user", "content": c},
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "user", "content": "Next step."},
        ]
        items = mgr.classify_all(messages)
        # Message index 1 should be recognized as constraint
        assert items[1].zone == ContextZone.CRITICAL, f"Failed on: {c}"
        assert items[1].protected is True, f"Failed on: {c}"
        assert items[1].source == "constraint", f"Failed on: {c}"


def test_rich_compact_preview_preserves_salient_lines():
    from minicode.context_budget import build_compact_preview

    # Test error extraction
    pytest_err = (
        "=================== test session starts ===================\n"
        "tests/test_a.py .\n"
        "FAILED tests/test_b.py::test_fail - AssertionError: Status code 500\n"
        "=================== 1 failed in 0.3s ==================="
    )
    preview_err = build_compact_preview(pytest_err, max_chars=120)
    assert "AssertionError" in preview_err or "FAILED" in preview_err
    assert len(preview_err) <= 135

    # Test constraint line preservation
    conv_with_constraint = (
        "We are analyzing the project structure.\n"
        "Never edit files under auth/.\n"
        "Proceeding with the remaining tests."
    )
    preview_c = build_compact_preview(conv_with_constraint, max_chars=140)
    assert "Never edit files under auth/." in preview_c


def test_tool_pair_integrity_validation():
    from minicode.context_budget import validate_tool_pair_integrity

    # Valid pair
    valid_msgs = [
        {"role": "user", "content": "read file"},
        {"role": "assistant_tool_call", "toolName": "read_file", "toolUseId": "call_1", "input": {}},
        {"role": "tool_result", "toolName": "read_file", "toolUseId": "call_1", "content": "file contents"},
    ]
    assert validate_tool_pair_integrity(valid_msgs) is True

    # Offloaded pair (content replaced with artifact reference)
    offloaded_msgs = [
        {"role": "assistant_tool_call", "toolName": "read_file", "toolUseId": "call_1", "input": {}},
        {"role": "tool_result", "toolName": "read_file", "toolUseId": "call_1", "content": "[Context Artifact]\nid: ctx_123"},
    ]
    assert validate_tool_pair_integrity(offloaded_msgs) is True

    # Evicted tombstone pair
    evicted_msgs = [
        {"role": "assistant_tool_call", "toolName": "read_file", "toolUseId": "call_1", "input": {}},
        {"role": "tool_result", "toolName": "read_file", "toolUseId": "call_1", "content": "[Output cleared]"},
    ]
    assert validate_tool_pair_integrity(evicted_msgs) is True

    # Broken: orphan result with no call
    broken_orphan = [
        {"role": "user", "content": "hello"},
        {"role": "tool_result", "toolName": "read_file", "toolUseId": "call_missing", "content": "data"},
    ]
    assert validate_tool_pair_integrity(broken_orphan) is False

    # Broken: call with no result
    broken_unanswered = [
        {"role": "assistant_tool_call", "toolName": "read_file", "toolUseId": "call_unanswered", "input": {}},
    ]
    assert validate_tool_pair_integrity(broken_unanswered) is False

    # Broken: mismatched call ID
    broken_mismatch = [
        {"role": "assistant_tool_call", "toolName": "read_file", "toolUseId": "call_1", "input": {}},
        {"role": "tool_result", "toolName": "read_file", "toolUseId": "call_2", "content": "data"},
    ]
    assert validate_tool_pair_integrity(broken_mismatch) is False


def test_real_compactor_fallback_and_compliance(tmp_path):
    from minicode.context_compactor import ContextCompactor

    compactor = ContextCompactor(workspace=tmp_path)
    mgr = ContextBudgetManager()

    messages = [
        {"role": "system", "content": "You are assistant."},
        {"role": "user", "content": "Do a refactoring task."},
    ]
    for i in range(10):
        messages.append({
            "role": "assistant_tool_call",
            "toolName": "grep",
            "toolUseId": f"call_{i}",
            "input": {"query": f"match_{i}"},
        })
        messages.append({
            "role": "tool_result",
            "toolName": "grep",
            "toolUseId": f"call_{i}",
            "content": f"match_{i}_result_data\n" * 80,
        })

    # Available budget = 100
    target_budget = 100
    phase3_msgs, plan = mgr.plan_and_apply(
        messages=messages,
        available_budget=target_budget,
        compactor=compactor,
    )

    # Verify fallback tracking fields
    assert plan.fallback_attempted is True
    assert isinstance(plan.fallback_effective, bool)
    # Compliance must strictly be evaluated against budget
    tokens_final = sum(len(str(m)) // 4 for m in phase3_msgs)
    if plan.tokens_after_estimate <= target_budget:
        assert plan.budget_compliant is True
    else:
        assert plan.budget_compliant is False


def test_protected_context_over_budget_violation():
    mgr = ContextBudgetManager()
    # Huge task constraint that exceeds available budget of 50 tokens
    huge_constraint = "CRITICAL CONSTRAINT: " + "Must strictly preserve legacy endpoints without any alterations. " * 10
    messages = [
        {"role": "system", "content": "System prompt with base rules."},
        {"role": "user", "content": huge_constraint},
        {"role": "user", "content": "Now run next step."},
    ]

    phase3_msgs, plan = mgr.plan_and_apply(messages=messages, available_budget=50)

    # Protected context cannot fit into 50 tokens -> must report violation, NOT fake compliant
    assert plan.budget_compliant is False
    assert plan.budget_violation_reason == "protected_context_exceeds_budget"
    # Crucially, protected constraint must NOT be deleted
    assert any("Must strictly preserve legacy endpoints" in str(m.get("content", "")) for m in phase3_msgs)


def test_plan_available_budget_none_runtime_pressure():
    """Verify plan() with available_budget=None computes model budget and does not throw TypeError on eviction path."""
    mgr = ContextBudgetManager()

    # Construct fixture exceeding GPT-4o input budget (~123k tokens -> ~500k chars)
    # Using 150 conversation messages of 3,500 chars each (~130,000 tokens)
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Solve large migration task."},
    ]
    for i in range(150):
        messages.append({
            "role": "assistant",
            "content": f"Detailed architectural migration step {i} notes and analysis: " + ("x" * 3500),
        })
        messages.append({
            "role": "user",
            "content": f"Feedback on step {i}: continue with the next module.",
        })

    # Call plan with available_budget=None
    plan = mgr.plan(messages=messages, model="gpt-4o", available_budget=None)

    assert plan.budget_tokens > 100_000
    assert plan.tokens_before > plan.budget_tokens
    # Must enter evict_over_budget path without TypeError
    assert plan.evict_count > 0
    assert any("evict_over_budget" in d.reason for d in plan.decisions)


def test_error_resolution_semantics_case_a():
    """CASE A: pytest fail -> edit -> pytest pass.
    Pass must be protected VERIFICATION, earlier fail must have protected=False.
    """
    mgr = ContextBudgetManager()
    messages = [
        {"role": "system", "content": "You are assistant."},
        {"role": "user", "content": "Run tests and fix failures."},
        # Index 2: pytest fail
        {
            "role": "tool_result",
            "toolName": "pytest",
            "content": "=== test session starts ===\nFAILED tests/test_core.py::test_foo - AssertionError: expected 1 got 0\n1 failed in 0.12s",
        },
        {"role": "assistant", "content": "Fixing the assertion."},
        # Index 4: edit success
        {
            "role": "tool_result",
            "toolName": "edit",
            "content": "File core.py modified successfully.",
        },
        {"role": "assistant", "content": "Re-running tests."},
        # Index 6: pytest pass
        {
            "role": "tool_result",
            "toolName": "pytest",
            "content": "=== test session starts ===\n1 passed in 0.05s",
        },
    ]

    items = mgr.classify_all(messages)

    fail_item = items[2]
    pass_item = items[6]

    # Earlier fail must NOT be protected
    assert fail_item.protected is False
    assert fail_item.zone == ContextZone.VERIFICATION

    # Latest pass MUST be protected
    assert pass_item.protected is True
    assert pass_item.zone == ContextZone.VERIFICATION
    assert "latest_verification" in pass_item.reasons


def test_error_resolution_semantics_case_b():
    """CASE B: pytest pass -> edit -> pytest fail.
    Latest fail must be protected VERIFICATION evidence, earlier pass must have protected=False.
    """
    mgr = ContextBudgetManager()
    messages = [
        {"role": "system", "content": "You are assistant."},
        {"role": "user", "content": "Refactor codebase."},
        # Index 2: pytest pass
        {
            "role": "tool_result",
            "toolName": "pytest",
            "content": "=== test session starts ===\n5 passed in 0.20s",
        },
        {"role": "assistant", "content": "Introducing breaking changes."},
        # Index 4: edit success
        {
            "role": "tool_result",
            "toolName": "edit",
            "content": "File engine.py updated.",
        },
        {"role": "assistant", "content": "Checking regression tests."},
        # Index 6: pytest fail
        {
            "role": "tool_result",
            "toolName": "pytest",
            "content": "=== test session starts ===\nFAILED tests/test_engine.py::test_pipe - AssertionError\n1 failed in 0.15s",
        },
    ]

    items = mgr.classify_all(messages)

    pass_item = items[2]
    fail_item = items[6]

    # Earlier pass is not latest verification
    assert pass_item.protected is False

    # Latest fail is protected
    assert fail_item.protected is True
    assert fail_item.zone == ContextZone.VERIFICATION
    assert "latest_verification" in fail_item.reasons
    assert "unresolved_verification_failure" in fail_item.reasons


def test_error_resolution_semantics_case_c():
    """CASE C: generic tool error and no subsequent successful recovery evidence.
    Must remain protected ERROR_EVIDENCE.
    """
    mgr = ContextBudgetManager()
    messages = [
        {"role": "system", "content": "You are assistant."},
        {"role": "user", "content": "Build the frontend assets."},
        # Index 2: tool call
        {
            "role": "assistant_tool_call",
            "toolName": "run_command",
            "toolUseId": "call_cmd_1",
            "input": {"command": "npm run build"},
        },
        # Index 3: generic tool error
        {
            "role": "tool_result",
            "toolName": "run_command",
            "toolUseId": "call_cmd_1",
            "content": "Error: command failed with exit code 1. Missing dependency 'webpack'",
            "isError": True,
        },
    ]

    items = mgr.classify_all(messages)

    err_item = items[3]
    assert err_item.zone == ContextZone.ERROR_EVIDENCE
    assert err_item.protected is True
    assert "latest_error_evidence" in err_item.reasons

