"""Tests for LayeredContext and dynamic budget alignment."""
from __future__ import annotations

import pytest

from minicode.layered_context import (
    ContextBudget,
    ContextLayer,
    LayerContent,
    LayeredContext,
)


def test_context_budget_from_effective_input():
    # 200k effective input with 5% safety margin -> 190,000
    budget = ContextBudget.from_effective_input(effective_input=200_000, safety_margin=0.05)
    assert budget.total_limit == 190_000
    assert budget.system_limit == int(190_000 * 0.15)
    assert budget.project_limit == int(190_000 * 0.25)
    assert budget.session_limit == int(190_000 * 0.45)
    assert budget.scratchpad_limit == int(190_000 * 0.15)

    # 128k effective input with 10% safety margin
    budget_128k = ContextBudget.from_effective_input(effective_input=128_000, safety_margin=0.10)
    assert budget_128k.total_limit == int(128_000 * 0.90)


def test_layered_context_legacy_default_budget():
    ctx = LayeredContext()
    assert ctx.budget.total_limit == 8000
    assert ctx.budget.system_limit == 1200
    assert ctx.budget.session_limit == 3600


def test_giant_item_token_accounting_no_fake_truncation():
    """Verify giant item does NOT fake tokens=limit while leaving text giant."""
    budget = ContextBudget(total_limit=1000)  # session limit = 450 tokens
    ctx = LayeredContext(budget=budget)

    # Add a giant text of ~2000 tokens
    giant_text = "word " * 2000
    ctx.add(ContextLayer.SESSION, giant_text)

    items = ctx.get(ContextLayer.SESSION)
    assert len(items) == 1
    giant_item = items[0]

    # In the bugged version, giant_item.tokens would be set to 450,
    # falsely claiming to be 450 tokens when giant_text was ~2000 tokens.
    assert giant_item.tokens > 1000  # True token count preserved!
    assert giant_item.overflow is True
    assert ctx.get_layer_tokens(ContextLayer.SESSION) == giant_item.tokens

    overflow_items = ctx.get_overflow_items()
    assert len(overflow_items) == 1
    assert overflow_items[0] is giant_item


def test_layer_priority_trimming_and_history():
    budget = ContextBudget(total_limit=1000)  # session limit = 450 tokens
    ctx = LayeredContext(budget=budget)

    # Add high priority item (150 tokens)
    ctx.add(ContextLayer.SESSION, "important info " * 30, tokens=150, priority=10)
    # Add low priority item (400 tokens) -> exceeds 450 total
    ctx.add(ContextLayer.SESSION, "low priority info " * 80, tokens=400, priority=1)

    items = ctx.get(ContextLayer.SESSION)
    # High priority item must be kept
    assert len(items) == 1
    assert items[0].priority == 10
    assert items[0].tokens == 150

    # Verify trim history was recorded
    history = ctx.get_trim_history()
    assert len(history) >= 1
    assert history[0]["layer"] == "session"
    assert history[0]["removed_count"] >= 1
