from __future__ import annotations

from benchmarks.skill_routing_eval import (
    create_base_skills,
    run_catalog_scale_eval,
    run_retrieval_eval,
)
from minicode.skill_router import SkillRouter


def test_skill_routing_benchmark_accuracy():
    router = SkillRouter(default_top_k=5)
    skills = create_base_skills()
    metrics = run_retrieval_eval(router, skills)

    assert metrics["Recall@1"] >= 0.8
    assert metrics["Recall@3"] == 1.0
    assert metrics["Recall@5"] == 1.0
    assert metrics["MRR"] >= 0.9


def test_skill_routing_benchmark_token_savings():
    router = SkillRouter(default_top_k=5)
    scale_metrics = run_catalog_scale_eval(router)

    assert len(scale_metrics) == 3
    # Check 10, 100, 500 skills
    for row in scale_metrics:
        assert row["tokens_saved"] > 0
        assert row["full_catalog_tokens"] > row["top_k_catalog_tokens"]
