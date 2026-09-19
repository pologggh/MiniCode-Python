from __future__ import annotations

import pytest

from minicode.skill_router import (
    CandidateSkill,
    SkillRouter,
    get_skill_router,
    normalize_query,
)
from minicode.skills import SkillSummary


@pytest.fixture
def sample_skills() -> list[SkillSummary]:
    return [
        SkillSummary(
            name="fastapi-debugging",
            description="Debug FastAPI request and response failures, pydantic validation errors",
            path="/skills/fastapi-debugging/SKILL.md",
            source="project",
            category="backend",
            tags=["python", "fastapi", "api", "debugging"],
            priority=10,
        ),
        SkillSummary(
            name="pytest-debugging",
            description="Diagnose and fix failing pytest test suites",
            path="/skills/pytest-debugging/SKILL.md",
            source="project",
            category="testing",
            tags=["python", "pytest", "tests", "tdd"],
            priority=8,
        ),
        SkillSummary(
            name="react-ui",
            description="Frontend React component library and state management",
            path="/skills/react-ui/SKILL.md",
            source="user",
            category="frontend",
            tags=["react", "javascript", "frontend", "ui"],
            priority=5,
        ),
        SkillSummary(
            name="docker-deploy",
            description="Docker container build, compose files, and deployment automation",
            path="/skills/docker-deploy/SKILL.md",
            source="user",
            category="devops",
            tags=["docker", "containers", "deployment"],
            priority=5,
        ),
        SkillSummary(
            name="sql-migration",
            description="SQL schema migration scripts and database index tuning",
            path="/skills/sql-migration/SKILL.md",
            source="project",
            category="database",
            tags=["sql", "database", "postgres"],
            priority=5,
        ),
    ]


def test_normalize_query_chinese_english_mixed():
    query = "请帮我修复 FastAPI 接口报错，以及运行 pytest 测试"
    lowered, tokens = normalize_query(query)
    assert "fastapi" in tokens
    assert "pytest" in tokens
    assert "修" in tokens or "报" in tokens or "测" in tokens
    assert "请" not in tokens  # stop word


def test_normalize_query_preserves_code_tokens():
    query = "Configuring pydantic models and docker container with git"
    _, tokens = normalize_query(query)
    assert "pydantic" in tokens
    assert "docker" in tokens
    assert "git" in tokens


def test_normalize_query_hyphen_underscore():
    query = "run test_driven_development with fastapi-debugging"
    _, tokens = normalize_query(query)
    assert "fastapi-debugging" in tokens
    assert "fastapi" in tokens
    assert "debugging" in tokens
    assert "test_driven_development" in tokens
    assert "test" in tokens
    assert "development" in tokens


def test_router_exact_name_match_ranks_highest(sample_skills):
    router = SkillRouter(default_top_k=3)
    ranked = router.rank("fastapi-debugging", sample_skills)
    assert len(ranked) == len(sample_skills)
    assert ranked[0].skill_name == "fastapi-debugging"
    assert ranked[0].score >= 15.0
    assert "exact_name" in ranked[0].matched_fields


def test_router_tag_match(sample_skills):
    router = SkillRouter(default_top_k=3)
    ranked = router.rank("Need help with api tests and postgres", sample_skills)
    # fastapi has tag 'api', sql-migration has tag 'postgres'
    skill_names = [c.skill_name for c in ranked if c.score > 0]
    assert "fastapi-debugging" in skill_names
    assert "sql-migration" in skill_names


def test_router_description_term_overlap(sample_skills):
    router = SkillRouter(default_top_k=3)
    # "validation errors" is in fastapi-debugging description
    ranked = router.rank("Validation errors during payload decoding", sample_skills)
    assert ranked[0].skill_name == "fastapi-debugging"
    assert "description" in ranked[0].matched_fields


def test_router_irrelevant_skills_ranked_lower(sample_skills):
    router = SkillRouter(default_top_k=5)
    query = "FastAPI POST endpoint returns 422 and pytest fails"
    ranked = router.rank(query, sample_skills)

    top_two = [c.skill_name for c in ranked[:2]]
    assert "fastapi-debugging" in top_two
    assert "pytest-debugging" in top_two

    # Irrelevant skills should have 0 score or significantly lower
    for c in ranked[2:]:
        assert c.score < ranked[1].score


def test_router_deterministic_ordering_with_tie_break():
    router = SkillRouter()
    # Skills with same score and different priority/name
    skills = [
        SkillSummary(name="b-skill", description="desc", path="/b", source="u", priority=5),
        SkillSummary(name="a-skill", description="desc", path="/a", source="u", priority=5),
        SkillSummary(name="c-skill", description="desc", path="/c", source="u", priority=10),
    ]
    ranked = router.rank("unrelated query with no keyword match", skills)
    # Since scores are 0, tie break: priority desc -> c-skill first, then a-skill before b-skill
    assert ranked[0].skill_name == "c-skill"  # priority 10
    assert ranked[1].skill_name == "a-skill"  # priority 5, alphabetical 'a'
    assert ranked[2].skill_name == "b-skill"  # priority 5, alphabetical 'b'


def test_router_top_k_selection(sample_skills):
    router = SkillRouter(default_top_k=2)
    query = "FastAPI POST endpoint returns 422 and pytest fails"
    selected, metrics = router.select(query, sample_skills, top_k=2)
    assert len(selected) == 2
    assert selected[0].skill_name in ("fastapi-debugging", "pytest-debugging")
    assert selected[1].skill_name in ("fastapi-debugging", "pytest-debugging")
    assert metrics.selected_count == 2
    assert metrics.candidate_count == len(sample_skills)
    assert metrics.catalog_token_estimate > 0
    assert metrics.selected_catalog_token_estimate > 0


def test_router_empty_query_returns_no_candidates(sample_skills):
    router = SkillRouter(default_top_k=3)
    selected, metrics = router.select("", sample_skills)
    assert len(selected) == 0
    assert metrics.selected_count == 0


def test_router_no_candidate_reaches_threshold(sample_skills):
    router = SkillRouter(default_top_k=3)
    # Query with no overlap to any skill
    selected, metrics = router.select("quantum mechanics and string theory", sample_skills)
    assert len(selected) == 0
    assert metrics.selected_count == 0


def test_router_category_filter(sample_skills):
    router = SkillRouter(default_top_k=5)
    query = "debugging things"
    selected, _ = router.select(query, sample_skills, category_filter="testing")
    for cand in selected:
        assert cand.category == "testing"


def test_router_token_budget(sample_skills):
    router = SkillRouter(default_top_k=5)
    query = "python api testing database"
    # Constrain to very tight budget
    selected, metrics = router.select(query, sample_skills, token_budget=10)
    assert len(selected) <= 2


def test_router_format_catalog():
    router = SkillRouter()
    cands = [
        CandidateSkill(
            skill_name="fastapi-dbg",
            score=12.5,
            matched_fields=["exact_name"],
            category="backend",
            tags=["fastapi"],
            source="project",
            selection_reason="matched exact name",
            description="FastAPI debug helper",
        )
    ]
    catalog = router.format_catalog(cands)
    assert "Available skills (task-relevant):" in catalog
    assert "fastapi-dbg" in catalog
    assert "category: backend" in catalog
    assert "reason: matched exact name" in catalog
    assert "FastAPI debug helper" in catalog


def test_router_format_fallback_catalog():
    router = SkillRouter()
    fallback_with_skills = router.format_fallback_catalog(has_skills=True)
    assert "No specific skills matched this task query" in fallback_with_skills
    assert "load_skill(name)" in fallback_with_skills

    fallback_no_skills = router.format_fallback_catalog(has_skills=False)
    assert "none discovered" in fallback_no_skills
