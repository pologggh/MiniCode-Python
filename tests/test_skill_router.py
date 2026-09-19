from __future__ import annotations

import pytest

from minicode.skill_router import (
    CandidateSkill,
    SkillRouter,
    get_skill_router,
    normalize_query,
    normalize_top_k,
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


def test_priority_cannot_independently_create_relevance():
    router = SkillRouter(default_top_k=5)
    skills = [
        SkillSummary(
            name="completely-irrelevant-high-priority",
            description="Quantum physics and string cosmology calculations",
            path="/skills/cosmo/SKILL.md",
            source="project",
            category="physics",
            tags=["quantum", "physics"],
            priority=100,  # Huge priority
        ),
        SkillSummary(
            name="relevant-low-priority",
            description="Fixing Python bugs and unit tests",
            path="/skills/py/SKILL.md",
            source="project",
            category="python",
            tags=["python"],
            priority=1,
        ),
    ]
    query = "Help me debug a python function"
    selected, metrics = router.select(query, skills)

    # Irrelevant skill must NEVER be selected even with priority=100
    selected_names = [c.skill_name for c in selected]
    assert "relevant-low-priority" in selected_names
    assert "completely-irrelevant-high-priority" not in selected_names

    # Check ranked score directly: zero lexical match produces exactly 0.0 score
    ranked = router.rank(query, skills)
    irrelevant_cand = next(c for c in ranked if c.skill_name == "completely-irrelevant-high-priority")
    assert irrelevant_cand.score == 0.0
    assert irrelevant_cand.matched_fields == []


def test_word_boundary_matching_prevents_false_positives():
    router = SkillRouter(default_top_k=5)
    skills = [
        SkillSummary(
            name="go-toolchain",
            description="Go compiler and module management",
            path="/skills/go/SKILL.md",
            source="project",
            category="backend",
            tags=["go", "golang"],
        ),
        SkillSummary(
            name="c-toolchain",
            description="C and C++ build systems with CMake",
            path="/skills/c/SKILL.md",
            source="project",
            category="systems",
            tags=["c", "cpp"],
        ),
        SkillSummary(
            name="contest-helper",
            description="Competitive programming tools",
            path="/skills/contest/SKILL.md",
            source="project",
            category="algo",
            tags=["contest", "algorithms"],
        ),
    ]

    # Query containing "django" should NOT trigger "go"
    ranked_django = router.rank("Build a web application with Django and PostgreSQL", skills)
    django_names = [c.skill_name for c in ranked_django if c.score > 0]
    assert "go-toolchain" not in django_names

    # Query containing "docker" should NOT trigger "c"
    ranked_docker = router.rank("Setup docker container and docker-compose", skills)
    docker_names = [c.skill_name for c in ranked_docker if c.score > 0]
    assert "c-toolchain" not in docker_names

    # Query containing "test" should NOT trigger "contest-helper"
    ranked_test = router.rank("Run unit tests and assert results", skills)
    test_names = [c.skill_name for c in ranked_test if c.score > 0]
    assert "contest-helper" not in test_names

    # Valid exact boundary matches should work
    ranked_go = router.rank("Write a microservice in go", skills)
    assert ranked_go[0].skill_name == "go-toolchain"
    assert ranked_go[0].score > 0

    ranked_c = router.rank("Compile this c program", skills)
    assert ranked_c[0].skill_name == "c-toolchain"
    assert ranked_c[0].score > 0


def test_strict_token_budget_hard_truncation():
    router = SkillRouter(default_top_k=5)
    skills = [
        SkillSummary(
            name="big-skill-one",
            description="This is a fairly long description intended to take a non-trivial amount of tokens.",
            path="/1",
            source="p",
            tags=["query"],
        ),
        SkillSummary(
            name="big-skill-two",
            description="Another long description also consuming tokens for testing budget limits.",
            path="/2",
            source="p",
            tags=["query"],
        ),
    ]
    query = "query"

    # 1. Very small budget (e.g. 2 tokens) cannot even fit the first candidate -> must return empty list []
    selected, _ = router.select(query, skills, token_budget=2)
    assert selected == []

    # 2. Budget <= 0 -> must return empty list []
    selected_zero, _ = router.select(query, skills, token_budget=0)
    assert selected_zero == []

    # 3. Budget that fits exactly 1 candidate
    from minicode.context_manager import estimate_tokens
    first_cand_tokens = estimate_tokens("big-skill-one: This is a fairly long description intended to take a non-trivial amount of tokens.")
    selected_one, _ = router.select(query, skills, token_budget=first_cand_tokens + 2)
    assert len(selected_one) == 1
    assert selected_one[0].skill_name == "big-skill-one"


def test_metrics_candidate_count_reflects_recalled_pool():
    router = SkillRouter(default_top_k=5)
    skills = [
        SkillSummary(name="test-1", description="desc 1", path="/1", source="p", category="testing", tags=["python"]),
        SkillSummary(name="test-2", description="desc 2", path="/2", source="p", category="testing", tags=["python"]),
        SkillSummary(name="devops-1", description="desc 3", path="/3", source="p", category="devops", tags=["docker"]),
        SkillSummary(name="devops-2", description="desc 4", path="/4", source="p", category="devops", tags=["docker"]),
    ]
    # Filter by category="testing" -> recalled pool is 2, total skills is 4
    _, metrics = router.select("python", skills, category_filter="testing")
    assert metrics.candidate_count == 2
    assert metrics.candidate_count != len(skills)


def test_normalize_top_k_defensive_handling():
    assert normalize_top_k(None) == 5
    assert normalize_top_k(None, default=3) == 3
    assert normalize_top_k(0) == 5
    assert normalize_top_k(-10) == 5
    assert normalize_top_k("invalid") == 5
    assert normalize_top_k("7") == 7
    assert normalize_top_k(100, max_k=20) == 20
    assert normalize_top_k(3.7) == 3
    assert normalize_top_k(True) == 5  # bool guarded against int subclass
    assert normalize_top_k(False) == 5
