"""Skill Routing Evaluation and Benchmark.

Evaluates SkillRouter on accuracy (Recall@1, Recall@3, Recall@5, Precision@3, MRR)
and measures prompt token consumption across catalog scales (10, 100, 500 skills).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from minicode.context_manager import estimate_tokens
from minicode.skill_router import SkillRouter
from minicode.skills import SkillSummary


@dataclass(slots=True)
class EvalTestCase:
    id: str
    domain: str
    query: str
    expected_relevant: list[str]
    irrelevant: list[str]


# ---------------------------------------------------------------------------
# Evaluation Fixtures
# ---------------------------------------------------------------------------

FIXED_TEST_CASES: list[EvalTestCase] = [
    EvalTestCase(
        id="case-1-python-debugging",
        domain="Python debugging",
        query="Diagnose unhandled TypeError and trace stacktrace in python worker",
        expected_relevant=["python-debugging"],
        irrelevant=["react-frontend", "docker-deploy", "sql-migration"],
    ),
    EvalTestCase(
        id="case-2-fastapi",
        domain="FastAPI",
        query="FastAPI POST endpoint returns 422 unprocessable entity on pydantic payload",
        expected_relevant=["fastapi-debugging"],
        irrelevant=["git-workflow", "react-frontend", "docker-deploy"],
    ),
    EvalTestCase(
        id="case-3-pytest",
        domain="pytest",
        query="Run pytest test suite, fix failing assertions and mock fixture teardown",
        expected_relevant=["pytest-debugging"],
        irrelevant=["docker-deploy", "sql-migration", "react-frontend"],
    ),
    EvalTestCase(
        id="case-4-git",
        domain="Git",
        query="Resolve git merge conflict and interactive rebase on feature branch",
        expected_relevant=["git-workflow"],
        irrelevant=["fastapi-debugging", "sql-migration", "pytest-debugging"],
    ),
    EvalTestCase(
        id="case-5-docker",
        domain="Docker",
        query="Build multi-stage Dockerfile and debug docker compose port binding error",
        expected_relevant=["docker-deploy"],
        irrelevant=["react-frontend", "python-debugging", "sql-migration"],
    ),
    EvalTestCase(
        id="case-6-sql",
        domain="SQL",
        query="Optimize slow postgres SQL query and create database migration index",
        expected_relevant=["sql-migration"],
        irrelevant=["git-workflow", "react-frontend", "fastapi-debugging"],
    ),
    EvalTestCase(
        id="case-7-frontend",
        domain="Frontend",
        query="Fix React component state hook infinite re-render loop in frontend UI",
        expected_relevant=["react-frontend"],
        irrelevant=["sql-migration", "docker-deploy", "python-debugging"],
    ),
    EvalTestCase(
        id="case-8-compound-fastapi-pytest",
        domain="FastAPI + pytest",
        query="FastAPI POST endpoint returns 422 and pytest test suite fails",
        expected_relevant=["fastapi-debugging", "pytest-debugging"],
        irrelevant=["react-frontend", "docker-deploy", "sql-migration"],
    ),
]


def create_base_skills() -> list[SkillSummary]:
    """Base curated skills covering the test domains."""
    return [
        SkillSummary(
            name="python-debugging",
            description="Systematic Python debugging, exception diagnosis, and trace inspection",
            path="/skills/python-debugging/SKILL.md",
            source="project",
            category="python",
            tags=["python", "debug", "debugging", "trace", "exception"],
            priority=10,
        ),
        SkillSummary(
            name="fastapi-debugging",
            description="Debug FastAPI endpoints, request body validation, and pydantic models",
            path="/skills/fastapi-debugging/SKILL.md",
            source="project",
            category="backend",
            tags=["python", "fastapi", "api", "pydantic", "debugging"],
            priority=10,
        ),
        SkillSummary(
            name="pytest-debugging",
            description="Diagnose failing pytest test suites, assertions, fixtures, and mocks",
            path="/skills/pytest-debugging/SKILL.md",
            source="project",
            category="testing",
            tags=["python", "pytest", "test", "testing", "assertions"],
            priority=9,
        ),
        SkillSummary(
            name="git-workflow",
            description="Git branching, rebase, merge conflict resolution, and commit hygiene",
            path="/skills/git-workflow/SKILL.md",
            source="user",
            category="vcs",
            tags=["git", "rebase", "merge", "conflict", "branch"],
            priority=8,
        ),
        SkillSummary(
            name="docker-deploy",
            description="Dockerfile optimization, multi-stage builds, and docker compose deployment",
            path="/skills/docker-deploy/SKILL.md",
            source="user",
            category="devops",
            tags=["docker", "compose", "containers", "deployment"],
            priority=7,
        ),
        SkillSummary(
            name="sql-migration",
            description="SQL schema migrations, postgres query tuning, index optimization",
            path="/skills/sql-migration/SKILL.md",
            source="project",
            category="database",
            tags=["sql", "database", "postgres", "migration", "query"],
            priority=8,
        ),
        SkillSummary(
            name="react-frontend",
            description="React frontend components, hooks lifecycle, state management, and UI styling",
            path="/skills/react-frontend/SKILL.md",
            source="user",
            category="frontend",
            tags=["react", "javascript", "frontend", "ui", "hooks"],
            priority=7,
        ),
        SkillSummary(
            name="rest-api-design",
            description="RESTful API architectural conventions, HTTP status codes, and routing",
            path="/skills/rest-api-design/SKILL.md",
            source="project",
            category="backend",
            tags=["api", "rest", "http", "architecture"],
            priority=6,
        ),
        SkillSummary(
            name="redis-caching",
            description="Redis cache invalidation, key TTL strategies, and pub-sub messaging",
            path="/skills/redis-caching/SKILL.md",
            source="project",
            category="backend",
            tags=["redis", "cache", "caching", "backend"],
            priority=6,
        ),
        SkillSummary(
            name="security-audit",
            description="Static security audit, secret leakage detection, and input sanitization",
            path="/skills/security-audit/SKILL.md",
            source="project",
            category="security",
            tags=["security", "audit", "sanitization", "auth"],
            priority=9,
        ),
    ]


def generate_synthetic_skills(target_count: int) -> list[SkillSummary]:
    """Generate realistic synthetic skills up to target_count."""
    base = create_base_skills()
    if target_count <= len(base):
        return base[:target_count]

    results = list(base)
    domains = [
        ("auth", "Authentication and OAuth2 JWT token lifecycle", ["auth", "jwt", "oauth"]),
        ("kafka", "Apache Kafka event streaming consumer producer", ["kafka", "streaming", "events"]),
        ("graphql", "GraphQL schema resolvers and mutation optimization", ["graphql", "api", "resolvers"]),
        ("k8s", "Kubernetes pod deployments ingress and Helm charts", ["kubernetes", "k8s", "devops"]),
        ("vue", "Vue3 composition API reactive state and components", ["vue", "frontend", "javascript"]),
        ("pandas", "Pandas data processing DataFrame aggregation and ETL", ["python", "pandas", "data"]),
        ("terraform", "Terraform cloud infrastructure as code modules", ["terraform", "iac", "cloud"]),
        ("ci-cd", "GitHub Actions CI/CD workflows and automated releases", ["ci", "cd", "github"]),
        ("logging", "Structured JSON logging and telemetry metrics aggregation", ["logging", "telemetry"]),
        ("nginx", "Nginx reverse proxy SSL termination and rate limiting", ["nginx", "proxy", "server"]),
    ]

    counter = len(results) + 1
    while len(results) < target_count:
        for cat, desc, tags in domains:
            if len(results) >= target_count:
                break
            idx = counter
            counter += 1
            results.append(
                SkillSummary(
                    name=f"{cat}-utility-{idx}",
                    description=f"{desc} (utility {idx})",
                    path=f"/skills/{cat}-utility-{idx}/SKILL.md",
                    source="synthetic",
                    category=cat,
                    tags=tags + [f"sub-{idx}"],
                    priority=max(1, 10 - (idx % 10)),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def run_retrieval_eval(router: SkillRouter, skills: list[SkillSummary]) -> dict[str, float]:
    """Run accuracy benchmark and return standard IR metrics."""
    recall_1_list: list[float] = []
    recall_3_list: list[float] = []
    recall_5_list: list[float] = []
    precision_3_list: list[float] = []
    reciprocal_ranks: list[float] = []

    for case in FIXED_TEST_CASES:
        ranked = router.rank(case.query, skills)
        ranked_names = [c.skill_name for c in ranked]
        expected_set = set(case.expected_relevant)

        top_1 = set(ranked_names[:1])
        top_3 = set(ranked_names[:3])
        top_5 = set(ranked_names[:5])

        r1 = len(top_1 & expected_set) / len(expected_set)
        r3 = len(top_3 & expected_set) / len(expected_set)
        r5 = len(top_5 & expected_set) / len(expected_set)
        p3 = len(top_3 & expected_set) / 3.0

        rr = 0.0
        for rank_idx, name in enumerate(ranked_names, start=1):
            if name in expected_set:
                rr = 1.0 / rank_idx
                break

        recall_1_list.append(r1)
        recall_3_list.append(r3)
        recall_5_list.append(r5)
        precision_3_list.append(p3)
        reciprocal_ranks.append(rr)

    return {
        "Recall@1": round(sum(recall_1_list) / len(recall_1_list), 4),
        "Recall@3": round(sum(recall_3_list) / len(recall_3_list), 4),
        "Recall@5": round(sum(recall_5_list) / len(recall_5_list), 4),
        "Precision@3": round(sum(precision_3_list) / len(precision_3_list), 4),
        "MRR": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
    }


def run_catalog_scale_eval(router: SkillRouter) -> list[dict[str, Any]]:
    """Measure token usage between full catalog vs Top-K catalog across scales."""
    scales = [10, 100, 500]
    sample_query = "FastAPI POST endpoint returns 422 and pytest fails"
    results: list[dict[str, Any]] = []

    for count in scales:
        skills = generate_synthetic_skills(count)
        # Full catalog formatting (as in unrouted system prompt)
        full_catalog_text = "Available skills:\n" + "\n".join(
            f"- {s.name}: {s.description}" for s in skills
        )
        full_tokens = estimate_tokens(full_catalog_text)

        # Routed Top-K catalog formatting (default top_k=5)
        selected, metrics = router.select(sample_query, skills, top_k=5)
        top_k_tokens = metrics.selected_catalog_token_estimate

        token_savings = full_tokens - top_k_tokens
        savings_ratio = round((token_savings / full_tokens) * 100.0, 2) if full_tokens > 0 else 0.0

        results.append(
            {
                "skill_count": count,
                "full_catalog_tokens": full_tokens,
                "top_k_catalog_tokens": top_k_tokens,
                "tokens_saved": token_savings,
                "savings_percentage": f"{savings_ratio}%",
                "selected_skill_count": len(selected),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill Routing Evaluation Benchmark")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    router = SkillRouter(default_top_k=5)
    base_skills = create_base_skills()

    accuracy_metrics = run_retrieval_eval(router, base_skills)
    scale_metrics = run_catalog_scale_eval(router)

    if args.json:
        payload = {
            "retrieval_accuracy": accuracy_metrics,
            "catalog_scale_benchmarks": scale_metrics,
        }
        print(json.dumps(payload, indent=2))
        return

    print("=" * 60)
    print("MiniCode-Python Skill Router Evaluation Benchmark")
    print("=" * 60)
    print("\n1. Retrieval Accuracy Metrics (Curated Test Suite):")
    for metric_name, val in accuracy_metrics.items():
        print(f"  - {metric_name:<14}: {val:.4f}")

    print("\n2. Prompt Token Consumption Across Catalog Scales (Query: FastAPI + pytest):")
    print(f"  {'Skills':<8} | {'Full Catalog':<14} | {'Top-5 Catalog':<14} | {'Saved Tokens':<14} | {'Reduction':<10}")
    print(f"  {'-'*8}-|-{'-'*14}-|-{'-'*14}-|-{'-'*14}-|-{'-'*10}")
    for row in scale_metrics:
        print(
            f"  {row['skill_count']:<8} | "
            f"{row['full_catalog_tokens']:<14} | "
            f"{row['top_k_catalog_tokens']:<14} | "
            f"{row['tokens_saved']:<14} | "
            f"{row['savings_percentage']:<10}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
