from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from minicode.context_manager import estimate_tokens
from minicode.skills import SkillSummary

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5
MIN_RELEVANCE_THRESHOLD = 1.0

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "but", "if", "then", "else", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "don", "should", "now", "this", "that", "these", "those", "my",
    "me", "you", "your", "it", "its", "we", "our", "帮我", "请", "如何", "怎么",
    "一个", "进行", "可以", "这个", "那个", "使用", "用", "以及", "并且",
})

_COMMON_CODE_TOKENS = frozenset({
    "fastapi", "pytest", "pydantic", "docker", "git", "react", "sql", "vue",
    "angular", "django", "flask", "redis", "postgres", "mysql", "mongodb",
    "sqlite", "graphql", "rest", "api", "tdd", "ci", "cd", "debug", "debugging",
    "test", "tests", "testing", "refactor", "refactoring", "review", "reviewing",
    "python", "typescript", "javascript", "golang", "rust", "cpp", "c", "bash",
    "linux", "windows", "macos", "html", "css", "node", "npm", "pip",
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CandidateSkill:
    """A scored candidate skill produced by SkillRouter."""

    skill_name: str
    score: float
    matched_fields: list[str]
    category: str
    tags: list[str]
    source: str
    selection_reason: str
    description: str = ""
    path: str = ""
    priority: int = 0
    version: str = "1.0"


@dataclass(slots=True)
class SkillRoutingMetrics:
    """Observable metrics produced on each skill routing execution."""

    query: str
    candidate_count: int
    selected_count: int
    selected_skills: list[str]
    scores: dict[str, float]
    matched_fields: dict[str, list[str]]
    catalog_token_estimate: int
    selected_catalog_token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "selected_skills": list(self.selected_skills),
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "matched_fields": {k: list(v) for k, v in self.matched_fields.items()},
            "catalog_token_estimate": self.catalog_token_estimate,
            "selected_catalog_token_estimate": self.selected_catalog_token_estimate,
        }


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

def normalize_query(query: str) -> tuple[str, list[str]]:
    """Normalize user task query and extract normalized lexical tokens.

    Supports mixed Chinese and English text, preserves common code tokens,
    handles hyphens/underscores without external NLP dependencies.
    """
    if not query:
        return "", []

    lowered = query.lower().strip()
    raw_tokens = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*|[\u4e00-\u9fff]", lowered)

    tokens: list[str] = []
    seen: set[str] = set()

    for tok in raw_tokens:
        tok_clean = tok.strip("-_.")
        if not tok_clean:
            continue

        if "-" in tok_clean or "_" in tok_clean:
            if tok_clean not in seen and tok_clean not in _STOP_WORDS:
                seen.add(tok_clean)
                tokens.append(tok_clean)
            subparts = re.split(r"[-_]", tok_clean)
            for sub in subparts:
                sub_clean = sub.strip()
                if sub_clean and sub_clean not in seen and sub_clean not in _STOP_WORDS:
                    seen.add(sub_clean)
                    tokens.append(sub_clean)
        else:
            if tok_clean not in seen and tok_clean not in _STOP_WORDS:
                seen.add(tok_clean)
                tokens.append(tok_clean)

    return lowered, tokens


def _coerce_skill_summary(skill: SkillSummary | dict[str, Any]) -> SkillSummary:
    """Safely adapt either a SkillSummary object or a dictionary."""
    if isinstance(skill, SkillSummary):
        return skill
    name = str(skill.get("name") or "").strip()
    desc = str(skill.get("description") or "").strip()
    path = str(skill.get("path") or "").strip()
    source = str(skill.get("source") or "unknown").strip()
    category = str(skill.get("category") or "").strip()
    tags = skill.get("tags")
    if isinstance(tags, list):
        normalized_tags = [str(t).strip() for t in tags if str(t).strip()]
    elif isinstance(tags, str):
        normalized_tags = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        normalized_tags = []
    priority = int(skill.get("priority", 0) or 0)
    version = str(skill.get("version", "1.0") or "1.0")
    return SkillSummary(
        name=name,
        description=desc,
        path=path,
        source=source,
        category=category,
        tags=normalized_tags,
        version=version,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Skill Router
# ---------------------------------------------------------------------------

class SkillRouter:
    """Deterministic, explainable lexical skill router.

    Ranks skills based on user task query using lexical signals:
    - Exact skill name match
    - Tag overlap
    - Category overlap
    - Description term overlap
    - Framework / code token clues
    - Priority tie-breaks
    """

    def __init__(self, default_top_k: int = DEFAULT_TOP_K) -> None:
        self.default_top_k = default_top_k

    def recall(
        self,
        query: str,
        skills: list[SkillSummary | dict[str, Any]],
        category_filter: str | None = None,
    ) -> list[SkillSummary]:
        """Recall candidate skills matching broad criteria (e.g. category filter)."""
        recalled: list[SkillSummary] = []
        for raw in skills:
            skill = _coerce_skill_summary(raw)
            if not skill.name:
                continue
            if category_filter:
                if skill.category.lower() != category_filter.lower():
                    continue
            recalled.append(skill)
        return recalled

    def rank(
        self,
        query: str,
        skills: list[SkillSummary | dict[str, Any]],
    ) -> list[CandidateSkill]:
        """Score and rank skills deterministically against the user task query.

        Sort order: score desc -> priority desc -> skill_name asc.
        """
        lowered_query, query_tokens = normalize_query(query)
        query_token_set = frozenset(query_tokens)

        candidates: list[CandidateSkill] = []

        for raw in skills:
            skill = _coerce_skill_summary(raw)
            if not skill.name:
                continue

            score = 0.0
            matched_fields: list[str] = []
            reasons: list[str] = []

            skill_name_lower = skill.name.lower()
            skill_name_normalized = skill_name_lower.replace("-", " ").replace("_", " ")

            # 1. Exact Name Matching
            if skill_name_lower == lowered_query or skill_name_normalized == lowered_query:
                score += 15.0
                matched_fields.append("exact_name")
                reasons.append("exact skill name match")
            elif skill_name_lower in lowered_query or skill_name_normalized in lowered_query:
                score += 10.0
                matched_fields.append("name_in_query")
                reasons.append(f"skill name '{skill.name}' mentioned in query")
            else:
                # Subparts of name matching query
                name_subparts = [p for p in re.split(r"[-_\s]+", skill_name_lower) if p and p not in _STOP_WORDS]
                if name_subparts and all(p in query_token_set for p in name_subparts):
                    score += 6.0
                    matched_fields.append("all_name_tokens")
                    reasons.append(f"all name tokens of '{skill.name}' matched")
                elif any(p in query_token_set for p in name_subparts):
                    overlap = [p for p in name_subparts if p in query_token_set]
                    score += 2.0 * len(overlap)
                    matched_fields.append("partial_name_tokens")
                    reasons.append(f"matched name keywords: {', '.join(overlap)}")

            # 2. Tag Matching
            matched_tags: list[str] = []
            for tag in skill.tags:
                tag_lower = tag.lower().strip()
                if not tag_lower:
                    continue
                if tag_lower in query_token_set or tag_lower in lowered_query:
                    matched_tags.append(tag)
                    score += 3.0

            if matched_tags:
                matched_fields.append("tags")
                reasons.append(f"matched tags: {', '.join(matched_tags)}")

            # 3. Category Matching
            if skill.category:
                cat_lower = skill.category.lower().strip()
                if cat_lower and (cat_lower in query_token_set or cat_lower in lowered_query):
                    score += 2.0
                    matched_fields.append("category")
                    reasons.append(f"matched category '{skill.category}'")

            # 4. Description Term Overlap
            if skill.description:
                desc_tokens = [
                    tok.strip("-_.") for tok in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", skill.description.lower())
                    if tok and tok not in _STOP_WORDS
                ]
                desc_matches = [t for t in desc_tokens if t in query_token_set]
                # Unique matching terms
                unique_desc_matches = list(dict.fromkeys(desc_matches))
                if unique_desc_matches:
                    overlap_score = min(len(unique_desc_matches) * 1.0, 5.0)
                    score += overlap_score
                    matched_fields.append("description")
                    reasons.append(f"matched description terms: {', '.join(unique_desc_matches[:4])}")

            # 5. Common Code Clues (framework/language)
            clue_matches = [c for c in _COMMON_CODE_TOKENS if c in skill_name_lower and c in query_token_set]
            if clue_matches and "exact_name" not in matched_fields and "name_in_query" not in matched_fields:
                score += 2.0
                matched_fields.append("framework_clue")
                reasons.append(f"matched framework clue '{clue_matches[0]}'")

            # 6. Priority small deterministic bonus
            if skill.priority > 0:
                score += min(skill.priority * 0.05, 1.0)

            reason_str = "; ".join(reasons) if reasons else "baseline catalog entry"

            candidates.append(
                CandidateSkill(
                    skill_name=skill.name,
                    score=round(score, 4),
                    matched_fields=matched_fields,
                    category=skill.category,
                    tags=skill.tags,
                    source=skill.source,
                    selection_reason=reason_str,
                    description=skill.description,
                    path=skill.path,
                    priority=skill.priority,
                    version=skill.version,
                )
            )

        # Deterministic sort: score desc -> priority desc -> skill_name asc
        candidates.sort(key=lambda c: (-c.score, -c.priority, c.skill_name))
        return candidates

    def select(
        self,
        query: str,
        skills: list[SkillSummary | dict[str, Any]],
        top_k: int | None = None,
        category_filter: str | None = None,
        min_threshold: float = MIN_RELEVANCE_THRESHOLD,
        token_budget: int | None = None,
    ) -> tuple[list[CandidateSkill], SkillRoutingMetrics]:
        """Perform full recall -> rank -> select pipeline."""
        k = top_k if top_k is not None else self.default_top_k
        recalled = self.recall(query, skills, category_filter=category_filter)
        ranked = self.rank(query, recalled)

        # Filter by threshold (empty query yields empty selection if threshold > 0)
        has_query = bool(query and query.strip())
        if has_query:
            filtered = [c for c in ranked if c.score >= min_threshold]
        else:
            filtered = []

        selected = filtered[:k]

        # Budget constraint if specified
        if token_budget is not None and token_budget > 0:
            budget_trimmed: list[CandidateSkill] = []
            current_tokens = 0
            for cand in selected:
                cand_text = f"{cand.skill_name}: {cand.description}"
                cand_tokens = estimate_tokens(cand_text)
                if current_tokens + cand_tokens > token_budget and budget_trimmed:
                    break
                budget_trimmed.append(cand)
                current_tokens += cand_tokens
            selected = budget_trimmed

        # Token estimates
        full_catalog_text = "\n".join(f"- {s.name}: {s.description}" for s in recalled)
        catalog_tokens = estimate_tokens(full_catalog_text) if recalled else 0

        selected_catalog_text = self.format_catalog(selected) if selected else ""
        selected_tokens = estimate_tokens(selected_catalog_text) if selected else 0

        metrics = SkillRoutingMetrics(
            query=query,
            candidate_count=len(skills),
            selected_count=len(selected),
            selected_skills=[c.skill_name for c in selected],
            scores={c.skill_name: c.score for c in selected},
            matched_fields={c.skill_name: c.matched_fields for c in selected},
            catalog_token_estimate=catalog_tokens,
            selected_catalog_token_estimate=selected_tokens,
        )

        return selected, metrics

    def format_catalog(self, candidates: list[CandidateSkill]) -> str:
        """Format selected top-k candidates for injection into the system prompt."""
        if not candidates:
            return ""

        lines = ["Available skills (task-relevant):"]
        for idx, cand in enumerate(candidates, 1):
            category_line = f"   category: {cand.category}\n" if cand.category else ""
            reason_line = f"   reason: {cand.selection_reason}\n" if cand.selection_reason else ""
            lines.append(
                f"{idx}. {cand.skill_name}\n"
                f"{category_line}"
                f"{reason_line}"
                f"   description: {cand.description or 'No description provided.'}"
            )
        return "\n\n".join(lines)

    def format_fallback_catalog(self, has_skills: bool = True) -> str:
        """Format fallback message when no candidate matches the threshold."""
        if not has_skills:
            return (
                "Available skills:\n- none discovered\n"
                "Tip: Install skills via `npx superpowers-zh` in your project directory"
            )
        return (
            "Available skills (task-relevant):\n"
            "- No specific skills matched this task query.\n"
            "- You may call `load_skill(name)` directly if the user requests or you know a relevant installed skill."
        )


_global_router: SkillRouter | None = None


def get_skill_router(top_k: int = DEFAULT_TOP_K) -> SkillRouter:
    """Return a configured SkillRouter instance."""
    configured_k = os.environ.get("MINI_CODE_SKILL_TOP_K")
    if configured_k:
        try:
            top_k = int(configured_k)
        except ValueError:
            pass
    return SkillRouter(default_top_k=top_k)
