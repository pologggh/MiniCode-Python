"""Agent roles and role policies for Centralized Multi-Agent Orchestration.

Defines role allowlists, capabilities, and system prompts for:
- RESEARCH: Read-only exploration and analysis.
- CODING: Code generation and editing.
- TEST: Test execution and verification.
- REVIEWER: Code inspection and structured review verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentRole(str, Enum):
    RESEARCH = "research"
    CODING = "coding"
    TEST = "test"
    REVIEWER = "reviewer"


@dataclass(frozen=True)
class AgentRolePolicy:
    """Policy governing a sub-agent's permissions, toolset, and boundaries."""
    role: AgentRole
    allowed_tools: frozenset[str]
    is_writer: bool
    max_turns: int
    system_prompt: str


_RESEARCH_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "list_files",
    "grep_files",
    "file_tree",
    "find_symbols",
    "find_references",
    "get_ast_info",
    "web_fetch",
    "web_search",
})

_CODING_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "list_files",
    "grep_files",
    "edit_file",
    "write_file",
    "patch_file",
    "find_symbols",
    "get_ast_info",
    "file_tree",
})

_TEST_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "list_files",
    "grep_files",
    "test_runner",
    "load_context_artifact",
})

_REVIEWER_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "list_files",
    "grep_files",
    "code_review",
    "diff_viewer",
    "find_references",
})


ROLE_POLICIES: dict[AgentRole, AgentRolePolicy] = {
    AgentRole.RESEARCH: AgentRolePolicy(
        role=AgentRole.RESEARCH,
        allowed_tools=_RESEARCH_TOOLS,
        is_writer=False,
        max_turns=8,
        system_prompt=(
            "You are a specialized Research sub-agent. Your duty is to inspect the codebase, "
            "gather context, trace code paths, and analyze implementation or test requirements. "
            "You have READ-ONLY tools. You must not attempt to modify any files. "
            "Provide a concise, highly factual summary of your findings and recommendations."
        ),
    ),
    AgentRole.CODING: AgentRolePolicy(
        role=AgentRole.CODING,
        allowed_tools=_CODING_TOOLS,
        is_writer=True,
        max_turns=12,
        system_prompt=(
            "You are a specialized Coding sub-agent. Your duty is to implement code changes, "
            "refactors, or fixes precisely according to the plan and requirements. "
            "Ensure high code quality, follow existing patterns, and explain what changes were made."
        ),
    ),
    AgentRole.TEST: AgentRolePolicy(
        role=AgentRole.TEST,
        allowed_tools=_TEST_TOOLS,
        is_writer=False,
        max_turns=10,
        system_prompt=(
            "You are a specialized Test sub-agent. Your duty is to verify implementation correctness "
            "by executing test_runner. You have READ-ONLY access and cannot modify code or run arbitrary commands. "
            "You MUST execute test_runner and verify that tests pass (ok=True). If tests fail, report the exact "
            "failure details clearly so the team can address them."
        ),
    ),

    AgentRole.REVIEWER: AgentRolePolicy(
        role=AgentRole.REVIEWER,
        allowed_tools=_REVIEWER_TOOLS,
        is_writer=False,
        max_turns=6,
        system_prompt=(
            "You are a specialized Code Review sub-agent. Your duty is to inspect modified code, "
            "diffs, and verification results. You MUST provide a structured JSON verdict at the end:\n"
            "```json\n"
            "{\n"
            '  "verdict": "approve" | "reject",\n'
            '  "comments": "<summary of review>",\n'
            '  "issues": ["<issue 1>", ...]\n'
            "}\n"
            "```\n"
            "Be objective, thorough, and precise."
        ),
    ),
}


def get_role_policy(role: AgentRole | str) -> AgentRolePolicy:
    """Retrieve the policy configuration for a given agent role."""
    if isinstance(role, str):
        try:
            role_enum = AgentRole(role.lower())
        except ValueError:
            role_enum = AgentRole.RESEARCH
    else:
        role_enum = role

    return ROLE_POLICIES.get(role_enum, ROLE_POLICIES[AgentRole.RESEARCH])
