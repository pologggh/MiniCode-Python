from pathlib import Path

from minicode.prompt import build_system_prompt


def test_build_system_prompt_includes_skills_and_mcp(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        str(tmp_path),
        ["cwd: test"],
        {
            "skills": [{"name": "demo", "description": "demo skill"}],
            "mcpServers": [{"name": "fake", "status": "connected", "toolCount": 1, "resourceCount": 1, "promptCount": 1, "protocol": "newline-json"}],
        },
    )

    assert "Available skills:" in prompt
    assert "demo skill" in prompt
    assert "Configured MCP servers:" in prompt
    assert "fake: connected, tools=1" in prompt


def test_build_system_prompt_mentions_sequential_thinking_server(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        str(tmp_path),
        [],
        {
            "mcpServers": [
                {"name": "SequentialThinking", "status": "connected", "toolCount": 1}
            ]
        },
    )

    assert "SEQUENTIAL THINKING MCP SERVER IS CONNECTED" in prompt
    assert "sequential_thinking" in prompt


def test_build_system_prompt_includes_memory_context(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        str(tmp_path),
        [],
        {"memory_context": "# Project Memory\n\n- Always run pytest before release."},
    )

    assert "Project Memory & Context" in prompt
    assert "Always run pytest before release." in prompt


# ---------------------------------------------------------------------------
# Robustness: the system-prompt builder runs every turn; malformed MCP/skill/
# permission inputs must not crash it.
# ---------------------------------------------------------------------------


def test_build_system_prompt_bundle_handles_malformed_mcp_entry():
    """A partial MCP server dict (missing toolCount/name/status) or a non-dict
    entry must not KeyError/AttributeError the prompt build."""
    from minicode.prompt import build_system_prompt_bundle

    extras = {
        "mcpServers": [
            {"name": "broken", "status": "error"},  # missing toolCount
            "not-a-dict",  # wholly malformed
            {"name": "sequential-thinking", "status": "connected", "toolCount": 2},
        ],
        "skills": [],
        "memory_context": "",
        "runtime": {},
    }
    bundle = build_system_prompt_bundle(".", ["cwd: ."], extras)
    assert isinstance(bundle.prompt, str) and bundle.prompt
    assert "sequential-thinking" in bundle.prompt


def test_build_system_prompt_bundle_handles_none_in_permission_summary():
    """A None element in permission_summary must not crash join()."""
    from minicode.prompt import build_system_prompt_bundle

    bundle = build_system_prompt_bundle(
        ".", ["ok", None, "x"], {"mcpServers": [], "skills": [], "memory_context": "", "runtime": {}}
    )
    assert "Permission context" in bundle.prompt
    assert "ok" in bundle.prompt and "x" in bundle.prompt


def test_build_system_prompt_bundle_handles_malformed_skill():
    """A skill dict missing name/description must not KeyError the build."""
    from minicode.prompt import build_system_prompt_bundle

    bundle = build_system_prompt_bundle(
        ".", [], {"mcpServers": [], "skills": [{"name": "s"}], "memory_context": "", "runtime": {}}
    )
    assert isinstance(bundle.prompt, str) and bundle.prompt


def test_build_system_prompt_adaptive_skill_routing_injects_task_relevant_skills():
    from minicode.prompt import build_system_prompt

    skills = [
        {
            "name": "fastapi-debugging",
            "description": "Debug FastAPI endpoint failures",
            "category": "backend",
            "tags": ["fastapi", "api"],
            "priority": 10,
        },
        {
            "name": "react-ui",
            "description": "Frontend React component library",
            "category": "frontend",
            "tags": ["react", "ui"],
            "priority": 5,
        },
    ]

    prompt = build_system_prompt(
        ".",
        [],
        {
            "skills": skills,
            "user_query": "Help me fix this FastAPI request crash",
        },
    )

    assert "Available skills (task-relevant):" in prompt
    assert "fastapi-debugging" in prompt
    assert "reason:" in prompt
    assert "react-ui" not in prompt  # Irrelevant skill excluded from top-k


def test_build_system_prompt_adaptive_skill_routing_fallback_when_unmatched():
    from minicode.prompt import build_system_prompt

    skills = [
        {
            "name": "fastapi-debugging",
            "description": "Debug FastAPI endpoint failures",
            "category": "backend",
            "tags": ["fastapi", "api"],
            "priority": 10,
        }
    ]

    prompt = build_system_prompt(
        ".",
        [],
        {
            "skills": skills,
            "user_query": "unrelated quantum mechanics calculation",
        },
    )

    assert "No specific skills matched this task query" in prompt
    assert "load_skill(name)" in prompt


def test_build_system_prompt_does_not_inject_full_skill_body():
    from minicode.prompt import build_system_prompt

    full_secret_content = "DETAILED_INSTRUCTION_BODY_SECRET_CODE_12345"
    skills = [
        {
            "name": "secure-skill",
            "description": "A secure skill summary",
            "content": full_secret_content,  # LoadedSkill content should NOT be in prompt
        }
    ]

    prompt = build_system_prompt(
        ".",
        [],
        {
            "skills": skills,
            "user_query": "use secure-skill",
        },
    )

    assert "secure-skill" in prompt
    assert "A secure skill summary" in prompt
    assert full_secret_content not in prompt

