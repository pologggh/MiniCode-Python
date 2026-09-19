from pathlib import Path
import pytest

from minicode.skills import discover_skills, load_skill


@pytest.fixture(autouse=True)
def isolate_home(tmp_path: Path, monkeypatch) -> None:
    empty_home = tmp_path / "test_user_home"
    empty_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.setenv("USERPROFILE", str(empty_home))


def test_discover_skills_prefers_project_root(tmp_path: Path, monkeypatch) -> None:
    project_skill = tmp_path / ".mini-code" / "skills" / "demo" / "SKILL.md"
    project_skill.parent.mkdir(parents=True)
    project_skill.write_text("# Demo\n\nProject description\n", encoding="utf-8")

    user_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("USERPROFILE", str(user_home))
    user_skill = user_home / ".mini-code" / "skills" / "demo" / "SKILL.md"
    user_skill.parent.mkdir(parents=True)
    user_skill.write_text("# Demo\n\nUser description\n", encoding="utf-8")

    skills = discover_skills(tmp_path, force_refresh=True)

    assert len(skills) == 1
    assert skills[0].description == "Project description"
    loaded = load_skill(tmp_path, "demo")
    assert loaded is not None
    assert loaded.content.startswith("# Demo")


def test_skill_with_valid_frontmatter(tmp_path: Path) -> None:
    skill_file = tmp_path / ".mini-code" / "skills" / "fastapi-dbg" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    content = """---
name: fastapi-debugging
description: Debug FastAPI request and response failures
category: backend
tags:
  - python
  - fastapi
  - api
  - debugging
version: "1.0"
priority: 10
---

# FastAPI Debugging
This is the body content.
"""
    skill_file.write_text(content, encoding="utf-8")

    skills = discover_skills(tmp_path, force_refresh=True)
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "fastapi-debugging"
    assert skill.description == "Debug FastAPI request and response failures"
    assert skill.category == "backend"
    assert skill.tags == ["python", "fastapi", "api", "debugging"]
    assert skill.priority == 10
    assert skill.version == "1.0"
    assert skill.metadata is not None
    assert skill.metadata.name == "fastapi-debugging"

    loaded = load_skill(tmp_path, "fastapi-debugging")
    assert loaded is not None
    assert loaded.name == "fastapi-debugging"
    assert "This is the body content." in loaded.content


def test_skill_with_malformed_frontmatter_fallback(tmp_path: Path) -> None:
    skill_file = tmp_path / ".mini-code" / "skills" / "broken-yaml" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    content = """---
name: broken
description: [unclosed list
priority: bad_int: {invalid}
---

# Title
Fallback description paragraph.
"""
    skill_file.write_text(content, encoding="utf-8")

    # Should not crash, should fall back gracefully
    skills = discover_skills(tmp_path, force_refresh=True)
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name in ("broken-yaml", "broken")
    # Body description should be extracted when frontmatter is malformed
    assert "Fallback description paragraph." in skill.description or "No description" in skill.description


def test_skill_frontmatter_fallback_fields(tmp_path: Path) -> None:
    skill_file = tmp_path / ".mini-code" / "skills" / "my-skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    content = """---
category: testing
tags:
  - pytest
---

# Header
Extracted description from markdown.
"""
    skill_file.write_text(content, encoding="utf-8")

    skills = discover_skills(tmp_path, force_refresh=True)
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "my-skill"  # fallback to dir name
    assert skill.description == "Extracted description from markdown."
    assert skill.category == "testing"
    assert skill.tags == ["pytest"]
    assert skill.priority == 0


def test_load_skill_prevents_directory_traversal(tmp_path: Path) -> None:
    assert load_skill(tmp_path, "../outside") is None
    assert load_skill(tmp_path, "foo/bar") is None
    assert load_skill(tmp_path, "..\\outside") is None


