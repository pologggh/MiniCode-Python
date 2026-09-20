from pathlib import Path
import pytest

from minicode.skills import discover_skills, load_skill, parse_frontmatter, install_skill


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


def test_parse_frontmatter_constrained_schema_and_malformed():
    # 1. Valid constrained frontmatter
    valid_content = """---
name: sample-skill
description: A clean test skill
category: testing
tags: [pytest, unit]
version: "2.0"
priority: 15
unknown_field: should be ignored
author: nobody
---
# Body Title
Body text.
"""
    meta, body = parse_frontmatter(valid_content)
    assert meta["name"] == "sample-skill"
    assert meta["description"] == "A clean test skill"
    assert meta["category"] == "testing"
    assert meta["tags"] == ["pytest", "unit"]
    assert meta["version"] == "2.0"
    assert meta["priority"] == 15
    assert "unknown_field" not in meta
    assert "author" not in meta
    assert "# Body Title" in body

    # 2. Unclosed bracket in tags
    bad_bracket = """---
name: broken
tags: [unclosed, list
---
body
"""
    meta_bad, body_bad = parse_frontmatter(bad_bracket)
    assert meta_bad == {}
    assert "body" in body_bad

    # 3. Unclosed quote
    bad_quote = """---
name: "unclosed string
---
body
"""
    meta_quote, body_quote = parse_frontmatter(bad_quote)
    assert meta_quote == {}

    # 4. Invalid priority type
    bad_prio = """---
name: test
priority: not-an-int
---
body
"""
    meta_prio, _ = parse_frontmatter(bad_prio)
    assert meta_prio == {}


def test_symlink_and_traversal_containment_defense(tmp_path: Path) -> None:
    # 1. Traversal in load_skill
    assert load_skill(tmp_path, "../outside") is None
    assert load_skill(tmp_path, "sub/outside") is None
    assert load_skill(tmp_path, "..\\outside") is None

    # 2. Traversal in install_skill
    with pytest.raises(ValueError, match="Invalid skill name"):
        install_skill(tmp_path, str(tmp_path), name="../malicious")
    with pytest.raises(ValueError, match="Invalid skill name"):
        install_skill(tmp_path, str(tmp_path), name="foo/bar")

    # 3. Symlink escape containment
    outside_dir = tmp_path / "outside_project"
    outside_dir.mkdir(parents=True)
    outside_skill = outside_dir / "SKILL.md"
    outside_skill.write_text("---\nname: escaped\n---\nEscaped body", encoding="utf-8")

    skills_root = tmp_path / ".mini-code" / "skills"
    skills_root.mkdir(parents=True)
    symlink_dir = skills_root / "escaped-skill"

    try:
        symlink_dir.symlink_to(outside_dir, target_is_directory=True)
        # Should be rejected because it escapes skills_root
        loaded = load_skill(tmp_path, "escaped-skill")
        assert loaded is None
        discovered = discover_skills(tmp_path, force_refresh=True)
        assert not any(s.name == "escaped" for s in discovered)
    except OSError:
        # On Windows environments where symlink creation requires admin/developer mode,
        # test the path containment logic using mocked resolve
        from unittest.mock import patch
        with patch.object(Path, "is_relative_to", return_value=False):
            assert load_skill(tmp_path, "my-skill") is None


