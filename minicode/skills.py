from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(slots=True)
class SkillMetadata:
    """Structured metadata parsed from SKILL.md frontmatter."""

    name: str = ""
    description: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)
    version: str = "1.0"
    priority: int = 0


@dataclass(slots=True)
class SkillSummary:
    """Lightweight summary of a skill for catalog listing and prompt routing."""

    name: str
    description: str
    path: str
    source: str
    category: str = ""
    tags: list[str] = field(default_factory=list)
    version: str = "1.0"
    priority: int = 0
    metadata: SkillMetadata | None = None


@dataclass(slots=True)
class LoadedSkill(SkillSummary):
    """Full skill object with markdown content loaded on demand."""

    content: str = ""


_FRONTMATTER_PATTERN = re.compile(
    r"^---\r?\n(.*?)\r?\n---(?:\r?\n(.*))?$",
    re.DOTALL,
)


def _parse_skill_frontmatter(yaml_text: str) -> dict[str, Any]:
    """Deterministic, zero-dependency parser for constrained skill frontmatter.

    Strictly supports only the 6 defined schema fields:
    name, description, category, tags, version, priority.
    Raises ValueError on malformed syntax (e.g. unclosed brackets, braces, quotes).
    """
    result: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in yaml_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check for unclosed flow syntax or unclosed quotes
        if stripped.count("[") != stripped.count("]"):
            raise ValueError(f"Unclosed bracket in YAML line: {stripped}")
        if stripped.count("{") != stripped.count("}"):
            raise ValueError(f"Unclosed brace in YAML line: {stripped}")
        unquoted = stripped.replace(r'\"', '').replace(r"\'", '')
        if unquoted.count('"') % 2 != 0 or unquoted.count("'") % 2 != 0:
            raise ValueError(f"Unclosed quote in YAML line: {stripped}")

        # Bullet list item under current_key (e.g. - python)
        if stripped.startswith("- "):
            if current_key == "tags":
                item = stripped[2:].strip().strip("\"'")
                if "tags" not in result or not isinstance(result["tags"], list):
                    result["tags"] = []
                if item:
                    result["tags"].append(item)
            continue

        if ":" not in stripped:
            raise ValueError(f"Missing colon in YAML line: {stripped}")

        key, sep, val = stripped.partition(":")
        key = key.strip().lower()
        val = val.strip()

        # Ignore unsupported frontmatter fields to remain strictly constrained
        if key not in ("name", "description", "category", "tags", "version", "priority"):
            current_key = None
            continue

        current_key = key

        if not val:
            if key == "tags" and "tags" not in result:
                result["tags"] = []
            continue

        val_unquoted = val.strip("\"'")

        if key == "priority":
            try:
                result["priority"] = int(val_unquoted)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid integer for priority: {val}")
        elif key == "tags":
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1]
                parts = [p.strip().strip("\"'") for p in inner.split(",") if p.strip()]
                result["tags"] = parts
            elif "," in val:
                parts = [p.strip().strip("\"'") for p in val.split(",") if p.strip()]
                result["tags"] = parts
            else:
                result["tags"] = [val_unquoted] if val_unquoted else []
        else:
            result[key] = val_unquoted

    return result


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract and parse optional YAML frontmatter using the canonical constrained parser.

    Returns:
        (frontmatter_dict, body_content). If frontmatter is missing or
        malformed, returns ({}, content) without raising exceptions.
    """
    clean_text = content.lstrip("\ufeff")
    if not clean_text.startswith("---"):
        return {}, content

    match = _FRONTMATTER_PATTERN.match(clean_text)
    if not match:
        return {}, content

    yaml_text, body = match.groups()
    body = body or ""

    try:
        parsed = _parse_skill_frontmatter(yaml_text)
        if isinstance(parsed, dict):
            return parsed, body
    except Exception:
        pass

    return {}, body


def extract_skill_metadata(content: str, dir_name: str) -> tuple[SkillMetadata, str]:
    """Parse skill metadata from frontmatter or infer from body and directory name."""
    raw_meta, body = parse_frontmatter(content)

    name = str(raw_meta.get("name") or dir_name).strip()
    if not name:
        name = dir_name

    desc = str(raw_meta.get("description") or "").strip()
    if not desc:
        desc = extract_description(body)

    category = str(raw_meta.get("category") or "").strip()

    raw_tags = raw_meta.get("tags")
    if isinstance(raw_tags, list):
        tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    elif isinstance(raw_tags, str):
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    else:
        tags = []

    version = str(raw_meta.get("version") or "1.0").strip()

    raw_prio = raw_meta.get("priority")
    try:
        priority = int(raw_prio)
    except (TypeError, ValueError):
        priority = 0

    metadata = SkillMetadata(
        name=name,
        description=desc,
        category=category,
        tags=tags,
        version=version,
        priority=priority,
    )
    return metadata, body


def extract_description(markdown: str) -> str:
    normalized = markdown.replace("\r\n", "\n")
    paragraphs = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    for block in paragraphs:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        for line in lines:
            if not line.startswith("#"):
                return line.replace("`", "")
    for line in normalized.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped.replace("`", "")
    return "No description provided."


def _home_dir() -> Path:
    return Path.home()


def _skill_roots(cwd: str | Path) -> list[tuple[Path, str]]:
    base = Path(cwd)
    home = _home_dir()
    return [
        (base / ".mini-code" / "skills", "project"),
        (home / ".mini-code" / "skills", "user"),
        (base / ".claude" / "skills", "compat_project"),
        (home / ".claude" / "skills", "compat_user"),
    ]


def _list_skill_dirs(root: Path, source: str) -> list[LoadedSkill]:
    if not root.exists():
        return []
    try:
        resolved_root = root.resolve()
    except OSError:
        return []

    results: list[LoadedSkill] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []

    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            resolved_entry = entry.resolve()
            if not resolved_entry.is_relative_to(resolved_root):
                continue
        except (OSError, ValueError):
            # Windows: untrusted mount points, broken symlinks, escape attempts, etc.
            continue
        skill_path = entry / "SKILL.md"
        if not skill_path.exists():
            continue
        try:
            resolved_skill_path = skill_path.resolve()
            if not resolved_skill_path.is_relative_to(resolved_root):
                continue
            content = skill_path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            continue

        metadata, body = extract_skill_metadata(content, entry.name)
        results.append(
            LoadedSkill(
                name=metadata.name,
                description=metadata.description,
                path=str(skill_path),
                source=source,
                category=metadata.category,
                tags=metadata.tags,
                version=metadata.version,
                priority=metadata.priority,
                metadata=metadata,
                content=content,
            )
        )
    return results


_discovery_cache: dict[str, tuple[float, list[SkillSummary]]] = {}
_CACHE_TTL_SECONDS = 30.0


def clear_skill_cache() -> None:
    """Clear the cached skill discovery results."""
    _discovery_cache.clear()


def discover_skills(cwd: str | Path, *, force_refresh: bool = False) -> list[SkillSummary]:
    key = str(Path(cwd).resolve())
    now = time.monotonic()
    if not force_refresh and key in _discovery_cache:
        timestamp, cached = _discovery_cache[key]
        if now - timestamp < _CACHE_TTL_SECONDS:
            return list(cached)

    by_name: dict[str, LoadedSkill] = {}
    for root, source in _skill_roots(cwd):
        for skill in _list_skill_dirs(root, source):
            by_name.setdefault(skill.name, skill)

    summaries = [
        SkillSummary(
            name=skill.name,
            description=skill.description,
            path=skill.path,
            source=skill.source,
            category=skill.category,
            tags=skill.tags,
            version=skill.version,
            priority=skill.priority,
            metadata=skill.metadata,
        )
        for skill in by_name.values()
    ]
    _discovery_cache[key] = (now, summaries)
    return summaries


def load_skill(cwd: str | Path, name: str) -> LoadedSkill | None:
    normalized_name = name.strip()
    if not normalized_name:
        return None
    # Security: strictly prevent directory traversal
    if ".." in normalized_name or "/" in normalized_name or "\\" in normalized_name:
        return None

    # Fast path: check direct directory name match across root precedence
    for root, source in _skill_roots(cwd):
        if not root.exists():
            continue
        try:
            resolved_root = root.resolve()
        except OSError:
            continue
        skill_path = root / normalized_name / "SKILL.md"
        if skill_path.exists():
            try:
                resolved_skill = skill_path.resolve()
                if not resolved_skill.is_relative_to(resolved_root):
                    continue
                content = skill_path.read_text(encoding="utf-8")
                metadata, body = extract_skill_metadata(content, normalized_name)
                return LoadedSkill(
                    name=metadata.name,
                    description=metadata.description,
                    path=str(skill_path),
                    source=source,
                    category=metadata.category,
                    tags=metadata.tags,
                    version=metadata.version,
                    priority=metadata.priority,
                    metadata=metadata,
                    content=content,
                )
            except (OSError, ValueError):
                continue

    # Fallback path: scan directories in precedence order for matching metadata.name
    for root, source in _skill_roots(cwd):
        if not root.exists():
            continue
        try:
            resolved_root = root.resolve()
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
                resolved_entry = entry.resolve()
                if not resolved_entry.is_relative_to(resolved_root):
                    continue
                skill_path = entry / "SKILL.md"
                if not skill_path.exists():
                    continue
                resolved_skill = skill_path.resolve()
                if not resolved_skill.is_relative_to(resolved_root):
                    continue
                content = skill_path.read_text(encoding="utf-8")
                metadata, body = extract_skill_metadata(content, entry.name)
                if metadata.name == normalized_name:
                    return LoadedSkill(
                        name=metadata.name,
                        description=metadata.description,
                        path=str(skill_path),
                        source=source,
                        category=metadata.category,
                        tags=metadata.tags,
                        version=metadata.version,
                        priority=metadata.priority,
                        metadata=metadata,
                        content=content,
                    )
            except (OSError, ValueError):
                continue

    return None


def _managed_skill_root(scope: str, cwd: str | Path) -> Path:
    return (Path(cwd) / ".mini-code" / "skills") if scope == "project" else (_home_dir() / ".mini-code" / "skills")


def install_skill(cwd: str | Path, source_path: str, name: str | None = None, scope: str = "user") -> dict[str, str]:
    if name is not None:
        trimmed_name = name.strip()
        if ".." in trimmed_name or "/" in trimmed_name or "\\" in trimmed_name:
            raise ValueError(f"Invalid skill name: {name}")

    source = Path(source_path)
    if not source.is_absolute():
        source = Path(cwd) / source
    if source.is_dir():
        skill_file = source / "SKILL.md"
        inferred_name = source.name
    else:
        skill_file = source if source.name == "SKILL.md" else source / "SKILL.md"
        inferred_name = skill_file.parent.name
    if not skill_file.exists():
        raise RuntimeError(f"No SKILL.md found in {source}")

    skill_name = (name or inferred_name).strip()
    if not skill_name:
        raise RuntimeError("Skill name cannot be empty.")
    if ".." in skill_name or "/" in skill_name or "\\" in skill_name:
        raise ValueError(f"Invalid skill name: {skill_name}")

    target_dir = _managed_skill_root(scope, cwd) / skill_name
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(skill_file, target_dir / "SKILL.md")
    clear_skill_cache()
    return {"name": skill_name, "targetPath": str(target_dir / "SKILL.md")}


def remove_managed_skill(cwd: str | Path, name: str, scope: str = "user") -> dict[str, object]:
    target_path = _managed_skill_root(scope, cwd) / name
    if not target_path.exists():
        return {"removed": False, "targetPath": str(target_path)}
    shutil.rmtree(target_path)
    clear_skill_cache()
    return {"removed": True, "targetPath": str(target_path)}


