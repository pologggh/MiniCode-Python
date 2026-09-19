from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Optional PyYAML support with robust zero-dependency fallback
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


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


def _simple_yaml_parse(yaml_text: str) -> dict[str, Any]:
    """Lightweight, zero-dependency parser for standard skill frontmatter."""
    result: dict[str, Any] = {}
    current_key: str | None = None
    for line in yaml_text.splitlines():
        line = line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key is not None:
            item = stripped[2:].strip().strip("\"'")
            if not isinstance(result.get(current_key), list):
                result[current_key] = []
            result[current_key].append(item)
            continue
        if ":" in stripped:
            key, sep, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip("\"'")
            if not val:
                result[key] = []
                current_key = key
            else:
                if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
                    result[key] = int(val)
                elif val.lower() == "true":
                    result[key] = True
                elif val.lower() == "false":
                    result[key] = False
                else:
                    result[key] = val
                current_key = key
    return result


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract and parse optional YAML frontmatter from markdown content.

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

    # Try PyYAML if available
    if _HAS_YAML:
        try:
            parsed = yaml.safe_load(yaml_text)
            if isinstance(parsed, dict):
                return parsed, body
            return {}, body
        except Exception:
            # PyYAML failed to parse malformed frontmatter
            return {}, body

    # Fallback to internal lightweight parser when PyYAML is not installed
    try:
        parsed = _simple_yaml_parse(yaml_text)
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
    results: list[LoadedSkill] = []
    for entry in root.iterdir():
        try:
            if not entry.is_dir():
                continue
        except OSError:
            # Windows: untrusted mount points, broken symlinks, etc.
            continue
        skill_path = entry / "SKILL.md"
        if not skill_path.exists():
            continue
        try:
            content = skill_path.read_text(encoding="utf-8")
        except OSError:
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
        skill_path = root / normalized_name / "SKILL.md"
        if skill_path.exists():
            try:
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
            except OSError:
                continue

    # Fallback path: scan directories in precedence order for matching metadata.name
    for root, source in _skill_roots(cwd):
        if not root.exists():
            continue
        try:
            for entry in root.iterdir():
                if not entry.is_dir():
                    continue
                skill_path = entry / "SKILL.md"
                if not skill_path.exists():
                    continue
                try:
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
                except OSError:
                    continue
        except OSError:
            continue

    return None


def _managed_skill_root(scope: str, cwd: str | Path) -> Path:
    return (Path(cwd) / ".mini-code" / "skills") if scope == "project" else (_home_dir() / ".mini-code" / "skills")


def install_skill(cwd: str | Path, source_path: str, name: str | None = None, scope: str = "user") -> dict[str, str]:
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


