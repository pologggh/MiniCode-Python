from __future__ import annotations

import logging
from minicode.skills import load_skill
from minicode.tooling import ToolDefinition, ToolResult

logger = logging.getLogger("tools.load_skill")


def _validate(input_data: dict) -> dict:
    name = input_data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is required")
    name = name.strip()
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("Invalid skill name: path traversal characters are not allowed")
    return {"name": name}


def create_load_skill_tool(cwd: str) -> ToolDefinition:
    def _run(input_data: dict, _context) -> ToolResult:
        skill_name = input_data["name"]
        skill = load_skill(cwd, skill_name)
        if skill is None:
            logger.warning("Attempted to load unknown skill: %s", skill_name)
            return ToolResult(ok=False, output=f"Unknown skill: {skill_name}")

        logger.info(
            "Loaded skill %s from %s (source: %s, category: %s)",
            skill.name,
            skill.path,
            skill.source,
            skill.category,
        )
        return ToolResult(
            ok=True,
            output="\n".join(
                [
                    f"SKILL: {skill.name}",
                    f"SOURCE: {skill.source}",
                    f"PATH: {skill.path}",
                    "",
                    skill.content,
                ]
            ),
        )

    return ToolDefinition(
        name="load_skill",
        description="Load a local SKILL.md by name.",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        validator=_validate,
        run=_run,
    )
