"""Tool to recover bounded slices of offloaded context artifacts."""
from __future__ import annotations

import logging
from typing import Any

from minicode.context_artifacts import ContextArtifactStore
from minicode.tooling import ToolDefinition, ToolResult

logger = logging.getLogger("tools.load_context_artifact")


def _validate(input_data: dict[str, Any]) -> dict[str, Any]:
    artifact_id = input_data.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("artifact_id is required")
    artifact_id = artifact_id.strip()
    if ".." in artifact_id or "/" in artifact_id or "\\" in artifact_id:
        raise ValueError("Invalid artifact_id: path traversal characters are not allowed")

    offset = input_data.get("offset", 0)
    try:
        offset = int(offset)
        if offset < 0:
            offset = 0
    except (TypeError, ValueError):
        offset = 0

    limit = input_data.get("limit", 4000)
    try:
        limit = int(limit)
        if limit <= 0:
            limit = 4000
    except (TypeError, ValueError):
        limit = 4000
    limit = min(limit, 8000)

    return {"artifact_id": artifact_id, "offset": offset, "limit": limit}


def create_load_context_artifact_tool(
    cwd: str,
    store: ContextArtifactStore | None = None,
    metrics: Any = None,
) -> ToolDefinition:
    """Create the load_context_artifact tool bound to workspace store."""
    artifact_store = store or ContextArtifactStore(cwd)

    def _run(input_data: dict[str, Any], _context: Any) -> ToolResult:
        artifact_id = input_data["artifact_id"]
        offset = input_data.get("offset", 0)
        limit = input_data.get("limit", 4000)

        active_metrics = metrics or getattr(_context, "context_budget_metrics", None) or getattr(_context, "metrics", None)

        try:
            content, meta = artifact_store.read_range(
                artifact_id=artifact_id,
                offset=offset,
                limit=limit,
            )
        except Exception:
            content, meta = None, None
        if content is None or meta is None:
            if active_metrics and hasattr(active_metrics, "recovery_failures"):
                active_metrics.recovery_failures += 1
            logger.warning("Attempted to load missing/inaccessible artifact: %s", artifact_id)
            return ToolResult(
                ok=False,
                output=f"Context artifact not found or inaccessible: {artifact_id}",
            )

        if active_metrics and hasattr(active_metrics, "artifact_recovery_count"):
            active_metrics.artifact_recovery_count += 1

        logger.info(
            "Recovered context artifact %s range [%d:%d] (%d chars)",
            artifact_id,
            offset,
            offset + len(content),
            len(content),
        )
        output_parts = [
            f"ARTIFACT: {meta.artifact_id}",
            f"TOOL: {meta.tool_name}",
            f"RANGE: [offset={offset}, limit={len(content)}] of {meta.original_chars} chars",
            f"TOTAL TOKENS: ~{meta.estimated_tokens}",
            "",
            "--- CONTENT ---",
            content,
        ]
        return ToolResult(ok=True, output="\n".join(output_parts))

    return ToolDefinition(
        name="load_context_artifact",
        description="Load a bounded slice of an offloaded context artifact by its artifact ID (e.g. ctx_...).",
        input_schema={
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "The artifact ID (e.g. ctx_4fd8c13a...)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start reading from (default: 0)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum characters to read (default: 4000, max: 8000)",
                },
            },
            "required": ["artifact_id"],
        },
        validator=_validate,
        run=_run,
    )
