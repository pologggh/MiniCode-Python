"""Recoverable workspace-confined Context Artifact Store.

Persists oversized tool results and context data into workspace-confined disk artifacts
with stable SHA-256 IDs, metadata sidecars, secret-redacted previews, and bounded slice retrieval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from minicode.context_manager import estimate_tokens
from minicode.execution_trace import sanitize_text
from minicode.logging_config import get_logger

logger = get_logger("context_artifacts")

_ARTIFACT_ID_PATTERN = re.compile(r"^ctx_[a-f0-9]{16,64}$")
DEFAULT_MAX_RECOVERY_CHARS = 8000
DEFAULT_PREVIEW_CHARS = 400


@dataclass
class ContextArtifactMetadata:
    """Metadata tracking persisted context artifacts."""
    artifact_id: str
    tool_name: str
    created_at: float
    original_chars: int
    estimated_tokens: int
    sha256: str
    relative_path: str
    preview: str
    content_type: str = "text/plain"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextArtifactMetadata:
        return cls(
            artifact_id=data["artifact_id"],
            tool_name=data.get("tool_name", "unknown"),
            created_at=float(data.get("created_at", time.time())),
            original_chars=int(data.get("original_chars", 0)),
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            sha256=data.get("sha256", ""),
            relative_path=data.get("relative_path", ""),
            preview=data.get("preview", ""),
            content_type=data.get("content_type", "text/plain"),
        )


class ContextArtifactStore:
    """Workspace-confined storage for context artifacts with security hardening."""

    def __init__(
        self,
        workspace: str | Path | None = None,
        store_dir_name: str = ".mini-code-tool-results",
    ):
        self._workspace = Path(workspace).resolve() if workspace else Path.cwd().resolve()
        self._store_dir = (self._workspace / store_dir_name).resolve()

    @property
    def store_dir(self) -> Path:
        return self._store_dir

    def _ensure_dir(self) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def _validate_artifact_id(self, artifact_id: str) -> str:
        """Validate artifact ID format and prevent path traversal."""
        if not isinstance(artifact_id, str):
            raise ValueError("Artifact ID must be a string")
        clean_id = artifact_id.strip()
        if not _ARTIFACT_ID_PATTERN.match(clean_id):
            raise ValueError(
                f"Invalid artifact ID '{clean_id}': must match format 'ctx_<16-64 hex chars>'"
            )
        if ".." in clean_id or "/" in clean_id or "\\" in clean_id or "\x00" in clean_id:
            raise ValueError(f"Path traversal detected in artifact ID: '{clean_id}'")
        return clean_id

    def _get_paths(self, clean_id: str) -> tuple[Path, Path]:
        """Resolve content and meta paths with symlink escape validation."""
        self._ensure_dir()
        content_path = (self._store_dir / f"{clean_id}.txt").resolve()
        meta_path = (self._store_dir / f"{clean_id}.meta.json").resolve()
        # Strict Path containment validation (prevents sibling-prefix escapes and symlink escapes)
        resolved_store = Path(os.path.realpath(self._store_dir)).resolve()
        resolved_content = Path(os.path.realpath(content_path)).resolve()
        resolved_meta = Path(os.path.realpath(meta_path)).resolve()

        if not resolved_content.is_relative_to(resolved_store) or not resolved_meta.is_relative_to(resolved_store):
            raise ValueError(f"Symlink escape attempt detected for artifact ID '{clean_id}'")

        return content_path, meta_path

    def persist(
        self,
        content: str,
        tool_name: str = "unknown",
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
    ) -> ContextArtifactMetadata:
        """Persist text content into an artifact atomically with metadata."""
        self._ensure_dir()
        if not isinstance(content, str):
            content = "" if content is None else str(content)

        # Stable hash ID
        content_bytes = content.encode("utf-8")
        full_hash = hashlib.sha256(content_bytes).hexdigest()
        artifact_id = f"ctx_{full_hash[:16]}"
        content_path, meta_path = self._get_paths(artifact_id)

        # Generate secret-redacted preview preserving head and failure/tail lines
        lines = content.splitlines()
        if len(lines) <= 15 or len(content) <= preview_chars:
            raw_preview = content[:preview_chars]
        else:
            head_lines = lines[:6]
            tail_lines = lines[-10:]
            omitted = len(lines) - len(head_lines) - len(tail_lines)
            preview_parts = list(head_lines)
            if omitted > 0:
                preview_parts.append(f"... [{omitted} lines omitted] ...")
            preview_parts.extend(tail_lines)
            raw_preview = "\n".join(preview_parts)[: max(preview_chars, 800)]

        sanitized_preview = sanitize_text(raw_preview, max_length=len(raw_preview))

        tokens = estimate_tokens(content)
        rel_path = str(content_path.relative_to(self._workspace))

        meta = ContextArtifactMetadata(
            artifact_id=artifact_id,
            tool_name=tool_name,
            created_at=time.time(),
            original_chars=len(content),
            estimated_tokens=tokens,
            sha256=full_hash,
            relative_path=rel_path,
            preview=sanitized_preview,
            content_type="text/plain",
        )

        # Atomic write content
        tmp_fd, tmp_content = tempfile.mkstemp(
            dir=str(self._store_dir), prefix=".artifact_content_", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="") as f:
                f.write(content)
            os.replace(tmp_content, str(content_path))
        except Exception:
            if os.path.exists(tmp_content):
                try:
                    os.unlink(tmp_content)
                except OSError:
                    pass
            raise

        # Atomic write metadata
        tmp_fd, tmp_meta = tempfile.mkstemp(
            dir=str(self._store_dir), prefix=".artifact_meta_", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)
            os.replace(tmp_meta, str(meta_path))
        except Exception:
            if os.path.exists(tmp_meta):
                try:
                    os.unlink(tmp_meta)
                except OSError:
                    pass
            raise

        logger.debug("Persisted context artifact %s (%d chars)", artifact_id, len(content))
        return meta

    def get_metadata(self, artifact_id: str) -> ContextArtifactMetadata | None:
        """Retrieve metadata for an artifact if it exists."""
        clean_id = self._validate_artifact_id(artifact_id)
        try:
            _, meta_path = self._get_paths(clean_id)
            if not meta_path.is_file():
                return None
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ContextArtifactMetadata.from_dict(data)
        except Exception as exc:
            logger.warning("Failed to retrieve metadata for %s: %s", artifact_id, exc)
            return None

    def read(self, artifact_id: str) -> str | None:
        """Read full artifact content if it exists."""
        clean_id = self._validate_artifact_id(artifact_id)
        try:
            content_path, _ = self._get_paths(clean_id)
            if not content_path.is_file():
                return None
            with open(content_path, "r", encoding="utf-8", newline="") as f:
                return f.read()
        except Exception as exc:
            logger.warning("Failed to read artifact %s: %s", artifact_id, exc)
            return None

    def read_range(
        self,
        artifact_id: str,
        offset: int = 0,
        limit: int = 4000,
        max_allowed: int = DEFAULT_MAX_RECOVERY_CHARS,
    ) -> tuple[str, ContextArtifactMetadata] | tuple[None, None]:
        """Read a bounded slice of artifact content to prevent context blowup."""
        clean_id = self._validate_artifact_id(artifact_id)
        try:
            content_path, meta_path = self._get_paths(clean_id)
            if not content_path.is_file() or not meta_path.is_file():
                return None, None

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = ContextArtifactMetadata.from_dict(json.load(f))

            offset = max(0, int(offset))
            # Strict upper bound on recovery slice
            bounded_limit = max(1, min(int(limit), max_allowed))

            with open(content_path, "r", encoding="utf-8", newline="") as f:
                f.seek(offset)
                content = f.read(bounded_limit)

            return content, meta
        except Exception as exc:
            logger.warning("Failed to read range for artifact %s: %s", artifact_id, exc)
            return None, None

    def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists in the store."""
        try:
            clean_id = self._validate_artifact_id(artifact_id)
            content_path, _ = self._get_paths(clean_id)
            return content_path.is_file()
        except Exception:
            return False

    def cleanup(
        self,
        retention_days: int = 7,
        max_artifacts: int = 500,
        active_artifact_ids: set[str] | None = None,
    ) -> int:
        """Clean old artifacts without deleting currently referenced active artifacts."""
        if not self._store_dir.exists():
            return 0

        active_set = active_artifact_ids or set()
        cutoff = time.time() - (retention_days * 86400)
        deleted = 0

        files = list(self._store_dir.glob("ctx_*.txt"))
        # Sort oldest first
        files.sort(key=lambda p: p.stat().st_mtime)

        for f in files:
            clean_id = f.stem
            if clean_id in active_set:
                continue

            mtime = f.stat().st_mtime
            should_delete = (mtime < cutoff) or (len(files) - deleted > max_artifacts)
            if should_delete:
                try:
                    f.unlink(missing_ok=True)
                    meta_path = self._store_dir / f"{clean_id}.meta.json"
                    meta_path.unlink(missing_ok=True)
                    deleted += 1
                except OSError as exc:
                    logger.warning("Failed to delete old artifact %s: %s", clean_id, exc)

        return deleted
