"""Persistent task graph for cross-step workflow management.

Inspired by Learn Claude Code best practices:
- Distinguish between session-local planning and persistent task coordination
- Separate task definition (what) from execution slot (who is running / progress)
- Background task slot management with timed scheduling
- Worktree execution isolation for risky operations

Provides:
- TaskGraph: DAG of tasks with dependencies
- TaskSlot: Named execution slot with state tracking
- WorktreeIsolator: Temporary worktree for risky operations
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from minicode.config import MINI_CODE_DIR


# ---------------------------------------------------------------------------
# Task Graph
# ---------------------------------------------------------------------------

class TaskState(str, Enum):
    """Task execution state."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskDefinition:
    """What needs to be done (persistent, cross-session)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: int = 300
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSlot:
    """Who is running and current progress (session-local)."""

    task_id: str
    slot_name: str = "default"
    state: TaskState = TaskState.PENDING
    progress: float = 0.0  # 0.0 - 1.0
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: str | None = None


@dataclass
class TaskGraph:
    """Persistent task graph with execution slots."""

    name: str = ""
    definitions: dict[str, TaskDefinition] = field(default_factory=dict)
    slots: dict[str, TaskSlot] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # --- Definition API ---
    def add_task(
        self,
        name: str,
        description: str = "",
        dependencies: list[str] | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: int = 300,
    ) -> TaskDefinition:
        """Add a task definition to the graph."""
        task_def = TaskDefinition(
            name=name,
            description=description,
            dependencies=dependencies or [],
            priority=priority,
            timeout_seconds=timeout_seconds,
        )
        self.definitions[task_def.id] = task_def
        self.updated_at = time.time()
        return task_def

    # --- Slot API ---
    def assign_slot(self, task_id: str, slot_name: str = "default") -> TaskSlot:
        """Assign a task to an execution slot."""
        if task_id not in self.definitions:
            raise ValueError(f"Task {task_id} not found")

        slot = TaskSlot(task_id=task_id, slot_name=slot_name)
        slot_key = f"{slot_name}:{task_id}"
        self.slots[slot_key] = slot
        self.updated_at = time.time()
        return slot

    def start_task(self, slot_key: str) -> TaskSlot:
        """Mark a slot as running."""
        slot = self.slots.get(slot_key)
        if not slot:
            raise ValueError(f"Slot {slot_key} not found")
        slot.state = TaskState.RUNNING
        slot.started_at = time.time()
        slot.progress = 0.0
        self.updated_at = time.time()
        return slot

    def complete_task(self, slot_key: str, result: str = "") -> TaskSlot:
        """Mark a slot as completed."""
        slot = self.slots.get(slot_key)
        if not slot:
            raise ValueError(f"Slot {slot_key} not found")
        slot.state = TaskState.COMPLETED
        slot.completed_at = time.time()
        slot.progress = 1.0
        slot.result = result
        self.updated_at = time.time()
        return slot

    def fail_task(self, slot_key: str, error: str) -> TaskSlot:
        """Mark a slot as failed."""
        slot = self.slots.get(slot_key)
        if not slot:
            raise ValueError(f"Slot {slot_key} not found")
        slot.state = TaskState.FAILED
        slot.completed_at = time.time()
        slot.error = error
        self.updated_at = time.time()
        return slot

    def skip_task(self, slot_key: str, reason: str = "") -> TaskSlot:
        """Mark a slot as skipped (e.g. when upstream dependency failed)."""
        slot = self.slots.get(slot_key)
        if not slot:
            raise ValueError(f"Slot {slot_key} not found")
        slot.state = TaskState.SKIPPED
        slot.completed_at = time.time()
        slot.error = reason
        self.updated_at = time.time()
        return slot

    # --- Graph Logic ---
    def get_ready_tasks(self) -> list[TaskDefinition]:
        """Get tasks whose dependencies are all completed."""
        completed_task_ids = {
            slot.task_id for slot in self.slots.values()
            if slot.state == TaskState.COMPLETED
        }
        non_pending_ids = {
            slot.task_id for slot in self.slots.values()
            if slot.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.SKIPPED, TaskState.RUNNING)
        }

        ready = []
        for task_def in self.definitions.values():
            if task_def.id in non_pending_ids:
                continue
            # Check dependencies
            if all(dep in completed_task_ids for dep in task_def.dependencies):
                ready.append(task_def)

        # Sort by priority
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
        }
        ready.sort(key=lambda t: priority_order.get(t.priority, 2))
        return ready

    def is_graph_complete(self) -> bool:
        """Check if all tasks in the graph are completed."""
        if not self.definitions:
            return True
        completed_ids = {
            slot.task_id for slot in self.slots.values()
            if slot.state == TaskState.COMPLETED
        }
        return all(tid in completed_ids for tid in self.definitions)

    def get_progress_percentage(self) -> float:
        """Overall graph progress."""
        if not self.definitions:
            return 0.0
        completed = sum(
            1 for slot in self.slots.values()
            if slot.state == TaskState.COMPLETED
        )
        return (completed / len(self.definitions)) * 100

    # --- Persistence ---
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "definitions": {
                tid: {
                    "id": td.id,
                    "name": td.name,
                    "description": td.description,
                    "dependencies": td.dependencies,
                    "priority": td.priority.value,
                    "timeout_seconds": td.timeout_seconds,
                    "created_at": td.created_at,
                    "metadata": td.metadata,
                }
                for tid, td in self.definitions.items()
            },
            "slots": {
                sk: {
                    "task_id": s.task_id,
                    "slot_name": s.slot_name,
                    "state": s.state.value,
                    "progress": s.progress,
                    "started_at": s.started_at,
                    "completed_at": s.completed_at,
                    "error": s.error,
                    "result": s.result,
                }
                for sk, s in self.slots.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskGraph:
        """Deserialize from dictionary."""
        graph = cls(
            name=data.get("name", ""),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )
        for tid, td_data in data.get("definitions", {}).items():
            graph.definitions[tid] = TaskDefinition(
                id=td_data["id"],
                name=td_data["name"],
                description=td_data.get("description", ""),
                dependencies=td_data.get("dependencies", []),
                priority=TaskPriority(td_data.get("priority", "normal")),
                timeout_seconds=td_data.get("timeout_seconds", 300),
                created_at=td_data.get("created_at", time.time()),
                metadata=td_data.get("metadata", {}),
            )
        for sk, s_data in data.get("slots", {}).items():
            graph.slots[sk] = TaskSlot(
                task_id=s_data["task_id"],
                slot_name=s_data.get("slot_name", "default"),
                state=TaskState(s_data.get("state", "pending")),
                progress=s_data.get("progress", 0.0),
                started_at=s_data.get("started_at"),
                completed_at=s_data.get("completed_at"),
                error=s_data.get("error"),
                result=s_data.get("result"),
            )
        return graph


# ---------------------------------------------------------------------------
# Worktree Isolator (for risky operations)
# ---------------------------------------------------------------------------

class WorktreeIsolator:
    """Creates temporary git worktrees for risky task execution.

    Provides isolation so that exploratory or destructive operations
    don't affect the main working directory. Includes patch extraction,
    dry-run verification, and gated application.
    """

    def __init__(self, base_path: Path, prefix: str = "isolated_task") -> None:
        self.base_path = Path(base_path).resolve()
        self.prefix = prefix
        self.active_worktrees: list[Path] = []
        self._created_branches: dict[Path, str] = {}

    def is_git_repository(self) -> bool:
        """Check if base_path is within a valid git working tree."""
        import subprocess
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(self.base_path),
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return res.returncode == 0 and res.stdout.strip() == "true"
        except Exception:
            return False

    def create_worktree(self, task_id: str) -> Path | None:
        """Create a new worktree for the given task."""
        import subprocess

        if not self.is_git_repository():
            return None

        clean_id = "".join(c for c in task_id if c.isalnum() or c in ("-", "_"))[:32]
        timestamp = int(time.time() * 1000) % 1000000
        branch_name = f"{self.prefix}_{clean_id}_{timestamp}"
        worktree_path = self.base_path / ".worktrees" / branch_name

        try:
            worktree_path.parent.mkdir(parents=True, exist_ok=True)
            res = subprocess.run(
                ["git", "worktree", "add", "-b", branch_name, str(worktree_path)],
                cwd=str(self.base_path),
                capture_output=True,
                text=True,
                timeout=15.0,
            )
            if res.returncode != 0:
                return None

            self.active_worktrees.append(worktree_path)
            self._created_branches[worktree_path] = branch_name
            return worktree_path
        except Exception:
            return None

    def generate_patch(self, worktree_path: Path) -> str:
        """Generate git unified diff of changes made in the worktree."""
        import subprocess
        if not worktree_path.exists():
            return ""

        try:
            subprocess.run(
                ["git", "add", "-N", "."],
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            return res.stdout if res.returncode == 0 else ""
        except Exception:
            return ""

    def verify_patch(self, patch_content: str) -> bool:
        """Dry-run check if patch applies cleanly to base_path without conflicts."""
        import subprocess
        if not patch_content or not patch_content.strip():
            return True

        try:
            res = subprocess.run(
                ["git", "apply", "--check"],
                cwd=str(self.base_path),
                input=patch_content,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            return res.returncode == 0
        except Exception:
            return False

    def apply_patch(self, patch_content: str, permissions: Any | None = None) -> bool:
        """Apply patch to base_path, gated by permissions if present."""
        import subprocess
        if not patch_content or not patch_content.strip():
            return True

        if permissions and hasattr(permissions, "prompt") and permissions.prompt:
            try:
                decision = permissions.prompt({
                    "action": "apply_worktree_patch",
                    "patch_length": len(patch_content),
                    "base_path": str(self.base_path),
                })
                if isinstance(decision, dict) and decision.get("action") == "deny":
                    return False
            except Exception:
                pass

        try:
            res = subprocess.run(
                ["git", "apply"],
                cwd=str(self.base_path),
                input=patch_content,
                capture_output=True,
                text=True,
                timeout=15.0,
            )
            return res.returncode == 0
        except Exception:
            return False

    def cleanup_worktree(self, worktree_path: Path) -> None:
        """Remove a worktree and its associated branch."""
        import subprocess
        import shutil

        branch_name = self._created_branches.get(worktree_path)

        try:
            subprocess.run(
                ["git", "worktree", "remove", "-f", str(worktree_path)],
                cwd=str(self.base_path),
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except Exception:
            pass

        if branch_name:
            try:
                subprocess.run(
                    ["git", "branch", "-D", branch_name],
                    cwd=str(self.base_path),
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
            except Exception:
                pass
            self._created_branches.pop(worktree_path, None)

        if worktree_path.exists():
            try:
                shutil.rmtree(worktree_path, ignore_errors=True)
            except Exception:
                pass

        if worktree_path in self.active_worktrees:
            self.active_worktrees.remove(worktree_path)

    def cleanup_all(self) -> None:
        """Remove all active worktrees."""
        for wt in list(self.active_worktrees):
            self.cleanup_worktree(wt)



# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_TASK_GRAPH_DIR = MINI_CODE_DIR / "task_graphs"


def save_task_graph(graph: TaskGraph, graph_id: str) -> Path:
    """Save task graph to disk."""
    _TASK_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    graph_file = _TASK_GRAPH_DIR / f"{graph_id}.json"
    graph_file.write_text(
        json.dumps(graph.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return graph_file


def load_task_graph(graph_id: str) -> TaskGraph | None:
    """Load task graph from disk."""
    graph_file = _TASK_GRAPH_DIR / f"{graph_id}.json"
    if not graph_file.exists():
        return None
    try:
        data = json.loads(graph_file.read_text(encoding="utf-8"))
        return TaskGraph.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


def list_task_graphs() -> list[str]:
    """List all saved task graph IDs."""
    if not _TASK_GRAPH_DIR.exists():
        return []
    return [f.stem for f in _TASK_GRAPH_DIR.glob("*.json")]


def delete_task_graph(graph_id: str) -> bool:
    """Delete a saved task graph."""
    graph_file = _TASK_GRAPH_DIR / f"{graph_id}.json"
    if graph_file.exists():
        graph_file.unlink()
        return True
    return False
