from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from minicode.subagent_runner import SubAgentResult
from minicode.task_graph import WorktreeIsolator
from minicode.team_planner import TeamPlanner
from minicode.team_scheduler import TeamScheduler
from minicode.tooling import ToolContext


def test_worktree_is_git_repository(tmp_path):
    """WorktreeIsolator detects git vs non-git directories correctly."""
    # Current repository root is a git repository
    repo_isolator = WorktreeIsolator(base_path=Path("."))
    assert repo_isolator.is_git_repository()

    # Empty tmp directory is not a git repository
    empty_isolator = WorktreeIsolator(base_path=tmp_path)
    assert not empty_isolator.is_git_repository()


def test_worktree_lifecycle_with_patch_and_cleanup(tmp_path):
    """Test full worktree lifecycle: create, generate patch, verify patch, apply, and cleanup."""
    isolator = WorktreeIsolator(base_path=Path("."))
    assert isolator.is_git_repository()

    wt_path = isolator.create_worktree("test_lifecycle")
    try:
        assert wt_path is not None
        assert wt_path.exists()
        assert wt_path in isolator.active_worktrees

        # Create a new file in the worktree
        new_file = wt_path / "temp_wt_test_artifact.txt"
        new_file.write_text("worktree isolation test content\n", encoding="utf-8")

        # Generate patch
        patch_text = isolator.generate_patch(wt_path)
        assert "temp_wt_test_artifact.txt" in patch_text
        assert "worktree isolation test content" in patch_text

        # Dry-run verify patch
        can_apply = isolator.verify_patch(patch_text)
        assert can_apply
    finally:
        isolator.cleanup_all()
        assert not wt_path.exists()
        assert len(isolator.active_worktrees) == 0


def test_team_scheduler_with_worktree_flag(tmp_path):
    """TeamScheduler with use_worktree=True uses WorktreeIsolator and cleans up."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Isolated refactoring")

    scheduler = TeamScheduler(max_workers=2)
    context = ToolContext(cwd=str(tmp_path))

    mock_wt_path = tmp_path / ".worktrees" / "isolated_task_test"
    mock_wt_path.mkdir(parents=True, exist_ok=True)

    with patch("minicode.task_graph.WorktreeIsolator.is_git_repository", return_value=True), \
         patch("minicode.task_graph.WorktreeIsolator.create_worktree", return_value=mock_wt_path) as mock_create, \
         patch("minicode.task_graph.WorktreeIsolator.generate_patch", return_value="diff --git a/test b/test"), \
         patch("minicode.task_graph.WorktreeIsolator.verify_patch", return_value=True), \
         patch("minicode.task_graph.WorktreeIsolator.apply_patch", return_value=True), \
         patch("minicode.task_graph.WorktreeIsolator.cleanup_all") as mock_cleanup, \
         patch("minicode.team_scheduler.run_subagent") as mock_run_agent:

        mock_run_agent.return_value = SubAgentResult(
            ok=True,
            output='{"verdict": "approve", "comments": "ok", "issues": []}',
            final_message="All tests pass. ok=true",
            structured_data={"verdict": "approve", "comments": "ok", "issues": []},
            tool_calls_count=1,
        )

        res = scheduler.schedule_and_run(plan, context, use_worktree=True)
        assert res.success
        assert "Worktree Isolation" in res.summary
        assert mock_create.called
        assert mock_cleanup.called
