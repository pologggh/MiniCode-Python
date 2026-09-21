from __future__ import annotations

from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from minicode.permissions import PermissionManager
from minicode.subagent_runner import SubAgentResult, SubAgentToolEvent
from minicode.task_graph import WorktreeIsolator
from minicode.team_planner import TeamPlanner
from minicode.team_scheduler import TeamScheduler
from minicode.tooling import ToolContext


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary real git repository with an initial commit."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True, capture_output=True)
    
    init_file = repo_dir / "main.py"
    init_file.write_text("print('hello world')\n", encoding="utf-8")
    subprocess.run(["git", "add", "main.py"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    return repo_dir


def test_worktree_is_git_repository(tmp_path, git_repo):
    """WorktreeIsolator detects git vs non-git directories correctly."""
    repo_isolator = WorktreeIsolator(base_path=git_repo)
    assert repo_isolator.is_git_repository()

    empty_isolator = WorktreeIsolator(base_path=tmp_path / "not_a_repo")
    assert not empty_isolator.is_git_repository()


def test_worktree_parent_dirty_refuses_creation(git_repo):
    """WorktreeIsolator fail-closed: dirty parent repository refuses worktree execution."""
    isolator = WorktreeIsolator(base_path=git_repo)

    # Initially clean
    clean, msg = isolator.check_parent_workspace_clean()
    assert clean
    assert "Workspace clean" in msg

    # Make parent dirty (untracked file)
    untracked = git_repo / "untracked.txt"
    untracked.write_text("dirty content\n", encoding="utf-8")

    clean_dirty, msg_dirty = isolator.check_parent_workspace_clean()
    assert not clean_dirty
    assert "Parent workspace is dirty" in msg_dirty


def test_worktree_parent_fingerprint_mismatch(git_repo):
    """WorktreeIsolator detects workspace modifications during agent execution."""
    isolator = WorktreeIsolator(base_path=git_repo)
    clean, _ = isolator.check_parent_workspace_clean()
    assert clean

    # Initial check passes
    fp_ok, _ = isolator.verify_parent_fingerprint()
    assert fp_ok

    # Modify parent externally
    modified_file = git_repo / "main.py"
    modified_file.write_text("print('corrupted')\n", encoding="utf-8")

    fp_fail, fp_msg = isolator.verify_parent_fingerprint()
    assert not fp_fail
    assert "Parent workspace changed" in fp_msg


def test_worktree_detached_lifecycle_and_patch_writeback(git_repo):
    """Test full detached worktree lifecycle: create detached, patch, verify, apply, cleanup."""
    isolator = WorktreeIsolator(base_path=git_repo)
    clean, _ = isolator.check_parent_workspace_clean()
    assert clean

    wt_path = isolator.create_worktree("test_task")
    assert wt_path is not None
    assert wt_path.exists()
    assert wt_path in isolator.active_worktrees

    try:
        # Check that HEAD is detached in worktree
        res = subprocess.run(
            ["git", "status"],
            cwd=str(wt_path),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "HEAD detached" in res.stdout or "Not currently on any branch" in res.stdout

        # Modify existing file and create new file in worktree
        (wt_path / "main.py").write_text("print('enhanced world')\n", encoding="utf-8")
        (wt_path / "new_feature.py").write_text("def feature(): pass\n", encoding="utf-8")

        # Generate patch
        patch_text = isolator.generate_patch(wt_path)
        assert "main.py" in patch_text
        assert "new_feature.py" in patch_text

        # Verify patch
        assert isolator.verify_patch(patch_text)

        # Apply patch to parent
        ok, status = isolator.apply_patch(patch_text)
        assert ok
        assert status == "applied"

        # Verify parent got the changes
        assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('enhanced world')\n"
        assert (git_repo / "new_feature.py").read_text(encoding="utf-8") == "def feature(): pass\n"

    finally:
        isolator.cleanup_all()
        assert not wt_path.exists()
        assert len(isolator.active_worktrees) == 0


def test_worktree_permission_gated_apply(git_repo):
    """Apply patch must be gated by parent permissions and fail closed on denial."""
    isolator = WorktreeIsolator(base_path=git_repo)
    clean, _ = isolator.check_parent_workspace_clean()
    assert clean

    wt_path = isolator.create_worktree("perm_task")
    assert wt_path is not None

    try:
        (wt_path / "main.py").write_text("print('forbidden')\n", encoding="utf-8")
        patch_text = isolator.generate_patch(wt_path)
        assert isolator.verify_patch(patch_text)

        # Mock permission manager that denies
        denying_perms = MagicMock(spec=PermissionManager)
        denying_perms.ensure_edit.side_effect = RuntimeError("Permission denied: writing to main.py")

        ok, status = isolator.apply_patch(patch_text, permissions=denying_perms)
        assert not ok
        assert "permission_denied" in status

        # Main repo must NOT have been changed
        assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('hello world')\n"

    finally:
        isolator.cleanup_all()


def test_team_scheduler_worktree_fail_closed_on_dirty_parent(git_repo):
    """TeamScheduler with use_worktree=True fails immediately if parent workspace is dirty."""
    (git_repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    planner = TeamPlanner()
    plan = planner.plan(goal="Dirty test")
    scheduler = TeamScheduler(max_workers=2)
    context = ToolContext(cwd=str(git_repo))

    result = scheduler.schedule_and_run(plan, context, use_worktree=True)
    assert not result.success
    assert result.status == "parent_workspace_dirty"
    assert "parent_workspace_dirty" in result.summary


def test_team_scheduler_worktree_success_flow(git_repo):
    """TeamScheduler with use_worktree=True executes in isolated worktree and writes back cleanly."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Feature in worktree")
    scheduler = TeamScheduler(max_workers=2)
    context = ToolContext(cwd=str(git_repo))

    def mock_run_subagent(config):
        if "research" in config.name:
            return SubAgentResult(ok=True, output="Research complete")
        if "coding" in config.name:
            # Coding agent modifies main.py inside worktree cwd
            wt_cwd = Path(config.cwd)
            (wt_cwd / "main.py").write_text("print('hello from worktree')\n", encoding="utf-8")
            return SubAgentResult(ok=True, output="Code updated", changed_files=["main.py"])
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="All tests pass")
            return SubAgentResult(ok=True, output="10 passed", tool_events=[ev])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "approve", "comments": "Good worktree commit", "issues": []}',
                structured_data={"verdict": "approve", "comments": "Good worktree commit", "issues": []},
            )
        return SubAgentResult(ok=True, output="Done")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_run_subagent):
        result = scheduler.schedule_and_run(plan, context, use_worktree=True)

    assert result.success
    assert result.status == "completed"
    assert "Patch applied: True" in result.summary
    # Verify changes were written back to parent repository!
    assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('hello from worktree')\n"
