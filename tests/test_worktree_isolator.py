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
from minicode.tooling import ToolContext, ToolResult
from minicode.tools.write_file import write_file_tool


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

        # Verify patch dry-run
        assert isolator.verify_patch(patch_text)

        # Apply patch to parent using real parent PermissionManager with allow handler
        parent_perms = PermissionManager(
            workspace_root=str(git_repo),
            prompt=lambda req: {"decision": "allow_once"},
        )
        ok, status = isolator.apply_patch(patch_text, permissions=parent_perms)
        assert ok
        assert status == "applied"

        # Verify parent got the changes
        assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('enhanced world')\n"
        assert (git_repo / "new_feature.py").read_text(encoding="utf-8") == "def feature(): pass\n"

    finally:
        isolator.cleanup_all()
        assert not wt_path.exists()
        assert len(isolator.active_worktrees) == 0


def test_worktree_permission_gated_apply_denial(git_repo):
    """Apply patch must be gated by PermissionManager.ensure_edit and fail closed on denial."""
    isolator = WorktreeIsolator(base_path=git_repo)
    clean, _ = isolator.check_parent_workspace_clean()
    assert clean

    wt_path = isolator.create_worktree("perm_task")
    assert wt_path is not None

    try:
        (wt_path / "main.py").write_text("print('forbidden')\n", encoding="utf-8")
        patch_text = isolator.generate_patch(wt_path)
        assert isolator.verify_patch(patch_text)

        # Real PermissionManager with prompt handler denying edits
        denying_perms = PermissionManager(
            workspace_root=str(git_repo),
            prompt=lambda req: {"decision": "deny_once"},
        )

        ok, status = isolator.apply_patch(patch_text, permissions=denying_perms)
        assert not ok
        assert status == "permission_denied"

        # Main repo must NOT have been changed
        assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('hello world')\n"

    finally:
        isolator.cleanup_all()


def test_worktree_missing_permission_manager_fail_closed(git_repo):
    """When permissions=None and patch is non-empty, apply_patch must fail closed."""
    isolator = WorktreeIsolator(base_path=git_repo)
    clean, _ = isolator.check_parent_workspace_clean()
    assert clean

    wt_path = isolator.create_worktree("missing_perm_task")
    assert wt_path is not None

    try:
        (wt_path / "main.py").write_text("print('unauthorized')\n", encoding="utf-8")
        patch_text = isolator.generate_patch(wt_path)
        assert bool(patch_text.strip())

        # Calling apply_patch with permissions=None must fail closed
        ok, status = isolator.apply_patch(patch_text, permissions=None)
        assert not ok
        assert status == "permission_manager_missing"

        # Main repo must NOT have been changed
        assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('hello world')\n"

    finally:
        isolator.cleanup_all()


def test_worktree_scoped_writer_permission_manager(git_repo):
    """Worktree Scoped PermissionManager allows edits inside worktree but denies outside paths."""
    isolator = WorktreeIsolator(base_path=git_repo)
    wt_path = isolator.create_worktree("tool_test")
    assert wt_path is not None

    try:
        scoped_perms = isolator.create_isolated_permission_manager(str(wt_path))
        wt_context = ToolContext(cwd=str(wt_path), permissions=scoped_perms)

        # 1. Real write_file_tool inside worktree succeeds autonomously
        res_inside = write_file_tool.run(
            {"path": "added_by_tool.py", "content": "print('tool added')\n"},
            wt_context,
        )
        assert isinstance(res_inside, ToolResult)
        assert res_inside.ok
        assert (wt_path / "added_by_tool.py").exists()
        assert not (git_repo / "added_by_tool.py").exists()

        # 2. Writing outside worktree (e.g. attempting to modify parent or root) is strictly denied
        outside_path = str(git_repo / "hacked.py")
        with pytest.raises(Exception, match="outside cwd|denied"):
            scoped_perms.ensure_path_access(outside_path, "write")

        with pytest.raises(Exception, match="denied"):
            scoped_perms.ensure_edit(outside_path, diff_preview="bad diff")

    finally:
        isolator.cleanup_all()


def test_team_scheduler_worktree_fail_closed_on_dirty_parent(git_repo):
    """TeamScheduler with use_worktree=True fails immediately if parent workspace is dirty."""
    (git_repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    planner = TeamPlanner()
    plan = planner.plan(goal="Dirty test")
    scheduler = TeamScheduler(max_workers=2)
    parent_perms = PermissionManager(workspace_root=str(git_repo), prompt=None)
    context = ToolContext(cwd=str(git_repo), permissions=parent_perms)

    result = scheduler.schedule_and_run(plan, context, use_worktree=True)
    assert not result.success
    assert result.status == "parent_workspace_dirty"
    assert "parent_workspace_dirty" in result.summary


def test_team_scheduler_worktree_success_flow(git_repo):
    """TeamScheduler with use_worktree=True executes in isolated worktree, verifies gates, and applies patch."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Feature in worktree")
    scheduler = TeamScheduler(max_workers=2)

    parent_perms = PermissionManager(
        workspace_root=str(git_repo),
        prompt=lambda req: {"decision": "allow_once"},
    )
    context = ToolContext(cwd=str(git_repo), permissions=parent_perms)

    # Record parent git HEAD sha before run
    head_sha_before = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(git_repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

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

    # Verify no commit was made (HEAD sha remains identical, working tree has uncommitted change)
    head_sha_after = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(git_repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head_sha_before == head_sha_after

    status_out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(git_repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "M main.py" in status_out or " M main.py" in status_out


def test_team_scheduler_worktree_permission_denied_flow(git_repo):
    """When parent PermissionManager denies patch, writeback fails closed and parent file is untouched."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Feature denied by user")
    scheduler = TeamScheduler(max_workers=2)

    # Real PermissionManager denying edits
    parent_perms = PermissionManager(
        workspace_root=str(git_repo),
        prompt=lambda req: {"decision": "deny_once"},
    )
    context = ToolContext(cwd=str(git_repo), permissions=parent_perms)

    def mock_run_subagent(config):
        if "research" in config.name:
            return SubAgentResult(ok=True, output="ok")
        if "coding" in config.name:
            wt_cwd = Path(config.cwd)
            (wt_cwd / "main.py").write_text("print('unauthorized change')\n", encoding="utf-8")
            return SubAgentResult(ok=True, output="Code updated", changed_files=["main.py"])
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="All tests pass")
            return SubAgentResult(ok=True, output="pass", tool_events=[ev])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "approve", "comments": "Looks ok"}',
                structured_data={"verdict": "approve", "comments": "Looks ok"},
            )
        return SubAgentResult(ok=True, output="Done")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_run_subagent):
        result = scheduler.schedule_and_run(plan, context, use_worktree=True)

    assert not result.success
    assert result.status == "permission_denied"

    # Main repo must remain pristine!
    assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('hello world')\n"
    status_out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(git_repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert status_out == ""


def test_team_scheduler_worktree_missing_permission_flow(git_repo):
    """When context.permissions is None, worktree writeback fails closed with permission_manager_missing."""
    planner = TeamPlanner()
    plan = planner.plan(goal="Feature without permissions")
    scheduler = TeamScheduler(max_workers=2)
    context = ToolContext(cwd=str(git_repo), permissions=None)

    def mock_run_subagent(config):
        if "research" in config.name:
            return SubAgentResult(ok=True, output="ok")
        if "coding" in config.name:
            wt_cwd = Path(config.cwd)
            (wt_cwd / "main.py").write_text("print('rogue change')\n", encoding="utf-8")
            return SubAgentResult(ok=True, output="Code updated", changed_files=["main.py"])
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="All tests pass")
            return SubAgentResult(ok=True, output="pass", tool_events=[ev])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "approve", "comments": "Looks ok"}',
                structured_data={"verdict": "approve", "comments": "Looks ok"},
            )
        return SubAgentResult(ok=True, output="Done")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_run_subagent):
        result = scheduler.schedule_and_run(plan, context, use_worktree=True)

    assert not result.success
    assert result.status == "permission_manager_missing"

    # Parent repo must remain untouched!
    assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('hello world')\n"
    status_out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(git_repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert status_out == ""


def test_worktree_failed_patch_leakage_cases(git_repo):
    """Verify that in all 3 failure modes (Test Fail, Review Reject, Perm Deny), parent workspace leaked changes = 0."""
    planner = TeamPlanner()
    scheduler = TeamScheduler(max_workers=2)

    parent_perms = PermissionManager(
        workspace_root=str(git_repo),
        prompt=lambda req: {"decision": "allow_once"},
    )
    context = ToolContext(cwd=str(git_repo), permissions=parent_perms)

    # Scenario A: Test Gate Fails
    def mock_test_fail(config):
        if "research" in config.name:
            return SubAgentResult(ok=True, output="ok")
        if "coding" in config.name:
            wt_cwd = Path(config.cwd)
            (wt_cwd / "main.py").write_text("print('test fail dirty')\n", encoding="utf-8")
            return SubAgentResult(ok=True, output="ok", changed_files=["main.py"])
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=False, output_summary="AssertionError")
            return SubAgentResult(ok=True, output="failed", tool_events=[ev])
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_test_fail):
        res_a = scheduler.schedule_and_run(planner.plan("A"), context, use_worktree=True, max_replans=0)
    assert not res_a.success
    assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('hello world')\n"

    # Scenario B: Reviewer Rejects & replans exhausted
    def mock_review_reject(config):
        if "research" in config.name:
            return SubAgentResult(ok=True, output="ok")
        if "coding" in config.name:
            wt_cwd = Path(config.cwd)
            (wt_cwd / "main.py").write_text("print('review reject dirty')\n", encoding="utf-8")
            return SubAgentResult(ok=True, output="ok", changed_files=["main.py"])
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="pass")
            return SubAgentResult(ok=True, output="pass", tool_events=[ev])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "reject", "comments": "Bad architecture"}',
                structured_data={"verdict": "reject", "comments": "Bad architecture"},
            )
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_review_reject):
        res_b = scheduler.schedule_and_run(planner.plan("B"), context, use_worktree=True, max_replans=0)
    assert not res_b.success
    assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('hello world')\n"

    # Scenario C: Parent Permission Denies
    denying_perms = PermissionManager(
        workspace_root=str(git_repo),
        prompt=lambda req: {"decision": "deny_once"},
    )
    context_deny = ToolContext(cwd=str(git_repo), permissions=denying_perms)

    def mock_perm_deny(config):
        if "research" in config.name:
            return SubAgentResult(ok=True, output="ok")
        if "coding" in config.name:
            wt_cwd = Path(config.cwd)
            (wt_cwd / "main.py").write_text("print('perm deny dirty')\n", encoding="utf-8")
            return SubAgentResult(ok=True, output="ok", changed_files=["main.py"])
        if "test" in config.name:
            ev = SubAgentToolEvent(tool_name="test_runner", ok=True, output_summary="pass")
            return SubAgentResult(ok=True, output="pass", tool_events=[ev])
        if "reviewer" in config.name:
            return SubAgentResult(
                ok=True,
                output='{"verdict": "approve", "comments": "ok"}',
                structured_data={"verdict": "approve", "comments": "ok"},
            )
        return SubAgentResult(ok=True, output="ok")

    with patch("minicode.team_scheduler.run_subagent", side_effect=mock_perm_deny):
        res_c = scheduler.schedule_and_run(planner.plan("C"), context_deny, use_worktree=True)
    assert not res_c.success
    assert (git_repo / "main.py").read_text(encoding="utf-8") == "print('hello world')\n"

    # In all scenarios, git status is completely clean
    status_final = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(git_repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert status_final == ""
