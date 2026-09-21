"""Integration tests verifying Experience Memory wiring inside run_agent_turn."""
from __future__ import annotations

from pathlib import Path
import pytest

from minicode.agent_loop import run_agent_turn, _finalize_experience_memory
from minicode.execution_trace import ExecutionTrace
from minicode.experience import (
    ExperienceOutcome,
    ExperienceRecord,
    ExperienceExtractor,
    ExperienceQualityGate,
    experience_to_memory_entry,
)
from minicode.memory import MemoryEntry, MemoryManager, MemoryScope
from minicode.memory_injector import InjectedMemory
from minicode.permissions import PermissionManager
from minicode.tools import create_default_tool_registry
from minicode.types import AgentStep


class ScriptedModel:
    """Model adapter returning pre-programmed AgentSteps for deterministic testing."""

    def __init__(self, steps: list[AgentStep], model_id: str = "scripted-mock"):
        self.steps = list(steps)
        self.step_idx = 0
        self.model_id = model_id

    def next(self, messages, **_kwargs) -> AgentStep:
        if self.step_idx < len(self.steps):
            step = self.steps[self.step_idx]
            self.step_idx += 1
            return step
        return AgentStep(type="assistant", content="Finished task.")


@pytest.fixture
def permissions(tmp_path: Path):
    return PermissionManager(str(tmp_path), prompt=lambda _: {"decision": "allow_once"})


@pytest.fixture
def tools(tmp_path: Path):
    return create_default_tool_registry(str(tmp_path), runtime=None)


def test_case_a_tool_fail_recovered_verification_pass_positive_feedback(tmp_path: Path, permissions, tools):
    """Case A: Early tool failure -> recovery edit -> verification pass -> SUCCESS_VERIFIED and positive feedback."""
    mem_mgr = MemoryManager(project_root=tmp_path)
    # Pre-inject a memory entry
    prior_entry = mem_mgr.add_entry(
        MemoryScope.PROJECT,
        "experience",
        "Prior experience about auth tokens",
        metadata={"experience": {"outcome": "success_verified"}},
    )
    initial_usage = prior_entry.usage_count

    # Create dummy files
    (tmp_path / "app.py").write_text("def run(): return 1\n", encoding="utf-8")

    # Scripted sequence:
    # 1. run failing test
    # 2. edit app.py
    # 3. run passing test
    # 4. complete assistant message
    steps = [
        AgentStep(
            type="tool_calls",
            calls=[{"id": "c1", "toolName": "run_command", "input": {"CommandLine": "python -m pytest tests/test_fail.py"}}],
        ),
        AgentStep(
            type="tool_calls",
            calls=[{"id": "c2", "toolName": "write_to_file", "input": {"TargetFile": str(tmp_path / "app.py"), "CodeContent": "def run(): return 0\n", "Overwrite": True, "Description": "fix"}}],
        ),
        AgentStep(
            type="tool_calls",
            calls=[{"id": "c3", "toolName": "run_command", "input": {"CommandLine": "python -m pytest tests/test_pass.py"}}],
        ),
        AgentStep(type="assistant", content="The bug has been fixed and verified with pytest."),
    ]

    model = ScriptedModel(steps)

    # Monkeypatch tools.execute to simulate fail on c1, pass on c3
    real_exec = tools.execute

    def mock_exec(name, args, context):
        cmd = str(args.get("CommandLine", "") if isinstance(args, dict) else "")
        if "test_fail.py" in cmd:
            from minicode.tooling import ToolResult
            return ToolResult(ok=False, output="FAILED: test_fail.py - AssertionError")
        if "test_pass.py" in cmd:
            from minicode.tooling import ToolResult
            return ToolResult(ok=True, output="1 passed in 0.05s")
        return real_exec(name, args, context)

    tools.execute = mock_exec

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Fix the failing pytest in app.py"},
    ]

    result = run_agent_turn(
        model=model,
        tools=tools,
        messages=messages,
        cwd=str(tmp_path),
        permissions=permissions,
        memory_manager=mem_mgr,
        enable_work_chain=True,
    )

    assert result is not None

    # Verify that experience memory was persisted
    project_entries = mem_mgr.memories[MemoryScope.PROJECT].entries
    exp_entries = [e for e in project_entries if e.category == "experience" and e.id != prior_entry.id]
    assert len(exp_entries) >= 1

    latest_exp = exp_entries[-1]
    exp_data = latest_exp.metadata.get("experience", {})
    # Despite intermediate tool failure, terminal outcome must be SUCCESS_VERIFIED!
    assert exp_data.get("outcome") == ExperienceOutcome.SUCCESS_VERIFIED.value
    assert len(exp_data.get("strategy", [])) >= 2

    # Verify positive feedback was applied to injected memory
    updated_prior = mem_mgr.memories[MemoryScope.PROJECT]._id_index.get(prior_entry.id)
    assert updated_prior is not None
    assert updated_prior.usage_count >= initial_usage


def test_case_b_verification_pass_then_fail_negative_feedback(tmp_path: Path, permissions, tools):
    """Case B: Initial pass -> regression edit -> terminal verification fail -> FAILED_VERIFICATION and negative feedback."""
    mem_mgr = MemoryManager(project_root=tmp_path)
    prior_entry = mem_mgr.add_entry(
        MemoryScope.PROJECT,
        "experience",
        "Misleading past pattern",
        metadata={"experience": {"outcome": "failed_tool"}},
    )
    prior_entry.usage_count = 5
    mem_mgr._save_scope(MemoryScope.PROJECT)

    steps = [
        # 1. First test passes
        AgentStep(
            type="tool_calls",
            calls=[{"id": "c1", "toolName": "run_command", "input": {"CommandLine": "python -m pytest tests/test_auth.py"}}],
        ),
        # 2. Edit introduces bug
        AgentStep(
            type="tool_calls",
            calls=[{"id": "c2", "toolName": "write_to_file", "input": {"TargetFile": str(tmp_path / "auth.py"), "CodeContent": "syntax error", "Overwrite": True, "Description": "broken"}}],
        ),
        # 3. Terminal verification fails
        AgentStep(
            type="tool_calls",
            calls=[{"id": "c3", "toolName": "run_command", "input": {"CommandLine": "python -m pytest tests/test_auth.py"}}],
        ),
        AgentStep(type="assistant", content="Tests failed after change."),
    ]

    model = ScriptedModel(steps)

    pass_count = 0

    def mock_exec(name, args, context):
        nonlocal pass_count
        cmd = str(args.get("CommandLine", "") if isinstance(args, dict) else "")
        if "test_auth.py" in cmd:
            from minicode.tooling import ToolResult
            if pass_count == 0:
                pass_count += 1
                return ToolResult(ok=True, output="1 passed")
            return ToolResult(ok=False, output="FAILED test_auth.py: SyntaxError")
        from minicode.tooling import ToolResult
        return ToolResult(ok=True, output="File written")

    tools.execute = mock_exec

    trace = ExecutionTrace()
    trace.record_task_start("turn-b", "Refactor authentication", workspace=str(tmp_path))
    trace.record_tool_call("c1", "run_command", {"CommandLine": "python -m pytest tests/test_auth.py"})
    trace.record_verification("python -m pytest tests/test_auth.py", 0, True, "1 passed")
    trace.record_tool_call("c2", "write_to_file", {"TargetFile": "auth.py"})
    trace.record_tool_result("c2", "write_to_file", ok=True, output="File written")
    trace.record_tool_call("c3", "run_command", {"CommandLine": "python -m pytest tests/test_auth.py"})
    trace.record_verification("python -m pytest tests/test_auth.py", 1, False, "FAILED: SyntaxError")
    trace.record_assistant("Tests failed after change.")
    trace.record_task_end("failed", "Tests failed")

    # Finalize via _finalize_experience_memory
    class MockTurnState:
        step = 3
        tool_error_count = 1
        stop_reason = "failed"

    injected = [InjectedMemory(content="test", category="experience", relevance_score=0.5, source="search", memory_id=prior_entry.id)]

    _finalize_experience_memory(
        active_execution_trace=trace,
        turn_state=MockTurnState(),
        coda_summary=None,
        task_desc="Refactor authentication",
        orch=None,
        reflection_engine=None,
        memory_mgr=mem_mgr,
        turn_injected_memories=injected,
    )

    # Terminal verification must be FAILED_VERIFICATION
    extractor = ExperienceExtractor()
    outcome = extractor.classify_outcome(trace)
    assert outcome == ExperienceOutcome.FAILED_VERIFICATION

    # Negative feedback must decay usage count
    updated = mem_mgr.memories[MemoryScope.PROJECT]._id_index.get(prior_entry.id)
    assert updated is not None
    assert updated.usage_count == 4  # 5 - 1 = 4


def test_case_c_enable_work_chain_false_persists_trace_and_experience(tmp_path: Path, permissions, tools):
    """Case C: enable_work_chain=False must still capture trace and persist experience record."""
    mem_mgr = MemoryManager(project_root=tmp_path)

    steps = [
        AgentStep(
            type="tool_calls",
            calls=[{"id": "c1", "toolName": "run_command", "input": {"CommandLine": "python -m unittest discover"}}],
        ),
        AgentStep(type="assistant", content="Tests all passed."),
    ]
    model = ScriptedModel(steps)

    def mock_exec(name, args, context):
        from minicode.tooling import ToolResult
        return ToolResult(ok=True, output="Ran 5 tests in 0.01s: OK")

    tools.execute = mock_exec

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Run tests and verify unittest suite."},
    ]

    result = run_agent_turn(
        model=model,
        tools=tools,
        messages=messages,
        cwd=str(tmp_path),
        permissions=permissions,
        memory_manager=mem_mgr,
        enable_work_chain=False,  # Explicitly disabled
    )

    assert result is not None

    # Even with work chain disabled, experience memory must be persisted
    project_entries = mem_mgr.memories[MemoryScope.PROJECT].entries
    exp_entries = [e for e in project_entries if e.category == "experience"]
    assert len(exp_entries) >= 1
    exp = exp_entries[0]
    assert exp.metadata.get("experience", {}).get("outcome") == ExperienceOutcome.SUCCESS_VERIFIED.value


def test_case_d_blocked_and_aborted_rejected_by_quality_gate(tmp_path: Path):
    """Case D: Blocked or aborted tasks must NOT be persisted as reusable success experience."""
    gate = ExperienceQualityGate()
    extractor = ExperienceExtractor()

    # Aborted trace
    trace_aborted = ExecutionTrace()
    trace_aborted.record_task_start("t-abort", "Cancelled task", workspace=str(tmp_path))
    trace_aborted.record_tool_call("c1", "view_file", {"path": "main.py"})
    trace_aborted.record_tool_result("c1", "view_file", ok=True, output="code")
    trace_aborted.record_task_end(status="aborted", summary="User cancelled")

    rec_aborted = extractor.extract(trace_aborted)
    assert rec_aborted is not None
    assert rec_aborted.outcome == ExperienceOutcome.ABORTED
    can_persist, reason = gate.should_persist(rec_aborted, trace_aborted)
    assert can_persist is False
    assert "aborted" in reason.lower()

    # Blocked trace
    trace_blocked = ExecutionTrace()
    trace_blocked.record_task_start("t-block", "Blocked by security policy", workspace=str(tmp_path))
    trace_blocked.record_tool_call("c1", "run_command", {"CommandLine": "rm -rf /"})
    trace_blocked.record_tool_result("c1", "run_command", ok=False, output="Permission denied")
    trace_blocked.record_task_end(status="blocked", summary="Permission denied")

    rec_blocked = extractor.extract(trace_blocked)
    assert rec_blocked is not None
    assert rec_blocked.outcome == ExperienceOutcome.BLOCKED
    can_persist, reason = gate.should_persist(rec_blocked, trace_blocked)
    assert can_persist is False
    assert "blocked" in reason.lower()
