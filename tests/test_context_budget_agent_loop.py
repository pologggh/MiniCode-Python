"""Integration tests for ContextBudgetManager wiring in agent_loop.py."""
from __future__ import annotations

from pathlib import Path
import pytest

from minicode.agent_loop import run_agent_turn
from minicode.context_artifacts import ContextArtifactStore
from minicode.permissions import PermissionManager
from minicode.tools import create_default_tool_registry
from minicode.types import AgentStep


class RecordingModel:
    """Mock model that records all message batches passed to each turn step."""

    def __init__(self, steps: list[AgentStep], model_id: str = "scripted-test"):
        self.steps = list(steps)
        self.step_idx = 0
        self.model_id = model_id
        self.recorded_messages: list[list[dict]] = []

    def next(self, messages, **_kwargs) -> AgentStep:
        # Save shallow copy of messages dicts
        self.recorded_messages.append([dict(m) for m in messages])
        if self.step_idx < len(self.steps):
            step = self.steps[self.step_idx]
            self.step_idx += 1
            return step
        return AgentStep(type="assistant", content="Done.")


@pytest.fixture
def permissions(tmp_path: Path):
    return PermissionManager(str(tmp_path), prompt=lambda _: {"decision": "allow_once"})


@pytest.fixture
def tools(tmp_path: Path):
    return create_default_tool_registry(str(tmp_path), runtime=None)


def test_agent_loop_offloads_huge_tool_result_and_recovers(tmp_path: Path, permissions, tools):
    """Verify agent loop offloads large tool results before model request and recovers via tool."""
    huge_log = "Error log line from large test execution run\n" * 300  # ~13,500 chars
    (tmp_path / "huge.log").write_text(huge_log, encoding="utf-8")

    # Step 0: Model calls read_file to read huge.log
    # Step 1: Model inspects offloaded result and calls load_context_artifact
    # Step 2: Model finishes
    steps = [
        AgentStep(
            type="tool_calls",
            calls=[{
                "id": "c1",
                "toolName": "read_file",
                "input": {"path": str(tmp_path / "huge.log")},
            }],
        ),
        AgentStep(
            type="assistant",
            content="I saw the offloaded artifact and will now load full context.",
        ),
    ]

    model = RecordingModel(steps=steps)

    initial_messages = [
        {"role": "user", "content": "Analyze the log file huge.log"},
    ]

    final_messages = run_agent_turn(
        messages=initial_messages,
        system_prompt="You are an assistant.",
        model=model,
        tools=tools,
        cwd=str(tmp_path),
        permissions=permissions,
        max_steps=5,
        enable_work_chain=True,
    )

    # Model was called at least twice (initial, and after tool result)
    assert len(model.recorded_messages) >= 2

    # In step 1 (after read_file tool result), check messages passed to model
    second_call_messages = model.recorded_messages[1]

    # Find the tool result message
    tool_results = [m for m in second_call_messages if m.get("role") in ("tool", "tool_result")]
    assert len(tool_results) >= 1
    tr = tool_results[0]

    # Content must NOT be the huge 13,500 chars raw text!
    # It must be offloaded as an artifact
    assert len(tr.get("content", "")) < len(huge_log)
    assert "[Context Artifact]" in tr.get("content", "")
    assert "_context_artifact_id" in tr
    artifact_id = tr["_context_artifact_id"]
    assert artifact_id.startswith("ctx_")

    # Verify that the store contains this artifact and load_context_artifact can read it
    store = ContextArtifactStore(tmp_path)
    assert store.exists(artifact_id)
    recovered, meta = store.read_range(artifact_id, offset=0, limit=500)
    assert recovered is not None
    assert "Error log line from large test execution run" in recovered


def test_agent_loop_budget_manager_active_when_work_chain_disabled(tmp_path: Path, permissions, tools):
    """Verify ContextBudgetManager runs before model request even if enable_work_chain=False."""
    huge_log = "Detailed diagnostic line from system service\n" * 250
    (tmp_path / "diag.log").write_text(huge_log, encoding="utf-8")

    steps = [
        AgentStep(
            type="tool_calls",
            calls=[{
                "id": "c1",
                "toolName": "read_file",
                "input": {"path": str(tmp_path / "diag.log")},
            }],
        ),
        AgentStep(type="assistant", content="Analyzed."),
    ]

    model = RecordingModel(steps=steps)
    initial_messages = [{"role": "user", "content": "Check diag.log"}]

    final_messages = run_agent_turn(
        messages=initial_messages,
        system_prompt="System instructions.",
        model=model,
        tools=tools,
        cwd=str(tmp_path),
        permissions=permissions,
        max_steps=5,
        enable_work_chain=False,  # Explicitly disabled!
    )

    assert len(model.recorded_messages) >= 2
    second_call_messages = model.recorded_messages[1]

    tool_results = [m for m in second_call_messages if m.get("role") in ("tool", "tool_result")]
    assert len(tool_results) >= 1
    tr = tool_results[0]

    # Even with work chain disabled, the budget manager offloaded the huge tool result
    assert "[Context Artifact]" in tr.get("content", "")
    assert tr.get("_context_action") == "offload"


def test_agent_loop_model_calls_load_context_artifact_to_recover(tmp_path: Path, permissions, tools):
    """Verify model can call load_context_artifact to recover slice during multi-step turn."""
    full_text = "TRACE_LINE_DATA: Important stack trace frame #12345\n" * 200
    (tmp_path / "trace.txt").write_text(full_text, encoding="utf-8")

    class AdaptiveModel:
        def __init__(self):
            self.call_count = 0
            self.saved_artifact_id = ""

        def next(self, messages, **_kwargs) -> AgentStep:
            self.call_count += 1
            if self.call_count == 1:
                # Step 1: read file
                return AgentStep(
                    type="tool_calls",
                    calls=[{
                        "id": "c1",
                        "toolName": "read_file",
                        "input": {"path": str(tmp_path / "trace.txt")},
                    }],
                )
            elif self.call_count == 2:
                # Step 2: find offloaded artifact id from tool_result and call load_context_artifact
                for m in messages:
                    if m.get("_context_artifact_id"):
                        self.saved_artifact_id = m["_context_artifact_id"]
                        break
                assert self.saved_artifact_id.startswith("ctx_")
                return AgentStep(
                    type="tool_calls",
                    calls=[{
                        "id": "c2",
                        "toolName": "load_context_artifact",
                        "input": {
                            "artifact_id": self.saved_artifact_id,
                            "offset": 0,
                            "limit": 500,
                        },
                    }],
                )
            else:
                # Step 3: inspect recovered content and finish
                recovered_msgs = [
                    m for m in messages
                    if m.get("role") in ("tool", "tool_result") and "Important stack trace frame" in str(m.get("content", ""))
                ]
                assert len(recovered_msgs) >= 1
                return AgentStep(type="assistant", content="Successfully recovered slice.")

    model = AdaptiveModel()
    final_messages = run_agent_turn(
        messages=[{"role": "user", "content": "Analyze trace.txt"}],
        system_prompt="System instructions.",
        model=model,
        tools=tools,
        cwd=str(tmp_path),
        permissions=permissions,
        max_steps=5,
    )

    assert model.call_count == 3
    assert any("Successfully recovered slice." in str(m.get("content", "")) for m in final_messages)
