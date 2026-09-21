"""Centralized Security Policy Engine for MiniCode Python.

Provides unified evaluation of caller identity, tool capability classification,
argument sensitivity, execution boundaries, untrusted context taint, and permission modes
prior to tool execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import uuid

from minicode.auto_mode import PermissionMode
from minicode.security_rules import (
    classify_sensitive_path,
    is_catastrophic_command,
    is_development_command,
    is_git_internal_metadata,
    is_pure_readonly_command,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SecurityDecision(str, Enum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


class SecurityRisk(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SecurityActor(str, Enum):
    PARENT = "PARENT"
    CHILD = "CHILD"


class TrustLevel(str, Enum):
    TRUSTED_LOCAL = "TRUSTED_LOCAL"
    WORKSPACE = "WORKSPACE"
    UNTRUSTED_EXTERNAL = "UNTRUSTED_EXTERNAL"


class ApprovalRoute(str, Enum):
    NONE = "NONE"
    NATIVE_EDIT = "NATIVE_EDIT"
    NATIVE_COMMAND = "NATIVE_COMMAND"
    GENERIC_TOOL = "GENERIC_TOOL"


class ToolCategory(str, Enum):
    LOCAL_READ = "LOCAL_READ"
    LOCAL_WRITE = "LOCAL_WRITE"
    DESTRUCTIVE_LOCAL = "DESTRUCTIVE_LOCAL"
    EXECUTION = "EXECUTION"
    VERIFICATION = "VERIFICATION"
    GIT_READ = "GIT_READ"
    GIT_WRITE = "GIT_WRITE"
    EXTERNAL_READ = "EXTERNAL_READ"
    MCP = "MCP"
    ORCHESTRATION = "ORCHESTRATION"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Tool Security Policy Matrix
# ---------------------------------------------------------------------------

TOOL_SECURITY_POLICIES: dict[str, ToolCategory] = {
    # Local read tools
    "read_file": ToolCategory.LOCAL_READ,
    "list_files": ToolCategory.LOCAL_READ,
    "grep_files": ToolCategory.LOCAL_READ,
    "file_tree": ToolCategory.LOCAL_READ,
    "find_symbols": ToolCategory.LOCAL_READ,
    "find_references": ToolCategory.LOCAL_READ,
    "get_ast_info": ToolCategory.LOCAL_READ,
    "diff_viewer": ToolCategory.LOCAL_READ,
    "code_review": ToolCategory.LOCAL_READ,
    "ask_user": ToolCategory.LOCAL_READ,
    "todo_write": ToolCategory.LOCAL_READ,
    "load_skill": ToolCategory.LOCAL_READ,
    "load_context_artifact": ToolCategory.LOCAL_READ,
    
    # Local write tools
    "write_file": ToolCategory.LOCAL_WRITE,
    "edit_file": ToolCategory.LOCAL_WRITE,
    "patch_file": ToolCategory.LOCAL_WRITE,
    "batch_copy": ToolCategory.LOCAL_WRITE,
    "batch_move": ToolCategory.LOCAL_WRITE,
    
    # Destructive local tools
    "batch_delete": ToolCategory.DESTRUCTIVE_LOCAL,
    
    # Execution tools
    "run_command": ToolCategory.EXECUTION,
    
    # Verification tools
    "test_runner": ToolCategory.VERIFICATION,
    
    # Git tools (action-dependent, defaults below)
    "git": ToolCategory.GIT_READ,
    
    # External read tools
    "web_search": ToolCategory.EXTERNAL_READ,
    "web_fetch": ToolCategory.EXTERNAL_READ,
    
    # Orchestration tools
    "task": ToolCategory.ORCHESTRATION,
    "agent_team": ToolCategory.ORCHESTRATION,
}


def classify_tool_category(tool_name: str, input_data: Any = None) -> ToolCategory:
    """Classify tool into a standard security category."""
    if tool_name.startswith("mcp__") or tool_name in {
        "read_mcp_resource", "list_mcp_resources", "get_mcp_prompt", "list_mcp_prompts"
    }:
        return ToolCategory.MCP

    if tool_name == "git" and isinstance(input_data, dict):
        action = input_data.get("action", "")
        if action == "commit":
            return ToolCategory.GIT_WRITE
        return ToolCategory.GIT_READ

    return TOOL_SECURITY_POLICIES.get(tool_name, ToolCategory.UNKNOWN)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SecurityRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool_name: str = ""
    input_data: Any = field(default_factory=dict)
    cwd: str = ""
    actor: SecurityActor = SecurityActor.PARENT
    agent_role: str = "parent"
    agent_depth: int = 0
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    is_mcp: bool = False
    is_external: bool = False
    untrusted_context_seen: bool = False


@dataclass
class SecurityAssessment:
    decision: SecurityDecision
    risk: SecurityRisk
    rule_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    approval_route: ApprovalRoute = ApprovalRoute.NONE
    sensitive_paths: list[str] = field(default_factory=list)
    is_external: bool = False
    output_trust: TrustLevel = TrustLevel.TRUSTED_LOCAL
    hard_deny: bool = False


@dataclass
class SecurityPolicyMetrics:
    evaluations: int = 0
    allow_count: int = 0
    ask_count: int = 0
    deny_count: int = 0
    hard_denies: int = 0
    sensitive_file_requests: int = 0
    sensitive_file_denies: int = 0
    command_risk_events: int = 0
    mcp_requests: int = 0
    mcp_approval_requests: int = 0
    untrusted_outputs: int = 0
    injection_detections: int = 0
    child_policy_denies: int = 0
    audit_events: int = 0
    audit_redactions: int = 0


# ---------------------------------------------------------------------------
# Centralized Security Policy Engine
# ---------------------------------------------------------------------------

class SecurityPolicyEngine:
    """Centralized policy engine evaluated before and after tool execution."""

    def __init__(self, default_mode: PermissionMode = PermissionMode.DEFAULT) -> None:
        self.default_mode = default_mode
        self.metrics = SecurityPolicyMetrics()

    def evaluate(self, request: SecurityRequest) -> SecurityAssessment:
        """Evaluate a tool request and determine ALLOW / ASK / DENY policy decision."""
        self.metrics.evaluations += 1

        tool_name = request.tool_name
        category = classify_tool_category(tool_name, request.input_data)
        mode = request.permission_mode or self.default_mode
        actor = request.actor
        is_child = actor == SecurityActor.CHILD or request.agent_depth > 0

        rule_ids: list[str] = []
        reasons: list[str] = []
        sensitive_paths: list[str] = []

        # ===================================================================
        # 1. Child Agent Defense-in-Depth Policies
        # ===================================================================
        if is_child:
            # Child agents are strictly forbidden from spawning nested agents
            if category == ToolCategory.ORCHESTRATION:
                self.metrics.deny_count += 1
                self.metrics.hard_denies += 1
                self.metrics.child_policy_denies += 1
                return SecurityAssessment(
                    decision=SecurityDecision.DENY,
                    risk=SecurityRisk.CRITICAL,
                    rule_ids=["child_orchestration_denied"],
                    reasons=[f"Child agents (depth={request.agent_depth}) are forbidden from invoking orchestration tool '{tool_name}'"],
                    hard_deny=True,
                )

            # Child agents are strictly forbidden from calling MCP tools
            if category == ToolCategory.MCP or request.is_mcp:
                self.metrics.deny_count += 1
                self.metrics.hard_denies += 1
                self.metrics.child_policy_denies += 1
                return SecurityAssessment(
                    decision=SecurityDecision.DENY,
                    risk=SecurityRisk.HIGH,
                    rule_ids=["child_mcp_denied"],
                    reasons=[f"Child agents are forbidden from calling MCP tool '{tool_name}'"],
                    hard_deny=True,
                )

            # Role-specific containment
            child_role = request.agent_role.lower()
            if child_role in {"research", "test", "reviewer"}:
                if category in {
                    ToolCategory.LOCAL_WRITE,
                    ToolCategory.DESTRUCTIVE_LOCAL,
                    ToolCategory.GIT_WRITE,
                    ToolCategory.EXECUTION,
                }:
                    self.metrics.deny_count += 1
                    self.metrics.hard_denies += 1
                    self.metrics.child_policy_denies += 1
                    return SecurityAssessment(
                        decision=SecurityDecision.DENY,
                        risk=SecurityRisk.HIGH,
                        rule_ids=["child_role_write_denied"],
                        reasons=[f"Sub-agent role '{child_role}' is restricted to read-only tools; '{tool_name}' denied"],
                        hard_deny=True,
                    )

        # ===================================================================
        # 2. Catastrophic Command Detection (CRITICAL, Hard Deny)
        # ===================================================================
        if category == ToolCategory.EXECUTION and isinstance(request.input_data, dict):
            cmd = request.input_data.get("command", "")
            args = request.input_data.get("args") or []
            is_catastrophic, cat_reason = is_catastrophic_command(cmd, args)
            if is_catastrophic:
                self.metrics.deny_count += 1
                self.metrics.hard_denies += 1
                self.metrics.command_risk_events += 1
                return SecurityAssessment(
                    decision=SecurityDecision.DENY,
                    risk=SecurityRisk.CRITICAL,
                    rule_ids=["catastrophic_command_hard_deny"],
                    reasons=[f"Catastrophic command blocked: {cat_reason}"],
                    hard_deny=True,
                )

        # ===================================================================
        # 3. Sensitive Files & Repository Internal Boundaries
        # ===================================================================
        if isinstance(request.input_data, dict):
            # Extract target paths from standard tool arguments
            candidate_paths: list[str] = []
            for k in ("path", "destination", "source"):
                val = request.input_data.get(k)
                if isinstance(val, str) and val.strip():
                    candidate_paths.append(val.strip())

            for target in candidate_paths:
                # Disallow modifying or deleting .git/ internal metadata
                if category in {ToolCategory.LOCAL_WRITE, ToolCategory.DESTRUCTIVE_LOCAL}:
                    if is_git_internal_metadata(target):
                        self.metrics.deny_count += 1
                        self.metrics.hard_denies += 1
                        self.metrics.sensitive_file_denies += 1
                        return SecurityAssessment(
                            decision=SecurityDecision.DENY,
                            risk=SecurityRisk.CRITICAL,
                            rule_ids=["git_internal_metadata_write_denied"],
                            reasons=[f"Modifying internal git metadata is prohibited: {target}"],
                            hard_deny=True,
                        )

                # Check sensitive path classification
                is_sens, sens_reason = classify_sensitive_path(target, cwd=request.cwd)
                if is_sens:
                    sensitive_paths.append(target)
                    self.metrics.sensitive_file_requests += 1

            # Prevent batch_delete of workspace root or .git
            if tool_name == "batch_delete":
                del_path = str(request.input_data.get("path", "")).strip()
                if del_path in {".", "./", "", "/"}:
                    self.metrics.deny_count += 1
                    self.metrics.hard_denies += 1
                    return SecurityAssessment(
                        decision=SecurityDecision.DENY,
                        risk=SecurityRisk.CRITICAL,
                        rule_ids=["workspace_root_delete_denied"],
                        reasons=["Deleting the workspace root itself is prohibited"],
                        hard_deny=True,
                    )
                if is_git_internal_metadata(del_path):
                    self.metrics.deny_count += 1
                    self.metrics.hard_denies += 1
                    return SecurityAssessment(
                        decision=SecurityDecision.DENY,
                        risk=SecurityRisk.CRITICAL,
                        rule_ids=["git_directory_delete_denied"],
                        reasons=["Deleting .git directory is prohibited"],
                        hard_deny=True,
                    )

        # ===================================================================
        # 4. Sensitive File Read Policy
        # ===================================================================
        if category == ToolCategory.LOCAL_READ and sensitive_paths:
            # Sensitive file reads require explicit approval
            rule_ids.append("sensitive_file_read_approval")
            reasons.append(f"Sensitive configuration or key file access: {', '.join(sensitive_paths)}")
            if mode != PermissionMode.BYPASS:
                self.metrics.ask_count += 1
                return SecurityAssessment(
                    decision=SecurityDecision.ASK,
                    risk=SecurityRisk.HIGH,
                    rule_ids=rule_ids,
                    reasons=reasons,
                    approval_route=ApprovalRoute.GENERIC_TOOL,
                    sensitive_paths=sensitive_paths,
                )

        # ===================================================================
        # 5. External Tools & Output Trust Classification
        # ===================================================================
        output_trust = TrustLevel.TRUSTED_LOCAL
        is_external = False
        if category == ToolCategory.EXTERNAL_READ:
            output_trust = TrustLevel.UNTRUSTED_EXTERNAL
            is_external = True
        elif category == ToolCategory.MCP or request.is_mcp:
            self.metrics.mcp_requests += 1
            output_trust = TrustLevel.UNTRUSTED_EXTERNAL
            is_external = True

        # ===================================================================
        # 6. Mode & Category Evaluation
        # ===================================================================
        # Mode: PLAN (strictly read-only)
        if mode == PermissionMode.PLAN:
            if category in {ToolCategory.LOCAL_READ, ToolCategory.EXTERNAL_READ, ToolCategory.GIT_READ}:
                self.metrics.allow_count += 1
                return SecurityAssessment(
                    decision=SecurityDecision.ALLOW,
                    risk=SecurityRisk.SAFE,
                    rule_ids=["plan_mode_read_allowed"],
                    reasons=["Plan mode: read-only operation permitted"],
                    approval_route=ApprovalRoute.NONE,
                    output_trust=output_trust,
                    is_external=is_external,
                )
            self.metrics.deny_count += 1
            return SecurityAssessment(
                decision=SecurityDecision.DENY,
                risk=SecurityRisk.HIGH,
                rule_ids=["plan_mode_mutation_blocked"],
                reasons=[f"Plan mode disallows state-changing operations: '{tool_name}'"],
                hard_deny=False,
            )

        # Mode: BYPASS (skips normal prompts, but preserves hard denies checked earlier)
        if mode == PermissionMode.BYPASS:
            self.metrics.allow_count += 1
            return SecurityAssessment(
                decision=SecurityDecision.ALLOW,
                risk=SecurityRisk.LOW,
                rule_ids=["bypass_mode_approved"],
                reasons=["Bypass mode: interactive prompt skipped"],
                approval_route=ApprovalRoute.NONE,
                output_trust=output_trust,
                is_external=is_external,
            )

        # Evaluate by Category under AUTO or DEFAULT
        decision = SecurityDecision.ASK
        risk = SecurityRisk.MEDIUM
        route = ApprovalRoute.NONE

        if category == ToolCategory.LOCAL_READ:
            decision = SecurityDecision.ALLOW
            risk = SecurityRisk.SAFE
            route = ApprovalRoute.NONE

        elif category == ToolCategory.VERIFICATION:
            # test_runner is safe execution in workspace
            decision = SecurityDecision.ALLOW
            risk = SecurityRisk.SAFE
            route = ApprovalRoute.NONE

        elif category == ToolCategory.GIT_READ:
            decision = SecurityDecision.ALLOW
            risk = SecurityRisk.SAFE
            route = ApprovalRoute.NONE

        elif category == ToolCategory.EXTERNAL_READ:
            decision = SecurityDecision.ALLOW
            risk = SecurityRisk.LOW
            route = ApprovalRoute.NONE

        elif category == ToolCategory.LOCAL_WRITE:
            # write_file, edit_file, patch_file, batch_copy, batch_move
            risk = SecurityRisk.MEDIUM
            if mode == PermissionMode.AUTO:
                # AUTO mode allows normal edits unless sensitive
                if sensitive_paths:
                    decision = SecurityDecision.ASK
                    risk = SecurityRisk.HIGH
                    route = ApprovalRoute.NATIVE_EDIT
                else:
                    decision = SecurityDecision.ALLOW
                    route = ApprovalRoute.NONE
            else:
                decision = SecurityDecision.ASK
                route = ApprovalRoute.NATIVE_EDIT

        elif category == ToolCategory.DESTRUCTIVE_LOCAL:
            # batch_delete
            risk = SecurityRisk.HIGH
            decision = SecurityDecision.ASK
            route = ApprovalRoute.GENERIC_TOOL

        elif category == ToolCategory.GIT_WRITE:
            # git commit
            risk = SecurityRisk.MEDIUM
            decision = SecurityDecision.ASK
            route = ApprovalRoute.GENERIC_TOOL

        elif category == ToolCategory.EXECUTION:
            # run_command
            cmd = ""
            args: list[str] = []
            if isinstance(request.input_data, dict):
                cmd = request.input_data.get("command", "")
                args = request.input_data.get("args") or []

            if is_pure_readonly_command(cmd, args):
                decision = SecurityDecision.ALLOW
                risk = SecurityRisk.SAFE
                route = ApprovalRoute.NONE
            elif mode == PermissionMode.AUTO:
                # AUTO mode allows safe development commands unless risky
                if is_development_command(cmd):
                    decision = SecurityDecision.ALLOW
                    risk = SecurityRisk.LOW
                    route = ApprovalRoute.NONE
                else:
                    decision = SecurityDecision.ASK
                    risk = SecurityRisk.MEDIUM
                    route = ApprovalRoute.NATIVE_COMMAND
            else:
                # DEFAULT mode: development commands require approval
                decision = SecurityDecision.ASK
                risk = SecurityRisk.MEDIUM
                route = ApprovalRoute.NATIVE_COMMAND

        elif category == ToolCategory.MCP:
            # Unknown MCP defaults to ASK with generic approval
            self.metrics.mcp_approval_requests += 1
            decision = SecurityDecision.ASK
            risk = SecurityRisk.HIGH
            route = ApprovalRoute.GENERIC_TOOL

        elif category == ToolCategory.ORCHESTRATION:
            # Parent agent calling task or agent_team
            decision = SecurityDecision.ALLOW
            risk = SecurityRisk.LOW
            route = ApprovalRoute.NONE

        else:
            # UNKNOWN tool: fail-safe ASK
            decision = SecurityDecision.ASK
            risk = SecurityRisk.HIGH
            route = ApprovalRoute.GENERIC_TOOL
            rule_ids.append("unknown_tool_fail_safe")
            reasons.append(f"Unknown tool '{tool_name}' requires explicit approval")

        # ===================================================================
        # 7. Taint Escalation: External Injection Seen Earlier in Turn
        # ===================================================================
        if request.untrusted_context_seen:
            if decision == SecurityDecision.ALLOW and category in {
                ToolCategory.LOCAL_WRITE,
                ToolCategory.DESTRUCTIVE_LOCAL,
                ToolCategory.EXECUTION,
                ToolCategory.GIT_WRITE,
                ToolCategory.MCP,
            }:
                decision = SecurityDecision.ASK
                risk = SecurityRisk.HIGH
                rule_ids.append("taint_escalation_enforced")
                reasons.append("External untrusted content observed earlier in this turn; escalates to explicit approval")
                if category == ToolCategory.LOCAL_WRITE:
                    route = ApprovalRoute.NATIVE_EDIT
                elif category == ToolCategory.EXECUTION:
                    route = ApprovalRoute.NATIVE_COMMAND
                else:
                    route = ApprovalRoute.GENERIC_TOOL

        if decision == SecurityDecision.ALLOW:
            self.metrics.allow_count += 1
        elif decision == SecurityDecision.ASK:
            self.metrics.ask_count += 1
        else:
            self.metrics.deny_count += 1

        return SecurityAssessment(
            decision=decision,
            risk=risk,
            rule_ids=rule_ids,
            reasons=reasons,
            approval_route=route,
            sensitive_paths=sensitive_paths,
            is_external=is_external,
            output_trust=output_trust,
            hard_deny=False,
        )
