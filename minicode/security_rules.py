"""Centralized security classification rules for MiniCode Python.

Unifies catastrophic command recognition, development command classification,
pure read-only commands, and sensitive path detection across SecurityPolicyEngine,
PermissionManager, and run_command.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Sequence

# ---------------------------------------------------------------------------
# Catastrophic / High-Risk Command Detection
# ---------------------------------------------------------------------------

_CATASTROPHIC_PATTERNS = [
    # Git destructive commands
    (re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE), "git reset --hard discards all local working tree and index changes"),
    (re.compile(r"\bgit\s+clean\s+-[a-z]*f", re.IGNORECASE), "git clean with force flag deletes untracked files"),
    (re.compile(r"\bgit\s+push\s+.*(--force|-f)\b", re.IGNORECASE), "git push with force flag rewrites remote history"),
    (re.compile(r"\bgit\s+checkout\s+--\s+\.", re.IGNORECASE), "git checkout -- . overwrites working tree files"),
    
    # Recursive filesystem wiping
    (re.compile(r"\brm\s+-[a-z]*r[a-z]*f\b|\brm\s+-[a-z]*f[a-z]*r\b", re.IGNORECASE), "rm -rf can cause catastrophic and irrecoverable data loss"),
    (re.compile(r"\b(del|erase)\s+/[sfq]", re.IGNORECASE), "del/erase with /s or /q can recursively delete directory contents"),
    (re.compile(r"\b(rmdir|rd)\s+/s", re.IGNORECASE), "rmdir/rd with /s removes directory trees recursively"),
    
    # Disk formatting and raw block write
    (re.compile(r"\b(mkfs|mkfs\.[a-z0-9]+|fdisk)\b", re.IGNORECASE), "disk partition formatting commands destroy filesystems"),
    (re.compile(r"\bdd\s+if=", re.IGNORECASE), "dd disk block writing can overwrite drives and partitions"),
    (re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE), "format drive command destroys disk volume contents"),
    
    # Remote payload download and direct execution
    (re.compile(r"\b(curl|wget)\b.*\|\s*(sh|bash|zsh|fish)\b", re.IGNORECASE), "piping web downloads directly to a shell executes untrusted code"),
    (re.compile(r"\b(iwr|irm|invoke-webrequest|invoke-restmethod)\b.*\|\s*(iex|invoke-expression)\b", re.IGNORECASE), "PowerShell download piped to Invoke-Expression executes untrusted remote code"),
    (re.compile(r"\b(powershell|pwsh)\b.*(\biex\b|invoke-expression)", re.IGNORECASE), "invoking dynamic expression evaluation in PowerShell is dangerous"),
    
    # Excessive permissions
    (re.compile(r"\bchmod\s+(-[a-z]*R[a-z]*\s+)?777\b", re.IGNORECASE), "chmod 777 grants world-writable permissions"),
    (re.compile(r"\breg\s+delete\s+HKLM\b", re.IGNORECASE), "registry deletion in HKLM damages system configuration"),
]


def is_catastrophic_command(command: str, args: Sequence[str] | None = None) -> tuple[bool, str]:
    """Check if a command is catastrophic (CRITICAL risk, hard-denied even in BYPASS mode)."""
    full_cmd = command if not args else f"{command} {' '.join(args)}"
    normalized = " ".join(full_cmd.split())

    for pattern, reason in _CATASTROPHIC_PATTERNS:
        if pattern.search(normalized):
            return True, reason

    # Check command-specific token arguments
    cmd_lower = command.lower()
    if args:
        args_lower = [str(a).lower().strip() for a in args]
        if cmd_lower == "git":
            if "reset" in args_lower and "--hard" in args_lower:
                return True, "git reset --hard discards all local working tree and index changes"
            if "clean" in args_lower and any(a in {"-f", "-fd", "-df", "-xdf", "-fx"} for a in args_lower):
                return True, "git clean with force flag deletes untracked files"
            if "push" in args_lower and any(a in {"--force", "-f"} for a in args_lower):
                return True, "git push with force flag rewrites remote history"
        elif cmd_lower == "rm":
            combined_flags = "".join(a for a in args_lower if a.startswith("-"))
            if "r" in combined_flags and "f" in combined_flags:
                return True, "rm -rf can cause catastrophic and irrecoverable data loss"

    return False, ""


# ---------------------------------------------------------------------------
# Pure Read-Only Commands vs Development Commands
# ---------------------------------------------------------------------------

READONLY_COMMANDS = frozenset({
    # Unix read-only inspection tools
    "pwd", "ls", "cat", "grep", "rg", "find", "head", "tail", "wc",
    "echo", "df", "du", "whoami", "sed",
    # Windows equivalents
    "dir", "type", "where", "findstr", "more", "hostname",
})

DEVELOPMENT_COMMANDS = frozenset({
    "pytest", "python", "python3", "pythonw",
    "npm", "npx", "node", "yarn", "pnpm", "bun", "deno",
    "git", "pip", "pip3", "uv",
    "cargo", "go", "make", "cmake", "dotnet",
    "bash", "sh", "zsh", "powershell", "pwsh", "cmd",
})


def is_pure_readonly_command(command: str, args: Sequence[str] | None = None) -> bool:
    """Check if command is guaranteed read-only inspection that can be auto-approved."""
    cmd_name = Path(command).name.lower()
    if os.name == "nt" and cmd_name.endswith((".exe", ".cmd", ".bat")):
        cmd_name = cmd_name.rsplit(".", 1)[0]
    
    if cmd_name not in READONLY_COMMANDS:
        return False

    # If args contain redirection, pipe or dangerous modifiers, it's not pure read-only
    if args:
        for arg in args:
            if any(ch in str(arg) for ch in (">", "|", "&", ";")):
                return False
    return True


def is_development_command(command: str) -> bool:
    """Check if command is a development tool (requires approval in DEFAULT mode)."""
    cmd_name = Path(command).name.lower()
    if os.name == "nt" and cmd_name.endswith((".exe", ".cmd", ".bat")):
        cmd_name = cmd_name.rsplit(".", 1)[0]
    return cmd_name in DEVELOPMENT_COMMANDS


# ---------------------------------------------------------------------------
# Sensitive Path Classification
# ---------------------------------------------------------------------------

_SENSITIVE_FILENAME_PATTERNS = [
    # Environment configs containing credentials
    re.compile(r"^\.env(\..+)?$", re.IGNORECASE),
    # Private keys and certificates
    re.compile(r"^.*\.(pem|key|pfx|p12|pkcs12)$", re.IGNORECASE),
    re.compile(r"^id_(rsa|ed25519|ecdsa|dsa)(\..+)?$", re.IGNORECASE),
    # Common cloud & service credentials
    re.compile(r"^(credentials|service-account.*)\.json$", re.IGNORECASE),
    re.compile(r"^\.(npmrc|pypirc)$", re.IGNORECASE),
]

_SENSITIVE_DIR_PATTERNS = [
    re.compile(r"(^|[/\\])\.aws([/\\]|$)", re.IGNORECASE),
    re.compile(r"(^|[/\\])\.gcp([/\\]|$)", re.IGNORECASE),
    re.compile(r"(^|[/\\])\.azure([/\\]|$)", re.IGNORECASE),
    re.compile(r"(^|[/\\])\.ssh([/\\]|$)", re.IGNORECASE),
    re.compile(r"(^|[/\\])\.docker([/\\]config\.json)?$", re.IGNORECASE),
]


def is_git_internal_metadata(path: str | Path) -> bool:
    """Check if path targets internal .git metadata (not tracked files like .gitignore)."""
    p = Path(path)
    parts = p.parts
    # Exact check for .git directory or files inside .git
    for i, part in enumerate(parts):
        if part.lower() == ".git":
            # If it's the .git directory itself or something inside it
            return True
    return False


def classify_sensitive_path(target_path: str | Path, cwd: str | Path | None = None) -> tuple[bool, str]:
    """Classify if target_path points to a sensitive file or directory.
    
    Resolves symlinks to prevent symlink traversal attacks.
    """
    raw_path = Path(target_path)
    if not raw_path.is_absolute() and cwd:
        candidate = (Path(cwd) / raw_path)
    else:
        candidate = raw_path

    # Try resolving symlinks
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate

    # Check filename patterns
    fname = resolved.name
    for pattern in _SENSITIVE_FILENAME_PATTERNS:
        if pattern.match(fname):
            return True, f"Sensitive credentials or private key file: {fname}"

    # Check directory patterns
    resolved_str = str(resolved)
    for pattern in _SENSITIVE_DIR_PATTERNS:
        if pattern.search(resolved_str):
            return True, f"Sensitive configuration directory: {resolved_str}"

    return False, ""
