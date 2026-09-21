"""Centralized secret and sensitive data redaction for MiniCode.

Provides deterministic masking of credentials, private keys, authorization tokens,
and secret assignments across execution traces, tool outputs, and security audit logs.
"""
from __future__ import annotations

import re
from typing import Any

from minicode.release_readiness import (
    redact_sensitive_payload as _rr_redact_payload,
    redact_sensitive_text as _rr_redact_text,
)

# Standard maximum length for truncated summary text
DEFAULT_MAX_SUMMARY_LENGTH = 1500

# Private key block pattern
_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [A-Z0-9_-]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9_-]+ PRIVATE KEY-----",
    re.MULTILINE,
)

# JWT pattern
_JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")

# Bearer token pattern
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._~+/-]{10,})")

# API key / Secret assignment pattern
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<key>(?:password|secret|token|api_?key|auth|bearer)[\s:=]+)(?P<value>[^\s\"',;]{8,})",
    re.IGNORECASE,
)

# Environment file line pattern: KEY=VALUE where KEY suggests sensitive material
_ENV_SECRET_PATTERN = re.compile(
    r"(?im)^(?P<prefix>\s*[A-Za-z0-9_]*(?:SECRET|KEY|TOKEN|PASSWORD|AUTH|CREDENTIAL|PRIVATE)[A-Za-z0-9_]*\s*=\s*)(?P<val>[^\r\n]+)$"
)


def redact_text(text: str, max_length: int | None = None) -> str:
    """Deterministically redact secrets from plain text.
    
    Replaces private keys, bearer tokens, API keys, and sensitive environment variables
    with standard [REDACTED] or [REDACTED_PRIVATE_KEY] markers.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # First pass: private key blocks
    cleaned = _PRIVATE_KEY_PATTERN.sub("[REDACTED_PRIVATE_KEY]", text)

    # Second pass: Bearer tokens
    cleaned = _BEARER_PATTERN.sub(r"\g<1>[REDACTED_TOKEN]", cleaned)

    # Third pass: JWT tokens
    cleaned = _JWT_PATTERN.sub("[REDACTED_TOKEN]", cleaned)

    # Fourth pass: Secret assignments
    cleaned = _SECRET_ASSIGNMENT_PATTERN.sub(r"\g<key>[REDACTED]", cleaned)

    # Fifth pass: Environment variable lines
    cleaned = _ENV_SECRET_PATTERN.sub(r"\g<prefix>[REDACTED]", cleaned)

    # Sixth pass: standard release_readiness patterns
    cleaned = _rr_redact_text(cleaned)

    # Seventh pass: Ensure Bearer tokens are labeled [REDACTED_TOKEN]
    cleaned = re.sub(r"(?i)\bBearer\s+(\[REDACTED\]|[A-Za-z0-9._~+/-]{8,})", "Bearer [REDACTED_TOKEN]", cleaned)


    # Optional bounded truncation
    if max_length is not None and len(cleaned) > max_length:
        cleaned = cleaned[: max_length - 17] + "... [truncated]"

    return cleaned


def redact_payload(value: Any, max_length: int | None = None) -> Any:
    """Recursively redact secrets in nested dictionaries, lists, and strings."""
    # First pass through release_readiness payload sanitizer
    redacted = _rr_redact_payload(value)

    if isinstance(redacted, dict):
        return {
            str(k): redact_payload(v, max_length=max_length)
            for k, v in redacted.items()
        }
    if isinstance(redacted, (list, tuple)):
        return [redact_payload(item, max_length=max_length) for item in redacted]
    if isinstance(redacted, str):
        return redact_text(redacted, max_length=max_length)
    return redacted
