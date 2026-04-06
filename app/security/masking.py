"""Security helpers – secret masking and safe error messages."""

from __future__ import annotations

import re

# Minimum visible characters at the start/end when masking
_REVEAL = 4
_MASK_CHAR = "*"


def mask_secret(value: str | None, *, reveal: int = _REVEAL) -> str:
    """Mask a secret string, keeping only *reveal* chars at each end visible.

    Examples:
        mask_secret("sk-abc123xyz789")  -> "sk-a**********z789"
        mask_secret(None)               -> "<not set>"
        mask_secret("ab")               -> "****"
    """
    if not value:
        return "<not set>"
    if len(value) <= reveal * 2:
        return _MASK_CHAR * max(len(value), 4)
    return value[:reveal] + _MASK_CHAR * (len(value) - reveal * 2) + value[-reveal:]


def redact_key_in_text(text: str) -> str:
    """Replace anything that looks like an API key or credential in free text."""
    # Common patterns: sk-..., sk-ant-..., AI...
    redacted = re.sub(
        r"(sk-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]+",
        r"\1" + "****",
        text,
    )
    redacted = re.sub(
        r"\b(AI[A-Za-z0-9_-]{4})[A-Za-z0-9_-]+\b",
        r"\1" + "****",
        redacted,
    )
    # Bearer / token headers
    redacted = re.sub(
        r"(Bearer\s+)[A-Za-z0-9._\-]+",
        r"\1****",
        redacted,
    )
    # Passwords embedded in connection-style URIs (scheme://user:password@host)
    redacted = re.sub(
        r"(://[^:/?#]*:)[^@]+(@)",
        r"\1****\2",
        redacted,
    )
    return redacted


def safe_error_message(exc: Exception) -> str:
    """Return a redacted string representation of an exception."""
    return redact_key_in_text(str(exc))


# --- User-facing error messages ---

MSG_MISSING_API_KEY = "API key for {provider} is not configured. Please enter it in Settings → Credentials."
MSG_PROVIDER_UNREACHABLE = "Could not connect to {provider}. Check your network and credentials."
MSG_MODEL_FETCH_FAILED = "Failed to retrieve models from {provider}. {detail}"
MSG_EMPTY_OLLAMA_CATALOG = "No models found in your local Ollama instance. Pull a model first: `ollama pull <model>`."
MSG_DEFAULT_MODEL_UNAVAILABLE = (
    "The default model '{model_id}' was not found in the {provider} model list. "
    "The first available model has been selected instead."
)
