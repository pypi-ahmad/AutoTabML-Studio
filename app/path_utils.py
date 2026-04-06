"""Path-related helpers shared across app layers."""

from __future__ import annotations

import re


def safe_artifact_stem(name: str | None, default: str = "dataset") -> str:
    """Return a filesystem-safe stem for generated artifact filenames."""

    candidate = (name or default).strip()
    if not candidate:
        candidate = default
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
    sanitized = sanitized.strip("._-")
    return sanitized or default