"""Great Expectations context management.

Isolates GX initialization behind a factory so upstream code never imports
GX directly.  If GX is not installed the module degrades gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GX_AVAILABLE: bool | None = None


def is_gx_available() -> bool:
    """Return True if great_expectations is importable."""
    global _GX_AVAILABLE
    if _GX_AVAILABLE is None:
        try:
            import great_expectations  # noqa: F401
            _GX_AVAILABLE = True
        except ImportError:
            _GX_AVAILABLE = False
    return _GX_AVAILABLE


def get_ephemeral_context() -> Any:
    """Return an ephemeral (in-memory) GX DataContext.

    This uses the modern GX Core pattern for in-memory exploration and
    validation: ``gx.get_context()``.

    Returns the context object, or raises ValidationSetupError if GX is
    unavailable.
    """
    if not is_gx_available():
        from app.validation.errors import ValidationSetupError
        raise ValidationSetupError(
            "great_expectations is not installed.  "
            "Install it with: pip install 'great_expectations>=1.0'"
        )
    import great_expectations as gx  # noqa: WPS433

    return gx.get_context()


def get_file_context(context_root_dir: Path) -> Any:
    """Return a file-backed GX DataContext.

    TODO: implement this when local Data Docs are added for real.
    The current build intentionally avoids inventing a file-context/Data Docs
    workflow until it is verified against the exact GX Core version we ship.
    """
    from app.validation.errors import ValidationSetupError

    raise ValidationSetupError(
        "File-backed Great Expectations contexts for local Data Docs are "
        "not implemented yet. TODO: wire verified GX Core Data Docs support "
        f"for context root {context_root_dir}."
    )
