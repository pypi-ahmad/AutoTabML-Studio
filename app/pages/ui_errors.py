"""Shared UI exception logging helpers for Streamlit pages."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from app.errors import log_exception

logger = logging.getLogger("app.pages")


def log_ui_exception(
    exc: Exception,
    *,
    operation: str,
    context: Mapping[str, Any] | None = None,
    level: int = logging.WARNING,
) -> None:
    """Log a caught UI exception with structured context."""

    log_exception(logger, exc, operation=operation, level=level, context=context)


def log_ui_debug_exception(
    exc: Exception,
    *,
    operation: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Log a non-fatal UI fallback at debug level."""

    log_ui_exception(exc, operation=operation, context=context, level=logging.DEBUG)