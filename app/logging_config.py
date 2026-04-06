"""Logging configuration for AutoTabML Studio."""

from __future__ import annotations

import logging
import os
import sys

from app.security.masking import redact_key_in_text


class _RedactingFormatter(logging.Formatter):
    """Formatter that redacts obvious secret-like substrings from log output."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return redact_key_in_text(message)


def _configure_noisy_dependency_loggers() -> None:
    """Reduce non-actionable third-party log noise in normal app and batch runs."""

    logging.getLogger("great_expectations._docs_decorators").setLevel(logging.WARNING)
    logging.getLogger("great_expectations.expectations.registry").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.style.core").setLevel(logging.ERROR)
    logging.getLogger("visions.backends").setLevel(logging.WARNING)


def configure_logging(level: int | str | None = None) -> None:
    """Set up structured logging to stderr (12-factor compliant)."""
    _configure_noisy_dependency_loggers()

    root = logging.getLogger()
    if root.handlers:
        return  # already configured
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        _RedactingFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    effective_level = level or os.environ.get("AUTOTABML_LOG_LEVEL", "INFO")
    root.setLevel(effective_level)
    root.addHandler(handler)
