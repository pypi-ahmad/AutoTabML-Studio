"""Logging configuration for AutoTabML Studio.

Thin compatibility shim over :mod:`app.observability.logging_setup`.
Existing call sites (``app.main``, ``app.cli``, tests) keep using
:func:`configure_logging`; the implementation now adds correlation context
injection and an opt-in JSON formatter (``AUTOTABML_LOG_FORMAT=json``).
"""

from __future__ import annotations

import logging

from app.observability.logging_setup import (
    _RedactingTextFormatter as _RedactingFormatter,
)
from app.observability.logging_setup import configure_observability_logging

__all__ = ["configure_logging", "_RedactingFormatter"]


def _configure_noisy_dependency_loggers() -> None:
    """Reduce non-actionable third-party log noise in normal app and batch runs."""

    logging.getLogger("great_expectations._docs_decorators").setLevel(logging.WARNING)
    logging.getLogger("great_expectations.expectations.registry").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.style.core").setLevel(logging.ERROR)
    logging.getLogger("visions.backends").setLevel(logging.WARNING)


def configure_logging(level: int | str | None = None) -> None:
    """Configure structured logging to stderr (12-factor compliant).

    Honors the ``AUTOTABML_LOG_LEVEL`` and ``AUTOTABML_LOG_FORMAT`` env vars.
    Idempotent: subsequent calls only ensure the correlation filter is
    attached and do not stack handlers.
    """

    _configure_noisy_dependency_loggers()
    configure_observability_logging(level=level)
