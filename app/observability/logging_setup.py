"""Structured (JSON) logging with correlation injection.

The implementation is dependency-free: we serialize records via stdlib
:mod:`json`. A separate text formatter is preserved for local-dev ergonomics
and is selected via the ``AUTOTABML_LOG_FORMAT`` environment variable
(``json`` or ``text``; default ``text``).

All log records – regardless of format – are passed through
:func:`app.security.masking.redact_key_in_text` so that accidental secret
leakage in interpolated messages is scrubbed at the boundary.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from app.observability.context import current_context
from app.security.masking import redact_key_in_text

# Standard fields that already live on :class:`logging.LogRecord` and should
# not be duplicated into the JSON ``extra`` block.
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class CorrelationFilter(logging.Filter):
    """Inject the active correlation context onto every :class:`LogRecord`.

    Each key from :func:`current_context` is set as an attribute on the
    record, which makes it available both to :class:`JsonFormatter` (auto-
    serialized) and to text formatters via ``%(<key>)s``-style format
    strings.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - stdlib hook
        for key, value in current_context().items():
            # Avoid clobbering attributes the caller explicitly set via extra=.
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class _RedactingTextFormatter(logging.Formatter):
    """Plain-text formatter that scrubs obvious secrets from the message."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return redact_key_in_text(message)


class JsonFormatter(logging.Formatter):
    """Render :class:`LogRecord` as a single-line JSON document.

    The output is stable and shallow – nested objects are serialized via
    ``default=str`` so that unusual ``extra`` payloads cannot break the
    logging pipeline. Correlation context attached by
    :class:`CorrelationFilter` is emitted as top-level keys for easy grep /
    Loki / Elasticsearch ingestion.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_key_in_text(record.getMessage()),
        }
        if record.exc_info:
            payload["exc_info"] = redact_key_in_text(self.formatException(record.exc_info))
        if record.stack_info:
            payload["stack_info"] = redact_key_in_text(record.stack_info)

        # Promote any non-reserved attributes (correlation context + caller
        # ``extra=`` payloads) into the top-level document.
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                continue
            payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False)


def install_correlation_filter(logger: logging.Logger | None = None) -> CorrelationFilter:
    """Attach :class:`CorrelationFilter` to ``logger`` (root by default).

    Idempotent: re-installing on the same logger does not stack filters.
    Returns the installed filter so callers (such as tests) can detach it.
    """

    target = logger or logging.getLogger()
    for existing in target.filters:
        if isinstance(existing, CorrelationFilter):
            return existing
    correlation_filter = CorrelationFilter()
    target.addFilter(correlation_filter)
    # Also ensure handlers pick up the filter, since logger-level filters do
    # not propagate to handlers attached after the fact.
    for handler in target.handlers:
        if not any(isinstance(f, CorrelationFilter) for f in handler.filters):
            handler.addFilter(correlation_filter)
    return correlation_filter


def configure_observability_logging(
    level: int | str | None = None,
    *,
    fmt: str | None = None,
) -> None:
    """Configure the root logger for structured/observable output.

    Parameters
    ----------
    level:
        Override the root log level. Falls back to ``AUTOTABML_LOG_LEVEL``
        and finally to ``INFO``.
    fmt:
        ``"json"`` or ``"text"``. Falls back to ``AUTOTABML_LOG_FORMAT`` and
        finally to ``"text"`` so existing local-dev ergonomics are preserved.
    """

    root = logging.getLogger()
    if root.handlers:
        # configure_logging() in app/logging_config.py is idempotent; mirror
        # that contract here so re-imports (Streamlit hot-reload, pytest
        # session reuse) do not stack handlers.
        install_correlation_filter(root)
        return

    handler = logging.StreamHandler(sys.stderr)
    chosen_fmt = (fmt or os.environ.get("AUTOTABML_LOG_FORMAT", "text")).strip().lower()
    if chosen_fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            _RedactingTextFormatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    effective_level = level or os.environ.get("AUTOTABML_LOG_LEVEL", "INFO")
    root.setLevel(effective_level)
    root.addHandler(handler)
    install_correlation_filter(root)
