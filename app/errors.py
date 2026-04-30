"""Cross-cutting error-handling utilities.

This module provides:

* :class:`AutoTabMLError` — an opt-in umbrella base for application-level
  domain exceptions. Existing layer-specific exception hierarchies (ingestion,
  prediction, registry, tracking, etc.) deliberately do **not** all inherit
  from this class today. This base is provided for new code paths that want
  one place to catch any AutoTabML domain error without re-importing every
  layer.
* :func:`log_exception` — structured logging for caught exceptions, attaching
  an explicit operation name, exception class, and optional context fields so
  the failure is observable even when it is intentionally swallowed.
* :func:`log_and_wrap` — narrow shim used at module boundaries where we want
  to log a caught exception, then raise a domain-specific exception with the
  original chained as ``__cause__``.

These helpers exist so that ``except Exception`` blocks that *must* remain
broad (third-party dependency boundaries, optional features, UI fallbacks)
are still observable and never silent.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, NoReturn, TypeVar

ExceptionT = TypeVar("ExceptionT", bound=BaseException)


class AutoTabMLError(Exception):
    """Opt-in umbrella base class for application-level domain errors."""


def log_exception(
    logger: logging.Logger,
    exc: BaseException,
    *,
    operation: str,
    level: int = logging.WARNING,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Emit a structured log record for a caught exception.

    The record carries:

    * ``operation`` — short identifier of the failing operation
    * ``error_type`` — qualified class name of the exception
    * ``error_message`` — ``str(exc)``
    * any additional fields supplied via ``context``

    The exception traceback is attached via ``exc_info`` for ``WARNING`` and
    above so failures remain debuggable; lower levels stay quieter to avoid
    spamming optional-dependency fallbacks.
    """

    payload: dict[str, Any] = {
        "operation": operation,
        "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "error_message": str(exc),
    }
    if context:
        for key, value in context.items():
            if key in payload:
                continue
            payload[key] = value

    fields = " ".join(f"{key}={value!r}" for key, value in payload.items())
    logger.log(
        level,
        "operation_failed %s",
        fields,
        extra=payload,
        exc_info=level >= logging.WARNING,
    )


def log_and_wrap(
    logger: logging.Logger,
    exc: BaseException,
    *,
    operation: str,
    wrap_with: type[ExceptionT],
    message: str,
    level: int = logging.WARNING,
    context: Mapping[str, Any] | None = None,
) -> NoReturn:
    """Log a caught exception and re-raise it wrapped in a domain exception."""

    log_exception(logger, exc, operation=operation, level=level, context=context)
    raise wrap_with(message) from exc


__all__ = ["AutoTabMLError", "log_and_wrap", "log_exception"]
