"""Optional OpenTelemetry tracing with a stdlib-only fallback.

We deliberately do not declare ``opentelemetry-api`` as a hard dependency:
the call sites use :func:`start_span` and :func:`traced` unconditionally, and
this module decides at import time whether real spans or no-op spans are
emitted. Adopting OTel later is a single ``pip install`` away – no code
changes required.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Iterator, Protocol, TypeVar

from app.observability.context import current_context

F = TypeVar("F", bound=Callable[..., Any])


class SpanLike(Protocol):
    """The minimal span surface used by AutoTabML Studio call sites."""

    def set_attribute(self, key: str, value: Any) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        return None

    def record_exception(self, exception: BaseException) -> None:  # noqa: D401
        return None


try:  # pragma: no cover - exercised only when opentelemetry is installed
    from opentelemetry import trace as _otel_trace

    _TRACER = _otel_trace.get_tracer("autotabml-studio")
    _OTEL_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import-time failure must fall back cleanly
    _TRACER = None
    _OTEL_AVAILABLE = False


@contextlib.contextmanager
def start_span(name: str, **attributes: Any) -> Iterator[SpanLike]:
    """Open a span named ``name`` and yield a :class:`SpanLike`.

    When OpenTelemetry is installed this delegates to the global tracer and
    eagerly attaches the active correlation context as span attributes so
    that traces and logs share the same join keys. Otherwise it yields a
    no-op object that satisfies the protocol – callers can therefore use
    ``start_span`` unconditionally.
    """

    if not _OTEL_AVAILABLE or _TRACER is None:
        yield _NoopSpan()
        return

    with _TRACER.start_as_current_span(name) as span:  # pragma: no cover - import-gated
        for key, value in current_context().items():
            try:
                span.set_attribute(key, value)
            except Exception:  # noqa: BLE001 - never let telemetry break the caller
                pass
        for key, value in attributes.items():
            if value is None:
                continue
            try:
                span.set_attribute(key, value)
            except Exception:  # noqa: BLE001
                pass
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001 - re-raised after recording
            try:
                span.record_exception(exc)
            except Exception:  # noqa: BLE001
                pass
            raise


def traced(name: str | None = None) -> Callable[[F], F]:
    """Decorator that wraps a function call in :func:`start_span`."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with start_span(span_name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
