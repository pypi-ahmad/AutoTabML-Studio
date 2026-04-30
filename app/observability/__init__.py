"""Production-grade observability primitives for AutoTabML Studio.

This package provides four building blocks that compose into a single
12-factor friendly observability surface:

* :mod:`app.observability.context` – correlation context (run id, experiment id,
  request id) propagated via :class:`contextvars.ContextVar` so it survives
  async hops and thread pools.
* :mod:`app.observability.logging_setup` – structured JSON logging, a
  correlation-injecting :class:`logging.Filter`, and a secret-redacting
  formatter shared with the legacy text formatter.
* :mod:`app.observability.metrics` – a tiny pluggable metrics façade
  (counter / gauge / histogram) that defaults to an in-memory registry and
  can be swapped for Prometheus / StatsD / OpenTelemetry exporters by
  calling :func:`set_metrics_backend`.
* :mod:`app.observability.tracing` – a no-op tracing API that automatically
  upgrades to real OpenTelemetry spans when the ``opentelemetry-api`` package
  is importable. Safe to use unconditionally throughout the codebase.

The public surface is intentionally small; nothing here is required for the
app to function – everything degrades to a no-op when not configured.
"""

from __future__ import annotations

from app.observability.context import (
    bind_context,
    clear_context,
    correlation_scope,
    current_context,
    new_correlation_id,
)
from app.observability.logging_setup import (
    JsonFormatter,
    configure_observability_logging,
    install_correlation_filter,
)
from app.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    InMemoryMetricsBackend,
    MetricsBackend,
    NoopMetricsBackend,
    get_metrics_backend,
    set_metrics_backend,
)
from app.observability.tracing import (
    SpanLike,
    start_span,
    traced,
)

__all__ = [
    # context
    "bind_context",
    "clear_context",
    "correlation_scope",
    "current_context",
    "new_correlation_id",
    # logging
    "JsonFormatter",
    "configure_observability_logging",
    "install_correlation_filter",
    # metrics
    "Counter",
    "Gauge",
    "Histogram",
    "InMemoryMetricsBackend",
    "MetricsBackend",
    "NoopMetricsBackend",
    "get_metrics_backend",
    "set_metrics_backend",
    # tracing
    "SpanLike",
    "start_span",
    "traced",
]
