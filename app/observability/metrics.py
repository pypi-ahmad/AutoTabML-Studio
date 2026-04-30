"""Pluggable metrics façade.

Three primitive types are exposed – :class:`Counter`, :class:`Gauge`,
:class:`Histogram` – each backed by a swappable :class:`MetricsBackend`.
The default backend is :class:`InMemoryMetricsBackend`, which is fast,
introspectable from tests, and zero-dep. Production deployments can call
:func:`set_metrics_backend` once at startup with a Prometheus / StatsD /
OpenTelemetry adapter without touching any caller.

Design notes
------------

* The façade is intentionally synchronous. Backends are expected to be
  non-blocking (e.g. counter increments in a local map) or to fan out work
  to a background thread / queue internally.
* Labels passed to :meth:`Counter.inc` and friends are merged with the
  active correlation context so that a single ``run_id=...`` bind at the
  top of a workflow propagates to every metric emitted underneath. Backends
  that cannot accept arbitrary labels should drop unknown keys silently.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Protocol

from app.observability.context import current_context

LabelMap = Mapping[str, Any]


def _merge_labels(labels: LabelMap | None) -> dict[str, Any]:
    merged: dict[str, Any] = dict(current_context())
    if labels:
        merged.update(labels)
    # Stringify values so backends with strict label-type expectations
    # (Prometheus, StatsD) do not blow up on stray ints / UUIDs.
    return {k: str(v) for k, v in merged.items() if v is not None}


class MetricsBackend(Protocol):
    """Strategy interface implemented by metric exporters."""

    def incr(self, name: str, value: float, labels: LabelMap) -> None: ...
    def observe(self, name: str, value: float, labels: LabelMap) -> None: ...
    def gauge(self, name: str, value: float, labels: LabelMap) -> None: ...


class NoopMetricsBackend:
    """Backend that discards every observation. Used in tests / scripts."""

    def incr(self, name: str, value: float, labels: LabelMap) -> None:  # noqa: D401
        return None

    def observe(self, name: str, value: float, labels: LabelMap) -> None:  # noqa: D401
        return None

    def gauge(self, name: str, value: float, labels: LabelMap) -> None:  # noqa: D401
        return None


class InMemoryMetricsBackend:
    """Thread-safe backend that retains every observation in memory.

    Intended for unit tests, batch CLI runs, and as a default during early
    development. The structure is:

    * counters: ``{(name, frozenset(labels.items())): float}``
    * gauges:   ``{(name, frozenset(labels.items())): float}``
    * histograms: ``{(name, frozenset(labels.items())): list[float]}``
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, frozenset[tuple[str, str]]], float] = defaultdict(float)
        self.gauges: dict[tuple[str, frozenset[tuple[str, str]]], float] = {}
        self.histograms: dict[tuple[str, frozenset[tuple[str, str]]], list[float]] = defaultdict(list)

    @staticmethod
    def _key(name: str, labels: LabelMap) -> tuple[str, frozenset[tuple[str, str]]]:
        return (name, frozenset((k, str(v)) for k, v in labels.items()))

    def incr(self, name: str, value: float, labels: LabelMap) -> None:
        with self._lock:
            self.counters[self._key(name, labels)] += float(value)

    def observe(self, name: str, value: float, labels: LabelMap) -> None:
        with self._lock:
            self.histograms[self._key(name, labels)].append(float(value))

    def gauge(self, name: str, value: float, labels: LabelMap) -> None:
        with self._lock:
            self.gauges[self._key(name, labels)] = float(value)

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()


_backend_lock = threading.Lock()
_backend: MetricsBackend = InMemoryMetricsBackend()


def set_metrics_backend(backend: MetricsBackend) -> MetricsBackend:
    """Install ``backend`` as the active sink and return the previous one."""

    global _backend
    with _backend_lock:
        previous, _backend = _backend, backend
    return previous


def get_metrics_backend() -> MetricsBackend:
    """Return the currently installed metrics backend."""

    return _backend


class _Metric:
    """Shared base – holds a metric name and forwards to the active backend."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class Counter(_Metric):
    """Monotonically increasing counter."""

    def inc(self, value: float = 1.0, **labels: Any) -> None:
        _backend.incr(self.name, value, _merge_labels(labels))


class Gauge(_Metric):
    """Point-in-time numeric value."""

    def set(self, value: float, **labels: Any) -> None:
        _backend.gauge(self.name, value, _merge_labels(labels))


class Histogram(_Metric):
    """Distribution of observed values (e.g. durations in seconds)."""

    def observe(self, value: float, **labels: Any) -> None:
        _backend.observe(self.name, value, _merge_labels(labels))

    def time(self) -> _Timer:
        """Return a context manager that records elapsed seconds on exit."""

        return _Timer(self)


class _Timer:
    """Helper returned by :meth:`Histogram.time`."""

    __slots__ = ("_histogram", "_start")

    def __init__(self, histogram: Histogram) -> None:
        self._histogram = histogram
        self._start = 0.0

    def __enter__(self) -> _Timer:
        import time

        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        import time

        self._histogram.observe(time.perf_counter() - self._start)
