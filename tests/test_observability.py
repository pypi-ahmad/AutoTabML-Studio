"""Tests for the observability package."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading

import pytest

from app.observability import (
    Counter,
    Gauge,
    Histogram,
    InMemoryMetricsBackend,
    JsonFormatter,
    NoopMetricsBackend,
    bind_context,
    clear_context,
    configure_observability_logging,
    correlation_scope,
    current_context,
    get_metrics_backend,
    install_correlation_filter,
    new_correlation_id,
    set_metrics_backend,
    start_span,
    traced,
)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_context():
    clear_context()
    yield
    clear_context()


def test_correlation_scope_binds_and_restores():
    assert current_context() == {}
    with correlation_scope(run_id="r1", experiment_id="e1") as ctx:
        assert ctx["run_id"] == "r1"
        assert ctx["experiment_id"] == "e1"
        assert "correlation_id" in ctx
        assert current_context()["run_id"] == "r1"
    assert current_context() == {}


def test_correlation_scope_auto_generates_correlation_id():
    with correlation_scope(run_id="r"):
        cid = current_context()["correlation_id"]
    assert isinstance(cid, str)
    assert len(cid) == 32


def test_correlation_scope_nesting_overrides_then_restores():
    with correlation_scope(run_id="outer"):
        outer_cid = current_context()["correlation_id"]
        with correlation_scope(run_id="inner"):
            assert current_context()["run_id"] == "inner"
        assert current_context()["run_id"] == "outer"
        assert current_context()["correlation_id"] == outer_cid


def test_bind_context_skips_none_values():
    bind_context(run_id="r", experiment_id=None)
    ctx = current_context()
    assert ctx == {"run_id": "r"}


def test_new_correlation_id_is_unique_hex():
    a, b = new_correlation_id(), new_correlation_id()
    assert a != b
    assert all(c in "0123456789abcdef" for c in a)


def test_correlation_scope_is_thread_safe():
    results: dict[str, str] = {}

    def worker(name: str) -> None:
        with correlation_scope(run_id=name):
            results[name] = current_context()["run_id"]

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {f"t{i}": f"t{i}" for i in range(5)}


def test_correlation_scope_propagates_through_asyncio():
    async def inner() -> str:
        return current_context()["run_id"]

    async def outer() -> str:
        with correlation_scope(run_id="async-run"):
            return await inner()

    assert asyncio.run(outer()) == "async-run"


# ---------------------------------------------------------------------------
# JSON logging
# ---------------------------------------------------------------------------


def _make_record(message: str = "hello", level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name="test.logger",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_json_formatter_emits_minimal_fields():
    record = _make_record("hello world")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert "timestamp" in payload


def test_json_formatter_includes_correlation_context():
    formatter = JsonFormatter()
    record = _make_record()
    install_correlation_filter()
    with correlation_scope(run_id="r1", experiment_id="e9"):
        # Filter is attached to root; apply manually for this isolated record.
        for f in logging.getLogger().filters:
            f.filter(record)
        payload = json.loads(formatter.format(record))
    assert payload["run_id"] == "r1"
    assert payload["experiment_id"] == "e9"
    assert "correlation_id" in payload


def test_json_formatter_redacts_obvious_secrets():
    record = _make_record("token=sk-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH1234")
    payload = json.loads(JsonFormatter().format(record))
    assert "AAAABBBBCCCC" not in payload["message"]


def test_configure_observability_logging_json_mode(monkeypatch):
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_filters = list(root.filters)
    root.handlers = []
    root.filters = []
    monkeypatch.setenv("AUTOTABML_LOG_FORMAT", "json")
    try:
        configure_observability_logging(level="INFO")
        buffer = io.StringIO()
        capturing = logging.StreamHandler(buffer)
        capturing.setFormatter(JsonFormatter())
        # Re-attach the correlation filter to the new handler.
        for f in root.filters:
            capturing.addFilter(f)
        root.addHandler(capturing)

        with correlation_scope(run_id="abc"):
            logging.getLogger("obs.test").info("structured message")

        line = buffer.getvalue().strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["message"] == "structured message"
        assert payload["run_id"] == "abc"
    finally:
        for h in root.handlers:
            h.close()
        root.handlers = saved_handlers
        root.filters = saved_filters
        root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_backend():
    backend = InMemoryMetricsBackend()
    previous = set_metrics_backend(backend)
    try:
        yield backend
    finally:
        set_metrics_backend(previous)


def test_counter_increments_in_memory(memory_backend):
    Counter("requests_total").inc()
    Counter("requests_total").inc(2.5)
    key = ("requests_total", frozenset())
    assert memory_backend.counters[key] == pytest.approx(3.5)


def test_counter_attaches_correlation_labels(memory_backend):
    with correlation_scope(run_id="r-42"):
        Counter("training_runs").inc(extra="x")
    key, value = next(iter(memory_backend.counters.items()))
    assert key[0] == "training_runs"
    label_dict = dict(key[1])
    assert label_dict["run_id"] == "r-42"
    assert label_dict["extra"] == "x"
    assert "correlation_id" in label_dict
    assert value == 1.0


def test_gauge_tracks_last_value(memory_backend):
    g = Gauge("queue_depth")
    g.set(5)
    g.set(2)
    assert memory_backend.gauges[("queue_depth", frozenset())] == 2.0


def test_histogram_records_observations(memory_backend):
    h = Histogram("op_duration_seconds")
    h.observe(0.1)
    h.observe(0.2)
    key = ("op_duration_seconds", frozenset())
    assert memory_backend.histograms[key] == [0.1, 0.2]


def test_histogram_time_context_records_elapsed(memory_backend):
    h = Histogram("op_duration_seconds")
    with h.time():
        pass
    key = ("op_duration_seconds", frozenset())
    samples = memory_backend.histograms[key]
    assert len(samples) == 1
    assert samples[0] >= 0.0


def test_set_metrics_backend_returns_previous(memory_backend):
    new = NoopMetricsBackend()
    previous = set_metrics_backend(new)
    try:
        assert previous is memory_backend
        assert get_metrics_backend() is new
    finally:
        set_metrics_backend(memory_backend)


def test_noop_backend_swallows_observations():
    backend = NoopMetricsBackend()
    backend.incr("x", 1, {})
    backend.observe("x", 1, {})
    backend.gauge("x", 1, {})


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def test_start_span_yields_span_like_object():
    with start_span("unit.op", attr="value") as span:
        # In the no-op path these are silent; in the OTel path they succeed.
        span.set_attribute("k", "v")
        span.record_exception(RuntimeError("ignored"))


def test_traced_decorator_invokes_wrapped_function():
    calls: list[int] = []

    @traced("unit.fn")
    def fn(x: int) -> int:
        calls.append(x)
        return x * 2

    assert fn(3) == 6
    assert calls == [3]


def test_traced_decorator_propagates_exceptions():
    @traced()
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        boom()
