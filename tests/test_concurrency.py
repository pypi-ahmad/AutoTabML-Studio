"""Tests for the async concurrency helpers."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.concurrency import gather_with_concurrency, to_thread_many


class TestGatherWithConcurrency:
    @pytest.mark.asyncio
    async def test_returns_results_in_input_order(self) -> None:
        async def _produce(value: int, delay: float) -> int:
            await asyncio.sleep(delay)
            return value

        results = await gather_with_concurrency(
            [_produce(0, 0.02), _produce(1, 0.01), _produce(2, 0.0)],
            limit=3,
        )

        assert results == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_caps_concurrent_executions(self) -> None:
        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def _task() -> None:
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1

        await gather_with_concurrency([_task() for _ in range(10)], limit=3)

        assert peak <= 3

    @pytest.mark.asyncio
    async def test_runs_in_parallel(self) -> None:
        async def _slow() -> int:
            await asyncio.sleep(0.05)
            return 1

        start = time.perf_counter()
        results = await gather_with_concurrency([_slow() for _ in range(5)], limit=5)
        elapsed = time.perf_counter() - start

        assert sum(results) == 5
        # Sequential would be ~0.25s; parallel should be far below 0.20s.
        assert elapsed < 0.20

    @pytest.mark.asyncio
    async def test_return_exceptions_collects_failures(self) -> None:
        async def _ok() -> int:
            return 42

        async def _boom() -> int:
            raise RuntimeError("nope")

        results = await gather_with_concurrency(
            [_ok(), _boom(), _ok()], limit=2, return_exceptions=True
        )

        assert results[0] == 42
        assert isinstance(results[1], RuntimeError)
        assert results[2] == 42

    @pytest.mark.asyncio
    async def test_rejects_non_positive_limit(self) -> None:
        with pytest.raises(ValueError):
            await gather_with_concurrency([], limit=0)


class TestToThreadMany:
    @pytest.mark.asyncio
    async def test_runs_blocking_function_concurrently(self) -> None:
        def _square(x: int) -> int:
            time.sleep(0.02)
            return x * x

        batches = [((i,), {}) for i in range(5)]
        start = time.perf_counter()
        results = await to_thread_many(_square, batches, limit=5)
        elapsed = time.perf_counter() - start

        assert results == [0, 1, 4, 9, 16]
        # Sequential would be ~0.10s; parallel should be well under.
        assert elapsed < 0.08

    @pytest.mark.asyncio
    async def test_supports_kwargs(self) -> None:
        def _add(a: int, b: int = 0) -> int:
            return a + b

        results = await to_thread_many(
            _add,
            [((1,), {"b": 10}), ((2,), {"b": 20})],
            limit=2,
        )

        assert results == [11, 22]

    @pytest.mark.asyncio
    async def test_return_exceptions(self) -> None:
        def _maybe(x: int) -> int:
            if x == 1:
                raise ValueError("bad")
            return x

        results = await to_thread_many(
            _maybe,
            [((0,), {}), ((1,), {}), ((2,), {})],
            limit=3,
            return_exceptions=True,
        )

        assert results[0] == 0
        assert isinstance(results[1], ValueError)
        assert results[2] == 2
