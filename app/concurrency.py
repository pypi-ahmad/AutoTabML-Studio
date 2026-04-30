"""Small async concurrency helpers used across the app.

These helpers exist so callers can parallelize independent operations without
each module re-implementing semaphore/`asyncio.gather` boilerplate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


async def gather_with_concurrency(
    coros: Iterable[Awaitable[T]],
    *,
    limit: int,
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """Run awaitables concurrently with at most ``limit`` running at a time.

    ``limit`` must be >= 1. Each input awaitable is awaited inside a semaphore
    slot so memory/connection pressure stays bounded even with thousands of
    inputs.
    """

    if limit < 1:
        raise ValueError("limit must be >= 1")

    semaphore = asyncio.Semaphore(limit)

    async def _run(awaitable: Awaitable[T]) -> T:
        async with semaphore:
            return await awaitable

    return await asyncio.gather(
        *(_run(c) for c in coros),
        return_exceptions=return_exceptions,
    )


async def to_thread_many(
    func: Callable[..., T],
    arg_batches: Iterable[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    limit: int = 8,
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """Run a blocking ``func`` over many argument batches concurrently in threads.

    Each entry in ``arg_batches`` is a ``(args, kwargs)`` pair handed to
    ``asyncio.to_thread``. ``limit`` controls how many threads run at once.
    """

    coros = [asyncio.to_thread(func, *args, **kwargs) for args, kwargs in arg_batches]
    return await gather_with_concurrency(
        coros, limit=limit, return_exceptions=return_exceptions
    )


__all__ = ["gather_with_concurrency", "to_thread_many"]
