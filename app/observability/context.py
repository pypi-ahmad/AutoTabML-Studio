"""Correlation context for log/metric/trace records.

A small wrapper around :class:`contextvars.ContextVar` so that all
observability layers can read the same set of correlation keys without
threading them through every function signature.

Usage
-----

::

    from app.observability import correlation_scope, bind_context

    with correlation_scope(run_id="abc123", experiment_id="exp-7"):
        run_training()

    # Or bind ad-hoc inside an existing scope:
    bind_context(dataset_id="kaggle:titanic")

The context is read by:

* the JSON log formatter (auto-injects every key),
* :func:`app.observability.metrics.Counter.inc` (auto-attaches as labels when
  the backend supports it),
* :func:`app.observability.tracing.start_span` (sets matching span attributes).
"""

from __future__ import annotations

import contextlib
import uuid
from contextvars import ContextVar, Token
from typing import Any, Iterator, Mapping

# Single ContextVar holding an immutable mapping. Using a single var (rather
# than one per key) keeps copy-on-write semantics cheap and means a single
# `.set()` call atomically swaps the entire correlation snapshot.
_CONTEXT: ContextVar[Mapping[str, Any]] = ContextVar(
    "autotabml_observability_context",
    default={},
)


def current_context() -> dict[str, Any]:
    """Return a *copy* of the current correlation context mapping."""

    return dict(_CONTEXT.get())


def bind_context(**fields: Any) -> Token[Mapping[str, Any]]:
    """Merge ``fields`` into the current context and return a reset token.

    Callers that want lexical scoping should prefer :func:`correlation_scope`;
    :func:`bind_context` is exposed for cases where the bind site and reset
    site cannot live in the same function (for example, a Streamlit page
    handler that binds at the top and lets the run scope unwind naturally).
    """

    current = dict(_CONTEXT.get())
    current.update({k: v for k, v in fields.items() if v is not None})
    return _CONTEXT.set(current)


def clear_context() -> None:
    """Reset the correlation context to an empty mapping."""

    _CONTEXT.set({})


def new_correlation_id() -> str:
    """Generate a fresh correlation/request id (uuid4 hex, 32 chars)."""

    return uuid.uuid4().hex


@contextlib.contextmanager
def correlation_scope(**fields: Any) -> Iterator[dict[str, Any]]:
    """Bind ``fields`` for the lifetime of the ``with`` block.

    A new ``correlation_id`` is auto-generated when not provided, so that any
    log/metric/trace emitted inside the scope can be cross-referenced even if
    the caller did not explicitly pass one.
    """

    payload = {k: v for k, v in fields.items() if v is not None}
    payload.setdefault("correlation_id", new_correlation_id())
    merged = {**_CONTEXT.get(), **payload}
    token = _CONTEXT.set(merged)
    try:
        yield dict(merged)
    finally:
        _CONTEXT.reset(token)
