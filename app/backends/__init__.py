"""Execution backend package – factory for backend instances."""

from __future__ import annotations

from app.backends.base import BaseExecutionBackend
from app.backends.colab_mcp_backend import ColabMCPExecutionBackend
from app.backends.local_backend import LocalExecutionBackend
from app.config.enums import ExecutionBackend

__all__ = [
    "BaseExecutionBackend",
    "ColabMCPExecutionBackend",
    "LocalExecutionBackend",
    "build_backend",
]


def build_backend(backend: ExecutionBackend) -> BaseExecutionBackend:
    """Create the execution backend instance for the given enum value."""
    if backend == ExecutionBackend.COLAB_MCP:
        return ColabMCPExecutionBackend()
    if backend == ExecutionBackend.LOCAL:
        return LocalExecutionBackend()
    raise ValueError(f"Unknown execution backend: {backend!r}")
