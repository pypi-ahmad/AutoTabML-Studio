"""Abstract execution backend interface."""

from __future__ import annotations

import abc
from typing import Any

from app.config.enums import ExecutionBackend


class BaseExecutionBackend(abc.ABC):
    """Common interface for all execution backends."""

    backend_type: ExecutionBackend

    @abc.abstractmethod
    async def validate_backend(self) -> bool:
        """Check that the backend is reachable / properly configured."""

    @abc.abstractmethod
    async def prepare_session(self) -> dict[str, Any]:
        """Prepare any runtime context needed before a job runs."""

    @abc.abstractmethod
    async def run_job(self, job_payload: dict[str, Any]) -> dict[str, Any]:
        """Execute a job payload and return results.

        Concrete implementations will be added in a future step when the
        training / benchmark pipeline is built.
        """
