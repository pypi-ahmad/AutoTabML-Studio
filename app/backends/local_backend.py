"""Local execution backend – runs ML jobs on the user's machine."""

from __future__ import annotations

import logging
from typing import Any

from app.backends.base import BaseExecutionBackend
from app.config.enums import ExecutionBackend

logger = logging.getLogger(__name__)


class LocalExecutionBackend(BaseExecutionBackend):
    backend_type = ExecutionBackend.LOCAL

    async def validate_backend(self) -> bool:
        # Local always available
        return True

    async def prepare_session(self) -> dict[str, Any]:
        logger.info("Local execution session ready.")
        return {"backend": "local", "status": "ready"}

    async def run_job(self, job_payload: dict[str, Any]) -> dict[str, Any]:
        # TODO(local-pipeline): Wire in actual local training/benchmark pipeline.
        # Blocked until a unified job-dispatch layer is designed.  The existing
        # CLI-driven validation/profile/benchmark/experiment paths work without
        # this backend method; it is only needed for future notebook-style or
        # Streamlit-triggered async job execution.
        raise NotImplementedError("Local async job execution is not yet implemented.")
