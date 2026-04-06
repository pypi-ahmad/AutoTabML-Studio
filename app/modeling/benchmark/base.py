"""Base abstractions for benchmark services."""

from __future__ import annotations

import abc

import pandas as pd

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkResultBundle


class BaseBenchmarkService(abc.ABC):
    """Interface implemented by all benchmark service backends."""

    @abc.abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        config: BenchmarkConfig,
        *,
        dataset_name: str | None = None,
        dataset_fingerprint: str | None = None,
        execution_backend: ExecutionBackend = ExecutionBackend.LOCAL,
        workspace_mode: WorkspaceMode | None = None,
    ) -> BenchmarkResultBundle:
        """Run a benchmark and return a structured result bundle."""