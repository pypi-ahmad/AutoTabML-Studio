"""Base abstraction for profiling services."""

from __future__ import annotations

import abc

import pandas as pd

from app.profiling.schemas import ProfilingArtifactBundle, ProfilingConfig, ProfilingResultSummary


class BaseProfilingService(abc.ABC):
    """Interface that all profiling service implementations must satisfy."""

    @abc.abstractmethod
    def profile(
        self,
        df: pd.DataFrame,
        config: ProfilingConfig,
        *,
        dataset_name: str | None = None,
    ) -> tuple[ProfilingResultSummary, ProfilingArtifactBundle | None]:
        """Run profiling and return summary + optional artifacts."""
