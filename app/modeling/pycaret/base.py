"""Base abstractions for experiment services."""

from __future__ import annotations

import abc

import pandas as pd

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.modeling.pycaret.schemas import ExperimentConfig, ExperimentResultBundle, ModelSelectionSpec


class BaseExperimentService(abc.ABC):
    """Interface implemented by all experiment backends."""

    @abc.abstractmethod
    def setup_experiment(
        self,
        df: pd.DataFrame,
        config: ExperimentConfig,
        *,
        test_df: pd.DataFrame | None = None,
        dataset_name: str | None = None,
        dataset_fingerprint: str | None = None,
        execution_backend: ExecutionBackend = ExecutionBackend.LOCAL,
        workspace_mode: WorkspaceMode | None = None,
    ) -> ExperimentResultBundle:
        """Create the experiment context and return an initial bundle."""

    @abc.abstractmethod
    def compare_models(self, bundle: ExperimentResultBundle) -> ExperimentResultBundle:
        """Run compare_models for an existing bundle."""

    @abc.abstractmethod
    def tune_model(
        self,
        bundle: ExperimentResultBundle,
        selection: ModelSelectionSpec,
    ) -> ExperimentResultBundle:
        """Tune one selected model for an existing bundle."""