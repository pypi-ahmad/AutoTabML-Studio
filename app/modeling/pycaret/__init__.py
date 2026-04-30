"""PyCaret-backed experiment lab for AutoTabML Studio."""

import importlib

from app.modeling.pycaret.schemas import (
    CustomMetricSpec,
    ExperimentConfig,
    ExperimentResultBundle,
    ExperimentSummary,
    ExperimentTaskType,
    ModelSelectionSpec,
    SavedModelArtifact,
    SavedModelMetadata,
)

__all__ = [
    "CustomMetricSpec",
    "ExperimentConfig",
    "ExperimentResultBundle",
    "SavedModelArtifact",
    "ExperimentSummary",
    "ExperimentTaskType",
    "ModelSelectionSpec",
    "PyCaretExperimentService",
    "SavedModelMetadata",
]


def __getattr__(name: str):
    if name == "PyCaretExperimentService":
        return importlib.import_module("app.modeling.pycaret.service").PyCaretExperimentService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")