"""Prediction / inference package exports."""

import importlib

from app.prediction.schemas import (
    AvailableModelReference,
    BatchPredictionRequest,
    BatchPredictionResult,
    LoadedModel,
    ModelSourceType,
    PredictionArtifactBundle,
    PredictionHistoryEntry,
    PredictionInputSourceType,
    PredictionMode,
    PredictionRequest,
    PredictionResult,
    PredictionStatus,
    PredictionSummary,
    PredictionTaskType,
    PredictionValidationResult,
    SchemaValidationMode,
    SingleRowPredictionRequest,
)

__all__ = [
    "AvailableModelReference",
    "BasePredictionService",
    "BatchPredictionRequest",
    "BatchPredictionResult",
    "LoadedModel",
    "LocalFlamlModelLoader",
    "LocalPyCaretModelLoader",
    "MLflowModelLoader",
    "ModelLoader",
    "ModelSourceType",
    "PredictionArtifactBundle",
    "PredictionHistoryEntry",
    "PredictionInputSourceType",
    "PredictionMode",
    "PredictionRequest",
    "PredictionResult",
    "PredictionService",
    "PredictionStatus",
    "PredictionSummary",
    "PredictionTaskType",
    "PredictionValidationResult",
    "SchemaValidationMode",
    "SingleRowPredictionRequest",
]


def __getattr__(name: str):
    if name in {"BasePredictionService", "PredictionService"}:
        module = importlib.import_module("app.prediction.base")

        return {
            "BasePredictionService": module.BasePredictionService,
            "PredictionService": module.PredictionService,
        }[name]

    if name in {"LocalFlamlModelLoader", "LocalPyCaretModelLoader", "MLflowModelLoader", "ModelLoader"}:
        module = importlib.import_module("app.prediction.loader")

        return {
            "LocalFlamlModelLoader": module.LocalFlamlModelLoader,
            "LocalPyCaretModelLoader": module.LocalPyCaretModelLoader,
            "MLflowModelLoader": module.MLflowModelLoader,
            "ModelLoader": module.ModelLoader,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")