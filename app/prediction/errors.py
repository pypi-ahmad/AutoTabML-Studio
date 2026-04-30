"""Prediction-layer errors for inference workflows."""

from __future__ import annotations

from app.security.errors import TrustedArtifactError as BaseTrustedArtifactError


class PredictionError(Exception):
    """Base class for prediction-layer failures."""


class ModelDiscoveryError(PredictionError):
    """Raised when a model source cannot be discovered or resolved cleanly."""


class ModelLoadError(PredictionError):
    """Raised when a prediction model cannot be loaded."""


class TrustedArtifactError(BaseTrustedArtifactError, ModelLoadError):
    """Raised when a model artifact fails trust-boundary validation."""


class PredictionValidationError(PredictionError):
    """Raised when prediction-time input validation fails."""


class PredictionScoringError(PredictionError):
    """Raised when a loaded model cannot score the provided data."""


class PredictionArtifactError(PredictionError):
    """Raised when prediction artifacts cannot be written."""


class PredictionHistoryError(PredictionError):
    """Raised when prediction history cannot be persisted or queried."""