"""Custom exceptions for the model registry layer."""

from __future__ import annotations


class RegistryError(Exception):
    """Base exception for registry failures."""


class RegistryUnavailableError(RegistryError):
    """Raised when the MLflow model registry backend is not available."""


class ModelNotFoundError(RegistryError):
    """Raised when a registered model cannot be found."""


class VersionNotFoundError(RegistryError):
    """Raised when a model version cannot be found."""


class PromotionError(RegistryError):
    """Raised when a promotion action cannot be completed."""
