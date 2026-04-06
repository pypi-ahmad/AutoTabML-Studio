"""Custom exceptions for the tracking and history layer."""

from __future__ import annotations


class TrackingError(Exception):
    """Base exception for tracking failures."""


class TrackingUnavailableError(TrackingError):
    """Raised when MLflow tracking is not available or configured."""


class RunNotFoundError(TrackingError):
    """Raised when a requested run cannot be found."""


class ExperimentNotFoundError(TrackingError):
    """Raised when a requested MLflow experiment cannot be found."""


class ComparisonError(TrackingError):
    """Raised when a comparison cannot be completed."""
