"""Custom exceptions for the FLAML AutoML layer."""

from __future__ import annotations


class FlamlAutoMLError(Exception):
    """Base exception for FLAML AutoML failures."""


class FlamlConfigurationError(FlamlAutoMLError):
    """Raised when FLAML configuration is invalid."""


class FlamlDependencyError(FlamlAutoMLError):
    """Raised when the optional FLAML dependency is unavailable."""


class FlamlExecutionError(FlamlAutoMLError):
    """Raised when a FLAML operation fails."""


class FlamlTargetError(FlamlAutoMLError):
    """Raised when the selected target column cannot be used."""


class FlamlTrackingError(FlamlAutoMLError):
    """Raised when experiment tracking cannot be completed safely."""
