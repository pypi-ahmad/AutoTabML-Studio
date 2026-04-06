"""Custom exceptions for the PyCaret experiment layer."""

from __future__ import annotations


class PyCaretExperimentError(Exception):
    """Base exception for experiment failures."""


class PyCaretConfigurationError(PyCaretExperimentError):
    """Raised when experiment configuration is invalid."""


class PyCaretDependencyError(PyCaretExperimentError):
    """Raised when the optional PyCaret dependency is unavailable."""


class PyCaretExecutionError(PyCaretExperimentError):
    """Raised when a PyCaret operation fails."""


class PyCaretTargetError(PyCaretExperimentError):
    """Raised when the selected target column cannot be used."""


class UnsupportedExperimentTaskError(PyCaretExperimentError):
    """Raised when the requested task type is unsupported."""


class PyCaretTrackingError(PyCaretExperimentError):
    """Raised when experiment tracking cannot be completed safely."""