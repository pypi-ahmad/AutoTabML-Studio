"""Custom exceptions for the profiling layer."""

from __future__ import annotations


class ProfilingError(Exception):
    """Base exception for profiling failures."""


class ProfilingSetupError(ProfilingError):
    """Raised when the profiling library cannot be initialized."""
