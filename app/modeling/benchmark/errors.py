"""Custom exceptions for the benchmark layer."""

from __future__ import annotations


class BenchmarkError(Exception):
    """Base exception for benchmark failures."""


class BenchmarkConfigurationError(BenchmarkError):
    """Raised when a benchmark configuration is invalid."""


class BenchmarkDependencyError(BenchmarkError):
    """Raised when an optional runtime dependency is unavailable."""


class BenchmarkTargetError(BenchmarkError):
    """Raised when the selected benchmark target is invalid."""


class BenchmarkExecutionError(BenchmarkError):
    """Raised when benchmark execution fails."""


class UnsupportedBenchmarkTaskError(BenchmarkError):
    """Raised when the requested benchmark task type is unsupported."""


class BenchmarkTrackingError(BenchmarkError):
    """Raised when benchmark tracking cannot be completed safely."""