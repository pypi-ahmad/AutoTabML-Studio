"""Profiling artifact generation helpers."""

from __future__ import annotations

from app.profiling.schemas import ProfilingArtifactBundle

# Artifact writing is handled inside ydata_runner.YDataProfilingService._write_artifacts
# This module exists as the expected counterpart to validation/artifacts.py
# and can be extended if additional artifact formats are needed later.

__all__ = ["ProfilingArtifactBundle"]
