"""Automatic mode selection and DataFrame sampling for profiling."""

from __future__ import annotations

import logging

import pandas as pd

from app.config.models import ProfilingMode
from app.profiling.schemas import ProfilingConfig

logger = logging.getLogger(__name__)


def select_profiling_mode(
    df: pd.DataFrame,
    config: ProfilingConfig,
) -> ProfilingMode:
    """Choose profiling mode based on dataset size and config thresholds."""
    n_rows, n_cols = df.shape
    if (
        n_rows > config.large_dataset_row_threshold
        or n_cols > config.large_dataset_col_threshold
    ):
        logger.info(
            "Dataset size (%d rows, %d cols) exceeds threshold – using minimal mode.",
            n_rows,
            n_cols,
        )
        return ProfilingMode.MINIMAL
    return config.mode


def maybe_sample(
    df: pd.DataFrame,
    config: ProfilingConfig,
) -> tuple[pd.DataFrame, bool, int | None]:
    """Return (possibly sampled df, was_sampled, sample_size_used).

    Sampling is applied only if the row count exceeds the sampling threshold.
    """
    n_rows = len(df)
    if n_rows > config.sampling_row_threshold:
        sample_size = min(config.sample_size, n_rows)
        logger.info(
            "Sampling %d rows from %d for profiling.", sample_size, n_rows
        )
        sampled = df.sample(n=sample_size, random_state=42)
        return sampled, True, sample_size
    return df, False, None
