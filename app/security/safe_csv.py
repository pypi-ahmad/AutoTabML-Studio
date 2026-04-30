"""Safe CSV export helpers.

Excel and similar spreadsheet tools can evaluate cells beginning with formula
prefixes like ``=``, ``+``, ``-``, and ``@``. These helpers neutralize those
values before exporting them to CSV.
"""

from __future__ import annotations

import csv
from typing import Any

import pandas as pd


DANGEROUS_CSV_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@")


def _sanitize_csv_scalar(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_sanitize_csv_scalar(item) for item in value)
    if not isinstance(value, str):
        return value

    stripped = value.lstrip(" \t\r\n")
    if stripped.startswith(DANGEROUS_CSV_PREFIXES):
        return "'" + value
    return value


def sanitize_csv_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy safe for CSV export.

    Sanitizes string cells, column labels, index labels, and axis names so no
    exported spreadsheet cell begins with a dangerous formula prefix.
    """

    sanitized = dataframe.apply(lambda column: column.map(_sanitize_csv_scalar))
    sanitized = sanitized.rename(index=_sanitize_csv_scalar, columns=_sanitize_csv_scalar)

    if isinstance(sanitized.index, pd.MultiIndex):
        sanitized.index = sanitized.index.set_names(
            [_sanitize_csv_scalar(name) for name in sanitized.index.names],
        )
    else:
        sanitized.index.name = _sanitize_csv_scalar(sanitized.index.name)

    if isinstance(sanitized.columns, pd.MultiIndex):
        sanitized.columns = sanitized.columns.set_names(
            [_sanitize_csv_scalar(name) for name in sanitized.columns.names],
        )
    else:
        sanitized.columns.name = _sanitize_csv_scalar(sanitized.columns.name)

    return sanitized


def dataframe_to_safe_csv(dataframe: pd.DataFrame, *, index: bool = False) -> str:
    """Serialize ``dataframe`` to CSV with formula-safe cells and strict quoting."""

    sanitized = sanitize_csv_dataframe(dataframe)
    return sanitized.to_csv(
        index=index,
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )