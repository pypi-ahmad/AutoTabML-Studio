"""Normalization utilities for tabular datasets."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from app.ingestion.errors import EmptyDatasetError, IngestionError


def normalize_duplicate_column_names(columns: Iterable[object]) -> tuple[list[object], dict[str, str]]:
    """Make duplicate column names unique while preserving order.

    The first occurrence keeps its original label. Subsequent duplicates are
    renamed using the ``__N`` suffix.
    """

    new_columns: list[object] = []
    seen_labels: set[str] = set()
    rename_map: dict[str, str] = {}

    for original in columns:
        base_label = str(original)
        candidate = base_label
        suffix = 2

        while candidate in seen_labels:
            candidate = f"{base_label}__{suffix}"
            suffix += 1

        seen_labels.add(candidate)
        if candidate == base_label and not isinstance(original, str):
            new_columns.append(original)
            continue

        new_columns.append(candidate)
        if candidate != original:
            rename_map[str(original)] = candidate

    return new_columns, rename_map


def normalize_to_pandas(
    dataframe: pd.DataFrame,
    *,
    drop_fully_empty: bool = True,
    dedupe_columns: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize a raw tabular object to the canonical pandas DataFrame contract."""

    if not isinstance(dataframe, pd.DataFrame):
        raise IngestionError("normalize_to_pandas expects a pandas DataFrame.")

    normalized = dataframe.copy(deep=True)
    actions: list[str] = []

    if drop_fully_empty:
        row_count_before = len(normalized)
        normalized = normalized.dropna(axis=0, how="all")
        dropped_rows = row_count_before - len(normalized)
        if dropped_rows:
            actions.append(f"Dropped {dropped_rows} fully empty row(s).")

        column_count_before = len(normalized.columns)
        normalized = normalized.dropna(axis=1, how="all")
        dropped_columns = column_count_before - len(normalized.columns)
        if dropped_columns:
            actions.append(f"Dropped {dropped_columns} fully empty column(s).")

    if dedupe_columns and len(normalized.columns) != len(set(map(str, normalized.columns))):
        deduped_columns, rename_map = normalize_duplicate_column_names(normalized.columns)
        normalized.columns = deduped_columns
        actions.append(
            "Normalized duplicate column names: "
            + ", ".join(f"{source} -> {target}" for source, target in rename_map.items())
        )

    if normalized.empty or normalized.shape[1] == 0:
        raise EmptyDatasetError(
            "The dataset is empty after safe normalization. Check the source file or preview settings."
        )

    return normalized, actions
