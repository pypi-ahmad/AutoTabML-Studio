"""App-level validation rules that run directly against pandas DataFrames.

Rules that are better expressed as Great Expectations expectations are in
gx_builders.py.  This module covers heuristics and checks that are simpler
or more flexible as plain Python.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from app.validation.schemas import CheckResult, CheckSeverity, ValidationRuleConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_app_rules(
    df: pd.DataFrame,
    config: ValidationRuleConfig,
    *,
    gx_managed_rules: bool = False,
) -> list[CheckResult]:
    """Execute all app-level validation rules and return results."""
    results: list[CheckResult] = []
    results.append(_check_not_empty(df))
    results.append(_check_min_row_count(df, config.min_row_count))
    results.extend(_check_required_columns(df, config.required_columns))
    results.extend(_check_duplicate_column_names(df))
    results.append(_check_duplicate_rows(df))
    results.extend(_check_null_percentages(df, config.null_warn_pct, config.null_fail_pct))
    results.extend(_check_fully_null_columns(df))
    results.extend(_check_constant_columns(df))
    results.extend(_check_dtype_summary(df))

    if config.target_column:
        results.extend(_check_target_exists(df, config.target_column))
        results.extend(_check_target_sanity(df, config.target_column))

    results.extend(
        _check_uniqueness(
            df,
            config.uniqueness_columns,
            prereq_only=gx_managed_rules,
        )
    )
    results.extend(
        _check_numeric_ranges(
            df,
            config.numeric_range_checks,
            prereq_only=gx_managed_rules,
        )
    )
    results.extend(
        _check_allowed_categories(
            df,
            config.allowed_category_checks,
            prereq_only=gx_managed_rules,
        )
    )
    results.extend(_check_id_columns(df, config.id_columns))

    if config.enable_leakage_heuristics and config.target_column:
        results.extend(
            _check_leakage_heuristics(
                df, config.target_column, config.leakage_cardinality_threshold
            )
        )

    return results


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _ok(name: str, msg: str, severity: CheckSeverity = CheckSeverity.INFO, **details: Any) -> CheckResult:
    return CheckResult(check_name=name, passed=True, severity=severity, message=msg, details=details, source="app")


def _fail(name: str, msg: str, severity: CheckSeverity = CheckSeverity.ERROR, **details: Any) -> CheckResult:
    return CheckResult(check_name=name, passed=False, severity=severity, message=msg, details=details, source="app")


def _check_not_empty(df: pd.DataFrame) -> CheckResult:
    if df.empty:
        return _fail("dataset_not_empty", "Dataset is empty.", CheckSeverity.ERROR)
    return _ok("dataset_not_empty", "Dataset is not empty.")


def _check_min_row_count(df: pd.DataFrame, min_rows: int) -> CheckResult:
    n = len(df)
    if n < min_rows:
        return _fail(
            "min_row_count",
            f"Row count {n} is below minimum threshold {min_rows}.",
            CheckSeverity.ERROR,
            row_count=n,
            threshold=min_rows,
        )
    return _ok("min_row_count", f"Row count {n} meets minimum threshold {min_rows}.", row_count=n)


def _check_required_columns(df: pd.DataFrame, required: list[str]) -> list[CheckResult]:
    if not required:
        return []
    existing = set(df.columns)
    results: list[CheckResult] = []
    for col in required:
        if col not in existing:
            results.append(
                _fail("required_column_present", f"Required column '{col}' is missing.", CheckSeverity.ERROR, column=col)
            )
        else:
            results.append(
                _ok("required_column_present", f"Required column '{col}' is present.", column=col)
            )
    return results


def _check_duplicate_column_names(df: pd.DataFrame) -> list[CheckResult]:
    cols = list(df.columns)
    seen: dict[str, int] = {}
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        return [_fail(
            "no_duplicate_columns",
            f"Duplicate column names found: {dupes}",
            CheckSeverity.WARNING,
            duplicates=dupes,
        )]
    return [_ok("no_duplicate_columns", "No duplicate column names.")]


def _check_duplicate_rows(df: pd.DataFrame) -> CheckResult:
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        return _fail(
            "duplicate_rows",
            f"Found {n_dupes} duplicate row(s).",
            CheckSeverity.WARNING,
            duplicate_count=n_dupes,
        )
    return _ok("duplicate_rows", "No duplicate rows found.", duplicate_count=0)


def _check_null_percentages(
    df: pd.DataFrame, warn_pct: float, fail_pct: float
) -> list[CheckResult]:
    results: list[CheckResult] = []
    if df.empty:
        return results
    for col in df.columns:
        pct = float(df[col].isna().mean() * 100)
        if pct >= fail_pct:
            results.append(
                _fail(
                    "null_percentage",
                    f"Column '{col}' is {pct:.1f}% null (threshold {fail_pct}%).",
                    CheckSeverity.ERROR,
                    column=col,
                    null_pct=round(pct, 2),
                )
            )
        elif pct >= warn_pct:
            results.append(
                _fail(
                    "null_percentage",
                    f"Column '{col}' is {pct:.1f}% null (warn threshold {warn_pct}%).",
                    CheckSeverity.WARNING,
                    column=col,
                    null_pct=round(pct, 2),
                )
            )
    return results


def _check_fully_null_columns(df: pd.DataFrame) -> list[CheckResult]:
    results: list[CheckResult] = []
    for col in df.columns:
        if df[col].isna().all():
            results.append(
                _fail("fully_null_column", f"Column '{col}' is entirely null.", CheckSeverity.ERROR, column=col)
            )
    return results


def _check_constant_columns(df: pd.DataFrame) -> list[CheckResult]:
    results: list[CheckResult] = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.nunique() == 1:
            results.append(
                _fail(
                    "constant_column",
                    f"Column '{col}' has a single unique value.",
                    CheckSeverity.WARNING,
                    column=col,
                    unique_value=str(non_null.iloc[0]),
                )
            )
    return results


def _check_dtype_summary(df: pd.DataFrame) -> list[CheckResult]:
    results: list[CheckResult] = []
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna()
            if len(sample) == 0:
                continue
            types_found = set(type(v).__name__ for v in sample.head(200))
            if len(types_found) > 1:
                results.append(
                    _fail(
                        "mixed_types",
                        f"Column '{col}' has mixed types: {types_found}.",
                        CheckSeverity.WARNING,
                        column=col,
                        types_found=sorted(types_found),
                    )
                )
    return results


def _check_target_exists(df: pd.DataFrame, target: str) -> list[CheckResult]:
    if target not in df.columns:
        return [_fail("target_column_exists", f"Target column '{target}' not found.", CheckSeverity.ERROR, column=target)]
    return [_ok("target_column_exists", f"Target column '{target}' exists.", column=target)]


def _check_target_sanity(df: pd.DataFrame, target: str) -> list[CheckResult]:
    if target not in df.columns:
        return []
    results: list[CheckResult] = []
    col = df[target]
    non_null = col.dropna()

    if non_null.empty:
        results.append(
            _fail("target_not_null", f"Target column '{target}' is entirely null.", CheckSeverity.ERROR, column=target)
        )
        return results

    nunique = non_null.nunique()

    # Classification heuristic: low-cardinality non-numeric, or int with few values
    is_likely_classification = (
        not pd.api.types.is_float_dtype(col) and nunique <= 50
    )
    if is_likely_classification and nunique == 1:
        results.append(
            _fail(
                "target_classification_sanity",
                f"Classification target '{target}' has only one class.",
                CheckSeverity.ERROR,
                column=target,
                n_unique=nunique,
            )
        )

    # Regression heuristic: numeric with many unique values
    if pd.api.types.is_numeric_dtype(col):
        if non_null.std() == 0:
            results.append(
                _fail(
                    "target_regression_sanity",
                    f"Regression target '{target}' is constant (std=0).",
                    CheckSeverity.ERROR,
                    column=target,
                )
            )

    return results


def _check_uniqueness(
    df: pd.DataFrame,
    columns: list[str],
    *,
    prereq_only: bool = False,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for col in columns:
        if col not in df.columns:
            results.append(
                _fail("uniqueness_check", f"Column '{col}' not found for uniqueness check.", CheckSeverity.WARNING, column=col)
            )
            continue
        if prereq_only:
            continue
        n_unique = df[col].nunique()
        n_total = len(df)
        if n_unique < n_total:
            results.append(
                _fail(
                    "uniqueness_check",
                    f"Column '{col}' has {n_unique}/{n_total} unique values.",
                    CheckSeverity.WARNING,
                    column=col,
                    unique_count=n_unique,
                    total_count=n_total,
                )
            )
        else:
            results.append(
                _ok("uniqueness_check", f"Column '{col}' is fully unique.", column=col)
            )
    return results


def _check_numeric_ranges(
    df: pd.DataFrame,
    range_checks: dict[str, dict[str, float]],
    *,
    prereq_only: bool = False,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for col, bounds in range_checks.items():
        if col not in df.columns:
            results.append(
                _fail(
                    "numeric_range_check",
                    f"Column '{col}' not found for numeric range check.",
                    CheckSeverity.WARNING,
                    column=col,
                )
            )
            continue

        series = df[col].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.isna().any() and not series.empty:
                results.append(
                    _fail(
                        "numeric_range_check",
                        f"Column '{col}' is not numeric and cannot be range-checked safely.",
                        CheckSeverity.WARNING,
                        column=col,
                    )
                )
                continue
            series = coerced

        if prereq_only:
            continue

        min_value = bounds.get("min")
        max_value = bounds.get("max")
        invalid_mask = pd.Series(False, index=series.index)
        if min_value is not None:
            invalid_mask = invalid_mask | (series < min_value)
        if max_value is not None:
            invalid_mask = invalid_mask | (series > max_value)
        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            results.append(
                _fail(
                    "numeric_range_check",
                    f"Column '{col}' has {invalid_count} value(s) outside the configured numeric range.",
                    CheckSeverity.ERROR,
                    column=col,
                    invalid_count=invalid_count,
                    min_value=min_value,
                    max_value=max_value,
                )
            )
        else:
            results.append(
                _ok(
                    "numeric_range_check",
                    f"Column '{col}' satisfies the configured numeric range.",
                    column=col,
                )
            )

    return results


def _check_allowed_categories(
    df: pd.DataFrame,
    allowed_checks: dict[str, list[str]],
    *,
    prereq_only: bool = False,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for col, allowed_values in allowed_checks.items():
        if col not in df.columns:
            results.append(
                _fail(
                    "allowed_category_check",
                    f"Column '{col}' not found for allowed-category check.",
                    CheckSeverity.WARNING,
                    column=col,
                )
            )
            continue

        if prereq_only:
            continue

        observed = set(df[col].dropna().astype(str).unique())
        allowed = {str(value) for value in allowed_values}
        unexpected_values = sorted(observed - allowed)

        if unexpected_values:
            results.append(
                _fail(
                    "allowed_category_check",
                    f"Column '{col}' contains values outside the allowed category set.",
                    CheckSeverity.ERROR,
                    column=col,
                    unexpected_values=unexpected_values,
                )
            )
        else:
            results.append(
                _ok(
                    "allowed_category_check",
                    f"Column '{col}' contains only allowed category values.",
                    column=col,
                )
            )

    return results


def _check_id_columns(df: pd.DataFrame, id_columns: list[str]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for col in id_columns:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique()
        n_total = len(df)
        if n_unique == n_total:
            results.append(
                _ok("id_column_unique", f"ID column '{col}' is fully unique.", CheckSeverity.INFO, column=col)
            )
        else:
            results.append(
                _fail(
                    "id_column_unique",
                    f"ID column '{col}' has {n_unique}/{n_total} unique values – may not be a true identifier.",
                    CheckSeverity.WARNING,
                    column=col,
                )
            )
    return results


def _check_leakage_heuristics(
    df: pd.DataFrame,
    target: str,
    cardinality_threshold: float,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    target_lower = target.lower()
    n_rows = len(df)
    if n_rows == 0:
        return results

    for col in df.columns:
        if col == target:
            continue
        col_lower = col.lower()

        # Name similarity
        if _names_too_similar(col_lower, target_lower):
            results.append(
                _fail(
                    "leakage_name_similarity",
                    f"Column '{col}' has a name suspiciously similar to target '{target}'.",
                    CheckSeverity.WARNING,
                    column=col,
                )
            )

        # ID-like column heuristic
        if re.search(r'(?:^|_)id(?:$|_)|(?:^|_)Id(?:$|_)', col) and df[col].nunique() == n_rows:
            results.append(
                _fail(
                    "leakage_id_like",
                    f"Column '{col}' looks like an identifier (fully unique, name contains 'id').",
                    CheckSeverity.INFO,
                    column=col,
                )
            )

        # High-cardinality string column
        if df[col].dtype == object:
            ratio = df[col].nunique() / n_rows if n_rows > 0 else 0
            if ratio >= cardinality_threshold:
                results.append(
                    _fail(
                        "leakage_high_cardinality",
                        f"String column '{col}' has very high cardinality ({ratio:.2%} unique).",
                        CheckSeverity.WARNING,
                        column=col,
                        uniqueness_ratio=round(ratio, 4),
                    )
                )

    return results


def _names_too_similar(a: str, b: str) -> bool:
    """Heuristic: names are too similar if one is a prefix/suffix variant of the other."""
    if a == b:
        return False
    # Strip common decorators
    stripped_a = re.sub(r'[_\-\s]', '', a)
    stripped_b = re.sub(r'[_\-\s]', '', b)
    if not stripped_a or not stripped_b:
        return False
    if stripped_a == stripped_b:
        return True
    if stripped_a.startswith(stripped_b) or stripped_b.startswith(stripped_a):
        return True
    return False
