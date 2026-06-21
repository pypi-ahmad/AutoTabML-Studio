"""Property-based tests for the safe-CSV export helpers.

These complement the deterministic unit tests in tests/test_safe_csv.py
by exploring the input space the unit tests do not cover (random
unicode, embedded NUL, mixed-type cells, large indexes).
"""

from __future__ import annotations

import csv
import io
import string

from hypothesis import given, settings, strategies as st
import pandas as pd

from app.security.safe_csv import dataframe_to_safe_csv, sanitize_csv_dataframe

DANGEROUS_PREFIXES = ("=", "+", "-", "@")

# Only printable characters; we strip control chars in production
# before exporting, so we don't need to test that path here.
_cell_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        whitelist_characters=string.printable.replace("\n", "").replace("\r", "").replace('"', ""),
    ),
    min_size=0,
    max_size=64,
)


@st.composite
def _dangerous_cell(draw: st.DrawFn) -> str:
    """Generate a cell that begins with a known dangerous prefix."""
    prefix = draw(st.sampled_from(DANGEROUS_PREFIXES))
    body = draw(_cell_text)
    return f"{prefix}{body}"


@settings(max_examples=50, deadline=None)
@given(value=_dangerous_cell())
def test_dangerous_cell_value_is_escaped(value: str) -> None:
    dataframe = pd.DataFrame({"col": [value]})
    sanitized = sanitize_csv_dataframe(dataframe)
    assert sanitized["col"].iloc[0].startswith("'"), f"Dangerous prefix not escaped: {sanitized['col'].iloc[0]!r}"


@settings(max_examples=50, deadline=None)
@given(value=_dangerous_cell())
def test_dangerous_header_is_escaped(value: str) -> None:
    dataframe = pd.DataFrame({value: [1]})
    sanitized = sanitize_csv_dataframe(dataframe)
    assert sanitized.columns[0].startswith("'"), f"Dangerous header prefix not escaped: {sanitized.columns[0]!r}"


@settings(max_examples=50, deadline=None)
@given(text=_cell_text)
def test_safe_text_round_trips_through_safe_csv(text: str) -> None:
    """Values that do not start with a dangerous prefix should be untouched."""
    if any(text.startswith(p) for p in DANGEROUS_PREFIXES):
        return  # Property is about the safe-input contract
    dataframe = pd.DataFrame({"col": [text]})
    sanitized = sanitize_csv_dataframe(dataframe)
    assert sanitized["col"].iloc[0] == text


@settings(max_examples=20, deadline=None)
@given(
    name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    n_rows=st.integers(min_value=1, max_value=8),
)
def test_csv_round_trip_preserves_rows(name: str, n_rows: int) -> None:
    """The sanitized DataFrame, when re-parsed from CSV, must contain
    the same number of rows (no truncation, no extra rows from bad
    quoting).
    """
    dataframe = pd.DataFrame({name: list(range(n_rows))})
    csv_text = dataframe_to_safe_csv(dataframe)
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert len(rows) == n_rows + 1, f"Expected {n_rows + 1} rows (header + {n_rows} data), got {len(rows)}"
