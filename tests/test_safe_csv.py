"""Tests for safe CSV export helpers."""

from __future__ import annotations

import csv
import io

import pandas as pd

from app.security.safe_csv import dataframe_to_safe_csv, sanitize_csv_dataframe


def _parse_rows(csv_text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(csv_text)))


class TestSafeCsv:
    def test_sanitizes_dangerous_prefixes_in_headers_and_values(self):
        dataframe = pd.DataFrame(
            {
                "=name": ["=cmd", "+sum", "-1+2", "@lookup", "safe"],
                "note": ["plain", "still safe", "ok", "fine", "hello"],
            },
            index=pd.Index(["@row0", "row1", "row2", "row3", "row4"], name="-id"),
        )

        rows = _parse_rows(dataframe_to_safe_csv(dataframe, index=True))

        assert rows[0][0] == "'-id"
        assert rows[0][1] == "'=name"
        assert rows[1][0] == "'@row0"
        assert rows[1][1] == "'=cmd"
        assert rows[2][1] == "'+sum"
        assert rows[3][1] == "'-1+2"
        assert rows[4][1] == "'@lookup"
        assert rows[5][1] == "safe"

    def test_enforces_quote_all_output(self):
        dataframe = pd.DataFrame({"value": ["hello, \"world\"", "plain"]})

        csv_text = dataframe_to_safe_csv(dataframe, index=False)
        lines = csv_text.splitlines()

        assert lines[0] == '"value"'
        assert lines[1] == '"hello, ""world"""'
        assert lines[2] == '"plain"'

    def test_sanitize_csv_dataframe_does_not_mutate_original(self):
        dataframe = pd.DataFrame({"name": ["=cmd"]})

        sanitized = sanitize_csv_dataframe(dataframe)

        assert dataframe.iloc[0, 0] == "=cmd"
        assert sanitized.iloc[0, 0] == "'=cmd"