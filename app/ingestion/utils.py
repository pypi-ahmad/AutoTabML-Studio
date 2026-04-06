"""Shared helpers for the ingestion layer."""

from __future__ import annotations

import csv


def sniff_delimiter(sample_text: str) -> str | None:
    """Use csv.Sniffer to guess the delimiter from a text sample.

    Returns None when the sniffer cannot determine the delimiter.
    """
    try:
        dialect = csv.Sniffer().sniff(sample_text[:4096], delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except csv.Error:
        return None
