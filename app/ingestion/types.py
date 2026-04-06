"""Types and enums for the ingestion layer."""

from __future__ import annotations

from enum import Enum


class IngestionSourceType(str, Enum):
    """Supported dataset source types for tabular ingestion."""

    CSV = "csv"
    DELIMITED_TEXT = "delimited_text"
    EXCEL = "excel"
    HTML_TABLE = "html_table"
    DATAFRAME = "dataframe"
    KAGGLE = "kaggle"
    URL_FILE = "url_file"
    UCI_REPO = "uci_repo"


DELIMITED_FILE_SUFFIXES = {".csv", ".tsv", ".txt", ".data"}
EXCEL_FILE_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".xlsb"}
HTML_FILE_SUFFIXES = {".html", ".htm"}
