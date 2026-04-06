"""Unit tests for ingestion normalization, metadata, hashing, and parsing helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.ingestion.csv_loader import CSVLoader
from app.ingestion.errors import ParseFailureError
from app.ingestion.metadata import compute_content_hash, compute_schema_hash, extract_dataset_metadata
from app.ingestion.normalizer import normalize_to_pandas
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import IngestionSourceType


class TestMetadataAndHashing:
    def test_schema_hash_is_deterministic(self):
        dataframe = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        hash_one = compute_schema_hash(dataframe)
        hash_two = compute_schema_hash(dataframe.copy())

        assert hash_one == hash_two

    def test_content_hash_changes_when_rows_change(self):
        first = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        second = pd.DataFrame({"a": [1, 3], "b": ["x", "z"]})

        assert compute_content_hash(first) != compute_content_hash(second)

    def test_content_hash_ignores_index_gaps(self):
        """Same data with different index values must produce the same hash."""
        df_a = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        df_b = pd.DataFrame({"a": [1, 2]}, index=[5, 10])

        assert compute_content_hash(df_a) == compute_content_hash(df_b)

    def test_extract_dataset_metadata_fields(self, tmp_path: Path):
        csv_path = tmp_path / "metadata.csv"
        csv_path.write_text("one,two\n1,2\n", encoding="utf-8")
        spec = DatasetInputSpec(source_type=IngestionSourceType.CSV, path=csv_path, project_id="proj-1", tags=["demo"])
        dataframe = pd.DataFrame({"one": [1], "two": [2]})

        metadata = extract_dataset_metadata(
            dataframe,
            spec,
            normalization_actions=["Dropped 1 fully empty row(s)."],
            source_details={"source_kind": "path"},
        )

        assert metadata.row_count == 1
        assert metadata.column_count == 2
        assert metadata.file_extension == ".csv"
        assert metadata.project_id == "proj-1"
        assert metadata.tags == ["demo"]
        assert metadata.normalization_actions == ["Dropped 1 fully empty row(s)."]


class TestNormalization:
    def test_normalize_to_pandas_drops_empty_rows_and_columns_and_dedupes(self):
        dataframe = pd.DataFrame(
            [[1, 2, None], [None, None, None]],
            columns=["dup", "dup", "empty_col"],
        )

        normalized, actions = normalize_to_pandas(dataframe)

        assert normalized.shape == (1, 2)
        assert normalized.columns.tolist() == ["dup", "dup__2"]
        assert any(action == "Dropped 1 fully empty row(s)." for action in actions)
        assert any(action == "Dropped 1 fully empty column(s)." for action in actions)
        assert any("Normalized duplicate column names" in action for action in actions)


class TestParseFailures:
    def test_bad_csv_raises_parse_failure(self):
        loader = CSVLoader()
        broken_path = Path("broken.csv")
        spec = DatasetInputSpec(source_type=IngestionSourceType.CSV, path=broken_path)

        with pytest.raises(ParseFailureError, match="Failed to parse the delimited file"):
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(Path, "exists", lambda self: True)
                monkeypatch.setattr(
                    pd,
                    "read_csv",
                    lambda *args, **kwargs: (_ for _ in ()).throw(pd.errors.ParserError("bad csv")),
                )
                loader.load_raw_dataframe(spec)