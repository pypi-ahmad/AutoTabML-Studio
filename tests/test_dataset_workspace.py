"""Tests for shared dataset workspace helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.ingestion import DatasetInputSpec, DatasetMetadata, IngestionSourceType, LoadedDataset
from app.pages import dataset_workspace as dataset_workspace_module
from app.pages.dataset_workspace import (
    build_local_path_input_spec,
    build_url_input_spec,
    get_active_loaded_dataset,
    infer_local_source_type,
    resolve_session_dataset_name,
    set_active_dataset,
)
from app.storage import AppMetadataStore


def _make_loaded_dataset(
    *,
    display_name: str | None = None,
    path: str | None = None,
    url: str | None = None,
    schema_hash: str = "schema-hash",
    content_hash: str | None = None,
) -> LoadedDataset:
    source_type = IngestionSourceType.DATAFRAME
    input_spec = DatasetInputSpec(source_type=source_type, dataframe=pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    if path is not None:
        input_spec = DatasetInputSpec(source_type=IngestionSourceType.CSV, path=Path(path), display_name=display_name)
    elif url is not None:
        input_spec = DatasetInputSpec(source_type=IngestionSourceType.URL_FILE, url=url, display_name=display_name)

    metadata = DatasetMetadata(
        source_type=input_spec.source_type,
        source_locator=input_spec.locator,
        display_name=display_name,
        ingestion_timestamp=datetime.now(timezone.utc),
        row_count=2,
        column_count=2,
        column_names=["a", "b"],
        dtype_summary={"a": "int64", "b": "int64"},
        schema_hash=schema_hash,
        content_hash=content_hash,
    )
    return LoadedDataset(
        dataframe=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        metadata=metadata,
        input_spec=input_spec,
    )


class TestInferLocalSourceType:
    def test_csv_and_excel_and_delimited_are_supported(self):
        assert infer_local_source_type("train.csv") == IngestionSourceType.CSV
        assert infer_local_source_type("train.tsv") == IngestionSourceType.DELIMITED_TEXT
        assert infer_local_source_type("train.xlsx") == IngestionSourceType.EXCEL

    def test_unsupported_suffix_raises(self):
        with pytest.raises(ValueError, match="Unsupported dataset file type"):
            infer_local_source_type("train.parquet")


class TestInputSpecBuilders:
    def test_build_local_path_input_spec_infers_source_type(self):
        spec = build_local_path_input_spec(r"artifacts\demo\train.csv", display_name="demo-train")

        assert spec.source_type == IngestionSourceType.CSV
        assert spec.path == Path(r"artifacts\demo\train.csv")
        assert spec.display_name == "demo-train"

    def test_build_url_input_spec_uses_url_file(self):
        spec = build_url_input_spec("https://example.com/data.csv", display_name="remote-data")

        assert spec.source_type == IngestionSourceType.URL_FILE
        assert spec.url == "https://example.com/data.csv"
        assert spec.display_name == "remote-data"


class TestResolveSessionDatasetName:
    def test_prefers_explicit_name(self):
        loaded = _make_loaded_dataset(path="artifacts/demo/train.csv")

        name = resolve_session_dataset_name("Customer Churn", loaded, [])

        assert name == "Customer Churn"

    def test_falls_back_to_path_stem_and_uniquifies(self):
        loaded = _make_loaded_dataset(path="artifacts/demo/train.csv")

        name = resolve_session_dataset_name(None, loaded, ["train", "train (2)"])

        assert name == "train (3)"

    def test_falls_back_to_url_name(self):
        loaded = _make_loaded_dataset(url="https://example.com/datasets/iris.csv")

        name = resolve_session_dataset_name(None, loaded, [])

        assert name == "iris"


class TestActiveDatasetSelection:
    def test_set_active_dataset_persists_session_and_workspace_metadata(self, monkeypatch, tmp_path: Path):
        store = AppMetadataStore(tmp_path / "app.sqlite3")
        dataset = _make_loaded_dataset(path="artifacts/demo/train.csv", schema_hash="schema-train")
        fake_streamlit = SimpleNamespace(session_state={"loaded_datasets": {"train": dataset}})
        monkeypatch.setattr(dataset_workspace_module, "st", fake_streamlit)

        selected = set_active_dataset("train", metadata_store=store)
        project = store.get_workspace_project()

        assert selected == "train"
        assert fake_streamlit.session_state["active_dataset_name"] == "train"
        assert project.metadata["active_dataset_name"] == "train"
        assert project.metadata["active_dataset_key"] == "schema-train"

    def test_get_active_loaded_dataset_restores_selection_from_workspace_metadata(self, monkeypatch, tmp_path: Path):
        store = AppMetadataStore(tmp_path / "app.sqlite3")
        persisted_project = store.get_workspace_project()
        store.upsert_project(
            persisted_project.model_copy(
                update={
                    "metadata": {
                        **persisted_project.metadata,
                        "active_dataset_name": "original-name",
                        "active_dataset_key": "schema-restored",
                    }
                }
            )
        )

        restored_dataset = _make_loaded_dataset(path="artifacts/demo/reloaded.csv", schema_hash="schema-restored")
        fake_streamlit = SimpleNamespace(session_state={"loaded_datasets": {"reloaded": restored_dataset}})
        monkeypatch.setattr(dataset_workspace_module, "st", fake_streamlit)

        selected_name, selected_dataset = get_active_loaded_dataset(metadata_store=store)

        assert selected_name == "reloaded"
        assert selected_dataset == restored_dataset
        assert fake_streamlit.session_state["active_dataset_name"] == "reloaded"