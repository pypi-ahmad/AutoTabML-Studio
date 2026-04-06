"""Tests for UCI ML Repository ingestion (loader, factory, schemas, CLI)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.cli import _build_input_spec
from app.ingestion.errors import IngestionError, RemoteAccessError
from app.ingestion.factory import get_loader, load_dataset
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import IngestionSourceType
from app.ingestion.uci_loader import UCIRepoLoader, list_available_uci_datasets

# ---------------------------------------------------------------------------
# IngestionSourceType enum
# ---------------------------------------------------------------------------

class TestUCISourceType:
    def test_uci_repo_enum_exists(self):
        assert IngestionSourceType.UCI_REPO.value == "uci_repo"

    def test_uci_repo_in_all_types(self):
        assert IngestionSourceType.UCI_REPO in IngestionSourceType


# ---------------------------------------------------------------------------
# DatasetInputSpec for UCI
# ---------------------------------------------------------------------------

class TestUCIInputSpec:
    def test_spec_with_uci_id(self):
        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=53)
        assert spec.uci_id == 53
        assert spec.locator == "uci:53"

    def test_spec_with_uci_name(self):
        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_name="Iris")
        assert spec.uci_name == "Iris"
        assert spec.locator == "uci:Iris"

    def test_spec_uci_requires_id_or_name(self):
        with pytest.raises(ValueError, match="dataset ID or name"):
            DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO)

    def test_spec_uci_rejects_id_and_name_together(self):
        with pytest.raises(ValueError, match="either a dataset ID or a name, not both"):
            DatasetInputSpec(
                source_type=IngestionSourceType.UCI_REPO,
                uci_id=53,
                uci_name="Iris",
            )


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

class TestUCIFactory:
    def test_factory_returns_uci_loader(self):
        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=53)
        loader = get_loader(spec)
        assert isinstance(loader, UCIRepoLoader)


# ---------------------------------------------------------------------------
# UCIRepoLoader with mock
# ---------------------------------------------------------------------------

def _build_mock_uci_dataset(
    *,
    name: str = "Iris",
    uci_id: int = 53,
    num_instances: int = 150,
    ids: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    targets: pd.DataFrame | None = None,
    original: pd.DataFrame | None = None,
) -> MagicMock:
    """Build a mock object matching the ucimlrepo.fetch_ucirepo return shape."""
    if features is None:
        features = pd.DataFrame({"sepal_length": [5.1, 4.9], "sepal_width": [3.5, 3.0]})
    if targets is None:
        targets = pd.DataFrame({"class": ["setosa", "setosa"]})
    if ids is None:
        ids = pd.DataFrame()
    if original is None:
        original_frames = [frame for frame in (ids, features, targets) if frame is not None and not frame.empty]
        original = pd.concat(original_frames, axis=1) if original_frames else pd.DataFrame()

    mock_ds = MagicMock()
    mock_ds.data.ids = ids
    mock_ds.data.features = features
    mock_ds.data.targets = targets
    mock_ds.data.original = original
    mock_ds.data.headers = original.columns
    mock_ds.metadata.uci_id = uci_id
    mock_ds.metadata.name = name
    mock_ds.metadata.num_instances = num_instances
    mock_ds.metadata.num_features = len(features.columns)
    mock_ds.metadata.abstract = "Test abstract"
    mock_ds.metadata.area = "Life Science"
    mock_ds.metadata.task = "Classification"
    mock_ds.metadata.feature_types = "Real"
    mock_ds.metadata.target_col = targets.columns.tolist() if targets is not None else []
    mock_ds.metadata.index_col = ids.columns.tolist() if ids is not None and not ids.empty else []
    mock_ds.metadata.has_missing_values = "no"
    mock_ds.metadata.missing_values_symbol = "?"
    mock_ds.metadata.year_of_dataset_creation = 1988
    mock_ds.metadata.dataset_doi = "10.24432/C56C76"
    mock_ds.metadata.creators = ["R. A. Fisher"]
    mock_ds.metadata.intro_paper = {"title": "Iris data paper"}
    mock_ds.metadata.repository_url = "https://archive.ics.uci.edu/dataset/53/iris"
    mock_ds.metadata.data_url = "https://archive.ics.uci.edu/static/public/53/iris.zip"
    mock_ds.metadata.external_url = None
    mock_ds.metadata.characteristics = ["Multivariate"]
    mock_ds.metadata.additional_info = {
        "summary": "Classic iris classification dataset.",
        "purpose": "Demonstration",
        "recommended_data_splits": "Standard train/test split",
    }
    mock_ds.metadata.get = lambda key, default=None: getattr(mock_ds.metadata, key, default)
    mock_ds.variables = pd.DataFrame({
        "name": ["sepal_length", "sepal_width", "class"],
        "role": ["Feature", "Feature", "Target"],
        "type": ["Continuous", "Continuous", "Categorical"],
    })
    return mock_ds


class TestUCIRepoLoader:
    def test_supported_source_types(self):
        loader = UCIRepoLoader()
        assert loader.supported_source_types == (IngestionSourceType.UCI_REPO,)

    def test_load_by_id(self, monkeypatch):
        mock_ds = _build_mock_uci_dataset()
        mock_fetch = MagicMock(return_value=mock_ds)
        ucimlrepo = pytest.importorskip("ucimlrepo")
        monkeypatch.setattr(ucimlrepo, "fetch_ucirepo", mock_fetch)

        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=53)
        loader = UCIRepoLoader()
        df, source_details = loader.load_raw_dataframe(spec)

        mock_fetch.assert_called_once_with(id=53)
        assert "sepal_length" in df.columns
        assert "class" in df.columns
        assert df.shape == (2, 3)
        assert source_details["source_kind"] == "uci_repo"
        assert source_details["uci_id"] == 53
        assert source_details["uci_name"] == "Iris"
        assert source_details["feature_columns"] == ["sepal_length", "sepal_width"]
        assert source_details["target_columns"] == ["class"]
        assert source_details["uci_additional_info"]["summary"] == "Classic iris classification dataset."
        assert source_details["uci_headers"] == ["sepal_length", "sepal_width", "class"]

    def test_load_preserves_original_and_id_columns(self, monkeypatch):
        ids = pd.DataFrame({"record_id": [101, 102]})
        features = pd.DataFrame({"feature_a": [1.1, 2.2]})
        targets = pd.DataFrame({"target": [0, 1]})
        original = pd.concat([ids, features, targets], axis=1)
        mock_ds = _build_mock_uci_dataset(ids=ids, features=features, targets=targets, original=original)
        ucimlrepo = pytest.importorskip("ucimlrepo")
        monkeypatch.setattr(ucimlrepo, "fetch_ucirepo", MagicMock(return_value=mock_ds))

        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=1)
        loader = UCIRepoLoader()
        df, source_details = loader.load_raw_dataframe(spec)

        assert df.columns.tolist() == ["record_id", "feature_a", "target"]
        assert source_details["id_columns"] == ["record_id"]
        assert source_details["uci_index_col"] == ["record_id"]
        assert source_details["uci_original_shape"] == [2, 3]

    def test_load_by_name(self, monkeypatch):
        mock_ds = _build_mock_uci_dataset(name="Heart Disease", uci_id=45)
        mock_fetch = MagicMock(return_value=mock_ds)
        ucimlrepo = pytest.importorskip("ucimlrepo")
        monkeypatch.setattr(ucimlrepo, "fetch_ucirepo", mock_fetch)

        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_name="Heart Disease")
        loader = UCIRepoLoader()
        df, source_details = loader.load_raw_dataframe(spec)

        mock_fetch.assert_called_once_with(name="Heart Disease")
        assert source_details["uci_name"] == "Heart Disease"

    def test_row_limit_applied(self, monkeypatch):
        features = pd.DataFrame({"a": range(100)})
        targets = pd.DataFrame({"t": range(100)})
        mock_ds = _build_mock_uci_dataset(features=features, targets=targets)
        ucimlrepo = pytest.importorskip("ucimlrepo")
        monkeypatch.setattr(ucimlrepo, "fetch_ucirepo", MagicMock(return_value=mock_ds))

        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=1)
        loader = UCIRepoLoader()
        df, _ = loader.load_raw_dataframe(spec, row_limit=10)

        assert len(df) == 10

    def test_empty_targets_handled(self, monkeypatch):
        features = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        targets = pd.DataFrame()
        mock_ds = _build_mock_uci_dataset(features=features, targets=targets)
        ucimlrepo = pytest.importorskip("ucimlrepo")
        monkeypatch.setattr(ucimlrepo, "fetch_ucirepo", MagicMock(return_value=mock_ds))

        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=1)
        loader = UCIRepoLoader()
        df, source_details = loader.load_raw_dataframe(spec)

        assert df.shape == (2, 2)
        assert source_details["target_columns"] == []

    def test_none_targets_handled(self, monkeypatch):
        features = pd.DataFrame({"a": [1, 2]})
        mock_ds = _build_mock_uci_dataset(features=features, targets=pd.DataFrame(), original=features)
        mock_ds.data.targets = None
        ucimlrepo = pytest.importorskip("ucimlrepo")
        monkeypatch.setattr(ucimlrepo, "fetch_ucirepo", MagicMock(return_value=mock_ds))

        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=1)
        loader = UCIRepoLoader()
        df, source_details = loader.load_raw_dataframe(spec)

        assert df.shape == (2, 1)
        assert source_details["target_columns"] == []

    def test_fetch_failure_raises_remote_access_error(self, monkeypatch):
        ucimlrepo = pytest.importorskip("ucimlrepo")
        monkeypatch.setattr(
            ucimlrepo, "fetch_ucirepo", MagicMock(side_effect=Exception("Network error"))
        )

        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=999)
        loader = UCIRepoLoader()

        with pytest.raises(RemoteAccessError, match="Failed to fetch UCI dataset.*999"):
            loader.load_raw_dataframe(spec)

    def test_missing_ucimlrepo_package(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "ucimlrepo":
                raise ImportError("no ucimlrepo")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        spec = DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=53)
        loader = UCIRepoLoader()

        with pytest.raises(RemoteAccessError, match="ucimlrepo"):
            loader.load_raw_dataframe(spec)

    def test_requires_id_or_name(self):
        pytest.importorskip("ucimlrepo")
        spec = DatasetInputSpec.__new__(DatasetInputSpec)
        object.__setattr__(spec, "uci_id", None)
        object.__setattr__(spec, "uci_name", None)
        object.__setattr__(spec, "source_type", IngestionSourceType.UCI_REPO)

        loader = UCIRepoLoader()
        with pytest.raises(IngestionError, match="dataset ID.*or name"):
            loader.load_raw_dataframe(spec)

    def test_requires_exactly_one_of_id_or_name(self):
        pytest.importorskip("ucimlrepo")
        spec = DatasetInputSpec.__new__(DatasetInputSpec)
        object.__setattr__(spec, "uci_id", 53)
        object.__setattr__(spec, "uci_name", "Iris")
        object.__setattr__(spec, "source_type", IngestionSourceType.UCI_REPO)

        loader = UCIRepoLoader()
        with pytest.raises(IngestionError, match="either a dataset ID or a name, not both"):
            loader.load_raw_dataframe(spec)


class TestUCICatalog:
    def test_list_available_datasets_parses_results(self, monkeypatch):
        ucimlrepo = pytest.importorskip("ucimlrepo")

        def _fake_list_available_datasets(*, filter=None, search=None, area=None):
            print("-------------------------------------------------------------")
            print('The following datasets are available for search query "iris":')
            print("-------------------------------------------------------------")
            print("Dataset Name    ID")
            print("------------    --")
            print("Iris            53")

        monkeypatch.setattr(ucimlrepo, "list_available_datasets", _fake_list_available_datasets)

        rows = list_available_uci_datasets(search="iris")

        assert rows == [{"uci_id": 53, "name": "Iris"}]

    def test_list_available_datasets_handles_no_results(self, monkeypatch):
        ucimlrepo = pytest.importorskip("ucimlrepo")

        monkeypatch.setattr(ucimlrepo, "list_available_datasets", lambda **kwargs: print("No datasets found"))

        assert list_available_uci_datasets(search="missing") == []


# ---------------------------------------------------------------------------
# Full pipeline (load_dataset) with mock
# ---------------------------------------------------------------------------

class TestUCIFullPipeline:
    def test_load_dataset_through_pipeline(self, monkeypatch):
        mock_ds = _build_mock_uci_dataset()
        ucimlrepo = pytest.importorskip("ucimlrepo")
        monkeypatch.setattr(ucimlrepo, "fetch_ucirepo", MagicMock(return_value=mock_ds))

        loaded = load_dataset(
            DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=53)
        )

        assert loaded.dataframe.shape[0] == 2
        assert loaded.metadata.source_type == IngestionSourceType.UCI_REPO
        assert loaded.metadata.source_details["source_kind"] == "uci_repo"
        assert "uci_id" in loaded.metadata.source_details
        assert loaded.metadata.source_details["uci_loaded_shape"] == [2, 3]

    def test_metadata_includes_uci_details(self, monkeypatch):
        mock_ds = _build_mock_uci_dataset()
        ucimlrepo = pytest.importorskip("ucimlrepo")
        monkeypatch.setattr(ucimlrepo, "fetch_ucirepo", MagicMock(return_value=mock_ds))

        loaded = load_dataset(
            DatasetInputSpec(
                source_type=IngestionSourceType.UCI_REPO,
                uci_id=53,
                display_name="My Iris",
            )
        )

        assert loaded.metadata.display_name == "My Iris"
        assert loaded.metadata.source_locator == "uci:53"
        assert loaded.metadata.source_details["uci_repository_url"]
        assert loaded.metadata.source_details["uci_variables"]


# ---------------------------------------------------------------------------
# CLI locator support
# ---------------------------------------------------------------------------

class TestCLIUCILocator:
    def test_uci_id_locator_parsed(self):
        spec = _build_input_spec("uci:53")
        assert spec.source_type == IngestionSourceType.UCI_REPO
        assert spec.uci_id == 53
        assert spec.uci_name is None

    def test_uci_name_locator_parsed(self):
        spec = _build_input_spec("uci:Heart Disease")
        assert spec.source_type == IngestionSourceType.UCI_REPO
        assert spec.uci_id is None
        assert spec.uci_name == "Heart Disease"

    def test_uci_empty_value_raises(self):
        with pytest.raises(ValueError, match="uci:<id>.*uci:<name>"):
            _build_input_spec("uci:")

    def test_uci_numeric_string_parsed_as_id(self):
        spec = _build_input_spec("uci:45")
        assert spec.uci_id == 45


# ---------------------------------------------------------------------------
# Packaging: uci extra
# ---------------------------------------------------------------------------

class TestPackagingUCIExtra:
    def test_uci_extra_includes_ucimlrepo(self):
        import tomllib
        data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        uci_deps = data["project"]["optional-dependencies"]["uci"]
        assert any("ucimlrepo" in dep for dep in uci_deps)
