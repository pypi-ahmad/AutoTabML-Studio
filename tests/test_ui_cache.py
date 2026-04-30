from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from app.config.models import AppSettings
from app.ingestion.schemas import DatasetInputSpec, DatasetMetadata, LoadedDataset
from app.ingestion.types import IngestionSourceType
from app.pages import ui_cache
from app.registry.schemas import RegistryModelSummary


def _settings_for(tmp_path: Path) -> AppSettings:
    return AppSettings.model_validate({"artifacts": {"root_dir": str(tmp_path / "artifacts")}})


def _loaded_dataset_for(input_spec: DatasetInputSpec) -> LoadedDataset:
    frame = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    metadata = DatasetMetadata(
        source_type=input_spec.source_type,
        source_locator=input_spec.locator,
        display_name=input_spec.display_name,
        ingestion_timestamp=datetime.now(timezone.utc),
        row_count=len(frame),
        column_count=len(frame.columns),
        column_names=list(frame.columns),
        dtype_summary={column: str(dtype) for column, dtype in frame.dtypes.items()},
        schema_hash="schema-hash",
        content_hash="content-hash",
    )
    return LoadedDataset(dataframe=frame, metadata=metadata, input_spec=input_spec)


def test_load_dataset_for_ui_uses_cache_until_invalidated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ui_cache.invalidate_all_ui_caches()

    dataset_path = tmp_path / "train.csv"
    dataset_path.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
    input_spec = DatasetInputSpec(source_type=IngestionSourceType.CSV, path=dataset_path)

    calls = {"count": 0}

    def fake_load_dataset(spec: DatasetInputSpec) -> LoadedDataset:
        calls["count"] += 1
        return _loaded_dataset_for(spec)

    monkeypatch.setattr(ui_cache, "load_dataset", fake_load_dataset)

    ui_cache.load_dataset_for_ui(input_spec)
    ui_cache.load_dataset_for_ui(input_spec)

    assert calls["count"] == 1

    ui_cache.invalidate_dataset_cache()
    ui_cache.load_dataset_for_ui(input_spec)

    assert calls["count"] == 2


def test_list_cached_registered_models_uses_cache_until_invalidated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ui_cache.invalidate_all_ui_caches()
    settings = _settings_for(tmp_path)
    calls = {"count": 0}

    def fake_list_models(self) -> list[RegistryModelSummary]:  # noqa: ANN001
        calls["count"] += 1
        return [RegistryModelSummary(name="demo-model")]

    monkeypatch.setattr(ui_cache.RegistryService, "list_models", fake_list_models)

    ui_cache.list_cached_registered_models(settings)
    ui_cache.list_cached_registered_models(settings)

    assert calls["count"] == 1

    ui_cache.invalidate_mlflow_query_cache()
    ui_cache.list_cached_registered_models(settings)

    assert calls["count"] == 2


def test_get_prediction_service_reuses_resource_instance(tmp_path: Path) -> None:
    ui_cache.invalidate_all_ui_caches()
    settings = _settings_for(tmp_path)

    first = ui_cache.get_prediction_service(settings)
    second = ui_cache.get_prediction_service(settings)

    assert first is second