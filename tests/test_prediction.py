"""Tests for prediction / inference workflows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.config.models import AppSettings
from app.modeling.pycaret.schemas import ExperimentTaskType, SavedModelMetadata
from app.observability import InMemoryMetricsBackend, install_correlation_filter, set_metrics_backend
from app.prediction import (
    BatchPredictionRequest,
    LoadedModel,
    LocalPyCaretModelLoader,
    MLflowModelLoader,
    ModelSourceType,
    PredictionHistoryEntry,
    PredictionMode,
    PredictionRequest,
    PredictionResult,
    PredictionService,
    PredictionStatus,
    PredictionSummary,
    PredictionTaskType,
    PredictionValidationResult,
    SchemaValidationMode,
    SingleRowPredictionRequest,
)
from app.prediction.artifacts import write_prediction_artifacts
from app.prediction.errors import ModelDiscoveryError, ModelLoadError, PredictionValidationError
from app.prediction.history import PredictionHistoryStore
from app.prediction.schemas import PredictionArtifactBundle, PredictionValidationSeverity
from app.prediction.selectors import (
    build_mlflow_registered_model_uri,
    build_mlflow_run_model_uri,
    discover_local_saved_models,
    load_saved_model_metadata_file,
    resolve_local_model_reference,
)
from app.security.errors import TrustedArtifactError
from app.security.trusted_artifacts import TRUSTED_MODEL_SOURCE, compute_sha256, write_checksum_file
from app.prediction.validators import normalize_single_row_input, validate_prediction_dataframe


class _FakeClassifier:
    classes_ = ["no", "yes"]

    def predict(self, dataframe: pd.DataFrame):
        return pd.Series(
            ["yes" if value >= 0.5 else "no" for value in dataframe["a"]],
            index=dataframe.index,
        )

    def predict_proba(self, dataframe: pd.DataFrame):
        rows = []
        for value in dataframe["a"]:
            if value >= 0.5:
                rows.append([0.2, 0.8])
            else:
                rows.append([0.9, 0.1])
        return rows


class _FakeRegressor:
    def predict(self, dataframe: pd.DataFrame):
        return pd.Series([value * 10 for value in dataframe["a"]], index=dataframe.index)


def _write_trusted_saved_model_metadata(metadata: SavedModelMetadata, metadata_path: Path) -> SavedModelMetadata:
    model_sha256 = compute_sha256(metadata.model_path)
    write_checksum_file(metadata.model_path, checksum=model_sha256)
    trusted_metadata = metadata.model_copy(
        update={
            "artifact_format": "pycaret_pickle",
            "trusted_source": TRUSTED_MODEL_SOURCE,
            "model_sha256": model_sha256,
        }
    )
    metadata_path.write_text(trusted_metadata.model_dump_json(indent=2), encoding="utf-8")
    write_checksum_file(metadata_path)
    return trusted_metadata


def _make_loaded_model(
    *,
    task_type: PredictionTaskType = PredictionTaskType.CLASSIFICATION,
    feature_columns: list[str] | None = None,
    native_model=None,
    scorer_kind: str = "sklearn_like",
) -> LoadedModel:
    return LoadedModel(
        source_type=ModelSourceType.LOCAL_SAVED_MODEL,
        task_type=task_type,
        model_identifier="demo-model",
        load_reference="artifacts/models/demo-model.pkl",
        loader_name="TestLoader",
        scorer_kind=scorer_kind,
        supported_prediction_modes=[PredictionMode.SINGLE_ROW, PredictionMode.BATCH],
        feature_columns=feature_columns or ["a", "b"],
        target_column="target",
        metadata={"feature_dtypes": {"a": "float64", "b": "int64"}},
        native_model=native_model or _FakeClassifier(),
    )


class TestPredictionSelectors:
    def test_build_mlflow_run_model_uri(self):
        assert build_mlflow_run_model_uri("run-123", "model") == "runs:/run-123/model"

    def test_build_mlflow_registered_model_uri_for_alias(self):
        assert build_mlflow_registered_model_uri("pricing-model", alias="champion") == "models:/pricing-model@champion"

    def test_discover_and_resolve_local_saved_model(self, tmp_path: Path):
        model_dir = tmp_path / "models"
        metadata_dir = tmp_path / "metadata"
        model_dir.mkdir()
        metadata_dir.mkdir()
        model_path = model_dir / "house_price.pkl"
        model_path.write_bytes(b"fake-model")

        metadata = SavedModelMetadata(
            task_type=ExperimentTaskType.REGRESSION,
            target_column="price",
            model_id="lr",
            model_name="HousePriceModel",
            model_path=model_path,
            dataset_fingerprint="fp-123",
            feature_columns=["sqft", "beds"],
            feature_dtypes={"sqft": "float64", "beds": "int64"},
            target_dtype="float64",
        )
        metadata_path = metadata_dir / "housing_saved_model_metadata_20260404T010101.json"
        _write_trusted_saved_model_metadata(metadata, metadata_path)

        references = discover_local_saved_models([model_dir], [metadata_dir])

        assert len(references) == 1
        resolved = resolve_local_model_reference("HousePriceModel", references)
        assert resolved.model_path == model_path
        assert resolved.metadata_path == metadata_path
        assert resolved.task_type == PredictionTaskType.REGRESSION


class TestLocalPyCaretModelLoader:
    def test_loads_local_model_from_saved_metadata(self, tmp_path: Path, monkeypatch):
        model_dir = tmp_path / "models"
        metadata_dir = tmp_path / "metadata"
        model_dir.mkdir()
        metadata_dir.mkdir()
        model_path = model_dir / "classifier.pkl"
        model_path.write_bytes(b"fake-model")

        metadata = SavedModelMetadata(
            task_type=ExperimentTaskType.CLASSIFICATION,
            target_column="target",
            model_id="lr",
            model_name="ClassifierModel",
            model_path=model_path,
            dataset_fingerprint="fp-abc",
            feature_columns=["a", "b"],
            feature_dtypes={"a": "float64", "b": "int64"},
            target_dtype="int64",
        )
        metadata_path = metadata_dir / "classifier_saved_model_metadata_20260404T010101.json"
        _write_trusted_saved_model_metadata(metadata, metadata_path)

        monkeypatch.setattr(
            "app.modeling.pycaret.persistence.load_model_artifact",
            lambda task_type, model_name_or_path: {"task_type": task_type, "path": str(model_name_or_path)},
        )

        loader = LocalPyCaretModelLoader(model_dirs=[model_dir], metadata_dirs=[metadata_dir])
        loaded = loader.load(
            PredictionRequest(
                source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                model_identifier=str(model_path),
            )
        )

        assert loaded.task_type == PredictionTaskType.CLASSIFICATION
        assert loaded.feature_columns == ["a", "b"]
        assert loaded.metadata["target_column"] == "target"

    def test_local_model_without_trusted_metadata_is_rejected(self, tmp_path: Path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_path = model_dir / "orphan_model.pkl"
        model_path.write_bytes(b"fake-model")

        loader = LocalPyCaretModelLoader(model_dirs=[model_dir], metadata_dirs=[])

        with pytest.raises(ModelDiscoveryError, match="trusted saved model"):
            loader.load(
                PredictionRequest(
                    source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                    model_identifier=str(model_path),
                )
            )

    def _build_trusted_local_model(self, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        model_dir = tmp_path / "models"
        metadata_dir = tmp_path / "metadata"
        model_dir.mkdir()
        metadata_dir.mkdir()
        model_path = model_dir / "classifier.pkl"
        model_path.write_bytes(b"fake-model")

        metadata = SavedModelMetadata(
            task_type=ExperimentTaskType.CLASSIFICATION,
            target_column="target",
            model_id="lr",
            model_name="ClassifierModel",
            model_path=model_path,
            dataset_fingerprint="fp-abc",
            feature_columns=["a", "b"],
            feature_dtypes={"a": "float64", "b": "int64"},
            target_dtype="int64",
        )
        metadata_path = metadata_dir / "classifier_saved_model_metadata_20260404T010101.json"
        _write_trusted_saved_model_metadata(metadata, metadata_path)
        return model_dir, metadata_dir, model_path, metadata_path

    def test_tampered_model_bytes_are_rejected(self, tmp_path: Path):
        """Simulate post-save tampering of the pickle payload — must reject."""
        model_dir, metadata_dir, model_path, metadata_path = self._build_trusted_local_model(tmp_path)

        # Attacker rewrites the pickle bytes after a trusted save.
        model_path.write_bytes(b"malicious-payload")

        # Direct enforcement layer must raise on integrity mismatch.
        with pytest.raises(TrustedArtifactError, match="checksum mismatch"):
            load_saved_model_metadata_file(
                metadata_path,
                metadata_roots=[metadata_dir, model_dir],
                model_roots=[model_dir],
                raise_on_error=True,
            )

        # High-level loader must refuse to surface the tampered model at all.
        loader = LocalPyCaretModelLoader(model_dirs=[model_dir], metadata_dirs=[metadata_dir])
        assert loader.discover() == []
        with pytest.raises(ModelDiscoveryError):
            loader.load(
                PredictionRequest(
                    source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                    model_identifier=str(model_path),
                )
            )

    def test_tampered_metadata_is_rejected(self, tmp_path: Path):
        """Simulate post-save tampering of the metadata JSON — must reject."""
        model_dir, metadata_dir, _, metadata_path = self._build_trusted_local_model(tmp_path)

        # Attacker edits metadata (e.g. swaps feature columns) without updating sidecar.
        original = metadata_path.read_text(encoding="utf-8")
        metadata_path.write_text(original.replace('"a"', '"evil"'), encoding="utf-8")

        with pytest.raises(TrustedArtifactError, match="checksum mismatch"):
            load_saved_model_metadata_file(
                metadata_path,
                metadata_roots=[metadata_dir, model_dir],
                model_roots=[model_dir],
                raise_on_error=True,
            )

        loader = LocalPyCaretModelLoader(model_dirs=[model_dir], metadata_dirs=[metadata_dir])
        assert loader.discover() == []

    def test_untrusted_source_marker_is_rejected(self, tmp_path: Path):
        """Metadata without the trusted-source marker must be rejected."""
        model_dir = tmp_path / "models"
        metadata_dir = tmp_path / "metadata"
        model_dir.mkdir()
        metadata_dir.mkdir()
        model_path = model_dir / "classifier.pkl"
        model_path.write_bytes(b"fake-model")
        sha = compute_sha256(model_path)
        write_checksum_file(model_path, checksum=sha)

        # Forged metadata with VALID checksum sidecar but WRONG trusted_source marker.
        forged = SavedModelMetadata(
            task_type=ExperimentTaskType.CLASSIFICATION,
            target_column="target",
            model_id="lr",
            model_name="ClassifierModel",
            model_path=model_path,
            dataset_fingerprint="fp-abc",
            feature_columns=["a", "b"],
            feature_dtypes={"a": "float64", "b": "int64"},
            target_dtype="int64",
            artifact_format="pycaret_pickle",
            trusted_source="attacker_supplied_source",
            model_sha256=sha,
        )
        metadata_path = metadata_dir / "forged_saved_model_metadata.json"
        metadata_path.write_text(forged.model_dump_json(indent=2), encoding="utf-8")
        write_checksum_file(metadata_path)

        with pytest.raises(TrustedArtifactError, match="trusted source marker"):
            load_saved_model_metadata_file(
                metadata_path,
                metadata_roots=[metadata_dir, model_dir],
                model_roots=[model_dir],
                raise_on_error=True,
            )

        loader = LocalPyCaretModelLoader(model_dirs=[model_dir], metadata_dirs=[metadata_dir])
        assert loader.discover() == []

    def test_path_traversal_outside_trusted_root_is_rejected(self, tmp_path: Path):
        """Metadata pointing at a model file outside trusted roots must be rejected."""
        model_dir = tmp_path / "models"
        metadata_dir = tmp_path / "metadata"
        external_dir = tmp_path / "external"
        model_dir.mkdir()
        metadata_dir.mkdir()
        external_dir.mkdir()

        # Plant a "model" outside the trusted root with a fully valid checksum sidecar.
        external_model = external_dir / "evil_model.pkl"
        external_model.write_bytes(b"fake-model")
        sha = compute_sha256(external_model)
        write_checksum_file(external_model, checksum=sha)

        forged = SavedModelMetadata(
            task_type=ExperimentTaskType.CLASSIFICATION,
            target_column="target",
            model_id="lr",
            model_name="ClassifierModel",
            model_path=external_model,
            dataset_fingerprint="fp-abc",
            feature_columns=["a", "b"],
            feature_dtypes={"a": "float64", "b": "int64"},
            target_dtype="int64",
            artifact_format="pycaret_pickle",
            trusted_source=TRUSTED_MODEL_SOURCE,
            model_sha256=sha,
        )
        metadata_path = metadata_dir / "traversal_saved_model_metadata.json"
        metadata_path.write_text(forged.model_dump_json(indent=2), encoding="utf-8")
        write_checksum_file(metadata_path)

        with pytest.raises(TrustedArtifactError, match="outside"):
            load_saved_model_metadata_file(
                metadata_path,
                metadata_roots=[metadata_dir, model_dir],
                model_roots=[model_dir],
                raise_on_error=True,
            )

        loader = LocalPyCaretModelLoader(model_dirs=[model_dir], metadata_dirs=[metadata_dir])
        assert loader.discover() == []

    def test_discover_local_models_ignores_metadata_without_checksum_sidecar(self, tmp_path: Path):
        model_dir = tmp_path / "models"
        metadata_dir = tmp_path / "metadata"
        model_dir.mkdir()
        metadata_dir.mkdir()

        model_path = model_dir / "classifier.pkl"
        model_path.write_bytes(b"fake-model")
        metadata = SavedModelMetadata(
            task_type=ExperimentTaskType.CLASSIFICATION,
            target_column="target",
            model_id="lr",
            model_name="ClassifierModel",
            model_path=model_path,
            dataset_fingerprint="fp-abc",
            feature_columns=["a", "b"],
            feature_dtypes={"a": "float64", "b": "int64"},
            target_dtype="int64",
            artifact_format="pycaret_pickle",
            trusted_source=TRUSTED_MODEL_SOURCE,
            model_sha256=compute_sha256(model_path),
        )
        metadata_path = metadata_dir / "classifier_saved_model_metadata_20260404T010101.json"
        metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

        references = discover_local_saved_models([model_dir], [metadata_dir])

        assert references == []


class TestMLflowModelLoader:

    @staticmethod
    def _fake_import_module(name):
        """Return a lightweight fake for mlflow and mlflow.pyfunc imports."""
        if name == "mlflow.pyfunc":
            return SimpleNamespace(load_model=lambda model_uri, dst_path=None: {"uri": model_uri})
        # For any other submodule (e.g. mlflow itself referenced via importlib)
        return SimpleNamespace(
            set_tracking_uri=lambda uri: None,
            set_registry_uri=lambda uri: None,
        )

    def test_loads_run_model_via_mlflow_pyfunc(self, monkeypatch):
        monkeypatch.setattr("app.prediction.loader.is_mlflow_available", lambda: True)
        monkeypatch.setattr(
            "app.prediction.loader.importlib.import_module",
            self._fake_import_module,
        )

        import builtins
        _original_import = builtins.__import__

        def _patched_import(name, *args, **kwargs):
            if name == "mlflow":
                return SimpleNamespace(
                    set_tracking_uri=lambda uri: None,
                    set_registry_uri=lambda uri: None,
                )
            return _original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _patched_import)

        class _FakeHistoryService:
            def __init__(self, tracking_uri=None):
                self.tracking_uri = tracking_uri

            def get_run_detail(self, run_id):
                return SimpleNamespace(
                    run_id=run_id,
                    task_type="classification",
                    target_column="target",
                    experiment_name="autotabml-experiments",
                    run_name="experiment-classification-demo",
                )

        monkeypatch.setattr("app.prediction.loader.HistoryService", _FakeHistoryService)

        loader = MLflowModelLoader()
        loaded = loader.load(
            PredictionRequest(
                source_type=ModelSourceType.MLFLOW_RUN_MODEL,
                model_uri="runs:/run-123/model",
            )
        )

        assert loaded.load_reference == "runs:/run-123/model"
        assert loaded.task_type == PredictionTaskType.CLASSIFICATION
        assert loaded.metadata["run_id"] == "run-123"

    def test_loads_registered_model_via_registry_lookup(self, monkeypatch):
        monkeypatch.setattr("app.prediction.loader.is_mlflow_available", lambda: True)
        monkeypatch.setattr(
            "app.prediction.loader.importlib.import_module",
            self._fake_import_module,
        )

        import builtins
        _original_import = builtins.__import__

        def _patched_import(name, *args, **kwargs):
            if name == "mlflow":
                return SimpleNamespace(
                    set_tracking_uri=lambda uri: None,
                    set_registry_uri=lambda uri: None,
                )
            return _original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _patched_import)

        class _FakeRegistryService:
            def __init__(self, tracking_uri=None, registry_uri=None):
                self.tracking_uri = tracking_uri
                self.registry_uri = registry_uri

            def get_version_by_alias(self, model_name, alias):
                assert model_name == "pricing-model"
                assert alias == "champion"
                return SimpleNamespace(
                    version="4",
                    aliases=["champion"],
                    run_id="run-999",
                    source="runs:/run-999/model",
                    app_status="champion",
                )

        class _FakeHistoryService:
            def __init__(self, tracking_uri=None):
                self.tracking_uri = tracking_uri

            def get_run_detail(self, run_id):
                return SimpleNamespace(
                    run_id=run_id,
                    task_type="regression",
                    target_column="price",
                    experiment_name="autotabml-experiments",
                    run_name="experiment-regression-demo",
                )

        monkeypatch.setattr("app.prediction.loader.RegistryService", _FakeRegistryService)
        monkeypatch.setattr("app.prediction.loader.HistoryService", _FakeHistoryService)

        loader = MLflowModelLoader(registry_enabled=True)
        loaded = loader.load(
            PredictionRequest(
                source_type=ModelSourceType.MLFLOW_REGISTERED_MODEL,
                registry_model_name="pricing-model",
                registry_alias="champion",
            )
        )

        assert loaded.load_reference == "runs:/run-999/model"
        assert loaded.task_type == PredictionTaskType.REGRESSION
        assert loaded.metadata["registry_version"] == "4"
        assert loaded.metadata["resolved_source"] == "runs:/run-999/model"

    def test_load_pyfunc_sets_tracking_and_registry_uri(self, monkeypatch):
        """Regression: _load_pyfunc_model must call set_tracking_uri/set_registry_uri."""

        set_calls: dict[str, list[str]] = {"tracking": [], "registry": []}

        fake_mlflow = SimpleNamespace(
            set_tracking_uri=lambda uri: set_calls["tracking"].append(uri),
            set_registry_uri=lambda uri: set_calls["registry"].append(uri),
        )
        fake_pyfunc = SimpleNamespace(
            load_model=lambda model_uri, dst_path=None: {"uri": model_uri},
        )

        monkeypatch.setattr("app.prediction.loader.is_mlflow_available", lambda: True)

        import importlib as real_importlib

        def fake_import(name):
            if name == "mlflow.pyfunc":
                return fake_pyfunc
            if name == "mlflow":
                return fake_mlflow
            return real_importlib.import_module(name)

        monkeypatch.setattr("app.prediction.loader.importlib.import_module", fake_import)
        import app.prediction.loader as loader_mod
        monkeypatch.setattr(loader_mod, "mlflow", fake_mlflow, raising=False)

        # Patch `import mlflow` inside _load_pyfunc_model
        import builtins
        original_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "mlflow":
                return fake_mlflow
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)

        loader = MLflowModelLoader(
            tracking_uri="sqlite:///test.db",
            registry_uri="sqlite:///test-reg.db",
        )
        loader._load_pyfunc_model("runs:/abc123/model")

        assert set_calls["tracking"] == ["sqlite:///test.db"]
        assert set_calls["registry"] == ["sqlite:///test-reg.db"]


class TestPredictionValidation:
    def test_single_row_normalization(self):
        normalized = normalize_single_row_input({"a": 1, "b": 2})

        assert normalized.shape == (1, 2)

    def test_empty_single_row_payload_fails(self):
        with pytest.raises(PredictionValidationError, match="must not be empty"):
            normalize_single_row_input({})

    def test_missing_columns_fail_validation(self):
        loaded_model = _make_loaded_model(feature_columns=["a", "b"])
        dataframe = pd.DataFrame({"a": [1.0]})

        _, validation = validate_prediction_dataframe(
            dataframe,
            loaded_model,
            validation_mode=SchemaValidationMode.STRICT,
        )

        assert validation.can_score is False
        assert validation.missing_columns == ["b"]

    def test_extra_columns_warn_and_are_dropped_for_scoring(self):
        loaded_model = _make_loaded_model(feature_columns=["a", "b"])
        dataframe = pd.DataFrame({"a": [1.0], "b": [2], "extra": [3]})

        normalized, validation = validate_prediction_dataframe(
            dataframe,
            loaded_model,
            validation_mode=SchemaValidationMode.WARN,
        )

        assert validation.can_score is True
        assert validation.unexpected_columns == ["extra"]
        assert list(normalized.columns) == ["a", "b"]
        assert any(issue.severity == PredictionValidationSeverity.WARNING for issue in validation.issues)


class TestPredictionService:
    def test_single_row_prediction_returns_normalized_result(self, tmp_path: Path, monkeypatch):
        service = PredictionService(
            artifacts_dir=tmp_path / "artifacts",
            history_path=tmp_path / "history.jsonl",
            schema_validation_mode=SchemaValidationMode.STRICT,
            prediction_column_name="prediction",
            prediction_score_column_name="prediction_score",
            local_model_dirs=[],
            local_metadata_dirs=[],
        )
        loaded_model = _make_loaded_model()
        monkeypatch.setattr(service, "load_model", lambda request: loaded_model)

        result = service.predict_single(
            SingleRowPredictionRequest(
                source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                model_identifier="demo-model",
                task_type_hint=PredictionTaskType.CLASSIFICATION,
                row_data={"a": 0.75, "b": 3},
            )
        )

        assert isinstance(result, PredictionResult)
        assert result.predicted_label == "yes"
        assert result.predicted_score == pytest.approx(0.8)
        assert result.artifacts is not None
        assert result.artifacts.scored_csv_path is not None

    def test_batch_prediction_preserves_original_columns(self, tmp_path: Path, monkeypatch):
        service = PredictionService(
            artifacts_dir=tmp_path / "artifacts",
            history_path=tmp_path / "history.jsonl",
            schema_validation_mode=SchemaValidationMode.WARN,
            prediction_column_name="prediction",
            prediction_score_column_name="prediction_score",
            local_model_dirs=[],
            local_metadata_dirs=[],
        )
        loaded_model = _make_loaded_model()
        monkeypatch.setattr(service, "load_model", lambda request: loaded_model)

        result = service.predict_batch(
            BatchPredictionRequest(
                source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                model_identifier="demo-model",
                dataframe=pd.DataFrame({"a": [0.9, 0.1], "b": [1, 2], "extra": [5, 6]}),
                dataset_name="batch.csv",
                input_source_label="batch.csv",
                schema_validation_mode=SchemaValidationMode.WARN,
            )
        )

        assert list(result.scored_dataframe.columns) == ["a", "b", "extra", "prediction", "prediction_score"]
        assert result.summary.rows_scored == 2
        assert result.artifacts is not None
        assert result.artifacts.scored_csv_path is not None
        assert result.history_entry is not None

    def test_load_model_logs_resolved_run_context(self, tmp_path: Path, monkeypatch, caplog):
        service = PredictionService(
            artifacts_dir=tmp_path / "artifacts",
            history_path=tmp_path / "history.jsonl",
            schema_validation_mode=SchemaValidationMode.STRICT,
            prediction_column_name="prediction",
            prediction_score_column_name="prediction_score",
            local_model_dirs=[],
            local_metadata_dirs=[],
        )
        loaded_model = _make_loaded_model().model_copy(
            update={
                "metadata": {
                    "feature_dtypes": {"a": "float64", "b": "int64"},
                    "run_id": "run-123",
                    "experiment_name": "demo-exp",
                    "run_name": "nightly",
                }
            }
        )
        monkeypatch.setattr(service._local_loader, "load", lambda request: loaded_model)

        caplog.set_level(logging.INFO, logger="app.prediction.base")
        install_correlation_filter(logging.getLogger())

        result = service.load_model(
            PredictionRequest(
                source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                model_identifier="demo-model",
            )
        )

        assert result is loaded_model
        loaded_records = [record for record in caplog.records if record.getMessage() == "prediction_model_loaded"]
        assert loaded_records
        assert getattr(loaded_records[-1], "run_id", None) == "run-123"
        assert getattr(loaded_records[-1], "experiment_name", None) == "demo-exp"

    def test_batch_prediction_emits_metrics_and_correlated_logs(self, tmp_path: Path, monkeypatch, caplog):
        service = PredictionService(
            artifacts_dir=tmp_path / "artifacts",
            history_path=tmp_path / "history.jsonl",
            schema_validation_mode=SchemaValidationMode.WARN,
            prediction_column_name="prediction",
            prediction_score_column_name="prediction_score",
            local_model_dirs=[],
            local_metadata_dirs=[],
        )
        loaded_model = _make_loaded_model().model_copy(
            update={
                "metadata": {
                    "feature_dtypes": {"a": "float64", "b": "int64"},
                    "run_id": "run-456",
                    "experiment_name": "demo-exp",
                }
            }
        )
        monkeypatch.setattr(service, "load_model", lambda request: loaded_model)

        backend = InMemoryMetricsBackend()
        previous_backend = set_metrics_backend(backend)
        caplog.set_level(logging.INFO, logger="app.prediction.base")
        install_correlation_filter(logging.getLogger())
        try:
            service.predict_batch(
                BatchPredictionRequest(
                    source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                    model_identifier="demo-model",
                    dataframe=pd.DataFrame({"a": [0.9, 0.1], "b": [1, 2], "extra": [5, 6]}),
                    dataset_name="batch.csv",
                    input_source_label="batch.csv",
                    schema_validation_mode=SchemaValidationMode.WARN,
                )
            )
        finally:
            set_metrics_backend(previous_backend)

        completed = [record for record in caplog.records if record.getMessage() == "prediction_batch_completed"]
        assert completed
        assert getattr(completed[-1], "run_id", None) == "run-456"
        assert any(
            key[0] == "prediction_requests_total"
            and dict(key[1])["mode"] == "batch"
            and dict(key[1])["source_type"] == "local_saved_model"
            for key in backend.counters
        )
        assert any(
            key[0] == "prediction_duration_seconds"
            and dict(key[1])["status"] == "success"
            and dict(key[1])["mode"] == "batch"
            for key in backend.histograms
        )


class TestPredictionArtifactsAndHistory:
    def test_artifacts_use_output_stem(self, tmp_path: Path):
        loaded_model = _make_loaded_model(task_type=PredictionTaskType.REGRESSION, native_model=_FakeRegressor())
        summary = PredictionSummary(
            mode=PredictionMode.BATCH,
            source_type=ModelSourceType.LOCAL_SAVED_MODEL,
            task_type=PredictionTaskType.REGRESSION,
            model_identifier="regressor",
            input_source="batch.csv",
            input_row_count=2,
            rows_scored=2,
            prediction_column="prediction",
            prediction_score_column=None,
            validation_mode=SchemaValidationMode.STRICT,
        )
        artifacts = write_prediction_artifacts(
            loaded_model=loaded_model,
            scored_dataframe=pd.DataFrame({"a": [1.0], "prediction": [10.0]}),
            summary=summary,
            output_dir=tmp_path,
            output_stem="batch_scores",
        )

        assert artifacts.scored_csv_path is not None
        assert "batch_scores" in artifacts.scored_csv_path.name
        assert artifacts.summary_json_path is not None
        assert artifacts.metadata_json_path is not None
        assert artifacts.markdown_summary_path is not None
        markdown = artifacts.markdown_summary_path.read_text(encoding="utf-8")
        assert str(artifacts.scored_csv_path) in markdown

    def test_history_store_appends_and_lists_recent(self, tmp_path: Path):
        store = PredictionHistoryStore(tmp_path / "history.jsonl")
        now = datetime.now(timezone.utc)
        older = now - timedelta(hours=1)
        store.append(
            PredictionHistoryEntry(
                job_id="older",
                timestamp=older,
                status=PredictionStatus.SUCCESS,
                mode=PredictionMode.BATCH,
                model_source=ModelSourceType.LOCAL_SAVED_MODEL,
                model_identifier="model-a",
                task_type=PredictionTaskType.CLASSIFICATION,
                input_source="older.csv",
                row_count=10,
            )
        )
        store.append(
            PredictionHistoryEntry(
                job_id="newer",
                timestamp=now,
                status=PredictionStatus.SUCCESS,
                mode=PredictionMode.SINGLE_ROW,
                model_source=ModelSourceType.MLFLOW_RUN_MODEL,
                model_identifier="model-b",
                task_type=PredictionTaskType.REGRESSION,
                input_source="manual_row",
                row_count=1,
            )
        )

        recent = store.list_recent(limit=2)

        assert [entry.job_id for entry in recent] == ["newer", "older"]

    def test_history_write_warning_is_exposed_in_summary(self, tmp_path: Path, monkeypatch):
        service = PredictionService(
            artifacts_dir=tmp_path / "artifacts",
            history_path=tmp_path / "history.jsonl",
            schema_validation_mode=SchemaValidationMode.STRICT,
            prediction_column_name="prediction",
            prediction_score_column_name="prediction_score",
            local_model_dirs=[],
            local_metadata_dirs=[],
        )
        loaded_model = _make_loaded_model()
        monkeypatch.setattr(service, "load_model", lambda request: loaded_model)

        def _raise_history_error(entry):  # noqa: ANN001, ARG001
            raise OSError("disk full")

        monkeypatch.setattr(service._history_store, "append", _raise_history_error)

        result = service.predict_single(
            SingleRowPredictionRequest(
                source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                model_identifier="demo-model",
                task_type_hint=PredictionTaskType.CLASSIFICATION,
                row_data={"a": 0.75, "b": 3},
            )
        )

        assert any("Prediction history could not be written" in warning for warning in result.warnings)
        assert any("Prediction history could not be written" in warning for warning in result.summary.warnings)


class TestPredictionCLI:
    def test_predict_single_cli_uses_service(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr(cli_module, "load_settings", lambda: AppSettings())

        loaded_model = _make_loaded_model()
        result = PredictionResult(
            loaded_model=loaded_model,
            validation=PredictionValidationResult(),
            summary=PredictionSummary(
                mode=PredictionMode.SINGLE_ROW,
                source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                task_type=PredictionTaskType.CLASSIFICATION,
                model_identifier="demo-model",
                input_source="manual_row",
                input_row_count=1,
                rows_scored=1,
                prediction_column="prediction",
                prediction_score_column="prediction_score",
                validation_mode=SchemaValidationMode.STRICT,
            ),
            artifacts=PredictionArtifactBundle(),
            predicted_label="yes",
            predicted_score=0.88,
            scored_row={"prediction": "yes", "prediction_score": 0.88},
        )

        class _FakePredictionService:
            def predict_single(self, request):
                return result

        monkeypatch.setattr(cli_module, "_build_prediction_service", lambda settings: _FakePredictionService())

        args = type(
            "Args",
            (),
            {
                "model_source": "local_saved_model",
                "model_id": "demo-model",
                "model_path": None,
                "model_uri": None,
                "metadata_path": None,
                "task_type": "classification",
                "schema_mode": None,
                "run_id": None,
                "artifact_path": None,
                "model_name": None,
                "model_version": None,
                "model_alias": None,
                "output_dir": None,
                "output_stem": None,
                "row_json": '{"a": 1, "b": 2}',
                "row_file": None,
            },
        )()

        cli_module.cmd_predict_single(args)
        output = capsys.readouterr().out

        assert "Prediction: demo-model" in output
        assert "Predicted label: yes" in output