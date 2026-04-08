"""Tests for the FLAML AutoML integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.config.enums import ExecutionBackend
from app.config.models import AppSettings, FlamlSettings
from app.modeling.flaml.errors import (
    FlamlAutoMLError,
    FlamlDependencyError,
    FlamlExecutionError,
    FlamlTargetError,
)
from app.modeling.flaml.schemas import (
    DEFAULT_ESTIMATOR_LIST,
    FlamlConfig,
    FlamlLeaderboardRow,
    FlamlResultBundle,
    FlamlSavedModelMetadata,
    FlamlSearchConfig,
    FlamlSearchResult,
    FlamlSortDirection,
    FlamlSummary,
    FlamlTaskType,
)
from app.modeling.flaml.service import FlamlAutoMLService, metric_sort_direction
from app.modeling.flaml.setup_runner import resolve_task_type


@pytest.fixture
def classification_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_num": [1, 2, 3, 4, 5, 6],
            "feature_cat": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_num": [1, 2, 3, 4, 5, 6],
            "feature_cat": ["a", "b", "c", "d", "e", "f"],
            "target": [10.1, 20.2, 30.3, 40.4, 50.5, 60.6],
        }
    )


class _FakeAutoML:
    """Minimal mock of flaml.AutoML for unit tests."""

    def __init__(self) -> None:
        self.fit_kwargs: dict = {}
        self._best_estimator = "lgbm"
        self._best_config = {"n_estimators": 10, "learning_rate": 0.1}
        self._best_loss = 0.05
        self._best_config_train_time = 1.23
        self._time_to_find_best_model = 5.67
        self._best_loss_per_estimator = {
            "lgbm": 0.05,
            "rf": 0.08,
            "xgboost": 0.06,
        }
        self._best_config_per_estimator = {
            "lgbm": {"n_estimators": 10},
            "rf": {"n_estimators": 50},
            "xgboost": {"n_estimators": 20},
        }

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs

    @property
    def best_estimator(self):
        return self._best_estimator

    @property
    def best_config(self):
        return self._best_config

    @property
    def best_loss(self):
        return self._best_loss

    @property
    def best_config_train_time(self):
        return self._best_config_train_time

    @property
    def time_to_find_best_model(self):
        return self._time_to_find_best_model

    @property
    def best_loss_per_estimator(self):
        return self._best_loss_per_estimator

    @property
    def best_config_per_estimator(self):
        return self._best_config_per_estimator

    def predict(self, X):
        return [0] * len(X)


def _make_service(monkeypatch):
    monkeypatch.setattr("app.modeling.flaml.setup_runner.is_flaml_available", lambda: True)
    monkeypatch.setattr("app.modeling.flaml.service.require_flaml", lambda: None)
    return FlamlAutoMLService(
        artifacts_dir=None,
        models_dir=None,
        mlflow_experiment_name=None,
    )


# ── Schema tests ───────────────────────────────────────────────────────


class TestFlamlSchemas:
    def test_task_type_enum_values(self):
        assert FlamlTaskType.AUTO.value == "auto"
        assert FlamlTaskType.CLASSIFICATION.value == "classification"
        assert FlamlTaskType.REGRESSION.value == "regression"

    def test_default_search_config(self):
        config = FlamlSearchConfig()
        assert config.time_budget == 120
        assert config.metric == "auto"
        assert config.n_splits == 5
        assert config.ensemble is False
        assert config.early_stop is False

    def test_flaml_config_validation(self):
        config = FlamlConfig(target_column="target")
        assert config.target_column == "target"
        assert config.task_type == FlamlTaskType.AUTO

    def test_search_result_model(self):
        result = FlamlSearchResult(
            best_estimator="lgbm",
            best_config={"n": 10},
            best_loss=0.05,
            metric="accuracy",
        )
        assert result.best_estimator == "lgbm"
        assert result.best_loss == 0.05

    def test_leaderboard_row(self):
        row = FlamlLeaderboardRow(
            rank=1,
            estimator_name="lgbm",
            best_loss=0.05,
        )
        assert row.rank == 1
        assert row.estimator_name == "lgbm"

    def test_saved_model_metadata(self):
        meta = FlamlSavedModelMetadata(
            task_type=FlamlTaskType.CLASSIFICATION,
            target_column="target",
            model_name="test_model",
            model_path=Path("test.pkl"),
            framework="flaml",
        )
        assert meta.framework == "flaml"
        assert meta.task_type == FlamlTaskType.CLASSIFICATION

    def test_sort_direction_enum(self):
        assert FlamlSortDirection.ASCENDING.value == "ascending"
        assert FlamlSortDirection.DESCENDING.value == "descending"


# ── Settings tests ─────────────────────────────────────────────────────


class TestFlamlSettings:
    def test_default_flaml_settings(self):
        settings = FlamlSettings()
        assert settings.default_time_budget == 120
        assert settings.default_n_splits == 5
        assert settings.default_seed == 0
        assert settings.default_classification_metric == "accuracy"
        assert settings.default_regression_metric == "r2"
        assert len(settings.default_estimator_list) > 0
        assert settings.mlflow_experiment_name == "autotabml-flaml"

    def test_flaml_settings_in_app_settings(self):
        app_settings = AppSettings()
        assert hasattr(app_settings, "flaml")
        assert isinstance(app_settings.flaml, FlamlSettings)
        assert app_settings.flaml.default_time_budget == 120


# ── Setup runner tests ─────────────────────────────────────────────────


class TestFlamlSetupRunner:
    def test_resolve_task_type_classification(self, classification_df):
        task_type, warnings = resolve_task_type(
            classification_df["target"],
            FlamlTaskType.CLASSIFICATION,
        )
        assert task_type == FlamlTaskType.CLASSIFICATION

    def test_resolve_task_type_regression(self, regression_df):
        task_type, warnings = resolve_task_type(
            regression_df["target"],
            FlamlTaskType.REGRESSION,
        )
        assert task_type == FlamlTaskType.REGRESSION

    def test_resolve_task_type_auto_detects(self, classification_df):
        task_type, warnings = resolve_task_type(
            classification_df["target"],
            FlamlTaskType.AUTO,
        )
        assert task_type in (FlamlTaskType.CLASSIFICATION, FlamlTaskType.REGRESSION)
        assert any("auto-detected" in w.lower() for w in warnings)


# ── Metric helpers ─────────────────────────────────────────────────────


class TestMetricSortDirection:
    def test_higher_is_better_metrics(self):
        assert metric_sort_direction("accuracy") == FlamlSortDirection.DESCENDING
        assert metric_sort_direction("r2") == FlamlSortDirection.DESCENDING
        assert metric_sort_direction("f1") == FlamlSortDirection.DESCENDING

    def test_lower_is_better_metrics(self):
        assert metric_sort_direction("mae") == FlamlSortDirection.ASCENDING
        assert metric_sort_direction("mse") == FlamlSortDirection.ASCENDING
        assert metric_sort_direction("rmse") == FlamlSortDirection.ASCENDING
        assert metric_sort_direction("log_loss") == FlamlSortDirection.ASCENDING


# ── Service tests ──────────────────────────────────────────────────────


class TestFlamlService:
    def test_run_automl_classification(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch)
        fake = _FakeAutoML()

        with patch("flaml.AutoML", return_value=fake):
            bundle = service.run_automl(
                classification_df,
                FlamlConfig(target_column="target", task_type=FlamlTaskType.CLASSIFICATION),
                dataset_name="test_cls",
                execution_backend=ExecutionBackend.LOCAL,
            )

        assert bundle.task_type == FlamlTaskType.CLASSIFICATION
        assert bundle.search_result is not None
        assert bundle.search_result.best_estimator == "lgbm"
        assert bundle.search_result.best_loss == 0.05
        assert len(bundle.search_result.leaderboard) == 3
        assert bundle.feature_columns == ["feature_num", "feature_cat"]
        assert bundle.summary.dataset_name == "test_cls"

    def test_run_automl_regression(self, regression_df, monkeypatch):
        service = _make_service(monkeypatch)
        fake = _FakeAutoML()
        fake._best_estimator = "rf"
        fake._best_loss = 0.02

        with patch("flaml.AutoML", return_value=fake):
            bundle = service.run_automl(
                regression_df,
                FlamlConfig(target_column="target", task_type=FlamlTaskType.REGRESSION),
                dataset_name="test_reg",
                execution_backend=ExecutionBackend.LOCAL,
            )

        assert bundle.task_type == FlamlTaskType.REGRESSION
        assert bundle.search_result.best_estimator == "rf"
        assert bundle.search_result.best_loss == 0.02

    def test_run_automl_auto_detects_task(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch)
        fake = _FakeAutoML()

        with patch("flaml.AutoML", return_value=fake):
            bundle = service.run_automl(
                classification_df,
                FlamlConfig(target_column="target", task_type=FlamlTaskType.AUTO),
                dataset_name="test_auto",
            )

        assert bundle.task_type in (FlamlTaskType.CLASSIFICATION, FlamlTaskType.REGRESSION)

    def test_run_automl_missing_target_raises(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch)
        with pytest.raises(FlamlTargetError, match="not found"):
            service.run_automl(
                classification_df,
                FlamlConfig(target_column="nonexistent"),
            )

    def test_run_automl_non_local_backend_raises(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch)
        with pytest.raises(FlamlExecutionError, match="not implemented"):
            service.run_automl(
                classification_df,
                FlamlConfig(target_column="target"),
                execution_backend=ExecutionBackend.COLAB_MCP,
            )

    def test_run_automl_drops_null_targets(self, monkeypatch):
        service = _make_service(monkeypatch)
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4],
            "target": [0, None, 1, 0],
        })
        fake = _FakeAutoML()

        with patch("flaml.AutoML", return_value=fake):
            bundle = service.run_automl(
                df,
                FlamlConfig(target_column="target", task_type=FlamlTaskType.CLASSIFICATION),
            )

        assert any("dropped" in w.lower() for w in bundle.warnings)
        assert bundle.summary.source_row_count == 3

    def test_run_automl_all_null_targets_raises(self, monkeypatch):
        service = _make_service(monkeypatch)
        df = pd.DataFrame({
            "feature": [1, 2, 3],
            "target": [None, None, None],
        })
        with pytest.raises(FlamlTargetError, match="No rows remain"):
            service.run_automl(
                df,
                FlamlConfig(target_column="target"),
            )

    def test_run_automl_fit_kwargs(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch)
        fake = _FakeAutoML()

        with patch("flaml.AutoML", return_value=fake):
            service.run_automl(
                classification_df,
                FlamlConfig(
                    target_column="target",
                    task_type=FlamlTaskType.CLASSIFICATION,
                    search=FlamlSearchConfig(
                        time_budget=60,
                        metric="f1",
                        n_splits=3,
                        seed=42,
                        ensemble=True,
                    ),
                ),
            )

        assert fake.fit_kwargs["task"] == "classification"
        assert fake.fit_kwargs["time_budget"] == 60
        assert fake.fit_kwargs["metric"] == "f1"
        assert fake.fit_kwargs["n_splits"] == 3
        assert fake.fit_kwargs["seed"] == 42
        assert fake.fit_kwargs["ensemble"] is True

    def test_save_best_model(self, classification_df, monkeypatch, tmp_path):
        service = FlamlAutoMLService(
            artifacts_dir=None,
            models_dir=tmp_path,
            mlflow_experiment_name=None,
        )
        monkeypatch.setattr("app.modeling.flaml.setup_runner.is_flaml_available", lambda: True)
        monkeypatch.setattr("app.modeling.flaml.service.require_flaml", lambda: None)

        fake = _FakeAutoML()

        with patch("flaml.AutoML", return_value=fake):
            bundle = service.run_automl(
                classification_df,
                FlamlConfig(target_column="target", task_type=FlamlTaskType.CLASSIFICATION),
                dataset_name="test_save",
            )

        updated = service.save_best_model(bundle, save_name="TestModel")

        assert updated.saved_model_metadata is not None
        assert updated.saved_model_metadata.model_name == "TestModel"
        assert updated.saved_model_metadata.framework == "flaml"
        assert updated.summary.saved_model_name == "TestModel"
        assert Path(updated.saved_model_metadata.model_path).exists()

    def test_save_best_model_no_runtime_raises(self, monkeypatch):
        service = _make_service(monkeypatch)
        bundle = FlamlResultBundle(
            config=FlamlConfig(target_column="target"),
            task_type=FlamlTaskType.CLASSIFICATION,
            summary=FlamlSummary(
                target_column="target",
                task_type=FlamlTaskType.CLASSIFICATION,
                source_row_count=10,
                source_column_count=3,
                feature_column_count=2,
            ),
            runtime=None,
        )
        with pytest.raises(FlamlExecutionError, match="No FLAML runtime"):
            service.save_best_model(bundle, save_name="test")

    def test_leaderboard_is_sorted_by_loss(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch)
        fake = _FakeAutoML()

        with patch("flaml.AutoML", return_value=fake):
            bundle = service.run_automl(
                classification_df,
                FlamlConfig(target_column="target", task_type=FlamlTaskType.CLASSIFICATION),
            )

        lb = bundle.search_result.leaderboard
        assert len(lb) == 3
        losses = [row.best_loss for row in lb]
        assert losses == sorted(losses)
        assert lb[0].estimator_name == "lgbm"
        assert lb[0].rank == 1

    def test_custom_estimator_list(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch)
        fake = _FakeAutoML()

        with patch("flaml.AutoML", return_value=fake):
            service.run_automl(
                classification_df,
                FlamlConfig(
                    target_column="target",
                    task_type=FlamlTaskType.CLASSIFICATION,
                    search=FlamlSearchConfig(estimator_list=["lgbm", "rf"]),
                ),
            )

        assert fake.fit_kwargs["estimator_list"] == ["lgbm", "rf"]


# ── Error hierarchy tests ─────────────────────────────────────────────


class TestFlamlErrors:
    def test_error_hierarchy(self):
        assert issubclass(FlamlDependencyError, FlamlAutoMLError)
        assert issubclass(FlamlExecutionError, FlamlAutoMLError)
        assert issubclass(FlamlTargetError, FlamlAutoMLError)

    def test_errors_are_exceptions(self):
        with pytest.raises(FlamlAutoMLError):
            raise FlamlExecutionError("test")


# ── Page helpers ───────────────────────────────────────────────────────


class TestFlamlPageHelpers:
    def test_build_flaml_run_key(self):
        from app.pages.flaml_automl_page import _build_flaml_run_key

        key = _build_flaml_run_key("iris", "species", FlamlTaskType.CLASSIFICATION)
        assert key == "flaml::iris::species::classification"

    def test_metric_options_for_task(self):
        from app.pages.flaml_automl_page import _metric_options_for_task

        cls_metrics = _metric_options_for_task(FlamlTaskType.CLASSIFICATION)
        assert "accuracy" in cls_metrics
        assert "r2" not in cls_metrics

        reg_metrics = _metric_options_for_task(FlamlTaskType.REGRESSION)
        assert "r2" in reg_metrics
        assert "accuracy" not in reg_metrics

        auto_metrics = _metric_options_for_task(FlamlTaskType.AUTO)
        assert "accuracy" in auto_metrics
        assert "r2" in auto_metrics

    def test_estimator_labels_cover_defaults(self):
        from app.pages.flaml_automl_page import ESTIMATOR_LABELS

        for estimator in DEFAULT_ESTIMATOR_LIST:
            assert estimator in ESTIMATOR_LABELS


# ── Registry tests ─────────────────────────────────────────────────────


class TestFlamlPageRegistry:
    def test_flaml_page_is_registered(self):
        from app.pages.registry import get_page_registry

        pages = get_page_registry()
        labels = [page.label for page in pages]
        assert "FLAML AutoML" in labels

    def test_flaml_page_in_build_section(self):
        from app.pages.registry import get_page_registry

        pages = get_page_registry()
        flaml_page = next(p for p in pages if p.label == "FLAML AutoML")
        assert flaml_page.section == "build"
        assert flaml_page.module_path == "app.pages.flaml_automl_page"
        assert flaml_page.render_function == "render_flaml_automl_page"


# ── MLflow tracking tests ─────────────────────────────────────────────


class TestFlamlMLflowTracking:
    def test_tracker_skips_when_mlflow_unavailable(self, monkeypatch):
        monkeypatch.setattr("app.modeling.flaml.mlflow_tracking.is_mlflow_available", lambda: False)
        from app.modeling.flaml.mlflow_tracking import MLflowFlamlTracker

        tracker = MLflowFlamlTracker("test-experiment")
        bundle = FlamlResultBundle(
            config=FlamlConfig(target_column="target"),
            task_type=FlamlTaskType.CLASSIFICATION,
            summary=FlamlSummary(
                target_column="target",
                task_type=FlamlTaskType.CLASSIFICATION,
                source_row_count=10,
                source_column_count=3,
                feature_column_count=2,
            ),
        )
        run_id, warnings = tracker.log_flaml_bundle(bundle)
        assert run_id is None
        assert any("not installed" in w for w in warnings)


# ── Artifact tests ─────────────────────────────────────────────────────


class TestFlamlArtifacts:
    def test_write_artifacts_with_leaderboard(self, tmp_path):
        from app.modeling.flaml.artifacts import write_flaml_artifacts

        bundle = FlamlResultBundle(
            dataset_name="test",
            config=FlamlConfig(target_column="target"),
            task_type=FlamlTaskType.CLASSIFICATION,
            search_result=FlamlSearchResult(
                best_estimator="lgbm",
                best_loss=0.05,
                metric="accuracy",
                leaderboard=[
                    FlamlLeaderboardRow(rank=1, estimator_name="lgbm", best_loss=0.05),
                    FlamlLeaderboardRow(rank=2, estimator_name="rf", best_loss=0.08),
                ],
            ),
            summary=FlamlSummary(
                target_column="target",
                task_type=FlamlTaskType.CLASSIFICATION,
                source_row_count=100,
                source_column_count=5,
                feature_column_count=4,
            ),
        )

        artifact_bundle = write_flaml_artifacts(bundle, tmp_path)
        assert artifact_bundle.search_result_json_path is not None
        assert artifact_bundle.search_result_json_path.exists()
        assert artifact_bundle.leaderboard_csv_path is not None
        assert artifact_bundle.leaderboard_csv_path.exists()
        assert artifact_bundle.summary_json_path is not None
        assert artifact_bundle.summary_json_path.exists()
