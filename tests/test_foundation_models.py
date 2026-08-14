from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.modeling.foundation import (
    TABFM_CHECKPOINT,
    TIMESFM_CHECKPOINT,
    CheckpointResolver,
    ModelDownloadRequiredError,
    TabFMConfig,
    TabFMService,
    TimesFMConfig,
    TimesFMService,
)


class _Classifier:
    classes_ = np.array(["no", "yes"])

    def fit(self, X, y):  # noqa: ANN001, N803
        self.fit_rows = len(X)
        return self

    def predict(self, X):  # noqa: ANN001, N803
        return np.where(X["score"].to_numpy() >= 0.5, "yes", "no")

    def predict_proba(self, X):  # noqa: ANN001, N803
        p = X["score"].to_numpy()
        return np.column_stack([1 - p, p])


class _Regressor:
    def fit(self, X, y):  # noqa: ANN001, N803
        self.fit_rows = len(X)
        return self

    def predict(self, X):  # noqa: ANN001, N803
        return X["score"].to_numpy() * 10


class _TimesModel:
    def forecast(self, *, horizon, inputs):  # noqa: ANN001
        point = np.vstack([np.repeat(values[-1], horizon) for values in inputs])
        quantiles = np.stack([np.stack([np.linspace(value - 1, value + 1, 10) for value in row]) for row in point])
        return point, quantiles


def _cached_resolver(tmp_path: Path) -> CheckpointResolver:
    return CheckpointResolver(snapshot_download=lambda **kwargs: str(tmp_path / kwargs["repo_id"].replace("/", "_")))


def test_checkpoint_resolver_requires_confirmation_before_download(tmp_path: Path):
    calls: list[dict] = []

    def fake_download(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        if kwargs["local_files_only"]:
            raise FileNotFoundError
        return str(tmp_path / "downloaded")

    resolver = CheckpointResolver(snapshot_download=fake_download)

    with pytest.raises(ModelDownloadRequiredError):
        resolver.resolve(TABFM_CHECKPOINT, allow_download=False)

    path = resolver.resolve(TABFM_CHECKPOINT, allow_download=True)
    assert path == tmp_path / "downloaded"
    assert calls[-1]["revision"] == TABFM_CHECKPOINT.revision
    assert calls[-1]["repo_id"] == "google/tabfm-1.0.0-pytorch"


def test_tabfm_classification_run_is_licensed_sampled_and_revision_pinned(tmp_path: Path):
    dataframe = pd.DataFrame(
        {
            "score": np.linspace(0, 1, 40),
            "category": ["a", "b"] * 20,
            "target": ["no", "yes"] * 20,
        }
    )
    estimator = _Classifier()
    observed: dict = {}

    def factory(path, task_type, config):  # noqa: ANN001
        observed.update(path=path, task_type=task_type, config=config)
        return estimator

    service = TabFMService(
        checkpoint_resolver=_cached_resolver(tmp_path),
        estimator_factory=factory,
    )

    with pytest.raises(ValueError, match="non-commercial"):
        service.run(dataframe, TabFMConfig(target_column="target"), accept_license=False)

    result = service.run(
        dataframe,
        TabFMConfig(target_column="target", context_rows=12, n_estimators=4),
        accept_license=True,
        output_dir=tmp_path,
    )

    assert result.task_type == "classification"
    assert estimator.fit_rows == 12
    assert {"balanced_accuracy", "accuracy", "macro_f1", "log_loss"} <= result.metrics.keys()
    assert observed["path"] == tmp_path / "google_tabfm-1.0.0-pytorch"
    assert result.artifacts["predictions"].exists()
    assert result.artifacts["summary"].exists()


def test_tabfm_rejects_more_than_ten_classes(tmp_path: Path):
    dataframe = pd.DataFrame({"score": range(22), "target": [f"c{i % 11}" for i in range(22)]})
    service = TabFMService(
        checkpoint_resolver=_cached_resolver(tmp_path),
        estimator_factory=lambda *args: _Classifier(),
    )

    with pytest.raises(ValueError, match="at most 10 classes"):
        service.run(
            dataframe,
            TabFMConfig(target_column="target", task_type="classification"),
            accept_license=True,
        )


def test_tabfm_context_round_trip_has_checksums(tmp_path: Path):
    dataframe = pd.DataFrame({"score": np.arange(30, dtype=float), "target": np.arange(30, dtype=float) * 10})
    service = TabFMService(
        checkpoint_resolver=_cached_resolver(tmp_path),
        estimator_factory=lambda *args: _Regressor(),
    )
    result = service.run(
        dataframe,
        TabFMConfig(target_column="target", task_type="regression", context_rows=10),
        accept_license=True,
        output_dir=tmp_path,
    )

    saved = service.save_context(result, name="research-context", output_dir=tmp_path)
    loaded = service.load_context(saved.metadata_path, estimator_factory=lambda *args: _Regressor())

    assert saved.context_path.exists()
    assert saved.checksum_path.exists()
    assert loaded.feature_columns == ["score"]
    assert loaded.native_model.fit_rows == 10

    saved.context_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        service.load_context(saved.metadata_path, estimator_factory=lambda *args: _Regressor())


def test_timesfm_grouped_forecast_quantiles_backtest_and_revision(tmp_path: Path):
    timestamps = pd.date_range("2025-01-01", periods=50, freq="D")
    dataframe = pd.concat(
        [
            pd.DataFrame({"when": timestamps, "value": np.arange(50, dtype=float), "store": store})
            for store in ["north", "south"]
        ],
        ignore_index=True,
    )
    service = TimesFMService(
        checkpoint_resolver=_cached_resolver(tmp_path),
        model_factory=lambda *args: _TimesModel(),
    )

    result = service.run(
        dataframe,
        TimesFMConfig(
            timestamp_column="when",
            target_column="value",
            group_column="store",
            horizon=5,
            context_length=40,
        ),
        output_dir=tmp_path,
    )

    assert len(result.forecast) == 10
    assert {"forecast", "mean", "q10", "q50", "q90", "store"} <= set(result.forecast.columns)
    assert result.metrics["groups_forecast"] == 2
    assert {"mae", "rmse", "smape"} <= result.metrics.keys()
    assert result.checkpoint_path == tmp_path / "google_timesfm-2.5-200m-pytorch"
    assert result.artifacts["forecast"].exists()
    assert TIMESFM_CHECKPOINT.revision == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"


def test_timesfm_rejects_duplicate_timestamps(tmp_path: Path):
    dataframe = pd.DataFrame(
        {
            "when": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "value": [1, 2, 3],
        }
    )
    service = TimesFMService(
        checkpoint_resolver=_cached_resolver(tmp_path),
        model_factory=lambda *args: _TimesModel(),
    )

    with pytest.raises(ValueError, match="duplicate"):
        service.run(
            dataframe,
            TimesFMConfig(timestamp_column="when", target_column="value", horizon=1, min_context=2),
        )
