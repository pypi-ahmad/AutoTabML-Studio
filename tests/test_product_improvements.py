from pathlib import Path
import zipfile

import pandas as pd
import pytest

from app.autorun import AutoRunConfig, plan_auto_run, suggest_targets
from app.background_jobs import BackgroundJobService
from app.config.models import AppSettings
from app.deployment import export_deployment_bundle
from app.drift import DriftLevel, build_drift_baseline, compare_drift
from app.evaluation import evaluate_model
from app.explainability import explain_global, explain_prediction
from app.provenance import build_provenance, write_provenance
from app.storage import AppJobStatus, AppJobType, JobRecord, build_metadata_store


def test_drift_baseline_detects_shift_without_storing_rows():
    training = pd.DataFrame({"age": range(100), "segment": ["a"] * 50 + ["b"] * 50})
    baseline = build_drift_baseline(training)
    report = compare_drift(baseline, pd.DataFrame({"age": range(1000, 1100), "segment": ["z"] * 100}))
    assert report.level == DriftLevel.HIGH
    assert not hasattr(baseline, "dataframe")


def test_auto_run_plan_requires_confirmation_ready_target(monkeypatch):
    frame = pd.DataFrame({"id": [1, 2, 3, 4], "feature": [3, 4, 5, 6], "target": [0, 1, 0, 1]})
    monkeypatch.setattr("app.autorun.is_flaml_available", lambda: True)
    assert suggest_targets(frame)[0] == "target"
    plan = plan_auto_run(frame, AutoRunConfig(target_column="target"))
    assert plan.task_type == "classification"
    assert plan.engine == "flaml"


def test_background_job_cancel_is_persisted(tmp_path: Path):
    settings = AppSettings.model_validate({"artifacts": {"root_dir": str(tmp_path / "artifacts")}})
    store = build_metadata_store(settings)
    assert store is not None
    job_dir = tmp_path / "jobs" / "job-1"
    job_dir.mkdir(parents=True)
    store.record_job(
        JobRecord(
            job_id="job-1",
            job_type=AppJobType.FLAML,
            status=AppJobStatus.RUNNING,
            metadata={"job_dir": str(job_dir), "progress": 20},
        )
    )
    job = BackgroundJobService(store=store, jobs_dir=tmp_path / "jobs").cancel("job-1")
    assert job.status == AppJobStatus.CANCELLED
    assert store.get_job("job-1").status == AppJobStatus.CANCELLED


def test_evaluation_and_explanations_cover_classification():
    sklearn = pytest.importorskip("sklearn.linear_model")
    features = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "z": [1, 1, 0, 0, 1, 0]})
    target = pd.Series([0, 0, 0, 1, 1, 1])
    model = sklearn.LogisticRegression().fit(features, target)
    report = evaluate_model(model, features, target, task_type="classification")
    global_explanation = explain_global(model, features, target)
    local_explanation = explain_prediction(model, features.iloc[[0]], features)
    assert "balanced_accuracy" in report.metrics
    assert global_explanation.contributions
    assert local_explanation.method == "single_feature_perturbation"


def test_evaluation_covers_regression():
    sklearn = pytest.importorskip("sklearn.linear_model")
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([0.0, 2.0, 4.0, 6.0])
    report = evaluate_model(sklearn.LinearRegression().fit(features, target), features, target, task_type="regression")
    assert report.metrics["r2"] == 1.0


def test_provenance_and_deployment_bundle_are_portable(tmp_path: Path):
    model = tmp_path / "model.skops"
    metadata = tmp_path / "model.json"
    model.write_bytes(b"model")
    metadata.write_text("{}", encoding="utf-8")
    manifest = build_provenance(
        engine="lazypredict",
        task_type="classification",
        target_column="target",
        dataset_fingerprint="abc",
        row_count=10,
        column_count=3,
        random_seed=42,
        configuration={"token": "not-a-secret-field"},
        model_path=model,
    )
    provenance = write_provenance(manifest, tmp_path / "provenance.json")
    assert manifest.configuration["token"] == "[REDACTED]"
    bundle = export_deployment_bundle(
        model_path=model,
        metadata_path=metadata,
        provenance_path=provenance,
        output_path=tmp_path / "deploy",
    )
    with zipfile.ZipFile(bundle.archive_path) as archive:
        names = set(archive.namelist())
        predictor = archive.read("predict.py").decode()
    assert {"serve.py", "predict.py", "Dockerfile", "checksums.json"} <= names
    assert "PredictionService" in predictor


@pytest.mark.integration
def test_auto_run_saves_reloads_and_predicts(tmp_path: Path):
    pytest.importorskip("flaml")
    from app.autorun import AutoRunMode, run_auto_run
    from app.prediction import BatchPredictionRequest, ModelSourceType, PredictionService

    frame = pd.DataFrame(
        {
            "x": range(80),
            "z": [index % 3 for index in range(80)],
            "target": [0 if index < 40 else 1 for index in range(80)],
        }
    )
    result = run_auto_run(
        frame,
        AutoRunConfig(
            target_column="target",
            mode=AutoRunMode.BALANCED,
            time_budget=10,
            model_name="integration",
        ),
        artifacts_dir=tmp_path / "artifacts",
        models_dir=tmp_path / "models",
        job_id="integration",
    )
    service = PredictionService(
        artifacts_dir=tmp_path / "predictions",
        history_path=tmp_path / "history.jsonl",
        local_model_dirs=[tmp_path / "models"],
        local_metadata_dirs=[tmp_path / "models"],
        registry_enabled=False,
    )
    prediction = service.predict_batch(
        BatchPredictionRequest(
            source_type=ModelSourceType.LOCAL_SAVED_MODEL,
            model_path=result.model_path,
            metadata_path=result.metadata_path,
            dataframe=frame.drop(columns=["target"]).head(5),
        )
    )
    assert prediction.summary.rows_scored == 5
    assert "drift_report" in prediction.loaded_model.metadata
