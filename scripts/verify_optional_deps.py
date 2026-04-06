#!/usr/bin/env python
"""Verify optional dependency runtime paths with real data — no mocks.

This script exercises each optional dependency path with a real dataset
and prints a structured pass/fail report. Run it after installing extras
to confirm actual runtime behavior.

Usage:
    python scripts/verify_optional_deps.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------

def _make_iris_df() -> pd.DataFrame:
    from sklearn.datasets import load_iris
    return load_iris(as_frame=True).frame


def _section(title: str) -> None:
    print()
    print(f"{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _result(label: str, ok: bool, detail: str = "") -> dict:
    status = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return {"label": label, "ok": ok, "detail": detail}


# ---------------------------------------------------------------------------
# 1. GX Validation
# ---------------------------------------------------------------------------

def verify_gx(df: pd.DataFrame) -> list[dict]:
    _section("Great Expectations (GX) Validation")
    results = []

    # Probe
    try:
        from app.validation.gx_context import is_gx_available
        avail = is_gx_available()
        results.append(_result("GX importable", avail, f"is_gx_available()={avail}"))
    except Exception as exc:
        results.append(_result("GX importable", False, str(exc)))
        return results

    if not avail:
        results.append(_result("GX validation", False, "GX not available"))
        return results

    # App-native rules (always work)
    from app.validation.schemas import ValidationRuleConfig
    from app.validation.service import validate_dataset

    config = ValidationRuleConfig(target_column="target")
    try:
        summary, artifacts = validate_dataset(df, config, dataset_name="gx_verify_native")
        ok = summary.failed_count == 0
        results.append(_result(
            "App-native validation",
            ok,
            f"{summary.passed_count} passed, {summary.warning_count} warnings, {summary.failed_count} failed",
        ))
    except Exception as exc:
        results.append(_result("App-native validation", False, str(exc)))

    return results


# ---------------------------------------------------------------------------
# 2. ydata-profiling
# ---------------------------------------------------------------------------

def verify_ydata(df: pd.DataFrame) -> list[dict]:
    _section("ydata-profiling")
    results = []

    try:
        from app.profiling.ydata_runner import is_ydata_available
        avail = is_ydata_available()
        results.append(_result("ydata importable", avail))
    except Exception as exc:
        results.append(_result("ydata importable", False, str(exc)))
        return results

    if not avail:
        return results

    from app.config.models import ProfilingMode
    from app.profiling.service import profile_dataset

    try:
        t0 = time.monotonic()
        summary, artifacts = profile_dataset(
            df,
            dataset_name="ydata_verify",
            mode=ProfilingMode.MINIMAL,
            artifacts_dir=Path("artifacts/tmp"),
        )
        elapsed = round(time.monotonic() - t0, 1)
        results.append(_result(
            "Real profiling run",
            True,
            f"{summary.row_count} rows, {summary.column_count} cols in {elapsed}s",
        ))
    except Exception as exc:
        results.append(_result("Real profiling run", False, str(exc)))

    return results


# ---------------------------------------------------------------------------
# 3. LazyPredict + MLflow
# ---------------------------------------------------------------------------

def verify_lazypredict_mlflow(df: pd.DataFrame, tracking_uri: str | None, registry_uri: str | None) -> list[dict]:
    _section("LazyPredict Benchmark + MLflow")
    results = []

    # Probe LazyPredict
    try:
        from app.modeling.benchmark.lazypredict_runner import is_lazypredict_available
        avail = is_lazypredict_available()
        results.append(_result("LazyPredict importable", avail))
    except Exception as exc:
        results.append(_result("LazyPredict importable", False, str(exc)))
        return results

    if not avail:
        return results

    # Probe MLflow
    try:
        from app.tracking.mlflow_query import is_mlflow_available
        mlflow_ok = is_mlflow_available()
        results.append(_result("MLflow importable", mlflow_ok))
    except Exception as exc:
        results.append(_result("MLflow importable", False, str(exc)))

    # Real benchmark run
    from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkTaskType
    from app.modeling.benchmark.service import benchmark_dataset

    config = BenchmarkConfig(
        target_column="target",
        task_type=BenchmarkTaskType.CLASSIFICATION,
        include_models=["DummyClassifier"],
    )
    try:
        t0 = time.monotonic()
        result = benchmark_dataset(
            df,
            config=config,
            dataset_name="lazypredict_verify",
            artifacts_dir=Path("artifacts/tmp"),
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            mlflow_experiment_name="verify-optional-deps",
        )
        elapsed = round(time.monotonic() - t0, 1)
        results.append(_result(
            "Real benchmark run",
            True,
            f"best={result.summary.best_model_name} score={result.summary.best_score:.4f} in {elapsed}s",
        ))
        if result.mlflow_run_id:
            results.append(_result("MLflow run logged", True, f"run_id={result.mlflow_run_id}"))
        else:
            results.append(_result("MLflow run logged", False, "no run_id in bundle"))
    except Exception as exc:
        results.append(_result("Real benchmark run", False, str(exc)))

    return results


# ---------------------------------------------------------------------------
# 4. PyCaret
# ---------------------------------------------------------------------------

def verify_pycaret() -> list[dict]:
    _section("PyCaret Experiment Lab")
    results = []

    try:
        from app.modeling.pycaret.setup_runner import _probe_pycaret_import_error
        err = _probe_pycaret_import_error()
        if err:
            results.append(_result("PyCaret importable", False, err))
        else:
            results.append(_result("PyCaret importable", True))
    except Exception as exc:
        results.append(_result("PyCaret importable", False, str(exc)))

    return results


# ---------------------------------------------------------------------------
# 5. MLflow History / Registry
# ---------------------------------------------------------------------------

def verify_mlflow_query(tracking_uri: str | None, registry_uri: str | None) -> list[dict]:
    _section("MLflow History & Registry")
    results = []

    try:
        from app.tracking.mlflow_query import is_mlflow_available
        if not is_mlflow_available():
            results.append(_result("MLflow available", False))
            return results
        results.append(_result("MLflow available", True))
    except Exception as exc:
        results.append(_result("MLflow available", False, str(exc)))
        return results

    from app.tracking.history_service import HistoryService

    try:
        svc = HistoryService(tracking_uri=tracking_uri)
        runs = svc.list_runs(limit=3)
        results.append(_result("History list_runs", True, f"{len(runs)} run(s) returned"))
    except Exception as exc:
        results.append(_result("History list_runs", False, str(exc)))

    from app.registry.registry_service import RegistryService

    try:
        reg = RegistryService(tracking_uri=tracking_uri, registry_uri=registry_uri)
        models = reg.list_models()
        results.append(_result("Registry list_models", True, f"{len(models)} model(s) returned"))
    except Exception as exc:
        results.append(_result("Registry list_models", False, str(exc)))

    return results


# ---------------------------------------------------------------------------
# 6. Prediction (MLflow model)
# ---------------------------------------------------------------------------

def verify_prediction(df: pd.DataFrame, tracking_uri: str | None, registry_uri: str | None) -> list[dict]:
    _section("Prediction (MLflow registered model)")
    results = []

    try:
        from app.tracking.mlflow_query import is_mlflow_available
        if not is_mlflow_available():
            results.append(_result("MLflow available", False))
            return results
    except Exception:
        results.append(_result("MLflow available", False))
        return results

    from app.registry.registry_service import RegistryService

    try:
        reg = RegistryService(tracking_uri=tracking_uri, registry_uri=registry_uri)
        models = reg.list_models()
        iris_models = [m for m in models if "iris" in m.name.lower()]
        if not iris_models:
            results.append(_result("Iris model in registry", False, "no iris model found — skip predict"))
            return results
        model_name = iris_models[0].name
        results.append(_result("Iris model in registry", True, model_name))
    except Exception as exc:
        results.append(_result("Iris model in registry", False, str(exc)))
        return results

    from app.prediction.base import PredictionService
    from app.prediction.schemas import BatchPredictionRequest, ModelSourceType

    try:
        svc = PredictionService(
            artifacts_dir=Path("artifacts/predictions"),
            history_path=Path("artifacts/predictions/history.jsonl"),
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        features_df = df.drop(columns=["target"])
        request = BatchPredictionRequest(
            dataframe=features_df,
            source_type=ModelSourceType.MLFLOW_REGISTERED_MODEL,
            registry_model_name=model_name,
            registry_version="1",
            task_type_hint="classification",
            dataset_name="pred_verify",
        )
        pred_result = svc.predict_batch(request)
        n_scored = pred_result.summary.rows_scored
        results.append(_result("Batch prediction", True, f"{n_scored} rows scored"))
    except Exception as exc:
        results.append(_result("Batch prediction", False, str(exc)))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("  Optional Dependency Runtime Verification")
    print(f"  Python {sys.version.split()[0]}")
    print("=" * 60)

    from app.config.settings import load_settings
    settings = load_settings()
    tracking_uri = settings.mlflow.tracking_uri
    registry_uri = settings.mlflow.registry_uri
    print(f"  MLflow tracking: {tracking_uri}")
    print(f"  MLflow registry: {registry_uri}")

    df = _make_iris_df()
    all_results: list[dict] = []

    all_results.extend(verify_gx(df))
    all_results.extend(verify_ydata(df))
    all_results.extend(verify_lazypredict_mlflow(df, tracking_uri, registry_uri))
    all_results.extend(verify_pycaret())
    all_results.extend(verify_mlflow_query(tracking_uri, registry_uri))
    all_results.extend(verify_prediction(df, tracking_uri, registry_uri))

    # Summary
    _section("SUMMARY")
    passed = sum(1 for r in all_results if r["ok"])
    failed = sum(1 for r in all_results if not r["ok"])
    print(f"  {passed} passed, {failed} failed out of {len(all_results)} checks")
    print()

    if failed:
        print("  Failed checks:")
        unexpected_failures = 0
        for r in all_results:
            if not r["ok"]:
                is_expected = "pycaret" in r["label"].lower() and "No module" in str(r.get("detail", ""))
                tag = " (expected on Python 3.13)" if is_expected else ""
                print(f"    - {r['label']}: {r['detail']}{tag}")
                if not is_expected:
                    unexpected_failures += 1
        print()
        return 1 if unexpected_failures > 0 else 0

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
