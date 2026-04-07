"""Optional integration checks for heavy dependency imports.

These tests are marker-gated so the default unit suite stays lightweight.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from app.config.models import ProfilingMode
from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkTaskType
from app.modeling.benchmark.service import benchmark_dataset
from app.modeling.pycaret.errors import PyCaretDependencyError
from app.modeling.pycaret.setup_runner import require_pycaret
from app.profiling.service import profile_dataset
from app.validation.schemas import ValidationRuleConfig
from app.validation.service import validate_dataset

pytestmark = pytest.mark.integration


def _small_classification_df() -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    for index in range(40):
        rows.append(
            {
                "id": index,
                "feature_num": float(index),
                "feature_cat": "high" if index >= 20 else "low",
                "target": 1 if index >= 20 else 0,
            }
        )
    return pd.DataFrame(rows)


def _sqlite_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.as_posix()}"


@pytest.mark.parametrize(
    "module_name",
    [
        "mlflow",
        "lazypredict",
        "ydata_profiling",
    ],
)
def test_optional_dependency_imports_when_installed(module_name: str):
    pytest.importorskip(module_name)


def test_pycaret_imports_when_available():
    pytest.importorskip("pycaret.classification")


def test_gx_validation_executes_real_expectations(tmp_path: Path):
    pytest.importorskip("great_expectations")

    config = ValidationRuleConfig(
        uniqueness_columns=["id"],
        numeric_range_checks={"score": {"min": 0, "max": 100}},
        allowed_category_checks={"status": ["active", "inactive"]},
    )
    failing_df = pd.DataFrame(
        {
            "id": [1, 1, 3, 4],
            "score": [10, 999, 30, -5],
            "status": ["active", "unknown", "active", "inactive"],
        }
    )

    summary, bundle = validate_dataset(
        failing_df,
        config,
        dataset_name="gx_runtime",
        artifacts_dir=tmp_path / "validation",
    )

    gx_checks = [check for check in summary.checks if check.source == "gx"]

    assert len(gx_checks) == 3
    assert {check.check_name for check in gx_checks} == {
        "expect_column_values_to_be_unique",
        "expect_column_values_to_be_between",
        "expect_column_values_to_be_in_set",
    }
    assert all(not check.passed for check in gx_checks)
    assert summary.failed_count >= 3
    assert bundle is not None
    assert bundle.summary_json_path is not None and bundle.summary_json_path.exists()
    assert bundle.markdown_report_path is not None and bundle.markdown_report_path.exists()


def test_ydata_profile_executes_real_report(tmp_path: Path):
    pytest.importorskip("ydata_profiling")

    df = _small_classification_df()
    summary, bundle = profile_dataset(
        df,
        mode=ProfilingMode.MINIMAL,
        dataset_name="profile_runtime",
        artifacts_dir=tmp_path / "profiling",
    )

    assert summary.row_count == len(df)
    assert summary.report_mode == ProfilingMode.MINIMAL
    assert bundle is not None
    assert bundle.html_report_path is not None and bundle.html_report_path.exists()
    assert bundle.summary_json_path is not None and bundle.summary_json_path.exists()


def test_benchmark_executes_real_lazypredict_and_mlflow(tmp_path: Path):
    pytest.importorskip("lazypredict")
    pytest.importorskip("mlflow")

    df = _small_classification_df()
    tracking_uri = _sqlite_uri(tmp_path / "mlflow.db")
    experiment_name = f"integration-benchmark-{uuid4().hex[:8]}"

    bundle = benchmark_dataset(
        df,
        BenchmarkConfig(
            target_column="target",
            task_type=BenchmarkTaskType.CLASSIFICATION,
            include_models=["DummyClassifier", "GaussianNB"],
            top_k=2,
        ),
        dataset_name="benchmark_runtime",
        dataset_fingerprint="integration-benchmark-fingerprint",
        artifacts_dir=tmp_path / "benchmark",
        mlflow_experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )

    assert bundle.summary.target_column == "target"
    assert 1 <= bundle.summary.model_count <= 2
    assert bundle.leaderboard
    assert bundle.mlflow_run_id is not None
    assert bundle.artifacts is not None
    assert bundle.artifacts.summary_json_path is not None and bundle.artifacts.summary_json_path.exists()
    assert bundle.artifacts.leaderboard_csv_path is not None and bundle.artifacts.leaderboard_csv_path.exists()


def test_pycaret_unavailable_runtime_has_clean_guidance_on_python_313_plus():
    if sys.version_info < (3, 13):
        pytest.skip("Current interpreter may support a real PyCaret runtime.")

    with pytest.raises(PyCaretDependencyError, match="training engine is not available"):
        require_pycaret()
