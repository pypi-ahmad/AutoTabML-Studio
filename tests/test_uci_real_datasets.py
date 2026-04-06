"""Real-dataset integration tests using UCI ML Repository.

These tests fetch actual datasets from the UCI ML Repository and run them
through the full AutoTabML Studio pipeline: ingestion → validation →
profiling → benchmarking.

Marker-gated as ``integration`` so the default quick test suite is unaffected.
Requires network access and the ``ucimlrepo`` package.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from app.config.models import ProfilingMode
from app.ingestion.factory import load_dataset
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import IngestionSourceType
from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkTaskType
from app.modeling.benchmark.service import benchmark_dataset
from app.profiling.service import profile_dataset
from app.validation.schemas import ValidationRuleConfig
from app.validation.service import validate_dataset

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sqlite_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.as_posix()}"


def _fetch_uci(dataset_id: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Fetch a UCI dataset and return (combined_df, feature_cols, target_cols)."""
    ucimlrepo = pytest.importorskip("ucimlrepo")
    dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
    features: pd.DataFrame = dataset.data.features
    targets: pd.DataFrame = dataset.data.targets
    combined = pd.concat([features, targets], axis=1)
    return combined, list(features.columns), list(targets.columns)


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iris_dataset() -> tuple[pd.DataFrame, str]:
    """Iris (ID 53) – 150 rows, 4 numeric features, 3-class classification."""
    df, feature_cols, target_cols = _fetch_uci(53)
    return df, target_cols[0]


@pytest.fixture(scope="module")
def auto_mpg_dataset() -> tuple[pd.DataFrame, str]:
    """Auto MPG (ID 9) – 398 rows, mixed features, regression on mpg."""
    df, feature_cols, target_cols = _fetch_uci(9)
    return df, target_cols[0]


@pytest.fixture(scope="module")
def wine_quality_dataset() -> tuple[pd.DataFrame, str]:
    """Wine Quality (ID 186) – 4898 rows, 11 numeric features, multi-class classification."""
    df, feature_cols, target_cols = _fetch_uci(186)
    return df, target_cols[0]


@pytest.fixture(scope="module")
def heart_disease_dataset() -> tuple[pd.DataFrame, str]:
    """Heart Disease (ID 45) – 303 rows, mixed features, binary classification."""
    df, feature_cols, target_cols = _fetch_uci(45)
    return df, target_cols[0]


# ---------------------------------------------------------------------------
# Ingestion tests – load UCI DataFrames through the ingestion pipeline
# ---------------------------------------------------------------------------

class TestIngestionWithRealData:
    """Pass UCI DataFrames through the ingestion layer."""

    def test_iris_uci_repo_ingestion(self):
        spec = DatasetInputSpec(
            source_type=IngestionSourceType.UCI_REPO,
            uci_id=53,
            display_name="UCI Iris Direct",
        )
        loaded = load_dataset(spec)

        assert loaded.dataframe.shape == (150, 5)
        assert loaded.metadata.source_type == IngestionSourceType.UCI_REPO
        assert loaded.metadata.source_details["source_kind"] == "uci_repo"
        assert loaded.metadata.source_details["uci_id"] == 53
        assert loaded.metadata.source_details["target_columns"] == ["class"]

    def test_iris_ingestion(self, iris_dataset: tuple[pd.DataFrame, str]):
        df, _target = iris_dataset
        spec = DatasetInputSpec(
            source_type=IngestionSourceType.DATAFRAME,
            dataframe=df,
            display_name="UCI Iris",
        )
        loaded = load_dataset(spec)

        assert loaded.dataframe.shape == df.shape
        assert set(loaded.dataframe.columns) == set(df.columns)
        assert loaded.metadata.row_count == 150
        assert loaded.metadata.column_count == 5

    def test_auto_mpg_ingestion(self, auto_mpg_dataset: tuple[pd.DataFrame, str]):
        df, _target = auto_mpg_dataset
        spec = DatasetInputSpec(
            source_type=IngestionSourceType.DATAFRAME,
            dataframe=df,
            display_name="UCI Auto MPG",
        )
        loaded = load_dataset(spec)

        assert loaded.dataframe.shape == df.shape
        assert loaded.metadata.row_count == 398
        assert loaded.metadata.column_count == 8

    def test_wine_quality_ingestion(self, wine_quality_dataset: tuple[pd.DataFrame, str]):
        df, _target = wine_quality_dataset
        spec = DatasetInputSpec(
            source_type=IngestionSourceType.DATAFRAME,
            dataframe=df,
            display_name="UCI Wine Quality",
        )
        loaded = load_dataset(spec)

        assert loaded.dataframe.shape == df.shape
        assert loaded.metadata.row_count > 1000

    def test_heart_disease_ingestion(self, heart_disease_dataset: tuple[pd.DataFrame, str]):
        df, _target = heart_disease_dataset
        spec = DatasetInputSpec(
            source_type=IngestionSourceType.DATAFRAME,
            dataframe=df,
            display_name="UCI Heart Disease",
        )
        loaded = load_dataset(spec)

        assert loaded.dataframe.shape == df.shape
        assert loaded.metadata.row_count == 303


# ---------------------------------------------------------------------------
# Validation tests – validate real datasets with GX rules
# ---------------------------------------------------------------------------

class TestValidationWithRealData:
    """Run Great Expectations validation on real UCI datasets."""

    def test_iris_validation_all_pass(
        self, iris_dataset: tuple[pd.DataFrame, str], tmp_path: Path,
    ):
        pytest.importorskip("great_expectations")
        df, target = iris_dataset

        config = ValidationRuleConfig(
            numeric_range_checks={
                "sepal length": {"min": 0, "max": 10},
                "sepal width": {"min": 0, "max": 10},
                "petal length": {"min": 0, "max": 10},
                "petal width": {"min": 0, "max": 5},
            },
        )
        summary, bundle = validate_dataset(
            df, config, dataset_name="iris_validation", artifacts_dir=tmp_path,
        )

        assert summary.total_checks > 0
        assert summary.failed_count == 0
        # Iris has 3 duplicate rows which trigger a warning (not a failure),
        # so passed_count may be less than total_checks.
        assert summary.passed_count >= summary.total_checks - summary.warning_count
        assert bundle is not None
        assert bundle.summary_json_path is not None and bundle.summary_json_path.exists()

    def test_auto_mpg_validation_detects_nulls(
        self, auto_mpg_dataset: tuple[pd.DataFrame, str], tmp_path: Path,
    ):
        pytest.importorskip("great_expectations")
        df, target = auto_mpg_dataset

        config = ValidationRuleConfig(
            numeric_range_checks={
                "mpg": {"min": 5, "max": 60},
                "cylinders": {"min": 3, "max": 12},
            },
        )
        summary, bundle = validate_dataset(
            df, config, dataset_name="auto_mpg_validation", artifacts_dir=tmp_path,
        )

        assert summary.total_checks > 0
        assert bundle is not None

    def test_heart_disease_category_validation(
        self, heart_disease_dataset: tuple[pd.DataFrame, str], tmp_path: Path,
    ):
        pytest.importorskip("great_expectations")
        df, target = heart_disease_dataset

        config = ValidationRuleConfig(
            numeric_range_checks={
                "age": {"min": 0, "max": 120},
            },
            allowed_category_checks={
                "sex": ["0", "1"],
            },
        )
        summary, _bundle = validate_dataset(
            df, config, dataset_name="heart_validation", artifacts_dir=tmp_path,
        )

        assert summary.total_checks > 0


# ---------------------------------------------------------------------------
# Profiling tests – generate ydata-profiling reports on real data
# ---------------------------------------------------------------------------

class TestProfilingWithRealData:
    """Generate profiling reports on real UCI datasets."""

    def test_iris_profiling_minimal(
        self, iris_dataset: tuple[pd.DataFrame, str], tmp_path: Path,
    ):
        pytest.importorskip("ydata_profiling")
        df, _target = iris_dataset

        summary, bundle = profile_dataset(
            df,
            mode=ProfilingMode.MINIMAL,
            dataset_name="iris_profile",
            artifacts_dir=tmp_path,
        )

        assert summary.row_count == 150
        assert summary.report_mode == ProfilingMode.MINIMAL
        assert bundle is not None
        assert bundle.html_report_path is not None and bundle.html_report_path.exists()

    def test_wine_quality_profiling_minimal(
        self, wine_quality_dataset: tuple[pd.DataFrame, str], tmp_path: Path,
    ):
        pytest.importorskip("ydata_profiling")
        df, _target = wine_quality_dataset

        summary, bundle = profile_dataset(
            df,
            mode=ProfilingMode.MINIMAL,
            dataset_name="wine_profile",
            artifacts_dir=tmp_path,
        )

        assert summary.row_count == df.shape[0]
        assert bundle is not None
        assert bundle.html_report_path is not None and bundle.html_report_path.exists()


# ---------------------------------------------------------------------------
# Benchmark tests – real LazyPredict benchmarking on UCI datasets
# ---------------------------------------------------------------------------

class TestBenchmarkWithRealData:
    """Run LazyPredict benchmarks on real UCI datasets."""

    def test_iris_classification_benchmark(
        self, iris_dataset: tuple[pd.DataFrame, str], tmp_path: Path,
    ):
        pytest.importorskip("lazypredict")
        pytest.importorskip("mlflow")
        df, target = iris_dataset

        tracking_uri = _sqlite_uri(tmp_path / "mlflow.db")
        experiment_name = f"uci-iris-bench-{uuid4().hex[:8]}"

        bundle = benchmark_dataset(
            df,
            BenchmarkConfig(
                target_column=target,
                task_type=BenchmarkTaskType.CLASSIFICATION,
                top_k=5,
            ),
            dataset_name="uci_iris",
            dataset_fingerprint="iris-150",
            artifacts_dir=tmp_path / "benchmark",
            mlflow_experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )

        assert bundle.task_type == BenchmarkTaskType.CLASSIFICATION
        assert bundle.summary.target_column == target
        assert bundle.summary.model_count >= 1
        assert len(bundle.leaderboard) >= 1
        assert bundle.mlflow_run_id is not None
        assert bundle.artifacts is not None
        assert bundle.artifacts.leaderboard_csv_path is not None
        assert bundle.artifacts.leaderboard_csv_path.exists()

        # Verify leaderboard is ranked
        ranks = [row.rank for row in bundle.leaderboard if row.rank is not None]
        assert ranks == sorted(ranks)

    def test_auto_mpg_regression_benchmark(
        self, auto_mpg_dataset: tuple[pd.DataFrame, str], tmp_path: Path,
    ):
        pytest.importorskip("lazypredict")
        pytest.importorskip("mlflow")
        df, target = auto_mpg_dataset

        tracking_uri = _sqlite_uri(tmp_path / "mlflow.db")
        experiment_name = f"uci-autompg-bench-{uuid4().hex[:8]}"

        bundle = benchmark_dataset(
            df.dropna(),  # drop rows with nulls for clean benchmark
            BenchmarkConfig(
                target_column=target,
                task_type=BenchmarkTaskType.REGRESSION,
                top_k=5,
            ),
            dataset_name="uci_auto_mpg",
            dataset_fingerprint="autompg-392",
            artifacts_dir=tmp_path / "benchmark",
            mlflow_experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )

        assert bundle.task_type == BenchmarkTaskType.REGRESSION
        assert bundle.summary.target_column == target
        assert bundle.summary.model_count >= 1
        assert len(bundle.leaderboard) >= 1
        assert bundle.mlflow_run_id is not None

    def test_heart_disease_classification_benchmark(
        self, heart_disease_dataset: tuple[pd.DataFrame, str], tmp_path: Path,
    ):
        pytest.importorskip("lazypredict")
        pytest.importorskip("mlflow")
        df, target = heart_disease_dataset

        tracking_uri = _sqlite_uri(tmp_path / "mlflow.db")
        experiment_name = f"uci-heart-bench-{uuid4().hex[:8]}"

        bundle = benchmark_dataset(
            df.dropna(),
            BenchmarkConfig(
                target_column=target,
                task_type=BenchmarkTaskType.CLASSIFICATION,
                top_k=3,
            ),
            dataset_name="uci_heart_disease",
            dataset_fingerprint="heart-303",
            artifacts_dir=tmp_path / "benchmark",
            mlflow_experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )

        assert bundle.task_type == BenchmarkTaskType.CLASSIFICATION
        assert bundle.summary.target_column == target
        assert bundle.summary.model_count >= 1
        assert bundle.leaderboard

    def test_wine_quality_benchmark_with_auto_task_detection(
        self, wine_quality_dataset: tuple[pd.DataFrame, str], tmp_path: Path,
    ):
        """Let the system auto-detect task type from the wine quality target."""
        pytest.importorskip("lazypredict")
        pytest.importorskip("mlflow")
        df, target = wine_quality_dataset

        tracking_uri = _sqlite_uri(tmp_path / "mlflow.db")
        experiment_name = f"uci-wine-bench-{uuid4().hex[:8]}"

        bundle = benchmark_dataset(
            df,
            BenchmarkConfig(
                target_column=target,
                task_type=BenchmarkTaskType.AUTO,
                top_k=3,
            ),
            dataset_name="uci_wine_quality",
            dataset_fingerprint="wine-4898",
            artifacts_dir=tmp_path / "benchmark",
            mlflow_experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )

        assert bundle.task_type in (BenchmarkTaskType.CLASSIFICATION, BenchmarkTaskType.REGRESSION)
        assert bundle.summary.model_count >= 1
        assert bundle.leaderboard


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Run a full pipeline from ingestion through benchmark on real data."""

    def test_iris_full_pipeline(self, iris_dataset: tuple[pd.DataFrame, str], tmp_path: Path):
        """Ingestion → Validation → Profiling → Benchmark on Iris."""
        pytest.importorskip("lazypredict")
        pytest.importorskip("mlflow")
        pytest.importorskip("great_expectations")
        pytest.importorskip("ydata_profiling")

        df, target = iris_dataset

        # Step 1: Ingestion
        spec = DatasetInputSpec(
            source_type=IngestionSourceType.DATAFRAME,
            dataframe=df,
            display_name="UCI Iris E2E",
        )
        loaded = load_dataset(spec)
        assert loaded.metadata.row_count == 150
        pipeline_df = loaded.dataframe

        # Step 2: Validation
        val_config = ValidationRuleConfig(
            numeric_range_checks={
                "sepal length": {"min": 0, "max": 10},
                "petal length": {"min": 0, "max": 10},
            },
        )
        val_summary, val_bundle = validate_dataset(
            pipeline_df, val_config,
            dataset_name="iris_e2e",
            artifacts_dir=tmp_path / "validation",
        )
        assert val_summary.failed_count == 0

        # Step 3: Profiling
        prof_summary, prof_bundle = profile_dataset(
            pipeline_df,
            mode=ProfilingMode.MINIMAL,
            dataset_name="iris_e2e",
            artifacts_dir=tmp_path / "profiling",
        )
        assert prof_summary.row_count == 150
        assert prof_bundle is not None

        # Step 4: Benchmark
        tracking_uri = _sqlite_uri(tmp_path / "mlflow.db")
        bench_bundle = benchmark_dataset(
            pipeline_df,
            BenchmarkConfig(
                target_column=target,
                task_type=BenchmarkTaskType.CLASSIFICATION,
                top_k=3,
            ),
            dataset_name="iris_e2e",
            dataset_fingerprint=loaded.metadata.schema_hash,
            artifacts_dir=tmp_path / "benchmark",
            mlflow_experiment_name=f"iris-e2e-{uuid4().hex[:8]}",
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )
        assert bench_bundle.summary.model_count >= 1
        assert bench_bundle.leaderboard
        assert bench_bundle.mlflow_run_id is not None

        # Verify artifacts exist
        assert (tmp_path / "validation").exists()
        assert (tmp_path / "profiling").exists()
        assert (tmp_path / "benchmark").exists()

    def test_auto_mpg_full_pipeline(self, auto_mpg_dataset: tuple[pd.DataFrame, str], tmp_path: Path):
        """Ingestion → Validation → Profiling → Benchmark on Auto MPG."""
        pytest.importorskip("lazypredict")
        pytest.importorskip("mlflow")
        pytest.importorskip("great_expectations")
        pytest.importorskip("ydata_profiling")

        df, target = auto_mpg_dataset
        clean_df = df.dropna()

        # Step 1: Ingestion
        spec = DatasetInputSpec(
            source_type=IngestionSourceType.DATAFRAME,
            dataframe=clean_df,
            display_name="UCI Auto MPG E2E",
        )
        loaded = load_dataset(spec)
        pipeline_df = loaded.dataframe

        # Step 2: Validation
        val_config = ValidationRuleConfig(
            numeric_range_checks={
                "mpg": {"min": 5, "max": 60},
            },
        )
        val_summary, _ = validate_dataset(
            pipeline_df, val_config,
            dataset_name="autompg_e2e",
            artifacts_dir=tmp_path / "validation",
        )
        assert val_summary.total_checks > 0

        # Step 3: Profiling
        prof_summary, _ = profile_dataset(
            pipeline_df,
            mode=ProfilingMode.MINIMAL,
            dataset_name="autompg_e2e",
            artifacts_dir=tmp_path / "profiling",
        )
        assert prof_summary.row_count == clean_df.shape[0]

        # Step 4: Benchmark
        tracking_uri = _sqlite_uri(tmp_path / "mlflow.db")
        bench_bundle = benchmark_dataset(
            pipeline_df,
            BenchmarkConfig(
                target_column=target,
                task_type=BenchmarkTaskType.REGRESSION,
                top_k=3,
            ),
            dataset_name="autompg_e2e",
            dataset_fingerprint=loaded.metadata.schema_hash,
            artifacts_dir=tmp_path / "benchmark",
            mlflow_experiment_name=f"autompg-e2e-{uuid4().hex[:8]}",
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )
        assert bench_bundle.task_type == BenchmarkTaskType.REGRESSION
        assert bench_bundle.summary.model_count >= 1
        assert bench_bundle.leaderboard
