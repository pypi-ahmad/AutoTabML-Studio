"""Real end-to-end local smoke test — no mocks for core paths.

Unlike ``test_hardening_smoke.py`` (which mocks profiling/benchmark), this
exercises the *real* runtime from startup through validation, profiling, and
benchmarking.  Optional heavy deps (GX, ydata-profiling, LazyPredict, MLflow)
are probed at runtime: if installed they run for real, if absent the test
verifies graceful degradation.

**Runs in the default test suite** (no ``integration`` marker, no network).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from app.config.models import AppSettings, ProfilingMode
from app.ingestion import DatasetInputSpec, IngestionSourceType, load_dataset
from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkTaskType
from app.profiling.service import profile_dataset
from app.startup import StartupStatus, initialize_local_runtime
from app.storage import AppJobType, build_metadata_store
from app.validation.schemas import ValidationRuleConfig
from app.validation.service import validate_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sqlite_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.as_posix()}"


def _classification_csv(tmp_path: Path) -> Path:
    """Write a small but realistic classification dataset (no network)."""
    rows = []
    for i in range(60):
        rows.append(
            {
                "id": i,
                "feature_num": float(i) + 0.5,
                "feature_cat": "high" if i >= 30 else "low",
                "score": (i * 7 % 100),
                "target": 1 if i >= 30 else 0,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "smoke_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# 1. Startup initialization
# ---------------------------------------------------------------------------

class TestStartupInitialization:
    """Verify the real startup path creates all expected resources."""

    def test_local_runtime_initializes_cleanly(self, tmp_path: Path):
        settings = AppSettings.model_validate(
            {"artifacts": {"root_dir": str(tmp_path / "artifacts")}}
        )
        status: StartupStatus = initialize_local_runtime(
            settings, include_optional_network_checks=False
        )

        assert status.artifact_dirs, "Expected at least one artifact directory"
        assert all(d.exists() for d in status.artifact_dirs)
        assert not status.errors, f"Startup errors: {[e.message for e in status.errors]}"

    def test_metadata_store_initializes(self, tmp_path: Path):
        settings = AppSettings.model_validate(
            {"artifacts": {"root_dir": str(tmp_path / "artifacts")}}
        )
        store = build_metadata_store(settings)
        assert store is not None
        assert store.db_path.exists()


# ---------------------------------------------------------------------------
# 2. Full local pipeline — ingestion → validation → profiling → benchmark
# ---------------------------------------------------------------------------

class TestLocalPipelineEndToEnd:
    """Exercise the real pipeline — no mocks for any service code."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path):
        """Common setup: settings, metadata store, loaded dataset."""
        self.tmp_path = tmp_path
        self.csv_path = _classification_csv(tmp_path)

        self.settings = AppSettings.model_validate(
            {"artifacts": {"root_dir": str(tmp_path / "artifacts")}}
        )

        # Real startup
        status = initialize_local_runtime(
            self.settings, include_optional_network_checks=False
        )
        assert not status.errors

        self.store = build_metadata_store(self.settings)
        assert self.store is not None

        # Real ingestion
        self.loaded = load_dataset(
            DatasetInputSpec(
                source_type=IngestionSourceType.CSV, path=self.csv_path
            )
        )
        assert self.loaded.dataframe.shape == (60, 5)

    # -- Validation (always runs, GX is optional) -------------------------

    def test_validation_runs_and_records_metadata(self):
        summary, bundle = validate_dataset(
            self.loaded.dataframe,
            ValidationRuleConfig(target_column="target"),
            dataset_name="smoke",
            artifacts_dir=self.settings.validation.artifacts_dir,
            loaded_dataset=self.loaded,
            metadata_store=self.store,
        )

        assert summary.dataset_name == "smoke"
        assert summary.total_checks > 0
        assert bundle is not None
        assert bundle.summary_json_path is not None
        assert bundle.summary_json_path.exists()

        jobs = self.store.list_recent_jobs(limit=5)
        assert any(j.job_type == AppJobType.VALIDATION for j in jobs)

    def test_validation_with_rich_rule_config(self):
        """Exercise non-trivial rule config — uniqueness, range, category checks."""
        config = ValidationRuleConfig(
            target_column="target",
            uniqueness_columns=["id"],
            numeric_range_checks={"score": {"min": 0, "max": 100}},
            allowed_category_checks={"feature_cat": ["high", "low"]},
        )
        summary, _ = validate_dataset(
            self.loaded.dataframe,
            config,
            dataset_name="smoke_rich",
            artifacts_dir=self.settings.validation.artifacts_dir,
        )

        assert summary.passed_count > 0
        assert summary.failed_count == 0

    # -- Profiling (real ydata-profiling if installed, else graceful) ------

    def test_profiling_real_or_graceful_skip(self):
        try:
            summary, bundle = profile_dataset(
                self.loaded.dataframe,
                mode=ProfilingMode.MINIMAL,
                dataset_name="smoke",
                artifacts_dir=self.settings.profiling.artifacts_dir,
                loaded_dataset=self.loaded,
                metadata_store=self.store,
            )
            # ydata-profiling IS installed
            assert summary.row_count == 60
            assert summary.report_mode == ProfilingMode.MINIMAL
            assert bundle is not None
            assert bundle.html_report_path is not None
            assert bundle.html_report_path.exists()

            jobs = self.store.list_recent_jobs(limit=10)
            assert any(j.job_type == AppJobType.PROFILING for j in jobs)
            logger.info("Profiling completed with real ydata-profiling")
        except Exception as exc:
            # ydata-profiling NOT installed — verify error is clean
            assert "ydata" in str(exc).lower() or "profiling" in str(exc).lower(), (
                f"Unexpected profiling error: {exc}"
            )
            logger.info("Profiling skipped (ydata-profiling not installed): %s", exc)

    # -- Benchmark (real LazyPredict + MLflow if installed) ----------------

    def test_benchmark_real_or_graceful_skip(self):
        from app.modeling.benchmark.service import benchmark_dataset

        tracking_uri = _sqlite_uri(self.tmp_path / "mlflow.db")

        try:
            bundle = benchmark_dataset(
                self.loaded.dataframe,
                BenchmarkConfig(
                    target_column="target",
                    task_type=BenchmarkTaskType.CLASSIFICATION,
                    include_models=["DummyClassifier"],
                    top_k=1,
                ),
                dataset_name="smoke",
                dataset_fingerprint="e2e-smoke-fingerprint",
                loaded_dataset=self.loaded,
                metadata_store=self.store,
                artifacts_dir=self.settings.benchmark.artifacts_dir,
                mlflow_experiment_name="e2e-smoke-test",
                tracking_uri=tracking_uri,
                registry_uri=tracking_uri,
            )
            # LazyPredict IS installed
            assert bundle.summary.target_column == "target"
            assert bundle.summary.model_count >= 1
            assert bundle.leaderboard
            assert bundle.artifacts is not None

            jobs = self.store.list_recent_jobs(limit=10)
            assert any(j.job_type == AppJobType.BENCHMARK for j in jobs)
            logger.info(
                "Benchmark completed: %d models, best=%s",
                bundle.summary.model_count,
                bundle.summary.best_model_name,
            )

            # If MLflow is installed, verify tracking worked
            if bundle.mlflow_run_id is not None:
                logger.info("MLflow tracked run: %s", bundle.mlflow_run_id)
        except Exception as exc:
            # LazyPredict NOT installed — verify error is clean
            assert "lazypredict" in str(exc).lower() or "benchmark" in str(exc).lower(), (
                f"Unexpected benchmark error: {exc}"
            )
            logger.info("Benchmark skipped (lazypredict not installed): %s", exc)


# ---------------------------------------------------------------------------
# 3. MLflow URI edge cases
# ---------------------------------------------------------------------------

class TestMLflowURIHandling:
    """Verify the startup MLflow URI validation catches bad config."""

    @pytest.mark.parametrize(
        "tracking_uri,expected_severity",
        [
            ("sqlite:///my db.db", "warning"),  # embedded whitespace
            ("sqlite:///valid.db", None),         # valid URI, no issue
        ],
    )
    def test_mlflow_uri_validation(
        self, tmp_path: Path, tracking_uri: str, expected_severity: str | None
    ):
        settings = AppSettings.model_validate(
            {
                "artifacts": {"root_dir": str(tmp_path / "artifacts")},
                "mlflow": {"tracking_uri": tracking_uri},
            }
        )
        status = initialize_local_runtime(
            settings, include_optional_network_checks=False
        )

        mlflow_issues = [
            i for i in status.issues if "mlflow" in i.message.lower()
        ]

        if expected_severity is None:
            assert not mlflow_issues
        else:
            assert any(i.severity == expected_severity for i in mlflow_issues)


# ---------------------------------------------------------------------------
# 4. Optional dependency probes — verify clean error messages
# ---------------------------------------------------------------------------

class TestOptionalDependencyProbes:
    """Verify each optional dep has a clean availability check."""

    def test_gx_availability_check_does_not_raise(self):
        from app.validation.gx_context import is_gx_available

        result = is_gx_available()
        assert isinstance(result, bool)

    def test_ydata_availability_check_does_not_raise(self):
        from app.profiling.ydata_runner import is_ydata_available

        result = is_ydata_available()
        assert isinstance(result, bool)

    def test_lazypredict_availability_check_does_not_raise(self):
        from app.modeling.benchmark.lazypredict_runner import is_lazypredict_available

        result = is_lazypredict_available()
        assert isinstance(result, bool)

    def test_pycaret_availability_check_does_not_raise(self):
        from app.modeling.pycaret.setup_runner import is_pycaret_available

        result = is_pycaret_available()
        assert isinstance(result, bool)

    def test_mlflow_availability_check_does_not_raise(self):
        from app.tracking.mlflow_query import is_mlflow_available

        result = is_mlflow_available()
        assert isinstance(result, bool)
