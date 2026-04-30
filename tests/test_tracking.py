"""Tests for the tracking layer – history, comparison, and MLflow query wrappers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config.models import TrackingSettings
from app.tracking.artifacts import write_comparison_artifacts
from app.tracking.compare_service import ComparisonService
from app.tracking.filters import (
    RunHistoryFilter,
    RunHistorySort,
    RunSortField,
    SortDirection,
    build_mlflow_filter_string,
)
from app.tracking.history_service import HistoryService
from app.tracking.schemas import (
    RunDetailView,
    RunHistoryItem,
    RunStatus,
    RunType,
)
from app.tracking.summary import run_summary_line

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str = "abc123def456",
    run_name: str = "benchmark-classification-train",
    experiment_id: str = "1",
    experiment_name: str = "autotabml-benchmarks",
    status: RunStatus = RunStatus.FINISHED,
    run_type: RunType = RunType.BENCHMARK,
    task_type: str = "classification",
    target_column: str = "target",
    model_name: str | None = "Logistic Regression",
    primary_metric_name: str | None = "Accuracy",
    primary_metric_value: float | None = 0.91,
    duration_seconds: float | None = 1.5,
    dataset_fingerprint: str = "fp-1",
    params: dict[str, str] | None = None,
    metrics: dict[str, float] | None = None,
    **overrides,
) -> RunHistoryItem:
    if params is None:
        params = {
            "task_type": task_type,
            "target_column": target_column,
            "dataset_fingerprint": dataset_fingerprint,
        }
    if metrics is None:
        effective_metrics: dict[str, float] = {}
        if primary_metric_name and primary_metric_value is not None:
            effective_metrics[primary_metric_name] = primary_metric_value
            effective_metrics["best_baseline_score"] = primary_metric_value
        metrics = effective_metrics
    return RunHistoryItem(
        run_id=run_id,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        run_name=run_name,
        status=status,
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
        duration_seconds=duration_seconds,
        artifact_uri="file:///artifacts/1/abc123def456/artifacts",
        run_type=run_type,
        task_type=task_type,
        target_column=target_column,
        model_name=model_name,
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary_metric_value,
        dataset_fingerprint=dataset_fingerprint,
        params=params,
        metrics=metrics,
        tags={"mlflow.runName": run_name},
        **overrides,
    )


# ---------------------------------------------------------------------------
# Filter string builder
# ---------------------------------------------------------------------------


class TestFilterStringBuilder:
    def test_empty_filter_produces_empty_string(self):
        assert build_mlflow_filter_string(None) == ""
        assert build_mlflow_filter_string(RunHistoryFilter()) == ""

    def test_task_type_filter(self):
        result = build_mlflow_filter_string(
            RunHistoryFilter(task_type="classification")
        )
        assert "params.task_type = 'classification'" in result

    def test_status_filter(self):
        result = build_mlflow_filter_string(
            RunHistoryFilter(status=RunStatus.FINISHED)
        )
        assert "attributes.status = 'FINISHED'" in result

    def test_combined_filters_joined_with_and(self):
        result = build_mlflow_filter_string(
            RunHistoryFilter(
                task_type="regression",
                dataset_fingerprint="fp-123",
                status=RunStatus.FINISHED,
            )
        )
        assert " AND " in result
        assert "params.task_type = 'regression'" in result
        assert "params.dataset_fingerprint = 'fp-123'" in result
        assert "attributes.status = 'FINISHED'" in result

    def test_tag_filter(self):
        result = build_mlflow_filter_string(
            RunHistoryFilter(tags={"custom_key": "custom_value"})
        )
        assert "tags.`custom_key` = 'custom_value'" in result


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


class TestRunSummary:
    def test_summary_line_includes_key_fields(self):
        run = _make_run()
        line = run_summary_line(run)
        assert "benchmark" in line
        assert "Logistic Regression" in line
        assert "Accuracy" in line
        assert "0.91" in line

    def test_summary_line_handles_missing_fields(self):
        run = _make_run(
            model_name=None,
            primary_metric_name=None,
            primary_metric_value=None,
            duration_seconds=None,
        )
        line = run_summary_line(run)
        assert "benchmark" in line


# ---------------------------------------------------------------------------
# History service sorting
# ---------------------------------------------------------------------------


class TestHistoryServiceSorting:
    """Validates that history sorting is correct and not partial-dataset."""

    def _service_with_runs(self, monkeypatch, runs, *, captured: dict):
        from app.tracking import history_service as service_module

        def fake_search_runs(**kwargs):
            captured.update(kwargs)
            max_results = kwargs.get("max_results", len(runs))
            # Mimic MLflow's start_time DESC default ordering.
            ordered = sorted(runs, key=lambda r: r.start_time or 0, reverse=True)
            return ordered[:max_results]

        monkeypatch.setattr(service_module.mlflow_query, "search_runs", fake_search_runs)
        monkeypatch.setattr(
            service_module.HistoryService,
            "_resolve_experiment_ids",
            lambda self, history_filter: (["1"], {"1": "exp"}),
        )
        return HistoryService(default_limit=5)

    def test_start_time_sort_pushed_to_mlflow(self, monkeypatch):
        captured: dict = {}
        runs = [
            _make_run(run_id=f"r{i}", duration_seconds=float(i))
            for i in range(3)
        ]
        service = self._service_with_runs(monkeypatch, runs, captured=captured)

        service.list_runs(
            sort=RunHistorySort(field=RunSortField.START_TIME, direction=SortDirection.ASCENDING),
            limit=2,
        )

        assert captured["order_by"] == ["attributes.start_time ASC"]
        assert captured["max_results"] == 2

    def test_duration_sort_widens_pool_and_orders_full_dataset(self, monkeypatch):
        captured: dict = {}
        # Build 12 runs where duration is *anti-correlated* with start_time, so a
        # naive "fetch top-N by start_time then re-sort" yields the wrong winner.
        runs = []
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(12):
            run = _make_run(
                run_id=f"r{i:02d}",
                duration_seconds=float(12 - i),
            )
            run = run.model_copy(
                update={
                    "start_time": base.replace(hour=i),
                    "end_time": base.replace(hour=i, minute=int(12 - i)),
                }
            )
            runs.append(run)

        service = self._service_with_runs(monkeypatch, runs, captured=captured)

        result = service.list_runs(
            sort=RunHistorySort(field=RunSortField.DURATION, direction=SortDirection.DESCENDING),
            limit=3,
        )

        # The widened pool must be requested so client-side sort sees the full set.
        assert captured["max_results"] >= len(runs)
        # Top-3 by duration should be 12.0, 11.0, 10.0 — never a recent-but-short run.
        assert [r.duration_seconds for r in result] == [12.0, 11.0, 10.0]
        assert [r.run_id for r in result] == ["r00", "r01", "r02"]

    def test_primary_score_sort_is_not_partial(self, monkeypatch):
        captured: dict = {}
        runs = [
            _make_run(run_id=f"r{i:02d}", primary_metric_value=float(i) / 10)
            for i in range(8)
        ]
        service = self._service_with_runs(monkeypatch, runs, captured=captured)

        result = service.list_runs(
            sort=RunHistorySort(
                field=RunSortField.PRIMARY_SCORE, direction=SortDirection.DESCENDING
            ),
            limit=2,
        )

        assert captured["max_results"] >= len(runs)
        assert [r.primary_metric_value for r in result] == [0.7, 0.6]

    def test_duration_sort_uses_zero_for_missing_values(self, monkeypatch):
        captured: dict = {}
        runs = [
            _make_run(run_id="present", duration_seconds=4.0),
            _make_run(run_id="missing", duration_seconds=None),
            _make_run(run_id="longest", duration_seconds=9.5),
        ]
        service = self._service_with_runs(monkeypatch, runs, captured=captured)

        result = service.list_runs(
            sort=RunHistorySort(
                field=RunSortField.DURATION, direction=SortDirection.ASCENDING
            ),
            limit=3,
        )

        assert [r.run_id for r in result] == ["missing", "present", "longest"]


# ---------------------------------------------------------------------------
# Comparison service
# ---------------------------------------------------------------------------


class TestComparisonService:
    def test_comparable_runs_have_no_warnings(self):
        left = _make_run(run_id="left-111")
        right = _make_run(run_id="right-222", primary_metric_value=0.93)

        service = ComparisonService()
        bundle = service.compare(left, right)

        assert bundle.comparable is True
        assert bundle.warnings == []

    def test_different_datasets_produce_warning(self):
        left = _make_run(run_id="left-111", dataset_fingerprint="fp-1")
        right = _make_run(run_id="right-222", dataset_fingerprint="fp-2")

        service = ComparisonService()
        bundle = service.compare(left, right)

        assert not bundle.comparable
        assert any("different datasets" in w for w in bundle.warnings)

    def test_different_target_columns_produce_warning(self):
        left = _make_run(run_id="left-111", target_column="price")
        right = _make_run(run_id="right-222", target_column="label")

        service = ComparisonService()
        bundle = service.compare(left, right)

        assert any("different columns" in w for w in bundle.warnings)

    def test_different_task_types_produce_warning(self):
        left = _make_run(run_id="left-111", task_type="classification")
        right = _make_run(run_id="right-222", task_type="regression")

        service = ComparisonService()
        bundle = service.compare(left, right)

        assert any("different task types" in w for w in bundle.warnings)

    def test_different_run_types_produce_warning(self):
        left = _make_run(run_id="left-111", run_type=RunType.BENCHMARK)
        right = _make_run(run_id="right-222", run_type=RunType.EXPERIMENT)

        service = ComparisonService()
        bundle = service.compare(left, right)

        assert any("different types" in w for w in bundle.warnings)

    def test_metric_deltas_are_computed_correctly(self):
        left = _make_run(
            run_id="left-111",
            primary_metric_name="Accuracy",
            primary_metric_value=0.85,
            metrics={"Accuracy": 0.85, "AUC": 0.90},
        )
        right = _make_run(
            run_id="right-222",
            primary_metric_name="Accuracy",
            primary_metric_value=0.91,
            metrics={"Accuracy": 0.91, "AUC": 0.93},
        )

        service = ComparisonService()
        bundle = service.compare(left, right)

        accuracy_delta = next(d for d in bundle.metric_deltas if d.name == "Accuracy")
        assert accuracy_delta.left_value == 0.85
        assert accuracy_delta.right_value == 0.91
        assert accuracy_delta.delta == pytest.approx(0.06)
        assert accuracy_delta.better_side == "right"

    def test_lower_is_better_metric_inversion(self):
        left = _make_run(
            run_id="left-111",
            metrics={"RMSE": 1.5},
        )
        right = _make_run(
            run_id="right-222",
            metrics={"RMSE": 1.2},
        )

        service = ComparisonService()
        bundle = service.compare(left, right)

        rmse_delta = next(d for d in bundle.metric_deltas if d.name == "RMSE")
        assert rmse_delta.better_side == "right"

    def test_missing_metric_on_one_side(self):
        left = _make_run(
            run_id="left-111",
            metrics={"Accuracy": 0.85, "AUC": 0.90},
        )
        right = _make_run(
            run_id="right-222",
            metrics={"Accuracy": 0.91},
        )

        service = ComparisonService()
        bundle = service.compare(left, right)

        auc_delta = next(d for d in bundle.metric_deltas if d.name == "AUC")
        assert auc_delta.left_value == 0.90
        assert auc_delta.right_value is None
        assert auc_delta.delta is None

    def test_missing_dataset_metadata_produces_warning(self):
        left = _make_run(
            run_id="left-111",
            dataset_fingerprint=None,
            params={"task_type": "classification", "target_column": "target"},
        )
        right = _make_run(
            run_id="right-222",
            dataset_fingerprint=None,
            params={"task_type": "classification", "target_column": "target"},
        )

        service = ComparisonService()
        bundle = service.compare(left, right)

        assert not bundle.comparable
        assert any("dataset" in warning.lower() and "verified" in warning.lower() for warning in bundle.warnings)

    def test_config_differences_detected(self):
        left = _make_run(
            run_id="left-111",
            params={"task_type": "classification", "target_column": "target", "dataset_fingerprint": "fp-1"},
        )
        right = _make_run(
            run_id="right-222",
            params={"task_type": "regression", "target_column": "target", "dataset_fingerprint": "fp-1"},
        )

        service = ComparisonService()
        bundle = service.compare(left, right)

        task_diff = next(
            (d for d in bundle.config_differences if d.key == "task_type"),
            None,
        )
        assert task_diff is not None
        assert task_diff.left_value == "classification"
        assert task_diff.right_value == "regression"

    def test_tie_metric_delta(self):
        left = _make_run(run_id="left-111", metrics={"Accuracy": 0.90})
        right = _make_run(run_id="right-222", metrics={"Accuracy": 0.90})

        service = ComparisonService()
        bundle = service.compare(left, right)

        accuracy_delta = next(d for d in bundle.metric_deltas if d.name == "Accuracy")
        assert accuracy_delta.better_side == "tie"


# ---------------------------------------------------------------------------
# Comparison artifacts
# ---------------------------------------------------------------------------


class TestComparisonArtifacts:
    def test_write_comparison_artifacts(self, tmp_path: Path):
        left = _make_run(run_id="aaa111bbb222")
        right = _make_run(run_id="ccc333ddd444", primary_metric_value=0.93)
        bundle = ComparisonService().compare(left, right)

        paths = write_comparison_artifacts(bundle, tmp_path)

        assert "comparison_json" in paths
        assert paths["comparison_json"].exists()
        assert "metrics_csv" in paths
        assert paths["metrics_csv"].exists()
        assert "markdown" in paths
        assert paths["markdown"].exists()

        md_content = paths["markdown"].read_text(encoding="utf-8")
        assert "# Run Comparison" in md_content
        assert "aaa111bb" in md_content


# ---------------------------------------------------------------------------
# History service with mocked MLflow
# ---------------------------------------------------------------------------


class _FakeMLflowRun:
    def __init__(self, run_id, experiment_id, params, metrics, tags, status="FINISHED"):
        self.info = SimpleNamespace(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=tags.get("mlflow.runName", ""),
            status=status,
            start_time=1704067200000,
            end_time=1704067201500,
            artifact_uri=f"file:///artifacts/{experiment_id}/{run_id}/artifacts",
        )
        self.data = SimpleNamespace(params=params, metrics=metrics, tags=tags)


class _FakeMLflowExperiment:
    def __init__(self, experiment_id, name):
        self.experiment_id = experiment_id
        self.name = name
        self.lifecycle_stage = "active"
        self.artifact_location = f"file:///artifacts/{experiment_id}"
        self.creation_time = 1704067200000
        self.last_update_time = 1704067200000
        self.tags = {}


def _patch_mlflow_query(monkeypatch, experiments, runs):
    """Patch mlflow_query module functions for testing."""

    from app.tracking import mlflow_query

    monkeypatch.setattr(mlflow_query, "is_mlflow_available", lambda: True)
    monkeypatch.setattr(mlflow_query, "_require_mlflow", lambda: None)

    def fake_get_client(tracking_uri=None, registry_uri=None):
        client = SimpleNamespace()
        client.search_experiments = lambda view_type=1: experiments
        client.get_experiment_by_name = lambda name: next(
            (e for e in experiments if e.name == name), None
        )
        client.search_runs = lambda experiment_ids, filter_string="", order_by=None, max_results=200: [
            r for r in runs if r.info.experiment_id in experiment_ids
        ]
        client.get_run = lambda run_id: next(
            (r for r in runs if r.info.run_id == run_id), None
        )
        client.list_artifacts = lambda run_id, path=None: []
        return client

    monkeypatch.setattr(mlflow_query, "_get_client", fake_get_client)


class TestHistoryService:
    def test_list_runs_returns_normalized_items(self, monkeypatch):
        experiments = [
            _FakeMLflowExperiment("1", "autotabml-benchmarks"),
        ]
        runs = [
            _FakeMLflowRun(
                "run-001", "1",
                params={"task_type": "classification", "target_column": "target", "dataset_fingerprint": "fp-1"},
                metrics={"best_score": 0.92},
                tags={"mlflow.runName": "benchmark-classification-train"},
            ),
        ]
        _patch_mlflow_query(monkeypatch, experiments, runs)

        service = HistoryService(
            default_experiment_names=["autotabml-benchmarks"],
        )
        result = service.list_runs()

        assert len(result) == 1
        assert result[0].run_id == "run-001"
        assert result[0].run_type == RunType.BENCHMARK
        assert result[0].task_type == "classification"

    def test_filter_by_run_type_works_client_side(self, monkeypatch):
        experiments = [
            _FakeMLflowExperiment("1", "autotabml-benchmarks"),
            _FakeMLflowExperiment("2", "autotabml-experiments"),
        ]
        runs = [
            _FakeMLflowRun(
                "run-bench", "1",
                params={"task_type": "classification", "target_column": "t", "dataset_fingerprint": ""},
                metrics={},
                tags={"mlflow.runName": "benchmark-classification-ds"},
            ),
            _FakeMLflowRun(
                "run-exp", "2",
                params={"task_type": "classification", "target_column": "t", "dataset_fingerprint": ""},
                metrics={},
                tags={"mlflow.runName": "experiment-classification-ds"},
            ),
        ]
        _patch_mlflow_query(monkeypatch, experiments, runs)

        service = HistoryService(
            default_experiment_names=["autotabml-benchmarks", "autotabml-experiments"],
        )
        result = service.list_runs(
            history_filter=RunHistoryFilter(run_type=RunType.EXPERIMENT),
        )

        assert len(result) == 1
        assert result[0].run_type == RunType.EXPERIMENT

    def test_get_run_detail_returns_extended_view(self, monkeypatch):
        experiments = [_FakeMLflowExperiment("1", "autotabml-benchmarks")]
        runs = [
            _FakeMLflowRun(
                "run-detail", "1",
                params={"task_type": "regression", "target_column": "price", "dataset_fingerprint": "fp"},
                metrics={"R2": 0.88, "RMSE": 1.23},
                tags={"mlflow.runName": "benchmark-regression-housing"},
            ),
        ]
        _patch_mlflow_query(monkeypatch, experiments, runs)

        service = HistoryService()
        detail = service.get_run_detail("run-detail")

        assert isinstance(detail, RunDetailView)
        assert detail.run_id == "run-detail"
        assert detail.task_type == "regression"
        assert detail.metrics["R2"] == 0.88

    def test_get_run_detail_lists_nested_artifacts(self, monkeypatch):
        from app.tracking import mlflow_query

        class _Artifact:
            def __init__(self, path, is_dir=False):
                self.path = path
                self.is_dir = is_dir

        experiments = [_FakeMLflowExperiment("1", "autotabml-benchmarks")]
        runs = [
            _FakeMLflowRun(
                "run-detail", "1",
                params={"task_type": "regression", "target_column": "price", "dataset_fingerprint": "fp"},
                metrics={"R2": 0.88},
                tags={"mlflow.runName": "benchmark-regression-housing"},
            ),
        ]

        monkeypatch.setattr(mlflow_query, "is_mlflow_available", lambda: True)
        monkeypatch.setattr(mlflow_query, "_require_mlflow", lambda: None)

        artifact_tree = {
            None: [_Artifact("plots", is_dir=True), _Artifact("summary.json")],
            "plots": [_Artifact("plots/confusion.png")],
        }

        def fake_get_client(tracking_uri=None, registry_uri=None):
            client = SimpleNamespace()
            client.search_experiments = lambda view_type=1: experiments
            client.get_experiment_by_name = lambda name: experiments[0]
            client.search_runs = lambda experiment_ids, filter_string="", order_by=None, max_results=200: runs
            client.get_run = lambda run_id: runs[0]
            client.list_artifacts = lambda run_id, path=None: artifact_tree.get(path, [])
            return client

        monkeypatch.setattr(mlflow_query, "_get_client", fake_get_client)

        detail = HistoryService().get_run_detail("run-detail")

        assert "summary.json" in detail.artifact_paths
        assert "plots/confusion.png" in detail.artifact_paths

    def test_sort_by_primary_score(self, monkeypatch):
        experiments = [_FakeMLflowExperiment("1", "autotabml-benchmarks")]
        runs = [
            _FakeMLflowRun(
                "run-low", "1",
                params={"task_type": "classification", "target_column": "t", "dataset_fingerprint": ""},
                metrics={"best_score": 0.70},
                tags={"mlflow.runName": "benchmark-classification-a"},
            ),
            _FakeMLflowRun(
                "run-high", "1",
                params={"task_type": "classification", "target_column": "t", "dataset_fingerprint": ""},
                metrics={"best_score": 0.95},
                tags={"mlflow.runName": "benchmark-classification-b"},
            ),
        ]
        _patch_mlflow_query(monkeypatch, experiments, runs)

        service = HistoryService()
        result = service.list_runs(
            sort=RunHistorySort(
                field=RunSortField.PRIMARY_SCORE,
                direction=SortDirection.DESCENDING,
            ),
        )

        assert len(result) == 2
        assert result[0].run_id == "run-high"
        assert result[1].run_id == "run-low"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestTrackingConfig:
    def test_defaults(self):
        settings = TrackingSettings()
        assert settings.tracking_uri is None
        assert settings.registry_uri is None
        assert "autotabml-benchmarks" in settings.default_experiment_names
        assert "autotabml-experiments" in settings.default_experiment_names
        assert settings.history_page_default_limit == 50
        assert settings.champion_alias == "champion"
        assert settings.candidate_alias == "candidate"
        assert settings.registry_enabled is True


# ---------------------------------------------------------------------------
# MLflow normalization
# ---------------------------------------------------------------------------


class TestMLflowNormalization:
    def test_run_type_inferred_from_experiment_name(self):
        from app.tracking.mlflow_query import _infer_run_type

        assert _infer_run_type("autotabml-benchmarks", {}) == RunType.BENCHMARK
        assert _infer_run_type("autotabml-experiments", {}) == RunType.EXPERIMENT
        assert _infer_run_type("custom-experiment", {}) == RunType.UNKNOWN

    def test_run_type_inferred_from_run_name_tag(self):
        from app.tracking.mlflow_query import _infer_run_type

        assert _infer_run_type(None, {"mlflow.runName": "benchmark-classification-ds"}) == RunType.BENCHMARK
        assert _infer_run_type(None, {"mlflow.runName": "experiment-regression-ds"}) == RunType.EXPERIMENT
        assert _infer_run_type(None, {}) == RunType.UNKNOWN

    def test_safe_run_status(self):
        from app.tracking.mlflow_query import _safe_run_status

        assert _safe_run_status("FINISHED") == RunStatus.FINISHED
        assert _safe_run_status("RUNNING") == RunStatus.RUNNING
        assert _safe_run_status("INVALID") == RunStatus.UNKNOWN
        assert _safe_run_status(None) == RunStatus.UNKNOWN

    def test_primary_metric_extraction(self):
        from app.tracking.mlflow_query import _extract_primary_metric

        name, value = _extract_primary_metric(
            {"best_score": 0.91, "Accuracy": 0.91},
            {"compare_optimize_metric": "Accuracy"},
            RunType.EXPERIMENT,
        )
        assert name == "Accuracy"
        assert value == 0.91

    def test_primary_metric_fallback_to_best_score(self):
        from app.tracking.mlflow_query import _extract_primary_metric

        name, value = _extract_primary_metric(
            {"best_score": 0.88},
            {},
            RunType.BENCHMARK,
        )
        assert name == "best_score"
        assert value == 0.88

    def test_dataset_name_preserves_hyphenated_suffix(self):
        from app.tracking.mlflow_query import _normalize_run

        raw = _FakeMLflowRun(
            "run-001",
            "1",
            params={"task_type": "classification", "target_column": "target", "dataset_fingerprint": "fp-1"},
            metrics={},
            tags={"mlflow.runName": "benchmark-classification-credit-default"},
        )

        normalized = _normalize_run(raw, {"1": "autotabml-benchmarks"})

        assert normalized.dataset_name == "credit-default"


# ---------------------------------------------------------------------------
# CLI boundary
# ---------------------------------------------------------------------------


class TestHistoryCLI:
    def test_history_list_cli_invokes_service(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr(
            cli_module,
            "load_settings",
            lambda: type(
                "Settings",
                (),
                {"tracking": TrackingSettings()},
            )(),
        )

        captured_args = {}

        class _FakeHistoryService:
            def __init__(self, **kwargs):
                captured_args.update(kwargs)

            def list_runs(self, **kwargs):
                return [
                    _make_run(run_id="run-test-001"),
                ]

        monkeypatch.setattr(
            "app.tracking.history_service.HistoryService",
            _FakeHistoryService,
        )
        monkeypatch.setattr(
            "app.tracking.mlflow_query.is_mlflow_available",
            lambda: True,
        )

        args = type("Args", (), {
            "run_type": "all",
            "task_type": None,
            "sort_by": "start_time",
            "sort_dir": "descending",
            "limit": 20,
        })()

        cli_module.cmd_history_list(args)
        output = capsys.readouterr().out

        assert "Run History" in output
        assert "run-test-001" in output


class TestResolveRunId:
    """Regression tests for run-ID prefix resolution (HistoryService.resolve_run_id)."""

    def test_full_32_char_id_returned_as_is(self, monkeypatch):
        full_id = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
        experiments = [_FakeMLflowExperiment("1", "autotabml-benchmarks")]
        runs = []
        _patch_mlflow_query(monkeypatch, experiments, runs)

        service = HistoryService(default_experiment_names=["autotabml-benchmarks"])
        assert service.resolve_run_id(full_id) == full_id

    def test_unique_prefix_resolves_to_full_id(self, monkeypatch):
        experiments = [_FakeMLflowExperiment("1", "autotabml-benchmarks")]
        runs = [
            _FakeMLflowRun(
                "6357094c357e420cb322c4fb37a1754d", "1",
                params={"task_type": "classification", "target_column": "target", "dataset_fingerprint": "fp"},
                metrics={"best_score": 0.75},
                tags={"mlflow.runName": "benchmark-classification-demo"},
            ),
            _FakeMLflowRun(
                "98dbda0e2725458a926ea14565876d7e", "1",
                params={"task_type": "classification", "target_column": "target", "dataset_fingerprint": "fp"},
                metrics={"best_score": 0.58},
                tags={"mlflow.runName": "benchmark-classification-demo2"},
            ),
        ]
        _patch_mlflow_query(monkeypatch, experiments, runs)

        service = HistoryService(default_experiment_names=["autotabml-benchmarks"])
        assert service.resolve_run_id("6357094c") == "6357094c357e420cb322c4fb37a1754d"

    def test_no_match_prefix_raises_run_not_found(self, monkeypatch):
        from app.tracking.errors import RunNotFoundError

        experiments = [_FakeMLflowExperiment("1", "autotabml-benchmarks")]
        runs = [
            _FakeMLflowRun(
                "6357094c357e420cb322c4fb37a1754d", "1",
                params={"task_type": "classification", "target_column": "target", "dataset_fingerprint": "fp"},
                metrics={},
                tags={"mlflow.runName": "benchmark-classification-demo"},
            ),
        ]
        _patch_mlflow_query(monkeypatch, experiments, runs)

        service = HistoryService(default_experiment_names=["autotabml-benchmarks"])
        with pytest.raises(RunNotFoundError, match="No run found"):
            service.resolve_run_id("ffffffff")

    def test_ambiguous_prefix_raises_run_not_found(self, monkeypatch):
        from app.tracking.errors import RunNotFoundError

        experiments = [_FakeMLflowExperiment("1", "autotabml-benchmarks")]
        runs = [
            _FakeMLflowRun(
                "aabb094c357e420cb322c4fb37a1754d", "1",
                params={"task_type": "classification", "target_column": "target", "dataset_fingerprint": "fp"},
                metrics={},
                tags={"mlflow.runName": "benchmark-classification-a"},
            ),
            _FakeMLflowRun(
                "aabb0e2725458a926ea14565876d7e99", "1",
                params={"task_type": "classification", "target_column": "target", "dataset_fingerprint": "fp"},
                metrics={},
                tags={"mlflow.runName": "benchmark-classification-b"},
            ),
        ]
        _patch_mlflow_query(monkeypatch, experiments, runs)

        service = HistoryService(default_experiment_names=["autotabml-benchmarks"])
        with pytest.raises(RunNotFoundError, match="Ambiguous"):
            service.resolve_run_id("aabb")
