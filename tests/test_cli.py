"""Tests for CLI helpers and default wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app import APP_NAME, __version__
from app.cli import _build_input_spec
from app.config.enums import ExecutionBackend, WorkspaceMode
from app.config.models import AppSettings
from app.ingestion.types import IngestionSourceType
from app.startup import StartupIssue, StartupStatus
from app.storage.models import BatchItemStatus, BatchRunItemRecord, BatchRunRecord, BatchRunStatus


class TestBuildInputSpec:
    def test_infers_local_csv(self, tmp_path: Path):
        path = tmp_path / "train.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")

        spec = _build_input_spec(str(path))

        assert spec.source_type == IngestionSourceType.CSV
        assert spec.path == path

    def test_infers_local_excel(self, tmp_path: Path):
        path = tmp_path / "train.xlsx"
        path.write_bytes(b"not-a-real-excel-file")

        spec = _build_input_spec(str(path))

        assert spec.source_type == IngestionSourceType.EXCEL
        assert spec.path == path

    def test_infers_remote_url(self):
        spec = _build_input_spec("https://example.com/data.csv")

        assert spec.source_type == IngestionSourceType.URL_FILE
        assert spec.url == "https://example.com/data.csv"

    def test_respects_explicit_source_type(self, tmp_path: Path):
        path = tmp_path / "data.txt"
        path.write_text("a|b\n1|2\n", encoding="utf-8")

        spec = _build_input_spec(str(path), source_type="delimited_text")

        assert spec.source_type == IngestionSourceType.DELIMITED_TEXT

    def test_unsupported_suffix_raises(self, tmp_path: Path):
        path = tmp_path / "data.parquet"
        path.write_text("placeholder", encoding="utf-8")

        with pytest.raises(ValueError, match="unsupported file type"):
            _build_input_spec(str(path))


class TestOperationalCommands:
    def test_info_prints_app_defaults(self, monkeypatch, capsys):
        from app import cli as cli_module

        settings = AppSettings()
        settings.workspace_mode = WorkspaceMode.NOTEBOOK
        settings.execution.backend = ExecutionBackend.COLAB_MCP

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)

        cli_module.cmd_info(type("Args", (), {})())
        output = capsys.readouterr().out

        assert APP_NAME in output
        assert __version__ in output
        assert "Workspace mode: notebook" in output
        assert "Execution backend: colab_mcp" in output
        assert "CLI entrypoint: autotabml" in output
        assert "Streamlit entrypoint: app/main.py" in output

    def test_init_local_storage_prints_runtime_summary(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module

        status = StartupStatus(
            artifact_dirs=[tmp_path / "artifacts"],
            database_path=tmp_path / "app.sqlite3",
            issues=[StartupIssue(severity="info", message="ready")],
        )

        fake_settings = AppSettings()
        fake_settings.mlflow.tracking_uri = "sqlite:///test/mlflow.db"
        fake_settings.artifacts.root_dir = tmp_path / "artifacts"
        monkeypatch.setattr(cli_module, "load_settings", lambda: fake_settings)
        monkeypatch.setattr(cli_module, "save_settings", lambda s: None)
        monkeypatch.setattr(
            cli_module,
            "initialize_local_runtime",
            lambda settings, include_optional_network_checks=False: status,
        )

        cli_module.cmd_init_local_storage(type("Args", (), {})())
        output = capsys.readouterr().out

        assert "Local Storage Initialized" in output
        assert str(status.database_path) in output
        assert "[info] ready" in output

    def test_init_local_storage_sets_default_mlflow_uri_when_missing(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module

        status = StartupStatus(artifact_dirs=[tmp_path / "artifacts"])
        settings = AppSettings()
        settings.artifacts.root_dir = tmp_path / "artifacts"
        assert settings.mlflow.tracking_uri is None

        saved_settings = {}
        monkeypatch.setattr(cli_module, "load_settings", lambda: settings)
        monkeypatch.setattr(cli_module, "save_settings", lambda s: saved_settings.update({"uri": s.mlflow.tracking_uri}))
        monkeypatch.setattr(
            cli_module,
            "initialize_local_runtime",
            lambda settings, include_optional_network_checks=False: status,
        )

        cli_module.cmd_init_local_storage(type("Args", (), {})())
        output = capsys.readouterr().out

        assert settings.mlflow.tracking_uri == "sqlite:///artifacts/mlflow/mlflow.db"
        assert "MLflow tracking URI" in output
        assert saved_settings["uri"] == "sqlite:///artifacts/mlflow/mlflow.db"

    def test_doctor_exits_when_errors_are_reported(self, monkeypatch, capsys):
        from app import cli as cli_module

        status = StartupStatus(
            issues=[StartupIssue(severity="error", message="database unavailable")],
        )

        monkeypatch.setattr(cli_module, "load_settings", lambda: object())
        monkeypatch.setattr(
            cli_module,
            "initialize_local_runtime",
            lambda settings, include_optional_network_checks=True: status,
        )

        with pytest.raises(SystemExit):
            cli_module.cmd_doctor(type("Args", (), {})())

        output = capsys.readouterr().out
        assert "AutoTabML Doctor" in output
        assert "database unavailable" in output

    def test_main_version_flag_prints_version(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
        monkeypatch.setattr("sys.argv", ["autotabml", "--version"])

        with pytest.raises(SystemExit) as exc:
            cli_module.main()

        assert exc.value.code == 0
        assert capsys.readouterr().out.strip() == f"autotabml {__version__}"

    def test_uci_list_prints_catalog_rows(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr(
            cli_module,
            "list_available_uci_datasets",
            lambda search=None, area=None, filter=None: [
                {"uci_id": 53, "name": "Iris"},
                {"uci_id": 45, "name": "Heart Disease"},
            ],
        )

        cli_module.cmd_uci_list(
            type("Args", (), {"search": "i", "area": None, "filter": None, "limit": 1})()
        )
        output = capsys.readouterr().out

        assert "=== UCI Datasets ===" in output
        assert "53\tIris" in output
        assert "45\tHeart Disease" not in output

    def test_batch_show_recomputes_counts_from_items(self, monkeypatch, capsys):
        from app import cli as cli_module

        batch = BatchRunRecord(
            batch_id="batch-1",
            batch_name="demo",
            total_datasets=3,
            completed_count=99,
            failed_count=99,
            skipped_count=99,
            status=BatchRunStatus.RUNNING,
        )
        items = [
            BatchRunItemRecord(item_id="1", batch_id="batch-1", uci_id=1, dataset_name="A", status=BatchItemStatus.SUCCESS),
            BatchRunItemRecord(item_id="2", batch_id="batch-1", uci_id=2, dataset_name="B", status=BatchItemStatus.SUCCESS),
            BatchRunItemRecord(item_id="3", batch_id="batch-1", uci_id=3, dataset_name="C", status=BatchItemStatus.FAILED),
        ]

        class _FakeStore:
            def get_batch_run(self, batch_id):
                assert batch_id == "batch-1"
                return batch

            def list_batch_items(self, batch_id, limit=200):
                assert batch_id == "batch-1"
                return items

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda settings: _FakeStore())

        cli_module.cmd_batch_show(type("Args", (), {"batch_id": "batch-1"})())
        output = capsys.readouterr().out

        assert "Total: 3  Success: 2  Failed: 1  Skipped: 0" in output

    def test_batch_history_recomputes_counts_from_items(self, monkeypatch, capsys):
        from app import cli as cli_module

        batch = BatchRunRecord(
            batch_id="batch-1",
            batch_name="demo",
            total_datasets=2,
            completed_count=0,
            failed_count=0,
            skipped_count=9,
            status=BatchRunStatus.PARTIAL,
        )
        items = [
            BatchRunItemRecord(item_id="1", batch_id="batch-1", uci_id=1, dataset_name="A", status=BatchItemStatus.SUCCESS),
            BatchRunItemRecord(item_id="2", batch_id="batch-1", uci_id=2, dataset_name="B", status=BatchItemStatus.SKIPPED),
        ]

        class _FakeStore:
            def list_batch_runs(self, limit=20):
                return [batch]

            def list_batch_items(self, batch_id, limit=200):
                assert batch_id == "batch-1"
                return items

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda settings: _FakeStore())

        cli_module.cmd_batch_history(type("Args", (), {"limit": 10})())
        output = capsys.readouterr().out

        assert "success=1" in output
        assert "failed=0" in output
        assert "skipped=1" in output


class TestExperimentCommandsErrorHandling:
    """Regression: experiment-tune/evaluate/save must catch setup errors cleanly."""

    def _make_args(self, tmp_path: Path, command: str):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n", encoding="utf-8")
        return type("Args", (), {
            "dataset": str(csv),
            "target": "target",
            "task_type": "classification",
            "source_type": None,
            "model_id": "lr",
            "tune_metric": None,
            "n_iter": 10,
            "plot": [],
            "save_name": "test_model",
            "save_snapshot": False,
        })()

    def test_experiment_tune_catches_setup_error(self, monkeypatch, tmp_path):
        from app import cli as cli_module

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)

        class _FailingService:
            def setup_experiment(self, *a, **kw):
                raise RuntimeError("PyCaret is not available")

        monkeypatch.setattr(cli_module, "_build_pycaret_service", lambda s, **kw: _FailingService())

        args = self._make_args(tmp_path, "experiment-tune")
        with pytest.raises(SystemExit):
            cli_module.cmd_experiment_tune(args)

    def test_experiment_evaluate_catches_setup_error(self, monkeypatch, tmp_path):
        from app import cli as cli_module

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)

        class _FailingService:
            def setup_experiment(self, *a, **kw):
                raise RuntimeError("PyCaret is not available")

        monkeypatch.setattr(cli_module, "_build_pycaret_service", lambda s, **kw: _FailingService())

        args = self._make_args(tmp_path, "experiment-evaluate")
        with pytest.raises(SystemExit):
            cli_module.cmd_experiment_evaluate(args)

    def test_experiment_save_catches_setup_error(self, monkeypatch, tmp_path):
        from app import cli as cli_module

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)

        class _FailingService:
            def setup_experiment(self, *a, **kw):
                raise RuntimeError("PyCaret is not available")

        monkeypatch.setattr(cli_module, "_build_pycaret_service", lambda s, **kw: _FailingService())

        args = self._make_args(tmp_path, "experiment-save")
        with pytest.raises(SystemExit):
            cli_module.cmd_experiment_save(args)


class TestExperimentCommandsHappyPath:
    """Happy-path coverage for experiment-tune, experiment-evaluate, experiment-save."""

    def _make_args(self, tmp_path: Path, **overrides):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n", encoding="utf-8")
        defaults = {
            "dataset": str(csv),
            "target": "target",
            "task_type": "classification",
            "source_type": None,
            "model_id": "lr",
            "tune_metric": None,
            "n_iter": 10,
            "plot": ["confusion_matrix"],
            "save_name": "test_model",
            "save_snapshot": False,
        }
        defaults.update(overrides)
        return type("Args", (), defaults)()

    def test_experiment_tune_happy_path(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module

        bundle = SimpleNamespace(
            tuned_result=SimpleNamespace(
                optimize_metric="Accuracy",
                baseline_score=0.80,
                tuned_score=0.88,
            ),
        )

        class FakeService:
            def setup_experiment(self, *a, **kw):
                return bundle

            def tune_model(self, b, selection):
                return b

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)
        monkeypatch.setattr(cli_module, "_build_pycaret_service", lambda s, **kw: FakeService())

        cli_module.cmd_experiment_tune(self._make_args(tmp_path))
        output = capsys.readouterr().out

        assert "Experiment Tune: data" in output
        assert "Model: lr" in output
        assert "Optimization score: Accuracy" in output
        assert "Baseline score: 0.8" in output
        assert "Tuned score: 0.88" in output

    def test_experiment_evaluate_happy_path(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module

        bundle = SimpleNamespace(
            evaluation_plots=[
                SimpleNamespace(plot_id="confusion_matrix", path="/tmp/cm.png"),
            ],
            warnings=["Some warning"],
        )

        class FakeService:
            def setup_experiment(self, *a, **kw):
                return bundle

            def evaluate_model(self, b, selection):
                return b

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)
        monkeypatch.setattr(cli_module, "_build_pycaret_service", lambda s, **kw: FakeService())

        cli_module.cmd_experiment_evaluate(self._make_args(tmp_path))
        output = capsys.readouterr().out

        assert "Experiment Evaluate: data" in output
        assert "confusion_matrix" in output
        assert "Some warning" in output

    def test_experiment_save_happy_path(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module

        bundle = SimpleNamespace(
            saved_model_metadata=SimpleNamespace(
                model_path="/tmp/test_model",
                experiment_snapshot_path="/tmp/snapshot.pkl",
            ),
        )

        class FakeService:
            def setup_experiment(self, *a, **kw):
                return bundle

            def finalize_and_save_model(self, b, selection, save_name=None):
                return b

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)
        monkeypatch.setattr(cli_module, "_build_pycaret_service", lambda s, **kw: FakeService())

        cli_module.cmd_experiment_save(self._make_args(tmp_path))
        output = capsys.readouterr().out

        assert "Experiment Save: data" in output
        assert "Model path: /tmp/test_model" in output
        assert "Snapshot: /tmp/snapshot.pkl" in output


class TestCliOutputEncoding:
    """Regression tests for CLI output encoding on Windows cp1252."""

    def test_compare_output_is_ascii_safe(self):
        """Ensure compare CLI output uses ASCII-safe arrows (->), not Unicode → (\u2192)."""
        import inspect

        from app import cli as cli_module

        source = inspect.getsource(cli_module.cmd_compare_runs)
        assert "\u2192" not in source, "cmd_compare_runs still contains Unicode arrow \u2192"

    def test_registry_list_output_is_ascii_safe(self):
        import inspect

        from app import cli as cli_module

        source = inspect.getsource(cli_module.cmd_registry_list)
        assert "\u2192" not in source, "cmd_registry_list still contains Unicode arrow \u2192"

    def test_registry_promote_output_is_ascii_safe(self):
        import inspect

        from app import cli as cli_module

        source = inspect.getsource(cli_module.cmd_registry_promote)
        assert "\u2192" not in source, "cmd_registry_promote still contains Unicode arrow \u2192"

    def test_registry_service_alias_messages_are_ascii_safe(self):
        import inspect

        from app.registry import registry_service

        source = inspect.getsource(registry_service.RegistryService.promote)
        assert "\u2192" not in source, "RegistryService.promote still contains Unicode arrow \u2192"


class TestCmdValidate:
    """Direct functional tests for cmd_validate."""

    def _make_args(self, dataset_path: str, target: str = "approved"):
        return type("Args", (), {
            "dataset": dataset_path,
            "target": target,
            "source_type": None,
            "min_rows": None,
            "artifacts_dir": None,
        })()

    def test_validate_happy_path(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module
        from app.validation.schemas import (
            CheckResult,
            CheckSeverity,
            ValidationArtifactBundle,
            ValidationResultSummary,
        )

        csv = tmp_path / "data.csv"
        csv.write_text("a,b,approved\n1,2,0\n3,4,1\n", encoding="utf-8")

        summary = ValidationResultSummary(
            dataset_name="data",
            row_count=2,
            column_count=3,
            total_checks=2,
            passed_count=2,
            warning_count=0,
            failed_count=0,
            checks=[
                CheckResult(check_name="not_empty", passed=True, severity=CheckSeverity.INFO, message="OK"),
                CheckResult(check_name="target_exists", passed=True, severity=CheckSeverity.INFO, message="OK"),
            ],
        )
        bundle = ValidationArtifactBundle(summary_json_path=tmp_path / "v.json")

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)

        monkeypatch.setattr(
            "app.validation.service.validate_dataset",
            lambda *a, **kw: (summary, bundle),
        )

        cli_module.cmd_validate(self._make_args(str(csv)))
        output = capsys.readouterr().out

        assert "Validation: data" in output
        assert "Passed: 2" in output
        assert "Failed: 0" in output
        assert "[PASS] not_empty" in output

    def test_validate_exits_on_failure(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module
        from app.validation.schemas import (
            CheckResult,
            CheckSeverity,
            ValidationArtifactBundle,
            ValidationResultSummary,
        )

        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")

        summary = ValidationResultSummary(
            dataset_name="data",
            row_count=1,
            column_count=2,
            total_checks=1,
            passed_count=0,
            warning_count=0,
            failed_count=1,
            checks=[
                CheckResult(check_name="target_exists", passed=False, severity=CheckSeverity.ERROR, message="Missing"),
            ],
        )

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)
        monkeypatch.setattr(
            "app.validation.service.validate_dataset",
            lambda *a, **kw: (summary, ValidationArtifactBundle()),
        )

        with pytest.raises(SystemExit):
            cli_module.cmd_validate(self._make_args(str(csv), target="missing_col"))

        output = capsys.readouterr().out
        assert "Failed: 1" in output
        assert "[FAIL] target_exists" in output


class TestCmdBenchmark:
    def _make_args(self, prefer_gpu="auto"):
        return type("Args", (), {
            "dataset": "ignored.csv",
            "target": "target",
            "task_type": "classification",
            "source_type": None,
            "test_size": None,
            "random_state": None,
            "stratify": "auto",
            "ranking_metric": None,
            "sample_rows": 0,
            "top_k": None,
            "prefer_gpu": prefer_gpu,
            "include_model": [],
            "exclude_model": [],
            "artifacts_dir": None,
        })()

    def test_benchmark_uses_settings_gpu_preference_by_default(self, monkeypatch, capsys):
        from app import cli as cli_module

        settings = AppSettings()
        settings.benchmark.prefer_gpu = True

        loaded = SimpleNamespace(
            dataframe=pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]}),
            metadata=SimpleNamespace(content_hash=None, schema_hash="schema-hash"),
        )
        captured = {}

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)
        monkeypatch.setattr(cli_module, "_load_cli_dataset", lambda dataset, source_type=None: (loaded, "demo"))
        monkeypatch.setattr(cli_module, "_record_loaded_cli_dataset", lambda *a, **kw: None)

        def _fake_benchmark_dataset(df, config, **kwargs):
            captured["prefer_gpu"] = config.prefer_gpu
            return SimpleNamespace(
                summary=SimpleNamespace(
                    task_type=SimpleNamespace(value="classification"),
                    target_column="target",
                    ranking_metric="Balanced Accuracy",
                    ranking_direction=SimpleNamespace(value="descending"),
                    source_row_count=4,
                    benchmark_row_count=4,
                    train_row_count=3,
                    test_row_count=1,
                    best_model_name="ModelA",
                    best_score=0.9,
                    fastest_model_name="ModelA",
                    fastest_model_time_seconds=0.01,
                    benchmark_duration_seconds=0.02,
                    warnings=[],
                ),
                top_models=[],
                artifacts=None,
                mlflow_run_id=None,
            )

        monkeypatch.setattr("app.modeling.benchmark.service.benchmark_dataset", _fake_benchmark_dataset)

        cli_module.cmd_benchmark(self._make_args())

        assert captured["prefer_gpu"] is True
        assert "Benchmark: demo" in capsys.readouterr().out

    def test_benchmark_can_disable_gpu_preference_from_cli(self, monkeypatch, capsys):
        from app import cli as cli_module

        settings = AppSettings()
        settings.benchmark.prefer_gpu = True

        loaded = SimpleNamespace(
            dataframe=pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]}),
            metadata=SimpleNamespace(content_hash=None, schema_hash="schema-hash"),
        )
        captured = {}

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)
        monkeypatch.setattr(cli_module, "_load_cli_dataset", lambda dataset, source_type=None: (loaded, "demo"))
        monkeypatch.setattr(cli_module, "_record_loaded_cli_dataset", lambda *a, **kw: None)

        def _fake_benchmark_dataset(df, config, **kwargs):
            captured["prefer_gpu"] = config.prefer_gpu
            return SimpleNamespace(
                summary=SimpleNamespace(
                    task_type=SimpleNamespace(value="classification"),
                    target_column="target",
                    ranking_metric="Balanced Accuracy",
                    ranking_direction=SimpleNamespace(value="descending"),
                    source_row_count=4,
                    benchmark_row_count=4,
                    train_row_count=3,
                    test_row_count=1,
                    best_model_name="ModelA",
                    best_score=0.9,
                    fastest_model_name="ModelA",
                    fastest_model_time_seconds=0.01,
                    benchmark_duration_seconds=0.02,
                    warnings=[],
                ),
                top_models=[],
                artifacts=None,
                mlflow_run_id=None,
            )

        monkeypatch.setattr("app.modeling.benchmark.service.benchmark_dataset", _fake_benchmark_dataset)

        cli_module.cmd_benchmark(self._make_args(prefer_gpu="false"))

        assert captured["prefer_gpu"] is False
        assert "Benchmark: demo" in capsys.readouterr().out


class TestCmdHistoryList:
    """Direct functional tests for cmd_history_list."""

    def _make_args(self, run_type="all", limit=50, sort_by="start_time", sort_dir="descending", task_type=None):
        return type("Args", (), {
            "run_type": run_type,
            "limit": limit,
            "sort_by": sort_by,
            "sort_dir": sort_dir,
            "task_type": task_type,
        })()

    def test_history_list_prints_runs(self, monkeypatch, capsys):
        from app import cli as cli_module
        from app.tracking.schemas import RunHistoryItem

        fake_runs = [
            RunHistoryItem(
                run_id="abc123def456",
                experiment_id="0",
                run_name="benchmark-classification-demo",
                run_type="benchmark",
                task_type="classification",
                primary_metric_name="best_score",
                primary_metric_value=0.85,
                duration_seconds=0.5,
            ),
        ]

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)

        class FakeHistoryService:
            def __init__(self, **kw):
                pass

            def list_runs(self, **kw):
                return fake_runs

        monkeypatch.setattr("app.tracking.history_service.HistoryService", FakeHistoryService)

        cli_module.cmd_history_list(self._make_args())
        output = capsys.readouterr().out

        assert "Run History (1 runs)" in output
        assert "abc123def456" in output

    def test_history_list_no_runs(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)

        class FakeHistoryService:
            def __init__(self, **kw):
                pass

            def list_runs(self, **kw):
                return []

        monkeypatch.setattr("app.tracking.history_service.HistoryService", FakeHistoryService)

        cli_module.cmd_history_list(self._make_args())
        output = capsys.readouterr().out

        assert "No runs found." in output

    def test_history_list_exits_without_mlflow(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: False)

        with pytest.raises(SystemExit):
            cli_module.cmd_history_list(self._make_args())


class TestCmdRegistryList:
    """Direct functional test for cmd_registry_list."""

    def test_registry_list_prints_models(self, monkeypatch, capsys):
        from app import cli as cli_module
        from app.registry.schemas import RegistryModelSummary

        fake_models = [
            RegistryModelSummary(
                name="my-model",
                version_count=2,
                latest_version="2",
                aliases={"champion": "2"},
                description="A model",
            ),
        ]

        settings = AppSettings()
        settings.tracking.registry_enabled = True
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)

        class FakeRegistryService:
            def __init__(self, **kw):
                pass

            def list_models(self):
                return fake_models

        monkeypatch.setattr("app.registry.registry_service.RegistryService", FakeRegistryService)

        cli_module.cmd_registry_list(self._make_args())
        output = capsys.readouterr().out

        assert "Registered Models (1)" in output
        assert "my-model" in output
        assert "champion->v2" in output

    def _make_args(self):
        return type("Args", (), {})()


class TestCmdPredictSingle:
    """Direct functional test for cmd_predict_single."""

    def test_predict_single_prints_prediction_and_artifacts(self, monkeypatch, capsys):
        from app import cli as cli_module
        from app.prediction.schemas import ModelSourceType, PredictionTaskType

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "_validate_prediction_source_requirements", lambda *a, **kw: None)
        monkeypatch.setattr(
            cli_module,
            "_build_prediction_request_kwargs",
            lambda *a, **kw: {
                "source_type": ModelSourceType.MLFLOW_RUN_MODEL,
                "run_id": "run-123",
                "artifact_path": "model",
                "task_type_hint": PredictionTaskType.CLASSIFICATION,
            },
        )
        monkeypatch.setattr(
            cli_module,
            "_load_prediction_row_payload",
            lambda args: {"age": 35, "income": 55000.0, "credit_score": 720, "loan_amount": 15000.0},
        )

        result = SimpleNamespace(
            loaded_model=SimpleNamespace(
                model_identifier="run-123",
                source_type=ModelSourceType.MLFLOW_RUN_MODEL,
                task_type=PredictionTaskType.CLASSIFICATION,
            ),
            predicted_label=1.0,
            predicted_value=None,
            predicted_score=0.91,
            validation=SimpleNamespace(
                issues=[SimpleNamespace(severity=SimpleNamespace(value="warning"), message="Metadata missing")]
            ),
            artifacts=SimpleNamespace(
                scored_csv_path=Path("artifacts/predictions/pred.csv"),
                summary_json_path=Path("artifacts/predictions/pred.json"),
                metadata_json_path=Path("artifacts/predictions/pred_meta.json"),
            ),
        )

        class FakePredictionService:
            def predict_single(self, request):
                assert request.run_id == "run-123"
                return result

        monkeypatch.setattr(cli_module, "_build_prediction_service", lambda settings: FakePredictionService())

        args = type("Args", (), {})()
        cli_module.cmd_predict_single(args)
        output = capsys.readouterr().out

        assert "Prediction: run-123" in output
        assert "Predicted label: 1.0" in output
        assert "Prediction score: 0.91" in output
        assert "[warning] Metadata missing" in output
        assert "Scored CSV:" in output


class TestCmdPredictBatch:
    """Direct functional test for cmd_predict_batch."""

    def test_predict_batch_prints_summary_and_preview(self, monkeypatch, capsys):
        from app import cli as cli_module
        from app.prediction.schemas import ModelSourceType, PredictionTaskType

        settings = AppSettings()
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)
        monkeypatch.setattr(cli_module, "_validate_prediction_source_requirements", lambda *a, **kw: None)
        monkeypatch.setattr(
            cli_module,
            "_build_prediction_request_kwargs",
            lambda *a, **kw: {
                "source_type": ModelSourceType.MLFLOW_RUN_MODEL,
                "run_id": "run-123",
                "artifact_path": "model",
                "task_type_hint": PredictionTaskType.CLASSIFICATION,
            },
        )

        loaded = SimpleNamespace(
            dataframe=pd.DataFrame({"age": [35], "income": [55000.0], "credit_score": [720], "loan_amount": [15000.0]}),
        )
        monkeypatch.setattr(cli_module, "_load_cli_dataset", lambda *a, **kw: (loaded, "demo_batch"))
        monkeypatch.setattr(cli_module, "_record_loaded_cli_dataset", lambda *a, **kw: None)

        result = SimpleNamespace(
            loaded_model=SimpleNamespace(
                model_identifier="run-123",
                source_type=ModelSourceType.MLFLOW_RUN_MODEL,
                task_type=PredictionTaskType.CLASSIFICATION,
            ),
            summary=SimpleNamespace(
                rows_scored=1,
                input_row_count=1,
                output_artifact_path=Path("artifacts/predictions/batch.csv"),
            ),
            scored_dataframe=pd.DataFrame({"prediction": [1], "predicted_score": [0.88]}),
            validation=SimpleNamespace(issues=[]),
            artifacts=SimpleNamespace(
                scored_csv_path=Path("artifacts/predictions/batch.csv"),
                summary_json_path=Path("artifacts/predictions/batch.json"),
                metadata_json_path=Path("artifacts/predictions/batch_meta.json"),
            ),
        )

        class FakePredictionService:
            def predict_batch(self, request):
                assert request.dataset_name == "demo_batch"
                return result

        monkeypatch.setattr(cli_module, "_build_prediction_service", lambda settings: FakePredictionService())

        args = type(
            "Args",
            (),
            {
                "dataset": "demo.csv",
                "source_type": None,
                "output_path": None,
            },
        )()
        cli_module.cmd_predict_batch(args)
        output = capsys.readouterr().out

        assert "Batch Prediction: demo_batch" in output
        assert "Rows scored: 1/1" in output
        assert "Preview:" in output
        assert "prediction" in output


class TestCmdProfile:
    """Direct functional tests for cmd_profile."""

    def _make_args(self, dataset_path: str):
        return type("Args", (), {
            "dataset": dataset_path,
            "source_type": None,
            "artifacts_dir": None,
        })()

    def test_profile_happy_path(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module
        from app.config.models import ProfilingMode

        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

        summary = SimpleNamespace(
            row_count=2,
            column_count=2,
            numeric_column_count=2,
            categorical_column_count=0,
            missing_cells_pct=0.0,
            duplicate_row_count=0,
            report_mode=ProfilingMode.STANDARD,
            sampling_applied=False,
            high_cardinality_columns=[],
        )
        bundle = SimpleNamespace(
            html_report_path=tmp_path / "report.html",
            summary_json_path=tmp_path / "profile.json",
        )

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)
        monkeypatch.setattr("app.profiling.ydata_runner.is_ydata_available", lambda: True)
        monkeypatch.setattr(
            "app.profiling.service.profile_dataset",
            lambda *a, **kw: (summary, bundle),
        )

        cli_module.cmd_profile(self._make_args(str(csv)))
        output = capsys.readouterr().out

        assert "Profile: data" in output
        assert "Rows: 2  Columns: 2" in output
        assert "Numeric: 2" in output
        assert "Missing: 0.0%" in output

    def test_profile_exits_without_ydata(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module

        monkeypatch.setattr("app.profiling.ydata_runner.is_ydata_available", lambda: False)
        monkeypatch.setattr(
            "app.profiling.ydata_runner.profiling_install_guidance",
            lambda: "install ydata-profiling",
        )

        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")

        with pytest.raises(SystemExit):
            cli_module.cmd_profile(self._make_args(str(csv)))


class TestCmdPredictHistory:
    """Direct functional tests for cmd_predict_history."""

    def test_predict_history_prints_entries(self, monkeypatch, capsys):
        from datetime import datetime, timezone

        from app import cli as cli_module

        entry = SimpleNamespace(
            timestamp=datetime(2026, 4, 6, 12, 0, 0, tzinfo=timezone.utc),
            status=SimpleNamespace(value="success"),
            mode=SimpleNamespace(value="single"),
            model_identifier="lr-model",
            row_count=1,
            output_artifact_path=Path("artifacts/predictions/pred.csv"),
        )

        class FakeService:
            def list_history(self, limit=20):
                return [entry]

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "_build_prediction_service", lambda s: FakeService())

        cli_module.cmd_predict_history(type("Args", (), {"limit": 20})())
        output = capsys.readouterr().out

        assert "Prediction History (1)" in output
        assert "lr-model" in output
        assert "[success]" in output

    def test_predict_history_empty(self, monkeypatch, capsys):
        from app import cli as cli_module

        class FakeService:
            def list_history(self, limit=20):
                return []

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "_build_prediction_service", lambda s: FakeService())

        cli_module.cmd_predict_history(type("Args", (), {"limit": 20})())
        output = capsys.readouterr().out

        assert "No prediction history found." in output


class TestCmdHistoryShow:
    """Direct functional tests for cmd_history_show."""

    def test_history_show_prints_run_detail(self, monkeypatch, capsys):
        from app import cli as cli_module

        detail = SimpleNamespace(
            run_id="run-abc-123",
            run_name="benchmark-regression-diabetes",
            run_type=SimpleNamespace(value="benchmark"),
            task_type="regression",
            status=SimpleNamespace(value="FINISHED"),
            duration_seconds=4.5,
            model_name="LinearRegression",
            primary_metric_name="R2",
            primary_metric_value=0.42,
            params={"random_state": "42", "test_size": "0.2"},
            metrics={"R2": 0.42, "MAE": 53.1},
            artifact_paths=["model/MLmodel", "model/model.pkl"],
        )

        class FakeHistoryService:
            def __init__(self, **kw):
                pass

            def resolve_run_id(self, run_id):
                return run_id

            def get_run_detail(self, run_id):
                return detail

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)
        monkeypatch.setattr("app.tracking.history_service.HistoryService", FakeHistoryService)

        cli_module.cmd_history_show(type("Args", (), {"run_id": "run-abc-123"})())
        output = capsys.readouterr().out

        assert "Run Detail: run-abc-123" in output
        assert "Name: benchmark-regression-diabetes" in output
        assert "Type: benchmark  Task: regression" in output
        assert "Status: FINISHED" in output
        assert "Duration: 4.5s" in output
        assert "Model: LinearRegression" in output
        assert "Primary metric: R2 = 0.42" in output
        assert "random_state = 42" in output
        assert "R2 = 0.42" in output
        assert "model/MLmodel" in output

    def test_history_show_exits_without_mlflow(self, monkeypatch):
        from app import cli as cli_module

        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: False)

        with pytest.raises(SystemExit):
            cli_module.cmd_history_show(type("Args", (), {"run_id": "run-1"})())


class TestCmdRegistryShow:
    """Direct functional tests for cmd_registry_show."""

    def test_registry_show_prints_versions(self, monkeypatch, capsys):
        from app import cli as cli_module

        version = SimpleNamespace(
            version="3",
            status="READY",
            run_id="run-abc-123",
            app_status="champion",
            aliases=["champion"],
        )

        class FakeRegistryService:
            def __init__(self, **kw):
                pass

            def list_versions(self, model_name):
                assert model_name == "my-model"
                return [version]

        settings = AppSettings()
        settings.tracking.registry_enabled = True
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)
        monkeypatch.setattr("app.registry.registry_service.RegistryService", FakeRegistryService)

        cli_module.cmd_registry_show(type("Args", (), {"model_name": "my-model"})())
        output = capsys.readouterr().out

        assert "Versions of 'my-model' (1)" in output
        assert "v3" in output
        assert "status=READY" in output
        assert "app_status=champion" in output
        assert "champion" in output

    def test_registry_show_no_versions(self, monkeypatch, capsys):
        from app import cli as cli_module

        class FakeRegistryService:
            def __init__(self, **kw):
                pass

            def list_versions(self, model_name):
                return []

        settings = AppSettings()
        settings.tracking.registry_enabled = True
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)
        monkeypatch.setattr("app.registry.registry_service.RegistryService", FakeRegistryService)

        cli_module.cmd_registry_show(type("Args", (), {"model_name": "no-model"})())
        output = capsys.readouterr().out

        assert "No versions found" in output

    def test_registry_show_exits_when_disabled(self, monkeypatch, capsys):
        from app import cli as cli_module

        settings = AppSettings()
        settings.tracking.registry_enabled = False
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)

        with pytest.raises(SystemExit):
            cli_module.cmd_registry_show(type("Args", (), {"model_name": "m"})())


class TestCmdRegistryRegister:
    """Direct functional tests for cmd_registry_register."""

    def test_registry_register_prints_version(self, monkeypatch, capsys):
        from app import cli as cli_module

        registered = SimpleNamespace(model_name="my-model", version="1")

        class FakeRegistryService:
            def __init__(self, **kw):
                pass

            def register_model(self, model_name, source, run_id=None, description=""):
                assert model_name == "my-model"
                assert source == "runs:/run-1/model"
                return registered

        settings = AppSettings()
        settings.tracking.registry_enabled = True
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)
        monkeypatch.setattr("app.registry.registry_service.RegistryService", FakeRegistryService)

        args = type("Args", (), {
            "model_name": "my-model",
            "source": "runs:/run-1/model",
            "run_id": None,
            "description": None,
        })()
        cli_module.cmd_registry_register(args)
        output = capsys.readouterr().out

        assert "Registered model 'my-model' version 1" in output


class TestCmdRegistryPromote:
    """Direct functional tests for cmd_registry_promote."""

    def test_registry_promote_prints_result(self, monkeypatch, capsys):
        from app import cli as cli_module

        result = SimpleNamespace(
            model_name="my-model",
            version="2",
            action=SimpleNamespace(value="champion"),
            alias_changes=["champion -> v2"],
            tag_changes=["app.status = champion"],
            warnings=[],
        )

        class FakeRegistryService:
            def __init__(self, **kw):
                pass

            def promote(self, request):
                assert request.model_name == "my-model"
                assert request.version == "2"
                return result

        settings = AppSettings()
        settings.tracking.registry_enabled = True
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)
        monkeypatch.setattr("app.registry.registry_service.RegistryService", FakeRegistryService)

        args = type("Args", (), {
            "model_name": "my-model",
            "version": "2",
            "action": "champion",
        })()
        cli_module.cmd_registry_promote(args)
        output = capsys.readouterr().out

        assert "Promoted 'my-model' v2 -> champion" in output
        assert "champion -> v2" in output
        assert "app.status = champion" in output

    def test_registry_promote_exits_when_disabled(self, monkeypatch):
        from app import cli as cli_module

        settings = AppSettings()
        settings.tracking.registry_enabled = False
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)

        args = type("Args", (), {"model_name": "m", "version": "1", "action": "champion"})()
        with pytest.raises(SystemExit):
            cli_module.cmd_registry_promote(args)


class TestCmdDoctorHappyPath:
    """Direct test for cmd_doctor when no errors are found."""

    def test_doctor_happy_path_prints_summary(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module

        status = StartupStatus(
            artifact_dirs=[tmp_path / "artifacts"],
            database_path=tmp_path / "app.sqlite3",
            issues=[StartupIssue(severity="info", message="all good")],
        )

        monkeypatch.setattr(cli_module, "load_settings", lambda: AppSettings())
        monkeypatch.setattr(
            cli_module,
            "initialize_local_runtime",
            lambda settings, include_optional_network_checks=True: status,
        )

        cli_module.cmd_doctor(type("Args", (), {})())
        output = capsys.readouterr().out

        assert "AutoTabML Doctor" in output
        assert "CUDA available:" in output
        assert str(tmp_path / "app.sqlite3") in output
        assert "Artifact directories checked: 1" in output
        assert "[info] all good" in output

    def test_doctor_clean_system_no_issues(self, monkeypatch, capsys):
        from app import cli as cli_module

        status = StartupStatus(issues=[])
        monkeypatch.setattr(cli_module, "load_settings", lambda: AppSettings())
        monkeypatch.setattr(
            cli_module,
            "initialize_local_runtime",
            lambda settings, include_optional_network_checks=True: status,
        )

        cli_module.cmd_doctor(type("Args", (), {})())
        output = capsys.readouterr().out

        assert "[ok] no startup issues detected" in output


class TestCmdCompareRuns:
    """Direct functional tests for cmd_compare_runs."""

    def test_compare_runs_prints_comparison(self, monkeypatch, capsys):
        from app import cli as cli_module
        from app.tracking.schemas import (
            ComparisonBundle,
            ConfigDifference,
            MetricDelta,
            RunHistoryItem,
        )

        left = RunHistoryItem(
            run_id="aaaa1111bbbb2222",
            experiment_id="0",
            run_name="benchmark-run-left",
            run_type="benchmark",
        )
        right = RunHistoryItem(
            run_id="cccc3333dddd4444",
            experiment_id="0",
            run_name="benchmark-run-right",
            run_type="benchmark",
        )
        bundle = ComparisonBundle(
            left=left,
            right=right,
            comparable=True,
            metric_deltas=[
                MetricDelta(name="R2", left_value=0.42, right_value=0.55, delta=0.13, better_side="right"),
            ],
            config_differences=[
                ConfigDifference(key="random_state", left_value="42", right_value="99"),
            ],
        )

        detail_left = SimpleNamespace(
            run_id="aaaa1111bbbb2222",
            run_name="benchmark-run-left",
            run_type=SimpleNamespace(value="benchmark"),
            task_type="regression",
            status=SimpleNamespace(value="FINISHED"),
            duration_seconds=1.0,
            model_name=None,
            primary_metric_name=None,
            primary_metric_value=None,
            params={},
            metrics={"R2": 0.42},
            artifact_paths=[],
        )
        detail_right = SimpleNamespace(
            run_id="cccc3333dddd4444",
            run_name="benchmark-run-right",
            run_type=SimpleNamespace(value="benchmark"),
            task_type="regression",
            status=SimpleNamespace(value="FINISHED"),
            duration_seconds=2.0,
            model_name=None,
            primary_metric_name=None,
            primary_metric_value=None,
            params={},
            metrics={"R2": 0.55},
            artifact_paths=[],
        )

        class FakeHistoryService:
            def __init__(self, **kw):
                pass

            def resolve_run_id(self, run_id):
                return run_id

            def get_run_detail(self, run_id):
                return detail_left if run_id == "aaaa1111bbbb2222" else detail_right

        class FakeComparisonService:
            def compare(self, left, right):
                return bundle

        settings = AppSettings()
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)
        monkeypatch.setattr("app.tracking.history_service.HistoryService", FakeHistoryService)
        monkeypatch.setattr("app.tracking.compare_service.ComparisonService", FakeComparisonService)

        args = type("Args", (), {
            "left_run_id": "aaaa1111bbbb2222",
            "right_run_id": "cccc3333dddd4444",
            "artifacts_dir": None,
        })()
        cli_module.cmd_compare_runs(args)
        output = capsys.readouterr().out

        assert "Comparison:" in output
        assert "Comparable: Yes" in output
        assert "R2:" in output
        assert "0.4200" in output
        assert "0.5500" in output
        assert "+0.1300" in output
        assert "(right)" in output
        assert "random_state:" in output

    def test_compare_runs_exits_without_mlflow(self, monkeypatch):
        from app import cli as cli_module

        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: False)

        args = type("Args", (), {"left_run_id": "a", "right_run_id": "b", "artifacts_dir": None})()
        with pytest.raises(SystemExit):
            cli_module.cmd_compare_runs(args)


class TestCmdRegistryGates:
    """Test that registry commands exit cleanly when registry is disabled or mlflow missing."""

    def test_registry_list_exits_when_disabled(self, monkeypatch):
        from app import cli as cli_module

        settings = AppSettings()
        settings.tracking.registry_enabled = False
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)

        with pytest.raises(SystemExit):
            cli_module.cmd_registry_list(type("Args", (), {})())

    def test_registry_register_exits_when_disabled(self, monkeypatch):
        from app import cli as cli_module

        settings = AppSettings()
        settings.tracking.registry_enabled = False
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)

        args = type("Args", (), {"model_name": "m", "source": "s", "run_id": None, "description": None})()
        with pytest.raises(SystemExit):
            cli_module.cmd_registry_register(args)

    def test_registry_list_exits_without_mlflow(self, monkeypatch):
        from app import cli as cli_module

        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: False)

        with pytest.raises(SystemExit):
            cli_module.cmd_registry_list(type("Args", (), {})())

    def test_registry_register_exits_without_mlflow(self, monkeypatch):
        from app import cli as cli_module

        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: False)

        args = type("Args", (), {"model_name": "m", "source": "s", "run_id": None, "description": None})()
        with pytest.raises(SystemExit):
            cli_module.cmd_registry_register(args)

    def test_registry_promote_exits_without_mlflow(self, monkeypatch):
        from app import cli as cli_module

        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: False)

        args = type("Args", (), {"model_name": "m", "version": "1", "action": "champion"})()
        with pytest.raises(SystemExit):
            cli_module.cmd_registry_promote(args)


class TestCmdExperimentRun:
    """Direct functional tests for cmd_experiment_run."""

    def test_experiment_run_happy_path(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module

        csv = tmp_path / "data.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n", encoding="utf-8")

        bundle = SimpleNamespace(
            summary=SimpleNamespace(
                task_type=SimpleNamespace(value="classification"),
                target_column="target",
                compare_optimize_metric="Accuracy",
                best_baseline_model_name="LogisticRegression",
                best_baseline_score=0.85,
                experiment_duration_seconds=2.5,
            ),
            mlflow_run_id="run-exp-123",
        )

        class FakeService:
            def run_compare_pipeline(self, df, config, **kw):
                return bundle

        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)
        monkeypatch.setattr(cli_module, "_build_pycaret_service", lambda s, **kw: FakeService())

        args = type("Args", (), {
            "dataset": str(csv),
            "target": "target",
            "task_type": "classification",
            "source_type": None,
            "train_size": None,
            "fold": None,
            "fold_strategy": None,
            "preprocess": "auto",
            "ignore_feature": [],
            "compare_metric": None,
            "n_select": 1,
            "budget_time": None,
            "turbo": True,
            "use_gpu": None,
            "artifacts_dir": None,
        })()
        cli_module.cmd_experiment_run(args)
        output = capsys.readouterr().out

        assert "Experiment: data" in output
        assert "Task: classification" in output
        assert "Best baseline: LogisticRegression" in output
        assert "MLflow run id: run-exp-123" in output


class TestParserIntegration:
    """Tests that exercise argparse parsing through main()."""

    def test_info_subcommand_dispatches(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr("sys.argv", ["autotabml", "info"])

        cli_module.main()
        output = capsys.readouterr().out

        assert APP_NAME in output
        assert __version__ in output

    def test_doctor_subcommand_dispatches(self, monkeypatch, capsys):
        from app import cli as cli_module

        status = StartupStatus(issues=[])
        monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
        monkeypatch.setattr(cli_module, "load_settings", lambda: AppSettings())
        monkeypatch.setattr(
            cli_module,
            "initialize_local_runtime",
            lambda settings, include_optional_network_checks=True: status,
        )
        monkeypatch.setattr("sys.argv", ["autotabml", "doctor"])

        cli_module.main()
        output = capsys.readouterr().out

        assert "AutoTabML Doctor" in output

    def test_unknown_subcommand_exits(self, monkeypatch):
        from app import cli as cli_module

        monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
        monkeypatch.setattr("sys.argv", ["autotabml", "not-a-command"])

        with pytest.raises(SystemExit):
            cli_module.main()

    def test_no_subcommand_exits(self, monkeypatch):
        from app import cli as cli_module

        monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
        monkeypatch.setattr("sys.argv", ["autotabml"])

        with pytest.raises(SystemExit):
            cli_module.main()

    def test_init_local_storage_subcommand_dispatches(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module

        status = StartupStatus(artifact_dirs=[tmp_path / "artifacts"])
        settings = AppSettings()
        settings.mlflow.tracking_uri = "sqlite:///test.db"
        settings.artifacts.root_dir = tmp_path / "artifacts"
        monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
        monkeypatch.setattr(cli_module, "load_settings", lambda: settings)
        monkeypatch.setattr(cli_module, "save_settings", lambda s: None)
        monkeypatch.setattr(
            cli_module,
            "initialize_local_runtime",
            lambda settings, include_optional_network_checks=False: status,
        )
        monkeypatch.setattr("sys.argv", ["autotabml", "init-local-storage"])

        cli_module.main()
        output = capsys.readouterr().out

        assert "Local Storage Initialized" in output

    def test_history_list_subcommand_dispatches(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)

        class FakeHistoryService:
            def __init__(self, **kw):
                pass

            def list_runs(self, **kw):
                return []

        monkeypatch.setattr("app.tracking.history_service.HistoryService", FakeHistoryService)
        monkeypatch.setattr("sys.argv", ["autotabml", "history-list"])

        cli_module.main()
        output = capsys.readouterr().out

        assert "No runs found." in output

    def test_predict_history_subcommand_dispatches(self, monkeypatch, capsys):
        from app import cli as cli_module

        class FakeService:
            def list_history(self, limit=20):
                return []

        monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "_build_prediction_service", lambda s: FakeService())
        monkeypatch.setattr("sys.argv", ["autotabml", "predict-history"])

        cli_module.main()
        output = capsys.readouterr().out

        assert "No prediction history found." in output

    def test_registry_list_subcommand_dispatches(self, monkeypatch, capsys):
        from app import cli as cli_module

        settings = AppSettings()
        settings.tracking.registry_enabled = True
        monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: settings)
        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)

        class FakeRegistryService:
            def __init__(self, **kw):
                pass

            def list_models(self):
                return []

        monkeypatch.setattr("app.registry.registry_service.RegistryService", FakeRegistryService)
        monkeypatch.setattr("sys.argv", ["autotabml", "registry-list"])

        cli_module.main()
        output = capsys.readouterr().out

        assert "No registered models found." in output

    def test_validate_parser_resolves_args(self, monkeypatch, capsys, tmp_path: Path):
        from app import cli as cli_module
        from app.validation.schemas import (
            CheckResult,
            CheckSeverity,
            ValidationArtifactBundle,
            ValidationResultSummary,
        )

        csv = tmp_path / "sample.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")

        summary = ValidationResultSummary(
            dataset_name="sample",
            row_count=1,
            column_count=2,
            total_checks=1,
            passed_count=1,
            warning_count=0,
            failed_count=0,
            checks=[
                CheckResult(check_name="not_empty", passed=True, severity=CheckSeverity.INFO, message="OK"),
            ],
        )

        monkeypatch.setattr(cli_module, "configure_logging", lambda: None)
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())
        monkeypatch.setattr(cli_module, "build_metadata_store", lambda s: None)
        monkeypatch.setattr(
            "app.validation.service.validate_dataset",
            lambda *a, **kw: (summary, ValidationArtifactBundle()),
        )
        monkeypatch.setattr("sys.argv", ["autotabml", "validate", str(csv), "--target", "a"])

        cli_module.main()
        output = capsys.readouterr().out

        assert "Validation: sample" in output
        assert "Passed: 1" in output
    """Direct functional test for cmd_compare_runs."""

    def test_compare_runs_prints_deltas_and_config_diff(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr("app.tracking.mlflow_query.is_mlflow_available", lambda: True)
        monkeypatch.setattr(cli_module, "_load_runtime_settings", lambda: AppSettings())

        left = SimpleNamespace(run_id="left-run-1234567890")
        right = SimpleNamespace(run_id="right-run-0987654321")

        class FakeHistoryService:
            def __init__(self, **kw):
                pass

            def resolve_run_id(self, run_id):
                return run_id

            def get_run_detail(self, run_id):
                return left if run_id == "left-run-1234567890" else right

        bundle = SimpleNamespace(
            comparable=False,
            warnings=["Primary metric metadata is missing."],
            metric_deltas=[
                SimpleNamespace(
                    name="accuracy",
                    left_value=0.5,
                    right_value=0.7,
                    delta=0.2,
                    better_side="right",
                )
            ],
            config_differences=[
                SimpleNamespace(key="random_state", left_value="42", right_value="7")
            ],
        )

        class FakeComparisonService:
            def compare(self, left_run, right_run):
                return bundle

        monkeypatch.setattr("app.tracking.history_service.HistoryService", FakeHistoryService)
        monkeypatch.setattr("app.tracking.compare_service.ComparisonService", FakeComparisonService)

        args = type(
            "Args",
            (),
            {"left_run_id": left.run_id, "right_run_id": right.run_id, "artifacts_dir": None},
        )()
        cli_module.cmd_compare_runs(args)
        output = capsys.readouterr().out

        assert "Comparison: left-run-123 vs right-run-09" in output
        assert "Comparable: No" in output
        assert "accuracy: 0.5000 -> 0.7000  +0.2000 (right)" in output
        assert "random_state: 42 -> 7" in output