"""CLI entry points for AutoTabML Studio.

Usage:
    python -m app.cli --version
    python -m app.cli info
    python -m app.cli uci-list [--search <query>] [--area <area>] [--filter <filter>] [--limit <n>]
    python -m app.cli init-local-storage
    python -m app.cli doctor
    python -m app.cli validate <dataset> [--target <col>] [--source-type <type>] [--artifacts-dir <dir>]
    python -m app.cli profile  <dataset> [--source-type <type>] [--artifacts-dir <dir>]
    python -m app.cli benchmark <dataset> --target <col> [--task-type <type>] [--artifacts-dir <dir>]
    python -m app.cli experiment-run <dataset> --target <col> [--task-type <type>]

    Dataset locators:
        path/to/file.csv          Local file
        https://example.com/d.csv Remote URL
        uci:53                    UCI ML Repository by ID (e.g. Iris)
        uci:Heart Disease         UCI ML Repository by name
    python -m app.cli experiment-tune <dataset> --target <col> --model-id <id> --task-type <type>
    python -m app.cli experiment-evaluate <dataset> --target <col> --model-id <id> --task-type <type>
    python -m app.cli experiment-save <dataset> --target <col> --model-id <id> --task-type <type>
    python -m app.cli predict-single --model-source <source> [--row-json <json>]
    python -m app.cli predict-batch <dataset> --model-source <source>
    python -m app.cli predict-history [--limit <n>]
    python -m app.cli history-list [--run-type <type>] [--limit <n>]
    python -m app.cli history-show <run_id>
    python -m app.cli compare-runs <left_run_id> <right_run_id>
    python -m app.cli registry-list
    python -m app.cli registry-show <model_name>
    python -m app.cli registry-register <model_name> --source <uri>
    python -m app.cli registry-promote <model_name> <version> --action <action>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from app import APP_NAME, CLI_ENTRYPOINT, DIST_NAME, STREAMLIT_ENTRYPOINT, __version__
from app.config.settings import load_settings, save_settings
from app.gpu import cuda_summary
from app.ingestion import DatasetInputSpec, IngestionSourceType, load_dataset
from app.ingestion.errors import IngestionError
from app.ingestion.types import DELIMITED_FILE_SUFFIXES, EXCEL_FILE_SUFFIXES
from app.ingestion.uci_loader import list_available_uci_datasets
from app.logging_config import configure_logging
from app.modeling.pycaret.service import PyCaretExperimentService
from app.security.masking import safe_error_message
from app.startup import format_startup_issues, initialize_local_runtime
from app.storage import BatchItemStatus, build_metadata_store, ensure_dataset_record


def _cli_error(exc: Exception) -> None:
    """Print a redacted error message to stderr and exit."""
    print(f"Error: {safe_error_message(exc)}", file=sys.stderr)
    sys.exit(1)


def _load_runtime_settings():  # noqa: ANN201
    try:
        return load_settings()
    except Exception as exc:
        _cli_error(exc)


def _build_input_spec(locator: str, source_type: str | None = None) -> DatasetInputSpec:
    """Build an ingestion input spec from a CLI dataset locator."""

    # UCI Repository shorthand: uci:<id> or uci:<name>
    if locator.startswith("uci:"):
        uci_value = locator[4:].strip()
        if not uci_value:
            raise ValueError("UCI locator must be 'uci:<id>' or 'uci:<name>'")
        try:
            uci_id = int(uci_value)
            return DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=uci_id)
        except ValueError:
            return DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_name=uci_value)

    if source_type is not None:
        resolved_type = IngestionSourceType(source_type)
    elif _is_url(locator):
        resolved_type = IngestionSourceType.URL_FILE
    else:
        path = Path(locator)
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in EXCEL_FILE_SUFFIXES:
            resolved_type = IngestionSourceType.EXCEL
        elif suffix in DELIMITED_FILE_SUFFIXES:
            resolved_type = (
                IngestionSourceType.CSV if suffix == ".csv" else IngestionSourceType.DELIMITED_TEXT
            )
        else:
            raise ValueError(f"unsupported file type: {suffix}")

    if _is_url(locator):
        return DatasetInputSpec(source_type=resolved_type, url=locator)

    return DatasetInputSpec(source_type=resolved_type, path=Path(locator))


def _is_url(locator: str) -> bool:
    parsed = urlparse(locator)
    return parsed.scheme in {"http", "https"}


def _load_cli_dataset(locator: str, source_type: str | None = None):  # noqa: ANN201
    """Load a dataset through the ingestion layer for CLI use."""

    spec = _build_input_spec(locator, source_type=source_type)
    loaded = load_dataset(spec)
    dataset_name = loaded.metadata.display_name or loaded.metadata.source_locator
    if loaded.metadata.display_name is None and loaded.input_spec and loaded.input_spec.path:
        dataset_name = loaded.input_spec.path.stem
    return loaded, dataset_name


def _record_loaded_cli_dataset(metadata_store, loaded_dataset, dataset_name: str | None) -> None:  # noqa: ANN001
    if metadata_store is not None:
        ensure_dataset_record(metadata_store, loaded_dataset, dataset_name=dataset_name)


def cmd_validate(args: argparse.Namespace) -> None:
    """Run data validation from CLI."""
    from app.validation.schemas import ValidationRuleConfig
    from app.validation.service import validate_dataset

    settings = _load_runtime_settings()
    metadata_store = build_metadata_store(settings)

    try:
        loaded, name = _load_cli_dataset(args.dataset, source_type=args.source_type)
        _record_loaded_cli_dataset(metadata_store, loaded, name)
    except (FileNotFoundError, ValueError, IngestionError) as exc:
        _cli_error(exc)

    config = ValidationRuleConfig(
        target_column=args.target,
        min_row_count=args.min_rows or settings.validation.min_row_threshold,
        null_warn_pct=settings.validation.null_warn_pct,
        null_fail_pct=settings.validation.null_fail_pct,
    )

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else settings.validation.artifacts_dir
    summary, bundle = validate_dataset(
        loaded.dataframe,
        config,
        dataset_name=name,
        artifacts_dir=artifacts_dir,
        loaded_dataset=loaded,
        metadata_store=metadata_store,
    )

    print(f"\n=== Validation: {name} ===")
    print(f"Rows: {summary.row_count}  Columns: {summary.column_count}")
    print(f"Passed: {summary.passed_count}  Warnings: {summary.warning_count}  Failed: {summary.failed_count}")
    print()

    for check in summary.checks:
        icon = "PASS" if check.passed else ("WARN" if check.severity.value == "warning" else "FAIL")
        print(f"  [{icon}] {check.check_name}: {check.message}")

    if bundle:
        print("\nArtifacts:")
        if bundle.summary_json_path:
            print(f"  JSON:     {bundle.summary_json_path}")
        if bundle.markdown_report_path:
            print(f"  Markdown: {bundle.markdown_report_path}")

    if summary.has_failures:
        sys.exit(1)


def cmd_profile(args: argparse.Namespace) -> None:
    """Run data profiling from CLI."""
    from app.profiling.ydata_runner import is_ydata_available, profiling_install_guidance

    if not is_ydata_available():
        print(f"Error: {profiling_install_guidance()}", file=sys.stderr)
        sys.exit(1)

    from app.profiling.service import profile_dataset

    settings = _load_runtime_settings()
    metadata_store = build_metadata_store(settings)

    try:
        loaded, name = _load_cli_dataset(args.dataset, source_type=args.source_type)
        _record_loaded_cli_dataset(metadata_store, loaded, name)
    except (FileNotFoundError, ValueError, IngestionError) as exc:
        _cli_error(exc)

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else settings.profiling.artifacts_dir

    summary, bundle = profile_dataset(
        loaded.dataframe,
        mode=settings.profiling.default_mode,
        dataset_name=name,
        artifacts_dir=artifacts_dir,
        loaded_dataset=loaded,
        metadata_store=metadata_store,
        large_dataset_row_threshold=settings.profiling.large_dataset_row_threshold,
        large_dataset_col_threshold=settings.profiling.large_dataset_col_threshold,
        sampling_row_threshold=settings.profiling.sampling_row_threshold,
        sample_size=settings.profiling.sample_size,
    )

    print(f"\n=== Profile: {name} ===")
    print(f"Rows: {summary.row_count}  Columns: {summary.column_count}")
    print(f"Numeric: {summary.numeric_column_count}  Categorical: {summary.categorical_column_count}")
    print(f"Missing: {summary.missing_cells_pct:.1f}%  Duplicates: {summary.duplicate_row_count}")
    print(f"Mode: {summary.report_mode.value}  Sampled: {summary.sampling_applied}")

    if summary.high_cardinality_columns:
        print(f"High-cardinality: {', '.join(summary.high_cardinality_columns)}")

    if bundle:
        print("\nArtifacts:")
        if bundle.html_report_path:
            print(f"  HTML:  {bundle.html_report_path}")
        if bundle.summary_json_path:
            print(f"  JSON:  {bundle.summary_json_path}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run LazyPredict model benchmarking from CLI."""

    from app.modeling.benchmark.errors import BenchmarkError
    from app.modeling.benchmark.schemas import (
        BenchmarkConfig,
        BenchmarkSplitConfig,
        BenchmarkTaskType,
    )
    from app.modeling.benchmark.service import benchmark_dataset

    settings = _load_runtime_settings()
    metadata_store = build_metadata_store(settings)

    try:
        loaded, name = _load_cli_dataset(args.dataset, source_type=args.source_type)
        _record_loaded_cli_dataset(metadata_store, loaded, name)
    except (FileNotFoundError, ValueError, IngestionError) as exc:
        _cli_error(exc)

    if args.stratify == "auto":
        stratify_value = None if settings.benchmark.default_stratify else False
    else:
        stratify_value = args.stratify == "true"

    config = BenchmarkConfig(
        target_column=args.target,
        task_type=BenchmarkTaskType(args.task_type),
        prefer_gpu=settings.benchmark.prefer_gpu if args.prefer_gpu == "auto" else args.prefer_gpu == "true",
        split=BenchmarkSplitConfig(
            test_size=args.test_size or settings.benchmark.default_test_size,
            random_state=args.random_state if args.random_state is not None else settings.benchmark.default_random_state,
            stratify=stratify_value,
        ),
        ranking_metric=args.ranking_metric or None,
        sample_rows=args.sample_rows or None,
        include_models=args.include_model,
        exclude_models=args.exclude_model,
        top_k=args.top_k or settings.benchmark.ui_default_top_k,
        timeout_seconds=settings.benchmark.timeout_seconds,
    )

    dataset_fingerprint = loaded.metadata.content_hash or loaded.metadata.schema_hash
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else settings.benchmark.artifacts_dir

    try:
        bundle = benchmark_dataset(
            loaded.dataframe,
            config,
            dataset_name=name,
            dataset_fingerprint=dataset_fingerprint,
            loaded_dataset=loaded,
            metadata_store=metadata_store,
            execution_backend=settings.execution.backend,
            workspace_mode=settings.workspace_mode,
            artifacts_dir=artifacts_dir,
            classification_default_metric=settings.benchmark.default_classification_ranking_metric,
            regression_default_metric=settings.benchmark.default_regression_ranking_metric,
            sampling_row_threshold=settings.benchmark.sampling_row_threshold,
            suggested_sample_rows=settings.benchmark.suggested_sample_rows,
            mlflow_experiment_name=settings.benchmark.mlflow_experiment_name,
            tracking_uri=settings.tracking.tracking_uri,
            registry_uri=settings.tracking.registry_uri,
        )
    except BenchmarkError as exc:
        _cli_error(exc)

    print(f"\n=== Benchmark: {name} ===")
    print(f"Task: {bundle.summary.task_type.value}  Target: {bundle.summary.target_column}")
    print(
        f"Ranking: {bundle.summary.ranking_metric} "
        f"({bundle.summary.ranking_direction.value})"
    )
    print(
        f"Rows: source={bundle.summary.source_row_count} benchmark={bundle.summary.benchmark_row_count}  "
        f"Train/Test={bundle.summary.train_row_count}/{bundle.summary.test_row_count}"
    )
    if bundle.summary.best_model_name is not None:
        print(f"Best: {bundle.summary.best_model_name} ({bundle.summary.best_score})")
    if bundle.summary.fastest_model_name is not None:
        print(
            f"Fastest: {bundle.summary.fastest_model_name} "
            f"({bundle.summary.fastest_model_time_seconds:.4f}s)"
        )
    print(f"Duration: {bundle.summary.benchmark_duration_seconds:.2f}s")

    print("\nLeaderboard:")
    for row in bundle.top_models:
        print(
            f"  #{row.rank} {row.model_name}: {row.primary_score} "
            f"(time={row.training_time_seconds})"
        )

    if bundle.summary.warnings:
        print("\nWarnings:")
        for warning in bundle.summary.warnings:
            print(f"  - {warning}")

    if bundle.artifacts:
        print("\nArtifacts:")
        if bundle.artifacts.raw_results_csv_path:
            print(f"  Raw CSV:      {bundle.artifacts.raw_results_csv_path}")
        if bundle.artifacts.leaderboard_csv_path:
            print(f"  Leaderboard:  {bundle.artifacts.leaderboard_csv_path}")
        if bundle.artifacts.summary_json_path:
            print(f"  Summary JSON: {bundle.artifacts.summary_json_path}")

    if bundle.mlflow_run_id:
        print(f"\nMLflow run id: {bundle.mlflow_run_id}")


def cmd_experiment_run(args: argparse.Namespace) -> None:
    """Run PyCaret setup + compare_models from CLI."""

    from app.modeling.pycaret.schemas import (
        ExperimentCompareConfig,
        ExperimentConfig,
        ExperimentSetupConfig,
        ExperimentTaskType,
        ExperimentTuneConfig,
    )

    settings = _load_runtime_settings()
    metadata_store = build_metadata_store(settings)

    try:
        loaded, name = _load_cli_dataset(args.dataset, source_type=args.source_type)
        _record_loaded_cli_dataset(metadata_store, loaded, name)
    except (FileNotFoundError, ValueError, IngestionError) as exc:
        _cli_error(exc)

    fold_strategy = args.fold_strategy or _default_experiment_fold_strategy(settings, args.task_type)
    preprocess_value = settings.pycaret.default_preprocess if args.preprocess == "auto" else args.preprocess == "true"
    use_gpu = _resolve_cli_use_gpu(args, settings)
    service = _build_pycaret_service(
        settings,
        artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else settings.pycaret.artifacts_dir,
        metadata_store=metadata_store,
    )
    config = ExperimentConfig(
        target_column=args.target,
        task_type=ExperimentTaskType(args.task_type),
        mlflow_tracking_mode=_default_experiment_tracking_mode(settings),
        setup=ExperimentSetupConfig(
            session_id=settings.pycaret.default_session_id,
            train_size=args.train_size or settings.pycaret.default_train_size,
            fold=args.fold or settings.pycaret.default_fold,
            fold_strategy=fold_strategy,
            ignore_features=args.ignore_feature,
            preprocess=preprocess_value,
            use_gpu=use_gpu,
            log_experiment=_default_experiment_tracking_mode(settings) == _pycaret_native_tracking_mode(),
        ),
        compare=ExperimentCompareConfig(
            optimize=args.compare_metric or None,
            n_select=args.n_select,
            turbo=args.turbo,
            budget_time=args.budget_time,
        ),
        tune=ExperimentTuneConfig(optimize=None),
    )
    dataset_fingerprint = loaded.metadata.content_hash or loaded.metadata.schema_hash

    try:
        bundle = service.run_compare_pipeline(
            loaded.dataframe,
            config,
            dataset_name=name,
            dataset_fingerprint=dataset_fingerprint,
            execution_backend=settings.execution.backend,
            workspace_mode=settings.workspace_mode,
        )
    except Exception as exc:
        _cli_error(exc)

    print(f"\n=== Experiment: {name} ===")
    print(f"Task: {bundle.summary.task_type.value}  Target: {bundle.summary.target_column}")
    print(f"Compare metric: {bundle.summary.compare_optimize_metric}")
    if bundle.summary.best_baseline_model_name is not None:
        print(
            f"Best baseline: {bundle.summary.best_baseline_model_name} "
            f"({bundle.summary.best_baseline_score})"
        )
    print(f"Duration: {bundle.summary.experiment_duration_seconds:.2f}s")
    if bundle.mlflow_run_id:
        print(f"MLflow run id: {bundle.mlflow_run_id}")


def cmd_experiment_tune(args: argparse.Namespace) -> None:
    """Run PyCaret setup + tune_model from CLI."""

    from app.modeling.pycaret.schemas import (
        ExperimentConfig,
        ExperimentEvaluationConfig,
        ExperimentSetupConfig,
        ExperimentTaskType,
        ExperimentTuneConfig,
        ModelSelectionSpec,
    )

    settings = _load_runtime_settings()
    metadata_store = build_metadata_store(settings)
    try:
        loaded, name = _load_cli_dataset(args.dataset, source_type=args.source_type)
        _record_loaded_cli_dataset(metadata_store, loaded, name)
    except (FileNotFoundError, ValueError, IngestionError) as exc:
        _cli_error(exc)

    service = _build_pycaret_service(settings, metadata_store=metadata_store)
    use_gpu = _resolve_cli_use_gpu(args, settings)
    config = ExperimentConfig(
        target_column=args.target,
        task_type=ExperimentTaskType(args.task_type),
        mlflow_tracking_mode=_default_experiment_tracking_mode(settings),
        setup=ExperimentSetupConfig(
            session_id=settings.pycaret.default_session_id,
            train_size=settings.pycaret.default_train_size,
            fold=settings.pycaret.default_fold,
            fold_strategy=_default_experiment_fold_strategy(settings, args.task_type),
            use_gpu=use_gpu,
            log_experiment=_default_experiment_tracking_mode(settings) == _pycaret_native_tracking_mode(),
        ),
        tune=ExperimentTuneConfig(optimize=args.tune_metric or None, n_iter=args.n_iter),
        evaluation=ExperimentEvaluationConfig(plots=[]),
    )
    try:
        bundle = service.setup_experiment(
            loaded.dataframe,
            config,
            dataset_name=name,
            dataset_fingerprint=loaded.metadata.content_hash or loaded.metadata.schema_hash,
            execution_backend=settings.execution.backend,
            workspace_mode=settings.workspace_mode,
        )
        selection = ModelSelectionSpec(model_id=args.model_id, model_name=args.model_id)
        bundle = service.tune_model(bundle, selection)
    except Exception as exc:
        _cli_error(exc)
    print(f"\n=== Experiment Tune: {name} ===")
    print(f"Model: {selection.model_name}")
    print(f"Tune metric: {bundle.tuned_result.optimize_metric if bundle.tuned_result else 'N/A'}")
    if bundle.tuned_result is not None:
        print(f"Baseline score: {bundle.tuned_result.baseline_score}")
        print(f"Tuned score: {bundle.tuned_result.tuned_score}")


def cmd_experiment_evaluate(args: argparse.Namespace) -> None:
    """Run PyCaret setup + evaluation plots from CLI."""

    from app.modeling.pycaret.schemas import (
        ExperimentConfig,
        ExperimentEvaluationConfig,
        ExperimentSetupConfig,
        ExperimentTaskType,
        ModelSelectionSpec,
    )

    settings = _load_runtime_settings()
    metadata_store = build_metadata_store(settings)
    try:
        loaded, name = _load_cli_dataset(args.dataset, source_type=args.source_type)
        _record_loaded_cli_dataset(metadata_store, loaded, name)
    except (FileNotFoundError, ValueError, IngestionError) as exc:
        _cli_error(exc)

    service = _build_pycaret_service(settings, metadata_store=metadata_store)
    use_gpu = _resolve_cli_use_gpu(args, settings)
    config = ExperimentConfig(
        target_column=args.target,
        task_type=ExperimentTaskType(args.task_type),
        mlflow_tracking_mode=_default_experiment_tracking_mode(settings),
        setup=ExperimentSetupConfig(
            session_id=settings.pycaret.default_session_id,
            train_size=settings.pycaret.default_train_size,
            fold=settings.pycaret.default_fold,
            fold_strategy=_default_experiment_fold_strategy(settings, args.task_type),
            use_gpu=use_gpu,
            log_experiment=_default_experiment_tracking_mode(settings) == _pycaret_native_tracking_mode(),
        ),
        evaluation=ExperimentEvaluationConfig(plots=args.plot),
    )
    try:
        bundle = service.setup_experiment(
            loaded.dataframe,
            config,
            dataset_name=name,
            dataset_fingerprint=loaded.metadata.content_hash or loaded.metadata.schema_hash,
            execution_backend=settings.execution.backend,
            workspace_mode=settings.workspace_mode,
        )
        selection = ModelSelectionSpec(model_id=args.model_id, model_name=args.model_id)
        bundle = service.evaluate_model(bundle, selection)
    except Exception as exc:
        _cli_error(exc)
    print(f"\n=== Experiment Evaluate: {name} ===")
    for plot in bundle.evaluation_plots:
        print(f"  {plot.plot_id}: {plot.path}")
    if bundle.warnings:
        print("Warnings:")
        for warning in bundle.warnings:
            print(f"  - {warning}")


def cmd_experiment_save(args: argparse.Namespace) -> None:
    """Run PyCaret setup + finalize/save from CLI."""

    from app.modeling.pycaret.schemas import (
        ExperimentConfig,
        ExperimentPersistenceConfig,
        ExperimentSetupConfig,
        ExperimentTaskType,
        ModelSelectionSpec,
    )

    settings = _load_runtime_settings()
    metadata_store = build_metadata_store(settings)
    try:
        loaded, name = _load_cli_dataset(args.dataset, source_type=args.source_type)
        _record_loaded_cli_dataset(metadata_store, loaded, name)
    except (FileNotFoundError, ValueError, IngestionError) as exc:
        _cli_error(exc)

    service = _build_pycaret_service(settings, metadata_store=metadata_store)
    use_gpu = _resolve_cli_use_gpu(args, settings)
    config = ExperimentConfig(
        target_column=args.target,
        task_type=ExperimentTaskType(args.task_type),
        mlflow_tracking_mode=_default_experiment_tracking_mode(settings),
        setup=ExperimentSetupConfig(
            session_id=settings.pycaret.default_session_id,
            train_size=settings.pycaret.default_train_size,
            fold=settings.pycaret.default_fold,
            fold_strategy=_default_experiment_fold_strategy(settings, args.task_type),
            use_gpu=use_gpu,
            log_experiment=_default_experiment_tracking_mode(settings) == _pycaret_native_tracking_mode(),
        ),
        persistence=ExperimentPersistenceConfig(save_experiment_snapshot=args.save_snapshot),
    )
    try:
        bundle = service.setup_experiment(
            loaded.dataframe,
            config,
            dataset_name=name,
            dataset_fingerprint=loaded.metadata.content_hash or loaded.metadata.schema_hash,
            execution_backend=settings.execution.backend,
            workspace_mode=settings.workspace_mode,
        )
        selection = ModelSelectionSpec(model_id=args.model_id, model_name=args.model_id)
        bundle = service.finalize_and_save_model(bundle, selection, save_name=args.save_name)
    except Exception as exc:
        _cli_error(exc)
    print(f"\n=== Experiment Save: {name} ===")
    if bundle.saved_model_metadata is not None:
        print(f"Model path: {bundle.saved_model_metadata.model_path}")
        if bundle.saved_model_metadata.experiment_snapshot_path is not None:
            print(f"Snapshot: {bundle.saved_model_metadata.experiment_snapshot_path}")


def cmd_predict_single(args: argparse.Namespace) -> None:
    """Run single-row prediction with a saved local or MLflow-backed model."""

    from app.prediction import SingleRowPredictionRequest

    settings = _load_runtime_settings()
    _validate_prediction_source_requirements(args, settings)
    service = _build_prediction_service(settings)

    try:
        row_payload = _load_prediction_row_payload(args)
        request = SingleRowPredictionRequest(
            **_build_prediction_request_kwargs(args, settings),
            row_data=row_payload,
            input_source_label="manual_row",
        )
        result = service.predict_single(request)
    except Exception as exc:
        _cli_error(exc)

    print(f"\n=== Prediction: {result.loaded_model.model_identifier} ===")
    print(f"Source: {result.loaded_model.source_type.value}  Task: {result.loaded_model.task_type.value}")
    if result.predicted_label is not None:
        print(f"Predicted label: {result.predicted_label}")
    else:
        print(f"Predicted value: {result.predicted_value}")
    if result.predicted_score is not None:
        print(f"Prediction score: {result.predicted_score}")

    if result.validation.issues:
        print("\nValidation:")
        for issue in result.validation.issues:
            print(f"  [{issue.severity.value}] {issue.message}")

    if result.artifacts is not None:
        print("\nArtifacts:")
        if result.artifacts.scored_csv_path is not None:
            print(f"  Scored CSV:   {result.artifacts.scored_csv_path}")
        if result.artifacts.summary_json_path is not None:
            print(f"  Summary JSON: {result.artifacts.summary_json_path}")
        if result.artifacts.metadata_json_path is not None:
            print(f"  Metadata:     {result.artifacts.metadata_json_path}")


def cmd_predict_batch(args: argparse.Namespace) -> None:
    """Run batch prediction from a file-backed dataset."""

    from app.prediction import BatchPredictionRequest, PredictionInputSourceType

    settings = _load_runtime_settings()
    metadata_store = build_metadata_store(settings)
    _validate_prediction_source_requirements(args, settings)
    service = _build_prediction_service(settings)

    try:
        loaded, name = _load_cli_dataset(args.dataset, source_type=args.source_type)
        _record_loaded_cli_dataset(metadata_store, loaded, name)
        request = BatchPredictionRequest(
            **_build_prediction_request_kwargs(args, settings),
            dataframe=loaded.dataframe,
            dataset_name=name,
            input_source_label=name,
            input_source_type=PredictionInputSourceType.FILE,
            output_path=Path(args.output_path) if args.output_path else None,
        )
        result = service.predict_batch(request)
    except (FileNotFoundError, ValueError, IngestionError) as exc:
        _cli_error(exc)
    except Exception as exc:
        _cli_error(exc)

    print(f"\n=== Batch Prediction: {name} ===")
    print(f"Model: {result.loaded_model.model_identifier}")
    print(f"Source: {result.loaded_model.source_type.value}  Task: {result.loaded_model.task_type.value}")
    print(f"Rows scored: {result.summary.rows_scored}/{result.summary.input_row_count}")
    if result.summary.output_artifact_path is not None:
        print(f"Output: {result.summary.output_artifact_path}")

    preview_columns = [
        column
        for column in [
            settings.prediction.prediction_column_name,
            settings.prediction.prediction_score_column_name,
        ]
        if column in result.scored_dataframe.columns
    ]
    if preview_columns:
        print("\nPreview:")
        print(result.scored_dataframe[preview_columns].head().to_string())

    if result.validation.issues:
        print("\nValidation:")
        for issue in result.validation.issues:
            print(f"  [{issue.severity.value}] {issue.message}")

    if result.artifacts is not None:
        print("\nArtifacts:")
        if result.artifacts.scored_csv_path is not None:
            print(f"  Scored CSV:   {result.artifacts.scored_csv_path}")
        if result.artifacts.summary_json_path is not None:
            print(f"  Summary JSON: {result.artifacts.summary_json_path}")
        if result.artifacts.metadata_json_path is not None:
            print(f"  Metadata:     {result.artifacts.metadata_json_path}")


def cmd_predict_history(args: argparse.Namespace) -> None:
    """Show recent local prediction jobs."""

    settings = _load_runtime_settings()
    service = _build_prediction_service(settings)

    try:
        entries = service.list_history(limit=args.limit)
    except Exception as exc:
        _cli_error(exc)

    if not entries:
        print("No prediction history found.")
        return

    print(f"\n=== Prediction History ({len(entries)}) ===")
    for entry in entries:
        output_path = entry.output_artifact_path or "N/A"
        print(
            f"  {entry.timestamp.isoformat()}  [{entry.status.value}]  {entry.mode.value}  "
            f"{entry.model_identifier}  rows={entry.row_count}  output={output_path}"
        )


def cmd_info(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Show lightweight release and local runtime metadata."""

    settings = _load_runtime_settings()

    print(f"\n=== {APP_NAME} ===")
    print(f"Version: {__version__}")
    print(f"Package: {DIST_NAME}")
    print(f"CLI entrypoint: {CLI_ENTRYPOINT}")
    print(f"Streamlit entrypoint: {STREAMLIT_ENTRYPOINT}")
    print(f"Workspace mode: {settings.workspace_mode.value}")
    print(f"Execution backend: {settings.execution.backend.value}")
    print(f"Artifacts root: {settings.artifacts.root_dir}")
    print(f"App metadata DB: {settings.database.path}")


def cmd_uci_list(args: argparse.Namespace) -> None:
    """List searchable UCI ML Repository datasets."""

    try:
        rows = list_available_uci_datasets(
            search=args.search,
            area=args.area,
            filter=args.filter,
        )
    except Exception as exc:
        _cli_error(exc)

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    if not rows:
        print("No UCI datasets found.")
        return

    print("\n=== UCI Datasets ===")
    for row in rows:
        print(f"{row['uci_id']}\t{row['name']}")


def cmd_init_local_storage(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Initialize local artifact directories and the app metadata database."""

    settings = _load_runtime_settings()

    # Ensure a sensible default MLflow tracking URI when none is configured.
    _DEFAULT_MLFLOW_URI = "sqlite:///artifacts/mlflow/mlflow.db"
    mlflow_uri_was_missing = not settings.mlflow.tracking_uri
    if mlflow_uri_was_missing:
        settings.mlflow.tracking_uri = _DEFAULT_MLFLOW_URI
        settings.mlflow.registry_uri = _DEFAULT_MLFLOW_URI
        try:
            save_settings(settings)
        except Exception as exc:
            print(f"  [warning] Could not persist default MLflow URI: {exc}")

    # Create the MLflow directory alongside the other artifact dirs.
    mlflow_dir = settings.artifacts.root_dir / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    status = initialize_local_runtime(settings, include_optional_network_checks=False)

    print("\n=== Local Storage Initialized ===")
    if status.database_path is not None:
        print(f"Database: {status.database_path}")
    if settings.mlflow.tracking_uri:
        print(f"MLflow tracking URI: {settings.mlflow.tracking_uri}")
    print("Artifacts:")
    for path in status.artifact_dirs:
        print(f"  {path}")
    for line in format_startup_issues(status):
        print(f"  {line}")
    if status.errors:
        sys.exit(1)


def cmd_doctor(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Run local startup diagnostics with conservative optional checks."""

    settings = _load_runtime_settings()
    status = initialize_local_runtime(settings, include_optional_network_checks=True)

    print("\n=== AutoTabML Doctor ===")
    gpu_info = cuda_summary()
    print(f"CUDA available: {gpu_info['cuda_available']}")
    if gpu_info['device_name']:
        print(f"CUDA device: {gpu_info['device_name']} (count: {gpu_info['device_count']})")
    if status.database_path is not None:
        print(f"Database: {status.database_path}")
    print(f"Artifact directories checked: {len(status.artifact_dirs)}")
    print(f"Stale temp files removed: {status.temp_files_removed}")
    print(f"Stale partial artifacts removed: {status.partial_files_removed}")
    if status.issues:
        print("Checks:")
        for line in format_startup_issues(status):
            print(f"  {line}")
    else:
        print("Checks: [ok] no startup issues detected")
    if status.errors:
        sys.exit(1)


def _resolve_cli_use_gpu(args: argparse.Namespace, settings) -> bool | str:  # noqa: ANN001
    """Resolve the CLI --use-gpu argument against settings defaults."""

    raw = getattr(args, "use_gpu", None)
    if raw is None:
        return settings.pycaret.default_use_gpu
    if raw == "force":
        return "force"
    return raw == "true"


# ---------------------------------------------------------------------------
# History, compare, and registry CLI commands
# ---------------------------------------------------------------------------


def cmd_history_list(args: argparse.Namespace) -> None:
    """List MLflow runs from the history center."""

    from app.tracking.filters import RunHistoryFilter, RunHistorySort, RunSortField, SortDirection
    from app.tracking.history_service import HistoryService
    from app.tracking.mlflow_query import is_mlflow_available
    from app.tracking.schemas import RunType
    from app.tracking.summary import run_summary_line

    if not is_mlflow_available():
        print("Error: mlflow is not installed.", file=sys.stderr)
        sys.exit(1)

    settings = _load_runtime_settings()
    service = HistoryService(
        tracking_uri=settings.tracking.tracking_uri,
        default_experiment_names=settings.tracking.default_experiment_names,
        default_limit=args.limit,
    )

    history_filter = RunHistoryFilter(
        run_type=RunType(args.run_type) if args.run_type != "all" else None,
        task_type=args.task_type or None,
    )
    sort = RunHistorySort(
        field=RunSortField(args.sort_by),
        direction=SortDirection(args.sort_dir),
    )

    try:
        runs = service.list_runs(history_filter=history_filter, sort=sort, limit=args.limit)
    except Exception as exc:
        _cli_error(exc)

    if not runs:
        print("No runs found.")
        return

    print(f"\n=== Run History ({len(runs)} runs) ===")
    for run in runs:
        print(f"  {run.run_id}  {run_summary_line(run)}")


def cmd_history_show(args: argparse.Namespace) -> None:
    """Show details for a single MLflow run."""

    from app.tracking.history_service import HistoryService
    from app.tracking.mlflow_query import is_mlflow_available

    if not is_mlflow_available():
        print("Error: mlflow is not installed.", file=sys.stderr)
        sys.exit(1)

    settings = _load_runtime_settings()
    service = HistoryService(
        tracking_uri=settings.tracking.tracking_uri,
        default_experiment_names=settings.tracking.default_experiment_names,
    )

    try:
        resolved_id = service.resolve_run_id(args.run_id)
        detail = service.get_run_detail(resolved_id)
    except Exception as exc:
        _cli_error(exc)

    print(f"\n=== Run Detail: {detail.run_id} ===")
    print(f"Name: {detail.run_name or 'N/A'}")
    print(f"Type: {detail.run_type.value}  Task: {detail.task_type or 'N/A'}")
    print(f"Status: {detail.status.value}")
    print(f"Duration: {detail.duration_seconds:.1f}s" if detail.duration_seconds else "Duration: N/A")
    if detail.model_name:
        print(f"Model: {detail.model_name}")
    if detail.primary_metric_name and detail.primary_metric_value is not None:
        print(f"Primary metric: {detail.primary_metric_name} = {detail.primary_metric_value}")

    if detail.params:
        print("\nParameters:")
        for key, value in sorted(detail.params.items()):
            print(f"  {key} = {value}")

    if detail.metrics:
        print("\nMetrics:")
        for key, value in sorted(detail.metrics.items()):
            print(f"  {key} = {value}")

    if detail.artifact_paths:
        print("\nArtifacts:")
        for path in detail.artifact_paths:
            print(f"  {path}")


def cmd_compare_runs(args: argparse.Namespace) -> None:
    """Compare two MLflow runs side-by-side."""

    from app.tracking.compare_service import ComparisonService
    from app.tracking.history_service import HistoryService
    from app.tracking.mlflow_query import is_mlflow_available

    if not is_mlflow_available():
        print("Error: mlflow is not installed.", file=sys.stderr)
        sys.exit(1)

    settings = _load_runtime_settings()
    history = HistoryService(
        tracking_uri=settings.tracking.tracking_uri,
        default_experiment_names=settings.tracking.default_experiment_names,
    )

    try:
        left_id = history.resolve_run_id(args.left_run_id)
        right_id = history.resolve_run_id(args.right_run_id)
        left = history.get_run_detail(left_id)
        right = history.get_run_detail(right_id)
    except Exception as exc:
        _cli_error(exc)

    comparison = ComparisonService()
    bundle = comparison.compare(left, right)

    print(f"\n=== Comparison: {left.run_id[:12]} vs {right.run_id[:12]} ===")
    print(f"Comparable: {'Yes' if bundle.comparable else 'No'}")

    if bundle.warnings:
        print("\nWarnings:")
        for warning in bundle.warnings:
            print(f"  - {warning}")

    if bundle.metric_deltas:
        print("\nMetric Deltas:")
        for delta in bundle.metric_deltas:
            left_str = f"{delta.left_value:.4f}" if delta.left_value is not None else "N/A"
            right_str = f"{delta.right_value:.4f}" if delta.right_value is not None else "N/A"
            delta_str = f"{delta.delta:+.4f}" if delta.delta is not None else ""
            better = f" ({delta.better_side})" if delta.better_side else ""
            print(f"  {delta.name}: {left_str} -> {right_str}  {delta_str}{better}")

    if bundle.config_differences:
        print("\nConfig Differences:")
        for diff in bundle.config_differences:
            print(f"  {diff.key}: {diff.left_value or 'N/A'} -> {diff.right_value or 'N/A'}")

    if args.artifacts_dir:
        from app.tracking.artifacts import write_comparison_artifacts

        paths = write_comparison_artifacts(bundle, Path(args.artifacts_dir))
        print("\nArtifacts:")
        for label, path in paths.items():
            print(f"  {label}: {path}")


def cmd_registry_list(args: argparse.Namespace) -> None:
    """List registered models."""

    from app.tracking.mlflow_query import is_mlflow_available

    if not is_mlflow_available():
        print("Error: mlflow is not installed.", file=sys.stderr)
        sys.exit(1)

    settings = _load_runtime_settings()
    if not settings.tracking.registry_enabled:
        print("Error: model registry is disabled in settings.", file=sys.stderr)
        sys.exit(1)

    from app.registry.registry_service import RegistryService

    service = RegistryService(
        tracking_uri=settings.tracking.tracking_uri,
        registry_uri=settings.tracking.registry_uri,
        champion_alias=settings.tracking.champion_alias,
        candidate_alias=settings.tracking.candidate_alias,
    )

    try:
        models = service.list_models()
    except Exception as exc:
        _cli_error(exc)

    if not models:
        print("No registered models found.")
        return

    print(f"\n=== Registered Models ({len(models)}) ===")
    for model in models:
        aliases = ", ".join(f"{k}->v{v}" for k, v in model.aliases.items()) if model.aliases else "none"
        print(f"  {model.name}  versions={model.version_count}  aliases=[{aliases}]")


def cmd_registry_show(args: argparse.Namespace) -> None:
    """Show versions for a registered model."""

    from app.tracking.mlflow_query import is_mlflow_available

    if not is_mlflow_available():
        print("Error: mlflow is not installed.", file=sys.stderr)
        sys.exit(1)

    settings = _load_runtime_settings()
    if not settings.tracking.registry_enabled:
        print("Error: model registry is disabled in settings.", file=sys.stderr)
        sys.exit(1)

    from app.registry.registry_service import RegistryService

    service = RegistryService(
        tracking_uri=settings.tracking.tracking_uri,
        registry_uri=settings.tracking.registry_uri,
    )

    try:
        versions = service.list_versions(args.model_name)
    except Exception as exc:
        _cli_error(exc)

    if not versions:
        print(f"No versions found for model '{args.model_name}'.")
        return

    print(f"\n=== Versions of '{args.model_name}' ({len(versions)}) ===")
    for version in versions:
        aliases = ", ".join(version.aliases) if version.aliases else ""
        print(
            f"  v{version.version}  status={version.status}  "
            f"run_id={version.run_id or 'N/A'}  "
            f"app_status={version.app_status or 'N/A'}  "
            f"aliases=[{aliases}]"
        )


def cmd_registry_register(args: argparse.Namespace) -> None:
    """Register a model from an artifact source."""

    from app.tracking.mlflow_query import is_mlflow_available

    if not is_mlflow_available():
        print("Error: mlflow is not installed.", file=sys.stderr)
        sys.exit(1)

    settings = _load_runtime_settings()
    if not settings.tracking.registry_enabled:
        print("Error: model registry is disabled in settings.", file=sys.stderr)
        sys.exit(1)

    from app.registry.registry_service import RegistryService

    service = RegistryService(
        tracking_uri=settings.tracking.tracking_uri,
        registry_uri=settings.tracking.registry_uri,
    )

    try:
        version = service.register_model(
            args.model_name,
            source=args.source,
            run_id=args.run_id,
            description=args.description or "",
        )
    except Exception as exc:
        _cli_error(exc)

    print(f"Registered model '{version.model_name}' version {version.version}.")


def cmd_registry_promote(args: argparse.Namespace) -> None:
    """Promote a model version."""

    from app.tracking.mlflow_query import is_mlflow_available

    if not is_mlflow_available():
        print("Error: mlflow is not installed.", file=sys.stderr)
        sys.exit(1)

    settings = _load_runtime_settings()
    if not settings.tracking.registry_enabled:
        print("Error: model registry is disabled in settings.", file=sys.stderr)
        sys.exit(1)

    from app.registry.registry_service import RegistryService
    from app.registry.schemas import PromotionAction, PromotionRequest

    service = RegistryService(
        tracking_uri=settings.tracking.tracking_uri,
        registry_uri=settings.tracking.registry_uri,
        champion_alias=settings.tracking.champion_alias,
        candidate_alias=settings.tracking.candidate_alias,
        archived_tag_key=settings.tracking.archived_tag_key,
    )

    request = PromotionRequest(
        model_name=args.model_name,
        version=args.version,
        action=PromotionAction(args.action),
    )

    try:
        result = service.promote(request)
    except Exception as exc:
        _cli_error(exc)

    print(f"Promoted '{result.model_name}' v{result.version} -> {result.action.value}.")
    for change in result.alias_changes + result.tag_changes:
        print(f"  {change}")
    for warning in result.warnings:
        print(f"  Warning: {warning}")


def cmd_batch_history(args: argparse.Namespace) -> None:
    """List batch run history from the local database."""

    settings = _load_runtime_settings()
    metadata_store = build_metadata_store(settings)
    if metadata_store is None:
        print("Error: local metadata store is not configured.", file=sys.stderr)
        sys.exit(1)

    batches = metadata_store.list_batch_runs(limit=args.limit)
    if not batches:
        print("No batch runs found.")
        return

    print(f"\n=== Batch Run History ({len(batches)} runs) ===")
    for b in batches:
        success_count, failed_count, skipped_count = _resolve_batch_status_counts(metadata_store, b)
        print(
            f"  {b.batch_id}  [{b.status.value}]  "
            f"total={b.total_datasets}  success={success_count}  "
            f"failed={failed_count}  skipped={skipped_count}  "
            f"started={b.started_at.isoformat()}"
        )


def cmd_batch_show(args: argparse.Namespace) -> None:
    """Show details for a single batch run."""

    settings = _load_runtime_settings()
    metadata_store = build_metadata_store(settings)
    if metadata_store is None:
        print("Error: local metadata store is not configured.", file=sys.stderr)
        sys.exit(1)

    batch = metadata_store.get_batch_run(args.batch_id)
    if batch is None:
        print(f"Batch run '{args.batch_id}' not found.")
        sys.exit(1)

    print(f"\n=== Batch Run: {batch.batch_id} ===")
    print(f"Name: {batch.batch_name}")
    print(f"Status: {batch.status.value}")
    success_count, failed_count, skipped_count = _resolve_batch_status_counts(metadata_store, batch)
    print(f"Total: {batch.total_datasets}  Success: {success_count}  "
          f"Failed: {failed_count}  Skipped: {skipped_count}")
    print(f"Started: {batch.started_at.isoformat()}")
    print(f"Updated: {batch.updated_at.isoformat()}")

    items = metadata_store.list_batch_items(args.batch_id, limit=max(batch.total_datasets, 200))
    if not items:
        return

    print(f"\nDatasets ({len(items)}):")
    for item in items:
        score_str = f"{item.best_score:.4f}" if item.best_score is not None else "N/A"
        dur_str = f"{item.duration_seconds:.1f}s" if item.duration_seconds is not None else "N/A"
        err = f"  err={item.error_message[:60]}" if item.error_message else ""
        print(
            f"  uci:{item.uci_id:<4d} {item.dataset_name:<45s} [{item.status.value:<7s}]  "
            f"best={item.best_model or 'N/A':<30s}  score={score_str}  "
            f"metric={item.ranking_metric or 'N/A'}  dur={dur_str}{err}"
        )


def _resolve_batch_status_counts(metadata_store, batch) -> tuple[int, int, int]:  # noqa: ANN001
    """Return batch status counts derived from item records when available."""

    items = metadata_store.list_batch_items(batch.batch_id, limit=max(batch.total_datasets, 200))
    if not items:
        return batch.completed_count, batch.failed_count, batch.skipped_count

    success_count = sum(1 for item in items if item.status == BatchItemStatus.SUCCESS)
    failed_count = sum(1 for item in items if item.status == BatchItemStatus.FAILED)
    skipped_count = sum(1 for item in items if item.status == BatchItemStatus.SKIPPED)
    return success_count, failed_count, skipped_count


def _load_prediction_row_payload(args: argparse.Namespace) -> dict:
    """Load one JSON row payload from CLI args."""

    if args.row_json:
        payload = json.loads(args.row_json)
    elif args.row_file:
        payload = json.loads(Path(args.row_file).read_text(encoding="utf-8"))
    else:
        raise ValueError("Provide either --row-json or --row-file for predict-single.")
    if not isinstance(payload, dict):
        raise ValueError("Prediction row payload must be a JSON object.")
    return payload


def _validate_prediction_source_requirements(args: argparse.Namespace, settings) -> None:  # noqa: ANN001
    from app.prediction import ModelSourceType
    from app.tracking.mlflow_query import is_mlflow_available

    source_type = ModelSourceType(args.model_source)
    if source_type != ModelSourceType.LOCAL_SAVED_MODEL and not is_mlflow_available():
        raise ValueError("mlflow is not installed, so MLflow-backed prediction is unavailable.")
    if source_type == ModelSourceType.MLFLOW_REGISTERED_MODEL and not settings.tracking.registry_enabled:
        raise ValueError("Model registry is disabled in settings.")


def _build_prediction_service(settings, *, metadata_store=None):  # noqa: ANN001, ANN201
    from app.prediction import PredictionService, SchemaValidationMode

    effective_metadata_store = metadata_store if metadata_store is not None else build_metadata_store(settings)

    return PredictionService(
        artifacts_dir=settings.prediction.artifacts_dir,
        history_path=settings.prediction.history_path,
        schema_validation_mode=SchemaValidationMode(settings.prediction.schema_validation_mode),
        prediction_column_name=settings.prediction.prediction_column_name,
        prediction_score_column_name=settings.prediction.prediction_score_column_name,
        local_model_dirs=settings.prediction.supported_local_model_dirs,
        local_metadata_dirs=settings.prediction.local_model_metadata_dirs,
        tracking_uri=settings.tracking.tracking_uri,
        registry_uri=settings.tracking.registry_uri,
        registry_enabled=settings.tracking.registry_enabled,
        metadata_store=effective_metadata_store,
    )


def _build_prediction_request_kwargs(args: argparse.Namespace, settings) -> dict:
    """Build base prediction request kwargs from CLI args."""

    from app.prediction import ModelSourceType, PredictionTaskType, SchemaValidationMode

    return {
        "source_type": ModelSourceType(args.model_source),
        "model_identifier": args.model_id or args.model_path or args.model_uri or args.model_name,
        "model_path": Path(args.model_path) if args.model_path else None,
        "model_uri": args.model_uri or None,
        "metadata_path": Path(args.metadata_path) if args.metadata_path else None,
        "task_type_hint": PredictionTaskType(args.task_type) if args.task_type != "unknown" else None,
        "schema_validation_mode": SchemaValidationMode(args.schema_mode) if args.schema_mode else None,
        "tracking_uri": settings.tracking.tracking_uri,
        "registry_uri": settings.tracking.registry_uri,
        "run_id": args.run_id or None,
        "artifact_path": args.artifact_path or None,
        "registry_model_name": args.model_name or None,
        "registry_version": args.model_version or None,
        "registry_alias": args.model_alias or None,
        "output_dir": Path(args.output_dir) if args.output_dir else settings.prediction.artifacts_dir,
        "output_stem": args.output_stem or settings.prediction.default_output_stem,
    }


def _add_prediction_model_source_args(parser: argparse.ArgumentParser) -> None:
    """Add common model-source selection flags to a prediction parser."""

    from app.prediction import ModelSourceType

    parser.add_argument(
        "--model-source",
        choices=[source.value for source in ModelSourceType],
        required=True,
        help="Prediction model source type",
    )
    parser.add_argument("--model-id", default=None, help="Local saved-model identifier or discovered model name")
    parser.add_argument("--model-path", default=None, help="Explicit local saved-model path")
    parser.add_argument("--model-uri", default=None, help="Explicit MLflow model URI")
    parser.add_argument("--metadata-path", default=None, help="Optional saved-model metadata JSON path")
    parser.add_argument(
        "--task-type",
        choices=["unknown", "classification", "regression"],
        default="unknown",
        help="Optional task-type hint when metadata is incomplete",
    )
    parser.add_argument(
        "--schema-mode",
        choices=["strict", "warn"],
        default=None,
        help="Override the configured schema validation mode",
    )
    parser.add_argument("--run-id", default=None, help="MLflow run id for runs:/ model loading")
    parser.add_argument("--artifact-path", default=None, help="Artifact path under an MLflow run")
    parser.add_argument("--model-name", default=None, help="Registered model name")
    parser.add_argument("--model-version", default=None, help="Registered model version")
    parser.add_argument("--model-alias", default=None, help="Registered model alias")
    parser.add_argument("--output-dir", default=None, help="Directory for prediction artifacts")
    parser.add_argument("--output-stem", default=None, help="Optional output stem for generated artifacts")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        prog="autotabml",
        description=f"{APP_NAME} CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser("info", help="Show app version and local runtime defaults")
    info_parser.set_defaults()

    uci_list_parser = subparsers.add_parser("uci-list", help="Search the UCI ML Repository catalog")
    uci_list_parser.add_argument("--search", default=None, help="Search query for dataset names")
    uci_list_parser.add_argument("--area", default=None, help="Optional UCI subject-area filter")
    uci_list_parser.add_argument("--filter", default=None, help="Optional ucimlrepo filter value")
    uci_list_parser.add_argument("--limit", type=int, default=25, help="Maximum results to print")

    # validate
    val_parser = subparsers.add_parser("validate", help="Run data validation")
    val_parser.add_argument("dataset", help="Dataset path or supported URL")
    val_parser.add_argument("--target", default=None, help="Target column name")
    val_parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in IngestionSourceType],
        default=None,
        help="Optional explicit ingestion source type",
    )
    val_parser.add_argument("--min-rows", type=int, default=None, help="Minimum row count")
    val_parser.add_argument("--artifacts-dir", default=None, help="Directory for artifacts")

    # profile
    prof_parser = subparsers.add_parser("profile", help="Run data profiling")
    prof_parser.add_argument("dataset", help="Dataset path or supported URL")
    prof_parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in IngestionSourceType],
        default=None,
        help="Optional explicit ingestion source type",
    )
    prof_parser.add_argument("--artifacts-dir", default=None, help="Directory for artifacts")

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run baseline model benchmarking")
    bench_parser.add_argument("dataset", help="Dataset path or supported URL")
    bench_parser.add_argument("--target", required=True, help="Target column name")
    bench_parser.add_argument(
        "--task-type",
        choices=["auto", "classification", "regression"],
        default="auto",
        help="Benchmark task type",
    )
    bench_parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in IngestionSourceType],
        default=None,
        help="Optional explicit ingestion source type",
    )
    bench_parser.add_argument("--test-size", type=float, default=None, help="Test split size")
    bench_parser.add_argument("--random-state", type=int, default=None, help="Random seed")
    bench_parser.add_argument(
        "--stratify",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to stratify the train/test split when applicable",
    )
    bench_parser.add_argument("--ranking-metric", default=None, help="Optional ranking metric")
    bench_parser.add_argument("--sample-rows", type=int, default=0, help="Optional sampled row count")
    bench_parser.add_argument("--top-k", type=int, default=None, help="Top-k shortlist size")
    bench_parser.add_argument(
        "--prefer-gpu",
        choices=["auto", "true", "false"],
        default="auto",
        help="Prefer GPU acceleration when supported; defaults to benchmark settings",
    )
    bench_parser.add_argument(
        "--include-model",
        action="append",
        default=[],
        help="Model name to include; repeat to include multiple models",
    )
    bench_parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Model name to exclude; repeat to exclude multiple models",
    )
    bench_parser.add_argument("--artifacts-dir", default=None, help="Directory for artifacts")

    run_parser = subparsers.add_parser("experiment-run", help="Run PyCaret setup + compare_models")
    run_parser.add_argument("dataset", help="Dataset path or supported URL")
    run_parser.add_argument("--target", required=True, help="Target column name")
    run_parser.add_argument(
        "--task-type",
        choices=["auto", "classification", "regression"],
        default="auto",
        help="Experiment task type",
    )
    run_parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in IngestionSourceType],
        default=None,
        help="Optional explicit ingestion source type",
    )
    run_parser.add_argument("--train-size", type=float, default=None, help="PyCaret train_size")
    run_parser.add_argument("--fold", type=int, default=None, help="Cross-validation folds")
    run_parser.add_argument("--fold-strategy", default=None, help="Optional fold strategy override")
    run_parser.add_argument(
        "--preprocess",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to let PyCaret preprocess the dataset",
    )
    run_parser.add_argument(
        "--ignore-feature",
        action="append",
        default=[],
        help="Feature to ignore during setup; repeat to include multiple",
    )
    run_parser.add_argument("--compare-metric", default=None, help="compare_models sort metric")
    run_parser.add_argument("--n-select", type=int, default=1, help="Top N models returned by compare_models")
    run_parser.add_argument("--budget-time", type=float, default=None, help="Optional compare_models budget time in minutes")
    run_parser.add_argument("--no-turbo", dest="turbo", action="store_false", help="Disable PyCaret turbo compare mode")
    run_parser.add_argument("--use-gpu", choices=["false", "true", "force"], default=None, help="GPU acceleration (false/true/force); defaults to settings")
    run_parser.add_argument("--artifacts-dir", default=None, help="Directory for artifacts")
    run_parser.set_defaults(turbo=True)

    tune_parser = subparsers.add_parser("experiment-tune", help="Run PyCaret tune_model for one model id")
    tune_parser.add_argument("dataset", help="Dataset path or supported URL")
    tune_parser.add_argument("--target", required=True, help="Target column name")
    tune_parser.add_argument("--model-id", required=True, help="PyCaret model id to create and tune")
    tune_parser.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        required=True,
        help="Experiment task type",
    )
    tune_parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in IngestionSourceType],
        default=None,
        help="Optional explicit ingestion source type",
    )
    tune_parser.add_argument("--tune-metric", default=None, help="tune_model optimize metric")
    tune_parser.add_argument("--n-iter", type=int, default=10, help="Number of tuning iterations")
    tune_parser.add_argument("--use-gpu", choices=["false", "true", "force"], default=None, help="GPU acceleration (false/true/force); defaults to settings")

    eval_parser = subparsers.add_parser("experiment-evaluate", help="Generate evaluation plots for one model id")
    eval_parser.add_argument("dataset", help="Dataset path or supported URL")
    eval_parser.add_argument("--target", required=True, help="Target column name")
    eval_parser.add_argument("--model-id", required=True, help="PyCaret model id to evaluate")
    eval_parser.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        required=True,
        help="Experiment task type",
    )
    eval_parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in IngestionSourceType],
        default=None,
        help="Optional explicit ingestion source type",
    )
    eval_parser.add_argument(
        "--plot",
        action="append",
        default=[],
        help="Evaluation plot id to generate; repeat to include multiple",
    )
    eval_parser.add_argument("--use-gpu", choices=["false", "true", "force"], default=None, help="GPU acceleration (false/true/force); defaults to settings")

    save_parser = subparsers.add_parser("experiment-save", help="Finalize and save one model id")
    save_parser.add_argument("dataset", help="Dataset path or supported URL")
    save_parser.add_argument("--target", required=True, help="Target column name")
    save_parser.add_argument("--model-id", required=True, help="PyCaret model id to finalize and save")
    save_parser.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        required=True,
        help="Experiment task type",
    )
    save_parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in IngestionSourceType],
        default=None,
        help="Optional explicit ingestion source type",
    )
    save_parser.add_argument("--save-name", default="selected_model", help="Base name for saved model artifacts")
    save_parser.add_argument("--save-snapshot", action="store_true", help="Also save a PyCaret experiment snapshot")
    save_parser.add_argument("--use-gpu", choices=["false", "true", "force"], default=None, help="GPU acceleration (false/true/force); defaults to settings")

    predict_single_parser = subparsers.add_parser("predict-single", help="Run single-row prediction")
    _add_prediction_model_source_args(predict_single_parser)
    predict_single_parser.add_argument("--row-json", default=None, help="Single-row JSON object payload")
    predict_single_parser.add_argument("--row-file", default=None, help="Path to a JSON file containing one row object")

    predict_batch_parser = subparsers.add_parser("predict-batch", help="Run batch prediction from a dataset")
    predict_batch_parser.add_argument("dataset", help="Dataset path or supported URL")
    predict_batch_parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in IngestionSourceType],
        default=None,
        help="Optional explicit ingestion source type",
    )
    _add_prediction_model_source_args(predict_batch_parser)
    predict_batch_parser.add_argument("--output-path", default=None, help="Optional explicit scored CSV output path")

    predict_history_parser = subparsers.add_parser("predict-history", help="Show recent prediction jobs")
    predict_history_parser.add_argument("--limit", type=int, default=20, help="Maximum number of history items")

    init_local_storage_parser = subparsers.add_parser(
        "init-local-storage",
        help="Initialize local artifact directories and app metadata storage",
    )
    init_local_storage_parser.set_defaults()

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run local startup diagnostics and conservative environment checks",
    )
    doctor_parser.set_defaults()

    # history-list
    hist_list_parser = subparsers.add_parser("history-list", help="List MLflow runs")
    hist_list_parser.add_argument(
        "--run-type",
        choices=["all", "benchmark", "experiment", "unknown"],
        default="all",
        help="Filter by run type",
    )
    hist_list_parser.add_argument("--task-type", default=None, help="Filter by task type")
    hist_list_parser.add_argument(
        "--sort-by",
        choices=["start_time", "duration", "model_name", "primary_score"],
        default="start_time",
        help="Sort field",
    )
    hist_list_parser.add_argument(
        "--sort-dir",
        choices=["ascending", "descending"],
        default="descending",
        help="Sort direction",
    )
    hist_list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of runs")

    # history-show
    hist_show_parser = subparsers.add_parser("history-show", help="Show details for a single run")
    hist_show_parser.add_argument("run_id", help="MLflow run id")

    # compare-runs
    cmp_parser = subparsers.add_parser("compare-runs", help="Compare two MLflow runs")
    cmp_parser.add_argument("left_run_id", help="Left run id")
    cmp_parser.add_argument("right_run_id", help="Right run id")
    cmp_parser.add_argument("--artifacts-dir", default=None, help="Directory for comparison artifacts")

    # registry-list
    reg_list_parser = subparsers.add_parser("registry-list", help="List registered models")

    # registry-show
    reg_show_parser = subparsers.add_parser("registry-show", help="Show versions for a model")
    reg_show_parser.add_argument("model_name", help="Registered model name")

    # registry-register
    reg_register_parser = subparsers.add_parser("registry-register", help="Register a model")
    reg_register_parser.add_argument("model_name", help="Model name to register")
    reg_register_parser.add_argument("--source", required=True, help="Artifact source path or URI")
    reg_register_parser.add_argument("--run-id", default=None, help="MLflow run id")
    reg_register_parser.add_argument("--description", default=None, help="Model description")

    # registry-promote
    reg_promote_parser = subparsers.add_parser("registry-promote", help="Promote a model version")
    reg_promote_parser.add_argument("model_name", help="Registered model name")
    reg_promote_parser.add_argument("version", help="Model version to promote")
    reg_promote_parser.add_argument(
        "--action",
        choices=["champion", "candidate", "archived"],
        required=True,
        help="Promotion action",
    )

    # batch-history
    batch_hist_parser = subparsers.add_parser("batch-history", help="List batch run history")
    batch_hist_parser.add_argument("--limit", type=int, default=20, help="Maximum number of batch runs")

    # batch-show
    batch_show_parser = subparsers.add_parser("batch-show", help="Show details for a batch run")
    batch_show_parser.add_argument("batch_id", help="Batch run ID")

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "uci-list":
        cmd_uci_list(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "profile":
        cmd_profile(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "experiment-run":
        cmd_experiment_run(args)
    elif args.command == "experiment-tune":
        cmd_experiment_tune(args)
    elif args.command == "experiment-evaluate":
        cmd_experiment_evaluate(args)
    elif args.command == "experiment-save":
        cmd_experiment_save(args)
    elif args.command == "predict-single":
        cmd_predict_single(args)
    elif args.command == "predict-batch":
        cmd_predict_batch(args)
    elif args.command == "predict-history":
        cmd_predict_history(args)
    elif args.command == "init-local-storage":
        cmd_init_local_storage(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "history-list":
        cmd_history_list(args)
    elif args.command == "history-show":
        cmd_history_show(args)
    elif args.command == "compare-runs":
        cmd_compare_runs(args)
    elif args.command == "registry-list":
        cmd_registry_list(args)
    elif args.command == "registry-show":
        cmd_registry_show(args)
    elif args.command == "registry-register":
        cmd_registry_register(args)
    elif args.command == "registry-promote":
        cmd_registry_promote(args)
    elif args.command == "batch-history":
        cmd_batch_history(args)
    elif args.command == "batch-show":
        cmd_batch_show(args)
    else:
        parser.print_help()
        sys.exit(1)


def _build_pycaret_service(settings, *, artifacts_dir: Path | None = None, metadata_store=None) -> PyCaretExperimentService:
    tracking_settings = getattr(settings, "tracking", None)
    return PyCaretExperimentService(
        artifacts_dir=artifacts_dir if artifacts_dir is not None else settings.pycaret.artifacts_dir,
        models_dir=settings.pycaret.models_dir,
        snapshots_dir=settings.pycaret.snapshots_dir,
        classification_compare_metric=settings.pycaret.default_compare_metric_classification,
        regression_compare_metric=settings.pycaret.default_compare_metric_regression,
        classification_tune_metric=settings.pycaret.default_tune_metric_classification,
        regression_tune_metric=settings.pycaret.default_tune_metric_regression,
        mlflow_experiment_name=settings.pycaret.mlflow_experiment_name,
        tracking_uri=getattr(tracking_settings, "tracking_uri", None),
        registry_uri=getattr(tracking_settings, "registry_uri", None),
        metadata_store=metadata_store,
    )


def _default_experiment_fold_strategy(settings, task_type: str) -> str | None:
    if task_type == "classification":
        return settings.pycaret.default_classification_fold_strategy
    if task_type == "regression":
        return settings.pycaret.default_regression_fold_strategy
    return None


def _pycaret_native_tracking_mode():
    from app.modeling.pycaret.schemas import MLflowTrackingMode

    return MLflowTrackingMode.PYCARET_NATIVE


def _default_experiment_tracking_mode(settings):
    from app.modeling.pycaret.schemas import MLflowTrackingMode

    try:
        return MLflowTrackingMode(settings.pycaret.default_tracking_mode)
    except ValueError:
        return MLflowTrackingMode.MANUAL


if __name__ == "__main__":
    main()
