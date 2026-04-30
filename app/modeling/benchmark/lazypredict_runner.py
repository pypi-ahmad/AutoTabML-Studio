"""LazyPredict-backed benchmark service."""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.errors import log_and_wrap
from app.gpu import is_cuda_available
from app.modeling.benchmark.artifacts import write_benchmark_artifacts
from app.modeling.benchmark.base import BaseBenchmarkService
from app.modeling.benchmark.errors import (
    BenchmarkConfigurationError,
    BenchmarkDependencyError,
    BenchmarkExecutionError,
    BenchmarkTargetError,
)
from app.modeling.benchmark.mlflow_tracking import MLflowBenchmarkTracker
from app.modeling.benchmark.ranker import rank_result_rows, resolve_ranking_metric
from app.modeling.benchmark.schemas import (
    BenchmarkConfig,
    BenchmarkResultBundle,
    BenchmarkTaskType,
)
from app.modeling.benchmark.selectors import (
    benchmark_reliability_warnings,
    choose_stratify_target,
    collect_nested_object_columns,
    resolve_task_type,
)
from app.modeling.benchmark.summary import build_benchmark_summary, build_result_rows

logger = logging.getLogger(__name__)


def is_lazypredict_available() -> bool:
    """Return True if lazypredict is importable."""

    try:
        import lazypredict.Supervised  # noqa: F401
        return True
    except ImportError:
        return False


def _lazypredict_gpu_usable() -> bool:
    """Return True only when LazyPredict can actually utilise the GPU.

    LazyPredict delegates GPU checks to ``torch.cuda.is_available()``.
    Our own ``is_cuda_available()`` also considers a ctypes driver probe,
    which may succeed even when ``torch`` is absent.  To avoid passing
    ``use_gpu=True`` only to have LazyPredict emit 'GPU requested but
    CUDA is not available' warnings, we require **both** our check and
    a positive ``torch.cuda`` result.
    """
    if not is_cuda_available():
        return False
    try:
        import torch  # noqa: F811
        return torch.cuda.is_available()
    except (ImportError, AttributeError, RuntimeError):
        return False


def _get_lazypredict_module() -> Any:
    return importlib.import_module("lazypredict.Supervised")


def _train_test_split(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
    from sklearn.model_selection import train_test_split

    return train_test_split(*args, **kwargs)


class LazyPredictBenchmarkService(BaseBenchmarkService):
    """Benchmark service backed by LazyClassifier/LazyRegressor."""

    def __init__(
        self,
        *,
        artifacts_dir: Path | None = None,
        classification_default_metric: str = "Balanced Accuracy",
        regression_default_metric: str = "Adjusted R-Squared",
        sampling_row_threshold: int = 100_000,
        suggested_sample_rows: int = 50_000,
        mlflow_experiment_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        super().__init__(
            artifacts_dir=artifacts_dir,
            mlflow_experiment_name=mlflow_experiment_name,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        self._classification_default_metric = classification_default_metric
        self._regression_default_metric = regression_default_metric
        self._sampling_row_threshold = sampling_row_threshold
        self._suggested_sample_rows = suggested_sample_rows

    def run(
        self,
        df: pd.DataFrame,
        config: BenchmarkConfig,
        *,
        dataset_name: str | None = None,
        dataset_fingerprint: str | None = None,
        execution_backend: ExecutionBackend = ExecutionBackend.LOCAL,
        workspace_mode: WorkspaceMode | None = None,
    ) -> BenchmarkResultBundle:
        if execution_backend != ExecutionBackend.LOCAL:
            raise BenchmarkExecutionError(
                f"Benchmark execution for backend '{execution_backend.value}' is not implemented yet. "
                "TODO: add remote benchmark execution scaffolding for this backend."
            )

        if not is_lazypredict_available():
            raise BenchmarkDependencyError(
                "lazypredict is not installed. Install it with: pip install lazypredict"
            )

        if config.target_column not in df.columns:
            raise BenchmarkTargetError(
                f"Target column '{config.target_column}' was not found in the dataset."
            )

        source_row_count = len(df)
        source_column_count = len(df.columns)
        benchmark_warnings: list[str] = []
        started_at = perf_counter()
        lazy_gpu_enabled = bool(config.prefer_gpu and _lazypredict_gpu_usable())

        if config.prefer_gpu:
            if lazy_gpu_enabled:
                benchmark_warnings.append(
                    "CUDA detected; LazyPredict GPU acceleration is enabled for supported models."
                )
            else:
                benchmark_warnings.append(
                    "GPU preference is enabled, but CUDA was not detected; benchmarking will run on CPU."
                )

        working_df = df.copy()
        dropped_null_targets = int(working_df[config.target_column].isna().sum())
        if dropped_null_targets > 0:
            working_df = working_df.loc[working_df[config.target_column].notna()].copy()
            benchmark_warnings.append(
                f"Dropped {dropped_null_targets} row(s) with null target values before benchmarking."
            )

        if working_df.empty:
            raise BenchmarkTargetError("No rows remain after dropping null target values.")

        task_type, task_warnings = resolve_task_type(
            working_df[config.target_column],
            config.task_type,
        )
        benchmark_warnings.extend(task_warnings)

        if config.sample_rows is not None:
            working_df, sampling_warnings = self._apply_row_sampling(working_df, config, task_type)
            benchmark_warnings.extend(sampling_warnings)
        elif len(working_df) > self._sampling_row_threshold:
            benchmark_warnings.append(
                f"Dataset exceeds the benchmark sampling threshold ({self._sampling_row_threshold:,} rows); "
                f"consider sample_rows={self._suggested_sample_rows:,} for faster baselines."
            )

        benchmark_warnings.extend(
            benchmark_reliability_warnings(working_df, config.target_column, task_type)
        )

        feature_frame, target = self._prepare_frame(working_df, config, task_type, benchmark_warnings)
        stratify_target, stratified_split_applied, split_warnings = choose_stratify_target(
            target,
            task_type,
            config.split,
        )
        benchmark_warnings.extend(split_warnings)

        try:
            X_train, X_test, y_train, y_test = _train_test_split(
                feature_frame,
                target,
                test_size=config.split.test_size,
                random_state=config.split.random_state,
                stratify=stratify_target,
            )
        except (TypeError, ValueError) as exc:
            log_and_wrap(
                logger,
                exc,
                operation="benchmark.train_test_split",
                wrap_with=BenchmarkExecutionError,
                message=f"Failed to create the train/test split: {exc}",
                context={"task_type": task_type.value},
            )

        raw_results = self._run_lazypredict(
            X_train,
            X_test,
            y_train,
            y_test,
            config,
            task_type,
        )

        if raw_results.empty:
            raise BenchmarkExecutionError("LazyPredict returned an empty benchmark result table.")

        result_rows = build_result_rows(
            raw_results,
            task_type=task_type,
            benchmark_backend=execution_backend,
        )

        ranking_metric, ranking_direction, ranking_warnings = resolve_ranking_metric(
            task_type,
            raw_results.columns,
            preferred_metric=config.ranking_metric,
            default_metric=self._default_metric_for(task_type),
            raw_results=raw_results,
        )
        benchmark_warnings.extend(ranking_warnings)

        ranked_rows = rank_result_rows(
            result_rows,
            ranking_metric=ranking_metric,
            direction=ranking_direction,
        )

        duration_seconds = perf_counter() - started_at
        sampled_row_count = len(working_df) if config.sample_rows is not None and len(working_df) < source_row_count else None

        summary = build_benchmark_summary(
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            config=config,
            task_type=task_type,
            benchmark_backend=execution_backend,
            workspace_mode=workspace_mode,
            ranking_metric=ranking_metric,
            ranking_direction=ranking_direction,
            ranked_rows=ranked_rows,
            source_row_count=source_row_count,
            source_column_count=source_column_count,
            benchmark_row_count=len(working_df),
            feature_column_count=feature_frame.shape[1],
            train_row_count=len(X_train),
            test_row_count=len(X_test),
            sampled_row_count=sampled_row_count,
            stratified_split_applied=stratified_split_applied,
            benchmark_duration_seconds=duration_seconds,
            warnings=benchmark_warnings,
        )

        bundle = BenchmarkResultBundle(
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            config=config,
            task_type=task_type,
            benchmark_backend=execution_backend,
            workspace_mode=workspace_mode,
            raw_results=raw_results,
            leaderboard=ranked_rows,
            top_models=ranked_rows[: config.top_k],
            summary=summary,
            warnings=list(benchmark_warnings),
        )

        if self._artifacts_dir is not None:
            bundle.artifacts = write_benchmark_artifacts(bundle, self._artifacts_dir)

        tracker = self._build_tracker(MLflowBenchmarkTracker)
        if tracker is not None:
            run_id, tracking_warnings = tracker.log_bundle(bundle)
            bundle.mlflow_run_id = run_id
            self._append_bundle_warnings(bundle, tracking_warnings)

        return bundle

    def _default_metric_for(self, task_type: BenchmarkTaskType) -> str:
        if task_type == BenchmarkTaskType.CLASSIFICATION:
            return self._classification_default_metric
        return self._regression_default_metric

    def _apply_row_sampling(
        self,
        frame: pd.DataFrame,
        config: BenchmarkConfig,
        task_type: BenchmarkTaskType,
    ) -> tuple[pd.DataFrame, list[str]]:
        warnings: list[str] = []
        sample_rows = config.sample_rows
        if sample_rows is None or sample_rows >= len(frame):
            return frame, warnings

        if sample_rows < 2:
            raise BenchmarkConfigurationError("sample_rows must be at least 2 when provided.")

        sampled = frame.sample(n=sample_rows, random_state=config.split.random_state)
        warnings.append(
            f"Benchmark sampled {sample_rows:,} row(s) from {len(frame):,} source row(s)."
        )
        if task_type == BenchmarkTaskType.CLASSIFICATION:
            warnings.append(
                "Sampling was applied before classification benchmarking; rare classes may be underrepresented."
            )
        return sampled, warnings

    def _prepare_frame(
        self,
        frame: pd.DataFrame,
        config: BenchmarkConfig,
        task_type: BenchmarkTaskType,
        benchmark_warnings: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        feature_frame = frame.drop(columns=[config.target_column]).copy()
        if feature_frame.empty:
            raise BenchmarkConfigurationError(
                "Benchmarking requires at least one feature column after removing the target."
            )

        datetime_columns = list(feature_frame.select_dtypes(include=["datetime", "datetimetz"]).columns)
        if datetime_columns:
            feature_frame[datetime_columns] = feature_frame[datetime_columns].astype("string")
            benchmark_warnings.append(
                "Converted datetime feature columns to strings for LazyPredict compatibility: "
                + ", ".join(str(column) for column in datetime_columns)
            )

        nested_columns = collect_nested_object_columns(feature_frame)
        if nested_columns:
            for column in nested_columns:
                feature_frame[column] = feature_frame[column].map(_safe_stringify)
            benchmark_warnings.append(
                "Converted nested object feature columns to strings for LazyPredict compatibility: "
                + ", ".join(nested_columns)
            )

        target = frame[config.target_column].copy()
        if task_type == BenchmarkTaskType.REGRESSION:
            target = pd.to_numeric(target, errors="raise")

        return feature_frame, target

    def _run_lazypredict(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        config: BenchmarkConfig,
        task_type: BenchmarkTaskType,
    ) -> pd.DataFrame:
        lazy_module = _get_lazypredict_module()
        estimators = self._resolve_estimators(lazy_module, task_type, config)

        core_kwargs: dict[str, Any] = {
            "verbose": 0,
            "ignore_warnings": config.ignore_warnings,
            "custom_metric": None,
            "predictions": False,
            "random_state": config.split.random_state,
        }

        # These kwargs exist in newer LazyPredict releases but may be
        # absent in older versions (e.g. PyPI 0.2.16).  Guard via
        # constructor introspection so both old and new installs work.
        extended_kwargs: dict[str, Any] = {
            "categorical_encoder": config.categorical_encoder,
            "use_gpu": bool(config.prefer_gpu and _lazypredict_gpu_usable()),
        }
        if config.timeout_seconds is not None:
            extended_kwargs["timeout"] = config.timeout_seconds
        if config.max_models is not None:
            extended_kwargs["max_models"] = config.max_models

        if task_type == BenchmarkTaskType.CLASSIFICATION:
            cls = lazy_module.LazyClassifier
            core_kwargs["classifiers"] = estimators if estimators is not None else "all"
        elif task_type == BenchmarkTaskType.REGRESSION:
            cls = lazy_module.LazyRegressor
            core_kwargs["regressors"] = estimators if estimators is not None else "all"
        else:
            raise BenchmarkExecutionError(f"Unsupported task type for LazyPredict: {task_type.value}")

        core_kwargs.update(_filter_constructor_kwargs(cls, extended_kwargs))
        runner = cls(**core_kwargs)

        try:
            results = runner.fit(X_train, X_test, y_train, y_test)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            log_and_wrap(
                logger,
                exc,
                operation="benchmark.run_lazypredict",
                wrap_with=BenchmarkExecutionError,
                message=f"LazyPredict benchmark execution failed: {exc}",
                context={"task_type": task_type.value},
            )

        scores = results[0] if isinstance(results, tuple) else results
        if not isinstance(scores, pd.DataFrame):
            raise BenchmarkExecutionError("LazyPredict did not return a pandas DataFrame of scores.")
        return scores.copy()

    def _resolve_estimators(
        self,
        lazy_module: Any,
        task_type: BenchmarkTaskType,
        config: BenchmarkConfig,
    ) -> list[type] | None:
        include_models = set(config.include_models)
        exclude_models = set(config.exclude_models)
        if not include_models and not exclude_models:
            return None

        available_pairs = getattr(
            lazy_module,
            "CLASSIFIERS" if task_type == BenchmarkTaskType.CLASSIFICATION else "REGRESSORS",
            None,
        )
        if available_pairs is None:
            raise BenchmarkConfigurationError(
                "LazyPredict estimator filtering is unavailable with the installed package version."
            )

        available_by_name = {name: estimator for name, estimator in available_pairs}
        unknown_models = sorted((include_models | exclude_models) - set(available_by_name))
        if unknown_models:
            raise BenchmarkConfigurationError(
                "Unknown model name(s) requested: " + ", ".join(unknown_models)
            )

        selected_names = list(available_by_name)
        if include_models:
            selected_names = [name for name in selected_names if name in include_models]
        if exclude_models:
            selected_names = [name for name in selected_names if name not in exclude_models]

        if not selected_names:
            raise BenchmarkConfigurationError(
                "Benchmark model include/exclude filters removed every available estimator."
            )

        return [available_by_name[name] for name in selected_names]


def _filter_constructor_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return only *kwargs* that the class constructor actually accepts."""

    try:
        sig = inspect.signature(cls.__init__)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        accepted = set(sig.parameters.keys()) - {"self"}
        return {k: v for k, v in kwargs.items() if k in accepted}
    except (ValueError, TypeError):
        return kwargs


def _safe_stringify(value: object) -> object:
    if value is None:
        return value
    return str(value)