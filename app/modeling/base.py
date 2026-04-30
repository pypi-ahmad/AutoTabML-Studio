"""Shared base classes for modeling services, trackers, and artifact writers."""

from __future__ import annotations

import abc
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Generic, TypeVar

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.errors import log_exception

BundleT = TypeVar("BundleT")
ArtifactBundleT = TypeVar("ArtifactBundleT")
TrackerT = TypeVar("TrackerT", bound="BaseTracker[Any]")


def is_mlflow_available() -> bool:
    """Return True when mlflow is importable."""

    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


def get_mlflow_module() -> Any:
    """Import and return the mlflow module."""

    import mlflow

    return mlflow


def mlflow_exception_types(mlflow_module: Any) -> tuple[type[BaseException], ...]:
    """Return common MLflow exception types plus generic boundary failures."""

    exception_types: list[type[BaseException]] = [
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ]
    exceptions_module = getattr(mlflow_module, "exceptions", None)
    mlflow_exception = getattr(exceptions_module, "MlflowException", None)
    if isinstance(mlflow_exception, type) and issubclass(mlflow_exception, BaseException):
        exception_types.insert(0, mlflow_exception)
    return tuple(exception_types)


class BaseService(abc.ABC):
    """Common modeling-service configuration and helpers."""

    def __init__(
        self,
        *,
        artifacts_dir: Path | None = None,
        mlflow_experiment_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        metadata_store: Any | None = None,
    ) -> None:
        self._artifacts_dir = artifacts_dir
        self._mlflow_experiment_name = mlflow_experiment_name
        self._tracking_uri = tracking_uri
        self._registry_uri = registry_uri
        self._metadata_store = metadata_store

    def _build_tracker(self, tracker_cls: type[TrackerT]) -> TrackerT | None:
        """Return a configured tracker instance when MLflow tracking is enabled."""

        if not self._mlflow_experiment_name:
            return None
        return tracker_cls(
            self._mlflow_experiment_name,
            tracking_uri=self._tracking_uri,
            registry_uri=self._registry_uri,
        )

    def _append_bundle_warnings(self, bundle: Any, warnings: list[str]) -> None:
        """Append warnings to the bundle and its summary without duplicates."""

        if not warnings:
            return

        summary = getattr(bundle, "summary", None)
        summary_warnings = getattr(summary, "warnings", None)
        bundle_warnings = getattr(bundle, "warnings", None)

        for warning in warnings:
            if isinstance(bundle_warnings, list) and warning not in bundle_warnings:
                bundle_warnings.append(warning)
            if isinstance(summary_warnings, list) and warning not in summary_warnings:
                summary_warnings.append(warning)


class BaseTracker(Generic[BundleT], abc.ABC):
    """Shared MLflow tracking lifecycle for modeling bundles."""

    def __init__(
        self,
        experiment_name: str,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._registry_uri = registry_uri
        self._logger = logger or logging.getLogger(self.__class__.__module__)

    def log_bundle(
        self,
        bundle: BundleT,
        *,
        existing_run_id: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """Log params, metrics, and artifacts for one modeling bundle."""

        warnings: list[str] = []
        if not self._is_mlflow_available():
            warnings.append("MLflow tracking skipped because mlflow is not installed.")
            return self._failure_run_id(existing_run_id), warnings

        mlflow_module = self._get_mlflow_module()
        handled_errors = mlflow_exception_types(mlflow_module)

        try:
            if self._tracking_uri:
                mlflow_module.set_tracking_uri(self._tracking_uri)
            if self._registry_uri:
                mlflow_module.set_registry_uri(self._registry_uri)
            mlflow_module.set_experiment(self._experiment_name)
            with mlflow_module.start_run(**self._start_run_kwargs(bundle, existing_run_id)) as run:
                params = self._build_params(bundle)
                if params:
                    mlflow_module.log_params(params)
                metrics = self._build_metrics(bundle)
                if metrics:
                    mlflow_module.log_metrics(metrics)
                self._log_artifacts(mlflow_module, bundle)
                return run.info.run_id, warnings
        except handled_errors as exc:
            log_exception(
                self._logger,
                exc,
                operation=self._operation_name(),
                context={"experiment_name": self._experiment_name},
            )
            warnings.append(f"MLflow tracking failed: {exc}")
            return self._failure_run_id(existing_run_id), warnings

    def _start_run_kwargs(self, bundle: BundleT, existing_run_id: str | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"run_name": self._build_run_name(bundle)}
        if existing_run_id is not None:
            kwargs["run_id"] = existing_run_id
        return kwargs

    def _failure_run_id(self, existing_run_id: str | None) -> str | None:
        return existing_run_id

    def _log_artifacts(self, mlflow_module: Any, bundle: BundleT) -> None:
        for path in self._artifact_paths(bundle):
            if path is None:
                continue
            artifact_path = Path(path)
            if artifact_path.exists():
                mlflow_module.log_artifact(str(artifact_path))

    @abc.abstractmethod
    def _is_mlflow_available(self) -> bool:
        """Return True if the subclass-specific MLflow boundary is available."""

    @abc.abstractmethod
    def _get_mlflow_module(self) -> Any:
        """Return the subclass-specific MLflow module handle."""

    @abc.abstractmethod
    def _operation_name(self) -> str:
        """Return the operation label used for structured error logging."""

    @abc.abstractmethod
    def _build_run_name(self, bundle: BundleT) -> str:
        """Return the MLflow run name for the bundle."""

    @abc.abstractmethod
    def _build_params(self, bundle: BundleT) -> dict[str, Any]:
        """Return MLflow params for the bundle."""

    @abc.abstractmethod
    def _build_metrics(self, bundle: BundleT) -> dict[str, float]:
        """Return MLflow metrics for the bundle."""

    @abc.abstractmethod
    def _artifact_paths(self, bundle: BundleT) -> Iterable[Path | str | None]:
        """Yield the artifact paths that should be uploaded to MLflow."""


class BaseArtifacts(Generic[BundleT, ArtifactBundleT], abc.ABC):
    """Shared helpers for artifact path generation and persistence."""

    artifact_kind: ArtifactKind
    artifact_bundle_cls: type[ArtifactBundleT]

    def __init__(
        self,
        bundle: BundleT,
        artifacts_dir: Path,
        *,
        manager: LocalArtifactManager | None = None,
    ) -> None:
        self.bundle = bundle
        self.artifacts_dir = artifacts_dir
        self.manager = manager or LocalArtifactManager()
        self.artifacts = self.artifact_bundle_cls()

    @abc.abstractmethod
    def build(self) -> ArtifactBundleT:
        """Write artifacts for the current bundle and return the bundle of paths."""

    def _artifact_path(
        self,
        *,
        label: str,
        suffix: str,
        stem: str | None = None,
        kind: ArtifactKind | None = None,
    ) -> Path:
        summary = getattr(self.bundle, "summary")
        dataset_name = stem if stem is not None else getattr(self.bundle, "dataset_name", None)
        return self.manager.build_artifact_path(
            kind=kind or self.artifact_kind,
            stem=dataset_name,
            label=label,
            suffix=suffix,
            timestamp=summary.run_timestamp,
            output_dir=self.artifacts_dir,
        )

    def _write_text(self, path: Path, content: str) -> Path:
        self.manager.write_text(path, content)
        return path

    def _write_json(self, path: Path, payload: Any) -> Path:
        self.manager.write_json(path, payload)
        return path

    def _write_dataframe(self, path: Path, dataframe, *, index: bool = False) -> Path:  # noqa: ANN001
        self.manager.write_dataframe_csv(path, dataframe, index=index)
        return path
