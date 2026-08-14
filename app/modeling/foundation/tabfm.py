"""TabFM evaluation and checksum-backed research-context persistence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.artifacts.manager import LocalArtifactManager
from app.modeling.foundation.checkpoints import TABFM_CHECKPOINT, CheckpointResolver
from app.security.trusted_artifacts import compute_sha256, verify_local_artifact, write_checksum_file

TABFM_WEIGHTS_LICENSE = "tabfm-non-commercial-v1.0"
_CONTEXT_SOURCE = "autotabml_tabfm_research_context_v1"


class TabFMConfig(BaseModel):
    """Configuration for a local TabFM holdout evaluation."""

    target_column: str
    task_type: Literal["auto", "classification", "regression"] = "auto"
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = 42
    context_rows: int = Field(default=100, ge=2)
    n_estimators: int = Field(default=32, ge=1)


@dataclass
class TabFMResult:
    task_type: Literal["classification", "regression"]
    metrics: dict[str, float]
    predictions: pd.DataFrame
    context: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    native_model: Any
    checkpoint_path: Path
    config: TabFMConfig
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class TabFMSavedContext:
    metadata_path: Path
    context_path: Path
    checksum_path: Path


@dataclass
class TabFMLoadedContext:
    native_model: Any
    task_type: Literal["classification", "regression"]
    feature_columns: list[str]
    target_column: str
    metadata: dict[str, Any]


EstimatorFactory = Callable[[Path, str, TabFMConfig], Any]


class TabFMService:
    """Run the pinned TabFM model without making it a deployable model type."""

    def __init__(
        self,
        *,
        checkpoint_resolver: CheckpointResolver | None = None,
        estimator_factory: EstimatorFactory | None = None,
        artifact_manager: LocalArtifactManager | None = None,
    ) -> None:
        self._resolver = checkpoint_resolver or CheckpointResolver()
        self._estimator_factory = estimator_factory or _build_estimator
        self._artifacts = artifact_manager or LocalArtifactManager()

    def run(
        self,
        dataframe: pd.DataFrame,
        config: TabFMConfig,
        *,
        accept_license: bool,
        allow_download: bool = False,
        output_dir: Path | None = None,
    ) -> TabFMResult:
        if not accept_license:
            raise ValueError(
                "TabFM weights use the tabfm-non-commercial-v1.0 license and require explicit non-commercial "
                "research acceptance."
            )
        data = dataframe.copy()
        if config.target_column not in data.columns:
            raise ValueError(f"Target column not found: {config.target_column}")
        if len(data) < 5:
            raise ValueError("TabFM evaluation requires at least 5 rows.")
        feature_columns = [column for column in data.columns if column != config.target_column]
        if not feature_columns:
            raise ValueError("TabFM requires at least one feature column.")
        if len(feature_columns) > 500:
            raise ValueError("TabFM supports at most 500 selected features.")

        y = data[config.target_column]
        task_type = _resolve_task_type(y, config.task_type)
        if task_type == "classification" and y.nunique(dropna=True) > 10:
            raise ValueError("TabFM classification supports at most 10 classes.")

        X = data[feature_columns]
        from sklearn.model_selection import train_test_split

        stratify = y if task_type == "classification" and y.value_counts(dropna=False).min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify,
        )
        sample_size = min(config.context_rows, len(X_train))
        sampled_indices = X_train.sample(n=sample_size, random_state=config.random_state).index
        X_context = X_train.loc[sampled_indices].copy()
        y_context = y_train.loc[sampled_indices].copy()
        context = X_context.copy()
        context[config.target_column] = y_context

        checkpoint_path = self._resolver.resolve(TABFM_CHECKPOINT, allow_download=allow_download)
        estimator = self._estimator_factory(checkpoint_path, task_type, config)
        estimator.fit(X_context, y_context)
        predicted = np.asarray(estimator.predict(X_test))
        metrics, predictions = _score_predictions(estimator, task_type, X_test, y_test, predicted)
        artifacts = self._write_run_artifacts(
            predictions,
            metrics,
            config,
            task_type,
            checkpoint_path,
            output_dir=output_dir,
        )
        return TabFMResult(
            task_type=task_type,
            metrics=metrics,
            predictions=predictions,
            context=context,
            feature_columns=feature_columns,
            target_column=config.target_column,
            native_model=estimator,
            checkpoint_path=checkpoint_path,
            config=config,
            artifacts=artifacts,
        )

    def save_context(self, result: TabFMResult, *, name: str, output_dir: Path) -> TabFMSavedContext:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_stem(name)
        context_path = output_dir / f"{stem}_tabfm_context.json"
        context_path.write_text(result.context.to_json(orient="table", index=False), encoding="utf-8")
        context_checksum = compute_sha256(context_path)
        checksum_path = write_checksum_file(context_path, checksum=context_checksum)
        metadata = {
            "artifact_format": "tabfm_context_v1",
            "trusted_source": _CONTEXT_SOURCE,
            "weights_license": TABFM_WEIGHTS_LICENSE,
            "license_accepted_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": {
                "repo_id": TABFM_CHECKPOINT.repo_id,
                "revision": TABFM_CHECKPOINT.revision,
            },
            "task_type": result.task_type,
            "target_column": result.target_column,
            "feature_columns": result.feature_columns,
            "context_file": context_path.name,
            "context_sha256": context_checksum,
            "config": result.config.model_dump(mode="json"),
            "research_only": True,
            "deployable": False,
        }
        metadata_path = output_dir / f"{stem}_tabfm_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        write_checksum_file(metadata_path)
        return TabFMSavedContext(metadata_path, context_path, checksum_path)

    def load_context(
        self,
        metadata_path: Path,
        *,
        estimator_factory: EstimatorFactory | None = None,
        allow_download: bool = False,
    ) -> TabFMLoadedContext:
        root = metadata_path.resolve().parent
        verified_metadata = verify_local_artifact(metadata_path, trusted_roots=[root], label="TabFM metadata")
        metadata = json.loads(verified_metadata.path.read_text(encoding="utf-8"))
        if metadata.get("trusted_source") != _CONTEXT_SOURCE or metadata.get("deployable") is not False:
            raise ValueError("TabFM metadata is not a trusted research-only context.")
        context_path = root / str(metadata["context_file"])
        try:
            verified_context = verify_local_artifact(
                context_path,
                trusted_roots=[root],
                expected_sha256=str(metadata["context_sha256"]),
                label="TabFM context",
            )
        except Exception as exc:
            raise ValueError(f"TabFM context checksum validation failed: {exc}") from exc
        context = pd.read_json(StringIO(verified_context.path.read_text(encoding="utf-8")), orient="table")
        config = TabFMConfig.model_validate(metadata["config"])
        task_type = str(metadata["task_type"])
        checkpoint_path = self._resolver.resolve(TABFM_CHECKPOINT, allow_download=allow_download)
        factory = estimator_factory or self._estimator_factory
        estimator = factory(checkpoint_path, task_type, config)
        feature_columns = [str(column) for column in metadata["feature_columns"]]
        target_column = str(metadata["target_column"])
        estimator.fit(context[feature_columns], context[target_column])
        return TabFMLoadedContext(
            native_model=estimator,
            task_type=task_type,  # type: ignore[arg-type]
            feature_columns=feature_columns,
            target_column=target_column,
            metadata=metadata,
        )

    def _write_run_artifacts(
        self,
        predictions: pd.DataFrame,
        metrics: dict[str, float],
        config: TabFMConfig,
        task_type: str,
        checkpoint_path: Path,
        *,
        output_dir: Path | None,
    ) -> dict[str, Path]:
        directory = output_dir or self._artifacts.settings.experiments_dir / "foundation"
        directory.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex[:8]
        paths = {
            "predictions": directory / f"tabfm_predictions_{token}.csv",
            "metrics": directory / f"tabfm_metrics_{token}.json",
            "summary": directory / f"tabfm_summary_{token}.json",
        }
        self._artifacts.write_dataframe_csv(paths["predictions"], predictions, index=False)
        self._artifacts.write_json(paths["metrics"], metrics)
        self._artifacts.write_json(
            paths["summary"],
            {
                "model": TABFM_CHECKPOINT.repo_id,
                "revision": TABFM_CHECKPOINT.revision,
                "checkpoint_path": str(checkpoint_path),
                "task_type": task_type,
                "config": config.model_dump(mode="json"),
                "weights_license": TABFM_WEIGHTS_LICENSE,
                "research_only": True,
            },
        )
        return paths


def _resolve_task_type(series: pd.Series, configured: str) -> Literal["classification", "regression"]:
    if configured != "auto":
        return configured  # type: ignore[return-value]
    if not pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=True) <= 10:
        return "classification"
    return "regression"


def _score_predictions(
    estimator: Any,
    task_type: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predicted: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    predictions = X_test.copy()
    predictions["actual"] = y_test.to_numpy()
    predictions["predicted"] = predicted
    if task_type == "classification":
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

        probabilities = np.asarray(estimator.predict_proba(X_test))
        classes = np.asarray(estimator.classes_)
        for index, label in enumerate(classes):
            predictions[f"probability_{label}"] = probabilities[:, index]
        metrics = {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro")),
            "log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        }
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {
            "mae": float(mean_absolute_error(y_test, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "r2": float(r2_score(y_test, predicted)),
        }
    return metrics, predictions


def _build_estimator(checkpoint_path: Path, task_type: str, config: TabFMConfig) -> Any:
    try:
        from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Install the 'tabfm' extra to run TabFM.") from exc
    model = tabfm_v1_0_0_pytorch.load(model_type=task_type, checkpoint_path=str(checkpoint_path))
    estimator_type = TabFMClassifier if task_type == "classification" else TabFMRegressor
    return estimator_type(
        model=model,
        n_estimators=config.n_estimators,
        max_num_features=500,
        max_num_rows=config.context_rows,
        random_state=config.random_state,
    )


def _safe_stem(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return normalized.strip("-_") or "tabfm-context"
