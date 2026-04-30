"""Page-facing workflow services for prediction-related Streamlit pages.

These helpers keep Streamlit page modules focused on widget rendering while the
workflow service owns request construction, selection normalization, state
reset rules, file parsing, and evaluation logic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.modeling.benchmark.persistence import load_saved_benchmark_model
from app.modeling.benchmark.schemas import BenchmarkSavedModelMetadata
from app.prediction import AvailableModelReference, BatchPredictionRequest, PredictionRequest


@dataclass(frozen=True)
class PredictionExecutionConfig:
    """Prediction execution settings distilled from app configuration."""

    tracking_uri: str | None
    registry_uri: str | None
    output_dir: Path
    output_stem: str
    prediction_column_name: str


@dataclass(frozen=True)
class ModelTestingSelection:
    """One normalized model-testing selection entry shown in the page."""

    label: str
    source_kind: str
    prediction_reference: AvailableModelReference | None = None
    benchmark_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelTestingRunResult:
    """Normalized prediction output returned to the model-testing page."""

    scored_dataframe: pd.DataFrame
    predictions: pd.Series | None


@dataclass(frozen=True)
class ModelTestingEvaluation:
    """Computed evaluation metrics for model-testing results."""

    task_type: str
    metrics: dict[str, float]
    primary_metric_key: str
    verdict: str


class PredictionWorkflowService:
    """Orchestrate page-level prediction workflows outside Streamlit pages."""

    def build_execution_config(self, prediction_settings, tracking_settings) -> PredictionExecutionConfig:  # noqa: ANN001
        return PredictionExecutionConfig(
            tracking_uri=tracking_settings.tracking_uri,
            registry_uri=tracking_settings.registry_uri,
            output_dir=prediction_settings.artifacts_dir,
            output_stem=prediction_settings.default_output_stem,
            prediction_column_name=prediction_settings.prediction_column_name,
        )

    def build_request_signature(self, request_kwargs: Mapping[str, Any]) -> str:
        return json.dumps(dict(request_kwargs), default=str, sort_keys=True)

    def reset_prediction_session_state(
        self,
        session_state: MutableMapping[str, Any],
        current_signature: str | None,
    ) -> None:
        previous_signature = session_state.get("prediction_loaded_signature")
        if previous_signature == current_signature:
            return
        session_state.pop("prediction_loaded_model", None)
        session_state.pop("prediction_single_result", None)
        session_state.pop("prediction_batch_result", None)
        if current_signature is None:
            session_state.pop("prediction_loaded_signature", None)

    def load_prediction_model(self, prediction_service, request_kwargs: Mapping[str, Any]):  # noqa: ANN001
        request = PredictionRequest(**dict(request_kwargs))
        return prediction_service.load_model(request)

    def resolve_single_row_payload(
        self,
        *,
        use_form: bool,
        row_payload: Mapping[str, Any] | None,
        row_json_text: str,
    ) -> dict[str, Any]:
        if use_form:
            return dict(row_payload or {})

        payload = json.loads(row_json_text)
        if not isinstance(payload, dict):
            raise ValueError("Prediction input data must be a JSON object.")
        return payload

    def build_model_testing_selections(
        self,
        prediction_refs: Sequence[AvailableModelReference],
        benchmark_refs: Sequence[Mapping[str, Any]],
    ) -> list[ModelTestingSelection]:
        selections: list[ModelTestingSelection] = []
        for ref in prediction_refs:
            task_label = self._format_task_label(ref.task_type.value)
            selections.append(
                ModelTestingSelection(
                    label=f"🔬 {ref.display_name} ({task_label})",
                    source_kind="pycaret",
                    prediction_reference=ref,
                )
            )
        for metadata in benchmark_refs:
            benchmark_metadata = dict(metadata)
            task_label = self._format_task_label(str(benchmark_metadata.get("task_type", "unknown")))
            selections.append(
                ModelTestingSelection(
                    label=(
                        f"🏁 {benchmark_metadata.get('model_name', 'Unknown model')}"
                        f" — {benchmark_metadata.get('dataset_name', '?')} ({task_label})"
                    ),
                    source_kind="benchmark",
                    benchmark_metadata=benchmark_metadata,
                )
            )
        return selections

    def build_model_testing_load_key(self, selection: ModelTestingSelection) -> str:
        return f"mt_loaded:{selection.source_kind}:{selection.label}"

    def reset_model_testing_state(
        self,
        session_state: MutableMapping[str, Any],
        load_key: str,
    ) -> None:
        if session_state.get("mt_loaded_key") == load_key:
            return
        session_state.pop("mt_loaded_model", None)
        session_state.pop("mt_loaded_source", None)
        session_state.pop("mt_batch_result", None)
        session_state.pop("mt_predictions", None)
        session_state.pop("mt_eval_result", None)

    def load_model_testing_model(
        self,
        *,
        selection: ModelTestingSelection,
        prediction_service,  # noqa: ANN001
        config: PredictionExecutionConfig,
        trusted_model_roots: Sequence[Path],
    ):
        if selection.source_kind == "pycaret":
            reference = selection.prediction_reference
            if reference is None:
                raise ValueError("PyCaret selection is missing its model reference.")
            request = PredictionRequest(
                source_type=reference.source_type,
                model_identifier=reference.load_reference,
                model_path=reference.model_path,
                metadata_path=reference.metadata_path,
                tracking_uri=config.tracking_uri,
                registry_uri=config.registry_uri,
                output_dir=config.output_dir,
                output_stem=config.output_stem,
            )
            return prediction_service.load_model(request)

        metadata = BenchmarkSavedModelMetadata.model_validate(selection.benchmark_metadata or {})
        return load_saved_benchmark_model(metadata, trusted_roots=list(trusted_model_roots))

    def load_uploaded_test_dataframe(self, uploaded_file) -> pd.DataFrame:  # noqa: ANN001
        filename = str(getattr(uploaded_file, "name", "")).lower()
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file)

    def run_model_testing_predictions(
        self,
        *,
        selection: ModelTestingSelection,
        loaded_model,  # noqa: ANN001
        test_dataframe: pd.DataFrame,
        data_label: str,
        prediction_service,  # noqa: ANN001
        config: PredictionExecutionConfig,
    ) -> ModelTestingRunResult:
        if selection.source_kind == "pycaret":
            reference = selection.prediction_reference
            if reference is None:
                raise ValueError("PyCaret selection is missing its model reference.")
            result = prediction_service.predict_batch(
                BatchPredictionRequest(
                    source_type=reference.source_type,
                    model_identifier=reference.load_reference,
                    model_path=reference.model_path,
                    metadata_path=reference.metadata_path,
                    tracking_uri=config.tracking_uri,
                    registry_uri=config.registry_uri,
                    output_dir=config.output_dir,
                    output_stem=config.output_stem,
                    dataframe=test_dataframe,
                    dataset_name=data_label,
                    input_source_label=data_label,
                )
            )
            predictions = None
            if config.prediction_column_name in result.scored_dataframe.columns:
                predictions = result.scored_dataframe[config.prediction_column_name]
            return ModelTestingRunResult(scored_dataframe=result.scored_dataframe, predictions=predictions)

        metadata = selection.benchmark_metadata or {}
        feature_columns = [str(column) for column in metadata.get("feature_columns", [])]
        available_columns = [column for column in feature_columns if column in test_dataframe.columns]
        if not available_columns:
            raise ValueError("Test data does not contain any of the expected input columns.")

        predictions = pd.Series(
            loaded_model.predict(test_dataframe[available_columns]),
            index=test_dataframe.index,
            name=config.prediction_column_name,
        )
        scored_dataframe = test_dataframe.copy()
        scored_dataframe[config.prediction_column_name] = predictions
        return ModelTestingRunResult(scored_dataframe=scored_dataframe, predictions=predictions)

    def target_column_for_testing(
        self,
        source_kind: str,
        loaded_model,  # noqa: ANN001
        selection: ModelTestingSelection,
    ) -> str | None:
        if source_kind == "pycaret":
            return getattr(loaded_model, "target_column", None)
        metadata = selection.benchmark_metadata or {}
        target_column = metadata.get("target_column")
        return str(target_column) if target_column else None

    def infer_model_testing_task_type(
        self,
        source_kind: str,
        loaded_model,  # noqa: ANN001
        selection: ModelTestingSelection,
    ) -> str:
        if source_kind == "pycaret":
            task_value = getattr(getattr(loaded_model, "task_type", None), "value", "classification")
            return str(task_value).lower()
        metadata = selection.benchmark_metadata or {}
        return str(metadata.get("task_type", "classification")).lower()

    def evaluate_predictions(
        self,
        *,
        y_true: pd.Series,
        y_pred: pd.Series,
        task_type: str,
    ) -> ModelTestingEvaluation:
        prediction_series = y_pred if isinstance(y_pred, pd.Series) else pd.Series(y_pred)
        mask = y_true.notna() & prediction_series.notna()
        if not bool(mask.any()):
            raise ValueError("No comparable rows remain after removing missing values.")

        normalized_task = task_type.lower()
        if "classif" in normalized_task:
            yt = y_true[mask]
            yp = prediction_series[mask]
            metrics = self._classification_metrics(yt, yp)
            accuracy = metrics["accuracy"]
            verdict = (
                "The model performs well on this test set."
                if accuracy >= 0.7
                else "The model's accuracy is low — consider tuning it or trying a different algorithm."
            )
            return ModelTestingEvaluation(
                task_type="classification",
                metrics=metrics,
                primary_metric_key="accuracy",
                verdict=verdict,
            )

        yt = y_true[mask].astype(float)
        yp = prediction_series[mask].astype(float)
        metrics = self._regression_metrics(yt, yp)
        r2_value = metrics["r2"]
        verdict = (
            "The model performs well on this test set."
            if r2_value >= 0.5
            else "The model's accuracy is low — consider tuning it or trying a different algorithm."
        )
        return ModelTestingEvaluation(
            task_type="regression",
            metrics=metrics,
            primary_metric_key="r2",
            verdict=verdict,
        )

    @staticmethod
    def _format_task_label(task_value: str) -> str:
        return task_value.replace("_", " ").strip().title()

    @staticmethod
    def _classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        accuracy = float((y_true == y_pred).mean())
        labels = pd.Index(pd.concat([y_true, y_pred]).astype(str).unique())
        total_support = float(len(y_true))

        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

        for label in labels:
            true_mask = y_true.astype(str) == label
            pred_mask = y_pred.astype(str) == label
            support = float(true_mask.sum())
            true_positive = float((true_mask & pred_mask).sum())
            false_positive = float((~true_mask & pred_mask).sum())
            false_negative = float((true_mask & ~pred_mask).sum())

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            weighted_precision += precision * support
            weighted_recall += recall * support
            weighted_f1 += f1 * support

        if total_support == 0:
            raise ValueError("No rows are available to evaluate classification metrics.")

        return {
            "accuracy": accuracy,
            "precision": weighted_precision / total_support,
            "recall": weighted_recall / total_support,
            "f1": weighted_f1 / total_support,
        }

    @staticmethod
    def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        errors = y_true - y_pred
        mae = float(errors.abs().mean())
        rmse = float(math.sqrt((errors.pow(2)).mean()))

        residual_sum_squares = float(errors.pow(2).sum())
        total_sum_squares = float((y_true - y_true.mean()).pow(2).sum())
        if total_sum_squares == 0.0:
            r2_value = 1.0 if residual_sum_squares == 0.0 else 0.0
        else:
            r2_value = 1.0 - (residual_sum_squares / total_sum_squares)

        return {
            "r2": r2_value,
            "mae": mae,
            "rmse": rmse,
        }


__all__ = [
    "ModelTestingEvaluation",
    "ModelTestingRunResult",
    "ModelTestingSelection",
    "PredictionExecutionConfig",
    "PredictionWorkflowService",
]