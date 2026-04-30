"""Tests for the page-level prediction workflow service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.pages.services.prediction_workflow import (
    ModelTestingSelection,
    PredictionWorkflowService,
)
from app.prediction import AvailableModelReference, ModelSourceType, PredictionTaskType


class _StubPredictionService:
    def __init__(self) -> None:
        self.loaded_requests = []
        self.batch_requests = []

    def load_model(self, request):  # noqa: ANN001
        self.loaded_requests.append(request)
        return "loaded-model"

    def predict_batch(self, request):  # noqa: ANN001
        self.batch_requests.append(request)
        return SimpleNamespace(
            scored_dataframe=pd.DataFrame({"feature": [1, 2], "prediction": ["yes", "no"]})
        )


class _BenchmarkModel:
    def predict(self, dataframe: pd.DataFrame):
        return pd.Series(["ok" for _ in range(len(dataframe))], index=dataframe.index)


def _prediction_ref() -> AvailableModelReference:
    return AvailableModelReference(
        source_type=ModelSourceType.LOCAL_SAVED_MODEL,
        display_name="Demo Model",
        model_identifier="demo-model",
        load_reference="models/demo.pkl",
        task_type=PredictionTaskType.CLASSIFICATION,
        model_path=Path("models/demo.pkl"),
        metadata_path=Path("models/demo.json"),
    )


def _execution_config():
    prediction_settings = SimpleNamespace(
        artifacts_dir=Path("artifacts/predictions"),
        default_output_stem="predictions",
        prediction_column_name="prediction",
    )
    tracking_settings = SimpleNamespace(
        tracking_uri="sqlite:///mlruns.db",
        registry_uri="sqlite:///registry.db",
    )
    return PredictionWorkflowService().build_execution_config(prediction_settings, tracking_settings)


class TestPredictionWorkflowService:
    def test_build_request_signature_is_stable(self) -> None:
        workflow = PredictionWorkflowService()

        left = workflow.build_request_signature({"b": 2, "a": 1})
        right = workflow.build_request_signature({"a": 1, "b": 2})

        assert left == right

    def test_reset_prediction_session_state_clears_when_signature_changes(self) -> None:
        workflow = PredictionWorkflowService()
        session_state = {
            "prediction_loaded_signature": "old",
            "prediction_loaded_model": object(),
            "prediction_single_result": object(),
            "prediction_batch_result": object(),
        }

        workflow.reset_prediction_session_state(session_state, "new")

        assert "prediction_loaded_model" not in session_state
        assert "prediction_single_result" not in session_state
        assert "prediction_batch_result" not in session_state
        assert session_state["prediction_loaded_signature"] == "old"

    def test_load_prediction_model_builds_request(self) -> None:
        workflow = PredictionWorkflowService()
        service = _StubPredictionService()

        loaded = workflow.load_prediction_model(
            service,
            {
                "source_type": ModelSourceType.LOCAL_SAVED_MODEL,
                "model_identifier": "demo-model",
            },
        )

        assert loaded == "loaded-model"
        assert service.loaded_requests[0].model_identifier == "demo-model"

    def test_resolve_single_row_payload_requires_json_object(self) -> None:
        workflow = PredictionWorkflowService()

        with pytest.raises(ValueError):
            workflow.resolve_single_row_payload(
                use_form=False,
                row_payload=None,
                row_json_text='["not", "an", "object"]',
            )

    def test_build_model_testing_selections_merges_sources(self) -> None:
        workflow = PredictionWorkflowService()

        selections = workflow.build_model_testing_selections(
            [_prediction_ref()],
            [{"model_name": "Benchmark A", "dataset_name": "Iris", "task_type": "classification"}],
        )

        assert [item.source_kind for item in selections] == ["pycaret", "benchmark"]
        assert selections[0].prediction_reference is not None
        assert selections[1].benchmark_metadata == {
            "model_name": "Benchmark A",
            "dataset_name": "Iris",
            "task_type": "classification",
        }

    def test_load_model_testing_model_for_pycaret_uses_prediction_service(self) -> None:
        workflow = PredictionWorkflowService()
        service = _StubPredictionService()
        selection = ModelTestingSelection(
            label="Demo",
            source_kind="pycaret",
            prediction_reference=_prediction_ref(),
        )

        loaded = workflow.load_model_testing_model(
            selection=selection,
            prediction_service=service,
            config=_execution_config(),
            trusted_model_roots=[Path("models")],
        )

        assert loaded == "loaded-model"
        assert service.loaded_requests[0].model_path == Path("models/demo.pkl")

    def test_run_model_testing_predictions_for_pycaret_returns_scored_dataframe(self) -> None:
        workflow = PredictionWorkflowService()
        service = _StubPredictionService()
        selection = ModelTestingSelection(
            label="Demo",
            source_kind="pycaret",
            prediction_reference=_prediction_ref(),
        )

        result = workflow.run_model_testing_predictions(
            selection=selection,
            loaded_model=object(),
            test_dataframe=pd.DataFrame({"feature": [1, 2]}),
            data_label="demo.csv",
            prediction_service=service,
            config=_execution_config(),
        )

        assert list(result.scored_dataframe.columns) == ["feature", "prediction"]
        assert result.predictions.tolist() == ["yes", "no"]
        assert service.batch_requests[0].dataset_name == "demo.csv"

    def test_run_model_testing_predictions_for_benchmark_uses_expected_features(self) -> None:
        workflow = PredictionWorkflowService()
        selection = ModelTestingSelection(
            label="Benchmark",
            source_kind="benchmark",
            benchmark_metadata={"feature_columns": ["a", "b"], "task_type": "classification"},
        )

        result = workflow.run_model_testing_predictions(
            selection=selection,
            loaded_model=_BenchmarkModel(),
            test_dataframe=pd.DataFrame({"a": [1], "b": [2], "c": [3]}),
            data_label="benchmark.csv",
            prediction_service=_StubPredictionService(),
            config=_execution_config(),
        )

        assert result.predictions.tolist() == ["ok"]
        assert result.scored_dataframe["prediction"].tolist() == ["ok"]

    def test_evaluate_predictions_for_classification(self) -> None:
        workflow = PredictionWorkflowService()

        evaluation = workflow.evaluate_predictions(
            y_true=pd.Series(["yes", "no", "yes"]),
            y_pred=pd.Series(["yes", "no", "yes"]),
            task_type="classification",
        )

        assert evaluation.task_type == "classification"
        assert evaluation.primary_metric_key == "accuracy"
        assert evaluation.metrics["accuracy"] == pytest.approx(1.0)

    def test_evaluate_predictions_for_regression(self) -> None:
        workflow = PredictionWorkflowService()

        evaluation = workflow.evaluate_predictions(
            y_true=pd.Series([1.0, 2.0, 3.0]),
            y_pred=pd.Series([1.0, 2.0, 3.0]),
            task_type="regression",
        )

        assert evaluation.task_type == "regression"
        assert evaluation.primary_metric_key == "r2"
        assert evaluation.metrics["r2"] == pytest.approx(1.0)