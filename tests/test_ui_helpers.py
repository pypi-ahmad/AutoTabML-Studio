"""Tests for shared UI helpers — label maps, formatters, and description templates."""

from __future__ import annotations

import pytest

from app.pages.ui_labels import (
    BACKEND_LABELS,
    PREDICTION_TASK_TYPE_LABELS,
    PROMOTION_LABELS,
    PROVIDER_LABELS,
    SOURCE_TYPE_LABELS,
    TASK_TYPE_LABELS,
    format_enum_value,
    make_format_func,
)
from app.storage.models import AppJobType
from app.tracking.description_generator import generate_template_description


# ── format_enum_value ─────────────────────────────────────────────────


class TestFormatEnumValue:
    def test_snake_case(self):
        assert format_enum_value("benchmark_completed") == "Benchmark Completed"

    def test_single_word(self):
        assert format_enum_value("success") == "Success"

    def test_upper_screaming_snake(self):
        assert format_enum_value("LOCAL_SAVED_MODEL") == "Local Saved Model"

    def test_empty_string(self):
        assert format_enum_value("") == ""


# ── make_format_func ──────────────────────────────────────────────────


class TestMakeFormatFunc:
    def test_returns_label_when_present(self):
        func = make_format_func({"foo": "Bar"})
        assert func("foo") == "Bar"

    def test_fallback_title(self):
        func = make_format_func({"foo": "Bar"})
        assert func("unknown_value") == "Unknown Value"

    def test_fallback_raw_when_disabled(self):
        func = make_format_func({"foo": "Bar"}, fallback_title=False)
        assert func("unknown_value") == "unknown_value"


# ── Label maps cover expected enum values ─────────────────────────────


class TestLabelMaps:
    def test_prediction_task_type_labels_cover_core_values(self):
        for key in ("unknown", "classification", "regression"):
            assert key in PREDICTION_TASK_TYPE_LABELS

    def test_source_type_labels_cover_core_values(self):
        for key in ("LOCAL_SAVED_MODEL", "MLFLOW_RUN_MODEL", "MLFLOW_REGISTERED_MODEL"):
            assert key in SOURCE_TYPE_LABELS

    def test_promotion_labels_cover_core_values(self):
        for key in ("PROMOTE_TO_CHAMPION", "PROMOTE_TO_CANDIDATE", "ARCHIVE"):
            assert key in PROMOTION_LABELS

    def test_provider_labels_have_no_raw_values(self):
        """All provider labels should be human-readable, not raw enum values."""
        for value in PROVIDER_LABELS.values():
            assert value[0].isupper(), f"Provider label '{value}' should be title-case"

    def test_backend_labels_have_no_raw_values(self):
        for value in BACKEND_LABELS.values():
            assert "colab_mcp" not in value.lower()


# ── Description templates ─────────────────────────────────────────────


class TestDescriptionTemplates:
    @pytest.mark.parametrize("job_type", list(AppJobType))
    def test_template_renders_without_error(self, job_type):
        result = generate_template_description(
            job_type,
            dataset_name="test_dataset",
            metadata={"row_count": 100, "column_count": 5},
        )
        assert isinstance(result, str)
        assert len(result) > 50

    def test_benchmark_template_no_mlflow_branding(self):
        result = generate_template_description(
            AppJobType.BENCHMARK,
            dataset_name="sales",
            metadata={"best_model_name": "Ridge", "best_score": 0.95, "ranking_metric": "Accuracy"},
        )
        assert "MLflow" not in result
        assert "artifacts" not in result.lower()

    def test_experiment_template_no_mlflow_branding(self):
        result = generate_template_description(
            AppJobType.EXPERIMENT,
            dataset_name="churn",
            metadata={"task_type": "classification"},
        )
        assert "MLflow" not in result
        assert "artifacts" not in result.lower()
        assert "PyCaret" not in result

    def test_experiment_template_uses_current_page_names(self):
        result = generate_template_description(
            AppJobType.EXPERIMENT,
            dataset_name="churn",
            metadata={},
        )
        assert "Train & Tune" in result
        assert "Predictions" in result
        assert "Test & Evaluate" in result

    def test_benchmark_template_uses_current_page_names(self):
        result = generate_template_description(
            AppJobType.BENCHMARK,
            dataset_name="iris",
            metadata={},
        )
        assert "Quick Benchmark" in result
        assert "Train & Tune" in result
        assert "Predictions" in result

    def test_prediction_template_no_inference_jargon(self):
        result = generate_template_description(
            AppJobType.PREDICTION,
            dataset_name="forecast",
            metadata={},
        )
        assert "Inference" not in result

    def test_validation_template_no_great_expectations(self):
        result = generate_template_description(
            AppJobType.VALIDATION,
            dataset_name="raw_data",
            metadata={"passed_count": 5, "warning_count": 1, "failed_count": 0, "row_count": 100, "column_count": 10},
        )
        assert "Great Expectations" not in result

    def test_footer_uses_tracking_not_mlflow(self):
        result = generate_template_description(
            AppJobType.BENCHMARK,
            dataset_name="test",
            metadata={},
            mlflow_run_id="abc123",
        )
        assert "Tracking Run ID" in result
        assert "MLflow Run ID" not in result
