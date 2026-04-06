"""Tests for the data validation layer.

All tests use small local DataFrames and never call remote services.
GX tests are skipped if great_expectations is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.validation.rules import run_app_rules
from app.validation.schemas import (
    CheckResult,
    CheckSeverity,
    ValidationResultSummary,
    ValidationRuleConfig,
)
from app.validation.service import validate_dataset
from app.validation.summary import build_summary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "feature_a": [10.0, 20.0, 30.0, 40.0, 50.0],
        "feature_b": ["a", "b", "c", "d", "e"],
        "target": [0, 1, 0, 1, 0],
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


@pytest.fixture
def null_heavy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "col_a": [None, None, None, None, 1],
        "col_b": [None, None, None, None, None],
        "col_c": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def constant_df() -> pd.DataFrame:
    return pd.DataFrame({
        "const": [42, 42, 42],
        "varied": [1, 2, 3],
    })


@pytest.fixture
def duplicate_rows_df() -> pd.DataFrame:
    return pd.DataFrame({
        "a": [1, 1, 2],
        "b": ["x", "x", "y"],
    })


# ---------------------------------------------------------------------------
# Empty dataset
# ---------------------------------------------------------------------------

class TestEmptyDataset:
    def test_empty_dataframe_fails(self, empty_df: pd.DataFrame):
        config = ValidationRuleConfig()
        results = run_app_rules(empty_df, config)
        not_empty = [r for r in results if r.check_name == "dataset_not_empty"]
        assert len(not_empty) == 1
        assert not not_empty[0].passed

    def test_validate_dataset_on_empty(self, empty_df: pd.DataFrame):
        summary, _ = validate_dataset(empty_df, dataset_name="empty_test")
        assert summary.failed_count > 0
        assert summary.has_failures


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

class TestRequiredColumns:
    def test_required_columns_present(self, good_df: pd.DataFrame):
        config = ValidationRuleConfig(required_columns=["id", "target"])
        results = run_app_rules(good_df, config)
        req_checks = [r for r in results if r.check_name == "required_column_present"]
        assert len(req_checks) == 2
        assert all(r.passed for r in req_checks)

    def test_required_columns_missing(self, good_df: pd.DataFrame):
        config = ValidationRuleConfig(required_columns=["id", "nonexistent"])
        results = run_app_rules(good_df, config)
        req_checks = [r for r in results if r.check_name == "required_column_present"]
        missing = [r for r in req_checks if not r.passed]
        assert len(missing) == 1
        assert "nonexistent" in missing[0].message


# ---------------------------------------------------------------------------
# Null-heavy columns
# ---------------------------------------------------------------------------

class TestNullColumns:
    def test_fully_null_column_detected(self, null_heavy_df: pd.DataFrame):
        config = ValidationRuleConfig()
        results = run_app_rules(null_heavy_df, config)
        fully_null = [r for r in results if r.check_name == "fully_null_column"]
        assert len(fully_null) == 1
        assert "col_b" in fully_null[0].message

    def test_null_percentage_warning(self, null_heavy_df: pd.DataFrame):
        config = ValidationRuleConfig(null_warn_pct=50.0, null_fail_pct=95.0)
        results = run_app_rules(null_heavy_df, config)
        null_checks = [r for r in results if r.check_name == "null_percentage"]
        # col_a is 80% null -> should trigger warning (50-95 range)
        col_a_checks = [r for r in null_checks if r.details.get("column") == "col_a"]
        assert len(col_a_checks) == 1
        assert col_a_checks[0].severity == CheckSeverity.WARNING

    def test_null_percentage_error(self):
        df = pd.DataFrame({"almost_empty": [None] * 99 + [1]})
        config = ValidationRuleConfig(null_fail_pct=95.0)
        results = run_app_rules(df, config)
        null_checks = [r for r in results if r.check_name == "null_percentage"]
        assert any(r.severity == CheckSeverity.ERROR for r in null_checks)


# ---------------------------------------------------------------------------
# Constant columns
# ---------------------------------------------------------------------------

class TestConstantColumns:
    def test_constant_column_detected(self, constant_df: pd.DataFrame):
        config = ValidationRuleConfig()
        results = run_app_rules(constant_df, config)
        const_checks = [r for r in results if r.check_name == "constant_column"]
        assert len(const_checks) == 1
        assert "const" in const_checks[0].message


# ---------------------------------------------------------------------------
# Duplicate rows
# ---------------------------------------------------------------------------

class TestDuplicateRows:
    def test_duplicate_rows_detected(self, duplicate_rows_df: pd.DataFrame):
        config = ValidationRuleConfig()
        results = run_app_rules(duplicate_rows_df, config)
        dup_checks = [r for r in results if r.check_name == "duplicate_rows"]
        assert len(dup_checks) == 1
        assert not dup_checks[0].passed
        assert dup_checks[0].details["duplicate_count"] == 1

    def test_no_duplicates(self, good_df: pd.DataFrame):
        config = ValidationRuleConfig()
        results = run_app_rules(good_df, config)
        dup_checks = [r for r in results if r.check_name == "duplicate_rows"]
        assert all(r.passed for r in dup_checks)


# ---------------------------------------------------------------------------
# Target sanity
# ---------------------------------------------------------------------------

class TestTargetSanity:
    def test_classification_single_class(self):
        df = pd.DataFrame({"target": [1, 1, 1], "feature": [10, 20, 30]})
        config = ValidationRuleConfig(target_column="target")
        results = run_app_rules(df, config)
        target_checks = [r for r in results if r.check_name == "target_classification_sanity"]
        assert len(target_checks) == 1
        assert not target_checks[0].passed

    def test_regression_constant_target(self):
        df = pd.DataFrame({"target": [5.0, 5.0, 5.0], "feature": [1, 2, 3]})
        config = ValidationRuleConfig(target_column="target")
        results = run_app_rules(df, config)
        reg_checks = [r for r in results if r.check_name == "target_regression_sanity"]
        assert len(reg_checks) == 1
        assert not reg_checks[0].passed

    def test_target_missing(self):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        config = ValidationRuleConfig(target_column="missing_target")
        results = run_app_rules(df, config)
        exists_checks = [r for r in results if r.check_name == "target_column_exists"]
        assert len(exists_checks) == 1
        assert not exists_checks[0].passed

    def test_target_all_null(self):
        df = pd.DataFrame({"target": [None, None, None], "feature": [1, 2, 3]})
        config = ValidationRuleConfig(target_column="target")
        results = run_app_rules(df, config)
        null_target = [r for r in results if r.check_name == "target_not_null"]
        assert len(null_target) == 1
        assert not null_target[0].passed


# ---------------------------------------------------------------------------
# Summary construction
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_counts(self, good_df: pd.DataFrame):
        checks = [
            CheckResult(check_name="a", passed=True, severity=CheckSeverity.INFO, message="ok"),
            CheckResult(check_name="b", passed=False, severity=CheckSeverity.WARNING, message="warn"),
            CheckResult(check_name="c", passed=False, severity=CheckSeverity.ERROR, message="fail"),
        ]
        summary = build_summary(checks, good_df, dataset_name="test")
        assert summary.passed_count == 1
        assert summary.warning_count == 1
        assert summary.failed_count == 1
        assert summary.total_checks == 3
        assert summary.has_failures


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_default_rule_config(self):
        config = ValidationRuleConfig()
        assert config.min_row_count == 1
        assert config.null_warn_pct == 50.0
        assert config.null_fail_pct == 95.0
        assert config.enable_leakage_heuristics is True

    def test_validation_settings_defaults(self):
        from app.config.models import ValidationSettings
        settings = ValidationSettings()
        assert settings.artifacts_dir == Path("artifacts/validation")
        assert settings.data_docs_enabled is False


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

class TestArtifacts:
    def test_write_artifacts(self, good_df: pd.DataFrame, tmp_path: Path):
        config = ValidationRuleConfig(target_column="target")
        summary, bundle = validate_dataset(
            good_df, config, dataset_name="test_ds", artifacts_dir=tmp_path
        )
        assert bundle is not None
        assert bundle.summary_json_path is not None
        assert bundle.summary_json_path.exists()
        assert bundle.markdown_report_path is not None
        assert bundle.markdown_report_path.exists()

        # Verify JSON is parseable
        data = json.loads(bundle.summary_json_path.read_text())
        assert data["dataset_name"] == "test_ds"
        assert "checks" in data


# ---------------------------------------------------------------------------
# Leakage heuristics
# ---------------------------------------------------------------------------

class TestLeakageHeuristics:
    def test_name_similarity_detected(self):
        df = pd.DataFrame({
            "price": [10, 20, 30],
            "price_predicted": [11, 21, 31],
            "feature": [1, 2, 3],
        })
        config = ValidationRuleConfig(target_column="price", enable_leakage_heuristics=True)
        results = run_app_rules(df, config)
        leakage = [r for r in results if r.check_name == "leakage_name_similarity"]
        assert any("price_predicted" in r.details.get("column", "") for r in leakage)

    def test_id_like_column_detected(self):
        df = pd.DataFrame({
            "row_id": [1, 2, 3],
            "target": [0, 1, 0],
            "feature": [10, 20, 30],
        })
        config = ValidationRuleConfig(target_column="target", enable_leakage_heuristics=True)
        results = run_app_rules(df, config)
        id_like = [r for r in results if r.check_name == "leakage_id_like"]
        assert any("row_id" in r.details.get("column", "") for r in id_like)

    def test_underscore_only_column_not_false_positive(self):
        """Column names that strip to empty (e.g. '__') must not trigger leakage."""
        df = pd.DataFrame({
            "__": [0.1, 0.2, 0.3],
            "target": [0, 1, 0],
            "feature": [10, 20, 30],
        })
        config = ValidationRuleConfig(target_column="target", enable_leakage_heuristics=True)
        results = run_app_rules(df, config)
        leakage = [r for r in results if r.check_name == "leakage_name_similarity"]
        assert not any("__" in r.details.get("column", "") for r in leakage)


# ---------------------------------------------------------------------------
# Expectation-style fallback checks
# ---------------------------------------------------------------------------

class TestExpectationFallbackChecks:
    def test_numeric_range_check_detects_out_of_bounds(self):
        df = pd.DataFrame({"score": [10, 20, 999]})
        config = ValidationRuleConfig(
            numeric_range_checks={"score": {"min": 0, "max": 100}}
        )

        results = run_app_rules(df, config)
        range_checks = [r for r in results if r.check_name == "numeric_range_check"]

        assert len(range_checks) == 1
        assert not range_checks[0].passed
        assert range_checks[0].details["invalid_count"] == 1

    def test_allowed_category_check_detects_unexpected_values(self):
        df = pd.DataFrame({"status": ["active", "inactive", "unknown"]})
        config = ValidationRuleConfig(
            allowed_category_checks={"status": ["active", "inactive"]}
        )

        results = run_app_rules(df, config)
        category_checks = [r for r in results if r.check_name == "allowed_category_check"]

        assert len(category_checks) == 1
        assert not category_checks[0].passed
        assert category_checks[0].details["unexpected_values"] == ["unknown"]


# ---------------------------------------------------------------------------
# Mixed types
# ---------------------------------------------------------------------------

class TestMixedTypes:
    def test_mixed_types_warning(self):
        df = pd.DataFrame({"mixed": [1, "two", 3.0, None, "five"]})
        config = ValidationRuleConfig()
        results = run_app_rules(df, config)
        mixed = [r for r in results if r.check_name == "mixed_types"]
        assert len(mixed) == 1
        assert not mixed[0].passed


# ---------------------------------------------------------------------------
# GX builder specs
# ---------------------------------------------------------------------------

class TestGXBuilders:
    def test_build_expectations_basic(self):
        from app.validation.gx_builders import build_expectations

        config = ValidationRuleConfig(
            min_row_count=5,
            required_columns=["a", "b"],
            target_column="b",
            uniqueness_columns=["a"],
            numeric_range_checks={"score": {"min": 0, "max": 100}},
            allowed_category_checks={"status": ["active", "inactive"]},
        )
        cols = ["a", "b", "score", "status"]
        specs = build_expectations(config, cols)
        types = [s["type"] for s in specs]
        assert "expect_column_values_to_be_unique" in types
        assert "expect_column_values_to_be_between" in types
        assert "expect_column_values_to_be_in_set" in types

    def test_gx_managed_rules_only_emit_prereq_warnings(self):
        df = pd.DataFrame({"a": [1, 2, 2]})
        config = ValidationRuleConfig(
            uniqueness_columns=["a", "missing_col"],
            numeric_range_checks={"score": {"min": 0, "max": 100}},
            allowed_category_checks={"status": ["active"]},
        )

        results = run_app_rules(df, config, gx_managed_rules=True)

        uniqueness = [r for r in results if r.check_name == "uniqueness_check"]
        numeric = [r for r in results if r.check_name == "numeric_range_check"]
        allowed = [r for r in results if r.check_name == "allowed_category_check"]

        assert len(uniqueness) == 1
        assert uniqueness[0].details["column"] == "missing_col"
        assert len(numeric) == 1
        assert numeric[0].details["column"] == "score"
        assert len(allowed) == 1
        assert allowed[0].details["column"] == "status"


class TestArtifactStemSanitization:
    def test_validation_artifacts_sanitize_dataset_name(self, good_df: pd.DataFrame, tmp_path: Path):
        summary, bundle = validate_dataset(
            good_df,
            dataset_name="folder/unsafe:name",
            artifacts_dir=tmp_path,
        )

        assert summary.dataset_name == "folder/unsafe:name"
        assert bundle is not None
        assert bundle.summary_json_path is not None
        assert "unsafe" in bundle.summary_json_path.name
        assert "/" not in bundle.summary_json_path.name
        assert ":" not in bundle.summary_json_path.name
