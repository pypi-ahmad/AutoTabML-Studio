"""Tests for the profiling layer.

GX and ydata-profiling are NOT required for these tests.
Tests that need ydata-profiling are skipped if the library is unavailable.
"""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import pandas as pd
import pytest

from app.config.models import ProfilingMode, ProfilingSettings
from app.profiling.schemas import ProfilingConfig, ProfilingResultSummary
from app.profiling.selectors import maybe_sample, select_profiling_mode
from app.profiling.ydata_runner import profiling_install_guidance

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "num_a": [1, 2, 3, 4, 5],
        "num_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["a", "b", "c", "d", "e"],
    })


@pytest.fixture
def large_df() -> pd.DataFrame:
    """DataFrame that exceeds default thresholds."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 60_000
    return pd.DataFrame({
        "value": rng.standard_normal(n),
        "category": rng.choice(["a", "b", "c"], n),
    })


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

class TestModeSelection:
    def test_standard_mode_for_small_dataset(self, small_df: pd.DataFrame):
        config = ProfilingConfig(mode=ProfilingMode.STANDARD)
        mode = select_profiling_mode(small_df, config)
        assert mode == ProfilingMode.STANDARD

    def test_minimal_mode_for_large_rows(self, large_df: pd.DataFrame):
        config = ProfilingConfig(
            mode=ProfilingMode.STANDARD,
            large_dataset_row_threshold=50_000,
        )
        mode = select_profiling_mode(large_df, config)
        assert mode == ProfilingMode.MINIMAL

    def test_minimal_mode_for_wide_dataset(self):
        # 3 rows but 120 columns
        df = pd.DataFrame({f"col_{i}": [1, 2, 3] for i in range(120)})
        config = ProfilingConfig(
            mode=ProfilingMode.STANDARD,
            large_dataset_col_threshold=100,
        )
        mode = select_profiling_mode(df, config)
        assert mode == ProfilingMode.MINIMAL


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_no_sampling_under_threshold(self, small_df: pd.DataFrame):
        config = ProfilingConfig(sampling_row_threshold=200_000)
        result_df, was_sampled, sample_size = maybe_sample(small_df, config)
        assert not was_sampled
        assert sample_size is None
        assert len(result_df) == len(small_df)

    def test_sampling_applied_over_threshold(self):
        import numpy as np
        rng = np.random.default_rng(42)
        big = pd.DataFrame({"v": rng.standard_normal(300_000)})
        config = ProfilingConfig(
            sampling_row_threshold=200_000,
            sample_size=50_000,
        )
        result_df, was_sampled, sample_size = maybe_sample(big, config)
        assert was_sampled
        assert sample_size == 50_000
        assert len(result_df) == 50_000


# ---------------------------------------------------------------------------
# Profiling config / settings defaults
# ---------------------------------------------------------------------------

class TestProfilingConfig:
    def test_default_profiling_config(self):
        config = ProfilingConfig()
        assert config.mode == ProfilingMode.STANDARD
        assert config.large_dataset_row_threshold == 50_000
        assert config.sample_size == 50_000

    def test_default_profiling_settings(self):
        settings = ProfilingSettings()
        assert settings.artifacts_dir == Path("artifacts/profiling")
        assert settings.default_mode == ProfilingMode.STANDARD
        assert settings.large_dataset_row_threshold == 50_000

    def test_profiling_install_guidance_pins_compatible_setuptools(self):
        guidance = profiling_install_guidance()

        assert 'setuptools<82' in guidance

    def test_profiling_install_guidance_mentions_pkg_resources_when_missing(self):
        guidance = profiling_install_guidance(ModuleNotFoundError("No module named 'pkg_resources'"))

        assert 'pkg_resources' in guidance
        assert 'setuptools<82' in guidance


class TestYDataProfilingNoiseSuppression:
    def test_generate_report_disables_progress_bar(self, monkeypatch, small_df: pd.DataFrame):
        from app.profiling.ydata_runner import YDataProfilingService

        captured_kwargs: dict[str, object] = {}

        class _FakeReport:
            pass

        def _fake_profile_report(df, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeReport()

        monkeypatch.setitem(
            sys.modules,
            "ydata_profiling",
            types.SimpleNamespace(ProfileReport=_fake_profile_report),
        )

        service = YDataProfilingService()
        report = service._generate_report(small_df, ProfilingMode.MINIMAL, "demo")

        assert isinstance(report, _FakeReport)
        assert captured_kwargs["progress_bar"] is False
        assert captured_kwargs["minimal"] is True
        assert captured_kwargs["title"] == "demo"

    def test_known_dependency_warnings_are_suppressed(self):
        from pyparsing.warnings import PyparsingDeprecationWarning

        from app.profiling.ydata_runner import _suppress_profiling_runtime_noise

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _suppress_profiling_runtime_noise():
                warnings.warn("'parseString' deprecated - use 'parse_string'", PyparsingDeprecationWarning)
                warnings.warn(
                    "datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version.",
                    DeprecationWarning,
                )
                warnings.warn("keep-me", UserWarning)

        assert len(caught) == 1
        assert str(caught[0].message) == "keep-me"

    def test_runtime_noise_context_suppresses_stdout_and_stderr(self, capsys):
        from app.profiling.ydata_runner import _suppress_profiling_runtime_noise

        with _suppress_profiling_runtime_noise():
            print("stdout-noise")
            print("stderr-noise", file=sys.stderr)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_write_artifacts_html_error_logs_exception(self, tmp_path, monkeypatch, small_df):
        """Regression: the HTML write error handler must bind 'exc' to avoid NameError."""
        from app.profiling.ydata_runner import YDataProfilingService

        service = YDataProfilingService(artifacts_dir=tmp_path)

        class _ExplodingReport:
            def to_file(self, path):
                raise OSError("disk full")

        summary = ProfilingResultSummary(
            row_count=3, column_count=2,
        )

        # Should log a warning, not crash with NameError
        bundle = service._write_artifacts(_ExplodingReport(), summary, "test_ds")
        assert bundle.html_report_path is None


# ---------------------------------------------------------------------------
# Summary schema
# ---------------------------------------------------------------------------

class TestProfilingSummary:
    def test_summary_fields(self):
        summary = ProfilingResultSummary(
            row_count=100,
            column_count=5,
            numeric_column_count=3,
            categorical_column_count=2,
            missing_cells_total=10,
            missing_cells_pct=2.0,
            duplicate_row_count=0,
            duplicate_row_pct=0.0,
            memory_bytes=1024,
            report_mode=ProfilingMode.STANDARD,
            sampling_applied=False,
        )
        assert summary.row_count == 100
        assert summary.report_mode == ProfilingMode.STANDARD
        assert not summary.sampling_applied

    def test_summary_with_sampling(self):
        summary = ProfilingResultSummary(
            row_count=300_000,
            column_count=10,
            sampling_applied=True,
            sample_size_used=50_000,
            report_mode=ProfilingMode.MINIMAL,
        )
        assert summary.sampling_applied
        assert summary.sample_size_used == 50_000


# ---------------------------------------------------------------------------
# Summary extraction (direct DataFrame fallback)
# ---------------------------------------------------------------------------

class TestSummaryExtraction:
    def test_extract_summary_fallback(self, small_df: pd.DataFrame):
        """Test that summary extraction works using direct DataFrame analysis."""
        from app.profiling.summary import extract_summary

        # Pass None as report to force fallback
        summary = extract_summary(
            None,
            small_df,
            effective_mode=ProfilingMode.STANDARD,
            was_sampled=False,
            sample_size_used=None,
            dataset_name="test",
        )
        assert summary.row_count == 5
        assert summary.column_count == 3
        assert summary.numeric_column_count == 2
        assert summary.categorical_column_count == 1
        assert summary.missing_cells_total == 0
        assert summary.dataset_name == "test"

    def test_extract_summary_uses_full_dataset_metrics_even_when_report_is_sampled(self):
        from app.profiling.summary import extract_summary

        full_df = pd.DataFrame(
            {
                "value": [1, 1, 2, 3],
                "category": ["a", "b", "c", None],
            }
        )

        class _FakeSampledReport:
            def get_description(self):
                return {
                    "table": {
                        "n_cells_missing": 0,
                        "n_duplicates": 0,
                        "types": {"Numeric": 99, "Categorical": 99},
                    }
                }

        summary = extract_summary(
            _FakeSampledReport(),
            full_df,
            effective_mode=ProfilingMode.MINIMAL,
            was_sampled=True,
            sample_size_used=2,
            dataset_name="sampled",
        )

        assert summary.row_count == 4
        assert summary.missing_cells_total == 1
        assert summary.duplicate_row_count == 0
        assert summary.numeric_column_count == 1
        assert summary.categorical_column_count == 1
        assert summary.sampling_applied


# ---------------------------------------------------------------------------
# Artifact path handling
# ---------------------------------------------------------------------------

class TestArtifactPaths:
    def test_artifacts_dir_from_settings(self):
        settings = ProfilingSettings(artifacts_dir=Path("/tmp/my_profiles"))
        assert settings.artifacts_dir == Path("/tmp/my_profiles")

    def test_profiling_result_bundle_defaults(self):
        from app.profiling.schemas import ProfilingArtifactBundle
        bundle = ProfilingArtifactBundle()
        assert bundle.html_report_path is None
        assert bundle.summary_json_path is None
