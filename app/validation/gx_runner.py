"""Execute Great Expectations validation against a pandas DataFrame.

This module wraps the modern GX Core API (>=1.0) for in-memory pandas
validation.  All GX imports are contained here and in gx_context.py /
gx_builders.py so the rest of the app never depends on GX directly.

Design notes:
- Uses ephemeral context for in-memory validation (no disk state needed).
- Expectation suite is built per-run from the builder specs.
- Results are translated into our CheckResult schema immediately.
- Spark support can be added later by creating a parallel runner that
  swaps the pandas data source for a Spark data source.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.validation.gx_builders import ExpectationSpec, build_expectations
from app.validation.gx_context import get_ephemeral_context, is_gx_available
from app.validation.schemas import CheckResult, CheckSeverity, ValidationRuleConfig

logger = logging.getLogger(__name__)


def run_gx_validation(
    df: pd.DataFrame,
    config: ValidationRuleConfig,
) -> list[CheckResult]:
    """Run GX expectations and return normalized CheckResult list.

    If GX is not installed, returns an empty list (app-level rules still run).
    """
    if not is_gx_available():
        logger.info("great_expectations not installed – skipping GX checks.")
        return []

    column_names = list(df.columns)
    specs = build_expectations(config, column_names)
    if not specs:
        return []

    try:
        return _execute_specs(df, specs)
    except Exception as exc:
        logger.warning("GX validation failed - falling back to app-only checks: %s", exc)
        return [
            CheckResult(
                check_name="gx_execution",
                passed=False,
                severity=CheckSeverity.WARNING,
                message="Great Expectations validation encountered an error. App-level checks still ran.",
                source="gx",
            )
        ]


def _execute_specs(df: pd.DataFrame, specs: list[ExpectationSpec]) -> list[CheckResult]:
    """Build an ephemeral suite, run it, and convert results."""
    import great_expectations as gx  # noqa: WPS433

    context = get_ephemeral_context()

    # Modern GX Core API: add pandas data source → DataFrame asset → batch
    data_source = context.data_sources.add_pandas(name="autotabml_ds")
    data_asset = data_source.add_dataframe_asset(name="validation_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name="validation_batch",
    )

    # Build expectation suite
    suite = context.suites.add(
        gx.ExpectationSuite(name="autotabml_validation_suite")
    )
    for spec in specs:
        # NOTE(gx-api-compat): ``spec["type"]`` is resolved to the current GX
        # expectation class via ``gx.expectations``.  If a future GX release
        # changes the constructor API, update ``_resolve_expectation_class``.
        expectation_class = _resolve_expectation_class(gx, spec["type"])
        if expectation_class is None:
            logger.warning("Unknown GX expectation type: %s – skipping.", spec["type"])
            continue
        suite.add_expectation(expectation_class(**spec["kwargs"]))

    # Run validation
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="autotabml_validation",
            data=batch_definition,
            suite=suite,
        )
    )
    results = validation_definition.run(batch_parameters={"dataframe": df})

    return _convert_results(results)


def _resolve_expectation_class(gx: Any, expectation_type: str) -> Any:
    """Resolve a GX expectation type name to its exported expectation class."""

    class_name = "".join(part.capitalize() for part in expectation_type.split("_"))
    return getattr(gx.expectations, class_name, None)


def _convert_results(results: Any) -> list[CheckResult]:
    """Translate GX validation results into CheckResult objects."""
    checks: list[CheckResult] = []

    try:
        for result in results.results:
            expectation_type = result.expectation_config.type
            success = result.success

            # Build a readable message
            kwargs_repr = dict(result.expectation_config.kwargs) if hasattr(result.expectation_config, 'kwargs') else {}
            msg = f"GX: {expectation_type}"
            if "column" in kwargs_repr:
                msg += f" on column '{kwargs_repr['column']}'"
            msg += " – PASSED" if success else " – FAILED"

            details: dict[str, Any] = {"expectation_type": expectation_type}
            if hasattr(result, "result") and isinstance(result.result, dict):
                # Include observed value if present
                for key in ("observed_value", "element_count", "unexpected_count", "unexpected_percent"):
                    if key in result.result:
                        details[key] = result.result[key]

            checks.append(
                CheckResult(
                    check_name=expectation_type,
                    passed=success,
                    severity=CheckSeverity.ERROR if not success else CheckSeverity.INFO,
                    message=msg,
                    details=details,
                    source="gx",
                )
            )
    except Exception as exc:
        logger.warning("Error converting GX results: %s", exc)
        checks.append(
            CheckResult(
                check_name="gx_result_parse",
                passed=False,
                severity=CheckSeverity.WARNING,
                message="Could not parse GX validation results.",
                source="gx",
            )
        )

    return checks
