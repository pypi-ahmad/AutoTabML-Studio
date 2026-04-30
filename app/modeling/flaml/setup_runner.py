"""FLAML availability checks and task type resolution."""

from __future__ import annotations

import logging
import sys

import pandas as pd

from app.errors import log_exception
from app.modeling.benchmark.schemas import BenchmarkTaskType
from app.modeling.benchmark.selectors import infer_task_type as benchmark_infer_task_type
from app.modeling.benchmark.selectors import validate_target as benchmark_validate_target
from app.modeling.flaml.errors import FlamlDependencyError
from app.modeling.flaml.schemas import FlamlTaskType

logger = logging.getLogger(__name__)


def _probe_flaml_import_error() -> Exception | None:
    """Return the import-time failure when FLAML is unusable."""

    try:
        from flaml import AutoML  # noqa: F401

        return None
    except Exception as exc:
        log_exception(logger, exc, operation="flaml.probe_import", level=logging.DEBUG)
        return exc


def is_flaml_available() -> bool:
    """Return True when FLAML is importable."""

    return _probe_flaml_import_error() is None


def flaml_install_guidance() -> str:
    """Return a user-facing installation hint for environments without FLAML."""

    message = (
        "FLAML is not available in this environment. "
        "Install it with: `pip install flaml[automl]`"
    )
    if sys.version_info >= (3, 13):
        message += (
            "\n\n**Note:** FLAML may not be compatible with your current Python version. "
            "Check the FLAML documentation for supported versions."
        )
    return message


def require_flaml() -> None:
    """Raise a clean dependency error when FLAML is unavailable."""

    if is_flaml_available():
        return

    import_error = _probe_flaml_import_error()
    if import_error is None:
        return

    guidance = flaml_install_guidance()
    raw_detail = import_error.args if import_error.args else [str(import_error)]
    detail = " ".join(str(part) for part in raw_detail if str(part).strip())
    if detail:
        raise FlamlDependencyError(f"{guidance} Root cause: {detail}") from import_error
    raise FlamlDependencyError(guidance) from import_error


def resolve_task_type(
    target: pd.Series,
    requested_task_type: FlamlTaskType,
) -> tuple[FlamlTaskType, list[str]]:
    """Resolve the effective FLAML task type and validate the target."""

    warnings: list[str] = []
    non_null_target = target.dropna()

    if requested_task_type == FlamlTaskType.AUTO:
        inferred = benchmark_infer_task_type(non_null_target)
        task_type = _map_from_benchmark_task_type(inferred)
        warnings.append(f"Task type auto-detected as {task_type.value}.")
    else:
        task_type = requested_task_type

    warnings.extend(
        benchmark_validate_target(
            non_null_target,
            _map_to_benchmark_task_type(task_type),
        )
    )
    return task_type, warnings


def _map_to_benchmark_task_type(task_type: FlamlTaskType) -> BenchmarkTaskType:
    if task_type == FlamlTaskType.CLASSIFICATION:
        return BenchmarkTaskType.CLASSIFICATION
    if task_type == FlamlTaskType.REGRESSION:
        return BenchmarkTaskType.REGRESSION
    return BenchmarkTaskType.AUTO


def _map_from_benchmark_task_type(task_type: BenchmarkTaskType) -> FlamlTaskType:
    if task_type == BenchmarkTaskType.CLASSIFICATION:
        return FlamlTaskType.CLASSIFICATION
    if task_type == BenchmarkTaskType.REGRESSION:
        return FlamlTaskType.REGRESSION
    return FlamlTaskType.AUTO
