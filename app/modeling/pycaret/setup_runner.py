"""PyCaret setup helpers and experiment construction."""

from __future__ import annotations

import logging
import sys
from typing import Any

from app.errors import log_exception
from app.gpu import resolve_use_gpu
from app.modeling.pycaret.errors import PyCaretDependencyError, UnsupportedExperimentTaskError
from app.modeling.pycaret.schemas import ExperimentConfig, ExperimentTaskType, MLflowTrackingMode

logger = logging.getLogger(__name__)


def _probe_pycaret_import_error() -> Exception | None:
    """Return the import-time failure when PyCaret is unusable in the current runtime."""

    try:
        import pycaret.classification  # noqa: F401
        import pycaret.regression  # noqa: F401
        return None
    except Exception as exc:  # pragma: no cover - exercised through public helpers
        log_exception(logger, exc, operation="pycaret.probe_import", level=logging.DEBUG)
        return exc


def _format_pycaret_import_error(exc: Exception) -> str:
    """Normalize import-time dependency failures into a stable one-line message."""

    if exc.args:
        return " ".join(str(part) for part in exc.args if str(part).strip())
    return str(exc)


def is_pycaret_available() -> bool:
    """Return True when PyCaret classification and regression modules are importable."""

    return _probe_pycaret_import_error() is None


def build_pycaret_experiment(task_type: ExperimentTaskType) -> Any:
    """Instantiate the correct OOP experiment class for the resolved task."""

    if task_type == ExperimentTaskType.CLASSIFICATION:
        from pycaret.classification import ClassificationExperiment

        return ClassificationExperiment()
    if task_type == ExperimentTaskType.REGRESSION:
        from pycaret.regression import RegressionExperiment

        return RegressionExperiment()
    raise UnsupportedExperimentTaskError(f"Unsupported experiment task type: {task_type.value}.")


def build_setup_call_kwargs(
    config: ExperimentConfig,
    *,
    task_type: ExperimentTaskType,
    mlflow_experiment_name: str | None = None,
    test_data_supplied: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return setup() kwargs and a normalized audit copy without raw data."""

    setup = config.setup
    actual: dict[str, Any] = {
        "target": config.target_column,
        "session_id": setup.session_id,
        "train_size": None if test_data_supplied else setup.train_size,
        "fold": setup.fold,
        "fold_strategy": setup.fold_strategy
        or ("stratifiedkfold" if task_type == ExperimentTaskType.CLASSIFICATION else "kfold"),
        "numeric_features": list(setup.numeric_features),
        "categorical_features": list(setup.categorical_features),
        "date_features": list(setup.date_features),
        "ignore_features": list(setup.ignore_features),
        "test_data_supplied": test_data_supplied,
        "preprocess": setup.preprocess,
        "html": False,
        "verbose": False,
        "system_log": setup.system_log,
        "n_jobs": setup.n_jobs,
        "use_gpu": resolve_use_gpu(setup.use_gpu),
        "experiment_name": setup.experiment_name,
        "log_experiment": False,
        "log_plots": False,
        "log_profile": False,
        "log_data": False,
    }

    if config.mlflow_tracking_mode == MLflowTrackingMode.PYCARET_NATIVE and setup.log_experiment:
        actual["log_experiment"] = "mlflow"
        actual["experiment_name"] = setup.experiment_name or mlflow_experiment_name
        actual["log_plots"] = setup.log_plots
        actual["log_profile"] = setup.log_profile
        actual["log_data"] = setup.log_data

    # Keys that belong in the audit record but are NOT valid PyCaret setup() params.
    _audit_only_keys = {"test_data_supplied"}

    call_kwargs = {
        key: value
        for key, value in actual.items()
        if value is not None and value != [] and key not in _audit_only_keys
    }
    return call_kwargs, actual


def require_pycaret() -> None:
    """Raise a clean dependency error when PyCaret is unavailable."""

    if is_pycaret_available():
        return

    import_error = _probe_pycaret_import_error()
    if import_error is None:
        return

    guidance = pycaret_install_guidance()
    detail = _format_pycaret_import_error(import_error)
    if detail:
        raise PyCaretDependencyError(f"{guidance} Root cause: {detail}") from import_error
    raise PyCaretDependencyError(guidance) from import_error


def pycaret_install_guidance() -> str:
    """Return a user-facing installation hint for environments without PyCaret."""

    message = (
        "The model training engine is not available in this environment. "
        "Please ask your administrator to install the required training packages."
    )
    if sys.version_info >= (3, 13):
        message += (
            "\n\n**Note:** The training engine may not be compatible with your current Python version. "
            "Your administrator may need to use a different Python version."
        )
    return message