"""compare_models and create_model wrappers."""

from __future__ import annotations

from typing import Any

from app.modeling.pycaret.schemas import ExperimentCompareConfig


def run_compare_models(experiment_handle, config: ExperimentCompareConfig, *, optimize_metric: str) -> Any:  # noqa: ANN001
    """Execute compare_models with explicit, testable kwargs."""

    include_models = config.include_models or None
    exclude_models = config.exclude_models or None
    fit_kwargs = config.fit_kwargs or None
    return experiment_handle.compare_models(
        include=include_models,
        exclude=exclude_models,
        sort=optimize_metric,
        n_select=config.n_select,
        budget_time=config.budget_time,
        turbo=config.turbo,
        errors=config.errors,
        cross_validation=config.cross_validation,
        fit_kwargs=fit_kwargs,
        verbose=False,
    )


def create_model(experiment_handle, estimator_id: str, *, fit_kwargs: dict[str, Any] | None = None):  # noqa: ANN001
    """Create one concrete model from a model id."""

    return experiment_handle.create_model(
        estimator_id,
        fit_kwargs=fit_kwargs or None,
        verbose=False,
    )