"""tune_model wrapper."""

from __future__ import annotations

from typing import Any

from app.modeling.pycaret.schemas import ExperimentTuneConfig


def run_tune_model(
    experiment_handle,
    estimator,
    config: ExperimentTuneConfig,
    *,
    optimize_metric: str,
) -> tuple[Any, Any | None]:  # noqa: ANN001
    """Execute tune_model and optionally capture the tuner object."""

    result = experiment_handle.tune_model(
        estimator,
        fold=config.fold,
        n_iter=config.n_iter,
        custom_grid=config.custom_grid,
        optimize=optimize_metric,
        search_library=config.search_library,
        search_algorithm=config.search_algorithm,
        early_stopping=config.early_stopping,
        early_stopping_max_iters=config.early_stopping_max_iters,
        choose_better=config.choose_better,
        fit_kwargs=config.fit_kwargs or None,
        return_tuner=True,
        verbose=False,
        tuner_verbose=False,
    )

    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, None