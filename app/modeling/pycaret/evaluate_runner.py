"""Evaluation helpers for plots and optional interactive evaluation."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path

from app.errors import log_exception
from app.modeling.pycaret.schemas import ExperimentEvaluationConfig, ExperimentPlotArtifact, ExperimentTaskType
from app.modeling.pycaret.selectors import supported_plots_for_task
from app.path_utils import safe_artifact_stem

logger = logging.getLogger(__name__)


def generate_evaluation_plots(
    experiment_handle,
    estimator,
    *,
    task_type: ExperimentTaskType,
    model_name: str,
    evaluation: ExperimentEvaluationConfig,
    output_dir: Path,
) -> tuple[list[ExperimentPlotArtifact], list[str]]:  # noqa: ANN001
    """Generate evaluation plot artifacts and continue past unsupported plots."""

    plot_artifacts: list[ExperimentPlotArtifact] = []
    warnings: list[str] = []
    requested_plots = evaluation.plots or []
    supported = set(supported_plots_for_task(task_type))
    output_dir.mkdir(parents=True, exist_ok=True)

    for plot_id in requested_plots:
        if plot_id not in supported:
            warnings.append(f"Plot '{plot_id}' is not supported for {task_type.value} experiments.")
            continue

        before = {path.resolve() for path in output_dir.glob("*.png")}
        try:
            with pushd(output_dir):
                returned_path = experiment_handle.plot_model(
                    estimator,
                    plot=plot_id,
                    save=True,
                    scale=evaluation.plot_scale,
                    plot_kwargs=evaluation.plot_kwargs or None,
                    verbose=False,
                )
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - exercised through tests
            log_exception(
                logger,
                exc,
                operation="pycaret.generate_plot",
                level=logging.DEBUG,
                context={"plot_id": plot_id, "model_name": model_name},
            )
            warnings.append(f"Plot '{plot_id}' could not be generated for {model_name}: {exc}")
            continue

        generated_path = _resolve_plot_path(returned_path, output_dir, before)
        if generated_path is None or not generated_path.exists():
            warnings.append(f"Plot '{plot_id}' did not produce a saved artifact for {model_name}.")
            continue

        final_name = (
            f"{safe_artifact_stem(model_name)}_{safe_artifact_stem(plot_id)}.png"
        )
        final_path = output_dir / final_name
        if final_path.exists() and final_path != generated_path:
            final_path.unlink()
        if generated_path != final_path:
            generated_path.replace(final_path)

        plot_artifacts.append(
            ExperimentPlotArtifact(
                plot_id=plot_id,
                model_name=model_name,
                path=final_path,
            )
        )

    if evaluation.interactive:
        try:
            experiment_handle.evaluate_model(estimator, plot_kwargs=evaluation.plot_kwargs or None)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - exercised through tests
            log_exception(
                logger,
                exc,
                operation="pycaret.evaluate_model",
                level=logging.DEBUG,
                context={"model_name": model_name},
            )
            warnings.append(f"Interactive evaluate_model could not be launched: {exc}")

    return plot_artifacts, warnings


@contextmanager
def pushd(path: Path):
    """Temporarily change the working directory."""

    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _resolve_plot_path(
    returned_path: str | None,
    output_dir: Path,
    before: set[Path],
) -> Path | None:
    if returned_path:
        candidate = Path(returned_path)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        if candidate.exists():
            return candidate

    after = {path.resolve() for path in output_dir.glob("*.png")}
    created = list(after - before)
    if not created:
        return None
    return max(created, key=lambda path: path.stat().st_mtime)