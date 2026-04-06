"""Central registry for Streamlit page navigation and rendering."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from app.config.enums import WorkspaceMode


@dataclass(frozen=True)
class PageSpec:
    """Declarative Streamlit page registration."""

    label: str
    description: str
    module_path: str
    render_function: str


_PAGES = [
    PageSpec("Dashboard", "Workspace overview and recent local activity.", "app.pages.dashboard_page", "render_dashboard_page"),
    PageSpec("Dataset Intake", "Load, preview, and activate datasets for downstream workflows.", "app.pages.dataset_intake_page", "render_dataset_intake_page"),
    PageSpec("Validation", "Run dataset validation checks and inspect local reports.", "app.pages.validation_page", "render_validation_page"),
    PageSpec("Profiling", "Generate local EDA and profiling summaries.", "app.pages.profiling_page", "render_profiling_page"),
    PageSpec("Benchmark", "Run baseline local benchmark comparisons.", "app.pages.benchmark_page", "render_benchmark_page"),
    PageSpec("Experiment", "Run PyCaret compare, tune, evaluate, and save workflows.", "app.pages.experiment_page", "render_experiment_page"),
    PageSpec("Prediction", "Load local or MLflow-backed models and score rows locally.", "app.pages.prediction_page", "render_prediction_page"),
    PageSpec("History", "Browse MLflow-backed run history and details.", "app.pages.history_page", "render_history_page"),
    PageSpec("Compare", "Compare two MLflow runs side by side.", "app.pages.compare_page", "render_compare_page"),
    PageSpec("Registry", "Inspect and promote MLflow registry models.", "app.pages.registry_page", "render_registry_page"),
    PageSpec("Notebook", "Execute notebooks on Google Colab (default) or locally.", "app.pages.notebook_page", "render_notebook_page"),
    PageSpec("Settings", "Configure local runtime, providers, and execution defaults.", "app.pages.settings_page", "render_settings_page"),
]


def get_page_registry() -> list[PageSpec]:
    """Return the registered Streamlit pages in display order."""

    return list(_PAGES)


def default_page_label(workspace_mode: WorkspaceMode) -> str:
    """Return the default page label for the selected workspace mode."""

    return "Notebook" if workspace_mode == WorkspaceMode.NOTEBOOK else "Dashboard"


def get_page_by_label(label: str) -> PageSpec:
    """Resolve one page from the registry by its label."""

    for page in _PAGES:
        if page.label == label:
            return page
    raise KeyError(f"Unknown page label: {label}")


def render_registered_page(label: str) -> None:
    """Import and render one registered page."""

    page = get_page_by_label(label)
    module = importlib.import_module(page.module_path)
    getattr(module, page.render_function)()