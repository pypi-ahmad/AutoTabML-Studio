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
    section: str = ""


# ── Sections keep the sidebar organised by stage ──────────────────────
NAV_SECTIONS: list[tuple[str, str]] = [
    ("start", "Start"),
    ("prepare", "① Prepare"),
    ("build", "② Build"),
    ("use", "③ Use"),
    ("review", "Review"),
    ("admin", "Admin"),
]

_PAGES = [
    # ── Start ──────────────────────────────────────────────────────────
    PageSpec("Home", "Your local-first workspace dashboard — see recent activity and jump to any workflow.", "app.pages.dashboard_page", "render_dashboard_page", section="start"),
    # ── Prepare ────────────────────────────────────────────────────────
    PageSpec("Load Data", "Upload or connect a dataset (CSV, Excel, URL, or public repository).", "app.pages.dataset_intake_page", "render_dataset_intake_page", section="prepare"),
    PageSpec("Validation", "Optional — Check your data for missing values, duplicates, and errors before modeling.", "app.pages.validation_page", "render_validation_page", section="prepare"),
    PageSpec("Profiling", "Optional — Visual summary of your data: distributions, correlations, and statistics.", "app.pages.profiling_page", "render_profiling_page", section="prepare"),
    # ── Build ──────────────────────────────────────────────────────────
    PageSpec("Quick Benchmark", "Quickly test dozens of algorithms to find a shortlist — no tuning, just a fast baseline.", "app.pages.benchmark_page", "render_benchmark_page", section="build"),
    PageSpec("Train & Tune", "Train, fine-tune, and save a production-ready model.", "app.pages.experiment_page", "render_experiment_page", section="build"),
    PageSpec("FLAML AutoML", "Automatic model selection and tuning powered by Microsoft FLAML.", "app.pages.flaml_automl_page", "render_flaml_automl_page", section="build"),
    # ── Use ────────────────────────────────────────────────────────────
    PageSpec("Predictions", "Make predictions on new data, or test a model against ground truth.", "app.pages.predictions_page", "render_predictions_page", section="use"),
    # ── Review ─────────────────────────────────────────────────────────
    PageSpec("Models", "Browse all your saved models in one place.", "app.pages.models_page", "render_models_page", section="review"),
    PageSpec("History", "View the full history of every job and dataset run.", "app.pages.history_page", "render_history_page", section="review"),
    PageSpec("Compare", "Compare algorithm performance side by side for any dataset.", "app.pages.compare_page", "render_compare_page", section="review"),
    PageSpec("Notebook", "Generate or run reproducible notebooks for your datasets and runs.", "app.pages.notebook_page", "render_notebook_page", section="review"),
    # ── Admin ──────────────────────────────────────────────────────────
    PageSpec("Registry", "Version and manage your best models for deployment.", "app.pages.registry_page", "render_registry_page", section="admin"),
    PageSpec("Settings", "Configure providers, execution, and workspace preferences.", "app.pages.settings_page", "render_settings_page", section="admin"),
]


def get_page_registry() -> list[PageSpec]:
    """Return the registered Streamlit pages in display order."""

    return list(_PAGES)


def get_nav_sections() -> list[tuple[str, list[PageSpec]]]:
    """Return pages grouped by section, preserving section order.

    Returns a list of ``(section_display_name, [PageSpec, ...])``.
    """
    section_lookup = {key: display for key, display in NAV_SECTIONS}
    groups: dict[str, list[PageSpec]] = {key: [] for key, _ in NAV_SECTIONS}
    for page in _PAGES:
        groups.setdefault(page.section, []).append(page)
    return [(section_lookup.get(key, key), pages) for key, pages in groups.items() if pages]


def default_page_label(workspace_mode: WorkspaceMode) -> str:
    """Return the default page label for the selected workspace mode."""

    return "Notebook" if workspace_mode == WorkspaceMode.NOTEBOOK else "Home"


# ── Compatibility aliases for renamed/merged pages ─────────────────────
_LABEL_ALIASES: dict[str, str] = {
    "Prediction": "Predictions",
    "Model Testing": "Predictions",
}


def get_page_by_label(label: str) -> PageSpec:
    """Resolve one page from the registry by its label.

    Accepts legacy aliases (e.g. ``"Prediction"`` → ``"Predictions"``).
    """
    resolved = _LABEL_ALIASES.get(label, label)
    for page in _PAGES:
        if page.label == resolved:
            return page
    raise KeyError(f"Unknown page label: {label}")


def render_registered_page(label: str) -> None:
    """Import and render one registered page."""

    page = get_page_by_label(label)
    module = importlib.import_module(page.module_path)
    getattr(module, page.render_function)()