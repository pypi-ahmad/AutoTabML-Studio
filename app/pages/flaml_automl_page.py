"""Streamlit page for the FLAML AutoML workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.modeling.flaml.errors import FlamlAutoMLError
from app.modeling.flaml.schemas import (
    DEFAULT_ESTIMATOR_LIST,
    FlamlConfig,
    FlamlSearchConfig,
    FlamlTaskType,
)
from app.modeling.flaml.service import FlamlAutoMLService
from app.modeling.flaml.setup_runner import is_flaml_available, flaml_install_guidance
from app.pages.dataset_workspace import (
    go_to_page,
    render_dataset_header,
)
from app.pages.workflow_guide import render_next_step_hint, render_workflow_banner
from app.path_utils import model_save_name
from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.storage import build_metadata_store, ensure_dataset_record


# ── Estimator display names ───────────────────────────────────────────
ESTIMATOR_LABELS: dict[str, str] = {
    "lgbm": "LightGBM",
    "xgboost": "XGBoost",
    "xgb_limitdepth": "XGBoost (limited depth)",
    "catboost": "CatBoost",
    "rf": "Random Forest",
    "extra_tree": "Extra Trees",
    "lrl1": "Logistic/Linear Regression (L1)",
    "lrl2": "Logistic/Linear Regression (L2)",
    "kneighbor": "K-Nearest Neighbors",
}

CLASSIFICATION_METRICS: list[str] = [
    "accuracy",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "f1",
    "micro_f1",
    "macro_f1",
    "log_loss",
]

REGRESSION_METRICS: list[str] = [
    "r2",
    "mae",
    "mse",
    "mape",
]


def _metric_options_for_task(task_type: FlamlTaskType) -> list[str]:
    if task_type == FlamlTaskType.CLASSIFICATION:
        return list(CLASSIFICATION_METRICS)
    if task_type == FlamlTaskType.REGRESSION:
        return list(REGRESSION_METRICS)
    return list(CLASSIFICATION_METRICS) + list(REGRESSION_METRICS)


def _build_flaml_run_key(
    dataset_name: str,
    target_column: str,
    task_type: FlamlTaskType,
) -> str:
    return f"flaml::{dataset_name}::{target_column}::{task_type.value}"


def render_flaml_automl_page() -> None:
    state = get_or_init_state()
    settings = state.settings.flaml
    metadata_store = build_metadata_store(state.settings)
    st.title("🔥 FLAML AutoML")
    render_workflow_banner(current_step=4)
    st.caption(
        "**Fast, lightweight AutoML** — automatically finds the best algorithm and hyperparameters "
        "within a time budget. Powered by Microsoft FLAML's cost-effective search."
    )

    if not is_flaml_available():
        st.error(flaml_install_guidance())
        return

    # ── Step 1: Choose Your Data ───────────────────────────────────────
    st.subheader("1. Choose Your Data")
    selected_name, loaded_dataset = render_dataset_header(
        "FLAML AutoML", key_prefix="flaml", metadata_store=metadata_store,
    )
    if selected_name is None or loaded_dataset is None:
        return

    df = loaded_dataset.dataframe
    metadata = loaded_dataset.metadata
    ensure_dataset_record(metadata_store, loaded_dataset, dataset_name=selected_name)

    st.caption(f"Rows: **{len(df):,}** · Columns: **{len(df.columns):,}**")

    # ── Step 2: Configure ─────────────────────────────────────────────
    st.subheader("2. Configure")
    target_column = st.selectbox(
        "Target column",
        list(df.columns),
        key="flaml_target",
        help="The column your model should learn to predict.",
    )
    from app.pages.ui_labels import TASK_TYPE_LABELS, make_format_func

    _flaml_task_values = [t.value for t in FlamlTaskType]
    task_type = FlamlTaskType(
        st.selectbox(
            "Task type",
            options=_flaml_task_values,
            format_func=make_format_func(TASK_TYPE_LABELS),
            index=0,
            key="flaml_task_type",
            help=(
                "Choose 'Classification' for categories, 'Regression' for numbers, "
                "or 'Auto' to let the system decide."
            ),
        )
    )

    # ── Time budget presets ────────────────────────────────────────────
    _preset_names = ["Quick (60s)", "Standard (120s)", "Deep (300s)", "Custom"]
    _preset_budgets = [60, 120, 300, None]
    preset_idx = st.radio(
        "Search budget",
        options=range(len(_preset_names)),
        format_func=lambda i: _preset_names[i],
        index=1,
        horizontal=True,
        key="flaml_preset",
        help="How long FLAML searches for the best model. Longer = more thorough.",
    )
    preset_budget = _preset_budgets[preset_idx]

    with st.expander("Advanced options"):
        # ── Time budget ────────────────────────────────────────────────
        st.caption("**Search settings**")
        col1, col2, col3 = st.columns(3)
        with col1:
            time_budget = int(
                st.number_input(
                    "Time budget (seconds)",
                    min_value=10,
                    value=preset_budget or int(settings.default_time_budget),
                    step=10,
                    key="flaml_time_budget",
                    help=(
                        "Maximum time in seconds for the AutoML search. "
                        "FLAML will try as many algorithm + hyperparameter combinations "
                        "as it can within this budget."
                    ),
                )
            )
        with col2:
            n_splits = int(
                st.number_input(
                    "Cross-validation folds",
                    min_value=2,
                    value=int(settings.default_n_splits),
                    step=1,
                    key="flaml_n_splits",
                    help="Number of cross-validation folds. More folds = more reliable scores but slower.",
                )
            )
        with col3:
            seed = int(
                st.number_input(
                    "Random seed",
                    value=int(settings.default_seed),
                    step=1,
                    key="flaml_seed",
                    help="A number that makes the experiment reproducible.",
                )
            )

        # ── Metric ────────────────────────────────────────────────────
        st.caption("**Optimization metric**")
        metric_options = ["auto"] + _metric_options_for_task(task_type)
        metric = st.selectbox(
            "Metric to optimize",
            options=metric_options,
            index=0,
            key="flaml_metric",
            help="The score FLAML optimizes. 'auto' picks the best default for your task type.",
        )

        # ── Estimators ────────────────────────────────────────────────
        st.caption("**Algorithms to try**")
        all_estimators = list(ESTIMATOR_LABELS.keys())
        default_estimators = [e for e in DEFAULT_ESTIMATOR_LIST if e in all_estimators]
        estimator_list = st.multiselect(
            "Algorithms",
            options=all_estimators,
            default=default_estimators,
            format_func=lambda e: ESTIMATOR_LABELS.get(e, e),
            key="flaml_estimators",
            help="Which algorithms FLAML should try. More algorithms = broader search but takes longer.",
        )

        # ── Additional options ────────────────────────────────────────
        st.caption("**Additional options**")
        acol1, acol2 = st.columns(2)
        with acol1:
            ensemble = st.checkbox(
                "Ensemble (combine best models)",
                value=False,
                key="flaml_ensemble",
                help=(
                    "After finding the best model, try combining top models "
                    "into an ensemble for potentially better accuracy."
                ),
            )
            early_stop = st.checkbox(
                "Early stopping",
                value=False,
                key="flaml_early_stop",
                help="Stop the search early if the results have converged.",
            )
        with acol2:
            sample = st.checkbox(
                "Sample large datasets",
                value=True,
                key="flaml_sample",
                help="For large datasets, FLAML will start with a sample and gradually increase the data size.",
            )
            retrain_full = st.checkbox(
                "Retrain on full data",
                value=True,
                key="flaml_retrain_full",
                help="After finding the best configuration, retrain the model on the full training data.",
            )

    # Use preset budget if not custom
    if preset_budget is not None:
        time_budget = preset_budget

    # ── Step 3: Start Search ───────────────────────────────────────────
    st.subheader("3. Start Search")
    requested_key = _build_flaml_run_key(selected_name, target_column, task_type)
    flaml_bundles = st.session_state.setdefault("flaml_bundles", {})

    if st.button("🔍 Start FLAML Search", key="flaml_run_search", type="primary"):
        service = _build_service(state.settings, metadata_store=metadata_store)
        config = FlamlConfig(
            target_column=target_column,
            task_type=task_type,
            search=FlamlSearchConfig(
                time_budget=time_budget,
                metric=metric,
                estimator_list=estimator_list or list(DEFAULT_ESTIMATOR_LIST),
                n_splits=n_splits,
                seed=seed,
                ensemble=ensemble,
                early_stop=early_stop,
                sample=sample,
                retrain_full=retrain_full,
            ),
        )
        dataset_fingerprint = metadata.content_hash or metadata.schema_hash
        with st.spinner(f"FLAML is searching for the best model (budget: {time_budget}s)…"):
            try:
                bundle = service.run_automl(
                    df,
                    config,
                    dataset_name=selected_name,
                    dataset_fingerprint=dataset_fingerprint,
                    execution_backend=state.execution_backend,
                    workspace_mode=state.workspace_mode,
                )
            except FlamlAutoMLError as exc:
                st.error(
                    f"FLAML search stopped: {safe_error_message(exc)}\n\n"
                    "**What to try:** Check your target column and task type, "
                    "or increase the time budget."
                )
                return
            except Exception as exc:
                st.error(
                    f"FLAML search failed unexpectedly: {safe_error_message(exc)}\n\n"
                    "**What to try:** Verify your data has no formatting issues, "
                    "reduce the number of rows, or reload the dataset."
                )
                return

        flaml_bundles[requested_key] = bundle
        st.success("FLAML search complete!")

    bundle = flaml_bundles.get(requested_key)
    if bundle is None:
        st.info(
            "**Ready to search.**\n\n"
            "Choose a target column above and click **Start FLAML Search**.\n\n"
            "**What happens:** FLAML automatically tests multiple algorithms and hyperparameter "
            "configurations within your time budget, then returns the best model."
        )
        return

    # ── Step 4: Results ───────────────────────────────────────────────
    st.subheader("4. Results")
    _render_flaml_results(bundle, state.settings, metadata_store)
    render_next_step_hint(current_step=4)


def _render_flaml_results(bundle, app_settings, metadata_store) -> None:  # noqa: ANN001
    summary = bundle.summary
    search = bundle.search_result

    # ── Summary ────────────────────────────────────────────────────────
    _best = summary.best_estimator or "N/A"
    _best_label = ESTIMATOR_LABELS.get(_best, _best)
    _loss = summary.best_loss
    _dur = f"{summary.search_duration_seconds:.1f}"
    _metric = summary.metric or "auto"
    _count = len(search.leaderboard) if search else 0

    st.info(
        f"**What happened:** Searched **{_count}** algorithm(s) in **{_dur}s**. "
        f"Best: **{_best_label}** (loss: **{_loss:.4f}**, metric: **{_metric}**).\n\n"
        f"**What to do next:** Save the best model for predictions, or adjust settings and re-run."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Best algorithm", _best_label)
    col2.metric("Best loss", f"{_loss:.4f}" if _loss is not None else "N/A")
    col3.metric("Search time (s)", _dur)

    if search and search.best_config:
        with st.expander("Best configuration", expanded=False):
            st.caption("The hyperparameters FLAML selected for the best model.")
            st.json(search.best_config)

    # ── Leaderboard ───────────────────────────────────────────────────
    if search and search.leaderboard:
        st.subheader("Leaderboard")
        lb_data = []
        for row in search.leaderboard:
            lb_data.append({
                "Rank": row.rank,
                "Algorithm": ESTIMATOR_LABELS.get(row.estimator_name, row.estimator_name),
                "Best Loss": f"{row.best_loss:.4f}" if row.best_loss is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(lb_data), width="stretch")

    # ── Save ──────────────────────────────────────────────────────────
    st.subheader("Save Model")
    save_col, predict_col = st.columns(2)
    if save_col.button("💾 Save Best Model", key="flaml_save", type="primary",
                       help="Save the best model so you can load it on the Predictions page."):
        service = _build_service(app_settings, metadata_store=metadata_store)
        _save_name = model_save_name(bundle.dataset_name, summary.best_estimator or "flaml_best")
        try:
            updated = service.save_best_model(bundle, save_name=_save_name)
            st.session_state["flaml_bundles"][
                _build_flaml_run_key(
                    bundle.dataset_name or "dataset",
                    bundle.config.target_column,
                    bundle.task_type,
                )
            ] = updated
            st.success(f"Model saved as **{_save_name}**.")
            bundle = updated
        except FlamlAutoMLError as exc:
            st.error(f"Save failed: {safe_error_message(exc)}")

    if predict_col.button("🔮 Go to Predictions", key="flaml_goto_predict"):
        go_to_page("Predictions")

    # ── Saved model info ──────────────────────────────────────────────
    if bundle.saved_model_metadata is not None:
        st.subheader("Saved Model")
        st.success("Model saved and ready for predictions.")
        with st.expander("Saved files", expanded=False):
            st.caption(f"Model file: **{Path(str(bundle.saved_model_metadata.model_path)).name}**")

    # ── Artifacts ─────────────────────────────────────────────────────
    if bundle.artifacts is not None:
        with st.expander("📤 Reports & Downloads", expanded=True):
            for label, path in [
                ("Search results", bundle.artifacts.search_result_json_path),
                ("Leaderboard CSV", bundle.artifacts.leaderboard_csv_path),
                ("Summary data", bundle.artifacts.summary_json_path),
            ]:
                if path is not None and path.exists():
                    st.download_button(
                        label=f"Download {label}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        key=f"flaml_dl_{label}_{path.name}",
                    )

    # ── Warnings ──────────────────────────────────────────────────────
    if bundle.warnings:
        with st.expander("Warnings"):
            for warning in bundle.warnings:
                st.warning(warning)


def _build_service(app_settings, *, metadata_store=None) -> FlamlAutoMLService:  # noqa: ANN001
    settings = app_settings.flaml
    return FlamlAutoMLService(
        artifacts_dir=settings.artifacts_dir,
        models_dir=settings.models_dir,
        default_classification_metric=settings.default_classification_metric,
        default_regression_metric=settings.default_regression_metric,
        mlflow_experiment_name=settings.mlflow_experiment_name,
        tracking_uri=app_settings.tracking.tracking_uri,
        registry_uri=app_settings.tracking.registry_uri,
        metadata_store=metadata_store,
    )
