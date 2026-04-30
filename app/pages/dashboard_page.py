"""Dashboard page – professional workspace home."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.pages.dataset_workspace import get_active_loaded_dataset, get_loaded_datasets, go_to_page
from app.pages.ui_cache import get_metadata_store
from app.pages.ui_labels import format_enum_value
from app.pages.workflow_guide import WORKFLOW_STEPS
from app.state.session import get_or_init_state


def _detect_completed_steps() -> int:
    """Return the highest *completed* step number (0 = nothing done yet).

    Heuristic based on session-state keys that downstream pages populate.
    Step 2 (Check Quality) is optional and auto-skipped for sequencing.
    """
    loaded = get_loaded_datasets()
    if not loaded:
        return 0                                       # nothing loaded → step 0
    if not st.session_state.get("benchmark_bundles"):
        return 2                                       # data loaded → skip optional step 2, benchmark next
    if not st.session_state.get("experiment_bundles") and not st.session_state.get("flaml_bundles"):
        return 3                                       # benchmarked, no experiment yet
    return 4                                           # trained → predict is next


# ── One-click example datasets ────────────────────────────────────────

_EXAMPLE_DATASETS: list[dict[str, str | int]] = [
    {
        "label": "🌸 Iris (classification, 150 rows)",
        "uci_id": 53,
        "name": "iris",
        "task": "classification",
    },
    {
        "label": "❤️ Heart Disease (classification, 303 rows)",
        "uci_id": 45,
        "name": "heart-disease",
        "task": "classification",
    },
    {
        "label": "🍷 Wine Quality (regression, 4 898 rows)",
        "uci_id": 186,
        "name": "wine-quality",
        "task": "regression",
    },
]


def _load_example_dataset(example: dict) -> None:
    """Load a UCI example into the session and navigate to Load Data."""
    from app.ingestion.schemas import DatasetInputSpec
    from app.ingestion.types import IngestionSourceType
    from app.pages.dataset_workspace import _load_into_session

    spec = DatasetInputSpec(
        source_type=IngestionSourceType.UCI_REPO,
        uci_id=int(example["uci_id"]),
        display_name=str(example["name"]),
    )
    state = get_or_init_state()
    metadata_store = get_metadata_store(state.settings)
    result = _load_into_session(spec, preferred_name=str(example["name"]), metadata_store=metadata_store)
    if result:
        go_to_page("Load Data")


def render_dashboard_page() -> None:
    state = get_or_init_state()
    metadata_store = get_metadata_store(state.settings)
    startup_status = st.session_state.get("startup_status")

    # ── Hero header ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding: 1rem 0 0.5rem 0;">
            <h1 style="margin:0;">AutoTabML Studio</h1>
            <p style="color: gray; margin-top: 0.25rem; font-size: 1.05rem;">
                Local-first automated machine learning for tabular data
            </p>
            <p style="color: gray; margin-top: 0; font-size: 0.85rem;">
                🔒 Your data never leaves your machine — everything runs locally and stays private.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Active dataset spotlight ───────────────────────────────────────
    active_name, active_dataset = get_active_loaded_dataset(metadata_store=metadata_store)
    completed = _detect_completed_steps()
    recommended = completed + 1  # 1-based step the user should do next

    if active_dataset is None or active_name is None:
        # ── First-run walkthrough ──────────────────────────────────────
        st.markdown("")
        st.markdown(
            "### 👋 Welcome!  Let's get started.\n"
            "AutoTabML Studio walks you through building a machine-learning model — "
            "from raw data to predictions — right on your own machine."
        )

        # Goal-based entry choices
        st.markdown("**What would you like to do?**")
        goal_cols = st.columns(3)
        with goal_cols[0]:
            st.markdown("**📥 Use my own data**")
            st.caption("Upload a CSV or Excel file and start exploring.")
            if st.button("Upload My Data", key="dash_goal_upload", type="primary", use_container_width=True):
                go_to_page("Load Data")
        with goal_cols[1]:
            st.markdown("**🧪 Try an example**")
            st.caption("Load a sample dataset and follow the guided workflow.")
            for ex in _EXAMPLE_DATASETS:
                if st.button(str(ex["label"]), key=f"dash_ex_{ex['name']}", use_container_width=True):
                    _load_example_dataset(ex)
        with goal_cols[2]:
            st.markdown("**🔮 Score new data**")
            st.caption("Already have a trained model? Jump straight to predictions.")
            if st.button("Go to Predictions", key="dash_goal_predict", use_container_width=True):
                go_to_page("Predictions")

        # ── Recommended journey ────────────────────────────────────────
        st.markdown("")
        st.markdown("#### Your path to predictions")
        st.caption(
            "Follow these five steps to go from raw data to a working model. "
            "Steps marked *optional* can be skipped."
        )
        for step in WORKFLOW_STEPS:
            num = int(step["number"])
            icon = step["icon"]
            label = step["label"]
            short = step.get("short", "")
            optional = step.get("optional", False)
            opt_tag = "  `optional`" if optional else ""
            if num == 1:
                st.markdown(f"**{icon} Step {num} · {label}** — *start here*{opt_tag}")
            else:
                st.markdown(f"{icon} **Step {num} · {label}**{opt_tag}")
            st.caption(f"  {short}")
            for alt in step.get("alternatives", []):
                st.caption(f"  {alt['icon']} *Alternative:* {alt['label']} — {alt['short']}")
    else:
        # ── Dataset loaded — show stats + sequential recommended step ──
        st.subheader(f"📂 Current Dataset: {active_name}")
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric("Rows", f"{len(active_dataset.dataframe):,}")
        stat2.metric("Features", f"{len(active_dataset.dataframe.columns):,}")

        numeric_count = active_dataset.dataframe.select_dtypes(include="number").shape[1]
        categorical_count = len(active_dataset.dataframe.columns) - numeric_count
        stat3.metric("Numeric", numeric_count)
        stat4.metric("Categorical", categorical_count)

        with st.expander("Preview", expanded=False):
            st.dataframe(active_dataset.preview(10), width="stretch")

        # Recommended next step — large primary CTA with explanation
        if recommended <= len(WORKFLOW_STEPS):
            next_step = WORKFLOW_STEPS[recommended - 1]
            st.markdown("")
            st.info(
                f"**Recommended next step →  {next_step['icon']}  Step {next_step['number']} · {next_step['label']}**\n\n"
                f"{next_step.get('short', '')}"
            )
            if st.button(
                f"Go to {next_step['label']}",
                key="dash_recommended_next",
                type="primary",
                use_container_width=True,
            ):
                go_to_page(next_step["page"])

        # Full workflow progress — completed steps checked, future dimmed
        st.markdown("")
        st.caption("YOUR PROGRESS")
        for step in WORKFLOW_STEPS:
            num = int(step["number"])
            icon = step["icon"]
            label = step["label"]
            optional = step.get("optional", False)
            opt_tag = "  `optional`" if optional else ""
            if num <= completed:
                col_a, col_b = st.columns([6, 2])
                col_a.markdown(f"~~✓ Step {num} · {label}~~ ✅")
                if col_b.button("Revisit", key=f"dash_revisit_{step['page']}", use_container_width=True):
                    go_to_page(step["page"])
            elif num == recommended:
                st.markdown(f"**▶ Step {num} · {label}** — *you are here*")
            else:
                st.caption(f"{icon} Step {num} · {label}{opt_tag}")

    # ── Startup issues ─────────────────────────────────────────────────
    if startup_status and startup_status.issues:
        with st.expander("⚠️ Environment Issues", expanded=bool(startup_status.errors)):
            for issue in startup_status.issues:
                if issue.severity == "error":
                    st.error(issue.message)
                elif issue.severity == "warning":
                    st.warning(issue.message)
                else:
                    st.info(issue.message)

    if metadata_store is None:
        st.info(
            "Run history is not available yet. "
            "Please complete the initial setup (see the README or ask your administrator)."
        )
        return

    # ── Fetch workspace data ───────────────────────────────────────────
    jobs = metadata_store.list_recent_jobs(limit=10)
    datasets = metadata_store.list_recent_datasets(limit=8)
    models = metadata_store.list_saved_local_models(limit=8)
    loaded = get_loaded_datasets()

    st.divider()

    # ── KPI summary bar ────────────────────────────────────────────────
    st.subheader("Workspace Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Jobs Run", len(jobs))
    kpi2.metric("Datasets Used", len(datasets))
    kpi3.metric("Saved Models", len(models))
    kpi4.metric("Loaded Now", len(loaded))

    # ── Recent activity ────────────────────────────────────────────────
    if jobs:
        st.subheader("Recent Activity")
        job_rows = []
        for job in jobs:
            friendly_name = _friendly_job_name(job)
            job_rows.append({
                "Dataset / Subject": friendly_name,
                "Job Type": format_enum_value(job.job_type.value),
                "Status": _status_badge(job.status.value),
                "Updated": job.updated_at,
            })
        st.dataframe(pd.DataFrame(job_rows), width="stretch", hide_index=True)
    else:
        st.info(
            "**No activity yet.** Load a dataset and run your first workflow — "
            "results will appear here automatically."
        )
        if st.button("📥 Load a Dataset", key="dash_activity_goto_load", type="primary"):
            go_to_page("Load Data")

    # ── Datasets + Models side by side ─────────────────────────────────
    ds_col, model_col = st.columns(2)

    with ds_col:
        st.subheader("Tracked Datasets")
        if datasets:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Name": ds.display_name or ds.source_locator,
                            "Rows": f"{ds.row_count:,}",
                            "Columns": ds.column_count,
                            "Source": ds.source_type,
                        }
                        for ds in datasets
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("Datasets appear here after you run a workflow.")

    with model_col:
        st.subheader("Saved Models")
        if models:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Model": model.model_name,
                            "Task": model.task_type.title(),
                            "Target": model.target_column or "—",
                        }
                        for model in models
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("Train and save models via Train & Tune or Quick Benchmark.")

    # ── Environment details (collapsed, bottom of page) ────────────────
    from app.pages.ui_labels import BACKEND_LABELS, MODE_LABELS, PROVIDER_LABELS
    with st.expander("⚙️ Environment", expanded=False):
        env_col1, env_col2, env_col3 = st.columns(3)
        env_col1.caption(f"**AI Provider:** {PROVIDER_LABELS.get(state.provider.value, state.provider.value)}")
        env_col2.caption(f"**Runs on:** {BACKEND_LABELS.get(state.execution_backend.value, state.execution_backend.value)}")
        env_col3.caption(f"**Mode:** {MODE_LABELS.get(state.workspace_mode.value, state.workspace_mode.value)}")


# ──────────────────────────────────────────────────────────────────────


def _friendly_job_name(job) -> str:  # noqa: ANN001
    """Return a user-friendly display name for a job record."""

    if job.dataset_name:
        return job.dataset_name
    if job.title:
        # title is like "Benchmark · iris" — extract the dataset portion
        parts = job.title.split("·")
        if len(parts) >= 2:
            return parts[-1].strip()
        return job.title
    return format_enum_value(job.job_type.value)


def _status_badge(status: str) -> str:
    """Return a status string with a visual indicator."""

    if status.lower() == "success":
        return "✅ Success"
    if status.lower() == "failed":
        return "❌ Failed"
    return status.title()
