"""Streamlit page for comparing algorithm performance on a dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.pages.dataset_workspace import go_to_page
from app.pages.ui_labels import format_enum_value
from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.storage import build_metadata_store
from app.storage.models import AppJobType


def render_compare_page() -> None:
    state = get_or_init_state()
    metadata_store = build_metadata_store(state.settings)
    st.title("⚖️ Algorithm Comparison")
    st.caption(
        "See how different algorithms performed on the same dataset. "
        "Pick a past run below to view the rankings and scores."
    )

    if metadata_store is None:
        st.info("Run history is not available yet. Complete the initial setup to enable this feature.")
        return

    # ── Fetch benchmark + experiment jobs ──────────────────────────────
    benchmark_jobs = metadata_store.list_recent_jobs(limit=50, job_type=AppJobType.BENCHMARK)
    experiment_jobs = metadata_store.list_recent_jobs(limit=50, job_type=AppJobType.EXPERIMENT)
    all_jobs = sorted(
        benchmark_jobs + experiment_jobs,
        key=lambda j: j.updated_at,
        reverse=True,
    )

    if not all_jobs:
        st.info(
            "**No runs to compare yet.**\n\n"
            "This page shows side-by-side algorithm rankings from your **Quick Benchmark** or **Train & Tune** runs.\n\n"
            "**Next step:** Load a dataset and run a benchmark — you'll be able to compare results here."
        )
        if st.button("🏁 Go to Find Best Model", key="compare_goto_bench", type="primary"):
            go_to_page("Quick Benchmark")
        return

    # ── Executive summary ──────────────────────────────────────────────
    n_bench = sum(1 for j in all_jobs if j.job_type == AppJobType.BENCHMARK)
    n_exp = sum(1 for j in all_jobs if j.job_type == AppJobType.EXPERIMENT)
    _datasets_seen = {j.dataset_name for j in all_jobs if j.dataset_name}
    st.info(
        f"**Overview:** You have **{len(all_jobs)}** past runs "
        f"(**{n_bench}** benchmarks, **{n_exp}** experiments) "
        f"across **{len(_datasets_seen)}** dataset(s). "
        "Select a run below to inspect its algorithm rankings and scores."
    )

    # ── Job selector ───────────────────────────────────────────────────
    job_labels = [
        f"{j.dataset_name or '—'} · {format_enum_value(j.job_type.value)} · {j.updated_at:%Y-%m-%d %H:%M}"
        for j in all_jobs
    ]
    selected_label = st.selectbox(
        "Pick a dataset run to compare",
        options=job_labels,
        key="cmp_job_select",
        help="Choose a previous Quick Benchmark or Train & Tune run to view its algorithm rankings.",
    )
    selected_job = all_jobs[job_labels.index(selected_label)]

    st.subheader(f"{selected_job.dataset_name or 'Dataset'} — {format_enum_value(selected_job.job_type.value)}")

    # ── Plain-English summary ──────────────────────────────────────────
    meta = selected_job.metadata or {}
    if selected_job.job_type == AppJobType.BENCHMARK:
        _bm = meta.get("best_model_name", "N/A")
        _bs = meta.get("best_score", "N/A")
        _rm = meta.get("ranking_metric", "N/A")
        st.info(
            f"**What happened:** Benchmark run on **{selected_job.dataset_name or 'this dataset'}**. "
            f"Best performing algorithm: **{_bm}** (scored **{_bs}** on {_rm}).\n\n"
            "**Looks good?** Compare the leaderboard below to see how close the runners-up are.\n\n"
            "**Next step:** If you see a promising algorithm, head to **Train & Tune** to build and save a production model."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Algorithm", _bm)
        c2.metric("Best Score", f"{_bs}")
        c3.metric("Ranked By", _rm)
    elif selected_job.job_type == AppJobType.EXPERIMENT:
        _bl = meta.get("best_baseline_model_name", "N/A")
        _tm = meta.get("tuned_model_name", "N/A")
        _sel = meta.get("selected_model_name", "N/A")
        st.info(
            f"**What happened:** Training run on **{selected_job.dataset_name or 'this dataset'}**. "
            f"Best baseline: **{_bl}**, tuned: **{_tm}**, saved: **{_sel}**.\n\n"
            "**Looks good?** Check the leaderboard to see how algorithms rank.\n\n"
            "**Next step:** Use the saved model in **Predict**, or run **Test & Evaluate** to measure real-world accuracy."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Baseline", _bl)
        c2.metric("Tuned", _tm)
        c3.metric("Saved", _sel)

    # ── Load leaderboard from artifacts ────────────────────────────────
    leaderboard_df = _load_leaderboard(selected_job)

    if leaderboard_df is not None and not leaderboard_df.empty:
        st.subheader("Algorithm Leaderboard")
        st.dataframe(
            leaderboard_df.style.format(precision=4, na_rep="—"),
            width="stretch",
            hide_index=True,
        )

        # Metric explanations
        from app.pages.glossary import render_metric_legend
        _cmp_metric_cols = [c for c in leaderboard_df.columns if c not in ("Rank", "Model")]
        render_metric_legend(_cmp_metric_cols, key_prefix="cmp_legend")

        # ── Metric chart ──────────────────────────────────────────────
        score_col = _find_score_column(leaderboard_df)
        model_col = _find_model_column(leaderboard_df)
        if score_col and model_col:
            chart_df = leaderboard_df[[model_col, score_col]].dropna(subset=[score_col]).copy()
            chart_df = chart_df.sort_values(score_col, ascending=False).head(20)
            st.subheader("Score Comparison")
            st.bar_chart(chart_df.set_index(model_col)[score_col])

        # ── Export ────────────────────────────────────────────────────
        st.subheader("📤 Download Results")
        dl_col1, dl_col2, _ = st.columns([2, 2, 4])
        dl_col1.download_button(
            "Download Leaderboard CSV",
            data=leaderboard_df.to_csv(index=False).encode("utf-8"),
            file_name=f"comparison_{(selected_job.dataset_name or 'results').replace(' ', '_')}.csv",
            mime="text/csv",
            key="cmp_dl_csv",
        )
        dl_col2.download_button(
            "Download Leaderboard Data",
            data=leaderboard_df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name=f"comparison_{(selected_job.dataset_name or 'results').replace(' ', '_')}.json",
            mime="application/json",
            key="cmp_dl_json",
        )
    else:
        st.info(
            "No results file found for this run. "
            "The results may have been deleted or the run didn't produce a leaderboard."
        )

    # ── Side-by-side MLflow comparison (optional) ──────────────────────
    _render_mlflow_comparison(state.settings.tracking)


# ── Helpers ───────────────────────────────────────────────────────────


def _load_leaderboard(job) -> pd.DataFrame | None:  # noqa: ANN001
    """Try to load a leaderboard from the job's artifact paths."""

    # 1. Benchmark: primary_artifact_path is the leaderboard CSV
    if job.job_type == AppJobType.BENCHMARK and job.primary_artifact_path:
        csv_path = Path(job.primary_artifact_path)
        if csv_path.exists() and csv_path.suffix == ".csv":
            try:
                return pd.read_csv(csv_path)
            except Exception:
                pass

    # 2. Try to find a leaderboard JSON sibling of the summary
    if job.summary_path:
        summary_dir = Path(job.summary_path).parent
        # Benchmark leaderboard JSONs
        for json_path in sorted(summary_dir.glob("*leaderboard*.json"), reverse=True):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return _leaderboard_rows_to_df(data)
            except Exception:
                continue
        # Experiment compare JSONs
        for json_path in sorted(summary_dir.glob("*compare*.json"), reverse=True):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return _leaderboard_rows_to_df(data)
            except Exception:
                continue
        # Leaderboard CSVs
        for csv_path in sorted(summary_dir.glob("*leaderboard*.csv"), reverse=True):
            try:
                return pd.read_csv(csv_path)
            except Exception:
                continue

    # 3. Scan the default benchmark artifacts dir
    artifacts_dir = Path("artifacts") / "benchmark"
    if job.dataset_name and artifacts_dir.exists():
        ds_lower = job.dataset_name.lower().replace(" ", "_")
        for json_path in sorted(artifacts_dir.glob(f"*{ds_lower}*leaderboard*.json"), reverse=True):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return _leaderboard_rows_to_df(data)
            except Exception:
                continue

    return None


def _leaderboard_rows_to_df(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of leaderboard row dicts to a clean DataFrame."""

    records = []
    for row in rows:
        record = {
            "Rank": row.get("rank"),
            "Model": row.get("model_name", row.get("Model", "")),
        }
        # Primary score
        score = row.get("primary_score")
        if score is not None:
            record["Score"] = score

        # Raw metrics (expanded)
        raw = row.get("raw_metrics", {})
        for key, val in raw.items():
            record[key] = val

        # Training time
        time_val = row.get("training_time_seconds")
        if time_val is not None:
            record["Training Time (s)"] = time_val

        records.append(record)

    df = pd.DataFrame(records)
    if "Rank" in df.columns:
        df = df.sort_values("Rank", na_position="last")
    return df


def _find_score_column(df: pd.DataFrame) -> str | None:
    """Find the primary score column in a leaderboard DataFrame."""

    for candidate in ["Score", "Primary Score", "primary_score", "Balanced Accuracy",
                      "Accuracy", "R-Squared", "Adjusted R-Squared", "F1", "AUC"]:
        if candidate in df.columns:
            return candidate
    # Fallback: first numeric column after Rank/Model
    for col in df.columns:
        if col not in ("Rank", "Model", "Task Type", "Benchmark Backend",
                        "Run Timestamp", "Warnings", "Training Time (s)"):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def _find_model_column(df: pd.DataFrame) -> str | None:
    for candidate in ["Model", "model_name"]:
        if candidate in df.columns:
            return candidate
    return None


def _render_mlflow_comparison(tracking_settings) -> None:  # noqa: ANN001
    """Optional MLflow side-by-side run comparison."""

    from app.tracking.mlflow_query import is_mlflow_available

    if not is_mlflow_available():
        return

    with st.expander("Side-by-Side Run Comparison (Advanced)", expanded=False):
        from app.tracking.compare_service import ComparisonService
        from app.tracking.history_service import HistoryService

        history = HistoryService(
            tracking_uri=tracking_settings.tracking_uri,
            default_experiment_names=tracking_settings.default_experiment_names,
            default_limit=tracking_settings.history_page_default_limit,
        )

        try:
            runs = history.list_runs(limit=50)
        except Exception as exc:
            st.error(
                f"Failed to query MLflow: {safe_error_message(exc)}\n\n"
                "**What to try:** Check that MLflow tracking is configured in **Settings** and the tracking server is running."
            )
            return

        if len(runs) < 2:
            st.caption("Need at least 2 MLflow runs for side-by-side comparison.")
            return

        run_labels = [
            f"{r.dataset_name or '—'} · {r.model_name or format_enum_value(r.run_type.value)}"
            for r in runs
        ]

        col1, col2 = st.columns(2)
        with col1:
            left_label = st.selectbox("Left run", options=run_labels, index=0, key="cmp_mlflow_left", help="First run to compare.")
        with col2:
            right_default = 1 if len(run_labels) > 1 else 0
            right_label = st.selectbox("Right run", options=run_labels, index=right_default, key="cmp_mlflow_right", help="Second run to compare against.")

        left_run = runs[run_labels.index(left_label)]
        right_run = runs[run_labels.index(right_label)]

        if left_run.run_id == right_run.run_id:
            st.warning("Select two different runs.")
            return

        comparison = ComparisonService()
        bundle = comparison.compare(left_run, right_run)

        if bundle.metric_deltas:
            left_display = f"{left_run.dataset_name or '—'} · {left_run.model_name or format_enum_value(left_run.run_type.value)}"
            right_display = f"{right_run.dataset_name or '—'} · {right_run.model_name or format_enum_value(right_run.run_type.value)}"
            rows = []
            for delta in bundle.metric_deltas:
                rows.append({
                    "Metric": delta.name,
                    left_display: delta.left_value,
                    right_display: delta.right_value,
                    "Delta": delta.delta,
                    "Better": delta.better_side or "",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # --- Save comparison ---
    if st.button("Save Comparison Results", key="cmp_save"):
        from app.tracking.artifacts import write_comparison_artifacts

        try:
            paths = write_comparison_artifacts(bundle, settings.comparison_artifacts_dir)
            st.success("Comparison results saved.")
            with st.expander("Saved files", expanded=False):
                for label, path in paths.items():
                    st.caption(f"{label}: {Path(str(path)).name}")
        except Exception as exc:
            st.error(
                f"Failed to save comparison results: {safe_error_message(exc)}\n\n"
                "**What to try:** Check that the output folder is writable, or try saving to a different location in **Settings**."
            )
