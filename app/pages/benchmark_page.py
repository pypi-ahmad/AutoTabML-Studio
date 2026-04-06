"""Streamlit page for LazyPredict benchmark execution and results."""

from __future__ import annotations

import streamlit as st

from app.gpu import cuda_summary
from app.pages.dataset_workspace import get_active_loaded_dataset, render_active_dataset_banner, render_dataset_gateway_notice
from app.modeling.benchmark.errors import BenchmarkError
from app.modeling.benchmark.lazypredict_runner import is_lazypredict_available
from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkSplitConfig, BenchmarkTaskType
from app.modeling.benchmark.service import benchmark_dataset
from app.modeling.benchmark.summary import leaderboard_to_dataframe
from app.storage import build_metadata_store
from app.state.session import get_or_init_state
from app.security.masking import safe_error_message


def build_benchmark_run_key(
    dataset_name: str,
    target_column: str,
    requested_task_type: BenchmarkTaskType,
) -> str:
    """Return a stable session-state key for benchmark results."""

    return f"{dataset_name}::{target_column}::{requested_task_type.value}"


def default_ranking_metric_for_task(
    task_type: BenchmarkTaskType,
    settings,
) -> str:
    """Return the default ranking metric to prefill in the UI."""

    if task_type == BenchmarkTaskType.CLASSIFICATION:
        return settings.default_classification_ranking_metric
    if task_type == BenchmarkTaskType.REGRESSION:
        return settings.default_regression_ranking_metric
    return ""


def render_benchmark_page() -> None:
    state = get_or_init_state()
    settings = state.settings.benchmark
    metadata_store = build_metadata_store(state.settings)
    st.title("🏁 Benchmark")

    selected_name, loaded_dataset = get_active_loaded_dataset(metadata_store=metadata_store)
    if selected_name is None or loaded_dataset is None:
        render_dataset_gateway_notice("Benchmark", key_prefix="benchmark")
        return

    render_active_dataset_banner(selected_name, key_prefix="benchmark")
    df = loaded_dataset.dataframe
    metadata = loaded_dataset.metadata
    gpu_info = cuda_summary()

    st.caption(f"Rows: **{len(df):,}** · Columns: **{len(df.columns):,}**")
    if gpu_info["cuda_available"]:
        st.caption(
            f"CUDA detected: {gpu_info['device_name'] or 'GPU available'}. "
            "Benchmark runs will prefer GPU-capable estimators when supported."
        )
    else:
        st.caption("CUDA not detected. Benchmark runs will fall back to CPU even when GPU preference is enabled.")

    target_column = st.selectbox("Target column", list(df.columns), key="bench_target")
    task_type = BenchmarkTaskType(
        st.selectbox(
            "Task type",
            options=[task.value for task in BenchmarkTaskType],
            index=0,
            key="bench_task_type",
        )
    )

    default_metric = default_ranking_metric_for_task(task_type, settings)
    ranking_metric = st.text_input(
        "Ranking metric (optional)",
        value=default_metric,
        key="bench_ranking_metric",
        help="Leave blank to use the task-aware fallback ranking metric.",
    ).strip()

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider(
            "Test size",
            min_value=0.1,
            max_value=0.5,
            value=float(settings.default_test_size),
            step=0.05,
            key="bench_test_size",
        )
    with col2:
        random_state = int(
            st.number_input(
                "Random seed",
                value=int(settings.default_random_state),
                step=1,
                key="bench_random_state",
            )
        )
    with col3:
        top_k = int(
            st.number_input(
                "Top-k shortlist",
                min_value=1,
                value=int(settings.ui_default_top_k),
                step=1,
                key="bench_top_k",
            )
        )

    sample_rows_default = int(settings.default_sample_rows or 0)
    sample_rows_input = int(
        st.number_input(
            "Sample rows (0 = full dataset)",
            min_value=0,
            value=sample_rows_default,
            step=1000,
            key="bench_sample_rows",
        )
    )

    if len(df) > settings.sampling_row_threshold and sample_rows_input == 0:
        st.warning(
            f"This dataset exceeds the benchmark sampling threshold ({settings.sampling_row_threshold:,} rows). "
            f"Consider sampling around {settings.suggested_sample_rows:,} rows for a faster baseline run."
        )

    with st.expander("Advanced options"):
        prefer_gpu = st.checkbox(
            "Prefer GPU acceleration when supported",
            value=bool(settings.prefer_gpu),
            key="bench_prefer_gpu",
            help="Uses LazyPredict GPU mode for supported models when CUDA is available.",
        )
        stratify_option = st.selectbox(
            "Stratify split",
            options=["auto", "true", "false"],
            index=0 if settings.default_stratify else 2,
            key="bench_stratify",
        )
        include_models_raw = st.text_input(
            "Include models (comma-separated names)",
            key="bench_include_models",
        )
        exclude_models_raw = st.text_input(
            "Exclude models (comma-separated names)",
            key="bench_exclude_models",
        )

    requested_result_key = build_benchmark_run_key(selected_name, target_column, task_type)

    if st.button("Run Benchmark", key="bench_run"):
        if not is_lazypredict_available():
            st.error("lazypredict is not installed. Install with: `pip install -e \".[benchmark]\"`")
            return

        include_models = [item.strip() for item in include_models_raw.split(",") if item.strip()]
        exclude_models = [item.strip() for item in exclude_models_raw.split(",") if item.strip()]
        stratify_value = None if stratify_option == "auto" else stratify_option == "true"
        config = BenchmarkConfig(
            target_column=target_column,
            task_type=task_type,
            prefer_gpu=prefer_gpu,
            split=BenchmarkSplitConfig(
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_value,
            ),
            ranking_metric=ranking_metric or None,
            sample_rows=sample_rows_input or None,
            include_models=include_models,
            exclude_models=exclude_models,
            top_k=top_k,
            timeout_seconds=settings.timeout_seconds,
        )

        dataset_fingerprint = metadata.content_hash or metadata.schema_hash
        try:
            bundle = benchmark_dataset(
                df,
                config,
                dataset_name=selected_name,
                dataset_fingerprint=dataset_fingerprint,
                loaded_dataset=loaded_dataset,
                metadata_store=metadata_store,
                execution_backend=state.execution_backend,
                workspace_mode=state.workspace_mode,
                artifacts_dir=settings.artifacts_dir,
                classification_default_metric=settings.default_classification_ranking_metric,
                regression_default_metric=settings.default_regression_ranking_metric,
                sampling_row_threshold=settings.sampling_row_threshold,
                suggested_sample_rows=settings.suggested_sample_rows,
                mlflow_experiment_name=settings.mlflow_experiment_name,
                tracking_uri=state.settings.tracking.tracking_uri,
                registry_uri=state.settings.tracking.registry_uri,
            )
        except BenchmarkError as exc:
            st.error(safe_error_message(exc))
            return
        except Exception as exc:
            st.error(f"Benchmark failed unexpectedly: {safe_error_message(exc)}")
            return

        bundles = st.session_state.setdefault("benchmark_bundles", {})
        bundles[requested_result_key] = bundle
        st.success("Benchmark complete.")

    bundle = st.session_state.get("benchmark_bundles", {}).get(requested_result_key)
    if bundle is None:
        st.caption("Run a benchmark to see the leaderboard, artifacts, and tracking details.")
        return

    _render_result_bundle(bundle)


def _render_result_bundle(bundle) -> None:  # noqa: ANN001
    summary = bundle.summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Best model", summary.best_model_name or "N/A", summary.best_score)
    col2.metric("Models evaluated", summary.model_count)
    col3.metric("Duration (s)", f"{summary.benchmark_duration_seconds:.2f}")

    timing_col1, timing_col2, timing_col3 = st.columns(3)
    timing_col1.metric("Train rows", summary.train_row_count)
    timing_col2.metric("Test rows", summary.test_row_count)
    timing_col3.metric(
        "Fastest model",
        summary.fastest_model_name or "N/A",
        summary.fastest_model_time_seconds,
    )

    st.caption(
        f"Ranking metric: **{summary.ranking_metric}** ({summary.ranking_direction.value}) · "
        f"Task: **{summary.task_type.value}**"
    )
    if bundle.mlflow_run_id:
        st.caption(f"MLflow run id: **{bundle.mlflow_run_id}**")

    if summary.warnings:
        with st.expander("Warnings"):
            for warning in summary.warnings:
                st.warning(warning)

    leaderboard_df = leaderboard_to_dataframe(bundle.leaderboard)
    st.subheader("Leaderboard")
    st.dataframe(leaderboard_df, width="stretch")

    if bundle.top_models:
        st.subheader("Top Shortlist")
        st.table(leaderboard_to_dataframe(bundle.top_models))

    if bundle.artifacts:
        with st.expander("Artifacts"):
            if bundle.artifacts.raw_results_csv_path:
                st.markdown(f"Raw results CSV: `{bundle.artifacts.raw_results_csv_path}`")
            if bundle.artifacts.leaderboard_csv_path:
                st.markdown(f"Leaderboard CSV: `{bundle.artifacts.leaderboard_csv_path}`")
            if bundle.artifacts.leaderboard_json_path:
                st.markdown(f"Leaderboard JSON: `{bundle.artifacts.leaderboard_json_path}`")
            if bundle.artifacts.summary_json_path:
                st.markdown(f"Summary JSON: `{bundle.artifacts.summary_json_path}`")
            if bundle.artifacts.markdown_summary_path:
                st.markdown(f"Markdown summary: `{bundle.artifacts.markdown_summary_path}`")
            if bundle.artifacts.score_chart_path:
                st.markdown(f"Score chart: `{bundle.artifacts.score_chart_path}`")
            if bundle.artifacts.training_time_chart_path:
                st.markdown(f"Time chart: `{bundle.artifacts.training_time_chart_path}`")