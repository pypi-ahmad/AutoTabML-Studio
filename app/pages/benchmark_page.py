"""Streamlit page for LazyPredict benchmark execution and results."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from app.gpu import cuda_summary
from app.modeling.benchmark.errors import BenchmarkError
from app.modeling.benchmark.lazypredict_runner import is_lazypredict_available
from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkSplitConfig, BenchmarkTaskType
from app.modeling.benchmark.service import benchmark_dataset
from app.modeling.benchmark.summary import leaderboard_to_dataframe
from app.pages.dataset_workspace import render_dataset_header
from app.pages.shared_history import render_past_runs_section, render_saved_models_section
from app.pages.workflow_guide import render_next_step_hint, render_workflow_banner
from app.path_utils import model_save_name
from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.storage import build_metadata_store
from app.storage.models import AppJobType

logger = logging.getLogger(__name__)


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
    st.title("🏁 Quick Benchmark")
    render_workflow_banner(current_step=3)
    st.caption(
        "**Quick screening** — test dozens of algorithms on your data in one click to see which ones work best. "
        "No tuning, no saving — just a ranked leaderboard so you know where to focus in **Train & Tune**."
    )

    # ── Step 1: Choose Your Data ───────────────────────────────────────
    st.subheader("1. Choose Your Data")
    selected_name, loaded_dataset = render_dataset_header("Benchmark", key_prefix="benchmark", metadata_store=metadata_store)
    if selected_name is None or loaded_dataset is None:
        return

    df = loaded_dataset.dataframe
    metadata = loaded_dataset.metadata
    gpu_info = cuda_summary()

    st.caption(f"Rows: **{len(df):,}** · Columns: **{len(df.columns):,}**")

    # ── Step 2: Configure ─────────────────────────────────────────────
    st.subheader("2. Configure")
    target_column = st.selectbox(
        "Target column",
        list(df.columns),
        key="bench_target",
        help="The column you want the model to predict. For example, 'Price' for a pricing model or 'Churn' for customer churn.",
    )
    from app.pages.ui_labels import TASK_TYPE_LABELS, make_format_func
    task_type = BenchmarkTaskType(
        st.selectbox(
            "Task type",
            options=[task.value for task in BenchmarkTaskType],
            format_func=make_format_func(TASK_TYPE_LABELS),
            index=0,
            key="bench_task_type",
            help="Choose 'Classification' if your target is a category, or 'Regression' if it's a number.",
        )
    )

    # ── Smart defaults (user can override in Advanced) ─────────────────
    default_metric = default_ranking_metric_for_task(task_type, settings)
    sample_rows_default = int(settings.default_sample_rows or 0)

    # ── Run mode presets ───────────────────────────────────────────────
    from app.pages.glossary import BENCHMARK_PRESETS

    _preset_names = list(BENCHMARK_PRESETS.keys())
    _preset_descs = [f"**{k}** — {v['description']}" for k, v in BENCHMARK_PRESETS.items()]
    run_mode = st.radio(
        "Run mode",
        options=_preset_names,
        index=1,
        horizontal=True,
        key="bench_run_mode",
        help="Quick = fast screening on a sample.  Standard = balanced.  Deep = full dataset, thorough evaluation.",
    )
    _preset = BENCHMARK_PRESETS[run_mode]
    st.caption(_preset["description"])

    with st.expander("Advanced options"):
        st.caption("**Scoring & display**")
        ranking_metric = st.text_input(
            "Ranking score",
            value=default_metric,
            key="bench_ranking_metric",
            help="The metric used to rank algorithms (e.g. 'Accuracy', 'R2'). Leave blank for the task-aware default.",
        ).strip()
        top_k = int(
            st.number_input(
                "Show top models",
                min_value=1,
                value=int(_preset["top_k"]),
                step=1,
                key="bench_top_k",
                help="How many of the best-performing algorithms to highlight in the results. All tested algorithms still appear in the full leaderboard.",
            )
        )

        st.caption("**Data split**")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider(
                "Held-back data (%)",
                min_value=0.1,
                max_value=0.5,
                value=float(_preset["test_size"]),
                step=0.05,
                key="bench_test_size",
                help="How much data to hold back for testing. 0.2 means 20% is used for testing, 80% for training.",
            )
        with col2:
            random_state = int(
                st.number_input(
                    "Random seed",
                    value=int(settings.default_random_state),
                    step=1,
                    key="bench_random_state",
                    help="A fixed number that makes results reproducible. Use the same seed to get the same split every time.",
                )
            )

        sample_rows_input = int(
            st.number_input(
                "Sample rows (0 = full dataset)",
                min_value=0,
                value=int(_preset["sample_rows"]),
                step=1000,
                key="bench_sample_rows",
                help="Limit the number of rows used. Use 0 for the full dataset. Sampling speeds up large datasets at the cost of less representative results.",
            )
        )

        stratify_option = st.selectbox(
            "Balance categories in split",
            options=["auto", "true", "false"],
            index=0 if settings.default_stratify else 2,
            key="bench_stratify",
            help=(
                "When your target has imbalanced categories (e.g. 90% 'No' and 10% 'Yes'), "
                "stratification ensures both training and test sets get the same proportion. "
                "'Auto' enables this for classification tasks. 'True' forces it on. 'False' disables it."
            ),
        )

        st.caption("**Hardware & model filtering**")
        if gpu_info["cuda_available"]:
            st.caption(f"⚡ GPU detected: {gpu_info['device_name'] or 'available'}")
        prefer_gpu = st.checkbox(
            "Prefer GPU acceleration when supported",
            value=bool(settings.prefer_gpu),
            key="bench_prefer_gpu",
            help="Uses GPU acceleration for algorithms that support it. Only has an effect when a CUDA GPU is detected.",
        )
        include_models_raw = st.text_input(
            "Include only these algorithms (comma-separated)",
            key="bench_include_models",
            help=(
                "Only test specific algorithms — useful for re-running with a shortlist. "
                "Use the exact algorithm names from a previous leaderboard (e.g. 'RandomForestClassifier, XGBClassifier'). "
                "Leave blank to test all available algorithms."
            ),
        )
        exclude_models_raw = st.text_input(
            "Exclude these algorithms (comma-separated)",
            key="bench_exclude_models",
            help=(
                "Skip specific algorithms during benchmarking — useful if one consistently crashes or is too slow. "
                "Use exact names from the leaderboard."
            ),
        )

    # ── Step 3: Run Benchmark ─────────────────────────────────────────
    st.subheader("3. Run Benchmark")
    if len(df) > settings.sampling_row_threshold and sample_rows_input == 0:
        st.warning(
            f"This dataset exceeds the benchmark sampling threshold ({settings.sampling_row_threshold:,} rows). "
            f"Consider sampling around {settings.suggested_sample_rows:,} rows for a faster baseline run. "
            "Open **Advanced options** above to set a sample size."
        )

    requested_result_key = build_benchmark_run_key(selected_name, target_column, task_type)

    if st.button("Run Benchmark", key="bench_run"):
        if not is_lazypredict_available():
            st.error(
                "The benchmarking engine is not available in this environment. "
                "Ask your administrator to install the required benchmarking packages, "
                "or switch to **Train & Tune** which uses a different engine."
            )
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
        with st.spinner("Running benchmark — testing dozens of algorithms. This may take several minutes for large datasets…"):
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
                st.error(
                    f"Benchmark stopped: {safe_error_message(exc)}\n\n"
                    "**What to try:** Check that your target column and task type are correct, "
                    "or reduce the dataset size in **Advanced options**."
                )
                return
            except Exception as exc:
                st.error(
                    f"Benchmark failed unexpectedly: {safe_error_message(exc)}\n\n"
                    "**What to try:** Try a smaller sample size in **Advanced options**, "
                    "check that your data has no formatting issues, or reload the dataset from **Load Data**."
                )
                return

        bundles = st.session_state.setdefault("benchmark_bundles", {})
        bundles[requested_result_key] = bundle
        st.success("Benchmark complete.")

    bundle = st.session_state.get("benchmark_bundles", {}).get(requested_result_key)
    if bundle is None:
        st.info(
            "**Ready to benchmark.**\n\n"
            "Choose a target column above and click **Run Benchmark**.\n\n"
            "**What happens:** The system trains dozens of algorithms on your data and ranks them by performance. "
            "You'll get a leaderboard showing which algorithm works best — typically takes 1–5 minutes."
        )
        return

    # ── Step 4: Results ───────────────────────────────────────────────
    st.subheader("4. Results")
    _render_result_bundle(bundle)

    # ── Save a model (optional) ───────────────────────────────────────
    st.divider()
    with st.expander("💾 Save a model from this benchmark", expanded=False):
        st.caption(
            "**Optional** — Retrain a top algorithm on the same data split and save it for predictions. "
            "For a fully tuned model, go to **Train & Tune** instead."
        )
        _render_save_best_model(bundle)

    st.divider()
    render_past_runs_section(metadata_store, AppJobType.BENCHMARK, key_prefix="bench")
    render_saved_models_section(state.settings.prediction, key_prefix="bench")
    render_next_step_hint(current_step=3)


def _render_result_bundle(bundle) -> None:  # noqa: ANN001
    from app.pages.ui_labels import format_enum_value
    summary = bundle.summary

    # ── Plain-English summary ──────────────────────────────────────────
    _count = summary.model_count
    _best = summary.best_model_name or "N/A"
    _score = summary.best_score
    _metric = summary.ranking_metric
    _dur = f"{summary.benchmark_duration_seconds:.1f}"
    _fast = summary.fastest_model_name or "N/A"
    _fast_t = summary.fastest_model_time_seconds

    # Quality assessment
    if _count >= 10:
        _verdict = "Solid coverage — many algorithms were tested. The leaderboard gives a reliable picture of what works."
    elif _count >= 3:
        _verdict = "Decent coverage. For broader confidence, try the **Deep** run mode."
    else:
        _verdict = "Only a few algorithms were tested. Consider benchmarking with more data or the **Standard** / **Deep** run mode."

    st.info(
        f"**What happened:** Tested **{_count}** algorithms in **{_dur}s**. "
        f"Best performer: **{_best}** ({_metric}: **{_score}**).\n\n"
        f"**Quality:** {_verdict}\n\n"
        f"**Next step:** Go to **Train & Tune** to fine-tune **{_best}** (or another top pick) and save a production-ready model."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Best algorithm", _best, _score)
    col2.metric("Algorithms tested", _count)
    col3.metric("Total time", f"{_dur}s")

    _DIRECTION_LABELS = {"maximize": "higher is better", "minimize": "lower is better"}
    st.caption(
        f"Ranking score: **{_metric}** ({_DIRECTION_LABELS.get(summary.ranking_direction.value, summary.ranking_direction.value)}) · "
        f"Task: **{format_enum_value(summary.task_type.value)}** · "
        f"Fastest: **{_fast}** ({_fast_t}s)"
    )
    if bundle.mlflow_run_id:
        with st.expander("Technical details", expanded=False):
            st.caption(f"Tracking run ID: `{bundle.mlflow_run_id}` — use this to look up the run in MLflow.")

    if summary.warnings:
        with st.expander("Warnings"):
            for warning in summary.warnings:
                st.warning(warning)

    leaderboard_df = leaderboard_to_dataframe(bundle.leaderboard)
    st.subheader("Leaderboard")
    st.dataframe(leaderboard_df, width="stretch")

    # Metric explanations
    from app.pages.glossary import render_metric_legend
    _metric_cols = [c for c in leaderboard_df.columns if c not in ("Rank", "Model", "Task Type", "Benchmark Backend", "Run Timestamp", "Warnings")]
    render_metric_legend(_metric_cols, key_prefix="bench_legend")

    if bundle.top_models:
        st.subheader("Top Shortlist")
        st.caption("The highest-scoring algorithms from the benchmark — good candidates for **Train & Tune**.")
        st.table(leaderboard_to_dataframe(bundle.top_models))

    if bundle.artifacts:
        with st.expander("📤 Reports & Downloads", expanded=True):
            _bench_files = [
                ("Full results CSV", bundle.artifacts.raw_results_csv_path),
                ("Leaderboard CSV", bundle.artifacts.leaderboard_csv_path),
                ("Leaderboard data", bundle.artifacts.leaderboard_json_path),
                ("Summary data", bundle.artifacts.summary_json_path),
                ("Markdown summary", bundle.artifacts.markdown_summary_path),
                ("Score chart", bundle.artifacts.score_chart_path),
                ("Time chart", bundle.artifacts.training_time_chart_path),
            ]
            for _label, _path in _bench_files:
                if _path is None:
                    continue
                if _path.exists():
                    st.download_button(
                        label=f"Download {_label}",
                        data=_path.read_bytes(),
                        file_name=_path.name,
                        key=f"bench_dl_{_label}_{_path.name}",
                    )


def _resolve_estimator_class(model_name: str, task_type: BenchmarkTaskType):
    """Resolve a LazyPredict model name to its sklearn-compatible estimator class."""

    try:
        import lazypredict.Supervised as lazy_module

        attr = "CLASSIFIERS" if task_type == BenchmarkTaskType.CLASSIFICATION else "REGRESSORS"
        pairs = getattr(lazy_module, attr, None)
        if pairs:
            for name, cls in pairs:
                if name == model_name:
                    return cls
    except ImportError:
        pass

    # Fallback: try sklearn's all_estimators
    try:
        from sklearn.utils import all_estimators

        type_filter = "classifier" if task_type == BenchmarkTaskType.CLASSIFICATION else "regressor"
        for name, cls in all_estimators(type_filter=type_filter):
            if name == model_name:
                return cls
    except Exception:
        pass

    return None


def _render_save_best_model(bundle) -> None:  # noqa: ANN001
    """Render controls to retrain and save the best model from benchmark results."""

    if not bundle.leaderboard:
        return

    model_options = [row.model_name for row in bundle.leaderboard]
    selected_model = st.selectbox(
        "Algorithm to save",
        options=model_options,
        index=0,
        key="bench_save_model_select",
        help="Pick an algorithm from the leaderboard. It will be retrained on the same data split and saved as a model file you can use for predictions.",
    )

    if st.button("Retrain & Save", key="bench_save_model_btn"):
        estimator_cls = _resolve_estimator_class(selected_model, bundle.task_type)
        if estimator_cls is None:
            st.error(f"Could not find the algorithm '{selected_model}'. It may not be installed or supported for this task type.")
            return

        loaded_datasets = st.session_state.get("loaded_datasets", {})
        if not bundle.dataset_name or bundle.dataset_name not in loaded_datasets:
            st.error(
                "The original dataset is no longer loaded in the session. "
                "Re-load the dataset from Load Data before saving."
            )
            return

        loaded_dataset = loaded_datasets[bundle.dataset_name]
        df = loaded_dataset.dataframe
        config = bundle.config

        try:
            from sklearn.model_selection import train_test_split

            target = df[config.target_column]
            features = df.drop(columns=[config.target_column])

            # Drop non-numeric / object columns the same way LazyPredict does
            numeric_features = features.select_dtypes(include=["number", "bool"])

            X_train, _X_test, y_train, _y_test = train_test_split(
                numeric_features,
                target,
                test_size=config.split.test_size,
                random_state=config.split.random_state,
            )

            model = estimator_cls()
            model.fit(X_train, y_train)

            import joblib

            state = get_or_init_state()
            models_dir = state.settings.pycaret.models_dir
            models_dir.mkdir(parents=True, exist_ok=True)
            save_stem = model_save_name(bundle.dataset_name, selected_model)
            model_path = models_dir / f"{save_stem}.pkl"
            joblib.dump(model, model_path)

            # Write metadata sidecar
            metadata = {
                "source": "benchmark",
                "model_name": save_stem,
                "task_type": bundle.task_type.value,
                "target_column": config.target_column,
                "dataset_name": bundle.dataset_name,
                "dataset_fingerprint": bundle.dataset_fingerprint,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "feature_columns": list(X_train.columns),
                "split_test_size": config.split.test_size,
                "split_random_state": config.split.random_state,
                "model_path": str(model_path),
            }
            metadata_path = model_path.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            st.success(f"Model **{save_stem}** saved successfully.")
            with st.expander("Saved files", expanded=False):
                st.caption(f"Model file: **{model_path.name}**")
                st.caption(f"Metadata file: **{metadata_path.name}**")

        except Exception as exc:
            st.error(
                f"Failed to retrain and save model: {safe_error_message(exc)}\n\n"
                "**What to try:** Check that the models folder exists and is writable, "
                "or try saving a different algorithm from the leaderboard."
            )