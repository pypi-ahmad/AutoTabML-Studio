"""Generate professional run descriptions for job runs.

Two modes:
- **Template** (default): fast, deterministic Markdown descriptions built from
  job metadata — no AI required.
- **AI-enhanced** (opt-in): sends the template context to the configured AI
  provider for a richer, more explanatory write-up.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

from app.storage.models import AppJobType

logger = logging.getLogger(__name__)


# ── Template descriptions (no LLM) ────────────────────────────────────────

_TRACKING_OVERVIEW = (
    "Every aspect of this run is tracked — parameters, metrics, and "
    "output files — so you can reproduce, compare, and deploy with confidence."
)


def generate_template_description(
    job_type: AppJobType,
    *,
    dataset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    mlflow_run_id: str | None = None,
) -> str:
    """Return a rich Markdown description for a job run, built from templates."""
    meta = metadata or {}
    ds = dataset_name or "Unknown Dataset"

    header = f"## {_job_icon(job_type)} {job_type.value.title()} Run — {ds}\n\n"
    body = _JOB_TEMPLATES.get(job_type, _generic_template)(ds, meta, mlflow_run_id)
    footer = _build_footer(job_type, mlflow_run_id)
    return header + body + footer


def _benchmark_template(ds: str, meta: dict, run_id: str | None) -> str:
    best = meta.get("best_model_name", "N/A")
    score = meta.get("best_score")
    metric = meta.get("ranking_metric", "N/A")
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
    return (
        f"{_TRACKING_OVERVIEW}\n\n"
        f"### What happened\n"
        f"A **Quick Benchmark** evaluated dozens of algorithms on "
        f"**{ds}** and ranked them by **{metric}**.\n\n"
        f"### Key results\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Best algorithm | **{best}** |\n"
        f"| Best score ({metric}) | **{score_str}** |\n"
        f"| Ranking metric | {metric} |\n\n"
        f"### Tracking details\n"
        f"- **Parameters logged:** dataset name, target column, task type, "
        f"split ratio, random state\n"
        f"- **Metrics logged:** best score, algorithm count, per-algorithm scores\n"
        f"- **Output files:** leaderboard CSV, summary JSON, benchmark config\n\n"
        f"### How to use\n"
        f"1. Compare runs side-by-side in the **Compare** page\n"
        f"2. Head to **Train & Tune** to build a production model from the best algorithm\n"
        f"3. Use the saved model in **Predictions** to score new data\n"
    )


def _experiment_template(ds: str, meta: dict, run_id: str | None) -> str:
    best_baseline = meta.get("best_baseline_model_name", "N/A")
    tuned = meta.get("tuned_model_name", "N/A")
    selected = meta.get("selected_model_name", "N/A")
    saved = meta.get("saved_model_name", "N/A")
    task = meta.get("task_type", "N/A")
    return (
        f"{_TRACKING_OVERVIEW}\n\n"
        f"### What happened\n"
        f"A **Train & Tune** run ({task}) on **{ds}** — comparing baselines, "
        f"optionally tuning hyperparameters, and saving the final model.\n\n"
        f"### Pipeline stages\n"
        f"| Stage | Result |\n|---|---|\n"
        f"| Best baseline | **{best_baseline}** |\n"
        f"| Tuned model | {tuned} |\n"
        f"| Selected model | {selected} |\n"
        f"| Saved model | {saved} |\n\n"
        f"### Tracking details\n"
        f"- **Parameters:** dataset name, target, task type, fold strategy, "
        f"metric, preprocessing flags\n"
        f"- **Metrics:** compare-grid scores, tuned score, final metrics\n"
        f"- **Output files:** saved model (.pkl), experiment snapshot, plots, "
        f"metadata JSON\n\n"
        f"### How to use\n"
        f"1. Load the saved model in **Predictions** to score new data\n"
        f"2. Use **Test & Evaluate** to measure real-world accuracy\n"
        f"3. Register the model in the **Registry** for versioned deployment\n"
        f"4. Re-run with different settings to compare approaches\n"
    )


def _profiling_template(ds: str, meta: dict, run_id: str | None) -> str:
    rows = meta.get("row_count", "N/A")
    cols = meta.get("column_count", "N/A")
    mode = meta.get("report_mode", "standard")
    return (
        f"{_TRACKING_OVERVIEW}\n\n"
        f"### What happened\n"
        f"A **data profiling** report was generated "
        f"for **{ds}** ({rows:,} rows × {cols} columns) in *{mode}* mode.\n\n"
        f"### Report contents\n"
        f"- Variable distributions, types, and missing values\n"
        f"- Correlation analysis and duplicate detection\n"
        f"- Statistical summaries and alerts\n\n"
        f"### Tracking details\n"
        f"- **Parameters:** dataset name, row/column count, report mode, "
        f"sampling flag\n"
        f"- **Output files:** HTML report, summary JSON\n\n"
        f"### How to use\n"
        f"1. Review the HTML report for data quality insights\n"
        f"2. Use findings to inform feature engineering in **Train & Tune**\n"
        f"3. Compare profiles across different datasets in **History**\n"
    )


def _validation_template(ds: str, meta: dict, run_id: str | None) -> str:
    passed = meta.get("passed_count", 0)
    warned = meta.get("warning_count", 0)
    failed = meta.get("failed_count", 0)
    rows = meta.get("row_count", "N/A")
    cols = meta.get("column_count", "N/A")
    return (
        f"{_TRACKING_OVERVIEW}\n\n"
        f"### What happened\n"
        f"**Data validation** checked **{ds}** "
        f"({rows:,} rows × {cols} columns) against quality expectations.\n\n"
        f"### Validation results\n"
        f"| Check | Count |\n|---|---|\n"
        f"| ✅ Passed | {passed} |\n"
        f"| ⚠️ Warnings | {warned} |\n"
        f"| ❌ Failed | {failed} |\n\n"
        f"### Tracking details\n"
        f"- **Parameters:** dataset name, row/column count\n"
        f"- **Metrics:** passed/warning/failed counts\n"
        f"- **Output files:** validation summary JSON\n\n"
        f"### How to use\n"
        f"1. Fix any failed checks before training a model\n"
        f"2. Compare validation runs across dataset versions\n"
        f"3. Use as a data-quality gate in your ML pipeline\n"
    )


def _prediction_template(ds: str, meta: dict, run_id: str | None) -> str:
    model = meta.get("model_identifier", "N/A")
    mode = meta.get("mode", "N/A")
    task = meta.get("task_type", "N/A")
    row_count = meta.get("row_count", "N/A")
    return (
        f"{_TRACKING_OVERVIEW}\n\n"
        f"### What happened\n"
        f"**Predictions** were run on **{ds}** ({row_count} rows) "
        f"using model **{model}** ({task}, {mode} mode).\n\n"
        f"### Tracking details\n"
        f"- **Parameters:** model identifier, task type, input source\n"
        f"- **Output files:** prediction output CSV, summary JSON\n\n"
        f"### How to use\n"
        f"1. Download predictions from the output file\n"
        f"2. Compare prediction runs across models in **History**\n"
    )


def _generic_template(ds: str, meta: dict, run_id: str | None) -> str:
    return (
        f"{_TRACKING_OVERVIEW}\n\n"
        f"This run processed **{ds}**. "
        f"Check the History page for full details.\n"
    )


_JOB_TEMPLATES = {
    AppJobType.BENCHMARK: _benchmark_template,
    AppJobType.EXPERIMENT: _experiment_template,
    AppJobType.PROFILING: _profiling_template,
    AppJobType.VALIDATION: _validation_template,
    AppJobType.PREDICTION: _prediction_template,
}

_JOB_ICONS = {
    AppJobType.BENCHMARK: "📊",
    AppJobType.EXPERIMENT: "🧪",
    AppJobType.PROFILING: "📋",
    AppJobType.VALIDATION: "✅",
    AppJobType.PREDICTION: "🔮",
}


def _job_icon(job_type: AppJobType) -> str:
    return _JOB_ICONS.get(job_type, "📄")


def _build_footer(job_type: AppJobType, run_id: str | None) -> str:
    lines = ["\n---\n"]
    if run_id:
        lines.append(f"**Tracking Run ID:** `{run_id}`\n\n")
    lines.append(
        "💡 *Every future run on this dataset will automatically receive "
        "tracking descriptions like this one. View and compare them in "
        "the History page.*\n"
    )
    return "".join(lines)


# ── LLM-enhanced descriptions ─────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """\
You are an expert ML engineer writing professional run descriptions for
a machine learning studio application. Given the job context below, write a clear,
informative Markdown description that explains:

1. What this specific run did (job type, dataset, key results)
2. What was tracked (parameters, metrics, output files)
3. How the user can leverage this run (next steps, comparison, deployment)
4. Any notable findings from the metrics/results

Keep the description professional, concise (300-500 words), and practical.
Use Markdown formatting with headers, tables, and bullet points.
"""


def _build_llm_prompt(
    job_type: AppJobType,
    *,
    dataset_name: str | None,
    metadata: dict[str, Any] | None,
    mlflow_run_id: str | None,
) -> str:
    """Build the user prompt sent to the LLM."""
    meta = metadata or {}
    return (
        f"Job type: {job_type.value}\n"
        f"Dataset: {dataset_name or 'Unknown'}\n"
        f"MLflow Run ID: {mlflow_run_id or 'N/A'}\n"
        f"Metadata: {meta}\n\n"
        f"Write a professional MLflow run description for this job."
    )


def _run_async(coro):  # noqa: ANN001
    """Execute an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def generate_llm_description(
    job_type: AppJobType,
    *,
    dataset_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    mlflow_run_id: str | None = None,
    provider,  # noqa: ANN001 — BaseProvider instance
    model_id: str | None = None,
) -> str:
    """Generate an LLM-enhanced description using the configured provider.

    Falls back to the template description on any error.
    """
    prompt = (
        _LLM_SYSTEM_PROMPT + "\n\n"
        + _build_llm_prompt(
            job_type,
            dataset_name=dataset_name,
            metadata=metadata,
            mlflow_run_id=mlflow_run_id,
        )
    )
    try:
        result = _run_async(
            provider.generate_text(prompt, model_id=model_id, max_tokens=1024)
        )
        return result.strip() if result else generate_template_description(
            job_type,
            dataset_name=dataset_name,
            metadata=metadata,
            mlflow_run_id=mlflow_run_id,
        )
    except Exception as exc:
        logger.warning("LLM description generation failed: %s — falling back to template", exc)
        return generate_template_description(
            job_type,
            dataset_name=dataset_name,
            metadata=metadata,
            mlflow_run_id=mlflow_run_id,
        )
