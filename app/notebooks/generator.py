"""Generate reproducible Jupyter notebooks from job history records.

All user-derived values (dataset names, task types, paths, metadata) are
serialized via :func:`json.dumps` and embedded as JSON literals rather than
interpolated into Python source. Code cells therefore contain only static
templates plus JSON-quoted constants, so a hostile string (with quotes,
backslashes, newlines, or null bytes) cannot break out of its literal and
inject executable code.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# ── Allowlists & validation ────────────────────────────────────────────

ALLOWED_JOB_TYPES: frozenset[str] = frozenset(
    {"benchmark", "experiment", "flaml", "profiling", "validation"}
)
ALLOWED_TASK_TYPES: frozenset[str] = frozenset({"classification", "regression"})

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_DATASET_NAME_LEN = 80
_MAX_TARGET_COLUMN_LEN = 200
_MAX_PATH_LEN = 4096


class NotebookGenerationError(ValueError):
    """Raised when notebook inputs fail validation."""


def _markdown_cell(source: str):
    return new_markdown_cell(source, metadata={"language": "markdown"})


def _python_cell(source: str):
    return new_code_cell(source, metadata={"language": "python"})


def _safe_filename_component(value: str, fallback: str) -> str:
    cleaned = _FILENAME_SAFE_RE.sub("_", value.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        return fallback
    return cleaned[:_MAX_DATASET_NAME_LEN]


def _validate_job_type(job_type: str) -> str:
    normalized = (job_type or "").strip().lower()
    if normalized not in ALLOWED_JOB_TYPES:
        raise NotebookGenerationError(
            f"Unsupported job_type {job_type!r}; expected one of {sorted(ALLOWED_JOB_TYPES)}."
        )
    return normalized


def _validate_task_type(task_type: str | None) -> str | None:
    if task_type is None:
        return None
    normalized = str(task_type).strip().lower()
    if not normalized:
        return None
    if normalized not in ALLOWED_TASK_TYPES:
        raise NotebookGenerationError(
            f"Unsupported task_type {task_type!r}; expected one of {sorted(ALLOWED_TASK_TYPES)}."
        )
    return normalized


def _validate_path(path_value: str | None, *, label: str) -> str | None:
    """Sanity-check a path string before embedding it as a JSON literal.

    Returns the canonicalized absolute path string, or ``None`` if input was
    ``None``. Rejects paths containing NUL bytes or pathologically long
    inputs. Existence is *not* required — the artifact may live on a
    different machine when the notebook is later opened.
    """

    if path_value is None:
        return None
    text = str(path_value)
    if "\x00" in text:
        raise NotebookGenerationError(f"{label} contains a NUL byte.")
    if len(text) > _MAX_PATH_LEN:
        raise NotebookGenerationError(f"{label} exceeds maximum length.")
    try:
        resolved = Path(text).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise NotebookGenerationError(f"{label} is not a valid path: {exc}") from exc
    return str(resolved)


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    try:
        resolved = output_dir.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise NotebookGenerationError(f"Invalid output_dir: {exc}") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise NotebookGenerationError(f"output_dir is not a directory: {resolved}")
    return resolved


def _safe_text(value: Any, *, max_len: int = 500) -> str:
    """Render an arbitrary value as a single-line markdown-safe string."""

    text = "" if value is None else str(value)
    # Strip control characters; collapse newlines so injected text cannot
    # break out of a markdown bullet line.
    text = "".join(ch if ch.isprintable() or ch in (" ", "\t") else " " for ch in text)
    text = text.replace("\r", " ").replace("\n", " ")
    # Neutralize backticks so values cannot open or close markdown code spans.
    text = text.replace("`", "ʼ")
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def _json_literal(value: Any) -> str:
    """Return a Python expression that evaluates to ``value`` at runtime.

    The expression form ``json.loads(<json string literal>)`` is used so the
    embedded string is just a JSON-escaped constant — no f-string
    interpolation, no ``repr`` of arbitrary objects.
    """

    return "json.loads(" + json.dumps(json.dumps(value, default=str)) + ")"


def _json_string(value: str) -> str:
    """Return a JSON-quoted string literal safe to embed as a Python constant."""

    return json.dumps(value)


# ── Public API ─────────────────────────────────────────────────────────


def generate_job_notebook(
    *,
    dataset_name: str,
    job_type: str,
    task_type: str | None = None,
    target_column: str | None = None,
    metadata: dict | None = None,
    artifact_path: str | None = None,
    summary_path: str | None = None,
    output_dir: Path,
) -> Path:
    """Generate a ``.ipynb`` notebook that documents a completed job.

    All user-derived values are validated and embedded via JSON literals so
    that no string interpolation reaches executable Python source.
    """

    job_type_norm = _validate_job_type(job_type)
    task_type_norm = _validate_task_type(task_type)
    artifact_path_norm = _validate_path(artifact_path, label="artifact_path")
    summary_path_norm = _validate_path(summary_path, label="summary_path")
    output_dir_norm = _validate_output_dir(output_dir)

    dataset_name_text = _safe_text(dataset_name or "unknown", max_len=_MAX_DATASET_NAME_LEN)
    target_column_text = (
        _safe_text(target_column, max_len=_MAX_TARGET_COLUMN_LEN) if target_column else None
    )
    metadata_clean: dict = dict(metadata or {})

    safe_name = _safe_filename_component(dataset_name_text, fallback="dataset")
    notebook_name = f"{job_type_norm}_{safe_name}.ipynb"
    notebook_path = (output_dir_norm / notebook_name).resolve()
    # Final guard: the resolved file must remain inside output_dir.
    if not notebook_path.is_relative_to(output_dir_norm):
        raise NotebookGenerationError(
            "Computed notebook path escapes output_dir; refusing to write."
        )

    cells: list[Any] = []

    # ── Title ──────────────────────────────────────────────────────────
    cells.append(
        _markdown_cell(
            "\n".join(
                [
                    f"# {_safe_text(job_type_norm.title())}: {dataset_name_text}",
                    "",
                    f"Auto-generated notebook for the **{_safe_text(job_type_norm)}** job "
                    f"on dataset **{dataset_name_text}**.",
                    "",
                    f"- **Task type:** {_safe_text(task_type_norm or 'N/A')}",
                    f"- **Target column:** {_safe_text(target_column_text or 'N/A')}",
                    f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ]
            )
        )
    )

    # ── Setup ──────────────────────────────────────────────────────────
    cells.append(
        _python_cell(
            "# Setup\n"
            "import json\n"
            "import warnings\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "warnings.filterwarnings('ignore')\n"
            "print('Setup complete')"
        )
    )

    # ── Per-job-type body ──────────────────────────────────────────────
    body_builders = {
        "benchmark": _benchmark_cells,
        "experiment": _experiment_cells,
        "flaml": _flaml_cells,
        "profiling": _profiling_cells,
        "validation": _validation_cells,
    }
    builder = body_builders[job_type_norm]
    cells.extend(
        builder(
            dataset_name=dataset_name_text,
            target_column=target_column_text,
            task_type=task_type_norm,
            metadata=metadata_clean,
            artifact_path=artifact_path_norm,
            summary_path=summary_path_norm,
        )
    )

    # ── Metadata ───────────────────────────────────────────────────────
    if metadata_clean:
        cells.append(_markdown_cell("## Job Metadata"))
        cells.append(
            _python_cell(
                f"metadata = {_json_literal(metadata_clean)}\n"
                "for k, v in metadata.items():\n"
                "    print(f'{k}: {v}')"
            )
        )

    # ── Generic artifact loader ────────────────────────────────────────
    if artifact_path_norm:
        cells.append(_markdown_cell("## Load Output Files"))
        cells.append(
            _python_cell(
                f"artifact_path = Path({_json_string(artifact_path_norm)})\n"
                "if artifact_path.exists():\n"
                "    if artifact_path.suffix == '.csv':\n"
                "        df_artifact = pd.read_csv(artifact_path)\n"
                "        display(df_artifact)\n"
                "    elif artifact_path.suffix == '.json':\n"
                "        data = json.loads(artifact_path.read_text())\n"
                "        print(json.dumps(data, indent=2))\n"
                "    else:\n"
                "        print(f'Artifact: {artifact_path}')\n"
                "else:\n"
                "    print(f'Artifact not found: {artifact_path}')"
            )
        )

    # ── Build & validate notebook ──────────────────────────────────────
    notebook = new_notebook(cells=cells)
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    # nbformat.validate raises ValidationError if the structure is malformed.
    nbformat.validate(notebook)

    with notebook_path.open("w", encoding="utf-8") as fh:
        nbformat.write(notebook, fh)
    return notebook_path


# ── Per-job builders ───────────────────────────────────────────────────


def _benchmark_cells(
    *,
    dataset_name: str,
    target_column: str | None,
    task_type: str | None,
    metadata: dict,
    artifact_path: str | None,
    summary_path: str | None,  # noqa: ARG001
) -> list[Any]:
    cells: list[Any] = [
        _markdown_cell(
            "\n".join(
                [
                    "## Benchmark Configuration",
                    "",
                    f"- **Best model:** {_safe_text(metadata.get('best_model_name', 'N/A'))}",
                    f"- **Best score:** {_safe_text(metadata.get('best_score', 'N/A'))}",
                    f"- **Ranking metric:** {_safe_text(metadata.get('ranking_metric', 'N/A'))}",
                ]
            )
        ),
        _markdown_cell("## Reproduce Benchmark"),
    ]

    target_for_code = target_column or "target"
    task_for_code = task_type or "classification"
    cells.append(
        _python_cell(
            "# Reproduction template — fill in the dataset path before running.\n"
            f"dataset_name = {_json_string(dataset_name)}\n"
            f"target_column = {_json_string(target_for_code)}\n"
            f"task_type = {_json_string(task_for_code)}\n"
            "\n"
            "# from app.modeling.benchmark.service import benchmark_dataset\n"
            "# from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkTaskType\n"
            "#\n"
            "# config = BenchmarkConfig(\n"
            "#     target_column=target_column,\n"
            "#     task_type=BenchmarkTaskType(task_type),\n"
            "# )\n"
            "# bundle = benchmark_dataset(df, config, dataset_name=dataset_name)\n"
            "print('Uncomment above to reproduce')"
        )
    )

    if artifact_path:
        cells.append(_markdown_cell("## Leaderboard"))
        cells.append(
            _python_cell(
                f"leaderboard_path = Path({_json_string(artifact_path)})\n"
                "if leaderboard_path.exists() and leaderboard_path.suffix == '.csv':\n"
                "    leaderboard = pd.read_csv(leaderboard_path)\n"
                "    display(leaderboard)\n"
                "else:\n"
                "    print(f'Leaderboard not found: {leaderboard_path}')"
            )
        )

    return cells


def _experiment_cells(
    *,
    dataset_name: str,  # noqa: ARG001
    target_column: str | None,  # noqa: ARG001
    task_type: str | None,
    metadata: dict,
    artifact_path: str | None,
    summary_path: str | None,  # noqa: ARG001
) -> list[Any]:
    cells: list[Any] = [
        _markdown_cell(
            "\n".join(
                [
                    "## Experiment Summary",
                    "",
                    f"- **Best baseline:** {_safe_text(metadata.get('best_baseline_model_name', 'N/A'))}",
                    f"- **Tuned model:** {_safe_text(metadata.get('tuned_model_name', 'N/A'))}",
                    f"- **Selected model:** {_safe_text(metadata.get('selected_model_name', 'N/A'))}",
                    f"- **Saved model:** {_safe_text(metadata.get('saved_model_name', 'N/A'))}",
                ]
            )
        ),
        _markdown_cell("## Load Saved Model"),
    ]

    if artifact_path:
        task_for_code = task_type or "classification"
        cells.append(
            _python_cell(
                f"model_path = Path({_json_string(artifact_path)})\n"
                f"task_type = {_json_string(task_for_code)}\n"
                "from app.security.trusted_artifacts import verify_local_artifact\n"
                "from app.modeling.pycaret.persistence import load_model_artifact\n"
                "from app.modeling.pycaret.schemas import ExperimentTaskType\n"
                "if model_path.exists():\n"
                "    verify_local_artifact(model_path, trusted_roots=[model_path.parent], label='model artifact')\n"
                "    model = load_model_artifact(ExperimentTaskType(task_type), str(model_path))\n"
                "    print(f'Model loaded: {type(model).__name__}')\n"
                "else:\n"
                "    print(f'Model not found: {model_path}')"
            )
        )
    else:
        cells.append(_python_cell("print('No saved model artifact path recorded.')"))

    return cells


def _flaml_cells(
    *,
    dataset_name: str,
    target_column: str | None,
    task_type: str | None,
    metadata: dict,
    artifact_path: str | None,
    summary_path: str | None,  # noqa: ARG001
) -> list[Any]:
    cells: list[Any] = [
        _markdown_cell(
            "\n".join(
                [
                    "## FLAML AutoML Summary",
                    "",
                    f"- **Best estimator:** {_safe_text(metadata.get('best_estimator', 'N/A'))}",
                    f"- **Best loss:** {_safe_text(metadata.get('best_loss', 'N/A'))}",
                    f"- **Metric:** {_safe_text(metadata.get('metric', 'N/A'))}",
                    f"- **Search duration:** {_safe_text(metadata.get('search_duration_seconds', 'N/A'))}s",
                ]
            )
        ),
        _markdown_cell("## Reproduce FLAML Search"),
    ]

    target_for_code = target_column or "target"
    task_for_code = task_type or "classification"
    cells.append(
        _python_cell(
            f"dataset_name = {_json_string(dataset_name)}\n"
            f"target_column = {_json_string(target_for_code)}\n"
            f"task_type = {_json_string(task_for_code)}\n"
            "\n"
            "# from flaml import AutoML\n"
            "# df = pd.read_csv('path/to/your/data.csv')\n"
            "# automl = AutoML()\n"
            "# automl.fit(\n"
            "#     X_train=df.drop(columns=[target_column]),\n"
            "#     y_train=df[target_column],\n"
            "#     task=task_type,\n"
            "#     time_budget=120,\n"
            "# )\n"
            "# print(f'Best estimator: {automl.best_estimator}')\n"
            "print('Uncomment above to reproduce')"
        )
    )

    if artifact_path:
        cells.append(_markdown_cell("## Load Saved FLAML Model (verified pickle)"))
        cells.append(
            _python_cell(
                "from app.security.trusted_artifacts import (\n"
                "    load_verified_pickle_artifact,\n"
                "    verify_local_artifact,\n"
                ")\n"
                f"model_path = Path({_json_string(artifact_path)})\n"
                "if model_path.exists():\n"
                "    verified = verify_local_artifact(\n"
                "        model_path,\n"
                "        trusted_roots=[model_path.parent],\n"
                "        label='FLAML model artifact',\n"
                "    )\n"
                "    automl = load_verified_pickle_artifact(\n"
                "        model_path,\n"
                "        trusted_roots=[model_path.parent],\n"
                "        expected_sha256=verified.checksum,\n"
                "    )\n"
                "    print(f'Model loaded: {type(automl).__name__}')\n"
                "    print(f'Best estimator: {automl.best_estimator}')\n"
                "else:\n"
                "    print(f'Model not found: {model_path}')"
            )
        )

    return cells


def _profiling_cells(
    *,
    dataset_name: str,  # noqa: ARG001
    target_column: str | None,  # noqa: ARG001
    task_type: str | None,  # noqa: ARG001
    metadata: dict,  # noqa: ARG001
    artifact_path: str | None,
    summary_path: str | None,  # noqa: ARG001
) -> list[Any]:
    cells: list[Any] = [_markdown_cell("## Profiling Report")]
    if artifact_path:
        cells.append(
            _python_cell(
                "from IPython.display import HTML, display\n"
                f"report_path = Path({_json_string(artifact_path)})\n"
                "if report_path.exists() and report_path.suffix == '.html':\n"
                "    display(HTML(report_path.read_text(encoding='utf-8')))\n"
                "else:\n"
                "    print(f'Report: {report_path}')"
            )
        )
    else:
        cells.append(_python_cell("print('No profiling report artifact found.')"))
    return cells


def _validation_cells(
    *,
    dataset_name: str,  # noqa: ARG001
    target_column: str | None,  # noqa: ARG001
    task_type: str | None,  # noqa: ARG001
    metadata: dict,  # noqa: ARG001
    artifact_path: str | None,
    summary_path: str | None,
) -> list[Any]:
    cells: list[Any] = [_markdown_cell("## Validation Results")]
    path = summary_path or artifact_path
    if path:
        cells.append(
            _python_cell(
                f"summary_path = Path({_json_string(path)})\n"
                "if summary_path.exists():\n"
                "    data = json.loads(summary_path.read_text())\n"
                "    for k, v in data.items():\n"
                "        print(f'{k}: {v}')\n"
                "else:\n"
                "    print(f'Validation summary not found: {summary_path}')"
            )
        )
    else:
        cells.append(_python_cell("print('No validation artifact found.')"))
    return cells
