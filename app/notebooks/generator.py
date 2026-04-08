"""Generate reproducible Jupyter notebooks from job history records."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


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
    """Generate a .ipynb notebook that reproduces a job for a dataset.

    Returns the path to the generated notebook file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = dataset_name.replace(" ", "_").lower()
    notebook_name = f"{job_type}_{safe_name}.ipynb"
    notebook_path = output_dir / notebook_name

    cells = []

    # Title
    cells.append(_md_cell(
        f"# {job_type.title()}: {dataset_name}\n\n"
        f"Auto-generated notebook for the **{job_type}** job on dataset **{dataset_name}**.\n\n"
        f"- **Task type:** {task_type or 'N/A'}\n"
        f"- **Target column:** {target_column or 'N/A'}\n"
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    ))

    # Setup cell
    cells.append(_code_cell(
        "# Setup\n"
        "import pandas as pd\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "print('Setup complete')"
    ))

    if job_type == "benchmark":
        cells.extend(_benchmark_cells(
            dataset_name=dataset_name,
            target_column=target_column,
            task_type=task_type,
            metadata=metadata or {},
            artifact_path=artifact_path,
        ))
    elif job_type == "experiment":
        cells.extend(_experiment_cells(
            dataset_name=dataset_name,
            target_column=target_column,
            task_type=task_type,
            metadata=metadata or {},
            artifact_path=artifact_path,
        ))
    elif job_type == "flaml":
        cells.extend(_flaml_cells(
            dataset_name=dataset_name,
            target_column=target_column,
            task_type=task_type,
            metadata=metadata or {},
            artifact_path=artifact_path,
        ))
    elif job_type == "profiling":
        cells.extend(_profiling_cells(
            dataset_name=dataset_name,
            artifact_path=artifact_path,
        ))
    elif job_type == "validation":
        cells.extend(_validation_cells(
            dataset_name=dataset_name,
            artifact_path=artifact_path,
            summary_path=summary_path,
        ))

    # Summary metadata cell
    if metadata:
        cells.append(_md_cell("## Job Metadata"))
        cells.append(_code_cell(
            f"metadata = {json.dumps(metadata, indent=2, default=str)}\n"
            "for k, v in metadata.items():\n"
            "    print(f'{k}: {v}')"
        ))

    # Artifact loading cell
    if artifact_path:
        cells.append(_md_cell("## Load Output Files"))
        cells.append(_code_cell(
            f"artifact_path = r'{artifact_path}'\n"
            "from pathlib import Path\n"
            "p = Path(artifact_path)\n"
            "if p.exists():\n"
            "    if p.suffix == '.csv':\n"
            "        df_artifact = pd.read_csv(p)\n"
            "        display(df_artifact)\n"
            "    elif p.suffix == '.json':\n"
            "        import json\n"
            "        data = json.loads(p.read_text())\n"
            "        print(json.dumps(data, indent=2))\n"
            "    else:\n"
            "        print(f'Artifact: {p}')\n"
            "else:\n"
            "    print(f'Artifact not found: {p}')"
        ))

    notebook = _build_notebook(cells)
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook_path


def _benchmark_cells(
    dataset_name: str,
    target_column: str | None,
    task_type: str | None,
    metadata: dict,
    artifact_path: str | None,
) -> list[dict]:
    cells = []
    cells.append(_md_cell(
        "## Benchmark Configuration\n\n"
        f"- **Best model:** {metadata.get('best_model_name', 'N/A')}\n"
        f"- **Best score:** {metadata.get('best_score', 'N/A')}\n"
        f"- **Ranking metric:** {metadata.get('ranking_metric', 'N/A')}"
    ))

    cells.append(_md_cell("## Reproduce Benchmark"))
    cells.append(_code_cell(
        f"# Load dataset and run LazyPredict benchmark\n"
        f"# Dataset: {dataset_name}\n"
        f"# Target: {target_column or 'TODO: set target column'}\n\n"
        f"# from app.modeling.benchmark.service import benchmark_dataset\n"
        f"# from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkSplitConfig, BenchmarkTaskType\n"
        f"#\n"
        f"# config = BenchmarkConfig(\n"
        f"#     target_column='{target_column or 'target'}',\n"
        f"#     task_type=BenchmarkTaskType.{(task_type or 'classification').upper()},\n"
        f"# )\n"
        f"# bundle = benchmark_dataset(df, config, dataset_name='{dataset_name}')\n"
        f"print('Uncomment above to reproduce')"
    ))

    if artifact_path:
        cells.append(_md_cell("## Leaderboard"))
        cells.append(_code_cell(
            f"leaderboard_path = r'{artifact_path}'\n"
            "from pathlib import Path\n"
            "p = Path(leaderboard_path)\n"
            "if p.exists() and p.suffix == '.csv':\n"
            "    leaderboard = pd.read_csv(p)\n"
            "    display(leaderboard)\n"
            "else:\n"
            "    print(f'Leaderboard not found: {p}')"
        ))

    return cells


def _experiment_cells(
    dataset_name: str,
    target_column: str | None,
    task_type: str | None,
    metadata: dict,
    artifact_path: str | None,
) -> list[dict]:
    cells = []
    cells.append(_md_cell(
        "## Experiment Summary\n\n"
        f"- **Best baseline:** {metadata.get('best_baseline_model_name', 'N/A')}\n"
        f"- **Tuned model:** {metadata.get('tuned_model_name', 'N/A')}\n"
        f"- **Selected model:** {metadata.get('selected_model_name', 'N/A')}\n"
        f"- **Saved model:** {metadata.get('saved_model_name', 'N/A')}"
    ))

    cells.append(_md_cell("## Load Saved Model"))
    if artifact_path:
        cells.append(_code_cell(
            f"model_path = r'{artifact_path}'\n"
            "from pathlib import Path\n"
            "p = Path(model_path)\n"
            "if p.exists():\n"
            "    from app.modeling.pycaret.persistence import load_model_artifact\n"
            f"    from app.modeling.pycaret.schemas import ExperimentTaskType\n"
            f"    model = load_model_artifact(ExperimentTaskType.{(task_type or 'classification').upper()}, model_path)\n"
            "    print(f'Model loaded: {type(model).__name__}')\n"
            "else:\n"
            "    print(f'Model not found: {p}')"
        ))
    else:
        cells.append(_code_cell("print('No saved model artifact path recorded.')"))

    return cells


def _flaml_cells(
    dataset_name: str,
    target_column: str | None,
    task_type: str | None,
    metadata: dict,
    artifact_path: str | None,
) -> list[dict]:
    cells = []
    cells.append(_md_cell(
        "## FLAML AutoML Summary\n\n"
        f"- **Best estimator:** {metadata.get('best_estimator', 'N/A')}\n"
        f"- **Best loss:** {metadata.get('best_loss', 'N/A')}\n"
        f"- **Metric:** {metadata.get('metric', 'N/A')}\n"
        f"- **Search duration:** {metadata.get('search_duration_seconds', 'N/A')}s"
    ))

    cells.append(_md_cell("## Reproduce FLAML Search"))
    cells.append(_code_cell(
        f"# Load dataset and run FLAML AutoML\n"
        f"# Dataset: {dataset_name}\n"
        f"# Target: {target_column or 'TODO: set target column'}\n\n"
        f"from flaml import AutoML\n"
        f"import pandas as pd\n\n"
        f"# df = pd.read_csv('path/to/your/data.csv')\n"
        f"# automl = AutoML()\n"
        f"# automl.fit(\n"
        f"#     X_train=df.drop(columns=['{target_column or 'target'}']),\n"
        f"#     y_train=df['{target_column or 'target'}'],\n"
        f"#     task='{task_type or 'classification'}',\n"
        f"#     time_budget=120,\n"
        f"# )\n"
        f"# print(f'Best estimator: {{automl.best_estimator}}')\n"
        f"# print(f'Best config: {{automl.best_config}}')\n"
        f"print('Uncomment above to reproduce')"
    ))

    if artifact_path:
        cells.append(_md_cell("## Load Saved FLAML Model"))
        cells.append(_code_cell(
            f"import pickle\n"
            f"from pathlib import Path\n\n"
            f"model_path = Path(r'{artifact_path}')\n"
            "if model_path.exists():\n"
            "    with model_path.open('rb') as f:\n"
            "        automl = pickle.load(f)\n"
            "    print(f'Model loaded: {type(automl).__name__}')\n"
            "    print(f'Best estimator: {automl.best_estimator}')\n"
            "else:\n"
            "    print(f'Model not found: {model_path}')"
        ))

    return cells


def _profiling_cells(dataset_name: str, artifact_path: str | None) -> list[dict]:
    cells = []
    cells.append(_md_cell("## Profiling Report"))
    if artifact_path:
        cells.append(_code_cell(
            f"report_path = r'{artifact_path}'\n"
            "from pathlib import Path\n"
            "from IPython.display import HTML, display\n"
            "p = Path(report_path)\n"
            "if p.exists() and p.suffix == '.html':\n"
            "    display(HTML(f'<iframe src=\"{p}\" width=\"100%\" height=\"600\"></iframe>'))\n"
            "else:\n"
            "    print(f'Report: {p}')"
        ))
    else:
        cells.append(_code_cell("print('No profiling report artifact found.')"))
    return cells


def _validation_cells(
    dataset_name: str,
    artifact_path: str | None,
    summary_path: str | None,
) -> list[dict]:
    cells = []
    cells.append(_md_cell("## Validation Results"))
    path = summary_path or artifact_path
    if path:
        cells.append(_code_cell(
            f"import json\nfrom pathlib import Path\n"
            f"p = Path(r'{path}')\n"
            "if p.exists():\n"
            "    data = json.loads(p.read_text())\n"
            "    for k, v in data.items():\n"
            "        print(f'{k}: {v}')\n"
            "else:\n"
            "    print(f'Validation summary not found: {p}')"
        ))
    else:
        cells.append(_code_cell("print('No validation artifact found.')"))
    return cells


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n"),
    }


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n"),
    }


def _build_notebook(cells: list[dict]) -> dict:
    # Fix source lines: add \n to all but last line per cell
    for cell in cells:
        lines = cell["source"]
        if len(lines) > 1:
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }
