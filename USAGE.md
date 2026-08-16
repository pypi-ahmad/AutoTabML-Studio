# AutoTabML Studio — Usage Guide

> Complete reference for the Streamlit UI and CLI. Covers every page, every command, and every configuration knob.

---

## Contents

- [Who this is for](#who-this-is-for)
- [Before you start](#before-you-start)
- [Starting the app](#starting-the-app)
- [Core workflow: Auto Run](#core-workflow-auto-run)
- [Pages reference](#pages-reference)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Input & output formats](#input--output-formats)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

---

## Who this is for

AutoTabML Studio is for **data scientists and ML engineers** who want to go from a raw CSV to a trained, evaluated, and deployable model — entirely on their own machine, without provisioning cloud infrastructure.

You bring your own API keys and data. Everything runs locally. No data leaves your machine unless you explicitly configure a remote LLM provider for AI summaries.

---

## Before you start

### Requirements

| Requirement | Notes |
| --- | --- |
| Python 3.10 – 3.13 | Use 3.11 or 3.12 for the broadest extra compatibility (PyCaret requires <3.13; TabFM requires >=3.11) |
| [uv](https://docs.astral.sh/uv/) | Fast Python package manager — install with `pip install uv` |
| OS | Windows, macOS, or Linux |

### Clone and install

```bash
git clone https://github.com/pypi-ahmad/AutoTabML-Studio.git
cd AutoTabML-Studio

# Install the base environment (UI + CLI, no ML engines)
uv sync --locked --group dev
```

### Optional extras

Install only the extras you need. Each adds a specific ML engine or integration:

| Extra | Installs | Python constraint |
| --- | --- | --- |
| `benchmark` | LazyPredict, XGBoost, LightGBM, CatBoost baselines | 3.10 – 3.13 |
| `experiment` | PyCaret full pipeline (compare, tune, evaluate, save) | < 3.13 only |
| `flaml` | Microsoft FLAML AutoML | 3.10 – 3.13 |
| `validation` | Great Expectations data validation | 3.10 – 3.13 |
| `profiling` | ydata-profiling EDA reports | 3.10 – 3.13 (**conflicts with `tabfm`**) |
| `tabfm` | Google TabFM research weights | >= 3.11 (**conflicts with `profiling`**) |
| `timesfm` | Google TimesFM 2.5 forecasting | 3.10 – 3.13 |
| `gpu` | GPU-enabled XGBoost, LightGBM, CatBoost | 3.10 – 3.13 |
| `kaggle` | Kaggle dataset download | 3.10 – 3.13 |
| `uci` | UCI ML Repository access | 3.10 – 3.13 |
| `colab` | Google Colab MCP remote execution | 3.10 – 3.13 |
| `providers` | OpenAI, Anthropic, Gemini, Ollama LLM summaries | 3.10 – 3.13 |
| `explain` | SHAP model explainability | 3.10 – 3.13 |
| `serve` | FastAPI + Uvicorn deployment server | 3.10 – 3.13 |

```bash
# Examples — combine extras as needed
uv sync --locked --group dev --extra benchmark
uv sync --locked --group dev --extra experiment
uv sync --locked --group dev --extra flaml
uv sync --locked --group dev --extra providers
```

> [!IMPORTANT]
> `tabfm` and `profiling` cannot be installed in the same virtual environment.
> They have incompatible `typeguard` requirements (`<3` vs `>=4`).
> Sync them into separate environments if you need both workflows.

---

## Starting the app

### Initialize storage (first run only)

```bash
uv run autotabml init-local-storage
```

Creates the artifact directories and the app metadata SQLite database. Safe to run again — it is idempotent.

### Verify your environment

```bash
uv run autotabml doctor
```

Reports CUDA availability, database status, artifact directories, optional dependency groups, and any stale files. Run this after install to confirm everything is wired up.

### Launch the Streamlit UI

**Windows** — double-click `Launch AutoTabML Studio.cmd` in the repo root. Opens `http://localhost:8561` in your browser automatically.

**Terminal:**

```bash
uv run streamlit run app/main.py
```

### Notebook mode

Set `AUTOTABML_WORKSPACE_MODE=notebook` to start in the Notebook page instead of the Dashboard. Useful for headless or Colab-style environments.

> [!NOTE]
> The default execution backend is `colab_mcp` (Google Colab MCP remote execution).
> If you are running everything locally, set `AUTOTABML_EXECUTION__BACKEND=local`
> in your `.env` file or environment before starting the app.

---

## Core workflow: Auto Run

**Auto Run** is the recommended end-to-end path. It submits a full guided AutoML pipeline as a background job and saves the model, holdout evaluation, provenance, and a drift baseline automatically.

### When to use Auto Run

- You want a trained, evaluated, saved model in one click or one command.
- You are not sure which algorithm or hyperparameter strategy to use.
- You want reproducible results with minimal configuration.

Use the manual workflow (individual pages or CLI commands) when you need fine-grained control over each step, want to compare specific algorithms, or are working with time-series data.

### UI path

1. Open **Auto Run** from the sidebar.
2. Load or select your dataset.
3. Select the target column. The app infers task type (classification vs. regression) automatically.
4. Review the inferred FLAML plan (time budget, mode, estimators).
5. Click **Start Auto Run**. A background job is submitted; progress updates in real time.

When the job completes, the model is saved to `artifacts/models/`, MLflow run logged, and a drift baseline written.

### CLI path

```bash
uv run autotabml auto-run data/train.csv --target price --mode auto --time-budget 120
```

Check job progress:

```bash
uv run autotabml job-list
uv run autotabml job-status <job-id>
```

Cancel if needed:

```bash
uv run autotabml job-cancel <job-id>
```

---

## Pages reference

### Home (Dashboard)

Shows workflow progress, recent activity, and quick links to every step. Use it as your launch pad — it highlights which steps have been completed and what to do next.

**Tip:** The Dashboard shows the most recent dataset and model. Navigate directly to any step from here.

---

### Load Data

**Section: ① Prepare**

Load a dataset from any supported source:

| Source | How to use |
| --- | --- |
| Local file | Drag and drop or browse for a CSV, Excel, or TSV file |
| Remote URL | Paste a direct link to a CSV or Excel file |
| UCI ML Repository | Search by name or ID — fetches and caches automatically |
| Kaggle | Enter a dataset slug (requires `KAGGLE_USERNAME` and `KAGGLE_KEY`; `kaggle` extra) |

Outputs a loaded dataset stored in session state for downstream pages.

> [!NOTE]
> Kaggle is available in the UI when the `kaggle` extra is installed and credentials are set.
> Large datasets (>100 K rows) are automatically sampled to 50 K rows for profiling and benchmarking.

---

### Validation

**Section: ① Prepare**

Checks the loaded dataset for quality issues before modeling:

- Null value counts and percentages (warn at 50%, fail at 95%)
- Duplicate rows
- Schema consistency
- Optional Great Expectations suite (requires `validation` extra)

Results are saved to `artifacts/validation/` as a summary JSON and optional GX data docs.

**Tip:** Run Validation before Profiling — it catches hard blockers early.

---

### Profiling

**Section: ① Prepare**

Generates a full EDA report using `ydata-profiling` (requires `profiling` extra):

- Distributions, correlations, missing values, interactions
- HTML report + machine-readable `summary.json`
- Sampling applied automatically for large datasets (>50 K rows: standard mode; >200 K: minimal mode)

Output saved to `artifacts/profiling/`.

> [!WARNING]
> `profiling` and `tabfm` cannot be installed together. See [Before you start](#before-you-start).

---

### Quick Benchmark

**Section: ② Build**

Screens 30+ algorithms in seconds using LazyPredict (requires `benchmark` extra):

- Trains all classifiers or regressors without tuning
- Ranks results by Balanced Accuracy (classification) or Adjusted R² (regression)
- Output: ranked leaderboard + MLflow run

Use this to identify the top 3–5 candidates before investing in tuning.

**Tip:** Sort the leaderboard by your target metric before proceeding to Train & Tune.

---

### Train & Tune

**Section: ② Build**

Full PyCaret experiment pipeline (requires `experiment` extra, Python < 3.13):

| Step | What it does |
| --- | --- |
| Compare | Runs `compare_models()` across all estimators; returns a ranked leaderboard |
| Tune | Runs `tune_model()` on a selected model ID; grid or random search |
| Evaluate | Generates evaluation plots (confusion matrix, AUC, residuals, etc.) |
| Save | Finalizes and saves the model to `artifacts/models/` with a SHA256 sidecar |

All steps log metrics and artifacts to MLflow automatically.

> [!NOTE]
> PyCaret requires Python < 3.13. On Python 3.13, use FLAML or the benchmark page instead.

---

### FLAML AutoML

**Section: ② Build**

Microsoft FLAML automated model selection (requires `flaml` extra):

- Time-budget or iteration-budget search
- Supports XGBoost, LightGBM, random forest, extra trees, linear models, k-NN
- Logs leaderboard and best model to MLflow

Use FLAML when you want fast AutoML without PyCaret's Python version constraint.

---

### Foundation Models

**Section: ② Build**

Two local foundation models for specialized use cases:

#### Google TabFM (research only)

Tabular foundation model for classification and regression. Non-commercial, research-only weights.

> [!WARNING]
> TabFM requires explicit license acceptance (`tabfm-non-commercial-v1.0`).
> TabFM-derived model contexts cannot be promoted in the registry or exported for deployment.
> Requires the `tabfm` extra and Python >= 3.11.

- Downloads a pinned Hugging Face snapshot on first use (opt-in required)
- Runs holdout evaluation; optionally saves a context artifact

#### Google TimesFM 2.5

Time-series forecasting with point and quantile (q10–q90) predictions.

- Requires the `timesfm` extra
- Downloads a pinned Hugging Face snapshot on first use (opt-in required)
- Supports single and grouped series; configurable horizon and frequency
- Optional holdout backtest

---

### Predictions

**Section: unlabeled**

Two modes in one page:

**Batch scoring** — score a full CSV file using any saved model. Outputs predictions CSV to `artifacts/predictions/`.

**Model testing** — score a file that includes ground-truth labels; computes accuracy/RMSE against actuals.

Select a model from the dropdown (shows all models in `artifacts/models/` and registered MLflow models).

---

### Models

**Section: unlabeled**

Browse all saved model files in `artifacts/models/`. Each entry shows the model name, task type, algorithm, and save date. Click a model to view its metadata, provenance, and associated MLflow run.

---

### History

**Section: Review**

Full run history from MLflow. Filter by run type (benchmark, experiment, flaml, tabfm, timesfm) and task type. Sort by start time, duration, model name, or primary score.

---

### Compare

**Section: Review**

Side-by-side comparison of any two MLflow runs: metric deltas, parameter diffs, and algorithm details. Select two runs from the dropdowns and click Compare.

---

### Notebook

**Section: Review**

Generates a reproducible Jupyter notebook for a dataset run. The notebook includes data loading, preprocessing, training, and evaluation — ready to run in Colab or locally.

---

### Registry

**Section: Admin**

MLflow Model Registry interface:

| Action | How |
| --- | --- |
| Register a model | Select a saved model artifact and click Register |
| Promote to Champion | Move a version to the `champion` alias |
| Set as Candidate | Move to `candidate` alias |
| Archive | Tag version as `archived` |

> [!WARNING]
> TabFM-derived model contexts are blocked from registry promotion. The app enforces the non-commercial license boundary.

---

### Settings

**Section: Admin**

Configure workspace preferences without editing files:

- **LLM provider** — select OpenAI, Anthropic, Gemini, or Ollama; enter your API key (kept in session state only, never written to disk)
- **Execution backend** — local or Colab MCP
- **Workspace mode** — dashboard or notebook
- **Model** — select the LLM model for AI summaries (with cost estimates)

Changes are saved to `~/.autotabml/settings.json`. API keys are not included.

---

## CLI reference

All commands: `uv run autotabml <command>` (or `autotabml <command>` with the venv active).

Global flag: `--version` — print version and exit.

---

### Diagnostics

| Command | Description |
| --- | --- |
| `info` | Show version, workspace mode, execution backend, artifact paths, and database path |
| `doctor` | Run startup diagnostics: CUDA, database, artifact dirs, optional extras, stale file cleanup |
| `init-local-storage` | Create artifact directories and initialize the app metadata SQLite database |

---

### Data

#### `validate`

```bash
uv run autotabml validate data/train.csv --target price
```

| Flag | Default | Description |
| --- | --- | --- |
| `<dataset>` | required | Path, URL, or `uci:<id>` |
| `--target <col>` | — | Target column name |
| `--min-rows <n>` | 1 | Fail if row count is below threshold |
| `--artifacts-dir <dir>` | artifacts/validation/ | Output directory |

#### `profile`

```bash
uv run autotabml profile data/train.csv
```

| Flag | Default | Description |
| --- | --- | --- |
| `<dataset>` | required | Path, URL, or `uci:<id>` |
| `--artifacts-dir <dir>` | artifacts/profiling/ | Output directory |

#### `uci-list`

```bash
uv run autotabml uci-list --search iris --limit 10
```

| Flag | Default | Description |
| --- | --- | --- |
| `--search <query>` | — | Filter by name |
| `--area <area>` | — | Filter by subject area |
| `--limit <n>` | 20 | Maximum results |

---

### Benchmarking

#### `benchmark`

```bash
uv run autotabml benchmark data/train.csv --target target --task-type auto
```

| Flag | Default | Description |
| --- | --- | --- |
| `<dataset>` | required | Path, URL, or `uci:<id>` |
| `--target <col>` | required | Target column |
| `--task-type` | `auto` | `auto`, `classification`, `regression` |
| `--test-size <float>` | 0.2 | Holdout fraction |
| `--random-state <int>` | 42 | Random seed |
| `--stratify` | `auto` | `auto`, `true`, `false` |
| `--ranking-metric <str>` | Balanced Accuracy / Adjusted R² | Metric to rank by |
| `--sample-rows <n>` | — | Sample large datasets to N rows |
| `--top-k <n>` | 5 | Top K models to return |
| `--prefer-gpu` | `auto` | `auto`, `true`, `false` |
| `--include-model <name>` | — | Repeat to include only these models |
| `--exclude-model <name>` | — | Repeat to exclude models |
| `--artifacts-dir <dir>` | artifacts/benchmark/ | Output directory |

---

### PyCaret experiments

#### `experiment-run`

```bash
uv run autotabml experiment-run data/train.csv --target target --task-type classification --n-select 3
```

| Flag | Default | Description |
| --- | --- | --- |
| `<dataset>` | required | |
| `--target <col>` | required | |
| `--task-type` | `auto` | `auto`, `classification`, `regression` |
| `--train-size <float>` | 0.7 | Training fraction |
| `--fold <int>` | 5 | Cross-validation folds |
| `--fold-strategy <str>` | stratifiedkfold / kfold | Fold strategy |
| `--preprocess` | `auto` | `auto`, `true`, `false` |
| `--ignore-feature <col>` | — | Repeat to exclude features |
| `--compare-metric <str>` | Accuracy / R2 | Metric for compare ranking |
| `--n-select <int>` | — | Top N models to return |
| `--budget-time <min>` | — | Time limit in minutes |
| `--no-turbo` | False | Disable turbo mode (all estimators) |
| `--use-gpu` | `false` | `false`, `true`, `force` |
| `--artifacts-dir <dir>` | artifacts/experiments/ | |

#### `experiment-tune`

```bash
uv run autotabml experiment-tune data/train.csv --target target --model-id rf --task-type classification
```

| Flag | Default | Description |
| --- | --- | --- |
| `--model-id <id>` | required | Algorithm ID from compare output |
| `--task-type` | required | `classification` or `regression` |
| `--tune-metric <str>` | AUC / R2 | Optimization metric |
| `--n-iter <int>` | — | Tuning iterations |
| `--use-gpu` | `false` | `false`, `true`, `force` |

#### `experiment-evaluate`

```bash
uv run autotabml experiment-evaluate data/train.csv --target target --model-id rf --task-type classification --plot confusion_matrix --plot auc
```

| Flag | Default | Description |
| --- | --- | --- |
| `--plot <plot_id>` | all | Repeat for multiple plots |
| `--use-gpu` | `false` | |

#### `experiment-save`

```bash
uv run autotabml experiment-save data/train.csv --target target --model-id rf --task-type classification --save-name my_model
```

| Flag | Default | Description |
| --- | --- | --- |
| `--save-name <name>` | auto-generated | Output model filename stem |
| `--save-snapshot` | False | Also save a PyCaret pipeline snapshot |
| `--use-gpu` | `false` | |

---

### FLAML AutoML

#### `flaml-run`

```bash
uv run autotabml flaml-run data/train.csv --target target --task-type auto --time-budget 120
```

| Flag | Default | Description |
| --- | --- | --- |
| `<dataset>` | required | |
| `--target <col>` | required | |
| `--task-type` | `auto` | `auto`, `classification`, `regression` |
| `--time-budget <secs>` | 120 | Search time limit |
| `--max-iter <int>` | — | Maximum iterations |
| `--metric <str>` | accuracy / r2 | Optimization metric |
| `--n-splits <int>` | 5 | Cross-validation splits |
| `--seed <int>` | 0 | Random seed |
| `--ensemble` | False | Enable ensembling |
| `--early-stop` | False | Enable early stopping |
| `--estimator <name>` | — | Repeat to restrict estimator set |

#### `flaml-save`

Same flags as `flaml-run`, plus:

| Flag | Default | Description |
| --- | --- | --- |
| `--save-name <name>` | auto-generated | Output model filename stem |

---

### Foundation models

#### `tabfm-run`

```bash
uv run autotabml tabfm-run data/train.csv --target target --accept-tabfm-license --allow-download
```

| Flag | Default | Description |
| --- | --- | --- |
| `<dataset>` | required | |
| `--target <col>` | required | |
| `--task-type` | `auto` | `auto`, `classification`, `regression` |
| `--context-rows <n>` | — | Rows to use as in-context examples |
| `--n-estimators <n>` | — | Ensemble size |
| `--allow-download` | False | **Required** to download checkpoint on first use |
| `--accept-tabfm-license` | False | **Required** — accepts the non-commercial license |
| `--save-context <NAME>` | — | Save context artifact with this name |
| `--artifacts-dir <dir>` | artifacts/experiments/ | |

> [!WARNING]
> `--accept-tabfm-license` is required on every run. It signals explicit acceptance of the
> `tabfm-non-commercial-v1.0` license. Do not use TabFM for commercial or production workloads.

#### `timesfm-forecast`

```bash
uv run autotabml timesfm-forecast data/demand.csv --timestamp date --target demand --horizon 12 --allow-download
```

| Flag | Default | Description |
| --- | --- | --- |
| `<dataset>` | required | |
| `--timestamp <col>` | required | Timestamp column |
| `--target <col>` | required | Value column to forecast |
| `--group <col>` | — | Optional grouping column for panel data |
| `--horizon <int>` | 12 | Forecast steps ahead |
| `--context-length <int>` | — | Historical context window |
| `--frequency <freq>` | — | Pandas frequency string (e.g., `D`, `M`) |
| `--no-backtest` | False | Skip holdout backtest |
| `--allow-download` | False | **Required** to download checkpoint on first use |
| `--artifacts-dir <dir>` | artifacts/experiments/ | |

---

### Predictions

#### `predict-single`

Score a single row:

```bash
uv run autotabml predict-single \
  --model-source local_saved_model \
  --model-path artifacts/models/my_model.pkl \
  --row-json '{"age": 35, "income": 50000, "score": 720}'
```

| Flag | Default | Description |
| --- | --- | --- |
| `--model-source` | required | `local_saved_model`, `mlflow_run`, `mlflow_registered_model` |
| `--model-path <path>` | — | Path to `.pkl` (for `local_saved_model`) |
| `--model-id <id>` | — | MLflow artifact path (for `mlflow_run`) |
| `--model-uri <uri>` | — | MLflow model URI |
| `--model-name <name>` | — | Registry model name (for `mlflow_registered_model`) |
| `--model-version <n>` | — | Registry version |
| `--model-alias <alias>` | — | Registry alias (e.g., `champion`) |
| `--metadata-path <path>` | — | Path to model metadata JSON |
| `--task-type` | — | Override task type |
| `--schema-mode` | `strict` | `strict` or `warn` |
| `--row-json <json>` | — | Input as JSON string |
| `--row-file <path>` | — | Input as single-row CSV file |
| `--run-id <id>` | — | MLflow run ID |
| `--output-dir <dir>` | artifacts/predictions/ | |

#### `predict-batch`

```bash
uv run autotabml predict-batch data/new.csv \
  --model-source local_saved_model \
  --model-path artifacts/models/my_model.pkl \
  --output-path artifacts/predictions/scored.csv
```

Same model-source flags as `predict-single`, plus:

| Flag | Default | Description |
| --- | --- | --- |
| `<dataset>` | required | Input file to score |
| `--output-path <path>` | artifacts/predictions/predictions.csv | Output CSV path |

#### `predict-history`

```bash
uv run autotabml predict-history --limit 20
```

---

### Operations

#### `drift-check`

```bash
uv run autotabml drift-check data/new.csv --baseline artifacts/models/model_drift_baseline.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `<dataset>` | required | New data to check |
| `--baseline <json>` | required | Saved baseline JSON from a previous run |
| `--source-type` | — | Override source type |

#### `explain`

```bash
uv run autotabml explain artifacts/models/explanation.json
```

Prints a saved SHAP explanation artifact as formatted JSON.

#### `deploy-export`

```bash
uv run autotabml deploy-export \
  --model artifacts/models/my_model.pkl \
  --metadata artifacts/models/my_model.json \
  --provenance artifacts/models/my_model_provenance.json \
  --output deploy/my_model_bundle
```

| Flag | Default | Description |
| --- | --- | --- |
| `--model <path>` | required | Path to `.pkl` model file |
| `--metadata <path>` | required | Path to model metadata JSON |
| `--provenance <path>` | — | Optional provenance JSON |
| `--output <path>` | required | Output bundle directory |

#### `batch-history` / `batch-show`

```bash
uv run autotabml batch-history --limit 20
uv run autotabml batch-show <batch-id>
```

---

### History & registry

#### `history-list`

```bash
uv run autotabml history-list --run-type experiment --limit 20
```

| Flag | Default | Description |
| --- | --- | --- |
| `--run-type` | `all` | `all`, `benchmark`, `experiment`, `flaml`, `tabfm`, `timesfm`, `unknown` |
| `--task-type` | — | Filter by classification or regression |
| `--sort-by` | `start_time` | `start_time`, `duration`, `model_name`, `primary_score` |
| `--sort-dir` | `descending` | `ascending` or `descending` |
| `--limit <n>` | 50 | Maximum rows |

#### `history-show`

```bash
uv run autotabml history-show <run-id>
```

#### `compare-runs`

```bash
uv run autotabml compare-runs <run-id-a> <run-id-b>
```

#### `registry-list` / `registry-show`

```bash
uv run autotabml registry-list
uv run autotabml registry-show my_model
```

#### `registry-register`

```bash
uv run autotabml registry-register my_model \
  --source runs://<run-id>/model \
  --run-id <run-id> \
  --description "Best classifier from experiment 2026-08-17"
```

#### `registry-promote`

```bash
uv run autotabml registry-promote my_model 3 --action champion
uv run autotabml registry-promote my_model 2 --action archived
```

`--action` choices: `champion`, `candidate`, `archived`.

---

### Background jobs

#### `auto-run`

```bash
uv run autotabml auto-run data/train.csv --target price --mode auto --time-budget 120 --save-name best_model
```

| Flag | Default | Description |
| --- | --- | --- |
| `<dataset>` | required | |
| `--target <col>` | required | |
| `--task-type` | `auto` | `auto`, `classification`, `regression` |
| `--mode` | `auto` | `auto`, `quick`, `balanced`, `deep` |
| `--time-budget <secs>` | 120 | FLAML time budget |
| `--save-name <name>` | auto-generated | Output model name |

| Mode | Description |
| --- | --- |
| `auto` | Automatically selects strategy based on dataset size and complexity |
| `quick` | Fast pass — short time budget, fewer estimators |
| `balanced` | Medium time budget, broad estimator set |
| `deep` | Long time budget, full estimator set with ensembling |

#### `job-list` / `job-status` / `job-cancel`

```bash
uv run autotabml job-list --limit 10
uv run autotabml job-status <job-id>
uv run autotabml job-cancel <job-id>
```

---

## Configuration

Settings resolve in this order: **Pydantic defaults → `~/.autotabml/settings.json` → environment variables**.

The Settings page writes to `~/.autotabml/settings.json`. API keys are never written there — they are kept in session state and read from environment variables only.

### Using a `.env` file

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

`uv run` and Streamlit both load `.env` automatically.

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `AUTOTABML_WORKSPACE_MODE` | `dashboard` | Startup mode: `dashboard` or `notebook` |
| `AUTOTABML_EXECUTION__BACKEND` | `colab_mcp` | Execution backend: `local` or `colab_mcp` — **set to `local` for local-only use** |
| `AUTOTABML_ARTIFACTS__ROOT_DIR` | `artifacts/` | Override artifact root directory |
| `AUTOTABML_DATABASE__PATH` | `artifacts/app/app_metadata.sqlite3` | App metadata SQLite path |
| `AUTOTABML_MLFLOW__TRACKING_URI` | `sqlite:///artifacts/mlflow/mlflow.db` | MLflow tracking URI |
| `AUTOTABML_MLFLOW__REGISTRY_URI` | `sqlite:///artifacts/mlflow/mlflow.db` | MLflow registry URI |
| `AUTOTABML_PROVIDER__BASE_URL` | — | Override LLM provider base URL |
| `AUTOTABML_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `AUTOTABML_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `AUTOTABML_LOG_FORMAT` | `text` | Log format: `text` or `json` |
| `OPENAI_API_KEY` | — | OpenAI API key (no prefix — read directly by provider) |
| `ANTHROPIC_API_KEY` | — | Anthropic Claude API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama fallback URL |
| `KAGGLE_USERNAME` | — | Kaggle username for dataset download |
| `KAGGLE_KEY` | — | Kaggle API key |

> [!NOTE]
> `AUTOTABML_EXECUTION__BACKEND` defaults to `colab_mcp` to support remote Colab workflows.
> If you are running entirely locally, add `AUTOTABML_EXECUTION__BACKEND=local` to your `.env`.

> [!IMPORTANT]
> API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) are not prefixed with
> `AUTOTABML_` because provider client libraries read them directly. They must never be
> committed to version control.

---

## Input & output formats

### Supported input formats

| Format | Extension | Notes |
| --- | --- | --- |
| CSV | `.csv` | Default; bounded parsing |
| TSV | `.tsv` | Tab-separated |
| Excel | `.xlsx`, `.xls`, `.xlsb` | First sheet by default |
| Remote URL | `https://...` | Direct link to CSV or Excel file; SSRF-hardened |
| UCI ML Repository | `uci:<id>` | Fetched and cached locally |
| Kaggle | dataset slug | CLI only; requires credentials |
| HTML table | URL with `--source-type html_table` | Extracts first table |

### Artifact output layout

All outputs are written under the artifacts root (default `artifacts/`, override with `AUTOTABML_ARTIFACTS__ROOT_DIR`):

```
artifacts/
  validation/          — Validation reports, Great Expectations outputs
  profiling/           — HTML profile report + summary.json
  benchmark/           — leaderboard.csv + summary.json
  experiments/         — PyCaret run artifacts
    snapshots/         — PyCaret pipeline snapshots
  flaml/               — FLAML run artifacts
  models/              — model.pkl + model.sha256 + model.json metadata
  predictions/         — batch prediction CSVs + history.jsonl
  comparisons/         — compare-runs diff artifacts
  mlflow/              — mlflow.db (MLflow SQLite backend)
  app/                 — app_metadata.sqlite3 (job/dataset history)
  tmp/                 — Temporary files (auto-cleaned after 24 h; failed runs after 48 h)
```

> [!NOTE]
> Every saved model file (`model.pkl`) has a corresponding `.sha256` sidecar.
> The app verifies this checksum before loading any model to prevent tampered artifact loading.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: lazypredict` | `benchmark` extra not installed | `uv sync --locked --extra benchmark` |
| `ModuleNotFoundError: pycaret` | `experiment` extra not installed | `uv sync --locked --extra experiment` |
| `ModuleNotFoundError: flaml` | `flaml` extra not installed | `uv sync --locked --extra flaml` |
| `ModuleNotFoundError: ydata_profiling` | `profiling` extra not installed | `uv sync --locked --extra profiling` |
| `PyCaret requires Python < 3.13` | Running Python 3.13 | Use Python 3.11 or 3.12: `uv sync --python 3.12 --extra experiment` |
| `tabfm requires --accept-tabfm-license` | License flag not passed | Add `--accept-tabfm-license` to `tabfm-run` command |
| `typeguard conflict between tabfm and profiling` | Both extras in same venv | Use separate venvs: one for `tabfm`, one for `profiling` |
| MLflow shows no runs | Tracking URI not initialized | Run `uv run autotabml init-local-storage` |
| GPU not detected, training on CPU | No NVIDIA GPU or CUDA not installed | Expected fallback — install CUDA toolkit if GPU is available |
| Dataset sampled to 50 K rows | Dataset exceeds 100 K rows | Expected behavior; increase `AUTOTABML_BENCHMARK__SAMPLING_ROW_THRESHOLD` in settings |
| Kaggle download not available in UI | Kaggle extra not installed or credentials missing | Install `kaggle` extra; set `KAGGLE_USERNAME` and `KAGGLE_KEY` |
| `API key not found` in Settings | Key not in environment | Set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` in `.env` |
| App uses Colab MCP backend instead of running locally | Default backend is `colab_mcp` | Set `AUTOTABML_EXECUTION__BACKEND=local` in `.env` |
| `TrustedArtifactError: checksum mismatch` | Model file was modified after save | Re-save the model using `experiment-save` or `flaml-save` |
| SQLite database locked | Another process holds the DB | Stop other `autotabml` or `streamlit` processes; retry |

---

## Known limitations

| Constraint | Detail |
| --- | --- |
| PyCaret requires Python < 3.13 | Use Python 3.11 or 3.12 for PyCaret features |
| TabFM: non-commercial research only | `tabfm-non-commercial-v1.0` license; cannot be used in production or commercial contexts; Python >= 3.11 required |
| tabfm + profiling conflict | Incompatible `typeguard` versions; install in separate virtual environments |
| First foundation model use | TabFM and TimesFM download pinned Hugging Face snapshots on first run — requires `--allow-download` |
| GPU training | Requires NVIDIA GPU with CUDA; falls back to CPU automatically |
| Large datasets | Datasets with >100 K rows are automatically sampled to 50 K rows for benchmarking and profiling |
| Kaggle | CLI-only — not available in the Streamlit UI |
| Single-user | One active training job at a time |
| Background concurrency | Background jobs queue; concurrent job execution is not supported |
| Drift scope | Input-distribution drift only — not concept drift or target drift |
| AI summaries | Require an API key (OpenAI, Anthropic, Gemini) or a running local Ollama instance |
| Default backend | `AUTOTABML_EXECUTION__BACKEND` defaults to `colab_mcp`; set to `local` for fully local operation |
