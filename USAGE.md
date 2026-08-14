# Usage Guide — AutoTabML Studio

> **Local-first automated machine learning for tabular data.**
> Everything runs on your machine. Your data never leaves your environment.

AutoTabML Studio is an interactive workbench that takes you from a raw dataset to a
trained, evaluated, and deployable model — without writing code.

---

## Table of Contents

- [Who This Is For](#who-this-is-for)
- [Before You Start](#before-you-start)
- [Starting the App](#starting-the-app)
- [Core Workflow](#core-workflow)
- [Pages Reference](#pages-reference)
  - [Dashboard](#dashboard)
  - [Load Data](#load-data)
  - [Data Validation](#data-validation)
  - [Data Profiling](#data-profiling)
  - [Quick Benchmark](#quick-benchmark)
  - [Train & Tune](#train--tune)
  - [FLAML AutoML](#flaml-automl)
  - [Foundation Models](#foundation-models)
  - [Predictions](#predictions)
  - [Test & Evaluate](#test--evaluate)
  - [Models](#models)
  - [History](#history)
  - [Algorithm Comparison](#algorithm-comparison)
  - [Model Registry](#model-registry)
  - [Notebooks](#notebooks)
  - [Settings](#settings)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Optional Dependencies](#optional-dependencies)
- [Input & Output](#input--output)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## Who This Is For

- **Data analysts** who want to find the best algorithm for a dataset without writing
  Python scripts.
- **Data scientists** who need a quick screening-to-production pipeline on their own
  machine.
- **Business users** who want to upload a spreadsheet, train a model, and score new
  data through a guided interface.

No cloud account is required. No data is uploaded to external servers.

---

## Before You Start

| Requirement | Details |
|-------------|---------|
| **Python**  | 3.10, 3.11, 3.12, or 3.13 |
| **OS**      | Windows, macOS, or Linux |
| **Install** | `uv sync --locked --extra benchmark --extra experiment` (standard modeling workflows) |

Install only the features you need:

```bash
# Core only (data loading, validation rules, settings)
pip install -e .

# Add optional capabilities
pip install -e ".[validation]"     # Great Expectations data checks
pip install -e ".[profiling]"      # ydata-profiling EDA reports
pip install -e ".[benchmark]"      # LazyPredict algorithm screening
pip install -e ".[experiment]"     # PyCaret model training (Python < 3.13)
pip install -e ".[flaml]"          # Microsoft FLAML AutoML
pip install -e ".[tabfm]"          # Google TabFM (Python 3.11+; research only)
pip install -e ".[timesfm]"        # Google TimesFM 2.5 forecasting
pip install -e ".[gpu]"            # GPU-accelerated training (XGBoost, LightGBM, CatBoost)

# Combine only the extras you need. Keep TabFM and profiling in separate environments.
pip install -e ".[validation,profiling,benchmark,experiment,flaml,gpu,timesfm]"
```

> **Note:** The `experiment` group requires Python < 3.13 due to a PyCaret dependency
> constraint. TabFM requires Python 3.11+ and conflicts with the `profiling` extra
> because their upstream `typeguard` requirements do not overlap. Other features
> support Python 3.10–3.13.

### First-Time Setup

```bash
# Create local directories and metadata database
autotabml init-local-storage

# Verify your environment
autotabml doctor
```

`init-local-storage` creates the `artifacts/` directory tree and a local SQLite
database for metadata. `doctor` checks CUDA availability, database status, artifact
directories, and reports any startup issues.

---

## Starting the App

### Streamlit UI (Interactive)

On Windows, double-click **`Launch AutoTabML Studio.cmd`** in the repository
folder. It starts the app through the project's `uv` environment.

From a terminal:

```bash
uv run streamlit run app/main.py
```

Opens in your browser at `http://localhost:8561`. The sidebar provides guided
navigation through every step.

### CLI (Scripted / Headless)

```bash
autotabml --help
```

Core data, modeling, prediction, tracking, and registry workflows also have CLI
equivalents. See
[CLI Reference](#cli-reference) below.

---

## Core Workflow

### Recommended: Auto Run

Load a dataset, open **Auto Run**, confirm the target and inferred task, then
launch. Training runs in a local subprocess; SQLite-backed status survives
navigation and browser refreshes. Completion produces a saved model, untouched
holdout metrics, explanations, provenance, and a drift baseline. One training
job runs at a time.

AutoTabML Studio follows a five-step workflow. Steps 2a and 2b are optional:

```
① Load Data → ②? Validate → ②? Profile → ③ Quick Benchmark → ④ Train & Tune → ⑤ Predict
```

Each page shows a workflow banner indicating your current step and progress.

| Step | Page | Purpose |
|------|------|---------|
| **1** | Load Data | Upload or connect to a dataset |
| **2** *(optional)* | Data Validation | Check data quality before training |
| **2** *(optional)* | Data Profiling | Visual summary of distributions and correlations |
| **3** | Quick Benchmark | Screen dozens of algorithms to find the best candidates |
| **4** | Train & Tune | Fine-tune the best algorithm and save a production model |
| **5** | Predictions | Score new data with your saved model |

You can skip optional steps and jump directly from Load Data to Quick Benchmark.
The **FLAML AutoML** and **Foundation Models** pages are alternative modeling paths,
not required steps in the five-step guided workflow.

---

## Pages Reference

### Dashboard

The landing page. Shows:

- **Welcome flow** for first-time users with example datasets (Iris, Heart Disease,
  Wine Quality)
- **Active dataset spotlight** with row/column counts and completion progress
- **Recommended next step** based on what you have already completed
- **Recent activity** from your job history

### Load Data

**Title:** 📥 Load Data

Load a dataset from one of four sources:

| Source | Description |
|--------|-------------|
| **📁 Upload** | Drag-and-drop or browse for CSV, TSV, TXT, DATA, XLSX, XLS, XLSM, or XLSB files |
| **📂 Local Path** | Enter a file path to a supported file on your machine |
| **🌐 Web URL** | Paste an HTTP/HTTPS link to a remote CSV or data file |
| **🏛️ UCI Dataset Library** | Search and load datasets from the UCI Machine Learning Repository |

After loading, you can:

- Preview the first 50 rows
- View a column summary (data type, non-null percentage, uniqueness)
- See which cleanup steps were applied automatically
- Set the loaded dataset as active for all downstream pages

The **Loaded** tab lists previously loaded datasets so you can switch between them.

### Data Validation

**Title:** ✅ Data Validation · *Optional step*

Run quality checks on your active dataset before training. Configuration options:

- **Target column** — the column your model will predict
- **Required columns** — columns that must be present
- **Uniqueness check columns** — columns expected to have unique values
- **Data leakage detection** — flag potential leakage from the target
- **Minimum row count** — fail if the dataset is too small

Results show a pass/fail summary with details on each check.

> Requires the `validation` optional dependency for Great Expectations integration.
> Without it, app-native validation rules still run.

### Data Profiling

**Title:** 📊 Data Profiling · *Optional step*

Generate a visual exploratory data analysis (EDA) report covering distributions,
correlations, missing values, and data types.

- For large datasets (50,000+ rows or 100+ columns), a compact mode with sampling is
  used automatically.
- The generated profile can be viewed directly in the app.

> Requires the `profiling` optional dependency (`ydata-profiling`).

### Quick Benchmark

**Title:** 🏁 Quick Benchmark

Screen dozens of algorithms on your data in one click. No tuning, no model saving —
just a ranked leaderboard.

**Configuration:**

1. **Target column** — the column to predict
2. **Task type** — Classification or Regression (or auto-detect)
3. **Run mode** — Quick (sampled, fast), Standard (balanced), or Deep (full dataset)
4. **Advanced options:**
   - Ranking metric (e.g., Balanced Accuracy, R²)
   - Number of top models to display
   - Held-back data percentage (10–50%)
   - Random seed for reproducibility

**Output:** A ranked leaderboard of algorithms with scores. The page suggests your
next step: go to Train & Tune to build a production model from the top-performing
algorithm.

> Requires the `benchmark` optional dependency.

### Train & Tune

**Title:** 🧪 Train & Tune

Build a production-ready model. Unlike Quick Benchmark, this step:

- Compares algorithms with cross-validation
- Tunes hyperparameters on the best candidate
- Lets you evaluate with diagnostic charts
- Saves a final model that can be used for predictions

**Configuration:**

1. **Target column** — the column to predict
2. **Task type** — Classification, Regression, or Auto
3. **Training options** — train/test split, cross-validation folds, fold strategy,
   preprocessing toggle
4. **Experiment tracking** — Automatic, Manual, or Off
5. **GPU** — Off, Auto, or Force

After training, you can:

- **Tune** the top model's hyperparameters
- **Evaluate** with charts (confusion matrix, AUC, residuals, feature importance, etc.)
- **Save** the model for use in Predictions

> Requires the `experiment` optional dependency (`pycaret`). Python < 3.13 only.

### FLAML AutoML

**Title:** 🔥 FLAML AutoML

An alternative to Train & Tune powered by Microsoft FLAML — a fast, lightweight AutoML
framework that searches for the best model within a time or iteration budget.

**When to use FLAML instead of Train & Tune:**

- You want automated model selection without manual tuning steps
- You prefer time-budgeted search (e.g. "find the best model in 2 minutes")
- You want to try multiple estimators (LightGBM, XGBoost, Random Forest, CatBoost, etc.) simultaneously

**Configuration:**

1. **Target column** — the column to predict
2. **Task type** — Classification, Regression, or Auto-detect
3. **Time budget** — Quick (60s), Standard (120s), Deep (300s), or Custom
4. **Advanced options** — estimator selection, metric, CV folds, ensemble, early stopping, random seed

**After search:**

- View the search summary (best estimator, best loss, duration)
- Browse the leaderboard of all estimators tried
- Save the best model for use in Predictions
- Download artifacts (leaderboard CSV, summary JSON)

**CLI support:**

```bash
autotabml flaml-run data/train.csv --target target --task-type auto --time-budget 120
autotabml flaml-save data/train.csv --target target --task-type classification --save-name my_model
```

> Requires the `flaml` optional dependency: `pip install -e ".[flaml]"`

### Foundation Models

**Title:** Foundation Models

Run one of two local, revision-pinned Google model workflows against the active
dataset:

- **TabFM 1.0** evaluates tabular classification or regression from a sampled
  context. You choose the target, task type, context-row count, and ensemble size.
  It requires explicit acceptance of the non-commercial weights license and
  separate consent before the first model download. An optional saved context can
  be reused by Predictions, but cannot be registered, exported, or used in
  production.
- **TimesFM 2.5** produces point forecasts and q10–q90 uncertainty for a single
  series or grouped series. You choose timestamp, numeric target, optional group,
  horizon, context length, and frequency override. Optional final-horizon
  backtesting reports aggregate error metrics.

Both workflows execute on the local machine even when the Colab MCP backend is
selected. The first approved download is about 1 GB; later runs reuse the pinned
Hugging Face cache. Run artifacts are written under
`artifacts/experiments/foundation/` and only aggregate metrics are sent to MLflow.

```bash
uv sync --locked --extra tabfm       # dedicated environment; do not combine with profiling
uv sync --locked --extra timesfm
```

### Predictions

**Title:** 🔮 Predictions

Score new data using a saved model. Two tabs:

**📄 Score a File (Batch)**
- Upload a CSV or Excel file
- Get predictions (and confidence scores where applicable) for every row
- Download results

**✏️ Predict One Record**
- Fill in a form with one row of data
- Get an instant prediction
- JSON input is also available behind an expander for power users

**Model sources:**
- Saved models from Train & Tune, FLAML AutoML, or Quick Benchmark (auto-discovered)
- Saved TabFM contexts (research-only, prediction-only)
- Manual file path
- MLflow registry (if configured)

### Test & Evaluate

**Title:** 📊 Test & Evaluate

Measure how well a trained model performs on data it has never seen.

1. Select a saved model
2. Upload a test dataset that includes ground-truth labels
3. View performance metrics

This page is accessed through the Predictions page as a second tab
("📊 Test & Evaluate").

### Models

**Title:** 🗂️ Models

Browse every model you have trained or registered:

- 🔬 Models from Train & Tune
- 🏁 Models from Quick Benchmark
- Research-only saved TabFM contexts
- 📦 Models from the Model Registry (if configured)

Each model card shows metadata: task type, dataset, creation date, and source.

### History

**Title:** 📋 History

Every workflow run — validation, profiling, benchmark, experiment, foundation-model,
or prediction — is
recorded here.

**Filters:**
- Workflow type (All / Validation / Profiling / Benchmark / Experiment / Prediction)
- Dataset name (text search)
- Result limit (5–500, default 50)

Expand any run to view its full summary, parameters, and metrics.

### Algorithm Comparison

**Title:** ⚖️ Algorithm Comparison

Compare how different algorithms performed on the same dataset. Select a past
benchmark or experiment run to view side-by-side rankings, scores, and the best
algorithm.

### Model Registry

**Title:** 🏷️ Model Registry

Manage versioned copies of your best models. Promotion stages:

| Stage | Meaning |
|-------|---------|
| ⭐ **Champion** | Production-ready — this is the model you trust |
| 🧪 **Candidate** | Under evaluation — promoted for testing |
| 📦 **Archived** | Retired from active use |

> Requires MLflow tracking to be enabled in Settings.

### Notebooks

**Title:** 📓 Notebooks

Auto-generated Jupyter notebooks for every job you have run. Each notebook
reproduces the steps of that run in plain Python. You can:

- Download notebooks
- Open directly in Google Colab (if the Colab backend is configured)

### Settings

**Title:** ⚙️ Settings

Two tabs:

**Essentials:**
- 🔒 Privacy reminder
- Workspace mode (Dashboard or Notebook)
- GPU status (read-only detection)
- Run summaries toggle (auto-generate plain-English summaries of each run)

**Advanced:**
- Execution backend (Local or Colab MCP)
- GPU configuration
- LLM provider and credentials (OpenAI, Anthropic, Gemini, Ollama)
- Model cost calculator for input/output token estimates and pricing-tier notes
- Model directory paths
- Tracking server configuration

The calculator uses these standard USD prices per 1M tokens:

| Model | Input | Output |
| --- | ---: | ---: |
| Sonnet 5 | $2.00 | $10.00 |
| Gemini Flash 3.7 | $0.75 | $3.75 |
| Gemini 3.5 Flash Lite | $0.30 | $2.50 |
| GPT 5.6 Luna | $0.20 | $1.20 |
| GPT 5.6 Terra | $2.00 | $12.00 |
| Grok 4.6 | $2.00 | $6.00 |

Select a model and enter the expected input and output token counts. The result
shows each component and the estimated total; the pricing note identifies batch,
cache, fast-mode, promotional, or long-context conditions that may change the bill.

API keys are stored locally and never sent to AutoTabML Studio servers — they go
directly to the provider you configure.

---

## CLI Reference

```bash
autotabml auto-run data/train.csv --target target --mode balanced --time-budget 120
autotabml job-list --limit 20
autotabml job-status <job-id>
autotabml job-cancel <job-id>
autotabml explain artifacts/autorun/<job-id>/explanation.json
autotabml drift-check data/current.csv --baseline artifacts/models/model_drift_baseline.json
autotabml deploy-export --model artifacts/models/model.pkl --metadata artifacts/models/model_metadata.json --output artifacts/deployments/model
```

Deployment ZIPs provide FastAPI `/health`, `/metadata`, and `/predict`
endpoints, a standalone prediction command, and a non-root Dockerfile. Add
authentication and TLS before public exposure.

All operations available in the UI can also be run from the command line.

### System

```bash
autotabml --version               # Print version
autotabml info                    # Environment summary
autotabml init-local-storage      # Create directories and database
autotabml doctor                  # Check CUDA, database, artifacts, cleanup
```

### Data Preparation

```bash
# Validate a dataset
autotabml validate path/to/data.csv --target Price

# Profile a dataset
autotabml profile path/to/data.csv

# Search the UCI repository
autotabml uci-list --search "heart" --limit 10
```

Dataset locators accept multiple formats:

| Format | Example |
|--------|---------|
| Local file | `data/sales.csv` |
| Remote URL | `https://example.com/data.csv` |
| UCI by ID | `uci:53` |
| UCI by name | `uci:Heart Disease` |

### Benchmarking

```bash
autotabml benchmark data.csv \
  --target Price \
  --task-type regression \
  --top-k 10 \
  --test-size 0.2
```

Key flags: `--task-type`, `--test-size`, `--random-state`, `--stratify`,
`--ranking-metric`, `--sample-rows`, `--top-k`, `--prefer-gpu`,
`--include-model`, `--exclude-model`.

### Experiment (Train & Tune)

```bash
# Compare algorithms
autotabml experiment-run data.csv --target Churn --task-type classification

# Tune the best model
autotabml experiment-tune data.csv --target Churn --model-id lr --task-type classification

# Evaluate with charts
autotabml experiment-evaluate data.csv --target Churn --model-id lr \
  --task-type classification --plot confusion_matrix --plot auc

# Save the final model
autotabml experiment-save data.csv --target Churn --model-id lr \
  --task-type classification --save-name my_model
```

### FLAML AutoML

```bash
# Run FLAML search
autotabml flaml-run data.csv --target Churn --task-type classification --time-budget 120

# Run FLAML search and save the best model
autotabml flaml-save data.csv --target Churn --task-type auto --save-name flaml_best

# Customize estimators and metric
autotabml flaml-run data.csv --target price --task-type regression \
  --estimator lgbm --estimator xgboost --metric r2 --n-splits 10
```

Key flags: `--time-budget`, `--max-iter`, `--metric`, `--n-splits`, `--seed`,
`--ensemble`, `--early-stop`, `--estimator`.

### Foundation Models

```bash
# TabFM classification/regression research evaluation
autotabml tabfm-run data/train.csv \
  --target target \
  --task-type auto \
  --accept-tabfm-license \
  --allow-download

# Save a prediction-only research context
autotabml tabfm-run data/train.csv \
  --target target \
  --accept-tabfm-license \
  --save-context tabfm_research_context

# TimesFM single or grouped forecast with a final-horizon backtest
autotabml timesfm-forecast data/demand.csv \
  --timestamp date \
  --target demand \
  --group store_id \
  --horizon 12 \
  --allow-download
```

The `--allow-download` flag is required only until the pinned checkpoint is in the
local Hugging Face cache. TabFM always requires `--accept-tabfm-license`.

### Prediction

```bash
# Batch prediction
autotabml predict-batch new_data.csv \
  --model-source local_saved_model \
  --model-path artifacts/models/my_model

# Single-row prediction
autotabml predict-single \
  --model-source local_saved_model \
  --model-path artifacts/models/my_model \
  --row-json '{"feature1": 42, "feature2": "A"}'

# View prediction history
autotabml predict-history --limit 10
```

Model sources: `local_saved_model`, `mlflow_run`, `mlflow_registered_model`.

### History & Comparison

```bash
autotabml history-list --run-type benchmark --limit 10
autotabml history-show <run_id>
autotabml compare-runs <run_id_1> <run_id_2>
```

### Model Registry

```bash
autotabml registry-list
autotabml registry-show my_model
autotabml registry-register my_model --source "runs:/<run_id>/model"
autotabml registry-promote my_model 1 --action champion
```

Promotion actions: `champion`, `candidate`, `archived`.

---

## Configuration

### Environment Variables

All settings can be overridden with environment variables. Copy `.env.example` to
`.env` and uncomment the values you want to change.

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTOTABML_WORKSPACE_MODE` | `dashboard` | UI mode (`dashboard` or `notebook`) |
| `AUTOTABML_EXECUTION__BACKEND` | `colab_mcp` | Execution backend (`local` or `colab_mcp`) |
| `AUTOTABML_ARTIFACTS__ROOT_DIR` | `artifacts` | Root directory for all output files |
| `AUTOTABML_DATABASE__PATH` | `artifacts/app/app_metadata.sqlite3` | SQLite metadata database |
| `AUTOTABML_MLFLOW__TRACKING_URI` | `sqlite:///artifacts/mlflow/mlflow.db` | MLflow tracking server URI |
| `AUTOTABML_MLFLOW__REGISTRY_URI` | `sqlite:///artifacts/mlflow/mlflow.db` | MLflow model registry URI |
| `AUTOTABML_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server for local AI summaries |
| `AUTOTABML_PROVIDER__BASE_URL` | *(none)* | Optional base-URL override for the selected hosted provider |
| `AUTOTABML_LOG_LEVEL` | `INFO` | Logging verbosity |
| `AUTOTABML_LOG_FORMAT` | `text` | Log encoding (`text` or newline-delimited `json`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Direct Ollama-provider fallback URL |
| `OPENAI_API_KEY` | *(none)* | OpenAI API key for AI-generated summaries |
| `ANTHROPIC_API_KEY` | *(none)* | Anthropic API key for AI-generated summaries |
| `GEMINI_API_KEY` | *(none)* | Google Gemini API key for AI-generated summaries |

Nested settings use double underscores: `AUTOTABML_MLFLOW__TRACKING_URI`.

### Optional Dependencies

| Group | What It Enables | Install |
|-------|----------------|---------|
| `validation` | Great Expectations data checks | `pip install -e ".[validation]"` |
| `profiling` | ydata-profiling EDA reports | `pip install -e ".[profiling]"` |
| `benchmark` | LazyPredict algorithm screening | `pip install -e ".[benchmark]"` |
| `experiment` | PyCaret model training (Python < 3.13) | `pip install -e ".[experiment]"` |
| `flaml` | Time-budgeted Microsoft FLAML AutoML | `pip install -e ".[flaml]"` |
| `tabfm` | Google TabFM research evaluation (Python 3.11+) | `pip install -e ".[tabfm]"` |
| `timesfm` | Google TimesFM 2.5 forecasting | `pip install -e ".[timesfm]"` |
| `gpu` | GPU-accelerated XGBoost, LightGBM, CatBoost | `pip install -e ".[gpu]"` |
| `kaggle` | Kaggle dataset downloads (CLI only) | `pip install -e ".[kaggle]"` |
| `uci` | UCI repository search and loading | `pip install -e ".[uci]"` |
| `colab` | Google Colab MCP backend | `pip install -e ".[colab]"` |
| `providers` | OpenAI, Anthropic, Gemini, and Ollama summaries | `pip install -e ".[providers]"` |
| `explain` | SHAP model explanations | `pip install -e ".[explain]"` |
| `serve` | FastAPI and Uvicorn deployment runtime | `pip install -e ".[serve]"` |
| `dev` group | pytest, build, lint, and type tools | `uv sync --locked --group dev` |

Pages that require optional dependencies show a guided message when the dependency
is missing, with the exact install command.

---

## Input & Output

### Supported Input Formats

| Format | Extensions | Source |
|--------|-----------|--------|
| CSV | `.csv` | Upload, Local Path, Web URL |
| Delimited text | `.tsv`, `.txt`, `.data` | Upload, Local Path |
| Excel | `.xlsx`, `.xls`, `.xlsm`, `.xlsb` | Upload, Local Path |
| UCI Repository | — | Built-in search by name or ID |

### Output Files

All output is stored under the `artifacts/` directory:

```
artifacts/
├── app/                  # Metadata database
│   └── app_metadata.sqlite3
├── benchmark/            # Benchmark results and leaderboards
├── autorun/              # Completed Auto Run reports, explanations, and baselines
├── jobs/                 # Background-job requests, status, and logs
├── comparisons/          # Side-by-side run comparisons
├── deployments/          # Exported FastAPI/CLI/Docker bundles
├── experiments/          # Experiment runs and snapshots
│   ├── foundation/       # TabFM and TimesFM run artifacts
│   └── snapshots/
├── mlflow/               # MLflow tracking database
│   └── mlflow.db
├── models/               # Saved production models
├── predictions/          # Prediction output files and history
│   └── history.jsonl
├── profiling/            # Data profiling reports
├── tmp/                  # Temporary files (auto-cleaned after 24h)
└── validation/           # Validation reports
```

Prediction output files include a `prediction` column (and `prediction_score` for
classification tasks with confidence).

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "Train & Tune" page shows missing dependency | `pycaret` not installed | `pip install -e ".[experiment]"` |
| "Quick Benchmark" page shows missing dependency | `lazypredict` not installed | `pip install -e ".[benchmark]"` |
| Profiling page shows missing dependency | `ydata-profiling` not installed | `pip install -e ".[profiling]"` |
| Foundation Models page shows missing TabFM/TimesFM dependency | Selected model extra is not installed | Install `.[tabfm]` or `.[timesfm]` and restart |
| `profiling` and `tabfm` cannot resolve together | Their upstream `typeguard` constraints conflict | Keep them in separate virtual environments |
| `autotabml doctor` reports no CUDA | GPU drivers not installed or no NVIDIA GPU | Install CUDA toolkit, or use CPU (default) |
| Model Registry page is empty | MLflow tracking not initialized | Run `autotabml init-local-storage` |
| Benchmark stalls on large datasets | Dataset exceeds sampling threshold | Use "Quick" run mode or set `--sample-rows` |
| "Could not load model" error on Predictions page | Model file moved or deleted | Re-save the model from Train & Tune |
| Settings changes not taking effect | Streamlit caches session state | Refresh the browser page after saving |
| `experiment` install fails on Python 3.13 | PyCaret does not support 3.13 yet | Use Python 3.10–3.12 for experiment features |

Run `autotabml doctor` to diagnose environment issues. It checks CUDA, database
connectivity, artifact directories, and stale temporary files.

---

## Limitations

- **PyCaret experiments require Python < 3.13.** TabFM requires Python 3.11+.
- **TabFM is research-only.** Its weights are non-commercial and its saved contexts
  cannot be registered or deployment-exported.
- **TabFM and profiling require separate environments** because their upstream
  `typeguard` constraints conflict.
- **TimesFM covariates and fine-tuning are not enabled.** The integration provides
  point/quantile forecasting and optional holdout backtesting.
- **GPU acceleration** requires compatible NVIDIA hardware and CUDA drivers. The app
  falls back to CPU automatically when GPU is unavailable.
- **Kaggle integration** is available as an optional CLI dependency but is not exposed
  in the Streamlit UI.
- **MLflow tracking** uses a local SQLite database by default. Remote MLflow servers
  can be configured via environment variables but are not tested as a primary use case.
- **Large datasets** (100,000+ rows) trigger automatic sampling in benchmark and
  profiling to maintain reasonable run times.
- **Concurrent users** are not explicitly supported. The app is designed for
  single-user, local-first operation.
- **AI-generated run summaries** require an API key (OpenAI, Anthropic, Gemini) or a
  local Ollama instance. Without one, summaries are not generated.
