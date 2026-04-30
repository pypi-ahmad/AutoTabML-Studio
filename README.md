<div align="center">

# AutoTabML Studio

<img src="docs/assets/social-preview/autotabml-social-preview.png" alt="AutoTabML Studio" width="100%">

**Local-first automated machine learning workbench for tabular data.**\
Go from raw CSV to trained, evaluated, and deployable model — entirely on your machine.

<br>

[![CI](https://img.shields.io/github/actions/workflow/status/pypi-ahmad/AutoTabML-Studio/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI)](https://github.com/pypi-ahmad/AutoTabML-Studio/actions/workflows/ci.yml)
[![Security](https://img.shields.io/github/actions/workflow/status/pypi-ahmad/AutoTabML-Studio/security.yml?branch=main&style=for-the-badge&logo=shieldsdotio&logoColor=white&label=Security)](https://github.com/pypi-ahmad/AutoTabML-Studio/actions/workflows/security.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-0F172A?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB?style=for-the-badge&logo=python&logoColor=white)

![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-DataFrames-150458?style=flat-square&logo=pandas&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Schemas-E92063?style=flat-square&logo=pydantic&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-AutoML-1D4ED8?style=flat-square)
![FLAML](https://img.shields.io/badge/FLAML-AutoML-00A4EF?style=flat-square&logo=microsoft&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Metadata-003B57?style=flat-square&logo=sqlite&logoColor=white)

[Features](#-features) · [Quick Start](#-quick-start) · [Screenshots](#-screenshots) · [Architecture](#-architecture) · [Observability](#-observability) · [Docs](#-documentation)

</div>

---

## Why AutoTabML Studio?

Most tabular ML work is scattered across notebooks, throwaway scripts, and manual model-file management. AutoTabML Studio consolidates the entire lifecycle into a single local workspace with a **Streamlit UI** and a **CLI** — backed by the same service layer so results are always reproducible.

- **Zero cloud dependency.** Data never leaves your machine. No default outbound telemetry or external uploads.
- **Three AutoML engines.** LazyPredict for quick benchmarks, PyCaret for full experiments, and Microsoft FLAML for fast, cost-efficient hyperparameter search.
- **End-to-end tracking.** Every run is logged to MLflow with metrics, parameters, and artifacts. Compare, version, and promote models from one place.
- **530+ unit tests** with a CI-enforced coverage gate. Security scanning on every push.

---

## ✨ Features

<table>
<tr>
<td width="50%">

### Data Preparation
- **Multi-source ingestion** — CSV, Excel, TSV, URLs, HTML tables, UCI ML Repository, optional Kaggle
- **Quality validation** — app-native checks + optional Great Expectations integration
- **EDA profiling** — `ydata-profiling` reports with sampling safeguards for large datasets

</td>
<td width="50%">

### Modeling
- **Quick Benchmark** — screen 30+ algorithms via LazyPredict in seconds
- **Train & Tune** — full PyCaret pipeline (compare → tune → evaluate → finalize → save)
- **FLAML AutoML** — Microsoft's fast, lightweight AutoML with time-budget control
- **Classification & Regression** task types

</td>
</tr>
<tr>
<td width="50%">

### Predictions & Evaluation
- **Batch & single-row scoring** with form-based or JSON input
- **Model testing** against held-out data with ground-truth labels
- **Downloadable notebooks** auto-generated for every run (Colab-compatible)

</td>
<td width="50%">

### Operations
- **MLflow model registry** — Champion / Candidate / Archived lifecycle
- **Run history & comparison** — side-by-side algorithm evaluation
- **Structured observability** — JSON logs, metrics hooks, optional tracing, run correlation
- **AI-generated summaries** — OpenAI, Anthropic, Gemini, or local Ollama
- **CLI** for scripted, repeatable workflows

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
| --- | --- |
| Python | 3.10 – 3.13 (`uv` defaults to 3.12 via `.python-version`; use 3.11 or 3.12 for PyCaret support) |
| OS | Windows, macOS, Linux |

### Install

```bash
# 1. Sync the lockfile into a local environment
uv sync --locked --extra dev

# 2. Add optional extras as needed
uv sync --locked --extra dev --extra benchmark   # LazyPredict + boosted baselines
uv sync --locked --extra dev --extra experiment  # PyCaret full pipeline (Python 3.11/3.12)
uv sync --locked --extra dev --extra flaml       # Microsoft FLAML AutoML
uv sync --locked --extra dev --extra validation  # Great Expectations
uv sync --locked --extra dev --extra profiling   # ydata-profiling EDA reports

# Or sync everything in the lockfile:
uv sync --locked --all-extras
```

`uv` uses the committed lockfile and the repo's `.python-version` (`3.12`) by default. CI enforces `uv lock --check` and `uv sync --locked` so local installs and GitHub Actions resolve the same environment.

If dependency metadata changes, refresh pinned versions with `uv lock --python 3.12` and commit the updated `uv.lock`.

### Run

```bash
uv run autotabml init-local-storage   # Initialize SQLite + artifact dirs
uv run autotabml doctor               # Verify runtime dependencies
uv run streamlit run app/main.py      # Launch the UI
```

---

## 🧭 Workflow

```
Load Data → Validate → Profile → Benchmark → Train / FLAML → Predict → Compare → Register
```

| Step | What happens |
| --- | --- |
| **Load Data** | Upload a file, paste a URL, or pick a UCI dataset |
| **Validate** *(optional)* | Check for missing values, schema issues, and data leakage |
| **Profile** *(optional)* | Generate a visual EDA summary |
| **Quick Benchmark** | Screen 30+ algorithms — ranked leaderboard in seconds |
| **Train & Tune** | Fine-tune the best algorithm with PyCaret and save a production model |
| **FLAML AutoML** | Run Microsoft FLAML with time-budget or iteration-budget constraints |
| **Predict** | Score new data (single row or batch file) with any saved model |
| **Compare & Register** | Review run history, compare results, promote models |

See **[USAGE.md](USAGE.md)** for the full step-by-step guide.

---

## 🖼️ Screenshots

<table>
<tr>
<td width="50%"><img src="docs/assets/screenshots/dashboard-overview.png" alt="Dashboard"><br><strong>Dashboard</strong> — Workflow progress and recent activity</td>
<td width="50%"><img src="docs/assets/screenshots/dataset-intake.png" alt="Load Data"><br><strong>Load Data</strong> — Files, URLs, or UCI repository</td>
</tr>
<tr>
<td width="50%"><img src="docs/assets/screenshots/validation-summary.png" alt="Validation"><br><strong>Validation</strong> — Target-aware quality checks</td>
<td width="50%"><img src="docs/assets/screenshots/benchmark-leaderboard.png" alt="Benchmark"><br><strong>Benchmark</strong> — Algorithm ranking leaderboard</td>
</tr>
<tr>
<td width="50%"><img src="docs/assets/screenshots/experiment-lab.png" alt="Experiment"><br><strong>Train & Tune</strong> — PyCaret experiment pipeline</td>
<td width="50%"><img src="docs/assets/screenshots/prediction-center.png" alt="Predictions"><br><strong>Predictions</strong> — Batch and single-row scoring</td>
</tr>
<tr>
<td width="50%"><img src="docs/assets/screenshots/registry-view.png" alt="Registry"><br><strong>Registry</strong> — Model versioning and promotion</td>
<td width="50%"><img src="docs/assets/screenshots/settings-view.png" alt="Settings"><br><strong>Settings</strong> — Workspace and provider configuration</td>
</tr>
</table>

<details>
<summary>More screenshots</summary>

| | |
|---|---|
| <img src="docs/assets/screenshots/profiling-report.png" alt="Profiling"> **Profiling** | <img src="docs/assets/screenshots/history-view.png" alt="History"> **History** |
| <img src="docs/assets/screenshots/compare-view.png" alt="Compare"> **Compare** | |

</details>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                      │
│  Dashboard · Load · Validate · Profile · Benchmark  │
│  Train & Tune · FLAML · Predict · Models · History  │
│  Compare · Notebook · Registry · Settings           │
├─────────────────────────────────────────────────────┤
│                   CLI (argparse)                    │
├──────────────┬──────────────┬───────────────────────┤
│  Ingestion   │  Validation  │     Profiling         │
├──────────────┼──────────────┼───────────────────────┤
│  LazyPredict │   PyCaret    │      FLAML            │
│  (Benchmark) │ (Experiment) │    (AutoML)           │
├──────────────┴──────────────┴───────────────────────┤
│  Prediction · Tracking · Registry · Observability   │
│  Storage                                             │
├─────────────────────────────────────────────────────┤
│  MLflow (SQLite) · SQLite Metadata · artifacts/     │
└─────────────────────────────────────────────────────┘
```

Streamlit pages are **thin entry points**. All business logic lives in the service layer.

<details>
<summary>Module map</summary>

| Module | Responsibility |
| --- | --- |
| `app/ingestion/` | Source routing, loaders, normalization, metadata hashing |
| `app/validation/` | Quality rules, optional Great Expectations checks |
| `app/profiling/` | Profiling orchestration, selectors, summaries |
| `app/modeling/benchmark/` | LazyPredict orchestration, ranking, MLflow logging |
| `app/modeling/pycaret/` | PyCaret compare, tune, evaluate, finalize, save |
| `app/modeling/flaml/` | Microsoft FLAML AutoML service, artifacts, tracking |
| `app/prediction/` | Model discovery, loading, schema checks, scoring |
| `app/tracking/` | MLflow queries, history, run comparison |
| `app/registry/` | MLflow model registration and promotion |
| `app/observability/` | Structured logging, correlation context, metrics hooks, optional tracing |
| `app/storage/` | SQLite metadata store |
| `app/providers/` | LLM integrations (OpenAI, Anthropic, Gemini, Ollama) |
| `app/notebooks/` | Jupyter notebook generation |
| `app/config/` | Pydantic settings, enums, environment binding |
| `app/pages/` | Streamlit page entry points |
| `app/cli.py` | CLI entry point |

</details>

### Tech Stack

| Layer | Technology |
| --- | --- |
| **UI** | Streamlit |
| **CLI** | argparse |
| **Data** | pandas, Pydantic, pydantic-settings |
| **Benchmarking** | LazyPredict, scikit-learn, XGBoost, LightGBM, CatBoost |
| **Training** | PyCaret, Microsoft FLAML |
| **Tracking** | MLflow (local SQLite backend) |
| **Observability** | JSON logging, metrics hooks, optional OpenTelemetry tracing |
| **Metadata** | SQLite |
| **AI Summaries** | OpenAI · Anthropic · Gemini · Ollama |
| **Testing** | pytest (530+ tests), pytest-cov, pytest-asyncio |

---

## 💻 CLI

Examples below assume the synced `.venv` is active. If you are not activating it, prefix commands with `uv run`.

```bash
autotabml --version
autotabml info
autotabml doctor
```

```bash
# Data preparation
autotabml validate data/train.csv --target price --artifacts-dir artifacts/validation
autotabml profile data/train.csv --artifacts-dir artifacts/profiling

# Modeling
autotabml benchmark data/train.csv --target target --task-type auto
autotabml experiment-run data/train.csv --target target --task-type classification --n-select 3
autotabml flaml-run data/train.csv --target target --task-type auto --time-budget 120
autotabml flaml-save data/train.csv --target target --save-name best_model

# Operations
autotabml predict-history --limit 10
autotabml history-list --run-type experiment --limit 10
autotabml registry-list
```

---

## ⚙️ Configuration

Settings are resolved in order: **Pydantic defaults → persisted `settings.json` → environment variables**.

```bash
# Core settings (AUTOTABML_ prefix)
AUTOTABML_WORKSPACE_MODE=dashboard
AUTOTABML_EXECUTION__BACKEND=local
AUTOTABML_MLFLOW__TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db

# LLM provider keys (no prefix)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

See [.env.example](.env.example) for the full list.

---

## 📈 Observability

Runtime observability is **local-first**. Nothing is exported anywhere unless you explicitly wire an exporter.

- Set `AUTOTABML_LOG_FORMAT=json` to emit one JSON document per log line to stderr. The default remains `text` for local development.
- Training and prediction workflows automatically attach correlation fields such as `correlation_id`, `run_id`, `experiment_name`, `run_name`, and `task_type` when those values are available.
- Metrics hooks live behind `app.observability`, so you can swap the default in-process backend for Prometheus, StatsD, or OTLP adapters at startup without changing workflow code.
- Tracing is a no-op by default and upgrades automatically when `opentelemetry-api` is installed and configured.

```bash
# Structured JSON logs
AUTOTABML_LOG_FORMAT=json
AUTOTABML_LOG_LEVEL=INFO
uv run streamlit run app/main.py
```

---

## 🧪 Testing & CI

```bash
uv run pytest                              # Unit tests
uv run pytest -m integration               # Integration suite
uv run pytest --cov=app --cov-fail-under=65  # Coverage gate
```

| Workflow | Purpose |
| --- | --- |
| [CI](.github/workflows/ci.yml) | Lint (ruff) · Unit tests (Python 3.11 + 3.13) · Coverage ≥ 65% · E2E smoke |
| [Security](.github/workflows/security.yml) | `detect-secrets` + `gitleaks` on every push and PR |
| [Release](.github/workflows/release-readiness.yml) | Build validation + `twine check` for tagged releases |

CI uses the committed `uv.lock` for deterministic installs before linting, testing, coverage, and release validation.

Dependabot is configured for weekly dependency updates.

---

## ⚠️ Known Limitations

| Constraint | Detail |
| --- | --- |
| PyCaret requires Python < 3.13 | All other features work on 3.10 – 3.13 |
| GPU training | Requires NVIDIA + CUDA; falls back to CPU automatically |
| Large datasets | 100K+ rows trigger automatic sampling |
| Kaggle | CLI-only; not exposed in the UI |
| Single-user | Designed for individual local use |
| AI summaries | Require an API key or local Ollama |

---

## 📚 Documentation

| Resource | Description |
| --- | --- |
| [USAGE.md](USAGE.md) | Complete usage guide with step-by-step instructions |
| [Developer Guide](docs/developer-guide.md) | Implementation notes and development workflow |
| [Contributing](CONTRIBUTING.md) | Contribution guidelines |

---

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE).
