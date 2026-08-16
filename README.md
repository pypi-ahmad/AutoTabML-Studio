<div align="center">

# AutoTabML Studio

<img src="docs/assets/social-preview/autotabml-social-preview.png" alt="AutoTabML Studio" width="100%">

**Local-first automated machine learning workbench for tabular data.**

Go from a raw CSV to a trained, evaluated, deployable model, entirely on your
machine. A single service layer powers both the Streamlit UI and the CLI, so
results are reproducible either way. No cloud account, no outbound telemetry.

[![CI](https://img.shields.io/github/actions/workflow/status/pypi-ahmad/AutoTabML-Studio/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI)](https://github.com/pypi-ahmad/AutoTabML-Studio/actions/workflows/ci.yml)
[![Security](https://img.shields.io/github/actions/workflow/status/pypi-ahmad/AutoTabML-Studio/security.yml?branch=main&style=for-the-badge&logo=shieldsdotio&logoColor=white&label=Security)](https://github.com/pypi-ahmad/AutoTabML-Studio/actions/workflows/security.yml)
[![License](https://img.shields.io/badge/License-MIT-0F172A?style=for-the-badge)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB?style=for-the-badge&logo=python&logoColor=white)

![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-DataFrames-150458?style=flat-square&logo=pandas&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Schemas-E92063?style=flat-square&logo=pydantic&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-AutoML-1D4ED8?style=flat-square)
![FLAML](https://img.shields.io/badge/FLAML-AutoML-00A4EF?style=flat-square&logo=microsoft&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Metadata-003B57?style=flat-square&logo=sqlite&logoColor=white)

**Repository:** [github.com/pypi-ahmad/AutoTabML-Studio](https://github.com/pypi-ahmad/AutoTabML-Studio)

[Features](#features) · [Quick start](#quick-start) · [Screenshots](#screenshots) · [Architecture](#architecture) · [Observability](#observability) · [Docs](#documentation)

</div>

---

## Contents

- [What is AutoTabML Studio?](#what-is-autotabml-studio)
- [Features](#features)
- [Quick start](#quick-start)
- [Workflow](#workflow)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [CLI](#cli)
- [Configuration](#configuration)
- [Observability](#observability)
- [Testing & CI](#testing--ci)
- [Known limitations](#known-limitations)
- [Documentation](#documentation)

---

## What is AutoTabML Studio?

Most tabular ML work is scattered across notebooks, throwaway scripts, and
manual model-file management: a benchmark run in one notebook, a tuned model
saved to some folder, a prediction script duplicated for every project. There
is rarely one place that remembers what data trained a model, what it scored,
or which version is actually deployed.

AutoTabML Studio is a **local-first AutoML workbench** that consolidates that
entire lifecycle — ingest, validate, profile, train, predict, track, register —
into one workspace. It ships two front ends, a **Streamlit UI** for interactive
work and a **CLI** for scripted/headless runs, both driven by the exact same
service layer in `app/`. Nothing is UI-only or CLI-only: whatever you can click
through, you can also automate, and the results are identical either way.

It targets a single user running everything on their own machine: a data
scientist or ML engineer who wants to go from a raw CSV to a trained, evaluated,
deployable model without provisioning cloud infrastructure, standing up a
tracking server, or wiring together separate tools for ingestion, training, and
model management.

- **Zero cloud dependency** — data never leaves your machine; no default outbound telemetry or external uploads
- **Three AutoML engines** — LazyPredict for quick benchmarks, PyCaret for full experiments, Microsoft FLAML for fast, cost-efficient hyperparameter search
- **Foundation models** — research-only TabFM classification/regression and TimesFM 2.5 forecasting, both revision-pinned and local
- **End-to-end tracking** — MLflow logs aggregate metrics, parameters, and safe summary artifacts when installed; compare, version, and promote deployable models from one place
- **729 unit tests**, a CI-enforced ≥ 65% coverage gate, `ruff` lint, and `detect-secrets` + `gitleaks` scanning on every push

> [!NOTE]
> "Local-first" is a hard boundary, not a marketing line: MLflow runs against a
> local SQLite backend, the model registry is a local store, and outbound
> network calls only happen when you explicitly opt in (an LLM provider key, a
> Kaggle download, an on-demand foundation-model checkpoint fetch).

---

## Features

<table>
<tr>
<td width="50%">

**Data preparation**
- Multi-source ingestion — CSV, Excel, TSV, URLs, HTML tables, UCI ML Repository, optional Kaggle
- Efficient CSV loading — native bounded parsing avoids intermediate chunk copies
- Quality validation — app-native checks + optional Great Expectations integration
- EDA profiling — `ydata-profiling` reports with sampling safeguards for large datasets

</td>
<td width="50%">

**Modeling**
- Quick Benchmark — screen 30+ algorithms via LazyPredict in seconds
- Train & Tune — full PyCaret pipeline (compare → tune → evaluate → finalize → save)
- FLAML AutoML — fast, lightweight AutoML with time-budget control
- Google TabFM — non-commercial research evaluation for mixed-type classification and regression
- Google TimesFM 2.5 — single/grouped time-series forecasts with q10–q90 uncertainty and holdout backtests
- Classification and regression task types

</td>
</tr>
<tr>
<td width="50%">

**Predictions & evaluation**
- Batch and single-row scoring with form-based or JSON input
- Model testing against held-out data with ground-truth labels
- Downloadable, Colab-compatible notebooks auto-generated for every run

</td>
<td width="50%">

**Operations**
- MLflow model registry — Champion / Candidate / Archived lifecycle
- Run history and side-by-side algorithm comparison
- Structured observability — JSON logs, metrics hooks, optional tracing, run correlation
- AI-generated summaries — OpenAI, Anthropic, Gemini, or local Ollama
- CLI for scripted, repeatable workflows

</td>
</tr>
</table>

---

## Quick start

### Prerequisites

| Requirement | Version |
| --- | --- |
| Python | 3.10 – 3.13 (`uv` defaults to 3.12 via `.python-version`; use 3.11 or 3.12 for PyCaret support) |
| OS | Windows, macOS, Linux |

### Install

```bash
# 1. Sync the lockfile into a local environment
uv sync --locked --group dev

# 2. Add optional extras as needed
uv sync --locked --group dev --extra benchmark   # LazyPredict + boosted baselines
uv sync --locked --group dev --extra experiment  # PyCaret full pipeline (Python 3.11/3.12)
uv sync --locked --group dev --extra flaml       # Microsoft FLAML AutoML
uv sync --locked --group dev --extra validation  # Great Expectations
uv sync --locked --group dev --extra profiling   # ydata-profiling EDA reports
uv sync --locked --group dev --extra tabfm       # Google TabFM (Python 3.11+; research-only weights)
uv sync --locked --group dev --extra timesfm     # Google TimesFM 2.5
```

`uv` resolves against the committed lockfile and the repo's `.python-version`
(`3.12`) by default. CI enforces `uv lock --check` and `uv sync --locked` so
local installs and GitHub Actions resolve the same environment. If dependency
metadata changes, refresh pins with `uv lock --python 3.12` and commit the
updated `uv.lock`.

> [!IMPORTANT]
> `tabfm` and `profiling` are declared as a uv conflict: TabFM requires
> `typeguard<3` while `ydata-profiling` requires `typeguard>=4`. Sync them into
> separate uv environments if you need both workflows.

### Run

On Windows, double-click **`Launch AutoTabML Studio.cmd`** in the repository
root. It uses `uv` and opens the UI in your browser at `http://localhost:8561`.

Or from a terminal:

```bash
uv run autotabml init-local-storage   # Initialize SQLite + artifact dirs
uv run autotabml doctor               # Verify runtime dependencies
uv run streamlit run app/main.py      # Launch the UI
```

---

## Workflow

The recommended path is **Auto Run**: confirm a target, review the inferred
task and FLAML plan, then launch a cancellable background job. It saves the
model, holdout evaluation, explanation, provenance, and a row-free drift
baseline.

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
| **FLAML AutoML** | Run FLAML with time-budget or iteration-budget constraints |
| **Foundation Models** | Evaluate TabFM under its non-commercial weights license, or forecast with TimesFM 2.5 |
| **Predict** | Score new data (single row or batch file) with any saved model |
| **Compare & Register** | Review run history, compare results, promote models |

See **[USAGE.md](USAGE.md)** for the full step-by-step guide.

---

## Screenshots

<table>
<tr>
<td width="50%"><img src="docs/assets/screenshots/dashboard-overview.png" alt="Dashboard"><br><strong>Dashboard</strong> — workflow progress and recent activity</td>
<td width="50%"><img src="docs/assets/screenshots/dataset-intake.png" alt="Load Data"><br><strong>Load Data</strong> — files, URLs, or UCI repository</td>
</tr>
<tr>
<td width="50%"><img src="docs/assets/screenshots/validation-summary.png" alt="Validation"><br><strong>Validation</strong> — target-aware quality checks</td>
<td width="50%"><img src="docs/assets/screenshots/benchmark-leaderboard.png" alt="Benchmark"><br><strong>Benchmark</strong> — algorithm ranking leaderboard</td>
</tr>
<tr>
<td width="50%"><img src="docs/assets/screenshots/experiment-lab.png" alt="Experiment"><br><strong>Train & Tune</strong> — PyCaret experiment pipeline</td>
<td width="50%"><img src="docs/assets/screenshots/prediction-center.png" alt="Predictions"><br><strong>Predictions</strong> — batch and single-row scoring</td>
</tr>
<tr>
<td width="50%"><img src="docs/assets/screenshots/registry-view.png" alt="Registry"><br><strong>Registry</strong> — model versioning and promotion</td>
<td width="50%"><img src="docs/assets/screenshots/settings-view.png" alt="Settings"><br><strong>Settings</strong> — workspace and provider configuration</td>
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

## Architecture

[Open the interactive architecture map](docs/autotabml-studio-architecture.html) for guided views, source-linked components, and diagram export.

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                      │
│  Dashboard · Load · Validate · Profile · Benchmark  │
│  Train & Tune · FLAML · Foundation Models · Predict │
│  Compare · Notebook · Registry · Settings           │
├─────────────────────────────────────────────────────┤
│                   CLI (argparse)                    │
├──────────────┬──────────────┬───────────────────────┤
│  Ingestion   │  Validation  │     Profiling         │
├──────────────┼──────────────┼───────────────────────┤
│ LazyPredict │ PyCaret │ FLAML │ TabFM │ TimesFM 2.5 │
├──────────────┴──────────────┴───────────────────────┤
│  Prediction · Tracking · Registry · Observability   │
│  Storage                                             │
├─────────────────────────────────────────────────────┤
│  MLflow (SQLite) · SQLite Metadata · artifacts/     │
└─────────────────────────────────────────────────────┘
```

Streamlit pages are thin entry points; all business logic lives in the
service layer. Services with a single runtime implementation stay concrete —
shared base classes are reserved for real multi-implementation or published
extension boundaries.

<details>
<summary>Module map</summary>

| Module | Responsibility |
| --- | --- |
| `app/ingestion/` | Source routing, loaders, normalization, metadata hashing |
| `app/validation/` | Quality rules, optional Great Expectations checks |
| `app/profiling/` | Profiling orchestration, selectors, summaries |
| `app/modeling/benchmark/` | LazyPredict orchestration, ranking, MLflow logging |
| `app/modeling/pycaret/` | PyCaret compare, tune, evaluate, finalize, save |
| `app/modeling/flaml/` | FLAML AutoML service, artifacts, tracking |
| `app/modeling/foundation/` | Pinned TabFM/TimesFM adapters, consent gates, artifacts, context persistence, MLflow summaries |
| `app/prediction/` | Model discovery, loading, schema checks, scoring |
| `app/tracking/` | MLflow queries, history, run comparison |
| `app/registry/` | MLflow model registration and promotion |
| `app/observability/` | Structured logging, correlation context, metrics hooks, optional tracing |
| `app/storage/` | SQLite metadata store |
| `app/providers/` | LLM integrations plus reference token pricing and cost estimates |
| `app/notebooks/` | Jupyter notebook generation |
| `app/config/` | Pydantic settings, enums, environment binding |
| `app/pages/` | Streamlit page entry points |
| `app/cli.py` | CLI entry point |

</details>

### Tech stack

| Layer | Technology |
| --- | --- |
| UI | Streamlit |
| CLI | argparse |
| Data | pandas, Pydantic, pydantic-settings |
| Benchmarking | LazyPredict, scikit-learn, XGBoost, LightGBM, CatBoost |
| Training / forecasting | PyCaret, FLAML, Google TabFM, Google TimesFM 2.5 |
| Tracking | MLflow (local SQLite backend) |
| Observability | JSON logging, metrics hooks, optional OpenTelemetry tracing |
| Metadata | SQLite |
| AI summaries | OpenAI · Anthropic · Gemini · Ollama |
| Testing | pytest (729 tests), pytest-cov, pytest-asyncio, respx |

---

## CLI

```bash
autotabml auto-run data/train.csv --target target --mode auto --time-budget 120
autotabml job-list
autotabml job-status <job-id>
autotabml job-cancel <job-id>
autotabml drift-check data/new.csv --baseline artifacts/models/model_drift_baseline.json
autotabml deploy-export --model artifacts/models/model.pkl --metadata artifacts/models/model.json --output deploy/model
```

Examples assume the synced `.venv` is active; prefix with `uv run` otherwise.

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
autotabml tabfm-run data/train.csv --target target --accept-tabfm-license --allow-download
autotabml timesfm-forecast data/demand.csv --timestamp date --target demand --horizon 12 --allow-download

# Operations
autotabml predict-history --limit 10
autotabml history-list --run-type experiment --limit 10
autotabml batch-history --limit 20
autotabml registry-list
```

---

## Configuration

Settings resolve in order: Pydantic defaults → persisted `settings.json` →
environment variables.

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

## Observability

Runtime observability is local-first: nothing is exported anywhere unless you
explicitly wire an exporter.

- Set `AUTOTABML_LOG_FORMAT=json` to emit one JSON document per log line to stderr; the default stays `text` for local development
- Training and prediction workflows automatically attach correlation fields (`correlation_id`, `run_id`, `experiment_name`, `run_name`, `task_type`) when available
- Metrics hooks live behind `app.observability`, so the default in-process backend can be swapped for Prometheus, StatsD, or OTLP adapters at startup without touching workflow code
- Tracing is a no-op by default and upgrades automatically when `opentelemetry-api` is installed and configured

```bash
AUTOTABML_LOG_FORMAT=json
AUTOTABML_LOG_LEVEL=INFO
uv run streamlit run app/main.py
```

---

## Testing & CI

```bash
uv run pytest                                # Unit tests
uv run pytest -m integration                 # Integration suite
uv run pytest --cov=app --cov-fail-under=65  # Coverage gate
```

| Workflow | Purpose |
| --- | --- |
| [CI](.github/workflows/ci.yml) | Lint (ruff) · unit tests (Python 3.11 + 3.13) · coverage ≥ 65% · E2E smoke |
| [Security](.github/workflows/security.yml) | `detect-secrets` + `gitleaks` on every push and PR |
| [Release](.github/workflows/release-readiness.yml) | Build validation + `twine check` for tagged releases |

CI uses the committed `uv.lock` for deterministic installs before linting,
testing, coverage, and release validation. Dependabot runs weekly dependency
updates.

> [!NOTE]
> The verified-output numbers below come from running the project's own
> test, lint, and verification scripts on a fresh `.venv` with Python 3.12.10 —
> the same commands CI and Security run on every push.

| Check | Command | Result |
| --- | --- | --- |
| Lockfile consistency | `uv lock --check` | passes |
| Unit tests | `pytest tests/ -q` | **729 passed**, 30 deselected |
| Coverage gate | `pytest --cov=app --cov-fail-under=65` | CI gate ≥ 65% |
| Lint | `ruff check app/ tests/ scripts/` | all checks passed |
| Release metadata | `python -m app.release_metadata` | passes |

---

## Known limitations

| Constraint | Detail |
| --- | --- |
| PyCaret requires Python < 3.13 | All other features work on 3.10 – 3.13 |
| TabFM weights | Separate `tabfm-non-commercial-v1.0` license; non-commercial, non-production research only; Python 3.11+ |
| First model use | TabFM and TimesFM require explicit approval before downloading their pinned Hugging Face snapshots |
| GPU training | Requires NVIDIA + CUDA; falls back to CPU automatically |
| Large datasets | 100K+ rows trigger automatic sampling |
| Kaggle | CLI-only; not exposed in the UI |
| Single-user | Designed for individual local use |
| Background concurrency | One training job runs at a time |
| Drift meaning | Input-distribution drift only, not target/concept drift |
| AI summaries | Require an API key or local Ollama |

> [!WARNING]
> TabFM's pretrained weights carry a separate non-commercial research license.
> The app blocks TabFM-derived contexts from registry promotion and deployment
> export to keep that boundary enforced.

---

## Documentation

[docs/README.md](docs/README.md) is the full documentation index (a "where do
I want to go" table plus the complete document map). The tables below cover the
same ground inline.

**Using the app**

| Resource | Description |
| --- | --- |
| [USAGE.md](USAGE.md) | Every page, every CLI command, every configuration knob, in detail |
| [Interactive architecture](docs/autotabml-studio-architecture.html) | Explorable system map with guided views and source evidence |

**Understanding & extending the codebase**

| Resource | Description |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Module map, data-flow diagrams, reliability/security/performance model, extension points |
| [docs/developer-guide.md](docs/developer-guide.md) | Local setup, common commands, test strategy, release hygiene, troubleshooting |
| [docs/operations.md](docs/operations.md) | Day-1 verification, day-2 monitoring, failure modes, reset procedures, container operations |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, verification gates, documentation expectations, PR guidance |

**Upgrading & release history**

| Resource | Description |
| --- | --- |
| [CHANGELOG.md](CHANGELOG.md) | Chronological list of changes (Keep a Changelog format) |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | Step-by-step upgrades across supported release lines |
| [UPGRADE_SUMMARY.md](UPGRADE_SUMMARY.md) | One-page upgrade cheat sheet |
| [RELEASE_NOTES_v0.4.0.md](RELEASE_NOTES_v0.4.0.md) | Current release highlights and upgrade steps |
| [RELEASE_NOTES_v0.3.0.md](RELEASE_NOTES_v0.3.0.md) | Historical v0.3.0 release announcement |
| [RELEASE_NOTES_v0.2.0.md](RELEASE_NOTES_v0.2.0.md) | Historical v0.2.0 release announcement |

**Policies**

| Resource | Description |
| --- | --- |
| [SECURITY.md](SECURITY.md) | Supported versions, disclosure channel, response SLA, hardening guide |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community expectations |
| [LICENSE](LICENSE) | MIT License |

<p align="center">Made with ❤️ by Ahmad Mujtaba</p>
