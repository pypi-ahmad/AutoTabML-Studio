<div align="center">

# AutoTabML Studio

<img src="docs/assets/social-preview/autotabml-social-preview.png" alt="AutoTabML Studio" width="100%">

**Local-first automated machine learning workbench for tabular data.**

Go from a raw CSV to a trained, evaluated, deployable model — entirely on your machine.
No cloud account, no outbound telemetry. A single service layer drives both the Streamlit UI and the CLI,
so every workflow is reproducible either way.

[![CI](https://img.shields.io/github/actions/workflow/status/pypi-ahmad/AutoTabML-Studio/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI)](https://github.com/pypi-ahmad/AutoTabML-Studio/actions/workflows/ci.yml)
[![Security](https://img.shields.io/github/actions/workflow/status/pypi-ahmad/AutoTabML-Studio/security.yml?branch=main&style=for-the-badge&logo=shieldsdotio&logoColor=white&label=Security)](https://github.com/pypi-ahmad/AutoTabML-Studio/actions/workflows/security.yml)
[![License](https://img.shields.io/badge/License-MIT-0F172A?style=for-the-badge)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB?style=for-the-badge&logo=python&logoColor=white)

![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-AutoML-1D4ED8?style=flat-square)
![FLAML](https://img.shields.io/badge/FLAML-AutoML-00A4EF?style=flat-square&logo=microsoft&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-DataFrames-150458?style=flat-square&logo=pandas&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Schemas-E92063?style=flat-square&logo=pydantic&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Metadata-003B57?style=flat-square&logo=sqlite&logoColor=white)

**Repository:** [github.com/pypi-ahmad/AutoTabML-Studio](https://github.com/pypi-ahmad/AutoTabML-Studio)

[Features](#features) · [Quick start](#quick-start) · [Workflow](#workflow) · [CLI](#cli) · [Configuration](#configuration) · [Architecture](#architecture) · [Docs](#documentation) · [Community](#community)

</div>

---

## Contents

- [What is AutoTabML Studio?](#what-is-autotabml-studio)
- [Features](#features)
- [Quick start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Install](#install)
  - [Run](#run)
- [Workflow](#workflow)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
  - [Module map](#module-map)
  - [Tech stack](#tech-stack)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
  - [Environment variables](#environment-variables)
- [Observability](#observability)
- [Testing & CI](#testing--ci)
- [Known limitations](#known-limitations)
- [Documentation index](#documentation-index)
- [Community](#community)
- [Disclaimer](#disclaimer)

---

## What is AutoTabML Studio?

Most tabular ML work is scattered across notebooks, throwaway scripts, and
manual model-file management: a benchmark run in one notebook, a tuned model
saved somewhere, a prediction script duplicated for every project. There is
rarely one place that remembers what data trained a model, what it scored,
or which version is deployed.

AutoTabML Studio consolidates that entire lifecycle — ingest, validate, profile,
benchmark, train, predict, track, register, and export — into one local workspace.
It ships a **Streamlit UI** for interactive exploration and a **CLI** for scripted
or headless runs; both use the same service layer, so results are identical either way.

- **Zero cloud dependency** — data never leaves your machine; no outbound telemetry by default
- **Three AutoML engines** — LazyPredict for quick sweeps, PyCaret for full experiments, Microsoft FLAML for time-budget search
- **Foundation models** — Google TabFM (research-only) and TimesFM 2.5 (forecasting), both revision-pinned and local
- **End-to-end tracking** — MLflow logs metrics, parameters, and safe artifacts; compare, version, and promote models from one place
- **Production-grade quality** — 729 unit tests, ≥ 65% CI coverage gate, `ruff` lint, secret scanning on every push

> [!NOTE]
> "Local-first" is a hard boundary: MLflow runs against a local SQLite backend,
> the model registry is a local store, and network calls only happen when you
> explicitly opt in (an LLM provider key, a Kaggle download, a foundation-model
> checkpoint fetch).

---

## Features

<table>
<tr>
<td width="50%">

**Data preparation**
- Multi-source ingestion — CSV, Excel, TSV, URLs, HTML tables, UCI ML Repository, optional Kaggle
- Bounded CSV parsing — avoids intermediate chunk copies on large files
- Quality validation — app-native checks + optional Great Expectations integration
- EDA profiling — `ydata-profiling` HTML reports with large-dataset sampling

</td>
<td width="50%">

**Modeling**
- **Quick Benchmark** — screen 30+ algorithms via LazyPredict in seconds
- **Train & Tune** — full PyCaret pipeline: compare → tune → evaluate → finalize → save
- **FLAML AutoML** — fast, cost-efficient hyperparameter search with time-budget control
- **Google TabFM** — non-commercial research evaluation for classification and regression
- **Google TimesFM 2.5** — single/grouped time-series forecasts with q10–q90 uncertainty
- **Auto Run** — guided end-to-end pipeline as a cancellable background job

</td>
</tr>
<tr>
<td width="50%">

**Predictions & evaluation**
- Batch and single-row scoring (form-based or JSON input)
- Model testing against held-out data with ground-truth labels
- Drift detection — compare new data against a saved baseline
- SHAP-based model explainability
- Downloadable, Colab-compatible notebooks auto-generated for every run

</td>
<td width="50%">

**Operations**
- MLflow model registry — Champion / Candidate / Archived lifecycle
- Run history, side-by-side comparison, and algorithm leaderboards
- Structured observability — JSON logs, metrics hooks, optional OpenTelemetry tracing
- AI-generated summaries via OpenAI, Anthropic, Gemini, or local Ollama
- Export deployment bundles (model + metadata + provenance + FastAPI server)

</td>
</tr>
</table>

---

## Quick start

### Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.10 – 3.13 | `uv` defaults to 3.12 via `.python-version`; use 3.11 or 3.12 for PyCaret support |
| [uv](https://docs.astral.sh/uv/) | Fast Python package manager; install with `pip install uv` |
| OS | Windows, macOS, Linux |

### Install

```bash
git clone https://github.com/pypi-ahmad/AutoTabML-Studio.git
cd AutoTabML-Studio

# Sync base environment (UI + CLI, no ML engines)
uv sync --locked --group dev
```

Install only the extras you need:

```bash
uv sync --locked --group dev --extra benchmark    # LazyPredict + boosted baselines
uv sync --locked --group dev --extra experiment   # PyCaret full pipeline (Python 3.11/3.12 only)
uv sync --locked --group dev --extra flaml        # Microsoft FLAML AutoML
uv sync --locked --group dev --extra validation   # Great Expectations
uv sync --locked --group dev --extra profiling    # ydata-profiling EDA reports
uv sync --locked --group dev --extra tabfm        # Google TabFM (Python 3.11+; research-only)
uv sync --locked --group dev --extra timesfm      # Google TimesFM 2.5
uv sync --locked --group dev --extra providers    # OpenAI / Anthropic / Gemini / Ollama
```

> [!IMPORTANT]
> `tabfm` and `profiling` declare a `uv` conflict — TabFM requires `typeguard<3`
> while `ydata-profiling` requires `typeguard>=4`. Sync them into separate virtual
> environments if you need both.

### Run

**Windows:** double-click **`Launch AutoTabML Studio.cmd`** in the repo root.
It opens the UI at `http://localhost:8561`.

**Terminal:**

```bash
uv run autotabml init-local-storage   # Initialize SQLite + artifact directories
uv run autotabml doctor               # Verify runtime dependencies
uv run streamlit run app/main.py      # Launch the Streamlit UI
```

> [!TIP]
> Run `autotabml doctor` after install to confirm all required paths, the
> database, and any optional dependency groups are wired up correctly.

---

## Workflow

The recommended path is **Auto Run**: confirm a target column, review the inferred
task type and FLAML plan, then launch a cancellable background job. It saves the
model, holdout evaluation, provenance, and a row-free drift baseline automatically.

```
Load Data → Validate → Profile → Benchmark → Train / FLAML / Foundation → Predict → Compare → Register
```

| Step | What happens |
| --- | --- |
| **Load Data** | Upload CSV/Excel, paste a URL, pick a UCI dataset, or pull from Kaggle |
| **Validate** *(optional)* | Check for nulls, schema issues, leakage, and distribution anomalies |
| **Profile** *(optional)* | Generate a visual EDA summary with `ydata-profiling` |
| **Quick Benchmark** | Screen 30+ algorithms — ranked leaderboard in seconds |
| **Train & Tune** | Fine-tune the best algorithm with PyCaret and save a production model |
| **FLAML AutoML** | Run time-budget or iteration-budget constrained AutoML |
| **Foundation Models** | Evaluate TabFM (research-only) or forecast with TimesFM 2.5 |
| **Predict** | Score new data (single row or batch file) with any saved model |
| **Compare & Register** | Review run history, compare results, promote models to Champion |
| **Export** | Package a model + FastAPI server bundle for deployment |

See **[USAGE.md](USAGE.md)** for the full step-by-step guide with screenshots.

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

[Open the interactive architecture map](docs/autotabml-studio-architecture.html) — explorable component map with guided views and source-linked evidence.

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit UI (app/pages/)              │
│  Dashboard · Load · Validate · Profile · Benchmark          │
│  Train & Tune · FLAML · Foundation Models · AutoRun         │
│  Predict · Compare · History · Registry · Settings          │
├─────────────────────────────────────────────────────────────┤
│                     CLI (app/cli.py)                        │
├──────────────┬──────────────┬──────────────────────────────┤
│  Ingestion   │  Validation  │     Profiling                │
│  (CSV·Excel· │  (native +   │     (ydata-profiling)        │
│  URL·Kaggle· │  Great Exp.) │                              │
│  UCI·HTML)   │              │                              │
├──────────────┴──────────────┴──────────────────────────────┤
│       ML Engines                                            │
│  LazyPredict · PyCaret · FLAML · TabFM · TimesFM 2.5       │
├─────────────────────────────────────────────────────────────┤
│  Prediction · Tracking · Registry · Observability          │
│  Notebooks · Deployment · Storage · Security               │
├─────────────────────────────────────────────────────────────┤
│  MLflow (SQLite) · SQLite Metadata · artifacts/            │
└─────────────────────────────────────────────────────────────┘
```

Streamlit pages are thin entry points — all business logic lives in the service layer.
Services with a single implementation stay concrete; base classes are reserved for
real multi-implementation or published extension boundaries.

### Module map

<details>
<summary>Click to expand</summary>

| Module | Responsibility |
| --- | --- |
| `app/ingestion/` | Source routing, loaders (CSV, Excel, URL, Kaggle, UCI, HTML), normalization, metadata hashing |
| `app/validation/` | Quality rules, Great Expectations builders, context, and runner |
| `app/profiling/` | ydata-profiling orchestration, selectors, summaries |
| `app/modeling/benchmark/` | LazyPredict orchestration, ranking, MLflow logging |
| `app/modeling/pycaret/` | PyCaret compare, tune, evaluate, finalize, save |
| `app/modeling/flaml/` | FLAML AutoML service, artifacts, tracking |
| `app/modeling/foundation/` | Pinned TabFM/TimesFM adapters, consent gates, artifacts, context persistence |
| `app/prediction/` | Model discovery, loading, schema checks, single-row and batch scoring |
| `app/tracking/` | MLflow history queries, comparison, run filtering |
| `app/registry/` | MLflow model registration and stage promotion |
| `app/observability/` | Structured logging, correlation context, metrics hooks, optional tracing |
| `app/storage/` | SQLite metadata store, recorders, repositories |
| `app/providers/` | LLM integrations (OpenAI, Anthropic, Gemini, Ollama) + pricing catalog |
| `app/notebooks/` | Jupyter notebook generation |
| `app/security/` | Secret masking, SSRF-resistant HTTP, formula-injection-safe CSV, trusted artifact loader |
| `app/config/` | Pydantic settings, enums, environment binding |
| `app/pages/` | Streamlit page entry points (20 pages) |
| `app/cli.py` | CLI entry point (32 commands) |
| `app/autorun.py` | Guided end-to-end AutoML job orchestration |
| `app/deployment.py` | Deployment bundle export (model + FastAPI server) |
| `app/drift.py` | Input-distribution drift detection |
| `app/explainability.py` | SHAP model explanation |

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
| Security | SSRF-safe HTTP, formula-injection-safe CSV, checksum-verified model loading |
| Testing | pytest (729 tests), pytest-cov, pytest-asyncio, respx |

---

## CLI reference

All commands are run as `uv run autotabml <command>` (or `autotabml <command>` with the venv active).

### Diagnostics

| Command | Description |
| --- | --- |
| `info` | Show version, workspace mode, execution backend, artifact paths |
| `doctor` | Run startup diagnostics (CUDA, DB, artifact dirs, stale files) |
| `init-local-storage` | Initialize SQLite database and artifact directories |

### Data

| Command | Description |
| --- | --- |
| `validate <csv> --target <col>` | Run data validation (null checks, schema, Great Expectations) |
| `profile <csv>` | Generate EDA profiling report (ydata-profiling HTML + JSON) |
| `uci-list [--search <term>]` | Browse/search the UCI ML Repository catalog |

### Modeling

| Command | Description |
| --- | --- |
| `benchmark <csv> --target <col>` | Screen 30+ algorithms with LazyPredict |
| `experiment-run <csv> --target <col>` | PyCaret compare_models across all estimators |
| `experiment-tune <csv> --target <col> --model <id>` | PyCaret tune_model for one estimator |
| `experiment-evaluate <csv> --target <col> --model <id>` | PyCaret evaluation plots |
| `experiment-save <csv> --target <col> --model <id>` | PyCaret finalize + save model |
| `flaml-run <csv> --target <col> --time-budget 120` | FLAML AutoML search with time budget |
| `flaml-save <csv> --target <col> --save-name <name>` | FLAML search + save best model |
| `tabfm-run <csv> --target <col> --accept-tabfm-license --allow-download` | TabFM holdout eval (research-only) |
| `timesfm-forecast <csv> --timestamp <col> --target <col> --horizon 12 --allow-download` | TimesFM 2.5 forecast |
| `auto-run <csv> --target <col> --mode auto --time-budget 120` | Guided end-to-end AutoML job |

### Predictions & operations

| Command | Description |
| --- | --- |
| `predict-single --model <path> --input '{"col": val}'` | Score a single JSON row |
| `predict-batch <csv> --model <path>` | Score a full dataset file |
| `predict-history [--limit 10]` | List recent prediction job history |
| `drift-check <csv> --baseline <json>` | Compare new data against a saved drift baseline |
| `explain --model <path>` | Print saved SHAP explanation artifact |
| `deploy-export --model <pkl> --metadata <json> --output <dir>` | Export a deployment bundle |

### History & registry

| Command | Description |
| --- | --- |
| `history-list [--run-type <type>] [--limit 20]` | List MLflow runs with optional filters |
| `history-show <run-id>` | Show parameters, metrics, and artifacts for one MLflow run |
| `compare-runs <run-id-a> <run-id-b>` | Side-by-side metric and config diff of two runs |
| `batch-history [--limit 20]` | List AutoML batch run history from SQLite |
| `batch-show <run-id>` | Show dataset-level detail for one batch run |
| `registry-list` | List all registered models |
| `registry-show <model-name>` | List all versions of one registered model |
| `registry-register --run-id <id> --name <name>` | Register a model with the MLflow registry |
| `registry-promote --name <name> --version <n> --stage champion` | Promote a model version |

### Background jobs

| Command | Description |
| --- | --- |
| `job-list` | List recent background jobs |
| `job-status <job-id>` | Show full JSON record for one job |
| `job-cancel <job-id>` | Cancel one active background job |

---

## Configuration

Settings resolve in this order: **Pydantic defaults → `~/.autotabml/settings.json` → environment variables**.

The Streamlit Settings page writes preferences to `~/.autotabml/settings.json`; API keys are kept in session state and environment variables only — never persisted to disk.

### Environment variables

```bash
# Core runtime (AUTOTABML_ prefix, __ as nested delimiter)
AUTOTABML_WORKSPACE_MODE=dashboard          # dashboard | headless
AUTOTABML_EXECUTION__BACKEND=local          # local | colab_mcp
AUTOTABML_MLFLOW__TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db
AUTOTABML_LOG_FORMAT=json                   # json | text
AUTOTABML_LOG_LEVEL=INFO

# LLM provider keys (no prefix — never committed)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Optional integrations
KAGGLE_USERNAME=...
KAGGLE_KEY=...
OLLAMA_BASE_URL=http://localhost:11434       # default
```

See [.env.example](.env.example) for the complete variable list. Copy it to `.env` and fill in your values.

> [!IMPORTANT]
> Never commit API keys. The security workflow runs `detect-secrets` and
> `gitleaks` on every push and PR to prevent accidental credential exposure.

---

## Observability

All observability is local-first — nothing is exported unless you explicitly wire an exporter.

- `AUTOTABML_LOG_FORMAT=json` emits one JSON document per log line to stderr
- Training and prediction workflows attach correlation fields (`correlation_id`, `run_id`, `experiment_name`) automatically
- Metrics hooks in `app/observability` can swap from the default in-process backend to Prometheus, StatsD, or OTLP without touching workflow code
- Tracing is a no-op by default; it upgrades automatically when `opentelemetry-api` is installed and configured

```bash
AUTOTABML_LOG_FORMAT=json AUTOTABML_LOG_LEVEL=INFO uv run streamlit run app/main.py
```

---

## Testing & CI

```bash
uv run pytest                                 # Run unit tests
uv run pytest -m integration                  # Integration suite
uv run pytest --cov=app --cov-fail-under=65   # Coverage gate (≥ 65%)
uv run ruff check app/ tests/ scripts/        # Lint
uv lock --check                               # Lockfile consistency
```

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| [CI](.github/workflows/ci.yml) | Push / PR | Lint · unit tests (3.11 + 3.13) · coverage ≥ 65% · E2E smoke |
| [Security](.github/workflows/security.yml) | Push / PR | `detect-secrets` + `gitleaks` + `bandit` + `pip-audit` |
| [Release](.github/workflows/release-readiness.yml) | Tag push | Build validation + `twine check` |

> [!NOTE]
> Verified results on a fresh `.venv` with Python 3.12.10:
> 729 tests pass, 30 deselected; ruff all checks passed; `uv lock --check` passes.

---

## Known limitations

| Constraint | Detail |
| --- | --- |
| PyCaret requires Python < 3.13 | All other features work on 3.10 – 3.13 |
| TabFM weights | Separate `tabfm-non-commercial-v1.0` license; research/non-production only; Python 3.11+ |
| Foundation model first use | TabFM and TimesFM require explicit opt-in before downloading their pinned Hugging Face snapshots |
| GPU training | Requires NVIDIA + CUDA; falls back to CPU automatically |
| Large datasets | 100K+ rows trigger automatic sampling |
| Kaggle | CLI-only; not exposed in the Streamlit UI |
| Single-user | Designed for individual local use |
| Background concurrency | One training job at a time |
| Drift scope | Input-distribution drift only — not concept drift |
| AI summaries | Require an API key or a running local Ollama instance |

> [!WARNING]
> TabFM's pretrained weights carry a separate non-commercial research license.
> The app blocks TabFM-derived model contexts from registry promotion and
> deployment export to keep that boundary enforced.

---

## Documentation index

### Using the app

| Resource | Description |
| --- | --- |
| [USAGE.md](USAGE.md) | Complete guide — every page, every CLI command, every configuration knob |
| [Interactive architecture](docs/autotabml-studio-architecture.html) | Explorable system map with guided views and source evidence |

### Understanding the codebase

| Resource | Description |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Module map, data-flow diagrams, reliability/security/performance model, extension points |
| [docs/developer-guide.md](docs/developer-guide.md) | Local setup, common commands, test strategy, release hygiene, troubleshooting |
| [docs/operations.md](docs/operations.md) | Day-1 verification, day-2 monitoring, failure modes, reset procedures, container operations |
| [docs/README.md](docs/README.md) | Full documentation index |

### Upgrading

| Resource | Description |
| --- | --- |
| [CHANGELOG.md](CHANGELOG.md) | Chronological change history (Keep a Changelog format) |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | Step-by-step upgrade instructions |
| [UPGRADE_SUMMARY.md](UPGRADE_SUMMARY.md) | One-page upgrade cheat sheet |
| [RELEASE_NOTES_v0.4.0.md](RELEASE_NOTES_v0.4.0.md) | Current release highlights |

### Policies & community

| Resource | Description |
| --- | --- |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, verification gates, PR guidance |
| [SECURITY.md](SECURITY.md) | Supported versions, vulnerability disclosure, response SLA, hardening guide |
| [DISCLAIMER.md](DISCLAIMER.md) | Data responsibility, no-warranty, no-financial-support statement |
| [SUPPORT.md](SUPPORT.md) | Where to get help, report bugs, and request features |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community expectations |
| [LICENSE](LICENSE) | MIT License |

---

## Community

AutoTabML Studio is free, open-source, and community-driven. Everyone is welcome —
whether you are filing your first bug report, suggesting a feature, or submitting a pull request.

| Want to… | Go here |
| --- | --- |
| Ask a question | [GitHub Discussions](https://github.com/pypi-ahmad/AutoTabML-Studio/discussions) |
| Report a bug | [Bug Report](https://github.com/pypi-ahmad/AutoTabML-Studio/issues/new?template=bug_report.yml) |
| Request a feature | [Feature Request](https://github.com/pypi-ahmad/AutoTabML-Studio/issues/new?template=feature_request.yml) |
| Contribute code | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Report a vulnerability | [Security Advisories](https://github.com/pypi-ahmad/AutoTabML-Studio/security/advisories/new) *(private)* |
| Get help | [SUPPORT.md](SUPPORT.md) |

> [!NOTE]
> **No financial support needed or wanted.** This project is free and will stay free.
> The maintainer does not accept donations, sponsorships, or any financial contributions.
> If you find it useful, open a PR or leave a ⭐.

Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

---

## Disclaimer

All data you load into AutoTabML Studio is processed **entirely on your machine**.
The project maintainer has no access to your data and accepts no responsibility for it.
You are fully responsible for ensuring you have the right to process the data you use
and for complying with applicable regulations (GDPR, HIPAA, CCPA, etc.).

AI summary features send summarized (not raw) context to the LLM provider you configure —
you are responsible for that provider relationship.

See [DISCLAIMER.md](DISCLAIMER.md) for the complete statement including the no-warranty and
no-financial-support policy.

---

<p align="center">Made with ❤️ by <a href="https://github.com/pypi-ahmad">Ahmad Mujtaba</a></p>
