# Architecture Reference

> **Last reviewed:** 2026-08-17 · **Version:** v0.4.1 · **Audience:** contributors, advanced users

AutoTabML Studio is a **local-first AutoML workbench** with two front ends — a Streamlit UI and a CLI — driven by the same service layer. This document explains how every piece fits together, why it exists, and how to extend it.

---

## Contents

- [Overview & Design Constraints](#overview--design-constraints)
- [High-level Architecture](#high-level-architecture)
- [Module Map](#module-map)
- [Data Flows](#data-flows)
- [ML Engine Integration](#ml-engine-integration)
- [Configuration System](#configuration-system)
- [Security Model](#security-model)
- [Storage Layout](#storage-layout)
- [Observability](#observability)
- [Extension Points](#extension-points)
- [Dependency Constraints](#dependency-constraints)

---

## Overview & Design Constraints

AutoTabML Studio consolidates the complete tabular ML lifecycle — ingest, validate, profile, benchmark, train, predict, track, register, and export — into a single local workspace. The architecture satisfies four hard constraints:

**1. Same answers, UI or CLI.**
A benchmark started from the Streamlit dashboard must produce the same leaderboard as `autotabml benchmark …` from the shell. All business logic lives in `app/` service modules; `app/pages/*` are thin entry points that call services and render results. There is no UI-only or CLI-only logic.

**2. Local-first by default.**
No data leaves the machine unless the user explicitly opts in (a provider API key, a UCI/Kaggle download, a URL fetch, a foundation-model checkpoint). This is enforced by `app/security/safe_http.py` (SSRF guard on every user-supplied URL), the `gatherUsageStats = false` setting in `.streamlit/config.toml`, and the fact that MLflow's tracking and registry URIs both default to local SQLite.

**3. Reproducible.**
MLflow logs aggregate parameters, metrics, and safe summary artifacts for every run. The local SQLite store (`app/storage/`) records jobs, datasets, batch runs, and saved-model metadata. `uv.lock` pins every dependency and CI enforces `uv lock --check` on every push.

**4. Purpose-specific engines, shared data contracts.**
LazyPredict screens broadly, PyCaret and FLAML train deeply, TabFM performs research-only in-context evaluation, and TimesFM 2.5 forecasts time series. Each engine produces a consistent result-bundle shape so `app/tracking/` and compare workflows remain engine-agnostic. Services stay concrete until a second implementation or published extension contract justifies a polymorphic boundary.

---

## High-level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Streamlit UI (app/main.py)                    │
│   Dashboard · Load · Validate · Profile · Benchmark             │
│   Train & Tune · FLAML · Foundation Models · AutoRun            │
│   Predictions · History · Compare · Registry · Settings         │
│                   13 pages, thin entry points                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ thin entry points
┌──────────────────────────┴──────────────────────────────────────┐
│                     CLI (app/cli.py)                            │
│                 32 subcommands via argparse                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ both call the same services
┌──────────────────────────┴──────────────────────────────────────┐
│                      Service Layer (app/)                       │
│                                                                 │
│  Ingestion   Validation   Profiling   Background Jobs           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  ML Engines                             │   │
│  │  LazyPredict · PyCaret · FLAML · TabFM · TimesFM 2.5   │   │
│  └─────────────────────────────────────────────────────────┘   │
│  Prediction  Tracking  Registry  Observability                  │
│  Notebooks   Deployment  Drift   Explainability  Provenance     │
│  Security    Providers   Storage Backends                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                        Storage                                  │
│   MLflow SQLite (mlflow.db) · App SQLite (app_metadata.sqlite3)│
│   artifacts/ directory tree                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Layer responsibilities

| Layer | Location | Responsibility |
| --- | --- | --- |
| UI | `app/main.py`, `app/pages/` | Navigation, rendering, session state — no business logic |
| CLI | `app/cli.py` | Argument parsing, calling services, formatting output — no business logic |
| Service | `app/modeling/`, `app/ingestion/`, `app/prediction/`, etc. | All domain logic, artifact writing, MLflow logging |
| Storage | `app/storage/`, `artifacts/` | SQLite metadata, MLflow backend, artifact files |

---

## Module Map

| Module | Responsibility |
| --- | --- |
| `app/__init__.py` | Package identity: `APP_NAME`, `__version__` (`0.4.0`), `DIST_NAME`, entry points |
| `app/main.py` | Streamlit entry point; calls `build_streamlit_navigation()` from `app/pages/registry.py` |
| `app/cli.py` | 32-command argparse CLI; maps subcommands to service calls |
| `app/autorun.py` | Guided end-to-end AutoML job orchestration — dataset ingestion through model save |
| `app/autorun_worker.py` | Worker thread for `auto-run` background jobs |
| `app/background_jobs.py` | Background job service — submit, track, cancel async jobs |
| `app/concurrency.py` | Concurrency primitives (one-job-at-a-time enforcement) |
| `app/deployment.py` | Deployment bundle export: model + metadata + provenance + FastAPI skeleton |
| `app/drift.py` | Input-distribution drift detection against a saved baseline JSON |
| `app/errors.py` | Shared exception base classes |
| `app/evaluation.py` | Hold-out evaluation utilities used across engine types |
| `app/explainability.py` | SHAP-based model explanation entry point |
| `app/gpu.py` | CUDA detection via `cuda_summary()` |
| `app/logging_config.py` | Logging bootstrap (calls `app/observability/logging_setup.py`) |
| `app/path_utils.py` | Canonical path resolution and workspace root discovery |
| `app/provenance.py` | Model provenance record — dataset hash, run ID, engine, timestamp |
| `app/startup.py` | Local runtime initialization and diagnostics (`doctor` command backend) |
| `app/py.typed` | PEP 561 typed-package marker |
| `app/release_metadata.py` | Release metadata verification (used in release-readiness CI) |
| `app/backends/` | Execution backends: `local_backend.py` (in-process), `colab_mcp_backend.py` (remote Colab) |
| `app/config/` | Pydantic settings (`models.py`), enums (`enums.py`), load/save (`settings.py`) |
| `app/ingestion/` | Multi-source dataset loading: CSV, Excel, URL, Kaggle, UCI, HTML tables; normalizer, metadata hashing |
| `app/modeling/benchmark/` | LazyPredict orchestration — leaderboard, ranking, MLflow logging |
| `app/modeling/flaml/` | FLAML AutoML service — setup, search, leaderboard, MLflow logging |
| `app/modeling/foundation/` | TabFM (`tabfm.py`) and TimesFM 2.5 (`timesfm.py`) adapters; consent gates; checkpoint management |
| `app/modeling/pycaret/` | Full PyCaret pipeline — compare, tune, evaluate, finalize, save; MLflow tracking |
| `app/notebooks/` | Jupyter notebook generation for reproducibility |
| `app/observability/` | Structured logging, correlation context, metrics hooks, tracing |
| `app/pages/` | Streamlit page entry points (13 pages) + shared UI helpers |
| `app/prediction/` | Single-row and batch prediction; model loading; scoring; prediction history |
| `app/profiling/` | ydata-profiling orchestration — HTML + JSON report generation |
| `app/providers/` | LLM provider integrations: OpenAI, Anthropic, Gemini, Ollama; pricing catalog |
| `app/registry/` | MLflow model registry — register, list, version, promote (champion/candidate/archived) |
| `app/security/` | Secret masking, SSRF-safe HTTP, formula-injection-safe CSV, trusted artifact loading |
| `app/state/` | Streamlit session state management (`session.py`) |
| `app/storage/` | SQLite app metadata store — datasets, jobs, batch runs, saved models, projects |
| `app/tracking/` | MLflow run history queries, filtering, side-by-side comparison, description generation |
| `app/validation/` | Data validation — app-native rules + Great Expectations integration |

---

## Data Flows

### Flow 1 — CSV → Benchmark → MLflow Leaderboard

```mermaid
sequenceDiagram
    participant U as User (UI or CLI)
    participant ING as app/ingestion/factory.py
    participant NRM as app/ingestion/normalizer.py
    participant BNS as app/modeling/benchmark/service.py
    participant LZR as app/modeling/benchmark/lazypredict_runner.py
    participant MLF as app/modeling/benchmark/mlflow_tracking.py
    participant ART as app/modeling/benchmark/artifacts.py
    participant STR as app/storage/

    U->>ING: load dataset (CSV path or URI)
    ING->>NRM: normalize to pandas DataFrame + metadata
    NRM-->>BNS: DataFrame, DatasetMetadata
    BNS->>LZR: run LazyClassifier / LazyRegressor
    LZR-->>BNS: leaderboard DataFrame
    BNS->>MLF: log experiment, params, metrics
    BNS->>ART: write leaderboard CSV + summary JSON to artifacts/benchmark/
    BNS->>STR: record batch run in app_metadata.sqlite3
    BNS-->>U: BenchmarkResult (leaderboard, run_id, artifact paths)
```

**Key invariants:**
- The ingestion factory (`app/ingestion/factory.py`) selects the correct loader via `IngestionSourceType` and always returns a normalized DataFrame + `DatasetMetadata`.
- LazyPredict is called through a thin `lazypredict_runner.py` wrapper that handles optional-import guards; if the `benchmark` extra is not installed, the service raises a clear `ImportError`.
- MLflow logging is additive — the benchmark produces useful output whether or not MLflow is configured.

---

### Flow 2 — Saved Model → Batch Prediction → History

```mermaid
sequenceDiagram
    participant U as User (UI or CLI)
    participant ING as app/ingestion/factory.py
    participant PRED as app/prediction/batch_predict.py
    participant LDR as app/prediction/secure_loader.py
    participant TA as app/security/trusted_artifacts.py
    participant SCR as app/prediction/scorer.py
    participant HST as app/prediction/history.py
    participant ART as app/prediction/artifacts.py

    U->>ING: load new dataset
    U->>LDR: load model (local path or MLflow URI)
    LDR->>TA: canonicalize path, check trust root, verify SHA256 sidecar
    TA-->>LDR: verified model object
    LDR-->>PRED: model + metadata
    ING-->>PRED: DataFrame
    PRED->>SCR: score DataFrame against model
    SCR-->>PRED: predictions DataFrame
    PRED->>ART: write predictions CSV to artifacts/predictions/
    PRED->>HST: append entry to history.jsonl
    PRED-->>U: PredictionResult (predictions, artifact path, history entry)
```

**Key invariants:**
- Every local model load goes through `app/security/trusted_artifacts.py` — no pickle is deserialized without path canonicalization, trust-root check, and SHA256 sidecar verification.
- The `prediction_column` and `prediction_score_column` names are configurable via `PredictionSettings` so downstream consumers have stable field names.

---

### Flow 3 — Auto Run → Background Job → Artifacts

```mermaid
sequenceDiagram
    participant U as User (UI or CLI)
    participant AR as app/autorun.py
    participant BG as app/background_jobs.py
    participant ARW as app/autorun_worker.py
    participant ING as app/ingestion/
    participant FL as app/modeling/flaml/service.py
    participant EVAL as app/evaluation.py
    participant EXP as app/explainability.py
    participant PRV as app/provenance.py
    participant DFT as app/drift.py
    participant STR as app/storage/

    U->>AR: submit auto-run (dataset, target, mode, time_budget)
    AR->>BG: create background job, return job_id
    BG->>ARW: spawn worker thread
    ARW->>ING: load + normalize dataset
    ARW->>FL: FLAML AutoML search
    FL-->>ARW: best model + leaderboard
    ARW->>EVAL: holdout evaluation
    ARW->>EXP: SHAP explanation (if available)
    ARW->>PRV: write provenance record
    ARW->>DFT: save drift baseline
    ARW->>STR: record completed run in app_metadata.sqlite3
    ARW-->>BG: update job status to COMPLETED
    U->>BG: job-status <job_id> (poll or check)
    BG-->>U: JobRecord (status, result paths, metrics)
```

**Key invariants:**
- `auto-run` is the only workflow that runs entirely asynchronously. `background_jobs.py` enforces one active training job at a time via `app/concurrency.py`.
- The worker writes a row-free drift baseline so future `drift-check` runs can detect distribution shift without retaining raw training data.

---

## ML Engine Integration

All ML engines follow the same integration pattern:

1. A **service class** in `app/modeling/<engine>/service.py` exposes a `run()` method that accepts a DataFrame + config and returns a typed result bundle.
2. An **MLflow tracking module** (`mlflow_tracking.py`) handles all experiment logging — parameters, metrics, safe artifacts. Calling the service never requires MLflow to be installed; tracking is additive.
3. An **artifacts module** (`artifacts.py`) writes engine-specific outputs (leaderboards, model files, plots) to the configured directory under `artifacts/`.
4. **Schemas** (`schemas.py`) define Pydantic models for inputs, outputs, and persisted records.

### Engine comparison

| Engine | Extra | Python | Service class | Purpose |
| --- | --- | --- | --- | --- |
| **LazyPredict** | `benchmark` | 3.10–3.13 | `benchmark.service.BenchmarkService` | Quick baseline sweep across 30+ algorithms |
| **PyCaret** | `experiment` | **< 3.13** | `pycaret.service.PyCaretExperimentService` | Full compare → tune → evaluate → finalize pipeline |
| **FLAML** | `flaml` | 3.10–3.13 | `flaml.service.FlamlAutoMLService` | Time-budget constrained AutoML search |
| **TabFM** | `tabfm` | **≥ 3.11** | `foundation.tabfm.TabFMService` | Research-only in-context tabular evaluation (non-commercial license) |
| **TimesFM 2.5** | `timesfm` | 3.10–3.13 | `foundation.timesfm.TimesFMService` | Point + quantile time-series forecasting |

### TabFM-specific constraints

> [!WARNING]
> TabFM's pretrained weights carry the `tabfm-non-commercial-v1.0` license. The CLI requires `--accept-tabfm-license` and `--allow-download` flags before the checkpoint is fetched. The service sets a `research_only=True` marker on all saved TabFM contexts; `app/registry/` and `app/deployment.py` both reject contexts carrying this marker to prevent accidental promotion or export.

Foundation-model checkpoints are managed by `app/modeling/foundation/checkpoints.py` — each model is pinned to an exact Hugging Face revision and is not downloaded until the user explicitly opts in.

### uv dependency conflicts

| Conflict pair | Reason | Resolution |
| --- | --- | --- |
| `tabfm` + `profiling` | TabFM requires `typeguard<3`; ydata-profiling requires `typeguard>=4` | Sync into separate virtual environments |

---

## Configuration System

All settings are managed by `app/config/models.py` using **Pydantic Settings**. Resolution order (later overrides earlier):

1. Pydantic field defaults (in `models.py`)
2. Persisted user preferences (`~/.autotabml/settings.json`, written by the Settings page)
3. Environment variables (`AUTOTABML_` prefix, `__` as nested delimiter)

> [!NOTE]
> API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) are intentionally **not** prefixed with `AUTOTABML_` because provider client libraries read them directly. They are never written to `settings.json`.

### Settings sections

| Section | Env prefix | Key fields |
| --- | --- | --- |
| `workspace_mode` | `AUTOTABML_WORKSPACE_MODE` | `dashboard` (default) or `notebook` |
| `execution` | `AUTOTABML_EXECUTION__*` | `backend`: `colab_mcp` (default) or `local` |
| `provider` | `AUTOTABML_PROVIDER__*` | `provider` (LLMProvider enum), `base_url` |
| `ui` | `AUTOTABML_UI__*` | `selected_model_id` |
| `artifacts` | `AUTOTABML_ARTIFACTS__*` | `root_dir` (default `artifacts/`); all subdirs auto-rebase |
| `database` | `AUTOTABML_DATABASE__*` | `path` (default `artifacts/app/app_metadata.sqlite3`), `initialize_on_startup` |
| `validation` | `AUTOTABML_VALIDATION__*` | `min_row_threshold`, `null_warn_pct` (50%), `null_fail_pct` (95%), `gx_context_dir` |
| `profiling` | `AUTOTABML_PROFILING__*` | `default_mode`, `large_dataset_row_threshold` (50k), `sampling_row_threshold` (200k) |
| `benchmark` | `AUTOTABML_BENCHMARK__*` | `default_test_size` (0.2), `timeout_seconds` (120), classification/regression ranking metrics |
| `pycaret` | `AUTOTABML_PYCARET__*` | `default_train_size` (0.7), `default_fold` (5), compare/tune metrics, `default_tracking_mode` |
| `flaml` | `AUTOTABML_FLAML__*` | `default_time_budget` (120s), `default_estimator_list`, classification/regression metrics |
| `mlflow` | `AUTOTABML_MLFLOW__*` | `tracking_uri`, `registry_uri`, `champion_alias`, `candidate_alias`, `registry_enabled` |
| `prediction` | `AUTOTABML_PREDICTION__*` | `schema_validation_mode`, `prediction_column_name`, `history_path` |

Top-level flags: `mlflow_descriptions_enabled` (True), `llm_descriptions_enabled` (False), `ollama_base_url` (`http://localhost:11434`).

The `load_settings()` / `save_settings()` functions in `app/config/settings.py` handle serialization. API keys are read from `st.session_state` (UI) or environment (CLI) and are never written to disk.

---

## Security Model

Four security controls are active by default. Each lives in `app/security/` and is called at its relevant trust boundary.

### 1. Secret masking — `app/security/masking.py`

Strips credential patterns from all log lines and exception messages before they are emitted or displayed:

- `sk-...` (OpenAI-style keys)
- `AI...` (Gemini/Anthropic-style keys)
- `Bearer <token>` (HTTP authorization headers)
- `scheme://user:password@host` (URI-embedded credentials)

`safe_error_message(exc)` wraps any exception for display; `redact_key_in_text(text)` is applied by the JSON log formatter.

### 2. Formula injection prevention — `app/security/safe_csv.py`

Any cell whose stripped value starts with `=`, `+`, `-`, or `@` is prefixed with a single quote before CSV export. This prevents a malicious dataset from triggering formula execution when the exported CSV is opened in Excel, LibreOffice, or Google Sheets. Applied in `sanitize_csv_dataframe()` and `dataframe_to_safe_csv()`.

### 3. SSRF-resistant HTTP client — `app/security/safe_http.py`

All user-supplied URLs (data ingestion paths) are validated before any network request:

- **Scheme allowlist:** only `http` and `https` pass.
- **IP blocklist:** every hostname is resolved via `socket.getaddrinfo`; loopback, link-local, private (RFC 1918/4193), multicast, reserved, and unspecified ranges are rejected.
- **Redirect tracing:** redirects are followed manually (not delegated to `httpx`) with a configurable hop cap; each redirected URL is re-validated.
- **Response limits:** responses are streamed and aborted at a configurable byte cap (default 200 MiB); `Content-Length` headers that exceed the cap are rejected immediately.
- **Content-Type allowlist:** only tabular content types (CSV, Excel, TSV) are accepted.
- **No environment proxies:** `trust_env=False` on the `httpx.Client` instance.

### 4. Trusted artifact loading — `app/security/trusted_artifacts.py`

Three layers of validation before any model file is deserialized:

1. **Path canonicalization** — every path is resolved with `.resolve(strict=True)`.
2. **Trust root check** — the resolved path must remain within a configured approved directory (`supported_local_model_dirs` or `local_model_metadata_dirs` from `PredictionSettings`).
3. **SHA256 sidecar verification** — every saved artifact has a `.sha256` companion written at save time; the hash is recomputed and compared before any `pickle.load()` or `skops` deserialization.

`require_trusted_source(metadata)` additionally enforces the `trusted_source: "autotabml_trusted_local_model_v1"` marker in the metadata JSON, and `require_metadata_checksum(metadata)` enforces the `model_sha256` field. Both must pass before `load_verified_pickle_artifact()` or `load_verified_skops_artifact()` proceed.

---

## Storage Layout

```
~/.autotabml/
  settings.json          # User preferences (no API keys)

artifacts/               # AUTOTABML_ARTIFACTS__ROOT_DIR
  validation/            # Great Expectations outputs (HTML data docs, JSON summary)
  profiling/             # ydata-profiling HTML reports + JSON summaries
  benchmark/             # LazyPredict leaderboard CSV + summary JSON per run
  experiments/           # PyCaret run artifacts; snapshots/ (model snapshots)
    foundation/          # TabFM context JSON + .sha256; TimesFM outputs
  flaml/                 # FLAML leaderboard, best-model pickle + .sha256
  models/                # Finalized .pkl files + .sha256 sidecars + metadata JSON
  predictions/           # Batch prediction CSVs; history.jsonl (prediction log)
  comparisons/           # compare-runs JSON artifacts
  mlflow/                # mlflow.db (MLflow tracking + registry, SQLite)
  app/                   # app_metadata.sqlite3 (datasets, jobs, batch runs, saved models)
  tmp/                   # Temporary files (24h retention; 48h for failed/partial runs)
```

**Override policy:** setting `AUTOTABML_ARTIFACTS__ROOT_DIR` rebases all subdirectories. Individual subdirectories can be overridden independently (e.g. `AUTOTABML_DATABASE__PATH` for the SQLite path, `AUTOTABML_MLFLOW__TRACKING_URI` for the MLflow URI).

**SQLite metadata store** (`app/storage/`): the `store.py` / `sqlite_connector.py` layer manages schema migrations (`migrations.py`) and exposes typed repository objects for datasets, jobs, batch runs, projects, and saved models via `app/storage/repositories/`.

**MLflow backend**: local SQLite at `artifacts/mlflow/mlflow.db` by default. The tracking URI and registry URI can be pointed at a remote MLflow server by setting `AUTOTABML_MLFLOW__TRACKING_URI` and `AUTOTABML_MLFLOW__REGISTRY_URI`.

---

## Observability

All observability is local-first and opt-in at each level.

### Structured logging — `app/observability/logging_setup.py`

| Setting | Env var | Values | Default |
| --- | --- | --- | --- |
| Log format | `AUTOTABML_LOG_FORMAT` | `text`, `json` | `text` |
| Log level | `AUTOTABML_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

In `json` mode the formatter runs `redact_key_in_text()` (from `app/security/masking.py`) before emitting each line, so credentials never appear in log output.

### Correlation context — `app/observability/context.py`

Training and prediction workflows attach structured fields to every log line via a context manager:

- `correlation_id` — unique per request/job
- `run_id` — MLflow run ID (when available)
- `experiment_name`, `run_name`, `task_type`

This allows log lines from concurrent sessions to be filtered and correlated without a centralized log aggregator.

### Metrics hooks — `app/observability/metrics.py`

Metrics are emitted through a hook interface. The default backend is in-process (counters and gauges in memory). The interface can be swapped for Prometheus, StatsD, or OTLP adapters at startup without modifying workflow code.

### Distributed tracing — `app/observability/tracing.py`

Tracing is a **no-op by default**. It upgrades automatically when `opentelemetry-api` is installed and a trace exporter is configured. No code changes are required to enable it.

---

## Extension Points

### Add a new ingestion source

1. Create a loader class in `app/ingestion/` implementing the base interface from `app/ingestion/base.py`.
2. Register it in `app/ingestion/factory.py` under a new `IngestionSourceType` enum value (defined in `app/config/enums.py`).
3. Add the new source type to `--source-type` choices in `app/cli.py`.
4. Add a corresponding UI control in `app/pages/dataset_intake_page.py`.

### Add a new ML engine

1. Create a module under `app/modeling/<engine>/` with at minimum: `service.py`, `mlflow_tracking.py`, `schemas.py`, `artifacts.py`, `errors.py`.
2. The service class must expose a `run()` method returning a typed result bundle compatible with `app/tracking/` history queries.
3. Add the optional extra to `pyproject.toml` and document the Python version constraint.
4. Add CLI subcommands in `app/cli.py`.
5. Add a Streamlit page in `app/pages/` (thin entry point calling your service).

### Add a new prediction loader

1. Implement a loader in `app/prediction/` following the interface in `app/prediction/base.py`.
2. Ensure all model files pass through `app/security/trusted_artifacts.py` before deserialization — never bypass the SHA256 and trust-root checks.
3. Register the loader in `app/prediction/loader.py`.

### Add a new Streamlit page

1. Create `app/pages/<name>_page.py` with a `render_<name>_page()` function.
2. Register the page in `app/pages/registry.py` by adding an entry to the navigation structure.
3. Keep all business logic in a service module — the page function should contain only Streamlit calls and service invocations.

---

## Dependency Constraints

### Python version matrix

| Feature | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 |
| --- | --- | --- | --- | --- |
| Core UI + CLI | ✅ | ✅ | ✅ | ✅ |
| Benchmark (LazyPredict) | ✅ | ✅ | ✅ | ✅ |
| FLAML AutoML | ✅ | ✅ | ✅ | ✅ |
| TimesFM 2.5 | ✅ | ✅ | ✅ | ✅ |
| PyCaret experiment | ✅ | ✅ | ✅ | ❌ |
| TabFM | ❌ | ✅ | ✅ | ✅ |

> [!TIP]
> Python 3.12 is the recommended version for the broadest extra coverage — PyCaret, TabFM, and all other extras work on 3.12.

### Optional extras map

| Extra | Installs | Notes |
| --- | --- | --- |
| `benchmark` | LazyPredict, scikit-learn, XGBoost, LightGBM, CatBoost, MLflow | Baseline sweep |
| `experiment` | PyCaret ≥ 3.0.4, MLflow, XGBoost, LightGBM, CatBoost | Python < 3.13 only |
| `flaml` | FLAML[automl] ≥ 2.5.0, scikit-learn, LightGBM, XGBoost | |
| `tabfm` | tabfm[pytorch] ≥ 1.0.1, huggingface-hub | Python ≥ 3.11; **conflicts with `profiling`** |
| `timesfm` | timesfm[torch] ≥ 2.0.2, huggingface-hub | |
| `profiling` | ydata-profiling ≥ 4.18.1 | **conflicts with `tabfm`** |
| `validation` | great_expectations ≥ 1.16 | |
| `kaggle` | kaggle | CLI-only dataset access |
| `uci` | ucimlrepo | UCI ML Repository access |
| `gpu` | XGBoost, LightGBM, CatBoost (GPU variants) | Requires NVIDIA + CUDA |
| `colab` | mcp ≥ 1.0 | Colab MCP execution backend |
| `providers` | openai ≥ 1.40, anthropic ≥ 0.40, google-genai ≥ 1.0, ollama ≥ 0.4 | LLM summaries |
| `explain` | shap ≥ 0.46 | SHAP explanations |
| `serve` | fastapi ≥ 0.115, uvicorn ≥ 0.30 | Deployment bundle server |

Core (always installed): numpy, pandas, streamlit, pydantic, pydantic-settings, httpx, openpyxl, xlrd, lxml, python-dotenv, nbformat.

---

*For developer setup, test strategy, and release hygiene see [docs/developer-guide.md](developer-guide.md). For day-2 operations and failure modes see [docs/operations.md](operations.md).*
