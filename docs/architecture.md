# Architecture Guide

> **Status:** v0.2.0 architecture. Last reviewed: 2026-06-22.

AutoTabML Studio is a **local-first ML workbench** with two faces
(Streamlit UI + CLI) backed by the same service layer. This document
explains *how the pieces fit together* and *why each piece exists*.

## Why this architecture?

Most tabular ML workflows are scattered across notebooks, ad-hoc
scripts, and manual model-file management. The project consolidates
the whole lifecycle — load, validate, profile, benchmark, train,
predict, compare, register — into one workspace that runs on a
laptop. To make that usable for an analyst, the architecture has to
satisfy four hard constraints:

1. **Same answers, UI or CLI.** A benchmark started from the
   Streamlit Dashboard must produce the same leaderboard as
   `autotabml benchmark …` from the shell. This is enforced by
   putting all business logic in `app/` and treating
   `app/pages/*` as thin entry points.
2. **Local-first by default.** No data leaves the machine unless
   the user explicitly points the workbench at a remote endpoint
   (UCI, Kaggle, a URL, an LLM provider). This is enforced by
   `app.security.safe_http` (SSRF guard) and the
   `gatherUsageStats = false` setting in `.streamlit/config.toml`.
3. **Reproducible.** Every run is logged to MLflow with
   parameters, metrics, and artifacts. The local SQLite store
   records jobs and the saved-model metadata. The lockfile
   (`uv.lock`) pins every dependency.
4. **Three AutoML engines, one interface.** LazyPredict for quick
   screening, PyCaret for full experiments, and FLAML for
   budgeted search all expose the same `BenchmarkResultBundle`
   / `ExperimentResultBundle` / `FlamlResultBundle` shape, so
   the History and Compare pages are engine-agnostic.

## High-level diagram

```
┌──────────────────────────────────────────────────────────────┐
│                  Streamlit UI (app/main.py)                  │
│   Dashboard · Load · Validation · Profiling · Benchmark       │
│   Train & Tune · FLAML · Predictions · Models · History      │
│   Compare · Notebook · Registry · Settings                   │
└────────────────────────┬─────────────────────────────────────┘
                         │  thin entry points
┌────────────────────────┴─────────────────────────────────────┐
│                      Service layer                          │
│                                                              │
│  Ingestion  │  Validation  │  Profiling  │  Modeling         │
│  ─────────  │  ──────────  │  ────────  │  ────────         │
│  loaders    │  rules + GX  │  ydata-pro │  benchmark        │
│  metadata   │  summary     │  (optional)│  pycaret          │
│  hash       │  artifacts   │  artifacts │  flaml            │
│                                                              │
│  Prediction │  Tracking    │  Registry  │  Observability    │
│  ─────────  │  ────────    │  ────────  │  ────────────     │
│  loader     │  MLflow      │  MLflow    │  JSON logging     │
│  scorer     │  history     │  promotion │  metrics          │
│  schema     │  compare     │  aliases   │  tracing (otel)   │
└──────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────┐
│  Storage · MLflow (SQLite) · SQLite metadata · artifacts/     │
│  Optional: providers/ → OpenAI · Anthropic · Gemini · Ollama │
└──────────────────────────────────────────────────────────────┘
```

## Module map

| Module                       | Responsibility                                     | Public surface                          |
| ---------------------------- | -------------------------------------------------- | ---------------------------------------- |
| `app/main.py`                | Streamlit entry point. Initializes state, renders the sectioned sidebar, dispatches to page registry. | `streamlit run app/main.py`             |
| `app/cli.py`                 | Argparse CLI. Same service calls as the UI. | `autotabml` console script              |
| `app/startup.py`             | Local-runtime init: artifact dirs, metadata DB, MLflow URI, optional Ollama reachability, optional Colab MCP prereqs. | `init_local_runtime(settings)`          |
| `app/config/`                | Pydantic `AppSettings` with 12 nested sections (artifacts, database, validation, profiling, benchmark, pycaret, flaml, mlflow, prediction, provider, ui, execution). | `AppSettings`                          |
| `app/ingestion/`             | Source routing, loaders, normalization, metadata hashing, error types. One loader per source type (CSV, Excel, HTML table, URL file, UCI repo, Kaggle). | `load_dataset(spec)`                    |
| `app/validation/`            | Target-aware quality rules + optional Great Expectations integration. Artifacts per run (rule results, summary JSON). | `validate_dataset(df, config)`          |
| `app/profiling/`             | `ydata-profiling` orchestration with sampling safeguards for large datasets. | `run_profile(df, config)`              |
| `app/modeling/benchmark/`    | LazyPredict orchestration, ranking, MLflow logging. | `benchmark_dataset(df, config)`        |
| `app/modeling/pycaret/`      | PyCaret compare → tune → evaluate → finalize → save pipeline. OOP API. | `PyCaretExperimentService`             |
| `app/modeling/flaml/`        | FLAML AutoML service, search → save pipeline, leaderboard, MLflow. | `FlamlAutoMLService`                    |
| `app/prediction/`            | Model discovery, secure loaders (pickle / skops with SHA + trust-root check), scoring. | `ModelLoader`, `LoadedModel`           |
| `app/tracking/`              | MLflow queries, history, run comparison, description generator. | `HistoryService`                        |
| `app/registry/`              | MLflow model registration and promotion (champion / candidate / archived). | `RegistryService`                       |
| `app/observability/`         | Structured JSON logging, correlation context, metrics hooks, optional OpenTelemetry tracing. | `configure_observability_logging()`     |
| `app/storage/`               | SQLite metadata store with repositories pattern. | `AppMetadataStore`                      |
| `app/providers/`             | LLM provider abstractions on top of the official SDKs (openai, anthropic, google-genai, ollama). | `BaseProvider`, `ModelItem`             |
| `app/notebooks/`             | Jupyter notebook generation from a run's metadata (no string interpolation; all user values embedded as JSON literals). | `generate_job_notebook(...)`            |
| `app/security/`              | SSRF-resistant HTTP, CSV/Excel formula-injection guard, trusted-artifact checks, secret masking, error types. | `safe_fetch`, `sanitize_csv_dataframe`, `verify_local_artifact` |
| `app/pages/`                 | Streamlit page entry points. Thin wrappers around the service layer. | `dashboard_page`, `benchmark_page`, ... |
| `app/security/trusted_artifacts` | `compute_sha256`, `verify_local_artifact`, `load_verified_pickle_artifact`, `load_verified_skops_artifact` | pickle/skops loading with checksum gate |
| `app/concurrency.py`          | `gather_with_concurrency`, `to_thread_many` for bounded async fan-out. | async helpers                          |
| `app/errors.py`               | `AutoTabMLError`, `log_exception`, `log_and_wrap` for consistent structured failure logging. | logging helpers                         |
| `app/gpu.py`                  | CUDA detection (torch + ctypes fallback). | `is_cuda_available`, `resolve_use_gpu`  |

## Data flow — load a CSV, benchmark, and inspect the leaderboard

1. **User** drops a CSV onto the **Load Data** page.
2. `app.pages.dataset_intake_page` builds a `DatasetInputSpec`
   and calls `app.ingestion.factory.load_dataset`.
3. The factory selects the `CSVLoader`, which calls
   `app.ingestion.normalizer.normalize_to_pandas` and
   `app.ingestion.metadata.extract_dataset_metadata`.
4. The returned `LoadedDataset` is stored in `st.session_state`.
5. On the **Quick Benchmark** page the user picks a target column
   and clicks *Run*. `app.pages.benchmark_page` calls
   `app.modeling.benchmark.service.benchmark_dataset(df, config)`.
6. The service instantiates `LazyPredictBenchmarkService`,
   handles GPU detection, sampling, missing-target rows, and
   ranking metric selection.
7. The `BenchmarkResultBundle` is returned to the page, which
   renders the leaderboard.
8. The service's tracker (`MLflowBenchmarkTracker`) logs
   parameters, metrics, and artifacts to MLflow at the configured
   tracking URI.
9. `app.storage.recorders.record_benchmark_job` writes the
   run to the local SQLite metadata store.
10. The user can now compare runs on the **History** page, see
    the leaderboard again on **Compare**, or register a model on
    the **Registry** page.

## Data flow — load a saved model and score a CSV

1. **User** picks a saved model on the **Predictions** page.
2. `app.prediction.batch_predict` builds a `PredictionRequest`
   (source type, model identifier, schema, task type hint).
3. `app.prediction.loader.ModelLoader.load(request)` selects
   the correct loader (`LocalPyCaretModelLoader`,
   `LocalFlamlModelLoader`, or `MLflowModelLoader`).
4. Local loaders call `verify_local_artifact(...)` first
   (path-in-trust-root + SHA256 sidecar check) before
   `pickle.load` or `skops.io.load`. MLflow loaders call
   `mlflow.pyfunc.load_model`.
5. The returned `LoadedModel` is passed to a scorer which
   reads the input CSV, validates schema, scores, and writes
   the output CSV.
6. The scorer appends a `PredictionHistoryEntry` to
   `~/.autotabml/predictions/history.jsonl` (JSONL, append-only).

## Reliability model

- **Retries** — `app.security.safe_fetch` retries on transient
  httpx errors only; SSRF and size violations are never retried.
- **Checksums** — every local model artifact carries a
  `.sha256` sidecar; loading verifies the sidecar before
  unpickling.
- **Atomic writes** — `app.config.settings.save_settings`
  writes to a temp file and renames into place; no half-written
  settings file.
- **Structured errors** — every caught exception flows through
  `app.errors.log_exception`, which attaches an operation name
  and context fields. The JSON formatter scrubs obvious
  secrets.
- **Correlation context** — every operation runs inside a
  `correlation_scope` that binds `correlation_id` (uuid4 hex),
  `run_id`, `experiment_name`, etc. Logs, metrics, and tracing
  all read the same context.

## Security model

- **SSRF guard** — `app.security.safe_http` resolves hostnames
  and rejects loopback / private / link-local / multicast /
  reserved / unspecified IPs before any TCP connect.
- **Formula-injection guard** — `app.security.safe_csv` prefixes
  dangerous-prefix cells with a single quote before CSV export.
- **Trusted-artifact guard** — `app.security.trusted_artifacts`
  enforces path-in-trust-root + SHA256 sidecar before
  unpickling.
- **Secret masking** — `app.security.masking.redact_key_in_text`
  scrubs common API key shapes from log output.
- **Container hardening** — `Dockerfile` runs as non-root
  (UID 10001) with `/app` writable only to the artifact
  subdirectory.
- **Dependency hygiene** — `uv.lock` is checked on every push;
  `bandit`, `pip-audit`, `detect-secrets`, and `gitleaks` run
  in CI.

## Performance model

- **Lazy loading** — all optional dependencies (PyCaret,
  FLAML, ydata-profiling, Great Expectations) are imported
  inside the function that uses them. Importing `app` does
  not load them.
- **Sampling** — benchmarks over 100K rows and profiles
  over 50K rows/100 columns automatically sample.
- **Streaming** — `app.security.safe_fetch` streams
  responses and aborts at the byte cap; large file
  downloads cannot exhaust memory.
- **Caching** — `app.gpu.is_cuda_available()` and
  `app.gpu.cuda_device_name()` are `@lru_cache`d.
- **HTTP/2** — when `httpx[http2]` is installed, the secure
  client enables it for concurrent fetches.

## Where to find the data

| Data          | Default location                                    | Override via                       |
| ------------- | --------------------------------------------------- | ---------------------------------- |
| Local SQLite  | `artifacts/app/app_metadata.sqlite3`                | `AUTOTABML_DATABASE__PATH`         |
| MLflow store  | `artifacts/mlflow/mlflow.db`                        | `AUTOTABML_MLFLOW__TRACKING_URI`    |
| Settings      | `~/.autotabml/settings.json`                        | n/a (the file is the source of truth) |
| Artifacts     | `artifacts/` tree                                   | `AUTOTABML_ARTIFACTS__ROOT_DIR`    |
| Prediction history | `artifacts/predictions/history.jsonl`           | `AUTOTABML_PREDICTION__HISTORY_PATH` |

## Extension points

- **New ingestion source** — add a loader under
  `app/ingestion/` and register it in
  `app/ingestion/factory.get_loader`.
- **New validation rule** — extend `app/validation/rules.py`
  and update `app/validation/schemas.py` if config changes.
- **New benchmark / experiment handler** — extend the
  service under `app/modeling/<engine>/service.py` and the
  tracker under `app/modeling/<engine>/mlflow_tracking.py`.
- **New prediction loader** — extend
  `app/prediction/loader.py` and register the source type
  in `app/prediction/schemas.py`.
- **New Streamlit page** — add a `*_page.py` under
  `app/pages/`, then register it in `app/pages/registry.py`.
