# Technical reference

## Stack

| Layer | Technology | Why it is here |
|---|---|---|
| UI | Streamlit ≥ 1.30 | Rapid interactive ML UI; pages are thin wrappers over the service layer |
| CLI | Python `argparse` | Stdlib; 32 sub-commands share the same service functions as the UI |
| Data schemas | Pydantic ≥ 2.0, pydantic-settings ≥ 2.0 | Validated config and inter-module contracts; `model_dump_json` used for all JSON serialization |
| DataFrames | pandas ≥ 2.1 | Canonical data carrier between all modules |
| HTTP (outbound) | `httpx` ≥ 0.27 | SSRF-resistant; `app/security/` wraps allowed hosts |
| Metadata persistence | SQLite via stdlib `sqlite3` | Zero-dependency embedded store; `app/storage/sqlite_connector.py` owns the connection pool |
| ML tracking | MLflow ≥ 2.12 (local SQLite backend) | Experiment runs, parameters, metrics, artifact URIs |
| Benchmarking | LazyPredict ≥ 0.3.0, scikit-learn ≥ 1.4 | Screen 30+ algorithms in seconds |
| Training | PyCaret ≥ 3.0.4 (Python < 3.13 only), FLAML ≥ 2.5.0 | Full pipeline and time-budget AutoML respectively |
| Boosted trees | XGBoost ≥ 2.0, LightGBM ≥ 4.0, CatBoost ≥ 1.2 | Available to all modeling engines |
| Foundation models | TabFM (`tabfm[pytorch]` ≥ 1.0.1, Python ≥ 3.11), TimesFM 2.5 (`timesfm[torch]` ≥ 2.0.2) | Research-only; revision-pinned Hugging Face checkpoints |
| Excel / HTML ingestion | `openpyxl`, `xlrd`, `lxml` | Multiple Excel formats and HTML table parsing |
| Explainability | `shap` ≥ 0.46 (`explain` extra) | SHAP-based model explanations |
| Deployment serving | FastAPI ≥ 0.115, Uvicorn ≥ 0.30 (`serve` extra) | Deployment bundle export server stub |
| LLM summaries | `openai`, `anthropic`, `google-genai`, `ollama` (`providers` extra) | AI-generated summaries; all four are optional |
| Notebook export | `nbformat` ≥ 5.10 | Colab-compatible notebook generation |
| Data validation | Great Expectations ≥ 1.16 (`validation` extra) | Optional deep quality suite |
| EDA profiling | ydata-profiling ≥ 4.18 (`profiling` extra) | HTML EDA reports; **conflicts with `tabfm`** (typeguard version clash) |
| Build backend | Hatchling ≥ 1.27 | Wheels and sdist; package root is `app/` |
| Package manager | uv | Lock-file managed (`uv.lock`); `uv sync --locked` for reproducible installs |
| Linter / formatter | Ruff ≥ 0.15, line length 120 | Enforced in CI |
| Type checking | Pyright (advisory), mypy 1.10 (advisory) | Not a CI gate; `app/ingestion/`, `app/profiling/`, `app/notebooks/` are excluded from mypy |
| Security scanning | Bandit, detect-secrets, gitleaks, pip-audit | Run in the `security` CI workflow on every push and PR |

## Important invariants

### Storage — lazy migration

`BaseRepository._read` and `_write` call `self._context.initialize()` before every operation. The `initialize` callable is the metadata store's migration runner. This ensures the schema is current even when the store is created in a context where `init-local-storage` was not run manually.

### Settings — secrets are never persisted

`app/config/settings.py:save_settings` explicitly excludes API keys. Keys exist only in:
1. Environment variables (loaded via `python-dotenv` at startup with `override=False`)
2. Streamlit session state in the UI

The `AUTOTABML_` prefix and `__` nested delimiter are enforced by `pydantic-settings`.

### Provenance — redaction

`app/provenance.py:_sanitize` walks the configuration dict and replaces values whose key contains `secret`, `token`, `password`, or `api_key` with `"[REDACTED]"`. This guard runs before the manifest is written to disk.

### Foundation models — hard registry block

`app/modeling/foundation/` marks TabFM-derived model contexts with `deployable=False` and `research_only=True` in their metadata. The registry promotion flow (`app/registry/`) checks this flag and refuses to set a Champion alias on such artifacts.

### Security — trusted model loading

`app/security/trusted_artifacts.py` loads `.pkl` files only after verifying a SHA-256 checksum sidecar. The sidecar is written by the save step. Pickle loading without a valid sidecar is refused, limiting the blast radius of a modified artifact.

### Concurrency — one active training job at a time

The background job service (`app/background_jobs.py`) enforces a single active job. A second submit while one is running either queues or raises, depending on configuration. This is documented in the README known limitations.

## Error handling

- CLI commands catch all exceptions via `_cli_error`, which calls `app/errors.py:log_exception` (structured JSON log to stderr) and `app/security/masking.py:safe_error_message` (strips internal paths and keys from user-facing output) before `sys.exit(1)`.
- Observability telemetry (`app/observability/`) swallows its own exceptions; errors in metrics hooks or tracing must never surface to callers.
- Streamlit page rendering is caught at `app/main.py` with a top-level try/except that logs and displays a safe error string.

## Persistence paths (defaults)

| What | Default path | Format |
|---|---|---|
| App settings | `~/.autotabml/settings.json` | JSON (Pydantic-serialized `AppSettings`) |
| App metadata DB | `artifacts/autotabml.db` (configurable) | SQLite 3 |
| MLflow tracking DB | `artifacts/mlflow/mlflow.db` | SQLite 3 (MLflow schema) |
| MLflow artifacts | `artifacts/mlflow/` | MLflow artifact store |
| Saved models | `artifacts/models/` | `.pkl` + `.json` metadata sidecar + `.sha256` checksum |
| Provenance manifests | Adjacent to saved model | JSON (`ProvenanceManifest.schema_version = 1`) |
| Drift baselines | `artifacts/models/` or user-specified | JSON (`DriftBaseline`) |

## Codec notes

- All JSON files are written via `pydantic` `model_dump_json` or `json.dumps(default=str)`.
- Datetime values in SQLite are stored as ISO-8601 strings (UTC-naive for internal use; `app/provenance.py` uses timezone-aware UTC).
- DataFrame CSV exports from prediction use formula-injection-safe escaping (see `app/security/`).

## Known limitations visible in code

| Constraint | Location |
|---|---|
| PyCaret requires Python < 3.13 | `pyproject.toml` extra `experiment`; PyCaret is not in `requires` |
| TabFM conflicts with ydata-profiling | `[tool.uv] conflicts` in `pyproject.toml` |
| TabFM Python ≥ 3.11 | `pyproject.toml` extra `tabfm` marker |
| mypy errors suppressed for ingestion / profiling / notebooks | `pyproject.toml [tool.mypy.overrides]` |
| Bandit B101 (assert) globally skipped | `pyproject.toml [tool.bandit] skips` |
| Pickle trusted-artifact note | `pyproject.toml [tool.bandit]` comment; reviewed in `docs/security.md` (not present in this tree) |
