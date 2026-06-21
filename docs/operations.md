# Operations Guide

> **Audience:** operators, SREs, and developers running AutoTabML
> Studio on a workstation, a server, or in a container.

This guide covers day-2 operations: how to keep the workbench
running, where to look when it is not, and how to recover from
common failure modes.

## Local runtime layout

When you run `autotabml init-local-storage` (or simply launch
the Streamlit UI for the first time), the project creates the
following on-disk layout:

```
~/.autotabml/
└── settings.json              # non-secret runtime preferences

<artifacts-root>/
├── app/                       # local app metadata
│   └── app_metadata.sqlite3   # run history, saved models, projects
├── validation/                # validation artifacts
├── profiling/                 # profiling artifacts
├── benchmark/                 # benchmark artifacts
├── experiments/               # experiment artifacts
│   └── snapshots/             # PyCaret experiment snapshots
├── models/                    # local model artifacts
│   └── <name>.sha256          # SHA256 sidecars
├── flaml/                     # FLAML artifacts
├── comparisons/               # comparison artifacts
├── predictions/
│   ├── history.jsonl          # append-only prediction log
│   └── <run>/                 # per-run output CSVs
├── mlruns/                    # MLflow local runs (if using file store)
├── mlflow/                    # MLflow SQLite store
└── tmp/                       # partial / failed artifacts
```

`<artifacts-root>` defaults to `./artifacts` (relative to the
working directory). Override with `AUTOTABML_ARTIFACTS__ROOT_DIR`.

## Day-1 verification

After install, run:

```bash
uv sync --locked --all-extras
uv run --no-sync autotabml init-local-storage
uv run --no-sync autotabml doctor
```

`doctor` reports on:

- **Python version** — must be 3.10 – 3.13.
- **Artifact dirs** — must all be writable.
- **Metadata store** — SQLite at the configured path.
- **MLflow URI** — must be reachable when set.
- **CUDA** — reports whether a CUDA-capable GPU is reachable.
- **Ollama** — when the Ollama provider is selected, hits
  `GET /api/tags` to confirm reachability.
- **Colab MCP** — when the Colab backend is selected, checks
  for `uvx` and the `mcp` SDK.

## Day-2 monitoring

The workbench has no built-in remote telemetry. To monitor it:

- **Logs** — the workbench emits structured JSON to stderr when
  `AUTOTABML_LOG_FORMAT=json`. Pipe to your log forwarder
  (journald, vector, fluentbit).
- **MLflow UI** — start the MLflow UI against the local
  SQLite store to inspect runs:
  ```bash
  uv run --no-sync mlflow ui \
      --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db
  ```
- **Local SQLite** — query the metadata store directly:
  ```bash
  sqlite3 artifacts/app/app_metadata.sqlite3 "SELECT * FROM jobs LIMIT 20"
  ```
- **Healthcheck endpoint** — the Streamlit container exposes
  `/_stcore/health` for liveness probes (the Dockerfile's
  `HEALTHCHECK` uses this).

## Common failure modes

### "autotabml doctor reports the metadata store is unavailable"

Check the SQLite path:

```bash
ls -l artifacts/app/app_metadata.sqlite3
sqlite3 artifacts/app/app_metadata.sqlite3 "PRAGMA integrity_check;"
```

If the file is corrupt, the simplest recovery is:

```bash
# 1. Stop any running autotabml processes
pkill -f "streamlit run app/main.py" || true

# 2. Move the corrupt database aside
mv artifacts/app/app_metadata.sqlite3 \
   artifacts/app/app_metadata.sqlite3.corrupt-$(date +%s)

# 3. Re-initialize
autotabml init-local-storage
```

This loses local run history but recovers the workspace.
MLflow runs are in a separate store (`artifacts/mlflow/`) and
are not affected.

### "Validation: 'great_expectations' is not installed"

The validation extra is optional. To enable Great Expectations
checks:

```bash
uv sync --locked --extra validation
```

App-native validation rules still run without Great Expectations.

### "Benchmark fails: 'lazypredict' is not installed"

```bash
uv sync --locked --extra benchmark
```

### "FLAML AutoML is unavailable"

```bash
uv sync --locked --extra flaml
```

### "Profiling page says 'ydata-profiling' is missing"

Install the profiling extra. On Python 3.12+ also pin
`setuptools<82` if `pkg_resources` import errors appear:

```bash
uv sync --locked --extra profiling
```

The upstream `ydata-profiling` package emits a
`DeprecationWarning` on import; a migration to
`fg-data-profiling` (`import data_profiling`) is on the
roadmap.

### "PyCaret is unavailable on Python 3.13"

PyCaret does not yet support Python 3.13. Use 3.11 or 3.12 for
PyCaret-backed experiments. The benchmark and FLAML paths work
on 3.13.

### "Saved model fails to load: 'artifact is missing the trusted source marker'"

The trusted source marker (`trusted_source:
autotabml_trusted_local_model_v1`) is added when a model is
saved through `app.modeling.pycaret.persistence.save_finalized_model`
or `app.modeling.flaml.service.save_best_model`. Re-save the
model from within the workbench; manually-placed pickle files
will be rejected.

### "Checksum mismatch for '<model>.pkl'"

The model file or its `.sha256` sidecar has been modified or
corrupted since the model was saved. Compare the expected
hash from the model metadata (`autotabml history-show
<run_id>`) against:

```bash
sha256sum artifacts/models/<name>.pkl
sha256sum artifacts/models/<name>.pkl.sha256
```

If they don't match, the artifact is no longer trustworthy;
re-train or re-import from a known-good source.

### "MLflow UI shows no runs"

The tracking URI and the registry URI must point to the same
store. The default is `sqlite:///artifacts/mlflow/mlflow.db`.
Override consistently with:

```bash
export AUTOTABML_MLFLOW__TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db
export AUTOTABML_MLFLOW__REGISTRY_URI=sqlite:///artifacts/mlflow/mlflow.db
```

### "Container healthcheck fails"

Check the streamlit health endpoint:

```bash
curl -fsS http://localhost:8501/_stcore/health
# expected: {"status": "ok"}
```

If it fails, check the container logs:

```bash
docker logs autotabml-studio --tail 200
```

Common causes: the artifact volume is not mounted, the
config dir is read-only, or a port is already in use.

## Backup strategy

What to back up:

1. **`~/.autotabml/settings.json`** — non-secret preferences.
2. **`<artifacts-root>/app/app_metadata.sqlite3`** — local
   run history. Backup with `sqlite3 .backup`.
3. **`<artifacts-root>/mlflow/mlflow.db`** — MLflow model
   registry, run records.
4. **`<artifacts-root>/models/*.pkl`** + `*.sha256` sidecars —
   the trained models.
5. **`<artifacts-root>/predictions/history.jsonl`** —
   prediction audit log.

What you do not need to back up:

- The `tmp/` directory (transient).
- The `.venv` directory (recreate with `uv sync`).
- `__pycache__` / `*.pyc` (recreated on import).

## Reset procedures

### Reset the local metadata store only

```bash
mv artifacts/app/app_metadata.sqlite3 \
   artifacts/app/app_metadata.sqlite3.bak
autotabml init-local-storage
```

### Reset MLflow only

```bash
mv artifacts/mlflow artifacts/mlflow.bak
autotabml init-local-storage
```

### Wipe everything and start fresh

```bash
rm -rf artifacts/ ~/.autotabml/settings.json
autotabml init-local-storage
```

## Container operations

### Build the image

```bash
docker build -t autotabml-studio:0.2.0 .
```

### Run with the bundled compose file

```bash
docker compose up -d
docker compose logs -f autotabml
docker compose exec autotabml autotabml --version
```

### Run a one-shot CLI command

```bash
docker run --rm -it \
  -v $(pwd)/data:/data:ro \
  -v $(pwd)/artifacts:/app/artifacts \
  autotabml-studio:0.2.0 \
  autotabml benchmark /data/train.csv --target label
```

### Inspect the running container

```bash
docker exec -it autotabml-studio bash
ls /app/artifacts
sqlite3 /app/artifacts/app/app_metadata.sqlite3 "SELECT COUNT(*) FROM jobs"
```

## Security operations

- **Rotate provider API keys** regularly. Set them via env vars
  at startup (`OPENAI_API_KEY=...`) and never write them to
  `settings.json` — that file is for non-secrets only.
- **Drop `app.metadata.sqlite3` to a read-only mount** if the
  container runs in a hostile environment.
- **Use Sigstore / cosign** to sign the container image at
  release time so consumers can verify authenticity. The
  `release-readiness.yml` workflow is the right place to add
  this.
- **Enable the FS watcher** (`streamlit run --server.enableWatcherServing true`)
  in development but disable it in production (the Dockerfile
  uses headless mode by default).

## What to escalate

If the workbench:

- Crashes during training with an OOM
- Reports checksum mismatches on artifacts you did not modify
- Has unexpected network connections (verify with `ss -tnp`)
- Logs authentication failures for a provider you did not call

Treat as a security incident: capture logs, isolate the host,
rebuild from a known-good commit, and rotate any provider
credentials in scope.
