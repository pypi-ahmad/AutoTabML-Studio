# Runbook

## Start

**Streamlit UI (Windows shortcut)**

Double-click `Launch AutoTabML Studio.cmd` in the repo root. This opens the UI at `http://localhost:8561`.

**Streamlit UI (terminal)**

```bash
uv run autotabml init-local-storage   # one-time; creates SQLite DB and artifact dirs
uv run streamlit run app/main.py
```

**CLI only**

```bash
uv run autotabml info      # verify version and paths
uv run autotabml doctor    # check runtime dependencies, DB, GPU, artifact dirs
```

**Headless / log to file**

```bash
AUTOTABML_LOG_FORMAT=json AUTOTABML_LOG_LEVEL=INFO uv run streamlit run app/main.py 2>autotabml.log
```

## Stop

Streamlit: `Ctrl+C` in the terminal, or close the terminal running the process.

Background jobs continue to run until completed or cancelled:

```bash
uv run autotabml job-list
uv run autotabml job-cancel <job-id>
```

## Logs

- All logging goes to **stderr** by default.
- Set `AUTOTABML_LOG_FORMAT=json` to get one JSON document per line.
- Each log record includes `correlation_id`, `run_id`, and `experiment_name` when emitted inside a training or prediction workflow.
- There is no dedicated log file by default; redirect stderr if you need persistence.

## Diagnostics

```bash
uv run autotabml doctor          # startup checks — CUDA, DB, artifact dirs, stale files
uv run autotabml info            # version, workspace mode, backend, artifact paths
```

`doctor` exits 0 on success and 1 if any error-severity check fails.

## Common failures

### "Metadata storage is unavailable"

The SQLite metadata database has not been initialised.

```bash
uv run autotabml init-local-storage
```

If the path is non-default, set `AUTOTABML__DATABASE__PATH` or check `~/.autotabml/settings.json`.

### "mlflow is not installed"

The `benchmark`, `experiment`, or `flaml` extras are not installed. Install the relevant extra:

```bash
uv sync --locked --group dev --extra benchmark
# or: --extra experiment --extra flaml
```

### "PyCaret requires Python < 3.13"

PyCaret is incompatible with Python 3.13. Switch to Python 3.11 or 3.12:

```bash
uv python pin 3.12
uv sync --locked --group dev --extra experiment
```

### Foundation model: "Please accept the license" / "allow_download not set"

TabFM and TimesFM require explicit opt-in before any network access. In the UI, use the Foundation Models page toggle. On the CLI, pass `--accept-tabfm-license --allow-download` (TabFM) or `--allow-download` (TimesFM).

### Foundation model checkpoint download is slow or fails

Models are fetched from Hugging Face at a pinned revision. Check your network access to `huggingface.co`. On subsequent runs the snapshot is cached by `huggingface_hub`.

### "typeguard" version conflict (TabFM + profiling)

TabFM requires `typeguard<3`; ydata-profiling requires `typeguard>=4`. Install them in separate virtual environments:

```bash
# Env 1 — with TabFM
uv sync --locked --extra tabfm
# Env 2 — with profiling
uv sync --locked --extra profiling
```

### Kaggle dataset download fails

Kaggle requires credentials. Set `KAGGLE_USERNAME` and `KAGGLE_KEY` (or place `~/.kaggle/kaggle.json`). Kaggle integration is CLI-only.

### GPU not detected

Run `uv run autotabml doctor`. If CUDA is reported unavailable, training falls back to CPU automatically. No action required unless GPU is mandatory.

### Stale lock files or partial artifacts from a crashed run

`autotabml doctor` removes stale temp files and partial artifacts automatically. Run it after an abnormal exit.

### "formula-injection-safe" CSV export writes leading apostrophes

This is intentional. Values starting with `=`, `+`, `-`, or `@` are prefixed with `'` to prevent spreadsheet formula injection. The raw scored DataFrame is also available in the returned object.

## Reset procedures

**Reset the metadata database only** (leaves MLflow and model artifacts intact):

```bash
# Delete the SQLite database, then reinitialize
del artifacts\autotabml.db       # Windows
rm artifacts/autotabml.db        # Unix
uv run autotabml init-local-storage
```

**Reset MLflow tracking** (deletes all run history, irreversible):

```bash
del artifacts\mlflow\mlflow.db   # Windows
rm artifacts/mlflow/mlflow.db    # Unix
uv run autotabml init-local-storage
```

**Reset settings** (reverts to Pydantic defaults):

```bash
del %USERPROFILE%\.autotabml\settings.json   # Windows
rm ~/.autotabml/settings.json                # Unix
```

## Container operations

A `Dockerfile` and `docker-compose.yml` are referenced in `pyproject.toml` sdist includes. Verify their presence with `ls Dockerfile docker-compose.yml` at the repo root. Container-specific run instructions are not documented here because the files were not inspected during this runbook's authoring; see those files directly.

## Unverified facts

- The exact Streamlit default port (8561 is from the Windows launcher script name pattern; the actual port in `Launch AutoTabML Studio.cmd` was not read; verify with `uv run streamlit run app/main.py --help`).
- Log file path if `Launch AutoTabML Studio.cmd` redirects stderr: not read.
- Exact Docker entry-point command: `Dockerfile` not read.

See `docs/operations.md` for additional day-2 monitoring and failure mode documentation.
