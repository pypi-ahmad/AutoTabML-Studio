# Developer Guide

## Local Setup

Use Python 3.11 or 3.12 for the full workflow including PyCaret experiments. Python 3.13 works for everything except PyCaret.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the base dev environment:

```bash
pip install -e ".[dev]"
```

Full local maintainer install:

```bash
pip install -e ".[dev,validation,profiling,benchmark,experiment,gpu,kaggle]"
```

Optional extras by workflow:

```bash
pip install -e ".[validation]"
pip install -e ".[profiling]"
pip install -e ".[benchmark]"
pip install -e ".[experiment]"
pip install -e ".[gpu]"
pip install -e ".[kaggle]"
```

The benchmark and experiment extras now include the GPU-capable boosting stack by default. Use `.[gpu]` only when you want those libraries without the rest of the benchmark or experiment workflow dependencies.

The app now prefers CUDA for both LazyPredict benchmark runs and PyCaret experiment workflows by default. When CUDA is unavailable, the default settings degrade safely back to CPU. Use `force` only when you want PyCaret setup to fail unless GPU execution is actually available.

## Provider Default Models

When a provider catalog cannot be fetched, the app falls back to verified stable model IDs:

- OpenAI: `gpt-5.4-mini`
- Anthropic: `claude-sonnet-4-6`
- Gemini: `gemini-2.5-flash`
- Ollama: no hardcoded fallback; the first local model returned by Ollama is used

These defaults were last verified against the current vendor model documentation during the April 2026 production-readiness pass.

## Common Commands

### Local runtime setup

```bash
autotabml init-local-storage
autotabml doctor
```

### Run the UI

```bash
streamlit run app/main.py
```

### Run tests

```bash
pytest
```

### Run lint

```bash
ruff check app/ tests/ scripts/
```

### Run the CI-equivalent coverage gate

```bash
pytest tests/ --cov=app --cov-report=term --cov-fail-under=65
```

### Run the E2E smoke test (requires validation, profiling, and benchmark extras)

```bash
pytest tests/test_e2e_local_smoke.py -v
```

### Run optional heavy-dependency integration checks

```bash
pytest -m integration
```

The integration marker now covers real Great Expectations execution, real `ydata-profiling` artifact generation, real LazyPredict plus MLflow benchmark smoke runs, and the clean PyCaret-unavailable path on interpreters where PyCaret is not supported.

### Canonical end-to-end CLI smoke flow

Run this sequence on a clean venv to verify the full local pipeline. Validated on Windows + Python 3.13.12 (2026-04-06). PyCaret experiment steps require Python ≤3.12.

```bash
# 0. Bootstrap
autotabml --version
autotabml info
autotabml init-local-storage
autotabml doctor

# 1. Prepare dataset (Iris via sklearn or any CSV with a target column)
python -c "
from sklearn.datasets import load_iris; import pandas as pd
df = load_iris(as_frame=True).frame
df.to_csv('artifacts/tmp/smoke_iris.csv', index=False)
df.drop(columns=['target']).head(10).to_csv('artifacts/tmp/smoke_iris_predict.csv', index=False)
"

# 2. Validate
autotabml validate artifacts/tmp/smoke_iris.csv --target target

# 3. Profile
autotabml profile artifacts/tmp/smoke_iris.csv

# 4. Benchmark (fast: 2 models only)
autotabml benchmark artifacts/tmp/smoke_iris.csv --target target \
  --task-type classification \
  --include-model DummyClassifier --include-model DecisionTreeClassifier

# 5. Experiment (requires Python ≤3.12 + pip install -e ".[experiment]")
autotabml experiment-run artifacts/tmp/smoke_iris.csv --target target \
  --task-type classification --n-select 1 --fold 3

# 6. Inspect history
autotabml history-list
autotabml history-show <RUN_ID>       # use id from benchmark output

# 7. Compare runs
autotabml compare-runs <RUN_ID_1> <RUN_ID_2>

# 8. Registry
autotabml registry-list
autotabml registry-show <MODEL_NAME>

# 9. Predict (feature-only CSV, no target column)
autotabml predict-batch artifacts/tmp/smoke_iris_predict.csv \
  --model-source mlflow_registered_model \
  --model-name <REGISTERED_MODEL_NAME> --model-version 1 \
  --task-type classification

# 10. Prediction history
autotabml predict-history
```

Expected outcomes:
- Steps 0-4, 6-10: all work on Python 3.13 with `.[dev,validation,profiling,benchmark]`
- Step 5: requires Python ≤3.12 with `.[experiment]` extra; prints clean error on 3.13
- Benchmark creates an MLflow run visible in `history-list`
- Predict uses a registered model and writes scored CSV + summary artifacts

### Discover CLI usage

```bash
autotabml --version
autotabml info
autotabml --help
```

## Release Hygiene

- update [CHANGELOG.md](../CHANGELOG.md) before cutting or announcing a version
- use [release-notes-template.md](release-notes-template.md) for manual release summaries
- keep [.env.example](../.env.example) aligned with the real settings keys
- keep notebook mode and `colab_mcp` messaging explicitly scoped to their current implemented state
- run `python -m app.release_metadata` before any public release build or tag; it verifies the committed license and public maintainer/contact metadata in `pyproject.toml`

### Public release packaging check

```bash
python -m app.release_metadata
python -m build
python -m twine check dist/*
```

The repository now exposes `.github/workflows/release-readiness.yml` for the same manual or tag-driven release gate in GitHub Actions.

## Test Strategy

The repo emphasizes local, hermetic tests:

- no live provider API calls in normal test runs
- no live MLflow server dependency in the unit suite
- `tmp_path` for filesystem isolation
- monkeypatched service boundaries for optional dependencies and network calls
- tests marked `integration` are reserved for optional heavy-dependency checks and are intended for the manual CI integration job

## Extension Points

- new ingestion sources: add loaders under `app/ingestion/`
- new validation rules: extend `app/validation/rules.py` and supporting schemas/builders as needed
- new benchmark/experiment result handling: extend the service and artifact layers before touching pages
- new prediction loaders: extend `app/prediction/loader.py`
- new page entrypoints: register them in `app/pages/registry.py`

## Local Troubleshooting

### Profiling page says `ydata-profiling` is missing

Install the profiling extra:

```bash
pip install -e ".[profiling]"
```

If the import still fails in a fresh environment, pin `setuptools<82` because `ydata-profiling` imports `pkg_resources` at runtime and `setuptools` 82 removed it.

### Benchmark or experiment flows are unavailable

Install the matching extras:

```bash
pip install -e ".[benchmark]"
pip install -e ".[experiment]"
```

Observed on this execution pass: PyCaret-backed experiment workflows did not install successfully on Python 3.13 in this environment. Use a PyCaret-compatible interpreter for experiment/save/local-saved-model flows until upstream support lands.

### MLflow pages show unavailable status

Install the workflows that require MLflow and set a local tracking URI if needed.

### The dashboard says the local metadata store is unavailable

Run:

```bash
autotabml init-local-storage
```
