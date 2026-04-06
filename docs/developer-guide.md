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

### Run the CI-equivalent coverage gate

```bash
pytest tests/ --cov=app --cov-report=term --cov-fail-under=65
```

### Run optional heavy-dependency integration checks

```bash
pytest -m integration
```

The integration marker now covers real Great Expectations execution, real `ydata-profiling` artifact generation, real LazyPredict plus MLflow benchmark smoke runs, and the clean PyCaret-unavailable path on interpreters where PyCaret is not supported.

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
