# Contributing

AutoTabML Studio is currently maintained as a local-first Python project. Keep changes aligned with the implemented local workflow and avoid expanding public-facing claims beyond the current product scope.

## Local Setup

Create and activate a virtual environment, then install the editable development environment:

```bash
python -m venv .venv
pip install -e ".[dev]"
```

If you want the full local workflow available during development, install the workflow extras too:

```bash
pip install -e ".[dev,validation,profiling,benchmark,experiment,kaggle]"
```

## Canonical Local Commands

```bash
autotabml --version
autotabml info
autotabml init-local-storage
autotabml doctor
streamlit run app/main.py
autotabml --help
pytest -q
```

## Release Hygiene

- update [CHANGELOG.md](CHANGELOG.md) for user-visible or maintainer-visible changes
- keep [.env.example](.env.example) aligned with the real settings model
- keep [README.md](README.md) and [docs/developer-guide.md](docs/developer-guide.md) aligned with actual entrypoints and local commands
- do not present notebook mode or `colab_mcp` as finished capabilities (MCP transport is validated; browser-connected execution is not yet CI-testable)
- if public owner/contact metadata or license terms are still undecided, leave them unset instead of inventing placeholders
- run `python -m app.release_metadata` before any public release or tag meant for distribution

## Optional Packaging Smoke Check

After installing the dev environment, you can build local distribution artifacts with:

```bash
python -m build
```

This repository does not currently automate release publishing. Treat build and release-note generation as manual maintainer steps.

Public release readiness is now guarded by `.github/workflows/release-readiness.yml`, which runs the metadata check, build, and `twine check` on manual dispatch and `v*` tags.