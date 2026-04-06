# Changelog

All notable changes to AutoTabML Studio should be recorded here.

This repository does not have a published tagged release yet. Keep upcoming release work under `Unreleased` until an actual version is cut.

## [Unreleased]

### Added

- `autotabml --version` and `autotabml info` for lightweight local release inspection.
- `CHANGELOG.md` and [docs/release-notes-template.md](docs/release-notes-template.md) for manual release packaging.
- [CONTRIBUTING.md](CONTRIBUTING.md) with the canonical local maintainer workflow.
- `python -m app.release_metadata` plus a manual/tagged release-readiness workflow to block public packaging when required license or maintainer/contact metadata is missing.
- CUDA detection, GPU-aware `doctor` output, and GPU-first PyCaret experiment defaults in the Streamlit UI and CLI.
- Apache-2.0 licensing plus public maintainer metadata for broader repository and package publication.

### Changed

- Project versioning now uses `app.__version__` as the single in-repo source of truth.
- `pyproject.toml` now declares an explicit setuptools build backend and package README metadata.
- `.env.example` now reflects the actual `AUTOTABML_*` settings keys used by the app, including local database and Ollama overrides.
- Release-facing docs now point to the same local install, CLI, Streamlit, and test commands.
- Provider fallback defaults now use vendor-verified production model IDs, including `gemini-2.5-flash` instead of a preview Gemini placeholder.
- Benchmark and experiment workflows now prefer CUDA by default where the installed runtime supports it, including LazyPredict GPU mode and GPU-ready experiment dependency bundles.

### Notes

- `0.1.0` is the current planned first public release version.
- No package registry publication or tagged public release has been performed from this repository yet.