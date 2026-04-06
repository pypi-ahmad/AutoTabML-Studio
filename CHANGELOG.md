# Changelog

All notable changes to AutoTabML Studio should be recorded here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-06

First public portfolio release. The repo is feature-complete for the demo scope:
full CLI + Streamlit UI covering dataset intake → validation → profiling →
benchmark → experiment → prediction → model registry, backed by MLflow tracking
and local artifact storage.

### Added

- **Streamlit dashboard** with 10 pages: Dashboard, Dataset Intake, Validation,
  Profiling, Benchmark, Experiment Lab, Prediction Center, History, Registry,
  Settings.
- **CLI** (`autotabml`) with 23 subcommands including `info`, `doctor`,
  `validate`, `profile`, `benchmark`, `experiment-run/tune/evaluate/save`,
  `train`, `predict-single/batch`, `history`, `compare-runs`, and full model
  registry lifecycle (`registry-list/promote/delete`).
- **Dataset workspace** — 5-tab layout supporting 8 formats (CSV, Parquet, JSON,
  Excel, Feather, ORC, SQLite, UCI), inline upload on every workflow page, sidebar
  status indicator, multi-dataset switcher.
- **Validation engine** powered by Great Expectations with 5 built-in checks and
  JSON artifact output.
- **Profiling** via ydata-profiling with HTML report generation.
- **Benchmark** via LazyPredict comparing 40+ models with GPU-first defaults.
- **Experiment Lab** via PyCaret (Python <3.13) with setup, compare, tune,
  evaluate, finalize, and save pipelines.
- **Prediction Center** — single-row and batch prediction from local or
  MLflow-backed models.
- **MLflow tracking** — automatic run logging, run history, cross-run comparison,
  and model registry with champion/challenger promotion.
- **Security** — credential masking (`app.security.masking`), secret-free codebase
  (detect-secrets + gitleaks CI), `SECURITY.md` with responsible disclosure policy.
- **CI/CD** — GitHub Actions: lint (ruff), unit tests (3.11 + 3.13), coverage
  gate (≥65%), E2E smoke tests, security scans, release-readiness check.
- **Testing** — 426 unit/integration tests, all hermetic. E2E smoke suite
  (`tests/test_e2e_local_smoke.py`) covering 13 real-dependency paths.
- **Documentation** — README with 8 real screenshots, demo guide with full CLI +
  Streamlit walkthrough, architecture overview, limitations disclosure,
  developer guide, contributing guide, issue/PR templates.
- **Colab MCP spike** — validated real MCP transport (server spawn, handshake,
  tool listing) as future integration proof-of-concept.
- GPU-aware `doctor` output and CUDA-first experiment defaults.
- Apache-2.0 license.

### Infrastructure

- `pyproject.toml` with optional dependency groups (`[dev]`, `[validation]`,
  `[profiling]`, `[benchmark]`, `[uci]`, `[colab]`).
- Version sourced from `app.__version__` (single truth).
- GitHub metadata: `[project.urls]`, issue/PR templates, `SECURITY.md`.
- Dependabot configuration for pip ecosystem.

### Notes

- PyCaret extras require Python <3.13 (tracked in `docs/limitations.md`).
- Colab MCP integration is transport-validated only; notebook execution is not
  wired end-to-end.
- This is a portfolio/demo release — not intended for unattended production use
  (see disclaimers in README and `docs/limitations.md`).