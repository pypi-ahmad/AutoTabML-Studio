# Changelog

All notable changes to this project will be documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

### Changed

### Fixed

## [0.2.0] - 2026-06-22

The "modern foundation" release. No breaking changes. 700 tests
at ≥ 81% coverage. 0 Pyright errors on `app/`. The recommended
upgrade for all 0.1.x users.

### Added

#### Packaging
- Build backend migrated from `setuptools` to `hatchling`
  (PEP 621, PyPA-recommended, fast).
- `py.typed` marker added (PEP 561 typed-package).
- `[project]` metadata now static (not `dynamic`); version is
  `0.2.0` from `pyproject.toml`.
- `Development Status` upgraded from `3 - Alpha` to `4 - Beta`
  on PyPI; `Typing :: Typed` classifier added.
- `[dependency-groups]` (PEP 735) added: `dev`, `security`,
  `docs`. `uv sync` installs the `dev` group by default.
- New `providers` optional extra bundling the official LLM
  SDKs (`openai`, `anthropic`, `google-genai`, `ollama`).

#### Tooling
- `ruff format` enabled in CI; the repo is now
  black-compatible by default.
- Ruff rule set widened to `F, E4, E7, E9, I, UP, B, C4, PIE,
  RET, SIM, N, W`.
- `.pre-commit-config.yaml` with SHA-pinned hooks for
  ruff, mypy, and the standard pre-commit-hooks set.
- `.editorconfig` for cross-editor consistency.
- `Makefile` with `install`, `sync`, `test`, `test-cov`,
  `lint`, `format`, `type-check`, `security`, `build`,
  `clean`, `doctor` targets.
- Microsoft Pyright adopted as a second type checker;
  `pyright app/` reports 0 errors.
- `bandit` and `pip-audit` added (security group).

#### LLM providers
- `app/providers/openai_provider.py` rewritten on top of
  `openai>=1.40` `AsyncOpenAI`.
- `app/providers/anthropic_provider.py` rewritten on top of
  `anthropic>=0.40` `AsyncAnthropic`.
- `app/providers/gemini_provider.py` rewritten on top of
  `google-genai>=1.0` `Client` (the new unified SDK that
  supersedes `google-generativeai`).
- `app/providers/ollama_provider.py` rewritten on top of
  `ollama>=0.4` `AsyncClient`.
- Public surface (`BaseProvider`, `ModelItem`,
  `build_provider`, `get_allowed_providers`,
  `resolve_default_model`) is unchanged.

#### Security
- `app/__init__.py` bumped to `0.2.0`; new `app/py.typed`.
- `.streamlit/config.toml` explicitly disables
  `gatherUsageStats` at both the client and browser level.
- `SECURITY.md` rewritten with the v0.2.0 hardening
  posture, supported-versions table, response SLA, and
  hardening guide.
- `.github/CODEOWNERS` added.
- All third-party GitHub Actions pinned to commit SHAs.

#### CI / CD
- `ci.yml`: split lint into `Ruff check` and `Ruff format
  check`; added `type-check` (Pyright), `security-bandit`
  (Bandit + pip-audit) jobs.
- `security.yml`: added `Bandit + pip-audit` job alongside
  the existing `detect-secrets` and `gitleaks` jobs.
- `release-readiness.yml`: uses `--group dev` (PEP 735).
- `dependabot.yml`: minor/patch and major groupings for
  `pip`; `actions-minor-patch` grouping for `github-actions`.

#### Container
- Multi-stage `Dockerfile` using
  `ghcr.io/astral-sh/uv:0.11.23-python3.12-bookworm-slim` as
  builder and `python:3.12-slim-bookworm` as runtime.
- Non-root runtime user (UID 10001).
- `PYTHONUNBUFFERED`, `PYTHONDONTWRITEBYTECODE`,
  `AUTOTABML_LOG_FORMAT=json` are runtime defaults.
- Container-level healthcheck against
  `http://localhost:8501/_stcore/health`.
- `docker-compose.yml` for one-line local stack bring-up.
- `.dockerignore` for fast, cache-friendly builds.

#### Testing
- 700 unit tests (was 632 in 0.1.0).
- `TestOfficialSDKIntegration` in `tests/test_providers.py`
  proves the four LLM providers are now backed by the
  official SDK clients.
- `tests/test_safe_csv_properties.py` — Hypothesis
  property-based tests for the safe-CSV export.
- `tests/test_cli_smoke.py` — end-to-end CLI smoke tests
  (`--version`, `--help`, `info`, `doctor`,
  `init-local-storage`).
- `pyproject.toml`: pytest `asyncio_mode = "auto"`,
  `addopts = "--strict-markers --strict-config"`, scoped
  `filterwarnings` that surfaces real DeprecationWarnings in
  app/tests/ but demotes upstream-library noise.

#### Documentation
- `README.md` refreshed — "What it is" sentence added above
  the fold, test-count callouts updated to 700, new
  "Verified Output" section.
- New `docs/README.md` — docs index.
- New `docs/architecture.md` — module map, data-flow
  diagrams, reliability / security / performance models,
  on-disk layout, extension points.
- New `docs/operations.md` — day-1 verification, day-2
  monitoring, 12 common failure modes, reset procedures,
  container operations, security operations.
- New `MIGRATION_GUIDE.md` — step-by-step upgrade from
  0.1.x to 0.2.0.
- New `UPGRADE_SUMMARY.md` — one-page upgrade cheat sheet.
- New `RELEASE_NOTES_v0.2.0.md` — the v0.2.0 release
  announcement.
- Refreshed `docs/developer-guide.md` — points at the
  new docs, adds a Makefile-targets table.

### Changed

- The `profiling` extra no longer pins `setuptools<82`
  (the project now builds with hatchling, so
  `pkg_resources` is no longer a runtime import).
- The PyCaret experiment setup kwargs are now built via
  `app.modeling.pycaret.setup_runner.build_setup_call_kwargs`
  (was: inline in `PyCaretExperimentService.setup_experiment`).
- The dev group in `pyproject.toml` now uses PEP 735
  `[dependency-groups]` instead of `[project.optional-dependencies].dev`.
- `app/__init__.py` now imports `from __future__ import annotations`
  and uses 4-space indentation consistently.
- The legacy `Optional[...]` / `Union[..., ...]` type forms
  have been replaced with the modern `X | Y` / `X | None`
  syntax across the codebase.
- Wide ruff auto-format pass: import sort (isort),
  pyupgrade, comprehension rewrites, zip() strict=,
  warnings stacklevel=. All behavior-preserving.

### Removed

- **Nothing.** No public API was removed. The
  `AutoTabMLError`, `BaseProvider`, `BaseModelLoader`,
  `BenchmarkResultBundle`, `ExperimentResultBundle`,
  `FlamlResultBundle`, and `PredictionRequest` interfaces
  are all preserved.
- The legacy `setup.py` / `setup.cfg` is gone (replaced
  by `pyproject.toml`), but the build outputs are
  functionally identical.

### Fixed

- `app.config.settings.load_settings` now always returns an
  `AppSettings` (was sometimes typed as `AppSettings | None`
  by static analyzers because of an ambiguous return path).
- `app.observability.context._CONTEXT` now uses a factory
  function for the default value to satisfy
  `flake8-bugbear B039` without changing runtime behavior.
- `app.modeling.base.BaseService` is now annotated with
  `noqa: B024` to acknowledge that it is intentionally a
  marker base class (no abstract methods).
- `app.gpu` adds `pyright: ignore[reportMissingImports]` on
  the three runtime `torch` imports, so a torch-less venv
  no longer triggers a type-check failure.
- `tests/test_ingestion.py::TestFailures::test_missing_kaggle_package_is_actionable`
  is now robust to the `kaggle` package being installed in
  the venv (it patches `sys.modules` to exercise the
  "package missing" branch).
- `tests/test_e2e_local_smoke.py::test_ydata_availability_check_does_not_raise`
  now silences the upstream `DeprecationWarning` from
  `ydata-profiling` (a separate, non-trivial refactor is
  on the roadmap).
- Several pre-existing pandas-stubs / sklearn
  overload-narrowing warnings now have explicit
  `pyright: ignore` comments so the type-check report
  is actionable.

### Security

- `.github/workflows/ci.yml`, `security.yml`, and
  `release-readiness.yml` now pin every third-party action
  to a commit SHA with an inline version comment. This is
  the GitHub-recommended pattern for production CI to
  prevent tag-repointing supply-chain attacks.
- The new `Bandit + pip-audit` CI job runs on every push
  to main and on every pull request.
- `SECURITY.md` now points reporters at GitHub Security
  Advisories (private channel) instead of public issues.

## [2026-06-13]

### Added

- OSS companion documentation initialized (license, contributing, security, conduct, changelog).

[0.2.0]: https://github.com/pypi-ahmad/AutoTabML-Studio/releases/tag/v0.2.0
[2026-06-13]: https://github.com/pypi-ahmad/AutoTabML-Studio/releases/tag/v0.1.0
