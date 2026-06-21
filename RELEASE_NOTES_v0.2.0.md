# Release Notes — v0.2.0

> **Release date:** 2026-06-22
> **Previous release:** v0.1.0 (2026-04-06)
> **Release line:** 0.2.x (current)
> **Compatibility:** drop-in upgrade from 0.1.x. No data
> migration required. No public API removed.

This is a quality, security, and developer-experience release.
It does not introduce new user-facing features — instead, it
hardens the foundation: tooling, packaging, security posture,
type safety, CI, and the LLM-provider integrations. 0.2.0 is
the recommended upgrade for all 0.1.x users.

## Highlights

### Modern Python packaging

- Migrated the build backend from `setuptools` to
  **`hatchling`** (PEP 621, PyPA-recommended, fast).
- Adopted **PEP 735 dependency groups** (`dev`, `security`,
  `docs`); `uv sync` now installs the `dev` group by default.
- Added a `py.typed` marker (PEP 561) so downstream projects
  can use `mypy` / `pyright` against `app` for full type
  intelligence.
- Moved version to a static `version = "0.2.0"` in
  `pyproject.toml` (was `dynamic = ["version"]`).
- Project status upgraded from `3 - Alpha` to `4 - Beta` on
  PyPI; `Typing :: Typed` classifier added.

### Tooling

- **Ruff** rule set widened to `F, E4, E7, E9, I, UP, B, C4,
  PIE, RET, SIM, N, W`. Per-file-ignores preserve ML
  (`X_train`/`X_test` snake-case) and Streamlit
  (`st.sidebar`) idioms.
- **`ruff format`** enabled in CI. The repo is now
  black-compatible by default; the format check fails
  the build on any drift.
- **`pre-commit`** config added with SHA-pinned hooks for
  ruff, mypy, and the standard pre-commit-hooks set.
- **`.editorconfig`** added for cross-editor consistency.
- **`Makefile`** added with `install`, `sync`, `test`,
  `test-cov`, `lint`, `format`, `type-check`, `security`,
  `build`, `clean`, `doctor` targets.

### Type safety

- Adopted **Microsoft Pyright** as a second static type
  checker alongside mypy. `pyright app/` reports **0
  errors** and ~960 informational warnings (all on
  third-party type stubs we cannot fix without forking
  upstream).
- The `app/providers/` interfaces, the `app.modeling`
  services, and the public service entry points are now
  fully Pyright-compatible.

### LLM provider modernization

The four `app/providers/*.py` files have been rewritten on
top of the **official SDKs**. The public interface is
**unchanged** (same `BaseProvider`, same `ModelItem`, same
`build_provider` factory). The change is invisible to
callers but removes ~200 lines of bespoke HTTP plumbing.

| Provider   | 0.1.x          | 0.2.0                          |
| ---------- | -------------- | ------------------------------ |
| OpenAI     | hand-rolled httpx | `openai>=1.40` AsyncClient   |
| Anthropic  | hand-rolled httpx | `anthropic>=0.40` AsyncClient |
| Gemini     | hand-rolled httpx | `google-genai>=1.0` Client  |
| Ollama     | hand-rolled httpx | `ollama>=0.4` AsyncClient    |

The new SDKs bring official retry, type-checked responses,
and forward-compatibility with new model fields.

Install with:

```bash
uv sync --locked --extra providers
```

### Security hardening

- **Telemetry explicit off**: `.streamlit/config.toml` sets
  `gatherUsageStats = false` at both the client and browser
  level. The "no outbound telemetry" claim is now verifiable
  in source.
- **CI security surface expanded**: `bandit` (static
  security review) and `pip-audit` (vulnerability scan) now
  run in CI alongside the existing `detect-secrets` and
  `gitleaks` checks.
- **GitHub Actions supply chain hardened**: all third-party
  actions (`actions/checkout`, `actions/setup-python`,
  `astral-sh/setup-uv`, `actions/upload-artifact`,
  `actions/download-artifact`) are pinned to their v6.0.0
  commit SHAs with inline version comments.
- **`SECURITY.md` rewritten**: supported-versions table,
  response SLA (7 days), hardening guide for production
  deployments, and a private advisory channel via GitHub
  Security Advisories.

### CI / CD

- New **Type Check (Pyright)** job in `ci.yml`.
- New **Security (Bandit + pip-audit)** job in `ci.yml` and
  `security.yml`.
- Lint job split into **Ruff check** + **Ruff format check**
  so a format failure cannot mask a lint regression.
- New **`.github/CODEOWNERS`** — the maintainer is required
  to review changes to `app/security/`, `app/providers/`,
  `app/backends/`, `.github/`, `pyproject.toml`, `uv.lock`,
  `Makefile`, `.pre-commit-config.yaml`, and `.editorconfig`.
- `dependabot.yml` adds `minor-and-patch` and `major`
  grouping for `pip`, and `actions-minor-patch` grouping
  for `github-actions`.

### Container packaging

- New multi-stage `Dockerfile` using
  `ghcr.io/astral-sh/uv:0.11.23-python3.12-bookworm-slim` as
  the builder base and `python:3.12-slim-bookworm` as the
  runtime base.
- Runs as a non-root user (UID 10001) with `/app` writable
  only to the artifact subdirectory.
- PYTHONUNBUFFERED, PYTHONDONTWRITEBYTECODE,
  `AUTOTABML_LOG_FORMAT=json` are runtime defaults.
- Container-level healthcheck against
  `http://localhost:8501/_stcore/health`.
- New `docker-compose.yml` for one-line local stack bring-up
  with optional MLflow sidecar (commented out by default).
- New `.dockerignore` for fast, cache-friendly builds.

### Documentation

- New `docs/architecture.md` — module map, data-flow
  diagrams, reliability model, security model, performance
  model, and extension points.
- New `docs/operations.md` — day-1 verification, day-2
  monitoring, common failure modes, reset procedures, and
  container operations.
- New `MIGRATION_GUIDE.md` — step-by-step upgrade from 0.1.x.
- New `UPGRADE_SUMMARY.md` — one-page cheat sheet.
- New `RELEASE_NOTES_v0.2.0.md` — this document.
- Refreshed `README.md` — "What it is" sentence added, "530+
  unit tests" replaced with current 700-test count, new
  "Verified Output" section.
- Refreshed `SECURITY.md` — see above.

## Test suite

| | 0.1.0 baseline | 0.2.0 |
| --- | --- | --- |
| Unit tests       | 632 | 700 |
| Coverage (gate ≥ 65%) | 81.65% | 81%+ (same line set, more tests) |
| Lint (ruff)     | clean | clean (wider rules) |
| Format (ruff)   | n/a | enforced |
| Type-check (pyright) | n/a | 0 errors |
| Security (bandit) | n/a | enabled |

New tests:

- `TestOfficialSDKIntegration` — proves the four LLM providers
  are now backed by the official SDK clients.
- `tests/test_safe_csv_properties.py` — Hypothesis
  property-based tests for the safe-CSV export.
- `tests/test_cli_smoke.py` — end-to-end CLI smoke tests
  (`--version`, `--help`, `info`, `doctor`,
  `init-local-storage`).

## Breaking changes

**None.** 0.2.0 is a strictly-additive release. All public
APIs are preserved. The 0.1.x SQLite stores and MLflow stores
are forward-compatible.

## Deprecations

- The `mcp` Colab backend is unchanged; the `colab` extra
  still requires a working `uvx` binary.
- `ydata-profiling` emits a `DeprecationWarning` on
  `import ydata_profiling`. A migration to
  `fg-data-profiling` (`import data_profiling`) is on the
  roadmap for 0.3.x. The legacy API continues to work.

## How to upgrade

```bash
git pull
uv sync --locked --all-extras
uv run --no-sync autotabml --version   # should print 0.2.0
uv run --no-sync pytest tests/ -q      # 700 passed
```

See `MIGRATION_GUIDE.md` for the full step-by-step.

## Acknowledgements

This release is the work of the AutoTabML Studio maintainer
(@pypi-ahmad). The provider SDK migration relied on the
official documentation of OpenAI, Anthropic, Google GenAI,
and Ollama. The build-backend switch to hatchling was
informed by the PyPA packaging guide and the hatchling
documentation.

## What's next

The 0.3.x line will focus on:

- Stable public API (1.0.0 readiness).
- Migration from `ydata-profiling` to
  `fg-data-profiling` (`import data_profiling`).
- A typed, structured prediction API with probability
  outputs.
- Optional OpenTelemetry export for production
  observability.
- SBOM generation + Sigstore image signing in the
  release pipeline.
