# AutoTabML Studio Documentation

Welcome to the AutoTabML Studio documentation. This is a zero-to-hero
handbook: each document explains what the project is, why each piece
exists, how it works, and how to use or extend it.

## Where to start

| If you want to …                                      | Read                                                                   |
| ----------------------------------------------------- | ---------------------------------------------------------------------- |
| Get a 5-minute overview                                | [README.md](../README.md)                                              |
| Install and run for the first time                    | [README.md](../README.md#-quick-start) → [USAGE.md](../USAGE.md)        |
| Build a complete model and inspect the leaderboard    | [USAGE.md](../USAGE.md#core-workflow)                                   |
| Run a headless benchmark, profile, or experiment       | [USAGE.md](../USAGE.md#cli-reference)                                   |
| Upgrade from a previous version                       | [UPGRADE_SUMMARY.md](../UPGRADE_SUMMARY.md) → [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) |
| Understand the system architecture                    | [architecture.md](architecture.md)                                     |
| Operate, monitor, troubleshoot                         | [operations.md](operations.md)                                         |
| Set up the dev environment and run the test suite     | [developer-guide.md](developer-guide.md)                               |
| Report a security issue                                | [SECURITY.md](../SECURITY.md)                                          |
| Read release notes                                    | [RELEASE_NOTES_v0.2.0.md](../RELEASE_NOTES_v0.2.0.md) → [CHANGELOG.md](../CHANGELOG.md) |

## Document map

### Top-level

- [README.md](../README.md) — project overview, quick start,
  workflow, screenshots, architecture summary, CLI reference,
  configuration, observability, testing & CI, known limitations.
- [USAGE.md](../USAGE.md) — every page, every CLI command,
  every configuration knob, in detail.
- [CHANGELOG.md](../CHANGELOG.md) — chronological list of
  changes (Keep-a-Changelog format).
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) — step-by-step
  upgrade from 0.1.x to 0.2.0.
- [UPGRADE_SUMMARY.md](../UPGRADE_SUMMARY.md) — one-page
  upgrade cheat sheet.
- [RELEASE_NOTES_v0.2.0.md](../RELEASE_NOTES_v0.2.0.md) — the
  v0.2.0 release announcement.
- [SECURITY.md](../SECURITY.md) — supported versions,
  disclosure channel, response SLA, hardening guide.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — how to contribute
  (if present in the repo).

### `docs/`

- [architecture.md](architecture.md) — module map, data-flow
  diagrams, reliability/security/performance models,
  extension points.
- [operations.md](operations.md) — day-1 verification,
  day-2 monitoring, common failure modes, reset procedures,
  container operations, security operations.
- [developer-guide.md](developer-guide.md) — local setup,
  common commands, release hygiene, test strategy, extension
  points, troubleshooting.

## What is in this release?

AutoTabML Studio is a **local-first automated machine-learning
workbench** that takes you from a raw CSV to a trained,
evaluated, and deployable model — entirely on your machine.
Three AutoML engines (LazyPredict, PyCaret, FLAML), end-to-end
MLflow tracking, an MLflow-style model registry, AI-generated
summaries (OpenAI, Anthropic, Gemini, Ollama), and a CLI that
mirrors the Streamlit UI. v0.2.0 hardens the foundation:
modern packaging, modern type safety, official SDK
integrations, and a production-grade CI/CD surface.

## What is the philosophy?

Local-first. No data leaves the machine unless you explicitly
ask for it. No outbound telemetry. No default cloud account.
The same service layer powers the UI and the CLI, so results
are always reproducible. The codebase is the source of truth:
typing, testing, and documentation are first-class.

## How is the codebase organized?

- `app/` — the Python package. All business logic.
- `app/main.py` — the Streamlit entry point.
- `app/cli.py` — the CLI entry point.
- `app/pages/` — Streamlit page entry points (thin wrappers
  around the service layer).
- `app/ingestion/`, `app/validation/`, `app/profiling/`,
  `app/modeling/`, `app/prediction/`, `app/tracking/`,
  `app/registry/`, `app/observability/`, `app/storage/`,
  `app/providers/`, `app/notebooks/`, `app/security/`,
  `app/config/` — feature modules.
- `tests/` — 700 tests, ≥ 81% coverage.
- `docs/` — this handbook.
- `scripts/` — developer-facing utility scripts.
- `Dockerfile` + `docker-compose.yml` — container packaging.
- `.github/workflows/` — CI, security, release-readiness.
- `pyproject.toml` + `uv.lock` — the single source of truth
  for packaging and dependencies.

See [architecture.md](architecture.md) for the full module map
and the data-flow diagrams.
