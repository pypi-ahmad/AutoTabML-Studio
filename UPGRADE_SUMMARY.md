# UPGRADE_SUMMARY.md

> One-page upgrade cheat sheet for AutoTabML Studio. Print this
> next to your terminal.

## 0.1.x → 0.2.0

### Did anything break?

**No.** 0.2.0 is additive. The CLI, the service layer, the SQLite
schema, the MLflow store layout, the artifact directory tree,
and `~/.autotabml/settings.json` are all unchanged. Your
existing data continues to work.

### What do I need to do?

1. Update the lockfile and reinstall:

   ```bash
   uv sync --locked --all-extras
   ```

2. Verify the upgrade:

   ```bash
   uv run --no-sync autotabml --version   # → 0.2.0
   uv run --no-sync autotabml doctor      # green
   uv run --no-sync pytest tests/ -q      # 700 passed
   ```

3. If you use the LLM providers in the UI, install the new
   `providers` extra:

   ```bash
   uv sync --locked --extra providers
   ```

### What's new in 0.2.0?

| Area          | What landed                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------- |
| Packaging     | `setuptools` → `hatchling` build backend, PEP 621 metadata, PEP 735 dependency groups, PEP 561 `py.typed` marker |
| Tooling       | `ruff format` enabled, wider rule set (B/C4/PIE/RET/SIM/N/W), `ruff` + `mypy` + `pyright` + `bandit` + `pip-audit` |
| DevX          | `Makefile` with `install`/`test`/`lint`/`format`/`type-check`/`security`/`build`/`doctor` targets, `.pre-commit-config.yaml` with SHA-pinned hooks, `.editorconfig` |
| LLM providers | Migrated to official SDKs: `openai.AsyncOpenAI`, `anthropic.AsyncAnthropic`, `google-genai.Client` (new SDK), `ollama.AsyncClient` |
| Security      | `.streamlit/config.toml` explicitly disables Streamlit telemetry; `SECURITY.md` rewritten with v0.2.0 hardening posture |
| CI / CD       | All GitHub Actions pinned to commit SHAs; new `type-check`, `security-bandit` jobs; `.github/CODEOWNERS`; `dependabot.yml` adds minor/patch grouping |
| Container     | New multi-stage `Dockerfile` (uv-based) and `docker-compose.yml`; non-root runtime user; healthcheck against `/_stcore/health` |
| Testing       | 700 tests (was 632); Hypothesis property tests for safe-CSV; CLI smoke tests for `--version`, `--help`, `info`, `doctor`, `init-local-storage` |
| Documentation  | New `docs/architecture.md`, `docs/operations.md`, `MIGRATION_GUIDE.md`, `UPGRADE_SUMMARY.md`, `RELEASE_NOTES_v0.2.0.md`; refreshed `README.md`, `SECURITY.md`, `CHANGELOG.md` |

### When is the next break?

We do not plan a breaking change before 1.0. The 0.3.x line
will add features; the 1.0.0 release will mark the public API
as stable.

### Need help?

- **Quick reference** — this document.
- **Step-by-step upgrade** — `MIGRATION_GUIDE.md`.
- **Detailed changelog** — `CHANGELOG.md`.
- **Detailed release notes** — `RELEASE_NOTES_v0.2.0.md`.
- **Bugs** — open a GitHub issue.
- **Security** — use GitHub Security Advisories (see
  `SECURITY.md`).
