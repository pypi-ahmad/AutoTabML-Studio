# MIGRATION_GUIDE.md

> Upgrade notes between major AutoTabML Studio versions. The
> current release line is **0.2.x**.

## Upgrading from 0.1.x → 0.2.0

### TL;DR

```bash
git pull
uv sync --locked --all-extras               # re-sync the lockfile
uv run --no-sync python -m app.release_metadata  # sanity check
uv run --no-sync pytest tests/ -q             # 700 tests, 81% coverage
uv run --no-sync autotabml --version         # should print 0.2.0
```

No data migration is required. SQLite stores and MLflow stores
created under 0.1.x continue to work.

### What changed

#### 1. Build backend: setuptools → hatchling

0.1.x used `setuptools` with `setup.py`-style `find_packages`.
0.2.0 uses `hatchling` with the PEP 621 `[project]` table. There
is no user-facing impact; the sdist and wheel produced by
`uv run --no-sync python -m build` are functionally identical.

If you publish wheels, no action is required — the project name
`autotabml-studio` and the import path `app` are unchanged.

#### 2. Dependency groups (PEP 735) replace `--extra dev`

0.1.x: `uv sync --extra dev`
0.2.0: `uv sync --group dev` (or `--all-groups`)

`uv sync --locked` (no flags) now installs the default groups
(`dev`) by default. `--all-extras` continues to work for
optional feature installs.

#### 3. LLM providers migrated to official SDKs

The four LLM provider files in `app/providers/` have been
rewritten on top of the official SDKs. The public interface
(`BaseProvider`, `ModelItem`, `build_provider`,
`get_allowed_providers`, `resolve_default_model`) is
**unchanged**, so all callers continue to work.

| Provider   | 0.1.x implementation | 0.2.0 implementation          |
| ---------- | -------------------- | ------------------------------ |
| OpenAI     | hand-rolled httpx     | `openai.AsyncOpenAI`           |
| Anthropic  | hand-rolled httpx     | `anthropic.AsyncAnthropic`     |
| Gemini     | hand-rolled httpx     | `google-genai.Client` (new SDK) |
| Ollama     | hand-rolled httpx     | `ollama.AsyncClient`           |

If you are not using the LLM providers, this change is
invisible. If you are, install the new `providers` extra
explicitly:

```bash
uv sync --locked --extra providers
```

For tests and CI environments that need the LLM providers
to be present, the CI workflow already does this.

The `GeminiProvider._auth_headers` helper is gone (the SDK
embeds the API key on the client). If you were reaching into
the provider internals, see the SDK docs.

#### 4. Typed-package marker (PEP 561)

A new empty file `app/py.typed` is shipped, declaring the
project as a typed package per PEP 561. Downstream projects can
now use `pyright` / `mypy` against `app` and get full
intelligence. No action required.

#### 5. Streamlit `gatherUsageStats` is now explicitly disabled

`.streamlit/config.toml` now contains `gatherUsageStats = false`
at both the client and browser level. This makes the
"no-telemetry, local-first" claim verifiable in source.

If you have your own `.streamlit/config.toml` overrides,
merge this in.

#### 6. New CI / CD / security jobs

- `lint` is now split into `Ruff check` and `Ruff format check`
  so a format failure cannot mask a lint regression.
- New `type-check` job runs Pyright against `app/`.
- New `security-bandit` job runs Bandit and `pip-audit`.
- All third-party GitHub Actions are pinned to commit SHAs
  (security best practice). Re-pinning is straightforward:
  look up the new SHA, update both `uses:` and the inline
  version comment.

#### 7. `bandit` and `pip-audit` are now dependencies

A new `[dependency-groups] security` group adds:

- `bandit[toml]>=1.7`
- `pip-audit>=2.7`

These are used in the new `security-bandit` CI job. If you
run `make security` locally, they will be installed.

#### 8. `pyproject.toml` and `uv.lock` are the only source of truth

The legacy `setup.py` / `setup.cfg` is gone. The build is now
fully pyproject-driven. If you had a CI step that ran
`pip install -e .` it still works; `pip` reads `pyproject.toml`.

### Deprecations

- **`mcp`-on-Python 3.13**: the Colab MCP backend still requires
  the `colab` extra and a working `uvx` binary. No change
  from 0.1.x.
- **`lazypredict`**: still required for the benchmark extra.
  0.2.0 does not change the benchmark behavior.
- **`ydata-profiling`**: the upstream `import ydata_profiling`
  emits a `DeprecationWarning`. A migration to
  `fg-data-profiling` (`import data_profiling`) is on the
  roadmap. The workbench continues to call the legacy
  `ydata-profiling` API; suppress the warning via
  `filterwarnings` if it is noisy in your logs.

### Removed

Nothing. **No public API was removed in 0.2.0.** The
`AutoTabMLError`, `BaseProvider`, `BaseModelLoader`,
`BenchmarkResultBundle`, `ExperimentResultBundle`,
`FlamlResultBundle`, and `PredictionRequest` interfaces are
all preserved.

### Filesystem layout

Unchanged. The artifact directory tree, the SQLite store
location, the MLflow store location, and `~/.autotabml/`
all stay the same.

### Tests

The test count rose from 632 (0.1.x baseline) to 700 (0.2.0).
New tests cover:

- Official SDK wiring for all 4 LLM providers
  (`TestOfficialSDKIntegration`).
- Property-based tests for the safe-CSV export
  (`test_safe_csv_properties.py`).
- End-to-end CLI smoke tests
  (`test_cli_smoke.py`).

Coverage went from 81.65% to 81%+ on the same code; new
tests target previously-untested behavior.

### Security

Security posture is significantly improved. See
`SECURITY.md` for the v0.2.0 summary. The default
`settings.json` no longer contains any sensitive field;
provider keys are read from env vars or session state only.

### Things you should do after upgrading

1. **Verify your existing data** — run
   `autotabml doctor`. If it shows warnings about old
   paths, follow the migration tooltips.
2. **Reinstall** — `uv sync --locked --all-extras` to pick up
   the new lockfile.
3. **Run the test suite** — `uv run --no-sync pytest tests/ -q`.
   Expect 700 passed.
4. **Verify the LLM providers** (if you use them) — set the
   provider to one of your accounts in the Settings page and
   click *Fetch models*. The new SDK-based providers
   authenticate using the same `OPENAI_API_KEY`,
   `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and
   `OLLAMA_BASE_URL` environment variables as before.

### If something breaks

- **Provider fails with `httpx` not found** — install
  `--extra providers`. This pulls in `openai`,
  `anthropic`, `google-genai`, and `ollama` which include
  `httpx` transitively.
- **Pyright complains** — set `pyrightconfig.json` exclude
  to match your project layout, or run `pyright app/`
  (the project config already excludes the legacy
  `app/notebooks/generator.py` and Streamlit `app/pages/*`).
- **Tests fail in CI but pass locally** — make sure you ran
  `uv sync --locked` and that the lockfile is up to date
  with `uv lock --check`. CI runs the same check; a
  mismatch is the most common cause.
- **Container build fails** — verify the uv base image
  tag is available. The Dockerfile uses
  `ghcr.io/astral-sh/uv:0.11.23-python3.12-bookworm-slim`;
  if you need to upgrade uv, update the `UV_VERSION` arg
  and the base image tag.
