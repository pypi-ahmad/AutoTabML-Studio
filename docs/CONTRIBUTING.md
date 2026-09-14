# Contributing

## Prerequisites

- Python 3.10–3.13 (`uv` defaults to 3.12 via `.python-version`; use 3.11 or 3.12 for PyCaret support)
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- Git

## Local setup

```bash
git clone https://github.com/pypi-ahmad/AutoTabML-Studio.git
cd AutoTabML-Studio

# Install all dev dependencies (no ML extras by default)
uv sync --locked --group dev

# Initialize local storage (SQLite DB + artifact dirs)
uv run autotabml init-local-storage

# Verify the environment
uv run autotabml doctor
```

Install only the extras relevant to the code you are changing:

```bash
uv sync --locked --group dev --extra benchmark    # LazyPredict tests
uv sync --locked --group dev --extra experiment   # PyCaret tests (Python 3.11/3.12 only)
uv sync --locked --group dev --extra flaml        # FLAML tests
uv sync --locked --group dev --extra validation   # Great Expectations tests
uv sync --locked --group dev --extra profiling    # ydata-profiling tests
```

## Verification gates

All of the following must pass before a PR is mergeable.

```bash
# Lint (enforced in CI)
uv run ruff check app/ tests/ scripts/

# Format check
uv run ruff format --check app/ tests/ scripts/

# Lockfile consistency
uv lock --check

# Unit tests (run by default; integration tests are opt-in)
uv run pytest

# Coverage gate — must stay at or above 65%
uv run pytest --cov=app --cov-fail-under=65

# Security lint (optional locally; always runs in CI)
uv run bandit -c pyproject.toml -r app/
```

Integration tests require optional extras and are excluded by default (`-m not integration` in `pyproject.toml`):

```bash
uv run pytest -m integration
```

## Branch expectations

- Branch from `main`. The default merge target for PRs is `main`.
- One logical change per PR. Unrelated cleanup belongs in a separate PR.
- Keep the `uv.lock` committed and consistent (`uv lock --check` must pass).

## CI workflows

| Workflow | File | Triggers | What it checks |
|---|---|---|---|
| CI | `.github/workflows/ci.yml` | Push, PR | Lint · unit tests on Python 3.11 + 3.13 · coverage ≥ 65% · E2E smoke |
| Security | `.github/workflows/security.yml` | Push, PR | `detect-secrets` + `gitleaks` + `bandit` + `pip-audit` |
| Release readiness | `.github/workflows/release-readiness.yml` | Tag push | Build validation + `twine check` |

## Code style

- Line length: 120 characters (Ruff enforced).
- Import order: `ruff --select I` (isort rules).
- Docstrings: module-level docstrings on all `app/` files; function docstrings where non-obvious.
- Type annotations: encouraged but not a hard gate (mypy is advisory).
- sklearn variable naming conventions (`X_train`, `X_test`) are explicitly allowed — `N803`, `N806`, `N815` are suppressed in `pyproject.toml`.

## Security

- Never commit API keys or credentials. The CI security workflow runs `detect-secrets` and `gitleaks` on every push.
- Outbound HTTP must go through `app/security/` SSRF-safe wrappers unless it is a known-safe SDK call.
- Model loading must use the checksum-verified loader in `app/security/trusted_artifacts.py` when loading `.pkl` files produced by this application.

See [SECURITY.md](../SECURITY.md) for vulnerability disclosure.

## What this repo is not for

- Do not add cloud-first features that require an account or send data by default.
- Do not add ML extras to the base `dependencies` list — keep them in optional extras.
- Do not pin exact patch versions for extras unless a specific bug makes it necessary.
