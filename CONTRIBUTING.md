# Contributing to AutoTabML Studio

Welcome, and thank you for considering a contribution! AutoTabML Studio is a
community-driven project and every contribution — a bug report, a documentation fix,
a test, or a new feature — is genuinely appreciated.

This is a local-first, fully free project. No financial contributions are needed or
wanted; the best way to support the project is with code, tests, and thoughtful
issue reports.

Keep changes focused, preserve the local-first data boundary, and include tests
and documentation for user-visible behavior.

## Set up the repository

Use Python 3.11 or 3.12 for the broadest optional-dependency support.

```bash
git clone https://github.com/pypi-ahmad/AutoTabML-Studio.git
cd AutoTabML-Studio
uv sync --locked --group dev
```

Add only the workflow extras needed for your change:

```bash
uv sync --locked --group dev --extra benchmark
uv sync --locked --group dev --extra experiment
uv sync --locked --group dev --extra flaml
uv sync --locked --group dev --extra profiling
uv sync --locked --group dev --extra tabfm
uv sync --locked --group dev --extra timesfm
```

Keep `tabfm` and `profiling` in separate virtual environments. Their upstream
`typeguard` requirements are intentionally declared as incompatible.

## Make and verify changes

- Keep the diff limited to one coherent purpose.
- Add or update tests for changed behavior.
- Update README, usage, operations, security, architecture, migration, and
  changelog pages when the corresponding public contract changes.
- Never commit API keys, tokens, local datasets, or generated model artifacts.

Run the narrow tests while developing, then the repository gates before opening a
pull request:

```bash
uv run --no-sync ruff check app tests scripts
uv run --no-sync pytest tests -q
uv run --no-sync pytest tests --cov=app --cov-report=term --cov-fail-under=65 -q
uv run --no-sync mypy app --config-file=pyproject.toml
uv lock --check
```

Optional integrations may be deselected when their dependencies are not installed.
Current collection reports 759 tests, with 729 selected in the standard environment.

## Submit the change

Open a focused pull request that explains the observable change, verification run,
and any compatibility or security implications. Follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

Report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/pypi-ahmad/AutoTabML-Studio/security/advisories/new),
not a public issue. See the [security policy](SECURITY.md) for details.

## Data responsibility

AutoTabML Studio processes data locally. Never commit real datasets, API keys,
or model artifacts to the repository. See [DISCLAIMER.md](DISCLAIMER.md) for
the full data-responsibility statement.
