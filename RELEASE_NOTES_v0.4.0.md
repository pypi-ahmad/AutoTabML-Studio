# Release Notes — v0.4.0

> **Release date:** 2026-08-14
>
> **Previous release:** v0.3.0 (2026-08-14)
>
> **Compatibility:** additive upgrade; no data migration or public API removal.

AutoTabML Studio 0.4.0 adds local, consent-gated Google foundation models and
moves the Streamlit workbench to port 8561 while preserving the existing AutoML,
prediction, tracking, registry, and deployment workflows.

## Highlights

### Google TabFM 1.0

- Evaluate mixed-type tabular classification and regression with a pinned model
  revision.
- Infer the task, report holdout metrics, and optionally save checksum-backed
  prediction contexts.
- Require explicit acceptance of the non-commercial weights license and block
  saved contexts from registry and deployment export.

### Google TimesFM 2.5

- Forecast single or grouped time series with point estimates and q10–q90
  uncertainty.
- Handle frequency selection, missing values, context limits, and optional
  final-horizon backtesting.
- Store local forecast artifacts while sending only aggregate metrics to MLflow.

### Runtime and delivery

- Serve Streamlit, Docker, Compose, health checks, and screenshot tooling on
  `http://localhost:8561`.
- Validate TabFM and profiling in separate CI environments because their upstream
  `typeguard` ranges do not overlap.
- Keep Docker and `make install` release-safe by installing the broad compatible
  extra set and leaving TabFM to its dedicated environment.
- Keep generated codebase-memory commit hashes out of detect-secrets false
  positives while retaining the separate Gitleaks repository scan.

## Verification

- 729 selected unit tests; 30 optional tests deselected in the standard environment.
- Ruff lint and format checks.
- CI coverage gate at 65% or higher.
- Public release metadata, wheel/sdist build, and Twine metadata validation.
- Locked dependency resolution through `uv.lock`.

## Upgrade

```bash
git pull
uv sync --locked --all-groups
uv run autotabml --version   # 0.4.0
uv run autotabml doctor
```

Install only the optional extras you use. Keep `tabfm` and `profiling` in separate
environments. Existing SQLite, MLflow, settings, and artifact layouts remain
compatible.
