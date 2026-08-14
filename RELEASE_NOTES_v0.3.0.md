# Release Notes — v0.3.0

> **Release date:** 2026-08-14
>
> **Previous release:** v0.2.0 (2026-06-22)
>
> **Release line:** 0.3.x (current)
>
> **Compatibility:** additive upgrade from 0.2.x; no data migration or public API removal.

AutoTabML Studio 0.3.0 turns the hardened 0.2 foundation into a more complete
local AutoML workflow. It adds guided background execution, stronger evaluation
and deployment outputs, practical desktop launch and model-cost tooling, and a
faster, simpler internal implementation.

## Highlights

### Guided Auto Run

- Run the recommended tabular-ML workflow from a single guided UI or CLI entry.
- Track persistent background progress, inspect completed artifacts, and cancel
  active jobs safely.
- Preserve holdout evaluation, explanations, provenance, and row-free drift
  baselines with each completed run.

### Evaluation and delivery

- Added model explanation and drift workflows around the saved holdout result.
- Added portable FastAPI, standalone CLI, and non-root Docker deployment bundles.
- Expanded local history, registry, and comparison workflows without changing
  existing artifact or metadata-store layouts.

### Desktop and AI-provider experience

- Added `Launch AutoTabML Studio.cmd` for one-file Windows launching.
- Added an Advanced Settings model-cost calculator with six supplied reference
  price tiers, exact Decimal arithmetic, and discount/tier notes.
- Kept external AI providers optional; datasets and model artifacts remain local
  unless the user explicitly configures an external integration.

### Performance and maintainability

- Reduced CSV ingestion allocations and simplified delimiter handling.
- Removed unused service abstractions, prototype scripts, parameters, and result
  transformation layers.
- Refreshed the interactive architecture map and committed code-understanding
  graphs for onboarding and maintenance.

### Security

- Bound local Streamlit and container ports to loopback by default.
- Strengthened untrusted URL ingestion, artifact-integrity enforcement, secret
  masking, and structured-log redaction.
- Retained SHA-pinned GitHub Actions and the existing security scanning pipeline.

## Verification

- 716 tests.
- Ruff clean across application, tests, and scripts.
- Pyright clean across the application release surface.
- Locked dependency resolution through `uv.lock`.

## Upgrade

```bash
git pull
uv sync --locked
uv run autotabml --version   # 0.3.0
uv run autotabml doctor
```

On current `main`, add only the optional extras you use. The newer `tabfm` and
`profiling` extras are intentionally incompatible and require separate
environments; see the root README for current install commands.

No SQLite, MLflow, settings, or artifact migration is required. Existing 0.2.x
workspaces continue to work.

Model-cost results are planning estimates based on the bundled reference table.
Provider discounts, cached-input pricing, fast modes, promotions, and
long-context tiers can change the final bill.
