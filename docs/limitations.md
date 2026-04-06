# Limitations

This file documents current constraints intentionally, so demos and public repo descriptions stay truthful.

## Product Scope Limits

- no deployment endpoints or serving API
- no background job scheduler or worker system
- no drift monitoring, fairness workflows, or automated retraining pipeline

## Partially Implemented Areas

### Notebook mode

Notebook mode exists as a first-class workspace mode and page entrypoint, but it is currently a placeholder rather than a full notebook execution environment.

### `colab_mcp`

The backend choice exists in config and settings, but real remote execution is not implemented yet.

### GX Data Docs

Configuration placeholders exist, but a full Data Docs workflow is not implemented.

## Dependency Constraints

- profiling requires `ydata-profiling` and currently relies on `setuptools<82` because `ydata-profiling` still imports `pkg_resources`
- benchmark workflows require the benchmark extra
- PyCaret experiment workflows require the experiment extra
- PyCaret-backed experiment/save/local-saved-model workflows did not validate on Python 3.13 in this environment because PyCaret was not installable here
- MLflow-backed history, comparison, registry, and prediction references require MLflow support in the local environment

## Data Engine Constraints

- downstream benchmark, experiment, and prediction flows operate on pandas DataFrames
- large datasets may require sampling or future alternate data-engine support
- the repo does not currently implement a DuckDB or Polars execution path

## Demo Constraints

- public demo packaging should avoid claiming features outside the currently implemented local-first workflow
