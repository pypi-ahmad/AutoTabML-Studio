# Architecture Overview

## Purpose

AutoTabML Studio is organized around local-first tabular ML workflows. The repo is intentionally split into service-oriented modules so the Streamlit pages and CLI can share the same underlying logic.

## Module Breakdown

- `app/config/`: pydantic models, runtime settings load/save, enums
- `app/ingestion/`: source routing, loaders, normalization, metadata hashing
- `app/validation/`: app-native checks, optional Great Expectations integration, validation artifacts
- `app/profiling/`: summary extraction, large-dataset selectors, optional `ydata-profiling`
- `app/modeling/benchmark/`: LazyPredict benchmark orchestration, ranking, artifacts, MLflow tracking
- `app/modeling/pycaret/`: PyCaret experiment setup, compare, tune, evaluate, finalize, persistence, MLflow tracking
- `app/prediction/`: model discovery/loading, schema validation, scoring, prediction artifacts, history
- `app/tracking/`: MLflow query wrappers, history service, comparison service, comparison artifacts
- `app/registry/`: model registration and promotion workflows on top of MLflow registry APIs
- `app/storage/`: SQLite metadata store and workflow recorders for local workspace activity
- `app/artifacts/`: canonical artifact path construction and cleanup
- `app/pages/`: Streamlit page entrypoints only
- `app/cli.py`: argparse command wiring over the same service layer

## Boundary Rules

- pages should call service-layer modules, not own business logic
- CLI should call the same services used by the UI where possible
- MLflow client access is centralized in `app/tracking/mlflow_query.py` or the dedicated tracker wrappers in benchmark/experiment modules
- local artifact path construction goes through `app/artifacts/manager.py`
- local app metadata should go through `app/storage/`

## Data Flow

Typical local flow:

1. ingestion normalizes an input source into a pandas DataFrame
2. metadata hashing captures schema/content information for lineage
3. validation, profiling, benchmark, or experiment services operate on the DataFrame
4. workflow artifacts are written locally under canonical artifact directories
5. benchmark and experiment runs optionally log to MLflow
6. local workspace metadata is recorded in the SQLite app database
7. prediction workflows reuse local saved models or MLflow-backed references

## MLflow vs App DB Responsibilities

### MLflow

MLflow is the source of truth for:

- benchmark and experiment run tracking
- run history inspection
- run comparison inputs
- model registry state
- MLflow-backed prediction model loading

### SQLite App DB

The local SQLite app database stores:

- dataset lineage metadata used by local workflows
- local job records for validation, profiling, benchmark, experiment, and prediction
- saved local model metadata

The SQLite app DB does not replace MLflow run history or registry state.

## Artifact Flow

All local workspace artifacts are written under `artifacts/` by `LocalArtifactManager`.

Canonical directories:

- `artifacts/validation`
- `artifacts/profiling`
- `artifacts/benchmark`
- `artifacts/experiments`
- `artifacts/experiments/snapshots`
- `artifacts/models`
- `artifacts/comparisons`
- `artifacts/predictions`
- `artifacts/tmp`
- `artifacts/app`
- `artifacts/mlflow`

Startup initialization also performs conservative cleanup of stale temp files and old partial artifacts.

## UI and CLI Entry Points

- Streamlit app: `app/main.py`
- CLI entrypoint: `autotabml` -> `app.cli:main`

Both surfaces share the same service modules rather than implementing duplicate business logic.
