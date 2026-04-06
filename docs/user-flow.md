# User Flow

## Primary Product Flow

The main product story is a local-first tabular ML workflow:

1. ingest a dataset
2. validate whether it is structurally usable
3. profile the dataset to inspect shape, missingness, and high-level EDA
4. benchmark baseline models with LazyPredict
5. run deeper PyCaret experiments on promising candidates
6. finalize and save one model or save all compared models locally
7. score new rows or batches through the prediction center
8. inspect run history, compare runs, and optionally register models in MLflow

## Page-Level Flow

### Dashboard

- check startup status
- confirm local metadata store is initialized
- review recent datasets, jobs, and saved local models

### Validation

- select a loaded dataset
- optionally set target, required columns, uniqueness checks, and leakage heuristics
- run validation
- inspect summary counts and generated artifacts

### Profiling

- select a loaded dataset
- generate profiling output
- inspect summary cards and artifact paths

### Benchmark

- select dataset and target
- choose classification or regression path
- run a baseline comparison
- inspect leaderboard, shortlist, artifacts, and MLflow run id if present

### Experiment

- select dataset and target
- run compare models
- optionally tune, evaluate, finalize/save a chosen model, or auto-save all compared models
- save local artifacts and model metadata

### Prediction

- load a saved local model or MLflow-backed model reference
- choose from discovered saved local models in the dropdown after experiment save/auto-save
- validate input schema compatibility
- run single-row or batch scoring
- inspect artifacts and local prediction history

### History / Compare / Registry

- browse MLflow-tracked benchmark and experiment runs
- compare two runs side by side
- inspect or promote registered models when MLflow registry APIs are available
