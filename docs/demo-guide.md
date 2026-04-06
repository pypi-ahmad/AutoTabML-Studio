# Demo Guide

## Demo Objective

Show AutoTabML Studio as a practical local-first tabular ML workflow tool, not as a deployment platform.

## Recommended Dataset Characteristics

Choose a small-to-medium tabular dataset that:

- fits comfortably in local memory
- has a clear target column
- contains a mix of numeric and categorical features
- is clean enough to finish the benchmark/experiment flow quickly

Good demo shape:

- a few thousand rows
- fewer than 50 columns
- binary classification or simple regression target

## Startup Steps

```bash
pip install -e ".[dev]"
autotabml init-local-storage
autotabml doctor
streamlit run app/main.py
```

For the full demo flow described below, install the matching extras first:

```bash
pip install -e ".[dev,validation,profiling,benchmark,experiment]"
```

If you want a lighter walkthrough, keep the base dev install and focus on validation plus the placeholder-aware UI tour.

## Ideal Demo Flow

### 1. Dashboard

Show:

- startup checks
- local metadata store initialized
- where recent jobs, datasets, and saved models appear

### 2. Validation

Show:

- selecting a dataset
- running validation with a target column
- failed/warning/passed counts
- generated report paths

### 3. Profiling

Show:

- profiling summary cards
- large-dataset warnings if relevant
- HTML and JSON artifact paths

### 4. Benchmark

Show:

- choosing task type and target
- running a baseline benchmark
- leaderboard output and ranking metric
- MLflow run id when available

### 5. Experiment

Show:

- compare flow
- one follow-up action: tune, evaluate, or finalize/save
- saved model output and local artifacts

### 6. Prediction

Show:

- loading the saved local model
- single-row prediction
- batch prediction and artifact output

### 7. History / Compare / Registry

Show:

- MLflow-backed history list
- run detail view
- optional side-by-side comparison
- registry page only if MLflow registry support is available locally

## What Not To Claim

Do not claim that the project currently provides:

- production deployment or serving
- background job orchestration
- remote execution through `colab_mcp`
- full notebook execution
- monitoring, drift detection, or fairness pipelines

## Fallback Paths If A Feature Fails

- if profiling dependencies are missing, skip profiling and continue with validation + benchmark
- if MLflow is not installed, focus on validation, profiling, benchmark, experiment save, and local prediction
- if registry APIs are unavailable, show history and comparison instead of registry actions
- if PyCaret extras are not installed or the current interpreter cannot install PyCaret, focus on validation, profiling, baseline benchmark, and MLflow history

## Talking Points

- this is useful because tabular ML work is often fragmented across notebooks, scripts, and tracking tools
- local-first matters because it lowers setup friction and keeps experimentation understandable
- MLflow helps with run history, artifact references, and registry semantics without pretending the app DB should own that state
- LazyPredict is used for quick baselines; PyCaret is used for deeper local experiment workflows

## Rehearsed CLI Demo Path

This exact sequence was validated end-to-end on the workspace environment. Copy and run it in order:

```bash
# 1. Bootstrap
autotabml init-local-storage
autotabml doctor

# 2. Validate
autotabml validate artifacts/tmp/e2e_demo.csv --target approved

# 3. Profile
autotabml profile artifacts/tmp/e2e_demo.csv

# 4. Benchmark
autotabml benchmark artifacts/tmp/e2e_demo.csv --target approved --task-type classification

# 5. History
autotabml history-list
autotabml history-show <run-id-from-step-4>

# 6. Predict (requires a logged MLflow model — see step 4's MLflow run id)
autotabml predict-single --model-source mlflow_run_model --run-id <run-id> --artifact-path model --task-type classification --row-json '{"age": 35, "income": 55000.0, "credit_score": 720, "loan_amount": 15000.0}'

# 7. Registry
autotabml registry-register my-classifier --source "runs:/<run-id>/model" --run-id <run-id>
autotabml registry-promote my-classifier 1 --action candidate
autotabml registry-list

# 8. Predict from registry
autotabml predict-single --model-source mlflow_registered_model --model-name my-classifier --model-alias candidate --task-type classification --row-json '{"age": 45, "income": 62000.0, "credit_score": 750, "loan_amount": 18000.0}'
```

For the Streamlit demo, use Dataset Intake to load the same CSV, then navigate through Validation, Profiling, Benchmark, History, and Registry pages.

## Rehearsed Streamlit Click Path

Use the same stable dataset throughout: `artifacts/tmp/e2e_demo.csv`.

1. Open Dashboard and mention the local-first scope plus recent jobs.
2. Open Dataset Intake, choose Local Path, load `artifacts/tmp/e2e_demo.csv`.
3. Open Validation, choose `approved` as the target, run validation, and point to the 5/5 passed summary.
4. Open Prediction to show that local and MLflow-backed model loading live in one surface.
5. Open History to show the recorded benchmark and prediction jobs.
6. Open Registry to show the promoted `candidate` alias on the demo model.

Keep the demo under five minutes. Do not branch into notebook mode or remote execution.
