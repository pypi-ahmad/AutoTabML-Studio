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

This exact sequence was validated end-to-end on a clean virtual environment (Python 3.13, `pip install -e ".[dev,validation,profiling,benchmark]"`). Copy and run it in order:

```bash
# 1. Bootstrap
autotabml init-local-storage
autotabml doctor
autotabml info

# 2. Validate
autotabml validate datasets/sklearn/Diabetes/diabetes.csv --target target

# 3. Profile
autotabml profile datasets/sklearn/Diabetes/diabetes.csv

# 4. Benchmark (43 models, CatBoostRegressor ranked #1)
autotabml benchmark datasets/sklearn/Diabetes/diabetes.csv \
  --target target --task-type regression --test-size 0.2 --ranking-metric r2 --top-k 5

# 5. History
autotabml history-list --limit 5
autotabml history-show <run-id-from-step-4>

# 6. Compare two benchmark runs (if you have a prior run)
autotabml compare-runs <left-run-id> <right-run-id>

# 7. Train + log a model (benchmark does not log a model artifact,
#    so train one with sklearn and log via MLflow for predict)
python -c "
import mlflow, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
mlflow.set_tracking_uri('sqlite:///artifacts/mlflow/mlflow.db')
mlflow.set_experiment('demo-walkthrough')
df = pd.read_csv('datasets/sklearn/Diabetes/diabetes.csv')
X, y = df.drop(columns=['target']), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Ridge(alpha=1.0); model.fit(X_train, y_train)
with mlflow.start_run(run_name='demo-ridge') as run:
    mlflow.log_metric('r2', model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, 'model', input_example=X_test.iloc[:1])
    print(f'Run ID: {run.info.run_id}')
"

# 8. Predict single row
autotabml predict-single \
  --model-source mlflow_run_model --run-id <run-id-from-step-7> --artifact-path model \
  --row-json '{"age":0.038,"sex":0.051,"bmi":0.062,"bp":-0.044,"s1":-0.035,"s2":-0.043,"s3":-0.002,"s4":0.002,"s5":0.020,"s6":-0.018}'

# 9. Predict batch (442 rows)
autotabml predict-batch datasets/sklearn/Diabetes/diabetes.csv \
  --model-source mlflow_run_model --run-id <run-id-from-step-7> --artifact-path model \
  --output-path artifacts/predictions/demo_scored.csv

# 10. Prediction history
autotabml predict-history --limit 5

# 11. Registry
autotabml registry-register demo-ridge \
  --source "runs:/<run-id-from-step-7>/model" --description "Ridge regression demo"
autotabml registry-promote demo-ridge 1 --action champion
autotabml registry-list
autotabml registry-show demo-ridge

# 12. Predict from registered champion
autotabml predict-single \
  --model-source mlflow_registered_model --model-name demo-ridge --model-alias champion \
  --row-json '{"age":0.038,"sex":0.051,"bmi":0.062,"bp":-0.044,"s1":-0.035,"s2":-0.043,"s3":-0.002,"s4":0.002,"s5":0.020,"s6":-0.018}'
```

On Python 3.13 the `experiment-run` command will exit with a clear message explaining PyCaret is unavailable.

For the Streamlit demo, use Dataset Intake to load the same diabetes CSV, then navigate through Validation, Profiling, Benchmark, History, and Registry pages.

## Rehearsed Streamlit Click Path

Use the same stable dataset throughout: `datasets/sklearn/Diabetes/diabetes.csv`.

1. Open Dashboard — mention local-first scope, recent jobs, active dataset banner.
2. Open Dataset Intake, choose Local Path, enter `datasets/sklearn/Diabetes/diabetes.csv`, click Load.
3. Scroll down to see the 4-metric bar, 50-row preview, schema table, and "Continue" buttons.
4. Open Validation, choose `target` as the target column, run validation, point to the 5/5 passed summary.
5. Open Profiling, generate the profile — show the summary cards and HTML artifact path.
6. Open Benchmark, set task to regression, target to `target`, run benchmark, show the leaderboard.
7. Open Prediction to show that local and MLflow-backed model loading live in one surface.
8. Open History to show the recorded benchmark and prediction jobs.
9. Open Compare to demonstrate side-by-side run comparison.
10. Open Registry to show registered models and promoted aliases.
11. Open Settings to show runtime defaults (backend, CUDA detection, artifact paths).

Keep the demo under five minutes. Do not branch into notebook mode or remote execution.
