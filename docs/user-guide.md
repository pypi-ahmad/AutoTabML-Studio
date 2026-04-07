# AutoTabML Studio — User Guide

> **AutoTabML Studio** is a local-first, end-to-end automated machine learning workbench for tabular data.
> It takes you from raw CSV to a production-ready model — no cloud account required.

---

## Table of Contents

1. [Getting Started](#getting-started)
   - [System Requirements](#system-requirements)
   - [Installation](#installation)
   - [First Launch](#first-launch)
2. [The Workflow at a Glance](#the-workflow-at-a-glance)
3. [Loading Your Data](#loading-your-data)
   - [Upload a File](#upload-a-file)
   - [Enter a Local Path](#enter-a-local-path)
   - [Paste a Web URL](#paste-a-web-url)
   - [Browse the UCI Dataset Library](#browse-the-uci-dataset-library)
4. [Check Data Quality (Validation)](#check-data-quality)
5. [Explore Your Data (Profiling)](#explore-your-data)
6. [Find the Best Algorithm (Benchmark)](#find-the-best-algorithm)
7. [Train & Tune a Model](#train--tune-a-model)
8. [Make Predictions](#make-predictions)
9. [Test a Model's Accuracy](#test-a-models-accuracy)
10. [Browse Saved Models](#browse-saved-models)
11. [Review Run History](#review-run-history)
12. [Compare Algorithms](#compare-algorithms)
13. [Version & Promote Models (Registry)](#version--promote-models)
14. [Auto-Generated Notebooks](#auto-generated-notebooks)
15. [Settings & Configuration](#settings--configuration)
    - [Where to Run (Execution)](#where-to-run)
    - [GPU Acceleration](#gpu-acceleration)
    - [AI Provider](#ai-provider)
    - [Auto-Generated Descriptions](#auto-generated-descriptions)
16. [Using the Command Line (CLI)](#using-the-command-line)
17. [Where Things Are Stored](#where-things-are-stored)
18. [Glossary](#glossary)

---

## Getting Started

### System Requirements

| Requirement   | Details                                   |
| ------------- | ----------------------------------------- |
| Python        | 3.10, 3.11, 3.12, or 3.13                |
| OS            | Windows, macOS, or Linux                  |
| RAM           | 4 GB minimum, 8 GB+ recommended          |
| Disk          | ~1 GB for dependencies + space for models |
| GPU (optional)| NVIDIA GPU with CUDA for faster training  |

### Installation

**Step 1 — Create a virtual environment:**

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

**Step 2 — Install the app:**

```bash
# Minimal install (validation + benchmarking + experiments)
pip install -e ".[dev,validation,profiling,benchmark,experiment]"
```

Or install only what you need:

| Feature          | Install Command                        |
| ---------------- | -------------------------------------- |
| Data validation  | `pip install -e ".[validation]"`       |
| Data profiling   | `pip install -e ".[profiling]"`        |
| Benchmarking     | `pip install -e ".[benchmark]"`        |
| Experiments      | `pip install -e ".[experiment]"`       |
| GPU acceleration | `pip install -e ".[gpu]"`              |
| Kaggle datasets  | `pip install -e ".[kaggle]"`           |

**Step 3 — Set up the workspace:**

```bash
autotabml init-local-storage
autotabml doctor
```

`init-local-storage` creates a local database to track your work.
`doctor` checks that your environment is set up correctly.

### First Launch

```bash
streamlit run app/main.py
```

Your browser will open to the **Home** page. From there, load a dataset and start exploring.

---

## The Workflow at a Glance

AutoTabML Studio follows a natural data-science workflow. You don't have to follow every step — jump to whichever stage you need.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Load    │ ──▶│  Check   │ ──▶│ Explore  │ ──▶│  Find    │ ──▶│ Train &  │
│  Data    │    │ Quality  │    │  Data    │    │  Best    │    │  Tune    │
│          │    │          │    │          │    │ Algorithm│    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                     │
                                                                     ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Version  │ ◀──│ Compare  │ ◀──│  Review  │ ◀──│  Test    │ ◀──│ Predict  │
│ & Deploy │    │ Results  │    │ History  │    │  Model   │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

**Quick-start path:** Load Data → Find Best Algorithm → Train & Tune → Predict

---

## Loading Your Data

Navigate to **Dataset Intake** (🧾) in the sidebar.

### Upload a File

1. Click the **Upload** tab.
2. Drag and drop a CSV or Excel file (or click to browse).
3. Optionally give the dataset a friendly name.
4. Click **Load Uploaded Dataset**.

**Supported file types:** `.csv`, `.tsv`, `.txt`, `.data`, `.xlsx`, `.xls`, `.xlsm`, `.xlsb`

### Enter a Local Path

1. Click the **Local Path** tab.
2. Type or paste the full file path (e.g. `C:\Data\sales.csv`).
3. Click **Load Local Path**.

### Paste a Web URL

1. Click the **Web URL** tab.
2. Paste a direct link to a CSV or Excel file.
3. Click **Load URL**.

### Browse the UCI Dataset Library

The [UCI Machine Learning Repository](https://archive.ics.uci.edu) hosts thousands of free, public datasets — perfect for learning and testing.

1. Click the **UCI Dataset Library** tab.
2. Search by keyword (e.g. "iris", "heart", "wine") or browse by subject area.
3. Select a dataset from the results and click **Use Selected Dataset**.
4. Click **Load UCI Dataset**.

> **Tip:** You can load multiple datasets in one session and switch between them at any time using the sidebar dataset picker.

---

## Check Data Quality

Navigate to **Validation** (✅).

This page scans your data for common problems before you start modelling:

| What it checks           | Why it matters                                              |
| ------------------------ | ----------------------------------------------------------- |
| Missing values           | Too many blanks can mislead a model                         |
| Duplicate rows           | Duplicates inflate results and waste training time          |
| Data leakage             | Columns that accidentally reveal the answer ruin a model    |
| Minimum row count        | Too few rows won't train a reliable model                   |
| Column uniqueness        | Are IDs truly unique? Are there unexpected duplicates?      |

**How to use:**

1. Select your **target column** — the column you want to predict.
2. Adjust thresholds if needed (most defaults work well).
3. Click **Run Validation**.
4. Review the report — green means good, yellow means warning, red means fix it.

> **Tip:** Expand "Advanced options" for fine-grained control over null thresholds and leakage detection.

---

## Explore Your Data

Navigate to **Exploratory Data Analysis** (📊).

This generates a rich visual report about your data:

- **Distributions** — see how values are spread across each column
- **Correlations** — discover which columns are related to each other
- **Missing values** — a heatmap showing where data is missing
- **Statistics** — min, max, mean, standard deviation for every column

**How to use:**

1. Click **Generate Profile**.
2. Wait for the report to render (large datasets may take a moment).
3. Browse the interactive report.

> **Note:** For datasets larger than 50,000 rows, the app automatically uses a compact mode or analyses a random sample to keep things fast.

---

## Find the Best Algorithm

Navigate to **Benchmark** (🏁).

This is the fastest way to see which algorithm works best on your data. Behind the scenes, it trains dozens of models and ranks them by performance.

**How to use:**

1. Select the **task type**:
   - **Classification** — predicting a category (e.g. spam / not spam, customer segment)
   - **Regression** — predicting a number (e.g. house price, temperature)
2. Pick the **target column** — the column you want to predict.
3. Set the **test size** — what percentage of your data to hold out for testing (default: 20%).
4. Choose how many **top models** to display (default: 5).
5. Click **Run Benchmark**.

**Reading the results:**

- Models are ranked from best to worst.
- The top row is the algorithm that performed best on your data.
- Key metrics are shown for each model (accuracy, R², etc.).
- You can **save any model** directly from the leaderboard.

> **Tip:** If you have a GPU, the app will detect it automatically (⚡ GPU detected) and use it to speed up training.

---

## Train & Tune a Model

Navigate to **Train & Tune** (🧪).

Once you've found a promising algorithm via Quick Benchmark, come here to fine-tune it for the best possible performance.

**How to use:**

1. Select the **task type** and **target column** (same as Benchmark).
2. Configure training options:
   - **Training data %** — how much data to use for training vs. validation (default: 70%).
   - **Cross-validation folds** — the data is split into N parts; each part takes a turn as the test set (default: 5 folds).
   - **Automatic preprocessing** — let the app handle missing values, encoding, and scaling automatically.
3. Click **Compare Models** to rank all candidates.
4. Select the best model and click **Tune** to optimise its settings.
5. Click **Finalize** to retrain on all available data.
6. Click **Save** to store the model for later use.

**Experiment tracking:**

Every experiment is automatically logged. You can choose:
- **Automatic** — full logging with zero effort
- **Manual** — log only what you choose
- **Off** — no logging

> **Tip:** Use "Columns to ignore" to exclude columns you know shouldn't be used (like IDs, names, or dates that aren't useful for prediction).

---

## Make Predictions

Navigate to **Prediction** (🔮).

Use a trained model to make predictions on new data.

**How to use:**

1. Choose **where the model is**:
   - **Saved model (on this machine)** — a model you saved from Benchmark or Experiment
   - **Model from an experiment run** — load directly from an MLflow experiment
   - **Registered production model** — a model promoted to the registry
2. Select the specific model.
3. Choose prediction mode:
   - **Single row** — enter values manually (great for quick tests)
   - **Batch** — upload a CSV/Excel file to score many rows at once
4. Click **Predict**.

**Results include:**
- The predicted value or category for each row
- Confidence/probability scores (when available)
- A downloadable results file for batch predictions

---

## Test a Model's Accuracy

Navigate to **Test & Evaluate** (📊) — available as a tab inside the **Predictions** page.

This lets you evaluate how well a saved model performs on a test dataset you provide.

**How to use:**

1. Select a model (🔬 = from Experiment, 🏁 = from Benchmark).
2. Upload a test dataset (CSV or Excel with the same columns the model expects).
3. Click **Run Test**.
4. Review the metrics:
   - **Classification:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix
   - **Regression:** R², RMSE (Root Mean Squared Error), MAE (Mean Absolute Error)

---

## Browse Saved Models

Navigate to **Models** (🗂️).

See every model you've trained in one place:

- **🔬 Train & Tune Models** — trained and tuned via Train & Tune
- **🏁 Benchmark Models** — saved from the benchmark leaderboard
- **📦 Registry Models** — registered for version control and deployment

Each model card shows:
- Task type (Classification or Regression)
- Target column and number of features
- File location on disk
- Dataset fingerprint (a unique ID for the exact data it was trained on)

> **Tip:** From here, you can jump to Predictions to put any model to work.

---

## Review History

Navigate to **History** (📋).

Every time you run a workflow — validation, profiling, benchmark, experiment, or prediction — it's recorded here.

**Filtering options:**
- **Workflow type** — show only benchmarks, experiments, etc.
- **Dataset name** — find runs for a specific dataset
- **Result limit** — control how many results to display

Each entry shows the job type, dataset, timestamp, and status. Click to expand for full details, artifacts, and AI-generated descriptions.

---

## Compare Algorithms

Navigate to **Compare** (⚖️).

See how different algorithms performed on the same dataset, side by side.

**How to use:**

1. Pick a past benchmark or experiment run from the dropdown.
2. View the ranking table — algorithms are sorted from best to worst.
3. Compare metrics across models to understand trade-offs (e.g. speed vs. accuracy).

---

## Version & Promote Models

Navigate to **Registry** (🏷️).

The registry keeps versioned copies of your best models, so you always know which model is in production and which is being tested.

**Key concepts:**

| Term        | Meaning                                                       |
| ----------- | ------------------------------------------------------------- |
| **Champion** | The current production-ready model (the one you trust most)  |
| **Candidate**| A newer model being evaluated before it replaces the champion|
| **Archived** | An older version kept for reference but no longer active     |

**How to use:**

1. Browse registered models and their versions.
2. Click ⭐ **Promote to Champion** to mark a version as production-ready.
3. Click 🧪 **Promote to Candidate** to flag a version for testing.
4. Click 📦 **Archive** to retire an old version.

---

## Auto-Generated Notebooks

Navigate to **Notebooks** (📓).

Every job you run (benchmark, experiment, profiling, validation) gets an auto-generated Jupyter notebook — ready to download, share, or open in Google Colab.

**How to use:**

1. Select a dataset from the dropdown.
2. Browse the list of jobs for that dataset.
3. Click **Generate & Download** to create the `.ipynb` file.
4. Click **Preview** to see the notebook contents inline.
5. Click **Download** to save the file.

> **Advanced:** Expand the "Run Code in the Cloud or Locally" section to connect to Google Colab for remote execution.

---

## Settings & Configuration

Navigate to **Settings** (⚙️).

### Where to Run

Choose where the app runs computations:

| Option                  | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| **Local (this machine)**| Everything runs on your computer — simple and private |
| **Cloud (Google Colab)**| Heavy computations run on Google's free cloud GPUs    |

### GPU Acceleration

If your machine has an NVIDIA GPU, the app detects it automatically.

| GPU Mode                          | Description                                    |
| --------------------------------- | ---------------------------------------------- |
| **Use GPU when available**        | Recommended — uses GPU if detected, CPU if not |
| **CPU only**                      | Always use the processor, never the GPU        |
| **Require GPU (stop if missing)** | Fail the job if no GPU is found                |

### AI Provider

Choose which AI service powers descriptions and smart features:

| Provider    | Requirements                        |
| ----------- | ----------------------------------- |
| **OpenAI**  | API key from openai.com             |
| **Anthropic** | API key from anthropic.com       |
| **Gemini**  | API key from Google AI Studio       |
| **Ollama**  | Free, runs locally — no key needed  |

Enter your API key in the **API Keys** section. Keys are stored in memory only — never written to disk.

### Auto-Generated Descriptions

Enable automatic plain-English summaries for every workflow run:

1. Toggle **Enable automatic run descriptions** on.
2. Optionally toggle **Use AI for richer descriptions** for more insightful, narrative summaries (requires an AI provider and API key).

---

## Using the Command Line

Every workflow available in the UI can also be run from the terminal.

### Quick Reference

```bash
# System
autotabml --version                    # Show version
autotabml doctor                       # Check environment health
autotabml init-local-storage           # Set up local database

# Data
autotabml uci-list                     # Browse UCI datasets
autotabml uci-list --search "iris"     # Search by keyword

# Workflows (replace <dataset> with a file path, URL, or uci:<id>)
autotabml validate <dataset> --target <column>
autotabml profile <dataset>
autotabml benchmark <dataset> --target <column>
autotabml experiment-run <dataset> --target <column>

# Predictions
autotabml predict-single --model-source <path> --row-json '{"col1": 5}'
autotabml predict-batch <dataset> --model-source <path>

# History & Comparison
autotabml history-list
autotabml history-show <run_id>
autotabml compare-runs <run_id_1> <run_id_2>

# Model Registry
autotabml registry-list
autotabml registry-promote <model> <version> --action promote-to-champion
```

### Dataset Locator Syntax

You can point to data in several ways:

```bash
# Local file
autotabml benchmark data/sales.csv --target revenue

# Web URL
autotabml benchmark https://example.com/data.csv --target target

# UCI dataset by ID
autotabml benchmark uci:53 --target species

# UCI dataset by name
autotabml benchmark uci:"Heart Disease" --target condition
```

---

## Where Things Are Stored

All data stays on your machine in the `artifacts/` folder:

```
artifacts/
├── app/
│   └── app_metadata.sqlite3     ← Local database (job history, dataset records)
├── benchmark/                   ← Benchmark leaderboards and saved models
├── experiments/                 ← Experiment snapshots and tuned models
├── models/                      ← Your saved models (.pkl + metadata)
├── predictions/                 ← Prediction logs and batch results
├── profiling/                   ← Data profiling reports (HTML)
├── validation/                  ← Data quality reports (JSON)
├── notebooks/                   ← Auto-generated Jupyter notebooks
└── mlflow/                      ← MLflow tracking data and registry
```

**Settings** are saved to `~/.autotabml/settings.json`.
**API keys** are never written to disk.

---

## Glossary

| Term                     | Plain-English Meaning                                                |
| ------------------------ | -------------------------------------------------------------------- |
| **Algorithm**            | A mathematical recipe for learning patterns from data                |
| **Benchmark**            | A quick test that ranks many algorithms on your data                 |
| **Classification**       | Predicting a category (e.g. "spam" or "not spam")                   |
| **Cross-validation**     | Splitting data multiple ways to test a model more thoroughly         |
| **Dataset fingerprint**  | A unique ID for the exact data a model was trained on                |
| **EDA**                  | Exploratory Data Analysis — visualising your data before modelling   |
| **Feature**              | A column in your data used as input to a model                       |
| **GPU**                  | Graphics Processing Unit — a chip that speeds up model training      |
| **Hyperparameter tuning**| Adjusting a model's internal settings to improve performance         |
| **MLflow**               | An open-source tool for tracking experiments and managing models      |
| **Model**                | The trained result of running an algorithm on your data              |
| **Preprocessing**        | Cleaning and transforming raw data so an algorithm can use it        |
| **Regression**           | Predicting a number (e.g. a price or temperature)                   |
| **Registry**             | A version-controlled catalogue of your best models                   |
| **Target column**        | The column you want the model to predict                             |
| **Test size**            | The percentage of data held back to evaluate the model fairly        |
| **Training**             | The process of feeding data into an algorithm so it learns patterns  |
| **UCI Repository**       | A popular public collection of datasets for ML research              |
| **Validation**           | Checking your data for problems before using it                      |

---

*Last updated: April 2026*
