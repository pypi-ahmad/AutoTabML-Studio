# Execution-Based Validation Report — Session 2

**Date:** 2026-04-04  
**Environment:** Python 3.13.12 · Windows 11 · `.venv` at `E:\Github\AutoTabML Studio`  
**Key packages:** streamlit 1.56.0, mlflow 3.10.1, lazypredict 0.3.0, ydata-profiling 4.18.1, scikit-learn 1.8.0, setuptools 81.0.0  
**PyCaret:** Not available (expected on Python 3.13 — `pyproject.toml` guards with `python_version < '3.13'`)

---

## 1. Commands Executed

| Command | Result |
|---|---|
| `autotabml --version` | ✅ `autotabml 0.1.0` |
| `autotabml info` | ✅ version, workspace mode, backend, entrypoints |
| `autotabml --help` | ✅ all 22 subcommands listed |
| `autotabml init-local-storage` | ✅ DB + 10 artifact dirs created |
| `autotabml doctor` | ✅ 0 issues |
| `autotabml validate smoke_train.csv --target approved` | ✅ 5/5 checks passed |
| `autotabml profile smoke_train.csv` | ✅ HTML + JSON artifacts |
| `autotabml benchmark smoke_train.csv --target approved --task-type auto` | ✅ 25 models, best: PassiveAggressiveClassifier 0.75 |
| `autotabml benchmark smoke_train.csv --target approved --sample-rows 50` | ✅ sampled, MLflow run logged |
| `autotabml experiment-run smoke_train.csv --target approved` | ✅ clean PyCaret-unavailable error |
| `autotabml experiment-tune smoke_train.csv --target approved --model-id lr --task-type classification` | ✅ clean error (fixed) |
| `autotabml experiment-evaluate smoke_train.csv --target approved --model-id lr --task-type classification` | ✅ clean error (fixed) |
| `autotabml experiment-save smoke_train.csv --target approved --model-id lr --task-type classification` | ✅ clean error (fixed) |
| `autotabml history-list` | ✅ 3 runs, full 32-char IDs (fixed) |
| `autotabml history-show <8-char-prefix>` | ✅ prefix resolution works (fixed) |
| `autotabml compare-runs <prefix1> <prefix2>` | ✅ metric deltas shown |
| `autotabml predict-single --model-source local_saved_model --model-id test --row-json '{...}'` | ✅ clean model-not-found error |
| `autotabml predict-batch smoke_predict.csv --model-source mlflow_run_model --run-id <id>` | ✅ clean model-artifact error (tracking URI fixed) |
| `autotabml predict-history` | ✅ 0 entries (no successful predictions) |
| `autotabml registry-list` | ✅ empty, then 1 after register |
| `autotabml registry-register test-model --source runs:/...` | ✅ v1 created |
| `autotabml registry-show test-model` | ✅ shows v1 |
| `autotabml registry-promote test-model 1 --action champion` | ✅ alias + tag applied |
| `streamlit run app/main.py` | ✅ all 10+ pages render without crashes |

---

## 2. Real Bugs Found and Fixed

### Bug 1: Streamlit Duplicate Sidebar Navigation
- **Symptom:** Raw Python filenames ("benchmark page", "compare page", etc.) appeared alongside the app's custom radio navigation in the sidebar.
- **Root Cause:** Streamlit 1.56.0 auto-discovers `*.py` files under `app/pages/` and `showSidebarNavigation` defaulted to `true`.
- **Fix:** Created `.streamlit/config.toml` with `[client] showSidebarNavigation = false`.
- **Verified:** Browser inspection confirmed clean single navigation.

### Bug 2: Run ID Truncation Usability Bug
- **Symptom:** `history-list` displayed truncated 12-char run IDs; `history-show` required full 32-char IDs, causing "Run not found" when pasting.
- **Root Cause:** `run.run_id[:12]` in `cmd_history_list` output formatting.
- **Fix:** (a) Show full IDs in listing, (b) Added `resolve_run_id()` prefix-matching to `HistoryService`, (c) Wired into `cmd_history_show` and `cmd_compare_runs`.
- **Verified:** 8-char prefix correctly resolves; ambiguous/missing prefix raises clean error.

### Bug 3: Prediction Model Loader Missing Tracking URI
- **Symptom:** `predict-batch` with MLflow run model source failed with "Run not found" even though `history-show` found the same run.
- **Root Cause:** `MLflowModelLoader._load_pyfunc_model()` didn't call `mlflow.set_tracking_uri()` before `pyfunc.load_model()`.
- **Fix:** Added `mlflow.set_tracking_uri()` and `mlflow.set_registry_uri()` calls before `pyfunc.load_model()`.
- **Verified:** Error changed to correct "Failed to download artifacts" (benchmark runs don't log model artifacts).

### Bug 4: Raw Tracebacks from experiment-tune/evaluate/save
- **Symptom:** Running `experiment-tune`, `experiment-evaluate`, or `experiment-save` produced raw Python tracebacks instead of clean error messages when PyCaret was unavailable.
- **Root Cause:** These three commands did NOT wrap their `service.setup_experiment()` calls in `try/except` like `cmd_experiment_run` did.
- **Fix:** Wrapped all three in `try/except` with `_cli_error(exc)`.
- **Verified:** All three produce clean one-line error messages.

---

## 3. Truth-Alignment Fixes

| Issue | Before | After |
|---|---|---|
| `.env.example` MLflow URI | `sqlite:///mlruns` | `sqlite:///artifacts/mlflow/mlflow.db` |
| README MLflow URI example | `sqlite:///mlruns` | `sqlite:///artifacts/mlflow/mlflow.db` |
| CLI module docstring | "validation and profiling" (6 commands) | "AutoTabML Studio" (22 commands) |
| README folder structure | Missing `.streamlit/` | Added `.streamlit/config.toml` |
| README Streamlit section | No mention of config.toml | Explains sidebar config purpose |

---

## 4. Files Modified (This Session)

| File | Change |
|---|---|
| `.streamlit/config.toml` | **NEW** — disable auto sidebar navigation |
| `app/cli.py` | Module docstring updated; full IDs in history-list; prefix resolution in history-show + compare-runs; try/except in experiment-tune/-evaluate/-save |
| `app/tracking/history_service.py` | Added `resolve_run_id()` method |
| `app/prediction/loader.py` | Added `set_tracking_uri()`/`set_registry_uri()` in `_load_pyfunc_model()` |
| `.env.example` | Corrected MLflow URI to `sqlite:///artifacts/mlflow/mlflow.db` |
| `README.md` | Fixed MLflow URI, added `.streamlit/` to folder tree, added config.toml explanation |
| `tests/test_tracking.py` | 4 new tests for `resolve_run_id()` prefix matching |
| `tests/test_prediction.py` | 1 new test for tracking URI propagation; fixed 2 existing tests for new `import mlflow` in loader |
| `tests/test_cli.py` | 3 new tests for experiment command error handling |
| `tests/test_hardening_smoke.py` | 1 new test for `.streamlit/config.toml` existence |

---

## 5. Test Suite

| Metric | Value |
|---|---|
| Total tests | **253** |
| Passed | **253** |
| Failed | **0** |
| New tests added | **9** |
| Previous session total | 244 |

---

## 6. Artifact Inventory

All expected artifacts were generated with correct content:

- **Validation:** 6 files (JSON + MD × 3 runs) — 5 checks, 0 failures
- **Profiling:** 4 files (HTML + JSON × 2 runs) — 100 rows, 6 cols
- **Benchmark:** 21+ files across 3 runs — leaderboards, summaries, plots, MD reports
- **MLflow:** 3 tracked runs in SQLite, 1 registered model (test-model v1, champion alias)
- **App metadata DB:** 5 tables, 15 jobs, 3 datasets, 1 project — all consistent with CLI output

---

## 7. Remaining Blockers

| Item | Status | Notes |
|---|---|---|
| PyCaret on Python 3.13 | Expected limitation | Documented in README, demo-guide fallback paths |
| Prediction from benchmark runs | Expected limitation | Benchmark runs don't log model artifacts (only leaderboards); need `experiment-save` to get predictable models |
| Notebook mode | Placeholder | Documented honestly |

---

## 8. Demo-Readiness Assessment

**Ready for demo** with the following flow:
1. `init-local-storage` → `doctor` → `streamlit run app/main.py`
2. Dashboard shows workspace metadata
3. Validation → Profiling → Benchmark all work end-to-end
4. History/Compare/Registry all functional with MLflow
5. Experiment commands produce clean "PyCaret unavailable" messages on Python 3.13
6. CLI and UI both operational, no crashes

**Fallback paths** documented in `docs/demo-guide.md` for PyCaret-unavailable scenarios.
