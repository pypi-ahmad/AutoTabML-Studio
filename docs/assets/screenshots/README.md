# Screenshots

Captured from a real local Streamlit session on April 6, 2026.

Primary capture dataset: `datasets/sklearn/Diabetes/diabetes.csv`.
Secondary live state: existing local MLflow history, registered models, and saved prediction artifacts already present in the workspace.

The capture flow is repeatable via `scripts/capture_screenshots.py`.

## Captured

- `dashboard-overview.png`: dashboard after loading a real dataset into session
- `dataset-intake.png`: Local Path dataset load using the sklearn diabetes CSV
- `validation-summary.png`: completed validation run with target column selected and 5 checks passed
- `profiling-report.png`: completed profiling run with summary cards and artifact section
- `prediction-center.png`: discovered local saved model plus recent prediction jobs
- `history-view.png`: MLflow-backed run history table with real benchmark runs
- `registry-view.png`: populated model registry with existing registered versions
- `compare-view.png`: compare workflow screen with live workspace state
- `settings-view.png`: execution/runtime settings, including detected CUDA state
- `experiment-lab.png`: experiment workspace page as captured in the current environment

## Captured But Not Used In README Gallery

- `benchmark-leaderboard.png`: automated benchmark-page capture from the live app; the current framing is less portfolio-ready than the rest of the set, so it is kept in the asset folder but omitted from the main README gallery

## Notes

- All screenshots are from the real app, not mocked composites.
- The benchmark page is slow enough in a cold local run that a perfectly framed completed leaderboard shot is still best taken manually after the run settles.
- A short 3-5 minute walkthrough video is still a worthwhile follow-up, but these screenshots are now good enough for portfolio and README use.
