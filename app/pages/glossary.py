"""Shared glossary, metric explainers, and inline term helpers.

Provides plain-English definitions for unavoidable ML terms and metrics
so that business users can understand results without external lookup.
"""

from __future__ import annotations

import streamlit as st

# ── Metric explanations ───────────────────────────────────────────────
# Keys are the metric names as they appear in leaderboards/summaries
# (case-insensitive lookup via metric_explanation()).

_METRIC_EXPLANATIONS: dict[str, str] = {
    # Classification
    "accuracy": "The percentage of predictions that were exactly right.",
    "balanced accuracy": "Like accuracy, but accounts for imbalanced classes — gives equal weight to each category.",
    "precision": "Of all the items the model *said* were positive, how many actually were?  Higher = fewer false alarms.",
    "recall": "Of all the items that *were* positive, how many did the model catch?  Higher = fewer missed cases.",
    "f1": "A single score that balances precision and recall.  Good when you care about both false alarms and missed cases.",
    "f1 score": "A single score that balances precision and recall.  Good when you care about both false alarms and missed cases.",
    "auc": "Area Under the ROC Curve — measures how well the model separates classes across all thresholds.  1.0 = perfect.",
    "roc auc": "Area Under the ROC Curve — measures how well the model separates classes across all thresholds.  1.0 = perfect.",
    "log loss": "Measures prediction confidence — lower is better.  Penalises confidently wrong predictions heavily.",
    "kappa": "Cohen's Kappa — agreement between predictions and actual values, adjusted for chance.  1.0 = perfect agreement.",
    "mcc": "Matthews Correlation Coefficient — a balanced measure even with imbalanced classes.  Ranges from −1 to +1.",
    # Regression
    "r2": "R-Squared — how much of the variation in the data the model explains.  1.0 = perfect, 0 = no better than guessing the average.",
    "r-squared": "How much of the variation in the data the model explains.  1.0 = perfect, 0 = no better than guessing the average.",
    "adjusted r-squared": "R-Squared adjusted for the number of input columns — prevents inflated scores when you have many features.",
    "mae": "Mean Absolute Error — the average size of prediction mistakes, in the same units as the target.  Lower is better.",
    "mean absolute error": "The average size of prediction mistakes, in the same units as the target.  Lower is better.",
    "mse": "Mean Squared Error — like MAE but penalises large mistakes more heavily.  Lower is better.",
    "mean squared error": "Like MAE but penalises large mistakes more heavily.  Lower is better.",
    "rmse": "Root Mean Squared Error — the square root of MSE, back in the original units.  Lower is better.",
    "mape": "Mean Absolute Percentage Error — average mistake as a percentage of the actual value.  Lower is better.",
    "rmsle": "Root Mean Squared Logarithmic Error — useful when targets span orders of magnitude.  Lower is better.",
    # Time
    "training time (s)": "Wall-clock seconds the algorithm took to train — useful for comparing speed.",
    "time taken": "Wall-clock seconds the algorithm took to train.",
}

# ── General glossary ──────────────────────────────────────────────────
# Terms that appear in the UI and may confuse business users.

_GLOSSARY: dict[str, str] = {
    "target column": "The column your model learns to predict (sometimes called the 'label' or 'outcome').",
    "feature": "An input column the model uses to make predictions — also called a 'variable' or 'attribute'.",
    "classification": "A task where the model predicts a category, like Yes/No, Approved/Denied, or product type.",
    "regression": "A task where the model predicts a number, like price, temperature, or revenue.",
    "training data": "The portion of your data used to teach the model.",
    "test data": "The portion held back to evaluate how well the model works on unseen data.",
    "cross-validation": "A technique that splits data into multiple rounds and trains/tests repeatedly to get a more reliable score. Shown as 'Validation rounds' in the app.",
    "fold": "One validation round.  With 5 rounds the model is trained and tested 5 times on different data slices (also called 'cross-validation folds').",
    "overfitting": "When a model memorises the training data instead of learning general patterns — performs well on training data but poorly on new data.",
    "leaderboard": "A ranked list of algorithms sorted by their performance score on your data.",
    "hyperparameter tuning": "Automatically adjusting model settings to squeeze out better performance — like fine-tuning a recipe.",
    "champion model": "The version of a model that is currently promoted for production use.",
    "candidate model": "A model version being evaluated before it replaces the current champion.",
    "model registry": "A versioned catalog where you store, promote, and retire model versions.",
    "mlflow": "An open-source platform for tracking experiments and managing models.  Used behind the scenes for history and registry.",
    "experiment": "A training run that compares algorithms, tunes the best one, and optionally saves a model.",
    "benchmark": "A quick screening run that tests many algorithms to find a short-list of the best performers.",
    "data leakage": "When information about the answer accidentally appears in the input data — leads to unrealistically high scores.",
    "gpu": "Graphics Processing Unit — specialised hardware that speeds up model training significantly.",
    "artifact": "A file produced by a run — such as a model file, performance chart, or summary report. Also called an 'output'.",
    "random seed": "A number that makes results reproducible — using the same seed produces the same random split each time.",
}

# ── Performance chart explanations ────────────────────────────────────
# Keys match the plot_id values used by PyCaret.

_PLOT_EXPLANATIONS: dict[str, str] = {
    "auc": "Shows how well the model separates classes at every decision threshold. The closer the curve hugs the top-left corner, the better.",
    "confusion_matrix": "A grid showing correct vs incorrect predictions for each category. Diagonal = correct; off-diagonal = mistakes.",
    "threshold": "Shows how precision and recall change as you adjust the model's decision boundary — helps you find the right trade-off.",
    "pr": "Shows the trade-off between precision (avoiding false alarms) and recall (catching all positives). Useful when categories are imbalanced.",
    "error": "Plots predicted values vs actual values. Points on the diagonal line are perfect predictions; scatter away from it shows error.",
    "class_report": "A per-class breakdown of precision, recall, and F1 — shows which categories the model handles well and which it struggles with.",
    "rfe": "Ranks input columns by importance — shows which features the model relies on most to make predictions.",
    "learning": "Shows how model performance changes as more training data is added. Helps diagnose underfitting or overfitting.",
    "manifold": "Projects high-dimensional data into 2D to visualise how well the model separates different classes.",
    "calibration": "Checks whether the model's confidence matches reality — e.g. when it says '80% likely', is it right 80% of the time?",
    "vc": "Shows how model accuracy changes with different hyperparameter values — helps understand sensitivity to settings.",
    "dimension": "A dimensionality-reduced scatter plot showing how data points cluster in feature space.",
    "feature": "Bar chart showing how much each input column contributes to the model's predictions.",
    "feature_all": "Feature importance across all compared models — shows which columns are universally important.",
    "parameter": "Displays the model's internal parameter values — mostly useful for advanced debugging.",
    "lift": "Shows how much better the model is than random guessing at identifying positive cases, sorted by confidence.",
    "gain": "Shows the cumulative percentage of positive cases captured as you move down the model's ranked predictions.",
    "tree": "A visual diagram of the decision tree (only available for tree-based models).",
    "ks": "Kolmogorov-Smirnov statistic — measures maximum separation between the positive and negative class distributions.",
    "residuals": "Plots prediction errors (residuals) vs predicted values — patterns indicate systematic model weaknesses.",
    "cooks": "Identifies influential data points that disproportionately affect the model. High values may indicate outliers worth investigating.",
}


def plot_explanation(plot_id: str) -> str | None:
    """Return a plain-English explanation for a performance chart, or None if unknown."""
    return _PLOT_EXPLANATIONS.get(plot_id.lower().strip())


def metric_explanation(metric_name: str) -> str | None:
    """Return a plain-English explanation for a metric, or None if unknown."""
    return _METRIC_EXPLANATIONS.get(metric_name.lower().strip())


def glossary_definition(term: str) -> str | None:
    """Return a plain-English definition for a glossary term, or None if unknown."""
    return _GLOSSARY.get(term.lower().strip())


def render_metric_legend(metric_names: list[str], *, key_prefix: str = "metric_legend") -> None:
    """Render a collapsible legend explaining the metrics shown in a table.

    Only includes metrics that have a known explanation.
    """
    explanations = []
    seen: set[str] = set()
    for name in metric_names:
        lower = name.lower().strip()
        if lower in seen:
            continue
        seen.add(lower)
        explanation = metric_explanation(name)
        if explanation:
            explanations.append((name, explanation))

    if not explanations:
        return

    with st.expander("📖 What do these metrics mean?", expanded=False):
        for name, explanation in explanations:
            st.markdown(f"**{name}** — {explanation}")


def render_glossary_sidebar() -> None:
    """Render a collapsible glossary in the sidebar."""
    with st.sidebar.expander("📖 Glossary", expanded=False):
        for term, definition in sorted(_GLOSSARY.items()):
            st.markdown(f"**{term.title()}** — {definition}")


# ── Preset run modes ─────────────────────────────────────────────────

BENCHMARK_PRESETS: dict[str, dict] = {
    "Quick": {
        "description": "Fast screening — sample up to 2 000 rows, smaller test split.",
        "sample_rows": 2000,
        "test_size": 0.2,
        "top_k": 5,
    },
    "Standard": {
        "description": "Balanced speed and coverage — up to 10 000 rows, standard split.",
        "sample_rows": 10000,
        "test_size": 0.25,
        "top_k": 10,
    },
    "Deep": {
        "description": "Full dataset, thorough evaluation — takes longer on large data.",
        "sample_rows": 0,
        "test_size": 0.25,
        "top_k": 15,
    },
}

EXPERIMENT_PRESETS: dict[str, dict] = {
    "Quick": {
        "description": "Fast exploration — 3 validation rounds, 80% training, top 3 models.",
        "fold": 3,
        "train_size": 0.8,
        "n_select": 3,
    },
    "Standard": {
        "description": "Balanced — 5 validation rounds, 70% training, top 5 models.",
        "fold": 5,
        "train_size": 0.7,
        "n_select": 5,
    },
    "Deep": {
        "description": "Thorough evaluation — 10 validation rounds, 70% training, top 10 models.",
        "fold": 10,
        "train_size": 0.7,
        "n_select": 10,
    },
}
