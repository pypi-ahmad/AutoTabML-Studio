"""Path-related helpers shared across app layers."""

from __future__ import annotations

import re

_MODEL_STOP_WORDS = {"classifier", "regressor", "estimator"}


def safe_artifact_stem(name: str | None, default: str = "dataset") -> str:
    """Return a filesystem-safe stem for generated artifact filenames."""

    candidate = (name or default).strip()
    if not candidate:
        candidate = default
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
    sanitized = sanitized.strip("._-")
    return sanitized or default


def model_save_name(dataset_name: str | None, model_name: str | None) -> str:
    """Return a short, human-readable save name: ``DatasetName_ModelAlgo``.

    Examples::

        model_save_name("iris", "Random Forest Classifier") -> "Iris_RandomForest"
        model_save_name("titanic", "Logistic Regression")   -> "Titanic_LogisticRegression"
        model_save_name("my data", "Extra Trees Regressor") -> "MyData_ExtraTrees"
    """
    dataset_part = _pascal_words(dataset_name or "Dataset")
    model_part = _pascal_model(model_name or "Model")
    return f"{dataset_part}_{model_part}"


def _pascal_words(name: str) -> str:
    """Convert any string to PascalCase by capitalising each alphanumeric word."""
    words = re.sub(r"[^A-Za-z0-9]+", " ", name).split()
    return "".join(w.capitalize() for w in words) if words else "Dataset"


def _pascal_model(model_name: str) -> str:
    """PascalCase model name, dropping generic suffixes (Classifier, Regressor, Estimator)."""
    words = re.sub(r"[^A-Za-z0-9]+", " ", model_name).split()
    filtered = [w for w in words if w.lower() not in _MODEL_STOP_WORDS]
    return "".join(w.capitalize() for w in (filtered or words))