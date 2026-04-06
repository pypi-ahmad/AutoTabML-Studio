"""Scoring helpers that normalize heterogeneous prediction backends."""

from __future__ import annotations

import pandas as pd

from app.modeling.pycaret.setup_runner import build_pycaret_experiment
from app.prediction.errors import PredictionScoringError
from app.prediction.schemas import LoadedModel, PredictionTaskType
from app.prediction.selectors import to_experiment_task_type

_PREDICTION_COLUMN_CANDIDATES = ["prediction_label", "prediction", "predictions", "label"]
_SCORE_COLUMN_CANDIDATES = ["prediction_score", "score", "probability", "confidence"]


class PredictionScorer:
    """Normalize scoring across local PyCaret and MLflow / sklearn-like models."""

    def score(
        self,
        loaded_model: LoadedModel,
        dataframe: pd.DataFrame,
        *,
        prediction_column_name: str,
        prediction_score_column_name: str,
    ) -> pd.DataFrame:
        """Score a dataframe and append normalized prediction columns."""

        if loaded_model.scorer_kind == "pycaret":
            return _score_with_pycaret(
                loaded_model,
                dataframe,
                prediction_column_name=prediction_column_name,
                prediction_score_column_name=prediction_score_column_name,
            )
        return _score_with_predict_api(
            loaded_model,
            dataframe,
            prediction_column_name=prediction_column_name,
            prediction_score_column_name=prediction_score_column_name,
        )


def _score_with_pycaret(
    loaded_model: LoadedModel,
    dataframe: pd.DataFrame,
    *,
    prediction_column_name: str,
    prediction_score_column_name: str,
) -> pd.DataFrame:
    try:
        experiment_handle = build_pycaret_experiment(to_experiment_task_type(loaded_model.task_type))
        raw_scored = experiment_handle.predict_model(
            loaded_model.native_model,
            data=dataframe.copy(),
            verbose=False,
        )
    except Exception as exc:
        raise PredictionScoringError(f"PyCaret prediction failed: {exc}") from exc

    if not isinstance(raw_scored, pd.DataFrame):
        raise PredictionScoringError("PyCaret prediction did not return a pandas DataFrame.")

    prediction_series = _extract_prediction_series(raw_scored)
    score_series = _extract_score_series(raw_scored)

    result = dataframe.copy()
    result[prediction_column_name] = _align_series(prediction_series, dataframe.index)
    if score_series is not None:
        result[prediction_score_column_name] = _align_series(score_series, dataframe.index)
    return result


def _score_with_predict_api(
    loaded_model: LoadedModel,
    dataframe: pd.DataFrame,
    *,
    prediction_column_name: str,
    prediction_score_column_name: str,
) -> pd.DataFrame:
    model = loaded_model.native_model
    if not hasattr(model, "predict"):
        raise PredictionScoringError("Loaded model does not expose a predict() method.")

    try:
        raw_predictions = model.predict(dataframe.copy())
    except Exception as exc:
        raise PredictionScoringError(f"Prediction failed: {exc}") from exc

    prediction_series, score_series = _normalize_predict_output(
        raw_predictions,
        dataframe,
        loaded_model=loaded_model,
        prediction_score_column_name=prediction_score_column_name,
    )

    if score_series is None and loaded_model.task_type == PredictionTaskType.CLASSIFICATION and hasattr(model, "predict_proba"):
        score_series = _extract_probability_scores(model, dataframe, prediction_series)

    result = dataframe.copy()
    result[prediction_column_name] = _align_series(prediction_series, dataframe.index)
    if score_series is not None:
        result[prediction_score_column_name] = _align_series(score_series, dataframe.index)
    return result


def _normalize_predict_output(
    raw_predictions,
    dataframe: pd.DataFrame,
    *,
    loaded_model: LoadedModel,
    prediction_score_column_name: str,
) -> tuple[pd.Series, pd.Series | None]:
    if isinstance(raw_predictions, pd.DataFrame):
        prediction_series = _extract_prediction_series(raw_predictions)
        score_series = _extract_score_series(raw_predictions, requested_name=prediction_score_column_name)
        return prediction_series, score_series

    if isinstance(raw_predictions, pd.Series):
        return raw_predictions, None

    try:
        prediction_series = pd.Series(raw_predictions, index=dataframe.index)
    except Exception:
        if len(dataframe.index) != 1:
            raise PredictionScoringError(
                "Model returned a scalar prediction for a multi-row batch, which cannot be normalized."
            )
        prediction_series = pd.Series([raw_predictions], index=dataframe.index)

    if len(prediction_series) != len(dataframe.index):
        raise PredictionScoringError(
            f"Prediction output length ({len(prediction_series)}) does not match input rows ({len(dataframe.index)})."
        )
    return prediction_series, None


def _extract_prediction_series(scored_frame: pd.DataFrame) -> pd.Series:
    for column in _PREDICTION_COLUMN_CANDIDATES:
        if column in scored_frame.columns:
            return scored_frame[column]
    if len(scored_frame.columns) == 1:
        return scored_frame.iloc[:, 0]
    raise PredictionScoringError(
        "Prediction output could not be normalized because no prediction column was found."
    )


def _extract_score_series(
    scored_frame: pd.DataFrame,
    *,
    requested_name: str | None = None,
) -> pd.Series | None:
    candidates = []
    if requested_name:
        candidates.append(requested_name)
    candidates.extend(_SCORE_COLUMN_CANDIDATES)
    for column in candidates:
        if column in scored_frame.columns:
            return scored_frame[column]
    return None


def _extract_probability_scores(
    model,
    dataframe: pd.DataFrame,
    prediction_series: pd.Series,
) -> pd.Series | None:
    try:
        raw_scores = model.predict_proba(dataframe.copy())
    except Exception:
        return None

    score_frame = pd.DataFrame(raw_scores, index=dataframe.index)
    if score_frame.empty:
        return None
    if hasattr(model, "classes_"):
        score_frame.columns = [str(label) for label in model.classes_]
        aligned_scores = []
        for row_index, prediction in prediction_series.items():
            key = str(prediction)
            if key in score_frame.columns:
                aligned_scores.append(float(score_frame.loc[row_index, key]))
            else:
                aligned_scores.append(float(score_frame.loc[row_index].max()))
        return pd.Series(aligned_scores, index=dataframe.index)
    return score_frame.max(axis=1)


def _align_series(series: pd.Series, index) -> pd.Series:  # noqa: ANN001
    if series.index.equals(index):
        return series
    return pd.Series(series.to_list(), index=index)
