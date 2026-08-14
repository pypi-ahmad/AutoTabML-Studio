"""TimesFM 2.5 local forecasting with grouped-series and backtest support."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.artifacts.manager import LocalArtifactManager
from app.modeling.foundation.checkpoints import TIMESFM_CHECKPOINT, CheckpointResolver


class TimesFMConfig(BaseModel):
    """Configuration for TimesFM 2.5 point and quantile forecasts."""

    timestamp_column: str
    target_column: str
    group_column: str | None = None
    horizon: int = Field(default=12, ge=1, le=1000)
    context_length: int = Field(default=1024, ge=2, le=16384)
    frequency: str | None = None
    backtest: bool = True
    min_context: int = Field(default=32, ge=2)
    normalize_inputs: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    fix_quantile_crossing: bool = True


@dataclass
class TimesFMResult:
    forecast: pd.DataFrame
    backtest: pd.DataFrame | None
    metrics: dict[str, float | int]
    warnings: list[str]
    checkpoint_path: Path
    artifacts: dict[str, Path]


ModelFactory = Callable[[Path, TimesFMConfig], Any]


class TimesFMService:
    """Forecast one or more independent regular time series with TimesFM 2.5."""

    def __init__(
        self,
        *,
        checkpoint_resolver: CheckpointResolver | None = None,
        model_factory: ModelFactory | None = None,
        artifact_manager: LocalArtifactManager | None = None,
    ) -> None:
        self._resolver = checkpoint_resolver or CheckpointResolver()
        self._model_factory = model_factory or _build_model
        self._artifacts = artifact_manager or LocalArtifactManager()

    def run(
        self,
        dataframe: pd.DataFrame,
        config: TimesFMConfig,
        *,
        allow_download: bool = False,
        output_dir: Path | None = None,
    ) -> TimesFMResult:
        required = [config.timestamp_column, config.target_column]
        if config.group_column:
            required.append(config.group_column)
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Required columns not found: {', '.join(missing)}")

        checkpoint_path = self._resolver.resolve(TIMESFM_CHECKPOINT, allow_download=allow_download)
        model = self._model_factory(checkpoint_path, config)
        forecast_frames: list[pd.DataFrame] = []
        backtest_frames: list[pd.DataFrame] = []
        warnings: list[str] = []
        groups = [(None, dataframe)]
        if config.group_column:
            groups = list(dataframe.groupby(config.group_column, dropna=False, sort=False))

        for group_name, raw_group in groups:
            label = str(group_name) if group_name is not None else "series"
            try:
                series, offset = _prepare_series(raw_group, config)
                if len(series) < config.min_context:
                    raise ValueError(f"requires at least {config.min_context} usable observations")
                forecast_frames.append(_forecast_group(model, series, offset, config, group_name))
                if config.backtest and len(series) >= config.min_context + config.horizon:
                    backtest_frames.append(_backtest_group(model, series, config, group_name))
                elif config.backtest:
                    warnings.append(f"Skipped backtest for {label}: insufficient history.")
            except ValueError as exc:
                if len(groups) == 1:
                    raise
                warnings.append(f"Skipped {label}: {exc}")

        if not forecast_frames:
            raise ValueError("No valid time series remained after validation.")
        forecast = pd.concat(forecast_frames, ignore_index=True)
        backtest = pd.concat(backtest_frames, ignore_index=True) if backtest_frames else None
        metrics: dict[str, float | int] = {"groups_forecast": len(forecast_frames)}
        if backtest is not None:
            metrics.update(_backtest_metrics(backtest))
            metrics["groups_backtested"] = len(backtest_frames)
        artifacts = self._write_artifacts(forecast, backtest, metrics, config, output_dir=output_dir)
        return TimesFMResult(forecast, backtest, metrics, warnings, checkpoint_path, artifacts)

    def _write_artifacts(
        self,
        forecast: pd.DataFrame,
        backtest: pd.DataFrame | None,
        metrics: dict[str, float | int],
        config: TimesFMConfig,
        *,
        output_dir: Path | None,
    ) -> dict[str, Path]:
        directory = output_dir or self._artifacts.settings.experiments_dir / "foundation"
        directory.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex[:8]
        paths = {
            "forecast": directory / f"timesfm_forecast_{token}.csv",
            "metrics": directory / f"timesfm_metrics_{token}.json",
            "summary": directory / f"timesfm_summary_{token}.json",
        }
        self._artifacts.write_dataframe_csv(paths["forecast"], forecast, index=False)
        if backtest is not None:
            paths["backtest"] = directory / f"timesfm_backtest_{token}.csv"
            self._artifacts.write_dataframe_csv(paths["backtest"], backtest, index=False)
        self._artifacts.write_json(paths["metrics"], metrics)
        self._artifacts.write_json(
            paths["summary"],
            {
                "model": TIMESFM_CHECKPOINT.repo_id,
                "revision": TIMESFM_CHECKPOINT.revision,
                "config": config.model_dump(mode="json"),
                "saved_model": False,
            },
        )
        return paths


def _prepare_series(group: pd.DataFrame, config: TimesFMConfig) -> tuple[pd.Series, pd.tseries.offsets.BaseOffset]:
    frame = group[[config.timestamp_column, config.target_column]].copy()
    frame[config.timestamp_column] = pd.to_datetime(frame[config.timestamp_column], errors="coerce")
    frame[config.target_column] = pd.to_numeric(frame[config.target_column], errors="coerce")
    if frame[config.timestamp_column].isna().any():
        raise ValueError("contains invalid timestamps")
    frame = frame.sort_values(config.timestamp_column)
    if frame[config.timestamp_column].duplicated().any():
        raise ValueError("contains duplicate timestamps")
    values = frame.set_index(config.timestamp_column)[config.target_column]
    first = values.first_valid_index()
    last = values.last_valid_index()
    if first is None or last is None:
        raise ValueError("contains no numeric target values")
    values = values.loc[first:last].interpolate(method="linear", limit_direction="both")
    if values.isna().any():
        raise ValueError("contains target gaps that could not be interpolated")
    frequency = config.frequency or pd.infer_freq(values.index)
    if not frequency:
        raise ValueError("frequency could not be inferred; provide a frequency override")
    try:
        offset = pd.tseries.frequencies.to_offset(frequency)
    except ValueError as exc:
        raise ValueError(f"invalid frequency override: {frequency}") from exc
    return values, offset


def _forecast_group(
    model: Any,
    series: pd.Series,
    offset: pd.tseries.offsets.BaseOffset,
    config: TimesFMConfig,
    group_name: Any,
) -> pd.DataFrame:
    inputs = [series.to_numpy(dtype=np.float32)[-config.context_length :]]
    point, quantiles = model.forecast(horizon=config.horizon, inputs=inputs)
    last_timestamp = pd.Timestamp(series.index[-1])
    future = pd.date_range(last_timestamp + offset, periods=config.horizon, freq=offset)
    result = _output_frame(future, point[0], quantiles[0], config.timestamp_column)
    if config.group_column:
        result[config.group_column] = group_name
    return result


def _backtest_group(model: Any, series: pd.Series, config: TimesFMConfig, group_name: Any) -> pd.DataFrame:
    history = series.iloc[: -config.horizon]
    actual = series.iloc[-config.horizon :]
    inputs = [history.to_numpy(dtype=np.float32)[-config.context_length :]]
    point, _ = model.forecast(horizon=config.horizon, inputs=inputs)
    result = pd.DataFrame(
        {
            config.timestamp_column: actual.index,
            "actual": actual.to_numpy(dtype=float),
            "forecast": point[0],
        }
    )
    if config.group_column:
        result[config.group_column] = group_name
    return result


def _output_frame(index: pd.DatetimeIndex, point: np.ndarray, quantiles: np.ndarray, timestamp: str) -> pd.DataFrame:
    payload: dict[str, Any] = {
        timestamp: index,
        "forecast": point,
        "mean": quantiles[:, 0],
    }
    for quantile_index, percentile in enumerate(range(10, 100, 10), start=1):
        payload[f"q{percentile}"] = quantiles[:, quantile_index]
    return pd.DataFrame(payload)


def _backtest_metrics(backtest: pd.DataFrame) -> dict[str, float]:
    actual = backtest["actual"].to_numpy(dtype=float)
    predicted = backtest["forecast"].to_numpy(dtype=float)
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(np.divide(2 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape": float(smape * 100),
    }


def _build_model(checkpoint_path: Path, config: TimesFMConfig) -> Any:
    try:
        import timesfm
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Install the 'timesfm' extra to run TimesFM.") from exc
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(checkpoint_path))
    model.compile(
        timesfm.ForecastConfig(
            max_context=config.context_length,
            max_horizon=config.horizon,
            normalize_inputs=config.normalize_inputs,
            use_continuous_quantile_head=True,
            force_flip_invariance=config.force_flip_invariance,
            infer_is_positive=config.infer_is_positive,
            fix_quantile_crossing=config.fix_quantile_crossing,
        )
    )
    return model
