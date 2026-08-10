from __future__ import annotations

import math
from datetime import datetime
from typing import Sequence

import numpy as np

from .config import MonitoringConfig, SensorConfig
from .models import Status, ThresholdForecast


_Q10_HAZARD = -math.log(0.90)
_Q50_HAZARD = -math.log(0.50)
_Q90_HAZARD = -math.log(0.10)


def _normal_survival(z: np.ndarray) -> np.ndarray:
    """Fast NumPy-only approximation of the standard-normal survival function."""
    scaled = math.sqrt(2.0 / math.pi) * (z + 0.044715 * np.power(z, 3))
    return 0.5 * (1.0 - np.tanh(scaled))


def _regression(
    timestamps: Sequence[datetime], values: Sequence[float]
) -> tuple[float, float, float, float, float] | None:
    if len(timestamps) != len(values) or len(values) < 3:
        return None

    origin = timestamps[0]
    x = np.array(
        [(timestamp - origin).total_seconds() for timestamp in timestamps],
        dtype=float,
    )
    y = np.asarray(values, dtype=float)

    if float(np.ptp(x)) <= 0:
        return None

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered_x = x - x_mean
    denominator = float(np.sum(centered_x**2))
    if denominator <= 0:
        return None

    slope = float(np.sum(centered_x * (y - y_mean)) / denominator)
    intercept = y_mean - slope * x_mean
    predicted = intercept + slope * x
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_total = float(np.sum((y - y_mean) ** 2))
    r_squared = 1.0 - ss_res / ss_total if ss_total > 1e-12 else 0.0
    residual_sigma = float(np.std(residuals, ddof=2))
    fitted_current = float(intercept + slope * x[-1])
    return slope, intercept, r_squared, residual_sigma, fitted_current


def _median_sample_seconds(timestamps: Sequence[datetime]) -> float | None:
    if len(timestamps) < 2:
        return None
    diffs = np.diff(
        np.array([timestamp.timestamp() for timestamp in timestamps], dtype=float)
    )
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return None
    return float(np.median(positive))


def _crossing_quantiles(
    *,
    target_value: float,
    current_value: float,
    fitted_current: float,
    slope_per_second: float,
    residual_sigma: float,
    sample_seconds: float,
    monitoring: MonitoringConfig,
) -> tuple[float | None, float | None, float | None]:
    if slope_per_second <= 0:
        return None, None, None

    grid_seconds = max(
        sample_seconds,
        float(monitoring.forecast_step_minutes) * 60.0,
    )
    max_seconds = float(monitoring.forecast_horizon_hours) * 3600.0
    future_seconds = np.arange(
        grid_seconds,
        max_seconds + grid_seconds,
        grid_seconds,
        dtype=float,
    )
    future_mean = fitted_current + slope_per_second * future_seconds

    if residual_sigma <= 1e-12:
        reached = np.flatnonzero(future_mean >= target_value)
        if len(reached) == 0:
            return None, None, None
        seconds = float(future_seconds[reached[0]])
        return seconds, seconds, seconds

    z = (target_value - future_mean) / residual_sigma
    point_probability = np.clip(_normal_survival(z), 0.0, 1.0)

    # A grid point may represent several original samples. Multiplying by the
    # number of opportunities approximates cumulative first-passage hazard.
    opportunities = max(grid_seconds / sample_seconds, 1.0)
    cumulative_hazard = np.cumsum(point_probability * opportunities)

    def crossing(hazard: float) -> float | None:
        indices = np.flatnonzero(cumulative_hazard >= hazard)
        if len(indices) == 0:
            return None
        return float(future_seconds[indices[0]])

    return (
        crossing(_Q50_HAZARD),
        crossing(_Q10_HAZARD),
        crossing(_Q90_HAZARD),
    )


def _unavailable_forecast(
    sensor: SensorConfig,
    current_status: Status,
    confidence: str,
    explanation: str,
    slope_per_hour: float | None = None,
    r_squared: float | None = None,
) -> ThresholdForecast:
    target_status = "WARNING" if current_status == Status.NORMAL else "CRITICAL"
    target_value = sensor.warning if current_status == Status.NORMAL else sensor.critical
    return ThresholdForecast(
        sensor=sensor.key,
        target_status=target_status,
        target_value=target_value,
        eta_seconds=None,
        earliest_seconds=None,
        latest_seconds=None,
        slope_per_hour=slope_per_hour,
        r_squared=r_squared,
        confidence=confidence,
        explanation=explanation,
        critical_eta_seconds=None,
        critical_earliest_seconds=None,
        critical_latest_seconds=None,
    )


def forecast_threshold(
    sensor: SensorConfig,
    current_value: float,
    current_status: Status,
    timestamps: Sequence[datetime],
    raw_values: Sequence[float],
    monitoring: MonitoringConfig,
) -> ThresholdForecast:
    if current_status == Status.CRITICAL:
        return ThresholdForecast(
            sensor=sensor.key,
            target_status="CRITICAL",
            target_value=sensor.critical,
            eta_seconds=0.0,
            earliest_seconds=0.0,
            latest_seconds=0.0,
            slope_per_hour=None,
            r_squared=None,
            confidence="reached",
            explanation="Critical manufacturer threshold is already exceeded.",
            critical_eta_seconds=0.0,
            critical_earliest_seconds=0.0,
            critical_latest_seconds=0.0,
        )

    if len(raw_values) < monitoring.minimum_forecast_points:
        return _unavailable_forecast(
            sensor,
            current_status,
            "insufficient_data",
            (
                f"At least {monitoring.minimum_forecast_points} readings are required "
                "before forecasting."
            ),
        )

    regression = _regression(timestamps, raw_values)
    sample_seconds = _median_sample_seconds(timestamps)
    if regression is None or sample_seconds is None:
        return _unavailable_forecast(
            sensor,
            current_status,
            "insufficient_data",
            "The recent timestamps do not contain enough time variation.",
        )

    slope, _, r_squared, residual_sigma, fitted_current = regression
    slope_per_hour = slope * 3600.0

    if slope <= 0:
        return _unavailable_forecast(
            sensor,
            current_status,
            "no_rising_trend",
            "The recent sensor trend is flat or decreasing.",
            slope_per_hour=slope_per_hour,
            r_squared=r_squared,
        )

    target_status = "WARNING" if current_status == Status.NORMAL else "CRITICAL"
    target_value = sensor.warning if current_status == Status.NORMAL else sensor.critical

    eta, earliest, latest = _crossing_quantiles(
        target_value=target_value,
        current_value=current_value,
        fitted_current=fitted_current,
        slope_per_second=slope,
        residual_sigma=residual_sigma,
        sample_seconds=sample_seconds,
        monitoring=monitoring,
    )
    critical_eta, critical_earliest, critical_latest = _crossing_quantiles(
        target_value=sensor.critical,
        current_value=current_value,
        fitted_current=fitted_current,
        slope_per_second=slope,
        residual_sigma=residual_sigma,
        sample_seconds=sample_seconds,
        monitoring=monitoring,
    )

    if eta is None and critical_eta is None:
        return _unavailable_forecast(
            sensor,
            current_status,
            "beyond_horizon",
            (
                "The rising trend does not reach the configured threshold within "
                f"{monitoring.forecast_horizon_hours:g} forecast hours."
            ),
            slope_per_hour=slope_per_hour,
            r_squared=r_squared,
        )

    interval_ratio = None
    if critical_eta and critical_earliest is not None and critical_latest is not None:
        interval_ratio = (critical_latest - critical_earliest) / critical_eta

    if r_squared >= 0.70 and (interval_ratio is None or interval_ratio <= 1.0):
        confidence = "high"
    elif r_squared >= monitoring.minimum_forecast_r_squared:
        confidence = "medium"
    else:
        confidence = "low"

    return ThresholdForecast(
        sensor=sensor.key,
        target_status=target_status,
        target_value=target_value,
        eta_seconds=eta,
        earliest_seconds=earliest,
        latest_seconds=latest,
        slope_per_hour=float(slope_per_hour),
        r_squared=float(r_squared),
        confidence=confidence,
        explanation=(
            "Probabilistic first-threshold forecast from the recent linear trend "
            "and observed sensor noise; the interval is the estimated 10% to 90% "
            "crossing range."
        ),
        critical_eta_seconds=critical_eta,
        critical_earliest_seconds=critical_earliest,
        critical_latest_seconds=critical_latest,
    )
