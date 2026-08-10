from __future__ import annotations

from .config import ProjectConfig
from .models import StatisticalForecast, Status, ThresholdForecast
from .smoothing import SmoothedSignal


def _target_status(target: str) -> Status:
    if target == "warning":
        return Status.WARNING
    if target == "critical":
        return Status.CRITICAL
    raise ValueError(f"Unsupported statistical forecast target: {target}")


def _target_value(sensor, target: str) -> float:
    return sensor.warning if target == "warning" else sensor.critical


def statistical_time_to_threshold(
    config: ProjectConfig,
    signals: dict[str, SmoothedSignal],
    sensor_statuses: dict[str, Status],
    history_points: int,
    *,
    target: str,
) -> StatisticalForecast:
    """Forecast the first Kalman threshold crossing for one health band.

    Warning and critical are independent forecasts.  Once a threshold is
    already reached the relevant estimate is exactly zero; it never tries to
    infer a maintenance/reset time.
    """
    status_target = _target_status(target)
    reached = [key for key, status in sensor_statuses.items() if status >= status_target]
    if reached:
        sensor_key = max(
            reached,
            key=lambda key: (
                int(sensor_statuses[key]),
                signals[key].kalman_level / max(_target_value(config.sensors[key], target), 1e-9),
            ),
        )
        return StatisticalForecast(
            estimated_hours=0.0,
            forecast_sensor=sensor_key,
            confidence="reached",
            reason=f"A raw manufacturer {target} threshold is already reached.",
            per_sensor_hours={
                key: 0.0 if status >= status_target else None
                for key, status in sensor_statuses.items()
            },
            target=target,
        )

    minimum_points = max(3, min(config.monitoring.minimum_forecast_points, 30))
    if history_points < minimum_points:
        return StatisticalForecast(
            estimated_hours=None,
            forecast_sensor=None,
            confidence="insufficient_history",
            reason="Insufficient accepted history for a stable Kalman rate estimate.",
            per_sensor_hours={key: None for key in config.sensors},
            target=target,
        )

    estimates: dict[str, float | None] = {}
    candidates: list[tuple[float, str, float]] = []
    for key, sensor in config.sensors.items():
        signal = signals[key]
        rate = signal.kalman_rate_per_hour
        distance = _target_value(sensor, target) - signal.kalman_level
        if distance <= 0:
            estimates[key] = 0.0
            candidates.append((0.0, key, rate))
        elif rate <= 1e-9:
            estimates[key] = None
        else:
            hours = distance / rate
            if hours <= config.monitoring.forecast_horizon_hours:
                estimates[key] = float(max(0.0, hours))
                candidates.append((float(max(0.0, hours)), key, rate))
            else:
                estimates[key] = None

    if not candidates:
        rates = [signals[key].kalman_rate_per_hour for key in config.sensors]
        if max(rates) <= 1e-9:
            reason = "No rising Kalman trend is currently present."
            confidence = "no_rising_trend"
        else:
            reason = f"Rising trends do not reach a {target} threshold within the configured horizon."
            confidence = "beyond_horizon"
        return StatisticalForecast(None, None, confidence, reason, estimates, target)

    hours, sensor_key, rate = min(candidates, key=lambda item: item[0])
    sensor = config.sensors[sensor_key]
    normalized_rate = rate / max(_target_value(sensor, target) - sensor.healthy_baseline, 1e-9)
    if history_points >= config.monitoring.minimum_forecast_points and normalized_rate > 0.02:
        confidence = "high"
    elif history_points >= 60 and normalized_rate > 0.005:
        confidence = "medium"
    else:
        confidence = "low"
    return StatisticalForecast(
        estimated_hours=hours,
        forecast_sensor=sensor_key,
        confidence=confidence,
        reason=(
            f"Earliest Kalman level/rate {target} crossing is {sensor.display_name}; "
            f"level={signals[sensor_key].kalman_level:.3f}, "
            f"rate={rate:.4f} {sensor.unit}/hour."
        ),
        per_sensor_hours=estimates,
        target=target,
    )


def statistical_time_to_warning(
    config: ProjectConfig,
    signals: dict[str, SmoothedSignal],
    sensor_statuses: dict[str, Status],
    history_points: int,
) -> StatisticalForecast:
    return statistical_time_to_threshold(
        config, signals, sensor_statuses, history_points, target="warning"
    )


def statistical_time_to_critical(
    config: ProjectConfig,
    signals: dict[str, SmoothedSignal],
    sensor_statuses: dict[str, Status],
    history_points: int,
) -> StatisticalForecast:
    return statistical_time_to_threshold(
        config, signals, sensor_statuses, history_points, target="critical"
    )


def sensor_kalman_threshold_forecast(
    config: ProjectConfig,
    sensor_key: str,
    signal: SmoothedSignal,
    current_status: Status,
    history_points: int,
) -> ThresholdForecast:
    """Per-sensor diagnostic forecast retained for the detailed CSV."""
    sensor = config.sensors[sensor_key]
    target_status = "WARNING" if current_status == Status.NORMAL else "CRITICAL"
    target_value = sensor.warning if current_status == Status.NORMAL else sensor.critical
    if current_status == Status.CRITICAL:
        return ThresholdForecast(
            sensor_key, "CRITICAL", sensor.critical, 0.0, 0.0, 0.0,
            signal.kalman_rate_per_hour, None, "reached",
            "Critical manufacturer threshold is already exceeded.",
            0.0, 0.0, 0.0,
        )
    minimum_points = max(3, min(config.monitoring.minimum_forecast_points, 30))
    if history_points < minimum_points:
        return ThresholdForecast(
            sensor_key, target_status, target_value, None, None, None,
            signal.kalman_rate_per_hour, None, "insufficient_data",
            f"At least {minimum_points} accepted readings are required for Kalman-rate forecasting.",
        )
    rate = signal.kalman_rate_per_hour
    if rate <= 1e-9:
        return ThresholdForecast(
            sensor_key, target_status, target_value, None, None, None,
            rate, None, "no_rising_trend", "No rising Kalman trend is currently present.",
        )
    eta_seconds = max(0.0, (target_value - signal.kalman_level) / rate * 3600.0)
    critical_eta = max(0.0, (sensor.critical - signal.kalman_level) / rate * 3600.0)
    if eta_seconds / 3600.0 > config.monitoring.forecast_horizon_hours:
        eta_seconds = None
    if critical_eta / 3600.0 > config.monitoring.forecast_horizon_hours:
        critical_eta = None
    confidence = "high" if history_points >= config.monitoring.minimum_forecast_points else "medium"
    return ThresholdForecast(
        sensor_key, target_status, target_value, eta_seconds, eta_seconds, eta_seconds,
        rate, None, confidence,
        "Threshold crossing calculated from the per-sensor Kalman level and rate.",
        critical_eta, critical_eta, critical_eta,
    )
