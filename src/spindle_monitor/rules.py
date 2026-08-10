from __future__ import annotations

import math

from .config import ProjectConfig, SensorConfig
from .models import Status


def validate_value(value: float, config: SensorConfig) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{config.display_name} is not finite: {value!r}")
    if not config.valid_min <= numeric <= config.valid_max:
        raise ValueError(
            f"{config.display_name} {numeric} {config.unit} is outside the configured "
            f"valid range [{config.valid_min}, {config.valid_max}]"
        )
    return numeric


def threshold_reached(value: float, threshold: float, project: ProjectConfig) -> bool:
    if project.threshold_comparison == "greater_or_equal":
        return value >= threshold
    return value > threshold


def status_for_value(value: float, sensor: SensorConfig, project: ProjectConfig) -> Status:
    if threshold_reached(value, sensor.critical, project):
        return Status.CRITICAL
    if threshold_reached(value, sensor.warning, project):
        return Status.WARNING
    return Status.NORMAL


def severity_for_value(value: float, sensor: SensorConfig) -> float:
    """Map engineering bands to an interpretable 0..1.5 degradation scale.

    Baseline -> 0.0, warning threshold -> 0.6, critical threshold -> 1.0,
    severity_cap_value -> 1.5. Values below baseline are clipped to zero.
    """
    if value <= sensor.healthy_baseline:
        return 0.0
    if value <= sensor.warning:
        span = sensor.warning - sensor.healthy_baseline
        return 0.6 * (value - sensor.healthy_baseline) / span
    if value <= sensor.critical:
        span = sensor.critical - sensor.warning
        return 0.6 + 0.4 * (value - sensor.warning) / span
    span = sensor.severity_cap_value - sensor.critical
    severity = 1.0 + 0.5 * (value - sensor.critical) / span
    return min(1.5, severity)


def health_from_severity(severity: float) -> float:
    return max(0.0, min(100.0, 100.0 * (1.0 - min(severity, 1.0))))


def critical_margin_percent(value: float, sensor: SensorConfig) -> float:
    """Remaining margin to the critical manufacturer threshold.

    100% means at or below the configured healthy baseline.
    0% means the critical threshold has been reached or exceeded.
    This is a threshold margin, not a measured percentage of physical life.
    """
    if value <= sensor.healthy_baseline:
        return 100.0
    span = sensor.critical - sensor.healthy_baseline
    if span <= 0:
        return 0.0
    margin = 100.0 * (sensor.critical - value) / span
    return max(0.0, min(100.0, margin))
