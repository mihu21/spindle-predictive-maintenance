from __future__ import annotations

from collections import deque
from bisect import bisect_left, insort
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np

from .config import ProjectConfig
from .models import FeatureHistoryAction, Status
from .smoothing import SmoothedSignal


WINDOWS_MINUTES: tuple[int, ...] = (5, 15, 30, 60, 180, 360, 720, 1440)


def _slope_per_hour(points: list[tuple[datetime, float]]) -> float:
    if len(points) < 3:
        return 0.0
    origin = points[0][0]
    x = np.asarray(
        [(timestamp - origin).total_seconds() / 3600.0 for timestamp, _ in points],
        dtype=float,
    )
    y = np.asarray([value for _, value in points], dtype=float)
    if float(np.ptp(x)) <= 0:
        return 0.0
    centered = x - float(np.mean(x))
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(centered * (y - float(np.mean(y)))) / denominator)


def _slope_arrays(x_seconds: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    x = (x_seconds - x_seconds[0]) / 3600.0
    if float(np.ptp(x)) <= 0:
        return 0.0
    centered = x - float(np.mean(x))
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(centered * (y - float(np.mean(y)))) / denominator)


def feature_names(
    sensor_keys: Iterable[str], windows_minutes: tuple[int, ...] = WINDOWS_MINUTES
) -> list[str]:
    names: list[str] = []
    per_sensor_current = (
        "raw",
        "rolling_median",
        "fast_ewma",
        "slow_ewma",
        "kalman_level",
        "kalman_rate_per_hour",
        "fast_slow_difference",
        "distance_warning",
        "distance_critical",
        "normalized_severity",
    )
    per_window = (
        "mean",
        "median",
        "minimum",
        "maximum",
        "std",
        "slope_per_hour",
        "change",
        "slope_acceleration_per_hour2",
        "warning_fraction",
        "critical_fraction",
        "threshold_crossings",
        "sampling_gap_mean_seconds",
        "sampling_gap_max_seconds",
        "recent_missing_fraction",
        "feature_available",
        "available_history_duration_seconds",
        "coverage_fraction",
        "window_mature",
        "sample_count",
        "gap_or_discontinuity",
    )
    for sensor in sensor_keys:
        names.extend(f"{sensor}__{name}" for name in per_sensor_current)
        for minutes in windows_minutes:
            names.extend(
                f"{sensor}__w{minutes}m__{name}" for name in per_window
            )
    names.extend(
        [
            "consecutive_warning_hours",
            "consecutive_critical_hours",
            "elapsed_lifecycle_hours",
            "maximum_normalized_severity",
            "weighted_normalized_severity",
        ]
    )
    return names


class _RollingTracker:
    """Incremental timestamp window with exact online statistics."""

    def __init__(self, minutes: int, expected_seconds: float = 60.0) -> None:
        self.minutes = minutes
        self.expected_seconds = expected_seconds
        self.points: deque[tuple[datetime, float, Status, float]] = deque()
        self.sorted_values: list[float] = []
        self.gaps: deque[float] = deque()
        self.sorted_gaps: list[float] = []
        self.sx = self.sy = self.sxx = self.sxy = self.syy = 0.0
        self.warning_count = self.critical_count = self.crossings = 0
        self.previous_slope = 0.0
        self.previous_timestamp: datetime | None = None
        self.current_slope = 0.0
        self.current_acceleration = 0.0
        self.gap_sum = 0.0
        self.missing_intervals = 0.0

    @staticmethod
    def _remove_sorted(values: list[float], value: float) -> None:
        index = bisect_left(values, value)
        if index < len(values):
            values.pop(index)

    def _remove_left(self) -> None:
        _, value, status, x = self.points.popleft()
        self._remove_sorted(self.sorted_values, value)
        self.sx -= x; self.sy -= value; self.sxx -= x * x; self.sxy -= x * value; self.syy -= value * value
        self.warning_count -= int(status >= Status.WARNING)
        self.critical_count -= int(status >= Status.CRITICAL)
        if self.points:
            self.crossings -= int(status != self.points[0][2])
            gap = self.gaps.popleft()
            self._remove_sorted(self.sorted_gaps, gap)
            self.gap_sum -= gap
            self.missing_intervals -= max(0.0, round(gap / self.expected_seconds) - 1.0)

    def update(self, timestamp: datetime, value: float, status: Status) -> None:
        cutoff = timestamp - timedelta(minutes=self.minutes)
        while self.points and self.points[0][0] < cutoff:
            self._remove_left()
        if self.points:
            gap = (timestamp - self.points[-1][0]).total_seconds()
            self.gaps.append(gap); insort(self.sorted_gaps, gap)
            self.gap_sum += gap
            self.missing_intervals += max(0.0, round(gap / self.expected_seconds) - 1.0)
            self.crossings += int(status != self.points[-1][2])
        x = timestamp.timestamp() / 3600.0
        self.points.append((timestamp, value, status, x)); insort(self.sorted_values, value)
        self.sx += x; self.sy += value; self.sxx += x * x; self.sxy += x * value; self.syy += value * value
        self.warning_count += int(status >= Status.WARNING)
        self.critical_count += int(status >= Status.CRITICAL)
        count = len(self.points)
        denominator = count * self.sxx - self.sx * self.sx
        slope = (count * self.sxy - self.sx * self.sy) / denominator if count >= 3 and abs(denominator) > 1e-9 else 0.0
        dt_hours = (timestamp - self.previous_timestamp).total_seconds() / 3600.0 if self.previous_timestamp else 0.0
        self.current_acceleration = (slope - self.previous_slope) / dt_hours if dt_hours > 0 else 0.0
        self.current_slope = slope
        self.previous_slope, self.previous_timestamp = slope, timestamp

    def stats(self, expected_seconds: float, maximum_gap_seconds: float) -> dict[str, float]:
        count = len(self.points)
        if not count:
            return {}
        middle = count // 2
        median_value = self.sorted_values[middle] if count % 2 else (self.sorted_values[middle - 1] + self.sorted_values[middle]) / 2.0
        mean = self.sy / count
        variance = max(0.0, self.syy / count - mean * mean)
        expected_count = max(1.0, self.minutes * 60.0 / expected_seconds)
        available_history = max(
            0.0, (self.points[-1][0] - self.points[0][0]).total_seconds()
        )
        gap_present = bool(self.sorted_gaps and self.sorted_gaps[-1] > maximum_gap_seconds)
        coverage = min(1.0, count / (expected_count + 1.0))
        mature = bool(
            available_history >= max(0.0, self.minutes * 60.0 - expected_seconds)
            and not gap_present
            and coverage >= 0.95
        )
        return {
            "mean": mean, "median": median_value, "minimum": self.sorted_values[0],
            "maximum": self.sorted_values[-1], "std": variance ** .5, "slope_per_hour": self.current_slope,
            "change": self.points[-1][1] - self.points[0][1],
            "slope_acceleration_per_hour2": self.current_acceleration,
            "warning_fraction": self.warning_count / count, "critical_fraction": self.critical_count / count,
            "threshold_crossings": float(self.crossings),
            "sampling_gap_mean_seconds": self.gap_sum / len(self.gaps) if self.gaps else 0.0,
            "sampling_gap_max_seconds": self.sorted_gaps[-1] if self.sorted_gaps else 0.0,
            "recent_missing_fraction": min(1.0, self.missing_intervals / expected_count),
            "feature_available": float(not self.sorted_gaps or self.sorted_gaps[-1] <= maximum_gap_seconds),
            "available_history_duration_seconds": available_history,
            "coverage_fraction": coverage,
            "window_mature": float(mature),
            "sample_count": float(count),
            "gap_or_discontinuity": float(gap_present),
        }


class FeatureGenerator:
    """Causal feature generator shared by runtime and training."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.history: dict[str, deque[tuple[datetime, float, Status]]] = {
            key: deque() for key in config.sensors
        }
        self.warning_since: datetime | None = None
        self.critical_since: datetime | None = None
        self.windows_minutes = tuple(config.feature_windows_minutes)
        self.names = feature_names(config.sensors.keys(), self.windows_minutes)
        self.trackers = {
            key: {minutes: _RollingTracker(minutes, config.target_sampling_interval_seconds) for minutes in self.windows_minutes}
            for key in config.sensors
        }

    def reset(self) -> None:
        """Clear causal feature history at a confirmed lifecycle boundary."""
        self.history = {key: deque() for key in self.config.sensors}
        self.warning_since = None
        self.critical_since = None
        self.trackers = {
            key: {minutes: _RollingTracker(minutes, self.config.target_sampling_interval_seconds) for minutes in self.windows_minutes}
            for key in self.config.sensors
        }

    def _trim(self, timestamp: datetime) -> None:
        cutoff = timestamp - timedelta(minutes=max(self.windows_minutes) + 1)
        for history in self.history.values():
            while history and history[0][0] < cutoff:
                history.popleft()

    @staticmethod
    def _status_fraction(points: list[tuple[datetime, float, Status]], level: Status) -> float:
        if not points:
            return 0.0
        return float(sum(status >= level for _, _, status in points) / len(points))

    @staticmethod
    def _crossings(points: list[tuple[datetime, float, Status]]) -> float:
        if len(points) < 2:
            return 0.0
        return float(
            sum(points[index][2] != points[index - 1][2] for index in range(1, len(points)))
        )

    def update(
        self,
        *,
        timestamp: datetime,
        raw_values: dict[str, float],
        smoothed: dict[str, SmoothedSignal],
        sensor_statuses: dict[str, Status],
        overall_status: Status,
        elapsed_lifecycle_hours: float,
        compute: bool = True,
        history_action: FeatureHistoryAction = FeatureHistoryAction.COMMIT_TO_FEATURE_HISTORY,
    ) -> dict[str, float]:
        self._trim(timestamp)
        if history_action == FeatureHistoryAction.COMMIT_TO_FEATURE_HISTORY:
            for key, value in raw_values.items():
                self.history[key].append((timestamp, float(value), sensor_statuses[key]))
                for tracker in self.trackers[key].values():
                    tracker.update(timestamp, float(value), sensor_statuses[key])

            if overall_status >= Status.WARNING:
                self.warning_since = self.warning_since or timestamp
            else:
                self.warning_since = None
            if overall_status >= Status.CRITICAL:
                self.critical_since = self.critical_since or timestamp
            else:
                self.critical_since = None

        if not compute:
            return {}

        values: dict[str, float] = {}
        severities: list[tuple[float, float]] = []
        for key, sensor in self.config.sensors.items():
            raw = float(raw_values[key])
            signal = smoothed[key]
            span = max(sensor.critical - sensor.healthy_baseline, 1e-9)
            severity = max(0.0, (signal.kalman_level - sensor.healthy_baseline) / span)
            severities.append((severity, sensor.weight))
            current = {
                "raw": raw,
                "rolling_median": signal.rolling_median,
                "fast_ewma": signal.fast_ewma,
                "slow_ewma": signal.slow_ewma,
                "kalman_level": signal.kalman_level,
                "kalman_rate_per_hour": signal.kalman_rate_per_hour,
                "fast_slow_difference": signal.fast_ewma - signal.slow_ewma,
                "distance_warning": sensor.warning - signal.kalman_level,
                "distance_critical": sensor.critical - signal.kalman_level,
                "normalized_severity": severity,
            }
            values.update({f"{key}__{name}": float(value) for name, value in current.items()})

            for minutes in self.windows_minutes:
                prefix = f"{key}__w{minutes}m"
                stats = self.trackers[key][minutes].stats(
                    float(self.config.target_sampling_interval_seconds),
                    self.config.maximum_interpolation_gap_minutes * 60.0,
                )
                values.update({f"{prefix}__{name}": float(value) for name, value in stats.items()})

        warning_hours = (
            (timestamp - self.warning_since).total_seconds() / 3600.0
            if self.warning_since else 0.0
        )
        critical_hours = (
            (timestamp - self.critical_since).total_seconds() / 3600.0
            if self.critical_since else 0.0
        )
        total_weight = sum(weight for _, weight in severities) or 1.0
        values.update(
            {
                "consecutive_warning_hours": float(warning_hours),
                "consecutive_critical_hours": float(critical_hours),
                "elapsed_lifecycle_hours": float(elapsed_lifecycle_hours),
                "maximum_normalized_severity": float(max(severity for severity, _ in severities)),
                "weighted_normalized_severity": float(
                    sum(severity * weight for severity, weight in severities) / total_weight
                ),
            }
        )
        return {name: float(values[name]) for name in self.names}
