from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


V2_4_SENSORS = ("vrms", "arms", "apeak", "crest", "temp")
V2_4_WINDOWS_HOURS = (1.0, 6.0, 24.0)


def _window_label(hours: float) -> str:
    return f"{int(hours)}h"


V2_4_RATE_FEATURES = tuple(
    name
    for sensor in V2_4_SENSORS
    for name in (
        *(f"v24_{sensor}_robust_slope_{_window_label(window)}" for window in V2_4_WINDOWS_HOURS),
        f"v24_{sensor}_slope_accel_1h_6h",
        f"v24_{sensor}_slope_accel_6h_24h",
        f"v24_{sensor}_ewma_fast",
        f"v24_{sensor}_ewma_slow",
        f"v24_{sensor}_ewma_divergence",
        f"v24_{sensor}_ewma_divergence_normalized",
        f"v24_{sensor}_monotonic_rise_fraction_6h",
        f"v24_{sensor}_trend_sign_consistency_6h",
    )
)

V2_4_STRESS_FEATURES = tuple(
    name
    for sensor in V2_4_SENSORS
    for name in (
        f"v24_{sensor}_prior_baseline_mean",
        f"v24_{sensor}_prior_baseline_std",
        f"v24_{sensor}_prior_baseline_z",
        f"v24_{sensor}_cumulative_positive_z_hours",
        f"v24_{sensor}_elevated_duration_hours",
    )
) + (
    "v24_baseline_ready_fraction",
    "v24_cumulative_combined_stress_hours",
    "v24_elapsed_machine_history_hours",
)

V2_4_CAUSAL_FEATURE_NAMES = V2_4_RATE_FEATURES + V2_4_STRESS_FEATURES


@dataclass
class _PriorBaseline:
    started_at: datetime | None = None
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    frozen: bool = False

    @property
    def std(self) -> float | None:
        if self.count < 2:
            return None
        return math.sqrt(max(0.0, self.m2 / (self.count - 1)))

    def update_after_emission(
        self,
        timestamp: datetime,
        value: float,
        *,
        baseline_hours: float,
        minimum_points: int,
    ) -> None:
        if self.started_at is None:
            self.started_at = timestamp
        if self.frozen:
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        elapsed = (timestamp - self.started_at).total_seconds() / 3600.0
        if elapsed >= baseline_hours and self.count >= minimum_points:
            self.frozen = True


@dataclass
class _SensorState:
    history: deque[tuple[datetime, float]] = field(default_factory=deque)
    baseline: _PriorBaseline = field(default_factory=_PriorBaseline)
    ewma_fast: float | None = None
    ewma_slow: float | None = None
    cumulative_positive_z_hours: float = 0.0
    elevated_duration_hours: float = 0.0
    last_timestamp: datetime | None = None
    last_positive_z: float = 0.0


@dataclass
class _MachineState:
    started_at: datetime
    sensors: dict[str, _SensorState] = field(default_factory=dict)


def _robust_current_slope(
    history: deque[tuple[datetime, float]],
    timestamp: datetime,
    window_hours: float,
) -> float:
    cutoff = timestamp - timedelta(hours=window_hours)
    points = [(ts, value) for ts, value in history if cutoff <= ts <= timestamp]
    if len(points) < 3:
        return math.nan
    current_ts, current_value = points[-1]
    prior = points[:-1]
    if len(prior) > 24:
        sample = np.linspace(0, len(prior) - 1, 24, dtype=int)
        prior = [prior[index] for index in sample]
    slopes = [
        (current_value - value) / elapsed
        for ts, value in prior
        if (elapsed := (current_ts - ts).total_seconds() / 3600.0) > 1e-12
    ]
    return float(np.median(slopes)) if slopes else math.nan


def _trend_consistency(
    history: deque[tuple[datetime, float]],
    timestamp: datetime,
    window_hours: float = 6.0,
) -> tuple[float, float]:
    cutoff = timestamp - timedelta(hours=window_hours)
    values = np.asarray([value for ts, value in history if cutoff <= ts <= timestamp], dtype=float)
    if len(values) < 3:
        return math.nan, math.nan
    differences = np.diff(values)
    nonzero = differences[np.abs(differences) > 1e-12]
    rise = float(np.mean(differences >= 0.0)) if len(differences) else math.nan
    if len(nonzero) == 0:
        return rise, 0.0
    positive = float(np.mean(nonzero > 0.0))
    sign_consistency = max(positive, 1.0 - positive)
    return rise, sign_consistency


class V24CausalFeatureBuilder:
    """Causal v2.4 degradation-rate and accumulated-stress state.

    Baseline features are emitted from prior observations and only then updated with the
    current observation. This makes the online and offline feature contract identical and
    prevents the current row from contaminating its own healthy baseline.
    """

    def __init__(
        self,
        *,
        baseline_hours: float = 3.0,
        baseline_minimum_points: int = 10,
        baseline_std_floor_fraction: float = 0.03,
        elevated_z_threshold: float = 1.5,
        fast_ewma_alpha: float = 0.30,
        slow_ewma_alpha: float = 0.05,
    ) -> None:
        self.baseline_hours = float(baseline_hours)
        self.baseline_minimum_points = int(baseline_minimum_points)
        self.baseline_std_floor_fraction = float(baseline_std_floor_fraction)
        self.elevated_z_threshold = float(elevated_z_threshold)
        self.fast_ewma_alpha = float(fast_ewma_alpha)
        self.slow_ewma_alpha = float(slow_ewma_alpha)
        self._machines: dict[str, _MachineState] = {}

    def reset_machine(self, machine_key: str) -> None:
        self._machines.pop(machine_key, None)

    def update(
        self,
        machine_key: str,
        timestamp: datetime,
        sensor_values: dict[str, float],
        *,
        state_reset: bool = False,
    ) -> dict[str, float]:
        if state_reset:
            self.reset_machine(machine_key)
        machine = self._machines.get(machine_key)
        if machine is None:
            machine = _MachineState(started_at=timestamp)
            self._machines[machine_key] = machine
        result: dict[str, float] = {}
        ready = 0
        combined_stress: list[float] = []

        for sensor in V2_4_SENSORS:
            value = float(sensor_values[sensor])
            state = machine.sensors.setdefault(sensor, _SensorState())
            baseline = state.baseline
            prior_mean = baseline.mean if baseline.count else math.nan
            prior_std = baseline.std
            scale_floor = (
                max(abs(prior_mean) * self.baseline_std_floor_fraction, 1e-6)
                if math.isfinite(prior_mean)
                else math.nan
            )
            scale = max(prior_std or 0.0, scale_floor) if math.isfinite(scale_floor) else math.nan
            prior_z = (
                (value - prior_mean) / scale
                if baseline.frozen and math.isfinite(prior_mean) and math.isfinite(scale) and scale > 0.0
                else math.nan
            )
            if baseline.frozen:
                ready += 1

            dt_hours = (
                max(0.0, (timestamp - state.last_timestamp).total_seconds() / 3600.0)
                if state.last_timestamp is not None
                else 0.0
            )
            positive_z = max(0.0, prior_z) if math.isfinite(prior_z) else 0.0
            if dt_hours > 0.0:
                state.cumulative_positive_z_hours += 0.5 * (
                    state.last_positive_z + positive_z
                ) * dt_hours
                if positive_z >= self.elevated_z_threshold:
                    state.elevated_duration_hours += dt_hours
            combined_stress.append(state.cumulative_positive_z_hours)

            state.ewma_fast = (
                value
                if state.ewma_fast is None
                else self.fast_ewma_alpha * value + (1.0 - self.fast_ewma_alpha) * state.ewma_fast
            )
            state.ewma_slow = (
                value
                if state.ewma_slow is None
                else self.slow_ewma_alpha * value + (1.0 - self.slow_ewma_alpha) * state.ewma_slow
            )
            divergence = float(state.ewma_fast - state.ewma_slow)
            normalized_divergence = (
                divergence / scale
                if math.isfinite(scale) and scale > 0.0
                else math.nan
            )

            state.history.append((timestamp, value))
            cutoff = timestamp - timedelta(hours=max(V2_4_WINDOWS_HOURS))
            while state.history and state.history[0][0] < cutoff:
                state.history.popleft()
            slopes = {
                window: _robust_current_slope(state.history, timestamp, window)
                for window in V2_4_WINDOWS_HOURS
            }
            rise_fraction, sign_consistency = _trend_consistency(state.history, timestamp)
            for window in V2_4_WINDOWS_HOURS:
                result[f"v24_{sensor}_robust_slope_{_window_label(window)}"] = slopes[window]
            result[f"v24_{sensor}_slope_accel_1h_6h"] = slopes[1.0] - slopes[6.0]
            result[f"v24_{sensor}_slope_accel_6h_24h"] = slopes[6.0] - slopes[24.0]
            result[f"v24_{sensor}_ewma_fast"] = float(state.ewma_fast)
            result[f"v24_{sensor}_ewma_slow"] = float(state.ewma_slow)
            result[f"v24_{sensor}_ewma_divergence"] = divergence
            result[f"v24_{sensor}_ewma_divergence_normalized"] = normalized_divergence
            result[f"v24_{sensor}_monotonic_rise_fraction_6h"] = rise_fraction
            result[f"v24_{sensor}_trend_sign_consistency_6h"] = sign_consistency
            result[f"v24_{sensor}_prior_baseline_mean"] = prior_mean
            result[f"v24_{sensor}_prior_baseline_std"] = prior_std if prior_std is not None else math.nan
            result[f"v24_{sensor}_prior_baseline_z"] = prior_z
            result[f"v24_{sensor}_cumulative_positive_z_hours"] = float(
                state.cumulative_positive_z_hours
            )
            result[f"v24_{sensor}_elevated_duration_hours"] = float(state.elevated_duration_hours)

            baseline.update_after_emission(
                timestamp,
                value,
                baseline_hours=self.baseline_hours,
                minimum_points=self.baseline_minimum_points,
            )
            state.last_timestamp = timestamp
            state.last_positive_z = positive_z

        result["v24_baseline_ready_fraction"] = ready / len(V2_4_SENSORS)
        result["v24_cumulative_combined_stress_hours"] = float(np.mean(combined_stress))
        result["v24_elapsed_machine_history_hours"] = max(
            0.0, (timestamp - machine.started_at).total_seconds() / 3600.0
        )
        return {name: float(result.get(name, math.nan)) for name in V2_4_CAUSAL_FEATURE_NAMES}


def sensor_values_from_runtime_features(features: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for sensor in V2_4_SENSORS:
        value = features.get(f"{sensor}_raw")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Missing finite runtime sensor feature {sensor}_raw")
        values[sensor] = float(value)
    return values
