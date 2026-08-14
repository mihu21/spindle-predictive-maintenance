from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import SensorConfig
from .models import VVB001Reading


@dataclass
class WindowState:
    points: deque[tuple[datetime, float]] = field(default_factory=deque)
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, timestamp: datetime, value: float, cutoff: datetime) -> None:
        self.points.append((timestamp, value))
        self.total += value
        self.total_sq += value * value
        while self.points and self.points[0][0] < cutoff:
            _, old = self.points.popleft()
            self.total -= old
            self.total_sq -= old * old

    def stats(self) -> tuple[int, float | None, float | None, float | None]:
        n = len(self.points)
        if n == 0:
            return 0, None, None, None
        mean = self.total / n
        variance = max(0.0, self.total_sq / n - mean * mean)
        std = math.sqrt(variance)
        trend = None
        if n >= 2:
            t0, v0 = self.points[0]
            t1, v1 = self.points[-1]
            hours = (t1 - t0).total_seconds() / 3600.0
            if hours > 0:
                trend = (v1 - v0) / hours
        return n, mean, std, trend


@dataclass
class BaselineState:
    started_at: datetime | None = None
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    frozen: bool = False

    def update(self, timestamp: datetime, value: float, baseline_hours: float, min_points: int) -> None:
        if self.started_at is None:
            self.started_at = timestamp
        if self.frozen:
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        elapsed_h = (timestamp - self.started_at).total_seconds() / 3600.0
        if elapsed_h >= baseline_hours and self.count >= min_points:
            self.frozen = True

    @property
    def std(self) -> float | None:
        if self.count < 2:
            return None
        return math.sqrt(max(0.0, self.m2 / (self.count - 1)))


@dataclass
class MetricState:
    ewma: float | None = None
    last_timestamp: datetime | None = None
    last_value: float | None = None
    windows: dict[int, WindowState] = field(default_factory=dict)
    baseline: BaselineState = field(default_factory=BaselineState)


class FeatureEngine:
    """Causal per-machine feature engine.

    Every (line_sel, machine_id) stream has isolated EWMA/window/baseline state. A long
    gap or explicit lifecycle reset clears all state for that machine, preventing one
    lifecycle or machine from contaminating another.
    """

    def __init__(self, config: SensorConfig) -> None:
        self.config = config
        self._state: dict[str, dict[str, MetricState]] = defaultdict(dict)
        self._last_machine_timestamp: dict[str, datetime] = {}

    def reset_machine(self, machine_key: str) -> None:
        self._state.pop(machine_key, None)
        self._last_machine_timestamp.pop(machine_key, None)


    def baseline_snapshot(self) -> dict[str, dict[str, dict[str, float | int | str | bool | None]]]:
        snapshot: dict[str, dict[str, dict[str, float | int | str | bool | None]]] = {}
        for machine_key, metrics in self._state.items():
            machine: dict[str, dict[str, float | int | str | bool | None]] = {}
            for metric, state in metrics.items():
                b = state.baseline
                if b.count:
                    machine[metric] = {
                        "started_at": b.started_at.isoformat() if b.started_at else None,
                        "count": b.count,
                        "mean": b.mean,
                        "m2": b.m2,
                        "frozen": b.frozen,
                    }
            if machine:
                snapshot[machine_key] = machine
        return snapshot

    def restore_baselines(self, snapshot: dict[str, dict[str, dict[str, object]]] | None) -> None:
        if not snapshot:
            return
        for machine_key, metrics in snapshot.items():
            for metric, raw in metrics.items():
                state = self._metric_state(machine_key, metric)
                started = raw.get("started_at")
                state.baseline = BaselineState(
                    started_at=datetime.fromisoformat(str(started)) if started else None,
                    count=int(raw.get("count", 0)),
                    mean=float(raw.get("mean", 0.0)),
                    m2=float(raw.get("m2", 0.0)),
                    frozen=bool(raw.get("frozen", False)),
                )

    def _metric_state(self, machine_key: str, metric: str) -> MetricState:
        machine = self._state[machine_key]
        if metric not in machine:
            machine[metric] = MetricState(
                windows={minutes: WindowState() for minutes in self.config.feature_windows_minutes}
            )
        return machine[metric]

    def process(self, reading: VVB001Reading) -> dict[str, float | int | str | None]:
        previous_machine_ts = self._last_machine_timestamp.get(reading.machine_key)
        state_reset = False
        if previous_machine_ts is not None:
            gap_h = (reading.timestamp - previous_machine_ts).total_seconds() / 3600.0
            if gap_h > self.config.history_hours:
                self.reset_machine(reading.machine_key)
                state_reset = True

        elapsed_since_previous_h = None
        if previous_machine_ts is not None and not state_reset:
            elapsed_since_previous_h = max(0.0, (reading.timestamp - previous_machine_ts).total_seconds() / 3600.0)
        features: dict[str, float | int | str | None] = {
            "machine_key": reading.machine_key,
            "acceleration_unit": self.config.acceleration_unit,
            "state_reset": int(state_reset),
            "elapsed_since_previous_hours": elapsed_since_previous_h,
        }
        baseline_ready_flags: list[bool] = []

        for metric, value in reading.sensor_values().items():
            state = self._metric_state(reading.machine_key, metric)
            previous_ewma = state.ewma
            state.ewma = value if previous_ewma is None else (
                self.config.ewma_alpha * value + (1.0 - self.config.ewma_alpha) * previous_ewma
            )
            rate = None
            if state.last_timestamp is not None and state.last_value is not None:
                elapsed_h = (reading.timestamp - state.last_timestamp).total_seconds() / 3600.0
                if elapsed_h > 0:
                    rate = (value - state.last_value) / elapsed_h

            state.baseline.update(
                reading.timestamp,
                value,
                baseline_hours=self.config.baseline_hours,
                min_points=self.config.baseline_min_points,
            )
            baseline_mean = state.baseline.mean if state.baseline.count else None
            baseline_std = state.baseline.std
            baseline_ready = state.baseline.frozen
            baseline_ready_flags.append(baseline_ready)
            baseline_delta = value - baseline_mean if baseline_mean is not None else None
            baseline_ratio = None
            baseline_z = None
            if baseline_mean is not None and abs(baseline_mean) > 1e-12:
                baseline_ratio = value / baseline_mean
            if baseline_mean is not None:
                scale_floor = max(abs(baseline_mean) * self.config.baseline_std_floor_fraction, 1e-6)
                scale = max(baseline_std or 0.0, scale_floor)
                baseline_z = (value - baseline_mean) / scale

            features[f"{metric}_raw"] = value
            features[f"{metric}_ewma"] = state.ewma
            features[f"{metric}_delta_ewma"] = value - state.ewma
            features[f"{metric}_rate_per_hour"] = rate
            features[f"{metric}_baseline_mean"] = baseline_mean
            features[f"{metric}_baseline_std"] = baseline_std
            features[f"{metric}_delta_baseline"] = baseline_delta
            features[f"{metric}_ratio_baseline"] = baseline_ratio
            features[f"{metric}_z_baseline"] = baseline_z
            features[f"{metric}_baseline_ready"] = int(baseline_ready)

            for minutes, window in state.windows.items():
                window.add(reading.timestamp, value, reading.timestamp - timedelta(minutes=minutes))
                count, mean, std, trend = window.stats()
                prefix = f"{metric}_{minutes}m"
                # Count is retained for diagnostics/warm-up visibility but excluded from ML
                # training because it can leak elapsed-lifecycle time.
                features[f"{prefix}_count"] = count
                features[f"{prefix}_mean"] = mean
                features[f"{prefix}_std"] = std
                features[f"{prefix}_trend_per_hour"] = trend
                features[f"{prefix}_delta_baseline"] = (mean - baseline_mean) if mean is not None and baseline_mean is not None else None
                features[f"{prefix}_ratio_baseline"] = (mean / baseline_mean) if mean is not None and baseline_mean is not None and abs(baseline_mean) > 1e-12 else None

            state.last_timestamp = reading.timestamp
            state.last_value = value

        if reading.arms > self.config.crest_consistency_min_arms:
            recomputed = reading.apeak / reading.arms
            features["crest_recomputed"] = recomputed
            features["crest_relative_error"] = abs(reading.crest - recomputed) / max(
                abs(reading.crest), abs(recomputed), 1e-12
            )
        else:
            features["crest_recomputed"] = None
            features["crest_relative_error"] = None

        features["baseline_ready"] = int(all(baseline_ready_flags))
        self._last_machine_timestamp[reading.machine_key] = reading.timestamp
        return features
