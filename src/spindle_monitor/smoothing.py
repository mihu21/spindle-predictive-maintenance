from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median
from copy import deepcopy

from .config import ProjectConfig
from .kalman import ConstantVelocityKalmanFilter


@dataclass(frozen=True)
class SmoothedSignal:
    rolling_median: float
    fast_ewma: float
    slow_ewma: float
    kalman_level: float
    kalman_rate_per_hour: float


class SignalSmoother:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        window = config.smoothing.rolling_median_window
        self.windows = {key: deque(maxlen=window) for key in config.sensors}
        self.fast: dict[str, float] = {}
        self.slow: dict[str, float] = {}
        self.filters = {
            key: ConstantVelocityKalmanFilter(
                config.smoothing.kalman_process_variance,
                config.smoothing.kalman_measurement_variance,
                minimum=sensor.valid_min,
                maximum=sensor.valid_max,
            )
            for key, sensor in config.sensors.items()
        }

    def reset(self, values: dict[str, float] | None = None) -> dict[str, SmoothedSignal] | None:
        """Discard trend state at a confirmed lifecycle boundary.

        A confirmed reset is a new machine-condition lifecycle. Carrying an
        old Kalman rate or EWMA level into it would create a false forecast.
        When the confirming reading is supplied, it seeds the new state.
        """
        window = self.config.smoothing.rolling_median_window
        self.windows = {key: deque(maxlen=window) for key in self.config.sensors}
        self.fast = {}
        self.slow = {}
        self.filters = {
            key: ConstantVelocityKalmanFilter(
                self.config.smoothing.kalman_process_variance,
                self.config.smoothing.kalman_measurement_variance,
                minimum=sensor.valid_min,
                maximum=sensor.valid_max,
            )
            for key, sensor in self.config.sensors.items()
        }
        if values is not None:
            return self.update(values, 1.0)
        return None

    def update(self, values: dict[str, float], dt_seconds: float) -> dict[str, SmoothedSignal]:
        output: dict[str, SmoothedSignal] = {}
        for key, value in values.items():
            window = self.windows[key]
            window.append(float(value))
            med = float(median(window))
            previous_fast = self.fast.get(key, med)
            previous_slow = self.slow.get(key, med)
            fast = self.config.smoothing.fast_ewma_alpha * med + (
                1.0 - self.config.smoothing.fast_ewma_alpha
            ) * previous_fast
            slow = self.config.smoothing.slow_ewma_alpha * med + (
                1.0 - self.config.smoothing.slow_ewma_alpha
            ) * previous_slow
            self.fast[key] = fast
            self.slow[key] = slow
            level, rate_per_second = self.filters[key].update(fast, dt_seconds)
            output[key] = SmoothedSignal(
                rolling_median=med,
                fast_ewma=fast,
                slow_ewma=slow,
                kalman_level=level,
                kalman_rate_per_hour=rate_per_second * 3600.0,
            )
        return output

    def preview(self, values: dict[str, float], dt_seconds: float) -> dict[str, SmoothedSignal]:
        """Evaluate a raw safety/audit row without committing it to trend state."""
        return deepcopy(self).update(values, dt_seconds)
