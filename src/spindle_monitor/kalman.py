from __future__ import annotations

import numpy as np


class ConstantVelocityKalmanFilter:
    """Two-state level/rate Kalman filter for an arbitrary engineering value."""

    def __init__(
        self,
        process_variance: float,
        measurement_variance: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)
        self.minimum = minimum
        self.maximum = maximum
        self.x = np.zeros((2, 1), dtype=float)
        self.p = np.eye(2, dtype=float)
        self.initialized = False

    def update(self, measurement: float, dt_seconds: float) -> tuple[float, float]:
        measurement = float(measurement)
        dt = max(float(dt_seconds), 1e-6)
        if not self.initialized:
            self.x[0, 0] = measurement
            self.x[1, 0] = 0.0
            self.p = np.diag([0.1, 0.01])
            self.initialized = True
            return measurement, 0.0

        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        h = np.array([[1.0, 0.0]], dtype=float)
        q_base = self.process_variance
        q = q_base * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=float
        )
        r = np.array([[self.measurement_variance]], dtype=float)

        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        innovation = np.array([[measurement]], dtype=float) - h @ self.x
        innovation_covariance = h @ self.p @ h.T + r
        kalman_gain = self.p @ h.T @ np.linalg.inv(innovation_covariance)
        self.x = self.x + kalman_gain @ innovation
        self.p = (np.eye(2, dtype=float) - kalman_gain @ h) @ self.p

        level = float(self.x[0, 0])
        if self.minimum is not None:
            level = max(level, self.minimum)
        if self.maximum is not None:
            level = min(level, self.maximum)
        self.x[0, 0] = level
        return level, float(self.x[1, 0])


class DegradationKalmanFilter(ConstantVelocityKalmanFilter):
    """Backward-compatible 0..1.5 degradation-state Kalman filter."""

    def __init__(self, process_variance: float, measurement_variance: float) -> None:
        super().__init__(
            process_variance,
            measurement_variance,
            minimum=0.0,
            maximum=1.5,
        )
