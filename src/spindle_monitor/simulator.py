from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .config import ProjectConfig
from .models import SensorReading


class DegradationSimulator:
    """Stateful degradation simulator with optional maintenance resets."""

    def __init__(
        self,
        config: ProjectConfig,
        seed: int = 42,
        degradation_per_sample: float = 0.0025,
        start_time: datetime | None = None,
        sample_interval_seconds: float = 60.0,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.degradation = 0.0
        self.degradation_per_sample = float(degradation_per_sample)
        self.timestamp = start_time or datetime.now()
        self.sample_interval_seconds = float(sample_interval_seconds)

    @staticmethod
    def _value_at_severity(severity: float, baseline: float, warning: float, critical: float) -> float:
        if severity <= 0.6:
            return baseline + (severity / 0.6) * (warning - baseline)
        if severity <= 1.0:
            return warning + ((severity - 0.6) / 0.4) * (critical - warning)
        return critical + (severity - 1.0) * (critical - warning)

    def maintenance_reset(self, remaining_degradation: float = 0.02) -> None:
        self.degradation = max(0.0, min(float(remaining_degradation), 0.20))

    def read(self) -> SensorReading:
        self.degradation = min(1.35, self.degradation + self.degradation_per_sample)
        values: dict[str, float] = {}
        noise_ratios = {
            "vibration_mps2": 0.025,
            "temperature_c": 0.018,
            "current_ampere": 0.02,
        }
        for key, sensor in self.config.sensors.items():
            base_value = self._value_at_severity(
                self.degradation,
                sensor.healthy_baseline,
                sensor.warning,
                sensor.critical,
            )
            scale = (sensor.critical - sensor.healthy_baseline) * noise_ratios.get(key, 0.02)
            values[key] = float(base_value + self.rng.normal(0.0, scale))
            values[key] = min(sensor.valid_max, max(sensor.valid_min, values[key]))
        reading = SensorReading(
            timestamp=self.timestamp,
            vibration_mps2=values["vibration_mps2"],
            temperature_c=values["temperature_c"],
            current_ampere=values["current_ampere"],
        )
        self.timestamp += timedelta(seconds=self.sample_interval_seconds)
        return reading
