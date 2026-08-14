from __future__ import annotations

import math

from .config import SensorConfig
from .models import SourceRecord, ValidationResult, VVB001Reading


class VVB001Validator:
    """Validate VVB001 process values against documented sensor measurement ranges.

    These are data-quality limits, not machine-health WARNING/CRITICAL thresholds.
    """

    VRMS_RANGE = (0.0, 45.0)       # mm/s
    CREST_RANGE = (1.0, 50.0)      # unitless
    TEMP_RANGE = (-30.0, 80.0)     # degC
    ACCEL_RANGE_M_S2 = (0.0, 490.3)
    ACCEL_RANGE_G = (0.0, 50.0)

    def __init__(self, config: SensorConfig) -> None:
        self.config = config

    def _range(self, name: str) -> tuple[float, float]:
        if name == "vrms":
            return self.VRMS_RANGE
        if name in {"arms", "apeak"}:
            return self.ACCEL_RANGE_G if self.config.acceleration_unit == "g" else self.ACCEL_RANGE_M_S2
        if name == "crest":
            return self.CREST_RANGE
        if name == "temp":
            return self.TEMP_RANGE
        raise KeyError(name)

    def validate_record(self, record: SourceRecord) -> ValidationResult:
        if record.reading is None:
            return ValidationResult(False, "INVALID", (record.parse_error or "unparseable source row",), None, record.raw)
        reading = record.reading
        reasons: list[str] = []
        for name, value in reading.sensor_values().items():
            if not math.isfinite(value):
                reasons.append(f"{name} is not finite")
                continue
            low, high = self._range(name)
            if value < low or value > high:
                reasons.append(f"{name}={value} is outside VVB001 measurement range [{low}, {high}]")
        if reasons:
            return ValidationResult(False, "INVALID", tuple(reasons), reading, record.raw)

        warnings: list[str] = []
        if reading.arms > self.config.crest_consistency_min_arms:
            expected = reading.apeak / reading.arms
            denominator = max(abs(reading.crest), abs(expected), 1e-12)
            relative_error = abs(reading.crest - expected) / denominator
            if relative_error > self.config.crest_consistency_relative_tolerance:
                warnings.append(
                    f"crest consistency warning: sensor crest={reading.crest:.6g}, "
                    f"apeak/arms={expected:.6g}, relative_error={relative_error:.3f}"
                )
        status = "VALID_WITH_WARNING" if warnings else "VALID"
        return ValidationResult(True, status, tuple(warnings), reading, record.raw)
