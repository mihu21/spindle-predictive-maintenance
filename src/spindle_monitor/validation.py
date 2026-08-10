from __future__ import annotations

import math
from datetime import datetime

from .config import ProjectConfig
from .models import SensorReading, ValidationResult
from .rules import status_for_value, validate_value


class InputValidator:
    """Stateful validation for chronological sensor streams.

    Invalid readings never advance the validator's last accepted timestamp/value
    state, which also keeps them out of lifecycle and feature histories.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.last_timestamp: datetime | None = None
        self.last_values: dict[str, float] = {}
        self.seen_timestamps: set[datetime] = set()

    def validate(
        self,
        reading: SensorReading,
        *,
        row_number: int | None = None,
        raw_row: dict[str, str] | None = None,
        source_label: str | None = None,
    ) -> ValidationResult:
        raw_safety_status = "UNKNOWN"
        try:
            statuses = [
                status_for_value(float(reading.values()[key]), sensor, self.config)
                for key, sensor in self.config.sensors.items()
                if math.isfinite(float(reading.values()[key]))
            ]
            if statuses:
                raw_safety_status = max(statuses).name
        except (TypeError, ValueError, KeyError):
            pass
        if source_label not in {None, "normal", "warning", "critical"}:
            return ValidationResult(
                False,
                f"unknown health_status value: {source_label!r}",
                row_number,
                raw_row,
                source_label=source_label,
                raw_safety_status=raw_safety_status,
            )
        if reading.timestamp in self.seen_timestamps:
            return ValidationResult(
                False,
                "duplicate timestamp",
                row_number,
                raw_row,
                source_label=source_label,
                raw_safety_status=raw_safety_status,
            )
        if self.last_timestamp is not None and reading.timestamp < self.last_timestamp:
            return ValidationResult(
                False,
                "timestamp is earlier than the last accepted timestamp",
                row_number,
                raw_row,
                source_label=source_label,
                raw_safety_status=raw_safety_status,
            )

        values = reading.values()
        try:
            for key, sensor in self.config.sensors.items():
                value = float(values[key])
                if not math.isfinite(value):
                    raise ValueError(f"{key} is not finite")
                validate_value(value, sensor)
                if key in self.last_values:
                    allowed_jump = self.config.smoothing.maximum_jump_multiplier * (
                        sensor.critical - sensor.healthy_baseline
                    )
                    if abs(value - self.last_values[key]) > allowed_jump:
                        raise ValueError(
                            f"possible sensor fault: {key} changed by "
                            f"{abs(value - self.last_values[key]):.3f}, above "
                            f"the configured jump limit {allowed_jump:.3f}"
                        )
        except (TypeError, ValueError) as exc:
            return ValidationResult(
                False,
                str(exc),
                row_number,
                raw_row,
                source_label=source_label,
                raw_safety_status=raw_safety_status,
            )

        self.last_timestamp = reading.timestamp
        self.last_values = values
        self.seen_timestamps.add(reading.timestamp)
        return ValidationResult(
            True,
            "",
            row_number,
            raw_row,
            reading,
            source_label,
            raw_safety_status=raw_safety_status,
        )
