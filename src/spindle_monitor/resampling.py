from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import SensorReading


@dataclass(frozen=True)
class ProcessedReading:
    raw: SensorReading | None
    processed: SensorReading
    interpolated: bool
    gap_seconds: float
    features_available: bool
    source_sampling_interval_seconds: float
    effective_resampling_interval_seconds: float


def safe_resample(
    readings: list[SensorReading], *, target_interval_seconds: int = 60,
    interpolation_enabled: bool = False, maximum_interpolation_gap_minutes: float = 5.0,
    large_gap_policy: str = "mark_unavailable",
    lifecycle_boundary_timestamps: tuple[datetime, ...] = (),
    source_statuses: list[str | None] | None = None,
) -> list[ProcessedReading]:
    """Resample causally without inventing lifecycle or safety events.

    Interpolation is limited to normal-to-normal spans, so a synthetic row cannot
    create or extend warning/critical confirmation.  Known reset boundaries and
    long gaps are hard barriers.
    """
    if target_interval_seconds <= 0:
        raise ValueError("target_interval_seconds must be positive")
    if large_gap_policy not in {"mark_unavailable", "reject"}:
        raise ValueError("Unsupported large_gap_policy")
    if source_statuses is not None and len(source_statuses) != len(readings):
        raise ValueError("source_statuses must align with readings")
    output: list[ProcessedReading] = []
    for index, reading in enumerate(readings):
        gap = 0.0 if index == 0 else (reading.timestamp - readings[index - 1].timestamp).total_seconds()
        if gap <= 0 and index:
            raise ValueError("Readings must have strictly increasing timestamps")
        if index:
            missing = max(0, int(gap // target_interval_seconds) - 1)
            max_gap_seconds = maximum_interpolation_gap_minutes * 60.0
            crosses_boundary = any(
                readings[index - 1].timestamp < boundary <= reading.timestamp
                for boundary in lifecycle_boundary_timestamps
            )
            safe_status_span = bool(
                source_statuses is None
                or (
                    source_statuses[index - 1] == "normal"
                    and source_statuses[index] == "normal"
                )
            )
            if (
                missing and gap <= max_gap_seconds and interpolation_enabled
                and not crosses_boundary and safe_status_span
            ):
                previous = readings[index - 1]
                for step in range(1, missing + 1):
                    fraction = step / (missing + 1)
                    values = {
                        key: previous.values()[key] + fraction * (reading.values()[key] - previous.values()[key])
                        for key in previous.values()
                    }
                    processed = SensorReading(
                        previous.timestamp + timedelta(seconds=target_interval_seconds * step),
                        values["vibration_mps2"], values["temperature_c"], values["current_ampere"],
                    )
                    output.append(ProcessedReading(
                        None, processed, True, target_interval_seconds, True,
                        gap, float(target_interval_seconds),
                    ))
            elif missing and gap > max_gap_seconds and large_gap_policy == "reject":
                raise ValueError(f"Sampling gap of {gap / 60.0:.1f} minutes exceeds configured maximum")
        available = not (index and gap > maximum_interpolation_gap_minutes * 60.0)
        output.append(ProcessedReading(
            reading, reading, False, gap, available,
            gap, float(target_interval_seconds),
        ))
    return output
