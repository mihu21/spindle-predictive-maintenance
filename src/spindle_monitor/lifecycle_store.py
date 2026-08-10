from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .models import LifecycleRecord


LIFECYCLE_FIELDS = [
    "lifecycle_id", "start_timestamp", "end_timestamp", "duration_hours",
    "highest_status", "first_warning_timestamp", "first_critical_timestamp",
    "first_raw_warning_timestamp", "first_confirmed_warning_timestamp",
    "first_raw_critical_timestamp", "first_confirmed_critical_timestamp",
    "critical_reached", "inferred_reset_timestamp", "reset_confidence",
    "reset_reason", "pre_reset_vibration", "post_reset_vibration",
    "pre_reset_temperature", "post_reset_temperature", "pre_reset_current",
    "post_reset_current",
    "operating_regime", "degradation_family",
]


def lifecycle_to_row(record: LifecycleRecord) -> dict[str, Any]:
    return {
        "lifecycle_id": record.lifecycle_id,
        "start_timestamp": record.start_timestamp.isoformat(),
        "end_timestamp": record.end_timestamp.isoformat(),
        "duration_hours": round(record.duration_hours, 4),
        "highest_status": record.highest_status,
        "first_warning_timestamp": record.first_warning_timestamp.isoformat() if record.first_warning_timestamp else "",
        "first_critical_timestamp": record.first_critical_timestamp.isoformat() if record.first_critical_timestamp else "",
        "first_raw_warning_timestamp": record.first_raw_warning_timestamp.isoformat() if record.first_raw_warning_timestamp else "",
        "first_confirmed_warning_timestamp": record.first_confirmed_warning_timestamp.isoformat() if record.first_confirmed_warning_timestamp else "",
        "first_raw_critical_timestamp": record.first_raw_critical_timestamp.isoformat() if record.first_raw_critical_timestamp else "",
        "first_confirmed_critical_timestamp": record.first_confirmed_critical_timestamp.isoformat() if record.first_confirmed_critical_timestamp else "",
        "critical_reached": record.critical_reached,
        "inferred_reset_timestamp": record.inferred_reset_timestamp.isoformat(),
        "reset_confidence": record.reset_confidence,
        "reset_reason": record.reset_reason,
        "pre_reset_vibration": record.pre_reset_values.get("vibration_mps2"),
        "post_reset_vibration": record.post_reset_values.get("vibration_mps2"),
        "pre_reset_temperature": record.pre_reset_values.get("temperature_c"),
        "post_reset_temperature": record.post_reset_values.get("temperature_c"),
        "pre_reset_current": record.pre_reset_values.get("current_ampere"),
        "post_reset_current": record.post_reset_values.get("current_ampere"),
        "operating_regime": record.operating_regime,
        "degradation_family": record.degradation_family,
    }


class LifecycleCSVStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.file = self.path.open("w", encoding="utf-8-sig", newline="")
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot overwrite {self.path}. Close it in Excel or choose another filename."
            ) from exc
        self.writer = csv.DictWriter(self.file, fieldnames=LIFECYCLE_FIELDS)
        self.writer.writeheader()

    def save(self, record: LifecycleRecord) -> None:
        self.writer.writerow(lifecycle_to_row(record))
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()
