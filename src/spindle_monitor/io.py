from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .config import ProjectConfig
from .models import SensorReading, ValidationResult
from .validation import InputValidator


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Unsupported timestamp format: {value!r}") from error


def _reading_from_row(row: dict[str, str]) -> SensorReading:
    return SensorReading(
        timestamp=parse_timestamp(row["timestamp"]),
        vibration_mps2=float(row["vibration_mps2"]),
        temperature_c=float(row["temperature_c"]),
        current_ampere=float(row["current_ampere"]),
    )


def read_csv_records(
    path: str | Path,
    config: ProjectConfig,
) -> Iterator[ValidationResult]:
    path = Path(path)
    validator = InputValidator(config)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "timestamp", "vibration_mps2", "temperature_c", "current_ampere",
            "health_status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            label = row.get("health_status")
            normalized_label = label.strip().lower() if label else None
            try:
                reading = _reading_from_row(row)
            except (TypeError, ValueError, KeyError) as exc:
                yield ValidationResult(
                    False,
                    f"invalid CSV row: {exc}",
                    row_number,
                    dict(row),
                    source_label=normalized_label,
                )
                continue
            yield validator.validate(
                reading,
                row_number=row_number,
                raw_row=dict(row),
                source_label=normalized_label,
            )


def read_csv_readings(path: str | Path) -> Iterator[tuple[SensorReading, str | None]]:
    """Backward-compatible strict CSV reader.

    It validates parsing only. The upgraded replay command uses read_csv_records,
    which audits and skips invalid rows instead of aborting the complete run.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "timestamp", "vibration_mps2", "temperature_c", "current_ampere",
            "health_status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                reading = _reading_from_row(row)
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"Invalid CSV row {row_number}: {error}") from error
            label = row.get("health_status")
            yield reading, label.strip().lower() if label else None
