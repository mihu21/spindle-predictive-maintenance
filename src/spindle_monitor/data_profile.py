from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .io import parse_timestamp, read_csv_records
from .model_registry import ModelRegistry
from .offline import prepare_offline_replay


INPUT_COLUMNS = [
    "timestamp", "vibration_mps2", "temperature_c", "current_ampere", "health_status"
]
SENSOR_COLUMNS = INPUT_COLUMNS[1:4]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("minimum", "p01", "p05", "p25", "median", "p75", "p95", "p99", "maximum")}
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)), "p01": float(np.quantile(array, .01)),
        "p05": float(np.quantile(array, .05)), "p25": float(np.quantile(array, .25)),
        "median": float(np.median(array)), "p75": float(np.quantile(array, .75)),
        "p95": float(np.quantile(array, .95)), "p99": float(np.quantile(array, .99)),
        "maximum": float(np.max(array)),
    }


def _sensor_statistics(values: list[float]) -> dict[str, float | None]:
    result = _percentiles(values)
    if values:
        result.update(mean=float(np.mean(values)), standard_deviation=float(np.std(values)))
        differences = np.diff(np.asarray(values, dtype=float))
        if len(differences):
            center = float(np.median(differences))
            mad = float(np.median(np.abs(differences - center)))
            # For independent adjacent measurement errors, diff sigma is sqrt(2)
            # times the per-reading sigma. MAD avoids treating degradation and
            # regime shifts as if all lifecycle variance were measurement noise.
            result["measurement_noise_estimate"] = 1.4826 * mad / math.sqrt(2.0)
        else:
            result["measurement_noise_estimate"] = None
    else:
        result.update(mean=None, standard_deviation=None, measurement_noise_estimate=None)
    result.pop("p25", None)
    result.pop("p75", None)
    return result


def _longest_status_runs(timestamps: list[datetime], statuses: list[str], median_seconds: float) -> dict[str, float]:
    longest: dict[str, float] = {"normal": 0.0, "warning": 0.0, "critical": 0.0}
    if not timestamps:
        return longest
    start = 0
    for index in range(1, len(statuses) + 1):
        if index == len(statuses) or statuses[index] != statuses[start]:
            duration = (timestamps[index - 1] - timestamps[start]).total_seconds() + median_seconds
            key = statuses[start]
            longest[key] = max(longest.get(key, 0.0), duration / 3600.0)
            start = index
    return longest


def _model_comparison(models_root: str | Path, sensors: dict[str, list[float]]) -> dict[str, Any]:
    registry = ModelRegistry(models_root)
    metadata = registry.load_metadata("production") or registry.load_metadata("candidate")
    if metadata is None:
        return {"available": False, "reason": "No selected model metadata exists."}
    names = metadata.get("feature_names", [])
    distribution = metadata.get("feature_distribution", {})
    lower = distribution.get("lower_quantile_01", [])
    upper = distribution.get("upper_quantile_99", [])
    comparison: dict[str, Any] = {}
    for sensor, values in sensors.items():
        feature = f"{sensor}__raw"
        if feature in names and len(lower) == len(names) == len(upper) and values:
            index = names.index(feature)
            comparison[sensor] = {
                "observed_minimum": min(values), "observed_maximum": max(values),
                "training_p01": lower[index], "training_p99": upper[index],
                "outside_training_range": min(values) < lower[index] or max(values) > upper[index],
            }
    return {
        "available": True,
        "model_version": metadata.get("model_version"),
        "data_domain": metadata.get("data_domain", "legacy_unclassified"),
        "sensors": comparison,
    }


def _input_validation_summary(
    path: Path,
    config: ProjectConfig,
    rows: list[dict[str, str]],
    missing_columns: list[str],
) -> dict[str, Any]:
    """Use the authoritative replay validator and retain actionable examples."""
    if missing_columns:
        reason = f"CSV is missing required columns: {sorted(missing_columns)}"
        return {
            "total_rows": len(rows),
            "valid_rows": 0,
            "invalid_rows": len(rows),
            "invalid_reason_counts": {reason: len(rows)},
            "representative_failures": {
                reason: [{"row_number": 2, "timestamp": rows[0].get("timestamp", "")}] if rows else []
            },
        }
    validations = list(read_csv_records(path, config))
    invalid = [value for value in validations if not value.valid]
    counts = Counter(value.reason for value in invalid)
    representatives: dict[str, list[dict[str, Any]]] = {}
    for value in invalid:
        examples = representatives.setdefault(value.reason, [])
        if len(examples) >= 3:
            continue
        raw = value.raw_row or {}
        examples.append({
            "row_number": value.row_number,
            "timestamp": raw.get("timestamp", ""),
        })
    return {
        "total_rows": len(validations),
        "valid_rows": sum(value.valid for value in validations),
        "invalid_rows": len(invalid),
        "invalid_reason_counts": dict(sorted(counts.items())),
        "representative_failures": dict(sorted(representatives.items())),
    }


def profile_data(
    input_path: str | Path,
    config: ProjectConfig,
    *,
    models_root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    missing_columns = [value for value in INPUT_COLUMNS if value not in columns]
    extra_columns = [value for value in columns if value not in INPUT_COLUMNS]
    timestamps: list[datetime] = []
    statuses: list[str] = []
    sensors: dict[str, list[float]] = {key: [] for key in SENSOR_COLUMNS}
    missing_sensors = Counter()
    invalid_numeric = Counter()
    invalid_timestamps = 0
    invalid_status_values = Counter()
    for row in rows:
        try:
            timestamps.append(parse_timestamp(row.get("timestamp", "")))
        except (TypeError, ValueError):
            invalid_timestamps += 1
            continue
        status = (row.get("health_status") or "").strip().lower()
        statuses.append(status)
        if status not in {"normal", "warning", "critical"}:
            invalid_status_values[status or "<blank>"] += 1
        for sensor in SENSOR_COLUMNS:
            value = row.get(sensor)
            if value is None or not value.strip():
                missing_sensors[sensor] += 1
                continue
            try:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError
                sensors[sensor].append(numeric)
            except ValueError:
                invalid_numeric[sensor] += 1
    input_validation = _input_validation_summary(path, config, rows, missing_columns)
    intervals = [(right - left).total_seconds() for left, right in zip(timestamps, timestamps[1:])]
    positive = [value for value in intervals if value > 0]
    median_seconds = float(np.median(positive)) if positive else 0.0
    missing_timestamp_count = int(sum(max(0, round(value / median_seconds) - 1) for value in positive)) if median_seconds else 0
    duplicates = sum(value == 0 for value in intervals)
    out_of_order = sum(value < 0 for value in intervals)
    contract_valid = not missing_columns and not extra_columns and columns == INPUT_COLUMNS
    plan_records = []
    lifecycle_error = None
    if not missing_columns:
        try:
            _, plan = prepare_offline_replay(path, config, models_root or path.parent / ".profile_models")
            plan_records = list(plan.records)
        except Exception as exc:  # report invalid input without mutating any registry
            lifecycle_error = str(exc)
    final_status = statuses[-1] if statuses else "invalid"
    open_state = f"open_{final_status}" if final_status in {"normal", "warning", "critical"} else "invalid"
    open_count = 1 if rows else 0
    reasons: list[str] = []
    warnings: list[str] = []
    if not contract_valid:
        reasons.append(f"Five-column contract mismatch; missing={missing_columns}, extra={extra_columns}.")
    if invalid_timestamps or invalid_numeric or missing_sensors:
        reasons.append("Invalid timestamp or numeric values are present.")
    if duplicates or out_of_order:
        reasons.append("Timestamp ordering is not valid for causal replay/training.")
    if invalid_status_values:
        reasons.append(f"Invalid health_status values are present: {dict(invalid_status_values)}.")
    completed = len(plan_records)
    required = config.ml.minimum_completed_lifecycles_for_training
    if completed < required:
        warnings.append(f"Dataset has only {completed} completed lifecycle(s); minimum required for training is {required}.")
    if open_count:
        warnings.append(f"The final lifecycle is {open_state} and censored; it is not an exact remaining-time target.")
    common_refusals: list[dict[str, Any]] = []
    def refuse(code: str, message: str, count: int | None = None) -> None:
        item: dict[str, Any] = {"code": code, "message": message}
        if count is not None:
            item["count"] = int(count)
        common_refusals.append(item)
    if missing_columns:
        refuse("missing_required_columns", "Required CSV columns are missing.", len(missing_columns))
    if extra_columns or columns != INPUT_COLUMNS:
        refuse("column_contract_mismatch", "CSV columns do not exactly match the preserved five-column contract.")
    if invalid_timestamps:
        refuse("invalid_timestamps", "One or more timestamps cannot be parsed.", invalid_timestamps)
    if duplicates:
        refuse("duplicate_timestamps", "Duplicate timestamps make causal replay ambiguous.", duplicates)
    if out_of_order:
        refuse("non_monotonic_timestamps", "Timestamps are not monotonically increasing.", out_of_order)
    if invalid_status_values:
        refuse("invalid_status_values", "health_status contains blank or unknown values.", sum(invalid_status_values.values()))
    if any(invalid_numeric.values()):
        refuse("non_numeric_sensor_values", "Required sensor columns contain non-numeric values.", sum(invalid_numeric.values()))
    if any(missing_sensors.values()):
        refuse("missing_sensor_values", "Required sensor values are missing.", sum(missing_sensors.values()))
    if len(timestamps) > 1 and not positive:
        refuse("invalid_sampling_intervals", "No positive sampling interval can be established.")
    for validation_reason, count in input_validation["invalid_reason_counts"].items():
        refuse(
            "input_validator_rejection",
            f"Replay InputValidator rejected row(s): {validation_reason}",
            int(count),
        )
    suitable_replay = not common_refusals
    suitable_features = suitable_replay
    suitable_training = suitable_features and completed >= required
    validation_minimum = config.ml.minimum_validation_lifecycles + config.ml.minimum_test_lifecycles
    suitable_validation = suitable_training and completed >= max(required, validation_minimum)
    suitable_promotion = suitable_validation and completed >= config.ml.minimum_completed_lifecycles_for_promotion
    result: dict[str, Any] = {
        "input_path": str(path.resolve()), "sha256": file_sha256(path), "row_count": len(rows),
        "column_names": columns,
        "contract_validation": {
            "valid": contract_valid, "expected_columns": INPUT_COLUMNS,
            "missing_columns": missing_columns, "extra_columns": extra_columns,
        },
        "start_timestamp": timestamps[0].isoformat() if timestamps else None,
        "end_timestamp": timestamps[-1].isoformat() if timestamps else None,
        "total_duration_hours": ((timestamps[-1] - timestamps[0]).total_seconds() / 3600.0) if len(timestamps) > 1 else 0.0,
        "sampling_interval_seconds": _percentiles(positive),
        "duplicate_timestamps": duplicates, "out_of_order_timestamps": out_of_order,
        "missing_timestamps": missing_timestamp_count, "invalid_timestamps": invalid_timestamps,
        "invalid_sampling_intervals": duplicates + out_of_order,
        "missing_sensor_values": dict(missing_sensors), "invalid_numeric_values": dict(invalid_numeric),
        "invalid_status_values": dict(invalid_status_values),
        "input_validation": input_validation,
        "total_invalid_rows": input_validation["invalid_rows"],
        "sensor_statistics": {key: _sensor_statistics(value) for key, value in sensors.items()},
        "health_status_counts": dict(Counter(statuses)),
        "raw_status_transition_count": sum(a != b for a, b in zip(statuses, statuses[1:])),
        "first_raw_warning_timestamp": next((timestamps[i].isoformat() for i, value in enumerate(statuses) if value == "warning"), None),
        "first_raw_critical_timestamp": next((timestamps[i].isoformat() for i, value in enumerate(statuses) if value == "critical"), None),
        "longest_continuous_status_hours": _longest_status_runs(timestamps, statuses, median_seconds),
        "detected_lifecycle_boundaries": completed, "completed_lifecycles": completed,
        "open_or_censored_lifecycles": open_count,
        "lifecycle_states": {open_state: open_count, "completed_with_reset": completed},
        "suitability": {
            "replay": suitable_replay, "feature_extraction": suitable_features,
            "training": suitable_training, "validation": suitable_validation,
            "promotion_evaluation": suitable_promotion,
        },
        "suitability_refusal_reasons": {
            "replay": list(common_refusals),
            "feature_extraction": list(common_refusals),
            "training": list(common_refusals) + ([] if completed >= required else [{
                "code": "insufficient_completed_lifecycles",
                "message": f"Training requires at least {required} completed lifecycles.",
                "count": completed,
            }]),
            "validation": list(common_refusals) + ([] if completed >= max(required, validation_minimum) else [{
                "code": "insufficient_validation_lifecycles",
                "message": "There are too few completed lifecycles for validation and untouched testing.",
                "count": completed,
            }]),
            "promotion_evaluation": list(common_refusals) + ([] if completed >= config.ml.minimum_completed_lifecycles_for_promotion else [{
                "code": "insufficient_promotion_lifecycles",
                "message": "There are too few completed lifecycles for promotion evaluation.",
                "count": completed,
            }]),
        },
        "rejection_reasons": reasons, "warning_reasons": warnings,
    }
    if lifecycle_error:
        result["lifecycle_detection_error"] = lifecycle_error
    if models_root is not None:
        result["model_training_distribution_comparison"] = _model_comparison(models_root, sensors)
    return result


def write_profile(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
