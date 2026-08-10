from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .data_profile import INPUT_COLUMNS, file_sha256
from .models import Status
from .status_policy import TimeBasedEventPolicy


GENERATOR_VERSION = "realistic_spindle_v4_load_degradation_independent"

# The bands deliberately overlap at their edges. Durations are sampled, never
# copied from a fixed template, and a lifecycle remains an indivisible unit.
DURATION_BANDS_HOURS: tuple[tuple[float, float], ...] = (
    (36.0, 71.5),
    (72.0, 167.5),
    (168.0, 299.0),
    (300.0, 499.0),
    (500.0, 749.0),
    (750.0, 999.0),
    (1000.0, 1250.0),
)
OPERATING_REGIMES = ("low_load", "medium_load", "high_load", "variable_mixed_load")
DEGRADATION_FAMILIES = ("gradual_multi_sensor", "rapid_multi_sensor", "vibration_leads", "thermal_leads")


def _profile_scale(profile: dict[str, Any], sensor: str, fallback: float) -> float:
    value = profile.get("sensor_statistics", {}).get(sensor, {}).get("measurement_noise_estimate")
    return max(float(value or fallback), fallback)


def _duration_plan_with_bands(
    lifecycle_count: int, rng: np.random.Generator,
) -> list[tuple[float, int]]:
    """Cover every duration band before repeating, preserving band evidence."""
    band_indices = [index % len(DURATION_BANDS_HOURS) for index in range(lifecycle_count)]
    rng.shuffle(band_indices)
    return [
        (float(rng.uniform(*DURATION_BANDS_HOURS[index])), int(index))
        for index in band_indices
    ]


def _duration_plan(lifecycle_count: int, rng: np.random.Generator) -> list[float]:
    """Backward-compatible duration-only view used by tests and callers."""
    return [value for value, _ in _duration_plan_with_bands(lifecycle_count, rng)]


def generate_realistic_mock(
    profile_path: str | Path,
    lifecycle_count: int,
    output_path: str | Path,
    metadata_output: str | Path,
    config: ProjectConfig,
    *,
    seed: int = 42,
    suite_role: str = "development",
) -> dict[str, Any]:
    """Generate hard-negative and varied hard-positive complete lifecycles.

    `suite_role=acceptance` is recorded in the sidecar and is rejected by the
    training entry point. It does not alter physics, only the audit contract.
    """
    if lifecycle_count < 1:
        raise ValueError("--lifecycles must be at least 1")
    if suite_role not in {"development", "acceptance"}:
        raise ValueError("suite_role must be development or acceptance")
    profile_file = Path(profile_path)
    profile = json.loads(profile_file.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    start_text = profile.get("start_timestamp") or "2026-01-01T00:00:00"
    timestamp = datetime.fromisoformat(start_text)
    sample_seconds = int(profile.get("sampling_interval_seconds", {}).get("median") or 60)
    sample_seconds = 60 if sample_seconds <= 0 else sample_seconds
    # Alternation guarantees at least floor(N/2) hard negatives and positives.
    lifecycle_kinds = ["degradation" if index % 2 == 0 else "long_healthy" for index in range(lifecycle_count)]
    # Duration support is stratified by outcome. With at least 14 lifecycles,
    # both healthy and degrading cohorts cover every configured duration band.
    planned_by_kind = {
        kind: iter(_duration_plan_with_bands(lifecycle_kinds.count(kind), rng))
        for kind in ("degradation", "long_healthy")
    }
    duration_plan = [(*next(planned_by_kind[kind]), kind) for kind in lifecycle_kinds]
    lifecycle_records: list[dict[str, Any]] = []
    row_count = 0
    kind_counts = {"degradation": 0, "long_healthy": 0}

    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=INPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for lifecycle_index, (duration_hours, duration_band_index, lifecycle_kind) in enumerate(duration_plan, 1):
            lifecycle_id = f"lifecycle_{lifecycle_index:04d}"
            sample_count = max(2, int(round(duration_hours * 3600 / sample_seconds)))
            lifecycle_start = timestamp
            recipe_index = kind_counts[lifecycle_kind]
            kind_counts[lifecycle_kind] += 1
            operating_regime = OPERATING_REGIMES[recipe_index % len(OPERATING_REGIMES)]
            degradation_family = (
                "healthy_hard_negative" if lifecycle_kind == "long_healthy"
                else DEGRADATION_FAMILIES[recipe_index % len(DEGRADATION_FAMILIES)]
            )
            noise_multiplier = float(rng.uniform(.65, 1.35))
            shift_hours = float(rng.uniform(6.0, 12.0))
            daily_phase = float(rng.uniform(0, 2 * np.pi))
            benign_drift_scale = float(rng.uniform(-.025, .035))
            # Long failing machines spend most of life healthy. The remaining
            # interval is always long enough to establish confirmed events.
            degradation_start_fraction = float(rng.uniform(.55, .82))
            if duration_hours < 72:
                degradation_start_fraction = float(rng.uniform(.30, .55))
            degradation_power = float(rng.uniform(.72, 1.65))
            # Operating level is sampled independently of lifecycle outcome.
            # It occupies a physically safe fraction of baseline-to-WARNING,
            # allowing sustained high-load healthy examples instead of making
            # elevated absolute level exclusive to degrading lifecycles.
            regime_level = {
                "low_load": float(rng.uniform(.05, .24)),
                "medium_load": float(rng.uniform(.30, .56)),
                "high_load": float(rng.uniform(.67, .92)),
                "variable_mixed_load": float(rng.uniform(.28, .62)),
            }[operating_regime]
            sensor_load_response = {
                key: float(rng.uniform(.92, 1.08)) for key in config.sensors
            }
            baseline_offsets = {
                key: float(rng.uniform(-.055, .055)) for key in config.sensors
            }
            first_raw_warning = first_confirmed_warning = None
            first_raw_critical = first_confirmed_critical = None
            event_policy = TimeBasedEventPolicy(config.lifecycle)
            transient_center = float(rng.uniform(.18, .78))
            transient_width = float(rng.uniform(.004, .018))
            for index in range(sample_count):
                elapsed_hours = index * sample_seconds / 3600.0
                fraction = index / max(sample_count - 1, 1)
                shift_phase = 2 * np.pi * elapsed_hours / shift_hours
                daily = np.sin(2 * np.pi * elapsed_hours / 24.0 + daily_phase)
                if operating_regime == "low_load":
                    load = regime_level + .025 * np.sin(shift_phase)
                elif operating_regime == "medium_load":
                    load = regime_level + .035 * np.sin(shift_phase)
                elif operating_regime == "high_load":
                    load = regime_level + .028 * np.sin(shift_phase)
                else:
                    block = int(elapsed_hours // shift_hours) % 3
                    load = (.16, .50, .84)[block] + .045 * np.sin(shift_phase)
                benign_drift = benign_drift_scale * np.sin(np.pi * fraction) + .010 * daily
                transient = np.exp(-.5 * ((fraction - transient_center) / transient_width) ** 2)
                shared_noise = rng.normal(0.0, 1.0)
                if lifecycle_kind == "degradation" and fraction > degradation_start_fraction:
                    progress = (fraction - degradation_start_fraction) / (1.0 - degradation_start_fraction)
                    degradation = float(np.clip(progress, 0.0, 1.0) ** degradation_power)
                else:
                    degradation = 0.0
                values: dict[str, float] = {}
                for sensor_index, (key, sensor) in enumerate(config.sensors.items()):
                    span = sensor.critical - sensor.healthy_baseline
                    safe_span = sensor.warning - sensor.healthy_baseline
                    profile_noise = _profile_scale(profile, key, span * .006)
                    noise_scale = min(profile_noise, span * .035)
                    noise = noise_multiplier * noise_scale * (
                        .45 * shared_noise + .55 * rng.normal(0.0, 1.0)
                    )
                    value = sensor.healthy_baseline + baseline_offsets[key] * safe_span
                    value += load * sensor_load_response[key] * safe_span
                    value += benign_drift * safe_span + transient * safe_span * rng.uniform(.02, .06)
                    if lifecycle_kind == "degradation":
                        lead = 1.0
                        if degradation_family == "vibration_leads":
                            lead = 1.18 if key == "vibration_mps2" else .88
                        elif degradation_family == "thermal_leads":
                            lead = 1.18 if key == "temperature_c" else .88
                        endpoint = sensor.critical + rng.uniform(.12, .30) * span
                        value += degradation * lead * (endpoint - sensor.healthy_baseline)
                    else:
                        # Hard negatives approach limits under load/transients
                        # but never become manufacturer WARNING.
                        value = min(value, sensor.warning - max(.025 * span, 3 * noise_scale))
                    values[key] = float(np.clip(value + noise, sensor.valid_min, sensor.valid_max))
                status = "normal"
                if any(values[key] > sensor.warning for key, sensor in config.sensors.items()):
                    status = "warning"
                    first_raw_warning = first_raw_warning or timestamp
                if any(values[key] > sensor.critical for key, sensor in config.sensors.items()):
                    status = "critical"
                    first_raw_critical = first_raw_critical or timestamp
                confirmed_status = event_policy.update(timestamp, Status.from_text(status))
                if confirmed_status >= Status.WARNING:
                    first_confirmed_warning = first_confirmed_warning or timestamp
                if confirmed_status >= Status.CRITICAL:
                    first_confirmed_critical = first_confirmed_critical or timestamp
                writer.writerow({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    **{key: f"{values[key]:.6f}" for key in config.sensors},
                    "health_status": status,
                })
                row_count += 1
                timestamp += timedelta(seconds=sample_seconds)
            maintenance_timestamp = timestamp
            lifecycle_records.append({
                "lifecycle_id": lifecycle_id,
                "start_timestamp": lifecycle_start.isoformat(),
                "end_timestamp": maintenance_timestamp.isoformat(),
                "duration_hours": sample_count * sample_seconds / 3600.0,
                "duration_band_index": int(duration_band_index),
                "duration_band_hours": list(DURATION_BANDS_HOURS[duration_band_index]),
                "lifecycle_kind": lifecycle_kind,
                "highest_status": "CRITICAL" if first_confirmed_critical else "WARNING" if first_confirmed_warning else "NORMAL",
                "highest_raw_status": "CRITICAL" if first_raw_critical else "WARNING" if first_raw_warning else "NORMAL",
                "first_raw_warning_timestamp": first_raw_warning.isoformat() if first_raw_warning else None,
                "first_confirmed_warning_timestamp": first_confirmed_warning.isoformat() if first_confirmed_warning else None,
                "first_raw_critical_timestamp": first_raw_critical.isoformat() if first_raw_critical else None,
                "first_confirmed_critical_timestamp": first_confirmed_critical.isoformat() if first_confirmed_critical else None,
                "first_warning_timestamp": first_confirmed_warning.isoformat() if first_confirmed_warning else None,
                "first_critical_timestamp": first_confirmed_critical.isoformat() if first_confirmed_critical else None,
                "critical_reached": first_confirmed_critical is not None,
                "maintenance_record_timestamp": maintenance_timestamp.isoformat(),
                "lifecycle_state": "completed_with_maintenance_record",
                "operating_regime": operating_regime,
                "degradation_family": degradation_family,
                "degradation_start_fraction": degradation_start_fraction if lifecycle_kind == "degradation" else None,
            })
            # Explicit maintenance boundary; this sample belongs to the next
            # lifecycle in causal replay and is not included in this record.
            writer.writerow({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                **{key: f"{sensor.healthy_baseline:.6f}" for key, sensor in config.sensors.items()},
                "health_status": "normal",
            })
            row_count += 1
            timestamp += timedelta(seconds=sample_seconds)

    duration_values = np.asarray([value["duration_hours"] for value in lifecycle_records], dtype=float)
    metadata: dict[str, Any] = {
        "data_domain": "realistic_synthetic",
        "dataset_role": suite_role,
        "model_stage": "realistic_candidate",
        "production_eligible": False,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "source_profile_path": str(profile_file.resolve()),
        "source_profile_sha256": file_sha256(profile_file),
        "output_path": str(output.resolve()),
        "dataset_sha256": file_sha256(output),
        "row_count": row_count,
        "lifecycle_count": lifecycle_count,
        "healthy_lifecycle_count": sum(value["lifecycle_kind"] == "long_healthy" for value in lifecycle_records),
        "degradation_lifecycle_count": sum(value["lifecycle_kind"] == "degradation" for value in lifecycle_records),
        "outcome_regime_counts": {
            kind: {
                regime: sum(
                    value["lifecycle_kind"] == kind and value["operating_regime"] == regime
                    for value in lifecycle_records
                )
                for regime in OPERATING_REGIMES
            }
            for kind in ("long_healthy", "degradation")
        },
        "operating_regime_sampled_independently_of_outcome": True,
        "sampling_interval_seconds": sample_seconds,
        "duration_distribution_hours": {
            "minimum": float(np.min(duration_values)), "p50": float(np.median(duration_values)),
            "p90": float(np.quantile(duration_values, .9)), "maximum": float(np.max(duration_values)),
        },
        "lifecycles": lifecycle_records,
        "assumptions": {
            "inferred_from_profile": ["sampling interval", "measurement-noise scale"],
            "manually_configured": [
                "complete maintenance-delimited lifecycles", "outcome-stratified randomized duration bands",
                "healthy/failing mix", "outcome-independent safe operating levels", "load regimes",
                "shared/independent noise", "daily and shift cycles",
                "benign drift and recoverable transients", "late, gradual, rapid, and sensor-leading degradation",
            ],
            "hard_negative_rule": (
                "Healthy load spans low through sustained high-but-safe operation and is capped below "
                "manufacturer WARNING; lifecycle outcome is sampled separately and labels are never rewritten."
            ),
            "accuracy_limitation": "Synthetic data validates software behavior only and is not plant accuracy evidence.",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata
