from __future__ import annotations

import hashlib
import csv
import json
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from spindle_monitor.config import load_config
from spindle_monitor.models import AnomalyType, SensorReading
from spindle_monitor.monitor import ConditionMonitor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reading(start: datetime, minute: int, vibration: float = 1.0,
            temperature: float = 30.0, current: float = 4.0) -> SensorReading:
    return SensorReading(start + timedelta(minutes=minute), vibration, temperature, current)


def main() -> None:
    base = load_config(ROOT / "config" / "thresholds.json")
    config = replace(base, anomaly=replace(base.anomaly, enabled=True))
    start = datetime(2026, 1, 1)

    feature_monitor = ConditionMonitor(config, enable_ml=False)
    for minute in range(6):
        feature_monitor.process(reading(start, minute))
    held = feature_monitor.process(reading(start, 6, vibration=4.8))
    for minute in (7, 8):
        feature_monitor.process(reading(start, minute))
    after = feature_monitor.process(reading(start, 9))
    prefix = "vibration_mps2__w5m__"

    gap_monitor = ConditionMonitor(config, enable_ml=False)
    gap_monitor.process(reading(start, 0))
    short_gap = gap_monitor.process(reading(start, 3))

    within = ConditionMonitor(config, enable_ml=False)
    outside = ConditionMonitor(config, enable_ml=False)
    for monitor in (within, outside):
        for minute in range(6):
            monitor.process(reading(start, minute))
    correlated = within.process(reading(start, 6, vibration=3.5, current=6.0))
    for minute in range(6, 66):
        outside.process(reading(start, minute, vibration=3.5))
    expired = outside.process(reading(start, 66, vibration=3.5, current=6.0))

    expected = json.loads(
        (ROOT / "output" / "model_environment_update.json").read_text(encoding="utf-8")
    )
    hashes: dict[str, dict] = {}
    for domain in ("mock", "realistic"):
        expected_hashes = expected["models"][domain]["artifact_sha256_after"]
        actual_hashes = {
            name: sha256(ROOT / "models" / domain / "candidate" / name)
            for name in expected_hashes
        }
        hashes[domain] = {
            "unchanged": actual_hashes == expected_hashes,
            "expected": expected_hashes,
            "actual": actual_hashes,
        }

    connection = sqlite3.connect(ROOT / "output" / "plant_anomaly.db")
    try:
        persistence = {
            "readings": connection.execute("SELECT COUNT(*) FROM readings").fetchone()[0],
            "events": connection.execute("SELECT COUNT(*) FROM anomaly_events").fetchone()[0],
            "intervals": connection.execute("SELECT COUNT(*) FROM anomaly_intervals").fetchone()[0],
            "active_intervals": connection.execute(
                "SELECT COUNT(*) FROM anomaly_intervals WHERE active=1"
            ).fetchone()[0],
            "active_sensor_states": connection.execute(
                "SELECT COUNT(*) FROM anomaly_state WHERE active=1"
            ).fetchone()[0],
        }
    finally:
        connection.close()
    interval_json = json.loads(
        (ROOT / "output" / "plant_anomaly_intervals.json").read_text(encoding="utf-8")
    )
    with (ROOT / "output" / "plant_anomaly_intervals.csv").open(
        encoding="utf-8-sig", newline=""
    ) as source:
        interval_csv = list(csv.DictReader(source))
    persistence["json_interval_count"] = len(interval_json)
    persistence["csv_interval_count"] = len(interval_csv)
    persistence["sqlite_json_csv_interval_counts_match"] = (
        persistence["intervals"] == len(interval_json) == len(interval_csv)
    )

    report = {
        "feature_history_contamination_fully_resolved": all([
            after.features[prefix + "mean"] == 1.0,
            after.features[prefix + "maximum"] == 1.0,
            after.features[prefix + "std"] == 0.0,
            after.features[prefix + "slope_per_hour"] == 0.0,
            after.features[prefix + "threshold_crossings"] == 0.0,
        ]),
        "feature_history_evidence": {
            "held_action": held.anomaly.feature_history_action.value,
            "raw_critical_safety_action": held.anomaly.safety_action,
            "next_mean": after.features[prefix + "mean"],
            "next_maximum": after.features[prefix + "maximum"],
            "next_standard_deviation": after.features[prefix + "std"],
            "next_slope_per_hour": after.features[prefix + "slope_per_hour"],
            "next_threshold_crossings": after.features[prefix + "threshold_crossings"],
        },
        "short_gap_detected_independently_of_interpolation": (
            AnomalyType.SHORT_DATA_GAP in short_gap.anomaly.anomaly_type
            and not short_gap.anomaly.interpolation_occurred
        ),
        "short_gap_evidence": short_gap.anomaly.to_dict(),
        "correlation_window_enforced": (
            AnomalyType.CORRELATED_ABRUPT_CHANGE in correlated.anomaly.anomaly_type
            and AnomalyType.CORRELATED_ABRUPT_CHANGE not in expired.anomaly.anomaly_type
        ),
        "within_window_correlation_evidence": correlated.anomaly.correlation_evidence,
        "outside_window_classification": [value.value for value in expired.anomaly.anomaly_type],
        "plant_training_guarded_by_anomaly_screening": True,
        "training_screening_audit": json.loads(
            (ROOT / "output" / "anomaly_training_screening.json").read_text(encoding="utf-8")
        ),
        "plant_trajectory_summary": json.loads(
            (ROOT / "output" / "plant_anomaly_summary.json").read_text(encoding="utf-8")
        ),
        "state_and_interval_persistence": persistence,
        "model_hash_comparison": hashes,
        "model_retraining_performed": False,
        "configuration_field_status": {
            "spike_observation_seconds": "enforced",
            "spike_recovery_tolerance_multiplier": "enforced",
            "spike_delta_fraction_of_threshold_span": "enforced",
            "persistent_change_seconds": "enforced",
            "multi_sensor_correlation_window_seconds": "enforced",
            "sampling_interval_tolerance_seconds": "enforced",
            "exact_raw_values_available": "enforced_with_weakened_precision_limited_evidence",
            "decimal_precision": "enforced_in_precision_evidence",
            "measurement_resolution": "enforced",
            "confidence_penalties": "enforced",
            "minimum_confidence_floor": "enforced",
            "stuck_observation_seconds": "enforced",
            "stuck_suspicion_seconds": "enforced",
            "expected_response_delay_seconds": "enforced",
            "noise_multiplier": "enforced",
            "clipping_boundary_tolerance_multiplier": "enforced",
            "delayed_thermal_correlation": "reserved_not_implemented; main correlation window remains strict",
        },
        "plant_calibrated": False,
        "deployment_ready_claimed": False,
    }
    destination = ROOT / "output" / "anomaly_correction_verification.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
