from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .anomaly import invalid_anomaly_result
from .config import ProjectConfig
from .models import AnomalyType, SensorReading
from .monitor import ConditionMonitor
from .validation import InputValidator


START = datetime(2026, 1, 1)


def _reading(minute: int, vibration: float = 1.0, temperature: float = 30.0,
             current: float = 4.0) -> SensorReading:
    return SensorReading(START + timedelta(minutes=minute), vibration, temperature, current)


def deterministic_fixtures() -> list[dict[str, Any]]:
    baseline = [_reading(index) for index in range(6)]
    noise = [_reading(index) for index in range(40)] + [
        _reading(index, vibration=1.0 + (0.35 if index % 2 else -0.35))
        for index in range(40, 66)
    ]
    return [
        {"name": "normal_highly_constant_temperature", "expected": "NONE", "normal": True,
         "readings": [_reading(index, temperature=31.2) for index in range(241)]},
        {"name": "normal_constant_current_fixed_load", "expected": "NONE", "normal": True,
         "readings": [_reading(index, current=4.25) for index in range(181)]},
        {"name": "rounded_constant_vibration", "expected": "NONE", "normal": True,
         "readings": [_reading(index, vibration=1.20) for index in range(181)]},
        {"name": "isolated_vibration_spike", "expected": "SINGLE_SAMPLE_SPIKE", "normal": False,
         "readings": baseline + [_reading(6, vibration=4.8), _reading(7), _reading(8), _reading(9)]},
        {"name": "persistent_vibration_increase", "expected": "PERSISTENT_STEP_CHANGE", "normal": False,
         "readings": baseline + [_reading(index, vibration=3.5) for index in range(6, 11)]},
        {"name": "multi_sensor_abrupt_deterioration", "expected": "CORRELATED_ABRUPT_CHANGE", "normal": False,
         "readings": baseline + [_reading(6, vibration=3.5, temperature=30.0, current=6.0)]},
        {"name": "correlation_window_boundary", "expected": "CORRELATED_ABRUPT_CHANGE", "normal": False,
         "readings": baseline + [
             _reading(index, vibration=3.5, current=(6.0 if index == 11 else 4.0))
             for index in range(6, 12)
         ]},
        {"name": "correlation_evidence_expired", "expected": "PERSISTENT_STEP_CHANGE", "normal": False,
         "forbidden": "CORRELATED_ABRUPT_CHANGE",
         "readings": baseline + [
             _reading(index, vibration=3.5, current=(6.0 if index == 12 else 4.0))
             for index in range(6, 13)
         ]},
        {"name": "sensor_stuck_during_known_load_change", "expected": "STUCK_SENSOR_SUSPECTED", "normal": False,
         "readings": [
             _reading(index, vibration=1.0, current=(4.0 if index < 61 else 5.0 if index < 121 else 6.0))
             for index in range(125)
         ]},
        {"name": "temperature_stable_within_thermal_delay", "expected": "POSSIBLE_STUCK_SENSOR", "normal": True,
         "readings": [
             _reading(index, temperature=31.0, current=(4.0 if index < 241 else 5.0))
             for index in range(245)
         ]},
        {"name": "temperature_fails_after_response_delay", "expected": "STUCK_SENSOR_SUSPECTED", "normal": False,
         "readings": [
             _reading(index, temperature=31.0, current=(4.0 if index < 241 else 5.0 if index < 421 else 6.0))
             for index in range(485)
         ]},
        {"name": "sensor_clipping_at_maximum", "expected": "CLIPPING_SUSPECTED", "normal": False,
         "readings": baseline + [_reading(index, vibration=20.0) for index in range(6, 9)]},
        {"name": "excessive_noise", "expected": "EXCESSIVE_NOISE", "normal": False,
         "readings": noise},
        {"name": "long_dropout", "expected": "LONG_DATA_GAP", "normal": False,
         "readings": baseline + [_reading(20)]},
        {"name": "sensor_drift_suspicion", "expected": "DRIFT_SUSPECTED", "normal": False,
         "readings": [_reading(index, vibration=1.0 + index * 0.025) for index in range(45)],
         "anomaly_overrides": {"drift_window_samples": 40, "drift_minimum_span_fraction": 0.25}},
        {"name": "real_gradual_degradation", "expected": "NONE", "normal": True,
         "readings": [_reading(index, vibration=1.0 + index * 0.005) for index in range(60)]},
    ]


def evaluate_anomaly_fixtures(config: ProjectConfig, output: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fixture in deterministic_fixtures():
        anomaly = replace(config.anomaly, enabled=True, **fixture.get("anomaly_overrides", {}))
        configured = replace(config, anomaly=anomaly)
        monitor = ConditionMonitor(configured, enable_ml=False)
        results = [monitor.process(value) for value in fixture["readings"]]
        observed = {
            item.value for result in results for item in result.anomaly.anomaly_type
            if item != AnomalyType.NONE
        }
        expected = fixture["expected"]
        forbidden = fixture.get("forbidden")
        passed = ((not observed) if expected == "NONE" else expected in observed) and (
            forbidden is None or forbidden not in observed
        )
        representative = next(
            (result for result in reversed(results)
             if expected in {item.value for item in result.anomaly.anomaly_type}),
            results[-1],
        )
        rows.append({
            "fixture_name": fixture["name"],
            "expected_classification": expected,
            "actual_classifications": sorted(observed) or ["NONE"],
            "quality_status": representative.anomaly.quality_status.value,
            "safety_action": representative.anomaly.safety_action,
            "model_action": representative.anomaly.model_action.value,
            "pass": passed,
            "normal_fixture": fixture["normal"],
            "forbidden_classification": forbidden,
        })

    # Safe interpolation and malformed input exercise paths that do not arise from
    # a complete SensorReading sequence alone.
    configured = replace(config, anomaly=replace(config.anomaly, enabled=True))
    monitor = ConditionMonitor(configured, enable_ml=False)
    monitor.process(_reading(0))
    short_disabled = monitor.process(_reading(3))
    rows.append({
        "fixture_name": "short_dropout_interpolation_disabled",
        "expected_classification": "SHORT_DATA_GAP",
        "actual_classifications": [item.value for item in short_disabled.anomaly.anomaly_type],
        "quality_status": short_disabled.anomaly.quality_status.value,
        "safety_action": short_disabled.anomaly.safety_action,
        "model_action": short_disabled.anomaly.model_action.value,
        "pass": (
            AnomalyType.SHORT_DATA_GAP in short_disabled.anomaly.anomaly_type
            and short_disabled.anomaly.estimated_missing_sample_count == 2
            and not short_disabled.anomaly.interpolation_occurred
        ),
        "normal_fixture": False, "forbidden_classification": None,
    })
    interpolating = replace(
        configured, interpolation_enabled=True,
        anomaly=replace(configured.anomaly, enabled=True),
    )
    monitor = ConditionMonitor(interpolating, enable_ml=False)
    monitor.process(_reading(0))
    short = monitor.process(
        _reading(1), interpolated=True, source_sampling_interval_seconds=180,
        effective_resampling_interval_seconds=60,
    )
    rows.append({
        "fixture_name": "short_dropout_interpolation_enabled", "expected_classification": "SHORT_DATA_GAP",
        "actual_classifications": [item.value for item in short.anomaly.anomaly_type],
        "quality_status": short.anomaly.quality_status.value,
        "safety_action": short.anomaly.safety_action,
        "model_action": short.anomaly.model_action.value,
        "pass": AnomalyType.SHORT_DATA_GAP in short.anomaly.anomaly_type,
        "normal_fixture": False, "forbidden_classification": None,
    })
    validation = InputValidator(configured).validate(
        SensorReading(START, float("nan"), 30.0, 4.0), raw_row={"vibration_mps2": "NaN"}
    )
    invalid = invalid_anomaly_result(validation)
    rows.append({
        "fixture_name": "invalid_value", "expected_classification": "MALFORMED_VALUE",
        "actual_classifications": [item.value for item in invalid.anomaly_type],
        "quality_status": invalid.quality_status.value, "safety_action": invalid.safety_action,
        "model_action": invalid.model_action.value,
        "pass": AnomalyType.MALFORMED_VALUE in invalid.anomaly_type,
        "normal_fixture": False, "forbidden_classification": None,
    })
    false_positives = [row["fixture_name"] for row in rows if row["normal_fixture"] and not row["pass"]]
    false_negatives = [row["fixture_name"] for row in rows if not row["normal_fixture"] and not row["pass"]]
    report = {
        "detector_version": config.anomaly.detector_version,
        "threshold_origin": config.anomaly.threshold_origin,
        "all_passed": not false_positives and not false_negatives,
        "fixture_count": len(rows),
        "false_positive_results_on_normal_constant_data": false_positives,
        "false_negative_results_on_fault_fixtures": false_negatives,
        "fixtures": rows,
        "remaining_limitations": [
            "Defaults are not plant calibrated.",
            "One censored trajectory from one machine cannot establish population behavior.",
            "No automatic root-cause diagnosis, recalibration, long-period reconstruction, or spoofing detector.",
        ],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
