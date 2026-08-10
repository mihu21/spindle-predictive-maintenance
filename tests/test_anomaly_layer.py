from __future__ import annotations

import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.anomaly import AnomalyDetector, invalid_anomaly_result
from spindle_monitor.anomaly_fixtures import evaluate_anomaly_fixtures
from spindle_monitor.config import load_config
from spindle_monitor.models import AnomalyType, ModelAction, QualityStatus, SensorReading, Status
from spindle_monitor.monitor import ConditionMonitor
from spindle_monitor.storage import CSVResultStore, SQLiteResultStore
from spindle_monitor.validation import InputValidator


class AnomalyLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        base = load_config(ROOT / "config" / "thresholds.json")
        self.config = replace(base, anomaly=replace(base.anomaly, enabled=True))
        self.start = datetime(2026, 1, 1)

    def reading(self, minute: int, vibration: float = 1.0, temperature: float = 30.0,
                current: float = 4.0) -> SensorReading:
        return SensorReading(
            self.start + timedelta(minutes=minute), vibration, temperature, current
        )

    def monitor_with_baseline(self) -> ConditionMonitor:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        for minute in range(6):
            monitor.process(self.reading(minute))
        return monitor

    def test_raw_critical_suspected_spike_preserves_immediate_safety(self) -> None:
        monitor = self.monitor_with_baseline()
        result = monitor.process(self.reading(6, vibration=4.8))
        self.assertEqual(result.raw_status, Status.CRITICAL)
        self.assertEqual(result.anomaly.safety_action, "IMMEDIATE_MANUFACTURER_CRITICAL_ACTION")
        self.assertEqual(result.maintenance_urgency, "IMMEDIATE_MAINTENANCE_REQUIRED")
        self.assertEqual(result.final_time_to_critical_hours, 0.0)
        self.assertIn(AnomalyType.SINGLE_SAMPLE_SPIKE, result.anomaly.anomaly_type)

    def test_stuck_or_disagreement_cannot_downgrade_raw_critical(self) -> None:
        monitor = self.monitor_with_baseline()
        monitor.anomaly_detector.confirm_stuck_sensor(
            "vibration_mps2", self.start + timedelta(minutes=6), "diagnostic open-circuit code"
        )
        result = monitor.process(self.reading(6, vibration=4.5))
        self.assertEqual(result.raw_status, Status.CRITICAL)
        self.assertEqual(result.anomaly.raw_safety_status, "CRITICAL")
        self.assertEqual(result.final_time_to_warning_hours, 0.0)
        self.assertEqual(result.final_time_to_critical_hours, 0.0)

    def test_correlated_change_escalates_but_keeps_forecast_available_at_low_confidence(self) -> None:
        monitor = self.monitor_with_baseline()
        result = monitor.process(self.reading(6, vibration=3.5, current=5.2))
        self.assertEqual(result.anomaly.quality_status, QualityStatus.POSSIBLE_MACHINE_EVENT)
        self.assertIn(AnomalyType.CORRELATED_ABRUPT_CHANGE, result.anomaly.anomaly_type)
        self.assertEqual(result.anomaly.model_action, ModelAction.USE_WITH_REDUCED_CONFIDENCE)
        self.assertTrue(result.anomaly.use_for_prediction)
        self.assertTrue(result.anomaly.use_for_features)
        self.assertFalse(result.anomaly.training_eligible)
        self.assertLess(result.anomaly.confidence_multiplier, 1.0)
        self.assertFalse(any(reason.startswith("anomaly:") for reason in result.withholding_reasons))


    def test_thermal_led_variable_load_change_is_not_treated_as_sensor_failure(self) -> None:
        monitor = self.monitor_with_baseline()
        # Physically valid load/temperature step: both values remain below
        # manufacturer WARNING, but the change is large enough to trigger the
        # abrupt/correlation observer.  This is a machine/regime hypothesis,
        # not evidence that the sensors themselves are broken.
        results = [
            monitor.process(self.reading(minute, temperature=45.0, current=5.2))
            for minute in range(6, 12)
        ]
        self.assertTrue(all(result.raw_status == Status.NORMAL for result in results))
        self.assertTrue(all(result.anomaly.use_for_prediction for result in results))
        self.assertTrue(any(
            AnomalyType.CORRELATED_ABRUPT_CHANGE in result.anomaly.anomaly_type
            or AnomalyType.PERSISTENT_STEP_CHANGE in result.anomaly.anomaly_type
            for result in results
        ))
        self.assertTrue(all(
            result.anomaly.model_action == ModelAction.USE_WITH_REDUCED_CONFIDENCE
            for result in results
        ))
        self.assertTrue(all(not result.anomaly.training_eligible for result in results))
        self.assertTrue(all(result.forecast_confidence == "LOW" for result in results))
        self.assertTrue(all(not result.recommendation_actionable for result in results))
        self.assertTrue(all(
            result.maintenance_urgency == "MONITOR CLOSELY — INSUFFICIENT FORECAST CONFIDENCE"
            for result in results
        ))

    def test_constant_sensors_and_low_variance_alone_are_valid(self) -> None:
        for sensor in ("vibration", "temperature", "current"):
            monitor = ConditionMonitor(self.config, enable_ml=False)
            last = None
            for minute in range(241):
                values = {"vibration": 1.2, "temperature": 31.1, "current": 4.2}
                last = monitor.process(self.reading(
                    minute, vibration=values["vibration"],
                    temperature=values["temperature"], current=values["current"]
                ))
            assert last is not None
            self.assertNotIn(AnomalyType.STUCK_SENSOR_SUSPECTED, last.anomaly.anomaly_type)
            self.assertNotIn(AnomalyType.STUCK_SENSOR_CONFIRMED, last.anomaly.anomaly_type)
            self.assertTrue(last.anomaly.training_eligible)

    def test_one_context_change_only_creates_weak_stuck_observation(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        observed = None
        for minute in range(65):
            observed = monitor.process(self.reading(minute, current=4.0 if minute < 61 else 5.0))
        assert observed is not None
        self.assertIn(AnomalyType.POSSIBLE_STUCK_SENSOR, observed.anomaly.anomaly_type)
        self.assertEqual(observed.anomaly.model_action, ModelAction.USE_NORMALLY)
        self.assertEqual(observed.anomaly.confidence_multiplier, 1.0)
        self.assertTrue(observed.anomaly.training_eligible)

    def test_multiple_context_changes_after_delay_create_suspicion_not_confirmation(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        result = None
        for minute in range(125):
            current = 4.0 if minute < 61 else 5.0 if minute < 121 else 6.0
            result = monitor.process(self.reading(minute, current=current))
        assert result is not None
        self.assertIn(AnomalyType.STUCK_SENSOR_SUSPECTED, result.anomaly.anomaly_type)
        self.assertNotIn(AnomalyType.STUCK_SENSOR_CONFIRMED, result.anomaly.anomaly_type)
        self.assertFalse(result.anomaly.training_eligible)
        self.assertTrue(result.anomaly.requires_human_review)

    def test_confirmation_requires_external_evidence_and_never_imputes(self) -> None:
        detector = AnomalyDetector(self.config)
        detector.confirm_stuck_sensor("vibration_mps2", self.start, "technician controlled test")
        reading = self.reading(0, vibration=1.234567)
        result = detector.detect(reading, Status.NORMAL)
        self.assertIn(AnomalyType.STUCK_SENSOR_CONFIRMED, result.anomaly_type)
        self.assertEqual(reading.vibration_mps2, 1.234567)
        self.assertEqual(result.model_action, ModelAction.SUSPEND_PREDICTION)

    def test_temperature_response_delay_prevents_premature_suspicion(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        result = None
        for minute in range(245):
            result = monitor.process(self.reading(minute, temperature=31.0,
                                                  current=4.0 if minute < 241 else 5.0))
        assert result is not None
        self.assertIn(AnomalyType.POSSIBLE_STUCK_SENSOR, result.anomaly.anomaly_type)
        self.assertNotIn(AnomalyType.STUCK_SENSOR_SUSPECTED, result.anomaly.anomaly_type)

    def test_spike_online_decision_is_pending_then_causally_resolved(self) -> None:
        monitor = self.monitor_with_baseline()
        pending = monitor.process(self.reading(6, vibration=3.9))
        self.assertEqual(pending.anomaly.model_action, ModelAction.HOLD_FOR_CONFIRMATION)
        self.assertIn("no future sample", pending.anomaly.causal_decision.lower())
        monitor.process(self.reading(7))
        monitor.process(self.reading(8))
        resolved = monitor.process(self.reading(9))
        self.assertFalse(resolved.anomaly.is_active)
        self.assertEqual(resolved.anomaly.final_offline_classification, "RESOLVED_TRANSIENT_SPIKE")
        self.assertEqual(pending.anomaly.final_offline_classification, "PENDING_OR_SAME_AS_CAUSAL")

    def test_persistent_single_sensor_change_is_not_finalized_as_spike(self) -> None:
        monitor = self.monitor_with_baseline()
        result = None
        for minute in range(6, 11):
            result = monitor.process(self.reading(minute, vibration=3.5))
        assert result is not None
        self.assertIn(AnomalyType.PERSISTENT_STEP_CHANGE, result.anomaly.anomaly_type)
        self.assertEqual(result.anomaly.suspected_origin, "machine_or_sensor_or_operating_regime")

    def test_long_gap_preserves_lifecycle_and_requires_recovery_history(self) -> None:
        monitor = self.monitor_with_baseline()
        lifecycle_id = monitor.process(self.reading(6)).lifecycle_id
        gap = monitor.process(self.reading(20))
        self.assertEqual(gap.lifecycle_id, lifecycle_id)
        self.assertIsNone(gap.completed_lifecycle)
        self.assertIn(AnomalyType.LONG_DATA_GAP, gap.anomaly.anomaly_type)
        self.assertFalse(gap.anomaly.use_for_prediction)
        recovery = monitor.process(self.reading(21))
        self.assertFalse(recovery.anomaly.use_for_prediction)

    def test_clipping_differs_from_stable_away_from_boundary(self) -> None:
        monitor = self.monitor_with_baseline()
        clipped = None
        for minute in range(6, 9):
            clipped = monitor.process(self.reading(minute, vibration=20.0))
        assert clipped is not None
        self.assertIn(AnomalyType.CLIPPING_SUSPECTED, clipped.anomaly.anomaly_type)
        stable = ConditionMonitor(self.config, enable_ml=False)
        for minute in range(20):
            normal = stable.process(self.reading(minute, vibration=1.2))
        self.assertNotIn(AnomalyType.CLIPPING_SUSPECTED, normal.anomaly.anomaly_type)

    def test_noise_reduces_confidence_but_monotonic_trend_is_not_noise(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        observed = []
        for minute in range(66):
            vibration = 1.0 if minute < 40 else 1.0 + (0.35 if minute % 2 else -0.35)
            observed.append(monitor.process(self.reading(minute, vibration=vibration)))
        noisy = next(result for result in observed if AnomalyType.EXCESSIVE_NOISE in result.anomaly.anomaly_type)
        self.assertLess(noisy.anomaly.confidence_multiplier, 1.0)
        self.assertFalse(noisy.anomaly.training_eligible)
        trend = ConditionMonitor(self.config, enable_ml=False)
        trend_results = [trend.process(self.reading(i, vibration=1.0 + i * 0.005)) for i in range(60)]
        self.assertFalse(any(
            AnomalyType.EXCESSIVE_NOISE in result.anomaly.anomaly_type for result in trend_results
        ))

    def test_invalid_raw_critical_is_rejected_for_model_but_preserves_safety(self) -> None:
        validation = InputValidator(self.config).validate(self.reading(0, vibration=21.0))
        self.assertFalse(validation.valid)
        anomaly = invalid_anomaly_result(validation)
        self.assertEqual(anomaly.raw_safety_status, "CRITICAL")
        self.assertEqual(anomaly.safety_action, "IMMEDIATE_MANUFACTURER_CRITICAL_ACTION")
        self.assertEqual(anomaly.model_action, ModelAction.REFUSE_PREDICTION)
        self.assertFalse(anomaly.training_eligible)

    def test_csv_and_sqlite_anomaly_fields_agree(self) -> None:
        monitor = self.monitor_with_baseline()
        result = monitor.process(self.reading(6, vibration=4.8))
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "audit.csv"
            db_path = Path(directory) / "audit.db"
            csv_store = CSVResultStore(csv_path)
            db_store = SQLiteResultStore(db_path)
            csv_store.save(result)
            db_store.save(result)
            csv_store.close()
            db_store.close()
            with csv_path.open(encoding="utf-8-sig", newline="") as source:
                csv_row = next(csv.DictReader(source))
            connection = sqlite3.connect(db_path)
            sql_row = connection.execute(
                "SELECT quality_status, anomaly_type, model_action FROM readings"
            ).fetchone()
            event_count = connection.execute("SELECT COUNT(*) FROM anomaly_events").fetchone()[0]
            connection.close()
            self.assertEqual((csv_row["quality_status"], csv_row["anomaly_type"], csv_row["model_action"]), sql_row)
            self.assertEqual(event_count, 1)

    def test_fixture_evaluation_has_zero_constant_false_positives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_anomaly_fixtures(self.config, Path(directory) / "report.json")
        self.assertTrue(report["all_passed"])
        self.assertEqual(report["false_positive_results_on_normal_constant_data"], [])
        self.assertEqual(report["false_negative_results_on_fault_fixtures"], [])

    def test_disabled_layer_preserves_existing_operational_result(self) -> None:
        disabled = replace(self.config, anomaly=replace(self.config.anomaly, enabled=False))
        first = ConditionMonitor(disabled, enable_ml=False)
        second = ConditionMonitor(disabled, enable_ml=False)
        left = [first.process(self.reading(i, vibration=1.0 + i * 0.01)) for i in range(10)]
        right = [second.process(self.reading(i, vibration=1.0 + i * 0.01)) for i in range(10)]
        self.assertEqual(
            [(item.lifecycle_id, item.lifecycle_state, item.raw_status, item.final_forecast_hours) for item in left],
            [(item.lifecycle_id, item.lifecycle_state, item.raw_status, item.final_forecast_hours) for item in right],
        )


if __name__ == "__main__":
    unittest.main()
