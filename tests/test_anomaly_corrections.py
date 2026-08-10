from __future__ import annotations

import argparse
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

from spindle_monitor.cli import build_parser, run_export_anomaly_intervals, run_train
from spindle_monitor.config import load_config
from spindle_monitor.models import (
    AnomalyType, FeatureHistoryAction, ModelAction, SensorReading, Status,
)
from spindle_monitor.monitor import ConditionMonitor
from spindle_monitor.resampling import safe_resample
from spindle_monitor.retraining import build_training_examples
from spindle_monitor.storage import CSVResultStore, SQLiteResultStore


class AnomalyCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        base = load_config(ROOT / "config" / "thresholds.json")
        self.config = replace(base, anomaly=replace(base.anomaly, enabled=True))
        self.start = datetime(2026, 1, 1)

    def reading(self, minute: int, vibration: float = 1.0, temperature: float = 30.0,
                current: float = 4.0) -> SensorReading:
        return SensorReading(
            self.start + timedelta(minutes=minute), vibration, temperature, current
        )

    def baseline(self, monitor: ConditionMonitor, end: int = 6) -> None:
        for minute in range(end):
            monitor.process(self.reading(minute))

    def test_held_spike_never_contaminates_subsequent_rolling_features(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        self.baseline(monitor)
        held = monitor.process(self.reading(6, vibration=4.8))
        self.assertEqual(
            held.anomaly.feature_history_action,
            FeatureHistoryAction.HOLD_OUTSIDE_FEATURE_HISTORY,
        )
        monitor.process(self.reading(7))
        monitor.process(self.reading(8))
        after = monitor.process(self.reading(9))
        prefix = "vibration_mps2__w5m__"
        self.assertEqual(after.features[prefix + "mean"], 1.0)
        self.assertEqual(after.features[prefix + "maximum"], 1.0)
        self.assertEqual(after.features[prefix + "std"], 0.0)
        self.assertEqual(after.features[prefix + "slope_per_hour"], 0.0)
        self.assertEqual(after.features[prefix + "threshold_crossings"], 0.0)

    def test_held_raw_critical_value_remains_in_csv_and_sqlite_audit(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        self.baseline(monitor)
        result = monitor.process(self.reading(6, vibration=4.8))
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "audit.csv"
            db_path = Path(directory) / "audit.db"
            csv_store = CSVResultStore(csv_path)
            db_store = SQLiteResultStore(db_path)
            csv_store.save(result); db_store.save(result)
            csv_store.close(); db_store.close()
            with csv_path.open(encoding="utf-8-sig", newline="") as source:
                row = next(csv.DictReader(source))
            connection = sqlite3.connect(db_path)
            details = json.loads(connection.execute("SELECT details_json FROM readings").fetchone()[0])
            connection.close()
        self.assertEqual(float(row["vibration_mps2"]), 4.8)
        self.assertEqual(details["vibration_mps2"], 4.8)
        self.assertEqual(row["safety_action"], "IMMEDIATE_MANUFACTURER_CRITICAL_ACTION")

    def test_persistent_policy_resumes_feature_history_at_reduced_confidence(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        self.baseline(monitor)
        results = [monitor.process(self.reading(i, vibration=3.5)) for i in range(6, 11)]
        persistent = results[-1]
        self.assertIn(AnomalyType.PERSISTENT_STEP_CHANGE, persistent.anomaly.anomaly_type)
        self.assertEqual(
            persistent.anomaly.feature_history_action,
            FeatureHistoryAction.COMMIT_TO_FEATURE_HISTORY,
        )
        self.assertTrue(persistent.anomaly.use_for_prediction)
        self.assertFalse(persistent.anomaly.training_eligible)
        # The initial ambiguous samples remain causally held out, but once the
        # persistent episode is established, current/future values may enter
        # feature history so real degradation is not hidden from prognostics.
        self.assertGreater(len(monitor.features.history["vibration_mps2"]), 6)

    def test_short_gap_detected_without_interpolation_and_evidence_is_exact(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        first = monitor.process(self.reading(0))
        gap = monitor.process(self.reading(3))
        self.assertIn(AnomalyType.SHORT_DATA_GAP, gap.anomaly.anomaly_type)
        self.assertEqual(gap.anomaly.estimated_missing_sample_count, 2)
        self.assertEqual(gap.anomaly.expected_sampling_interval_seconds, 60.0)
        self.assertEqual(gap.anomaly.source_sampling_interval_seconds, 180.0)
        self.assertFalse(gap.anomaly.interpolation_occurred)
        self.assertEqual(gap.lifecycle_id, first.lifecycle_id)
        self.assertIsNone(gap.completed_lifecycle)

    def test_short_gap_with_safe_interpolation_marks_generated_rows(self) -> None:
        readings = [self.reading(0), self.reading(3)]
        processed = safe_resample(
            readings, interpolation_enabled=True, maximum_interpolation_gap_minutes=5,
            target_interval_seconds=60, source_statuses=["normal", "normal"],
        )
        generated = [item for item in processed if item.interpolated]
        self.assertEqual(len(generated), 2)
        configured = replace(
            self.config,
            interpolation_enabled=True,
            anomaly=replace(self.config.anomaly, enabled=True),
        )
        monitor = ConditionMonitor(configured, enable_ml=False)
        monitor.process(self.reading(0))
        result = monitor.process(
            generated[0].processed, interpolated=True,
            source_sampling_interval_seconds=generated[0].source_sampling_interval_seconds,
            effective_resampling_interval_seconds=60,
        )
        self.assertIn(AnomalyType.SHORT_DATA_GAP, result.anomaly.anomaly_type)
        self.assertTrue(result.anomaly.interpolation_occurred)

    def test_gap_evidence_matches_csv_and_sqlite(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        monitor.process(self.reading(0))
        result = monitor.process(self.reading(3))
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "gap.csv"; db_path = Path(directory) / "gap.db"
            csv_store = CSVResultStore(csv_path); db_store = SQLiteResultStore(db_path)
            csv_store.save(result); db_store.save(result); csv_store.close(); db_store.close()
            with csv_path.open(encoding="utf-8-sig", newline="") as source:
                row = next(csv.DictReader(source))
            connection = sqlite3.connect(db_path)
            details = json.loads(connection.execute("SELECT details_json FROM readings").fetchone()[0])
            connection.close()
        self.assertEqual(int(row["estimated_missing_sample_count"]), details["estimated_missing_sample_count"])
        self.assertEqual(float(row["gap_duration_seconds"]), details["gap_duration_seconds"])

    def test_long_gap_still_suspends_and_recovery_is_not_immediate(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        monitor.process(self.reading(0))
        gap = monitor.process(self.reading(10))
        recovery = monitor.process(self.reading(11))
        self.assertIn(AnomalyType.LONG_DATA_GAP, gap.anomaly.anomaly_type)
        self.assertEqual(gap.anomaly.model_action, ModelAction.SUSPEND_PREDICTION)
        self.assertFalse(recovery.anomaly.use_for_prediction)

    def test_correlation_window_is_enforced_and_evidence_records_separation(self) -> None:
        within = ConditionMonitor(self.config, enable_ml=False); self.baseline(within)
        result = within.process(self.reading(6, vibration=3.5, current=6.0))
        self.assertIn(AnomalyType.CORRELATED_ABRUPT_CHANGE, result.anomaly.anomaly_type)
        self.assertFalse(result.anomaly.training_eligible)
        evidence = result.anomaly.correlation_evidence
        self.assertLessEqual(
            evidence["maximum_timestamp_separation_seconds"],
            evidence["configured_correlation_window_seconds"],
        )

        outside = ConditionMonitor(self.config, enable_ml=False); self.baseline(outside)
        for minute in range(6, 66):
            outside.process(self.reading(minute, vibration=3.5))
        late = outside.process(self.reading(66, vibration=3.5, current=6.0))
        self.assertNotIn(AnomalyType.CORRELATED_ABRUPT_CHANGE, late.anomaly.anomaly_type)

    def test_expired_single_sensor_event_remains_auditable_not_correlated(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False); self.baseline(monitor)
        persistent = None
        for minute in range(6, 15):
            persistent = monitor.process(self.reading(minute, vibration=3.5))
        assert persistent is not None
        self.assertIn(AnomalyType.PERSISTENT_STEP_CHANGE, persistent.anomaly.anomaly_type)
        for minute in range(15, 70):
            monitor.process(self.reading(minute, vibration=3.5))
        late = monitor.process(self.reading(70, vibration=3.5, temperature=50.0))
        self.assertNotIn(AnomalyType.CORRELATED_ABRUPT_CHANGE, late.anomaly.anomaly_type)

    def test_spike_observation_window_and_configured_penalty_are_enforced(self) -> None:
        changed = replace(
            self.config,
            anomaly=replace(
                self.config.anomaly,
                spike_observation_seconds=180,
                confidence_penalties={**self.config.anomaly.confidence_penalties, "suspected": 0.42},
            ),
        )
        monitor = ConditionMonitor(changed, enable_ml=False); self.baseline(monitor)
        pending = monitor.process(self.reading(6, vibration=3.9))
        early = monitor.process(self.reading(7))
        self.assertTrue(early.anomaly.is_active)
        self.assertEqual(pending.anomaly.confidence_multiplier, 0.42)
        self.assertEqual(pending.anomaly.configured_confidence_multiplier, 0.42)
        monitor.process(self.reading(8))
        resolved = monitor.process(self.reading(9))
        self.assertFalse(resolved.anomaly.is_active)
        self.assertEqual(resolved.anomaly.resolved_timestamp, self.start + timedelta(minutes=9))

    def test_stuck_timestamps_and_external_confirmation_metadata(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        result = None
        for minute in range(125):
            current = 4.0 if minute < 61 else 5.0 if minute < 121 else 6.0
            result = monitor.process(self.reading(minute, current=current))
        assert result is not None
        self.assertEqual(result.anomaly.first_observed_timestamp, self.start)
        self.assertEqual(result.anomaly.suspicion_timestamp, result.timestamp)
        confirmation_time = self.start + timedelta(minutes=125)
        monitor.anomaly_detector.confirm_stuck_sensor(
            "vibration_mps2", confirmation_time, "controlled response test failed",
            source="technician_test", actor="tech-17",
        )
        confirmed = monitor.process(self.reading(125, current=6.0))
        self.assertEqual(confirmed.anomaly.confirmed_timestamp, confirmation_time)
        self.assertEqual(confirmed.anomaly.confirmation_source, "technician_test")
        self.assertEqual(confirmed.anomaly.confirming_actor, "tech-17")

    def test_precision_limited_repetition_is_weakened_not_strong_stuck_evidence(self) -> None:
        vibration = replace(
            self.config.anomaly.sensors["vibration_mps2"],
            exact_raw_values_available=False,
            decimal_precision=1,
            stuck_observation_seconds=60,
            stuck_suspicion_seconds=120,
            expected_response_delay_seconds=60,
        )
        configured = replace(
            self.config,
            anomaly=replace(
                self.config.anomaly,
                sensors={**self.config.anomaly.sensors, "vibration_mps2": vibration},
            ),
        )
        monitor = ConditionMonitor(configured, enable_ml=False)
        result = None
        for minute in range(125):
            current = 4.0 if minute < 61 else 5.0 if minute < 121 else 6.0
            result = monitor.process(self.reading(minute, vibration=1.2, current=current))
        assert result is not None
        self.assertIn(AnomalyType.POSSIBLE_STUCK_SENSOR, result.anomaly.anomaly_type)
        self.assertNotIn(AnomalyType.STUCK_SENSOR_SUSPECTED, result.anomaly.anomaly_type)
        self.assertIn("repetition evidence is weakened", " ".join(result.anomaly.supporting_evidence))

    def _save_spike_episode(self, store: SQLiteResultStore, monitor: ConditionMonitor,
                            start_minute: int) -> None:
        store.save(monitor.process(self.reading(start_minute, vibration=4.8)))
        for minute in range(start_minute + 1, start_minute + 4):
            store.save(monitor.process(self.reading(minute)))

    def test_interval_export_consolidates_episodes_and_keeps_separate_spikes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "audit.db"
            store = SQLiteResultStore(db_path); monitor = ConditionMonitor(self.config, enable_ml=False)
            for minute in range(6): store.save(monitor.process(self.reading(minute)))
            self._save_spike_episode(store, monitor, 6)
            for minute in range(10, 15): store.save(monitor.process(self.reading(minute)))
            self._save_spike_episode(store, monitor, 15)
            store.close()
            connection = sqlite3.connect(db_path)
            rows = connection.execute(
                "SELECT active, end_or_resolution_timestamp, duration_seconds, event_row_count "
                "FROM anomaly_intervals ORDER BY interval_id"
            ).fetchall()
            connection.close()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row[0] == 0 and row[1] is not None for row in rows))
        self.assertTrue(all(row[2] == 180.0 and row[3] == 4 for row in rows))

    def test_active_interval_has_null_end_and_state_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "audit.db"
            store = SQLiteResultStore(db_path); monitor = ConditionMonitor(self.config, enable_ml=False)
            for minute in range(6): store.save(monitor.process(self.reading(minute)))
            store.save(monitor.process(self.reading(6, vibration=4.8)))
            store.close()
            reopened = SQLiteResultStore(db_path); reopened.close()
            connection = sqlite3.connect(db_path)
            interval = connection.execute(
                "SELECT active, end_or_resolution_timestamp FROM anomaly_intervals"
            ).fetchone()
            state = connection.execute(
                "SELECT active FROM anomaly_state WHERE sensor='vibration_mps2'"
            ).fetchone()
            connection.close()
        self.assertEqual(interval, (1, None))
        self.assertEqual(state, (1,))

    def test_normal_recovery_clears_only_affected_state_and_history_remains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "audit.db"
            store = SQLiteResultStore(db_path); monitor = ConditionMonitor(self.config, enable_ml=False)
            for minute in range(6): store.save(monitor.process(self.reading(minute)))
            self._save_spike_episode(store, monitor, 6); store.close()
            connection = sqlite3.connect(db_path)
            states = dict(connection.execute("SELECT sensor, active FROM anomaly_state"))
            interval_count = connection.execute("SELECT COUNT(*) FROM anomaly_intervals").fetchone()[0]
            event_count = connection.execute("SELECT COUNT(*) FROM anomaly_events").fetchone()[0]
            connection.close()
        self.assertEqual(states, {"current_ampere": 0, "temperature_c": 0, "vibration_mps2": 0})
        self.assertEqual(interval_count, 1)
        self.assertEqual(event_count, 4)

    def test_different_sensors_have_independent_current_state_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "audit.db"
            store = SQLiteResultStore(db_path); monitor = ConditionMonitor(self.config, enable_ml=False)
            for minute in range(6): store.save(monitor.process(self.reading(minute)))
            store.save(monitor.process(self.reading(6, vibration=3.5, current=6.0)))
            store.close()
            connection = sqlite3.connect(db_path)
            states = dict(connection.execute("SELECT sensor, active FROM anomaly_state"))
            connection.close()
        self.assertEqual(states["vibration_mps2"], 1)
        self.assertEqual(states["current_ampere"], 1)
        self.assertEqual(states["temperature_c"], 0)

    def test_json_csv_and_sqlite_interval_exports_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "audit.db"; json_path = Path(directory) / "i.json"; csv_path = Path(directory) / "i.csv"
            store = SQLiteResultStore(db_path); monitor = ConditionMonitor(self.config, enable_ml=False)
            for minute in range(6): store.save(monitor.process(self.reading(minute)))
            self._save_spike_episode(store, monitor, 6); store.close()
            run_export_anomaly_intervals(argparse.Namespace(
                database=str(db_path), output=str(json_path), csv=str(csv_path)
            ))
            exported = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8-sig", newline="") as source:
                csv_rows = list(csv.DictReader(source))
            connection = sqlite3.connect(db_path)
            sql_count = connection.execute("SELECT COUNT(*) FROM anomaly_intervals").fetchone()[0]
            connection.close()
        self.assertEqual(len(exported), len(csv_rows))
        self.assertEqual(len(exported), sql_count)
        self.assertEqual(exported[0]["interval_id"], csv_rows[0]["interval_id"])

    def test_plant_training_without_screening_is_refused_and_reproduction_mode_parses(self) -> None:
        parser = build_parser()
        reproduction = parser.parse_args(["train-model", "--reproduction-mode"])
        self.assertTrue(reproduction.reproduction_mode)
        args = parser.parse_args([
            "train-model", "--data-domain", "plant", "--models-root", "models/plant"
        ])
        with self.assertRaisesRegex(ValueError, "refused without --anomaly-screening"):
            run_train(args)

    def test_training_screening_counts_spike_exclusion_and_keeps_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "screen.csv"
            with input_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow(["timestamp", "vibration_mps2", "temperature_c", "current_ampere", "health_status"])
                for minute in range(10):
                    vibration = 4.8 if minute == 6 else 1.0
                    writer.writerow([
                        (self.start + timedelta(minutes=minute)).isoformat(), vibration, 30.0, 4.0,
                        "critical" if minute == 6 else "normal",
                    ])
            _, _, scan = build_training_examples(input_path, self.config, directory)
        self.assertTrue(scan["anomaly_screening_enabled"])
        self.assertGreater(scan["anomaly_training_eligible_rows"], 0)
        self.assertGreater(scan["anomaly_training_excluded_rows"], 0)
        self.assertIn("abrupt_reading_pending_confirmation", scan["anomaly_training_exclusion_counts"])
        self.assertTrue(scan["anomaly_configuration_sha256"])


if __name__ == "__main__":
    unittest.main()
