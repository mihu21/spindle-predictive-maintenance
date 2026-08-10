from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.config import load_config
from spindle_monitor.data_profile import profile_data
from spindle_monitor.lifecycle_detector import LifecycleDetector
from spindle_monitor.ml_forecaster import MLForecaster
from spindle_monitor.model_registry import ModelRegistry
from spindle_monitor.models import SensorReading, Status
from spindle_monitor.monitor import ConditionMonitor
from spindle_monitor.offline import prepare_offline_replay
from spindle_monitor.resampling import safe_resample
from spindle_monitor.status_policy import TimeBasedEventPolicy
from spindle_monitor.storage import SQLiteResultStore, result_to_flat_row


class FixedRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows):
        return np.full(len(rows), self.value, dtype=float)


class ReadinessRemediationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "config" / "thresholds.json")
        cls.start = datetime(2026, 1, 1)

    def test_ten_minute_warning_is_raw_but_not_confirmed(self) -> None:
        policy = TimeBasedEventPolicy(replace(
            self.config.lifecycle,
            warning_confirmation_minutes=30,
            maximum_confirmation_gap_minutes=5,
        ))
        values = [policy.update(self.start + timedelta(minutes=i), Status.WARNING) for i in range(10)]
        recovered = policy.update(self.start + timedelta(minutes=10), Status.NORMAL)
        self.assertTrue(all(value == Status.NORMAL for value in values))
        self.assertEqual(recovered, Status.NORMAL)

    def test_warning_and_critical_confirm_at_configured_elapsed_time(self) -> None:
        warning = TimeBasedEventPolicy(replace(
            self.config.lifecycle, warning_confirmation_minutes=30,
            maximum_confirmation_gap_minutes=5,
        ))
        for minute in range(30):
            self.assertLess(warning.update(self.start + timedelta(minutes=minute), Status.WARNING), Status.WARNING)
        self.assertEqual(warning.update(self.start + timedelta(minutes=30), Status.WARNING), Status.WARNING)

        critical = TimeBasedEventPolicy(replace(
            self.config.lifecycle, warning_confirmation_minutes=60,
            critical_confirmation_minutes=30, maximum_confirmation_gap_minutes=5,
        ))
        for minute in range(30):
            self.assertLess(critical.update(self.start + timedelta(minutes=minute), Status.CRITICAL), Status.CRITICAL)
        self.assertEqual(critical.update(self.start + timedelta(minutes=30), Status.CRITICAL), Status.CRITICAL)

    def test_raw_critical_protection_is_immediate_while_event_is_unconfirmed(self) -> None:
        monitor = ConditionMonitor(self.config, models_root=ROOT / "models", enable_ml=False)
        result = monitor.process(SensorReading(self.start, 4.8, 90.0, 9.0))
        self.assertEqual(result.raw_status, Status.CRITICAL)
        self.assertEqual(result.effective_status, Status.CRITICAL)
        self.assertEqual(result.event_status, Status.NORMAL)
        self.assertEqual(result.maintenance_urgency, "IMMEDIATE_MAINTENANCE_REQUIRED")

    def test_reset_waits_for_independent_minimum_critical_duration(self) -> None:
        lifecycle = replace(
            self.config.lifecycle,
            minimum_degraded_minutes=1,
            minimum_lifecycle_hours=0,
            reset_cooldown_hours=0,
            normal_confirmation_minutes=1,
            normal_reset_confirmation_minutes=1,
            pre_reset_window_minutes=60,
            post_reset_window_minutes=5,
            minimum_critical_duration_before_reset_minutes=30,
            minimum_sensor_drop_fraction=.05,
            minimum_overall_severity_drop=.1,
            healthy_baseline_tolerance=.6,
        )
        detector = LifecycleDetector(replace(self.config, lifecycle=lifecycle))
        high = {"vibration_mps2": 4.8, "temperature_c": 90.0, "current_ampere": 9.0}
        low = {"vibration_mps2": 1.2, "temperature_c": 33.0, "current_ampere": 4.2}
        for minute in range(30):
            detector.update(timestamp=self.start + timedelta(minutes=minute), status=Status.CRITICAL,
                            event_status=Status.CRITICAL, smoothed_values=high, maximum_severity=1.2)
        self.assertEqual(detector.completed, [])
        detector.update(timestamp=self.start + timedelta(minutes=30), status=Status.NORMAL,
                        event_status=Status.NORMAL, smoothed_values=low, maximum_severity=.1)
        completed = detector.update(timestamp=self.start + timedelta(minutes=31), status=Status.NORMAL,
                                    event_status=Status.NORMAL, smoothed_values=low, maximum_severity=.1)
        self.assertIsNotNone(completed.completed_lifecycle)

    def test_resampling_stops_at_long_gaps_and_reset_boundaries(self) -> None:
        readings = [
            SensorReading(self.start, 1, 30, 4),
            SensorReading(self.start + timedelta(minutes=10), 1.2, 31, 4.1),
        ]
        long_gap = safe_resample(readings, interpolation_enabled=True, maximum_interpolation_gap_minutes=5,
                                 source_statuses=["normal", "normal"])
        self.assertEqual(len(long_gap), 2)
        self.assertFalse(long_gap[-1].features_available)

        reset_span = safe_resample(
            [readings[0], SensorReading(self.start + timedelta(minutes=4), 1.2, 31, 4.1)],
            interpolation_enabled=True, maximum_interpolation_gap_minutes=5,
            lifecycle_boundary_timestamps=(self.start + timedelta(minutes=2),),
            source_statuses=["normal", "normal"],
        )
        self.assertEqual(len(reset_span), 2)

    def test_offline_workflow_marks_interpolated_rows_in_normal_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "small.csv"
            path.write_text(
                "timestamp,vibration_mps2,temperature_c,current_ampere,health_status\n"
                "2026-01-01 00:00:00,1,30,4,normal\n"
                "2026-01-01 00:02:00,1.2,32,4.2,normal\n",
                encoding="utf-8",
            )
            configured = replace(self.config, interpolation_enabled=True, maximum_interpolation_gap_minutes=5)
            validations, plan = prepare_offline_replay(path, configured, root / "models")
            valid = [value for value in validations if value.valid]
            self.assertEqual(len(valid), 3)
            interpolated = next(value for value in valid if value.interpolated)
            monitor = ConditionMonitor(configured, models_root=root / "models", lifecycle_plan=plan, enable_ml=False)
            result = monitor.process(
                interpolated.reading,
                interpolated=True,
                source_sampling_interval_seconds=interpolated.source_sampling_interval_seconds,
                effective_resampling_interval_seconds=interpolated.effective_resampling_interval_seconds,
            )
            self.assertEqual(result_to_flat_row(result)["row_provenance"], "interpolated")

    def test_profiler_refuses_duplicates_and_unknown_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.csv"
            duplicate.write_text(
                "timestamp,vibration_mps2,temperature_c,current_ampere,health_status\n"
                "2026-01-01 00:00:00,1,30,4,normal\n"
                "2026-01-01 00:00:00,1,30,4,normal\n", encoding="utf-8")
            duplicate_report = profile_data(duplicate, self.config)
            self.assertFalse(duplicate_report["suitability"]["replay"])
            self.assertIn("duplicate_timestamps", {
                value["code"] for value in duplicate_report["suitability_refusal_reasons"]["replay"]
            })

            unknown = root / "unknown.csv"
            unknown.write_text(
                "timestamp,vibration_mps2,temperature_c,current_ampere,health_status\n"
                "2026-01-01 00:00:00,1,30,4,banana\n", encoding="utf-8")
            unknown_report = profile_data(unknown, self.config)
            self.assertFalse(unknown_report["suitability"]["replay"])
            self.assertFalse(unknown_report["suitability"]["feature_extraction"])

    def test_early_24_hour_window_is_partial_and_immature(self) -> None:
        monitor = ConditionMonitor(self.config, models_root=ROOT / "models", enable_ml=False)
        result = monitor.process(SensorReading(self.start, 1.2, 33.0, 4.2))
        prefix = "vibration_mps2__w1440m"
        self.assertEqual(result.features[f"{prefix}__sample_count"], 1.0)
        self.assertLess(result.features[f"{prefix}__coverage_fraction"], .01)
        self.assertEqual(result.features[f"{prefix}__window_mature"], 0.0)

    def test_both_ends_of_target_support_are_flagged_without_clipping(self) -> None:
        for prediction in (-1.0, 10.0):
            with self.subTest(prediction=prediction), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "diagnostic"
                registry = ModelRegistry(root)
                metadata = {
                    "model_version": "p1", "data_domain": "accelerated_mock", "model_stage": "candidate_diagnostic",
                    "feature_names": ["f"], "input_schema_version": self.config.ml.schema_version,
                    "threshold_config_version": self.config.threshold_config_version,
                    "lifecycle_config_version": self.config.lifecycle_config_version,
                    "feature_distribution": {"lower_quantile_01": [0], "upper_quantile_99": [1]},
                    "target_support": {"time_to_warning": {"minimum": 0.1, "maximum": 5.0}},
                    "targets": {"time_to_warning": {}}, "probability_targets": {},
                }
                registry.save_candidate({"time_to_warning": FixedRegressor(prediction)}, metadata)
                registry.promote_targets(["time_to_warning"])
                forecast = MLForecaster(self.config, ["f"], root).predict({"f": .5})
                self.assertEqual(forecast.time_to_warning_hours, prediction)
                self.assertTrue(forecast.target_support_violations["time_to_warning"])
                self.assertEqual(forecast.physically_invalid_targets["time_to_warning"], prediction < 0)

    def test_sqlite_normal_record_contains_complete_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            monitor = ConditionMonitor(self.config, models_root=ROOT / "models", enable_ml=False,
                                       dataset_hash="abc", generator_version="v2")
            result = monitor.process(SensorReading(self.start, 1.2, 33.0, 4.2))
            store = SQLiteResultStore(Path(directory) / "audit.db")
            store.save(result, "normal")
            store.close()
            import sqlite3
            connection = sqlite3.connect(Path(directory) / "audit.db")
            row = connection.execute(
                "SELECT dataset_hash, generator_version, guardrail_result, details_json FROM readings"
            ).fetchone()
            connection.close()
            self.assertEqual(row[:2], ("abc", "v2"))
            self.assertIn("row_provenance", json.loads(row[3]))


if __name__ == "__main__":
    unittest.main()
