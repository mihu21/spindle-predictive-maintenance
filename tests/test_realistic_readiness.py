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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.config import load_config
from spindle_monitor.data_profile import file_sha256, profile_data
from spindle_monitor.domain import validate_registry_domain
from spindle_monitor.duration_evaluation import duration_bucket_evaluation
from spindle_monitor.ml_forecaster import MLForecaster
from spindle_monitor.model_registry import ModelRegistry
from spindle_monitor.models import SensorReading, Status
from spindle_monitor.realistic_generator import generate_realistic_mock
from spindle_monitor.resampling import safe_resample
from spindle_monitor.status_policy import TimeBasedEventPolicy
from spindle_monitor.storage import SQLiteResultStore


class FixedRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows):
        return np.full(len(rows), self.value, dtype=float)


class RealisticReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "config" / "thresholds.json")

    def test_provided_file_is_profiled_as_one_open_censored_lifecycle(self) -> None:
        report = profile_data(
            ROOT / "tests" / "fixtures" / "small_open_lifecycle.csv",
            self.config,
        )
        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["sampling_interval_seconds"]["median"], 60.0)
        self.assertEqual(report["raw_status_transition_count"], 1)
        self.assertEqual(report["completed_lifecycles"], 0)
        self.assertEqual(report["lifecycle_states"]["open_critical"], 1)
        self.assertFalse(report["suitability"]["training"])

    def test_profile_detects_duplicates_order_and_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text(
                "timestamp,vibration_mps2,temperature_c,current_ampere,health_status\n"
                "2026-01-01 00:01:00,1,30,4,normal\n"
                "2026-01-01 00:01:00,,30,4,normal\n"
                "2026-01-01 00:00:00,1,30,4,normal\n",
                encoding="utf-8",
            )
            report = profile_data(path, self.config)
            self.assertEqual(report["duplicate_timestamps"], 1)
            self.assertEqual(report["out_of_order_timestamps"], 1)
            self.assertEqual(report["missing_sensor_values"]["vibration_mps2"], 1)

    def test_time_confirmation_is_sampling_interval_independent(self) -> None:
        lifecycle = replace(
            self.config.lifecycle, warning_confirmation_minutes=30,
            maximum_confirmation_gap_minutes=20,
        )
        for minutes in (1, 5, 15):
            policy = TimeBasedEventPolicy(lifecycle)
            start = datetime(2026, 1, 1)
            status = Status.NORMAL
            for elapsed in range(0, 31, minutes):
                status = policy.update(start + timedelta(minutes=elapsed), Status.WARNING)
            if 30 % minutes:
                status = policy.update(start + timedelta(minutes=30), Status.WARNING)
            self.assertEqual(status, Status.WARNING)

    def test_large_gap_is_not_silently_interpolated(self) -> None:
        start = datetime(2026, 1, 1)
        readings = [
            SensorReading(start, 1, 30, 4),
            SensorReading(start + timedelta(minutes=10), 2, 40, 5),
        ]
        processed = safe_resample(readings, interpolation_enabled=True, maximum_interpolation_gap_minutes=5)
        self.assertEqual(len(processed), 2)
        self.assertFalse(processed[-1].features_available)
        self.assertFalse(processed[-1].interpolated)

    def test_realistic_generator_is_deterministic_and_domain_labelled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = {
                "start_timestamp": "2026-01-01T00:00:00", "total_duration_hours": 72,
                "sampling_interval_seconds": {"median": 60}, "sensor_statistics": {},
            }
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            outputs = []
            for suffix in ("a", "b"):
                csv_path = root / f"{suffix}.csv"
                meta_path = root / f"{suffix}_metadata.json"
                metadata = generate_realistic_mock(profile_path, 2, csv_path, meta_path, self.config, seed=42)
                outputs.append((file_sha256(csv_path), metadata))
            self.assertEqual(outputs[0][0], outputs[1][0])
            self.assertEqual(outputs[0][1]["data_domain"], "realistic_synthetic")
            self.assertFalse(outputs[0][1]["production_eligible"])
            lifecycle = outputs[0][1]["lifecycles"][0]
            self.assertIn("first_raw_warning_timestamp", lifecycle)
            self.assertIn("first_confirmed_warning_timestamp", lifecycle)
            self.assertNotEqual(
                lifecycle["first_raw_warning_timestamp"],
                lifecycle["first_confirmed_warning_timestamp"],
            )
            with (root / "a.csv").open(encoding="utf-8") as source:
                self.assertEqual(next(csv.reader(source)), ["timestamp", "vibration_mps2", "temperature_c", "current_ampere", "health_status"])

    def test_domain_registry_mismatch_and_synthetic_promotion_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Cannot use plant registry"):
                validate_registry_domain(root / "plant", "realistic_synthetic")
            registry = ModelRegistry(root / "realistic")
            registry.save_candidate({}, {"model_version": "r1", "data_domain": "realistic_synthetic", "targets": {}})
            with self.assertRaisesRegex(ValueError, "Cannot promote realistic_synthetic"):
                registry.promote_targets([])
            self.assertIn("promotion_refused", registry.audit_log.read_text(encoding="utf-8"))

    def test_support_metadata_flags_without_clipping_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "diagnostic"
            registry = ModelRegistry(root)
            metadata = {
                "model_version": "p1", "data_domain": "accelerated_mock", "feature_names": ["f"],
                "input_schema_version": self.config.ml.schema_version,
                "threshold_config_version": self.config.threshold_config_version,
                "lifecycle_config_version": self.config.lifecycle_config_version,
                "feature_distribution": {"lower_quantile_01": [0], "upper_quantile_99": [1]},
                "target_support": {"time_to_warning": {"maximum": 5.0}},
                "targets": {"time_to_warning": {}}, "probability_targets": {},
            }
            registry.save_candidate({"time_to_warning": FixedRegressor(10)}, metadata)
            registry.promote_targets(["time_to_warning"])
            forecast = MLForecaster(self.config, ["f"], root).predict({"f": .5})
            self.assertEqual(forecast.time_to_warning_hours, 10.0)
            self.assertTrue(forecast.beyond_training_duration_support)
            self.assertEqual(forecast.training_target_max_hours["time_to_warning"], 5.0)

    def test_duration_bucket_and_tolerance_metrics(self) -> None:
        report = duration_bucket_evaluation(
            [1, 1, 1, 1], [1.2, .5, 2, 1], ["a", "b", "c", "d"],
            {"a": 50, "b": 100, "c": 200, "d": 400},
        )
        self.assertEqual(report["buckets"]["short"]["lifecycle_count"], 1)
        self.assertEqual(report["buckets"]["very_long"]["lifecycle_count"], 1)
        self.assertGreater(report["buckets"]["long"]["late_by_more_than"]["30_minutes"], 0)

    def test_probability_horizons_are_configurable(self) -> None:
        configured = replace(self.config, ml=replace(self.config.ml, forecast_horizons_hours=(6, 12, 24, 48)))
        configured.validate()

    def test_sqlite_additive_migration_preserves_old_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.db"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE readings (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, source_label TEXT, status TEXT NOT NULL, lifecycle_id TEXT NOT NULL, lifecycle_state TEXT NOT NULL, final_forecast_hours REAL, forecast_confidence TEXT, details_json TEXT NOT NULL)")
            connection.commit(); connection.close()
            store = SQLiteResultStore(path); store.close()
            connection = sqlite3.connect(path)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(readings)")}
            connection.close()
            self.assertIn("data_domain", columns)
            self.assertIn("target_support_violation", columns)


if __name__ == "__main__":
    unittest.main()
