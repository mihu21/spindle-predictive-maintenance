from __future__ import annotations

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
from spindle_monitor.model_registry import ModelRegistry
from spindle_monitor.ml_forecaster import MLForecaster
from spindle_monitor.models import ForecastDecision, LifecycleRecord, SensorReading, Status
from spindle_monitor.monitor import ConditionMonitor
from spindle_monitor.offline import OfflineLifecyclePlan
from spindle_monitor.retraining import (
    ConstantProbabilityClassifier,
    LifecycleExamples,
    evaluate_candidate,
    lifecycle_sample_weights,
    probability_metrics,
    probability_target_eligibility,
    regression_metrics,
    split_lifecycles,
)
from spindle_monitor.storage import SQLiteResultStore


class ModelValidationFixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "config" / "thresholds.json")
        self.start = datetime(2026, 1, 1)

    def example(self, number: int, critical: bool = True) -> LifecycleExamples:
        timestamps = [self.start + timedelta(hours=value) for value in range(6)]
        return LifecycleExamples(
            lifecycle_id=f"lifecycle_{number:04d}",
            timestamps=timestamps,
            feature_rows=[[float(number), float(value)] for value in range(6)],
            baseline_warning=[None] * 6,
            baseline_critical=[None] * 6,
            first_warning_timestamp=self.start + timedelta(hours=4),
            first_critical_timestamp=(self.start + timedelta(hours=5)) if critical else None,
        )

    def test_regression_targets_are_strictly_pre_event_and_positive(self) -> None:
        example = self.example(1)
        _, warning, _, warning_times = example.target_rows("warning", 500)
        _, critical, _, critical_times = example.target_rows("critical", 500)
        self.assertTrue(all(timestamp < example.first_warning_timestamp for timestamp in warning_times))
        self.assertTrue(all(timestamp < example.first_critical_timestamp for timestamp in critical_times))
        self.assertTrue(all(value > 0 for value in warning))
        self.assertTrue(all(value > 0 for value in critical))
        self.assertEqual(warning[-1], 1.0)
        self.assertEqual(critical[-1], 1.0)

    def test_lifecycle_split_is_disjoint_deterministic_and_stratified(self) -> None:
        examples = [self.example(index, critical=index % 4 != 0) for index in range(1, 25)]
        first = split_lifecycles(examples, self.config)
        second = split_lifecycles(examples, self.config)
        first_ids = [[value.lifecycle_id for value in group] for group in first]
        second_ids = [[value.lifecycle_id for value in group] for group in second]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual([len(group) for group in first], [16, 4, 4])
        sets = [set(group) for group in first_ids]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertTrue(all(any(example.critical_reached for example in group) for group in first))

    def test_lifecycle_weights_contribute_equal_total_weight(self) -> None:
        ids = ["short"] * 2 + ["long"] * 20
        weights = lifecycle_sample_weights(ids)
        short = float(np.sum(weights[np.asarray(ids) == "short"]))
        long = float(np.sum(weights[np.asarray(ids) == "long"]))
        self.assertAlmostEqual(short, long)

    def test_micro_macro_and_per_lifecycle_metrics_are_reported(self) -> None:
        actual = [0.0] * 100 + [0.0]
        predicted = [1.0] * 100 + [10.0]
        metrics = regression_metrics(actual, predicted, ["long"] * 100 + ["short"])
        self.assertAlmostEqual(metrics["micro_mae_hours"], 110.0 / 101.0)
        self.assertAlmostEqual(metrics["macro_lifecycle_mae_hours"], 5.5)
        self.assertEqual(set(metrics["per_lifecycle_metrics"]), {"long", "short"})

    def test_probability_metrics_handle_single_class_without_invalid_auc(self) -> None:
        metrics = probability_metrics(np.ones(4), np.full(4, 0.99), 0.5)
        self.assertIsNone(metrics["roc_auc"])
        self.assertIn("Only one class", metrics["invalid_metric_reason"])
        self.assertIn("calibration_error", metrics)
        self.assertIn("log_loss", metrics)

    def test_nearly_constant_probability_target_is_rejected(self) -> None:
        eligible, reason = probability_target_eligibility(
            {"training": 0.9985, "validation": 1.0, "test": 0.99},
            {"training": 2, "validation": 1, "test": 2},
            self.config,
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "target_is_nearly_constant")

    def test_probability_calibration_limit_is_configurable(self) -> None:
        self.assertEqual(self.config.ml.maximum_probability_calibration_error, 0.10)

    def test_disagreement_thresholds_are_target_specific_and_configurable(self) -> None:
        self.assertGreaterEqual(self.config.ml.warning_disagreement_absolute_hours, 0.0)
        self.assertGreaterEqual(self.config.ml.critical_disagreement_absolute_hours, 0.0)
        self.assertGreaterEqual(self.config.ml.disagreement_relative_threshold, 0.0)

    def test_warning_and_critical_promotion_are_independent(self) -> None:
        config = replace(
            self.config,
            ml=replace(self.config.ml, minimum_deployment_lifecycles=3),
        )
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            common = {
                "training_lifecycle_ids": ["a", "b", "c"],
                "validation_lifecycle_ids": ["d"],
                "test_lifecycle_ids": ["e"],
            }
            def target(candidate: float, baseline: float):
                return {
                    **common,
                    "validation_metrics": {"macro_lifecycle_mae_hours": candidate},
                    "test_metrics": {"macro_lifecycle_mae_hours": candidate},
                    "baseline_metrics": {
                        "validation": {"macro_lifecycle_mae_hours": baseline},
                        "test": {"macro_lifecycle_mae_hours": baseline},
                    },
                }
            registry.save_candidate(
                {
                    "time_to_warning": ConstantProbabilityClassifier(0.5),
                    "time_to_critical": ConstantProbabilityClassifier(0.5),
                },
                {
                    "model_version": "independent",
                    "targets": {
                        "time_to_warning": target(3.0, 2.0),
                        "time_to_critical": target(1.0, 2.0),
                    },
                    "probability_targets": {},
                    "feature_names": [],
                    "input_schema_version": config.ml.schema_version,
                    "threshold_config_version": config.threshold_config_version,
                    "lifecycle_config_version": config.lifecycle_config_version,
                    "synthetic_data_only": True,
                },
            )
            report = evaluate_candidate(config, directory, promote=True)
            self.assertEqual(report["status"], "partially_promoted")
            self.assertFalse(report["targets"]["time_to_warning"]["eligible"])
            self.assertTrue(report["targets"]["time_to_critical"]["eligible"])
            self.assertFalse((Path(directory) / "production" / "time_to_warning.joblib").exists())
            self.assertTrue((Path(directory) / "production" / "time_to_critical.joblib").exists())
            production = registry.load_metadata("production")
            self.assertEqual(production["forecast_sources"]["time_to_critical"]["source"], "ml")
            self.assertEqual(production["forecast_sources"]["time_to_warning"]["source"], "statistical")

    def test_partial_promotion_preserves_existing_other_target_and_runtime_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            base_metadata = {
                "model_version": "warning-v1",
                "targets": {"time_to_warning": {}},
                "probability_targets": {},
                "feature_names": ["f"],
                "input_schema_version": self.config.ml.schema_version,
                "threshold_config_version": self.config.threshold_config_version,
                "lifecycle_config_version": self.config.lifecycle_config_version,
                "synthetic_data_only": True,
            }
            registry.save_candidate(
                {"time_to_warning": ConstantProbabilityClassifier(0.25)}, base_metadata
            )
            registry.promote_targets(["time_to_warning"])
            registry.save_candidate(
                {"time_to_critical": ConstantProbabilityClassifier(0.75)},
                {**base_metadata, "model_version": "critical-v2", "targets": {"time_to_critical": {}}},
            )
            registry.promote_targets(["time_to_critical"])
            self.assertTrue((Path(directory) / "production" / "time_to_warning.joblib").exists())
            self.assertTrue((Path(directory) / "production" / "time_to_critical.joblib").exists())
            production = registry.load_metadata("production")
            self.assertEqual(set(production["forecast_sources"]), {"time_to_warning", "time_to_critical"})
            forecaster = MLForecaster(self.config, ["f"], directory)
            forecast = forecaster.predict({"f": 1.0})
            self.assertIsNotNone(forecast.time_to_warning_hours)
            self.assertIsNotNone(forecast.time_to_critical_hours)
            self.assertEqual(forecast.maturity_stage, "deployed")

    def test_inconsistent_forecasts_are_withheld_not_swapped(self) -> None:
        warning, critical = ConditionMonitor._enforce_forecast_consistency(
            Status.NORMAL,
            ForecastDecision(8.0, None, "high", "ml", "warning"),
            ForecastDecision(4.0, None, "high", "ml", "critical"),
        )
        self.assertIsNone(warning.primary_hours)
        self.assertIsNone(critical.primary_hours)
        self.assertTrue(warning.withheld)
        self.assertEqual(warning.withholding_reason, "logical_consistency")
        self.assertIn("logically inconsistent", warning.reason)

    def test_offline_boundary_resets_state_and_assigns_new_lifecycle(self) -> None:
        record = LifecycleRecord(
            lifecycle_id="lifecycle_0001",
            start_timestamp=self.start,
            end_timestamp=self.start + timedelta(minutes=2),
            duration_hours=2 / 60,
            highest_status="CRITICAL",
            first_warning_timestamp=self.start,
            first_critical_timestamp=self.start,
            critical_reached=True,
            inferred_reset_timestamp=self.start + timedelta(minutes=4),
            reset_confidence="HIGH",
            reset_reason="confirmed test reset",
            pre_reset_values={"vibration_mps2": 4.5, "temperature_c": 85, "current_ampere": 9},
            post_reset_values={"vibration_mps2": 1.0, "temperature_c": 30, "current_ampere": 4},
        )
        plan = OfflineLifecyclePlan(self.start, (record,))
        with tempfile.TemporaryDirectory() as directory:
            monitor = ConditionMonitor(self.config, models_root=directory, lifecycle_plan=plan)
            monitor.process(SensorReading(self.start, 4.5, 85.0, 9.0))
            monitor.process(SensorReading(self.start + timedelta(minutes=1), 4.4, 84.0, 8.8))
            boundary = monitor.process(SensorReading(self.start + timedelta(minutes=2), 3.2, 30.0, 4.0))
            self.assertEqual(boundary.lifecycle_id, "lifecycle_0002")
            self.assertEqual(boundary.lifecycle_state, "RESET_CANDIDATE")
            self.assertEqual(boundary.raw_status, Status.WARNING)
            self.assertIsNone(boundary.final_time_to_warning_hours)
            self.assertEqual(boundary.forecast_confidence, "UNAVAILABLE")
            self.assertEqual(boundary.forecast_reason, "Reset confirmation is in progress.")
            self.assertEqual(len(monitor.timestamps), 1)
            self.assertTrue(all(len(history) == 1 for history in monitor.features.history.values()))
            self.assertTrue(all(len(window) == 1 for window in monitor.smoother.windows.values()))

    def test_sqlite_model_record_contains_auditable_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.db"
            store = SQLiteResultStore(path)
            metadata = {
                "model_version": "v1",
                "training_lifecycle_ids": ["a"],
                "validation_lifecycle_ids": ["b"],
                "test_lifecycle_ids": ["c"],
                "feature_names": ["feature"],
                "synthetic_data_only": True,
                "targets": {
                    "time_to_warning": {
                        "validation_metrics": {"macro_lifecycle_mae_hours": 1.0},
                        "test_metrics": {"macro_lifecycle_mae_hours": 1.2},
                        "baseline_metrics": {"test": {"macro_lifecycle_mae_hours": 2.0}},
                    }
                },
            }
            store.save_model_target_records(metadata, "candidate")
            store.close()
            connection = sqlite3.connect(path)
            row = connection.execute(
                "SELECT target_name, stage, test_lifecycle_ids_json, synthetic_data_only, metadata_json FROM model_records"
            ).fetchone()
            connection.close()
            self.assertEqual(row[:4], ("time_to_warning", "candidate", '["c"]', 1))
            self.assertEqual(json.loads(row[4])["feature_schema"], ["feature"])


if __name__ == "__main__":
    unittest.main()
