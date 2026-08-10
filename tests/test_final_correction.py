from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import warnings
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.config import load_config
from spindle_monitor.data_profile import profile_data
from spindle_monitor.environment import environment_compatibility, runtime_environment
from spindle_monitor.model_registry import ModelRegistry
from spindle_monitor.models import SensorReading
from spindle_monitor.monitor import ConditionMonitor
from spindle_monitor.offline import prepare_offline_replay, read_validated_input
from spindle_monitor.storage import result_to_flat_row


HEADER = "timestamp,vibration_mps2,temperature_c,current_ampere,health_status\n"


class FinalCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "config" / "thresholds.json")

    def _profile(self, body: str):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "input.csv"
        path.write_text(HEADER + body, encoding="utf-8")
        return directory, path, profile_data(path, self.config)

    def test_out_of_range_sensor_refuses_replay(self) -> None:
        directory, _, report = self._profile(
            "2026-01-01 00:00:00,21,30,4,normal\n"
        )
        try:
            self.assertFalse(report["suitability"]["replay"])
            self.assertEqual(report["total_invalid_rows"], 1)
            self.assertTrue(any("outside the configured" in reason for reason in report["input_validation"]["invalid_reason_counts"]))
        finally:
            directory.cleanup()

    def test_impossible_sensor_jump_refuses_replay(self) -> None:
        directory, _, report = self._profile(
            "2026-01-01 00:00:00,1,30,4,normal\n"
            "2026-01-01 00:01:00,20,30,4,critical\n"
        )
        try:
            self.assertFalse(report["suitability"]["replay"])
            self.assertTrue(any("possible sensor fault" in reason for reason in report["input_validation"]["invalid_reason_counts"]))
            examples = report["input_validation"]["representative_failures"]
            self.assertEqual(next(iter(examples.values()))[0]["row_number"], 3)
        finally:
            directory.cleanup()

    def test_profiler_and_replay_share_row_validity(self) -> None:
        directory, path, report = self._profile(
            "2026-01-01 00:00:00,1,30,4,normal\n"
            "2026-01-01 00:01:00,20,30,4,critical\n"
            "2026-01-01 00:02:00,1.1,31,4.1,normal\n"
        )
        try:
            replay = read_validated_input(path, self.config)
            self.assertEqual(report["input_validation"]["valid_rows"], sum(value.valid for value in replay))
            self.assertEqual(report["total_invalid_rows"], sum(not value.valid for value in replay))
        finally:
            directory.cleanup()

    def test_metadata_states_are_causal_without_changing_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "states.csv"
            start = datetime(2026, 1, 1)
            rows = []
            statuses = ["normal", "warning", "warning", "warning", "warning",
                        "critical", "critical", "critical", "normal", "normal",
                        "normal", "normal"]
            for minute, status in enumerate(statuses):
                values = {
                    "normal": (1.0, 30.0, 4.0),
                    "warning": (3.5, 65.0, 6.5),
                    "critical": (4.5, 85.0, 8.5),
                }[status]
                rows.append([
                    (start + timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M:%S"),
                    *values, status,
                ])
            with path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow(["timestamp", "vibration_mps2", "temperature_c", "current_ampere", "health_status"])
                writer.writerows(rows)
            # Put the explicit reset on the final observation as generated
            # realistic datasets do for their final lifecycle.
            boundary = start + timedelta(minutes=11)
            metadata = {
                "data_domain": "realistic_synthetic",
                "lifecycles": [{
                    "lifecycle_id": "lifecycle_0001",
                    "start_timestamp": start.isoformat(),
                    "end_timestamp": boundary.isoformat(),
                    "duration_hours": 11 / 60,
                    "highest_status": "CRITICAL",
                    "first_warning_timestamp": (start + timedelta(minutes=4)).isoformat(),
                    "first_critical_timestamp": (start + timedelta(minutes=7)).isoformat(),
                    "critical_reached": True,
                    "maintenance_record_timestamp": boundary.isoformat(),
                }],
            }
            path.with_name("states_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            monitoring = replace(
                self.config.monitoring,
                warning_vote_window=1, warning_votes_required=1,
                normal_recovery_readings=1, critical_recovery_readings=1,
            )
            lifecycle = replace(
                self.config.lifecycle,
                minimum_degraded_minutes=1,
                warning_confirmation_minutes=3,
                critical_confirmation_minutes=2,
                minimum_lifecycle_hours=0,
                reset_cooldown_hours=0,
                minimum_critical_duration_before_reset_minutes=0,
                normal_confirmation_minutes=2,
                normal_reset_confirmation_minutes=2,
            )
            config = replace(self.config, monitoring=monitoring, lifecycle=lifecycle)
            _, plan = prepare_offline_replay(path, config, root / "models")
            self.assertEqual(plan.boundary_timestamps, (boundary,))
            self.assertEqual(plan.snapshot(start).lifecycle_state, "HEALTHY")
            self.assertEqual(plan.snapshot(start + timedelta(minutes=2)).lifecycle_state, "DEGRADING")
            self.assertEqual(plan.snapshot(start + timedelta(minutes=4)).lifecycle_state, "WARNING")
            self.assertEqual(plan.snapshot(start + timedelta(minutes=7)).lifecycle_state, "CRITICAL")
            self.assertEqual(plan.snapshot(start + timedelta(minutes=8)).lifecycle_state, "RECOVERY_CONFIRMATION")
            reset = plan.snapshot(boundary)
            self.assertEqual(reset.lifecycle_state, "RESET_COMPLETE")
            self.assertEqual(reset.lifecycle_id, "lifecycle_0002")
            self.assertIsNotNone(reset.completed_lifecycle)

    def test_first_row_source_interval_is_unknown(self) -> None:
        monitor = ConditionMonitor(self.config, models_root=ROOT / "models", enable_ml=False)
        result = monitor.process(SensorReading(datetime(2026, 1, 1), 1.0, 30.0, 4.0))
        self.assertIsNone(result.source_sampling_interval_seconds)
        self.assertIsNone(result_to_flat_row(result)["source_sampling_interval_seconds"])
        self.assertEqual(result.sampling_gap_seconds, 0.0)

    def test_environment_compatibility_detects_material_mismatch(self) -> None:
        environment = runtime_environment()
        self.assertTrue(environment_compatibility(environment)["compatible"])
        incompatible = json.loads(json.dumps(environment))
        incompatible["packages"]["scikit-learn"] = "0.24.0"
        result = environment_compatibility(incompatible)
        self.assertFalse(result["compatible"])
        self.assertIn("scikit-learn", " ".join(result["reasons"]))

    def test_model_loader_refuses_materially_incompatible_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            training = runtime_environment()
            runtime_major = int(training["python_version"].split(".", 1)[0])
            training["python_version"] = f"{runtime_major + 1}.0.0"
            registry.save_candidate(
                {"time_to_warning": {"model": "never loaded"}},
                {
                    "model_version": "incompatible-test",
                    "training_environment": training,
                    "targets": {"time_to_warning": {}},
                    "probability_targets": {},
                },
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = registry.load_model("time_to_warning", stage="candidate")
            self.assertIsNone(loaded)
            self.assertTrue(any("materially incompatible" in str(item.message) for item in caught))
            self.assertIn("model_load_rejected", registry.audit_log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
