from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.config import load_config
from spindle_monitor.models import SensorReading
from spindle_monitor.monitor import ConditionMonitor
from spindle_monitor.storage import CSVResultStore, result_to_detailed_row, result_to_flat_row


class CSVOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "config" / "thresholds.json")
        self.monitor = ConditionMonitor(self.config, models_root=ROOT / "models")
        self.start = datetime(2026, 1, 1)

    def _rising_result(self):
        result = None
        for minute in range(80):
            result = self.monitor.process(
                SensorReading(
                    timestamp=self.start + timedelta(minutes=minute),
                    vibration_mps2=1.5 + minute * 0.004,
                    temperature_c=38.0 + minute * 0.04,
                    current_ampere=4.4 + minute * 0.003,
                )
            )
        assert result is not None
        return result

    def test_default_row_matches_upgrade_schema(self) -> None:
        row = result_to_flat_row(self._rising_result(), source_label="normal")
        required_audit = {
            "timestamp", "row_provenance", "data_domain", "dataset_hash",
            "generator_version", "model_stage", "model_version", "raw_status",
            "stabilized_status", "event_status", "lifecycle_id", "lifecycle_state",
            "lifecycle_censored", "feature_available",
            "available_history_duration_seconds", "source_sampling_interval_seconds",
            "effective_resampling_interval_seconds",
            "sampling_interval_out_of_distribution", "prediction_warning_hours_raw",
            "prediction_critical_hours_raw", "prediction_confidence",
            "training_support_warning", "guardrail_result",
            "refusal_or_fallback_reason", "window_24h_mature",
        }
        self.assertTrue(required_audit.issubset(row))
        self.assertNotIn("health_percent", row)

    def test_detailed_row_contains_features_and_kalman_fields(self) -> None:
        row = result_to_detailed_row(self._rising_result(), source_label="normal")
        self.assertGreater(len(row), 100)
        self.assertIn("vibration_mps2__kalman_rate_per_hour", row)
        self.assertIn("feature__vibration_mps2__w60m__slope_per_hour", row)
        self.assertEqual(row["source_label"], "normal")

    def test_csv_store_writes_excel_readable_file(self) -> None:
        result = self._rising_result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            store = CSVResultStore(path)
            store.save(result, source_label="warning")
            store.close()
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertIn("lifecycle_id", rows[0])
            self.assertIn("ml_time_to_warning_hours", rows[0])
            self.assertIn("probability_critical_24h", rows[0])
            self.assertNotIn("ml_time_to_reset_hours", rows[0])
            self.assertGreaterEqual(len(rows[0]), 70)


if __name__ == "__main__":
    unittest.main()
