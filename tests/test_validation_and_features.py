from __future__ import annotations

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
from spindle_monitor.feature_engineering import FeatureGenerator
from spindle_monitor.io import read_csv_records
from spindle_monitor.models import SensorReading, Status
from spindle_monitor.smoothing import SignalSmoother
from spindle_monitor.validation import InputValidator


class ValidationAndFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "config" / "thresholds.json")
        self.start = datetime(2026, 1, 1)

    def test_duplicate_timestamp_is_invalid_and_does_not_advance(self) -> None:
        validator = InputValidator(self.config)
        reading = SensorReading(self.start, 1.0, 30.0, 4.0)
        self.assertTrue(validator.validate(reading).valid)
        duplicate = validator.validate(reading)
        self.assertFalse(duplicate.valid)
        self.assertIn("duplicate", duplicate.reason)
        next_reading = SensorReading(self.start + timedelta(minutes=1), 1.1, 30.1, 4.1)
        self.assertTrue(validator.validate(next_reading).valid)

    def test_feature_generation_is_deterministic_and_causal(self) -> None:
        generators = [FeatureGenerator(self.config), FeatureGenerator(self.config)]
        smoothers = [SignalSmoother(self.config), SignalSmoother(self.config)]
        outputs = []
        for generator, smoother in zip(generators, smoothers):
            result = None
            for minute in range(10):
                values = {
                    "vibration_mps2": 1.0 + minute * 0.1,
                    "temperature_c": 30.0 + minute,
                    "current_ampere": 4.0 + minute * 0.05,
                }
                signals = smoother.update(values, 60.0)
                statuses = {key: Status.NORMAL for key in values}
                result = generator.update(
                    timestamp=self.start + timedelta(minutes=minute),
                    raw_values=values,
                    smoothed=signals,
                    sensor_statuses=statuses,
                    overall_status=Status.NORMAL,
                    elapsed_lifecycle_hours=minute / 60.0,
                )
            outputs.append(result)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(list(outputs[0]), generators[0].names)

    def test_fixed_five_column_input_schema_is_required_and_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.csv"
            valid.write_text(
                "timestamp,vibration_mps2,temperature_c,current_ampere,health_status\n"
                "2026-01-01T00:00:00,1.0,30.0,4.0,normal\n",
                encoding="utf-8",
            )
            records = list(read_csv_records(valid, self.config))
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0].valid)
            self.assertEqual(records[0].source_label, "normal")

            missing = Path(directory) / "missing.csv"
            missing.write_text(
                "timestamp,vibration_mps2,temperature_c,current_ampere\n"
                "2026-01-01T00:00:00,1.0,30.0,4.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "health_status"):
                list(read_csv_records(missing, self.config))


if __name__ == "__main__":
    unittest.main()
