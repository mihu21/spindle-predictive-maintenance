from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.config import load_config
from spindle_monitor.ml_forecaster import MLForecaster
from spindle_monitor.models import SensorReading, Status
from spindle_monitor.monitor import ConditionMonitor


class MonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "config" / "thresholds.json")
        self.monitor = ConditionMonitor(self.config)
        self.start = datetime(2026, 1, 1)

    def reading(self, minute: int, vibration: float, temperature: float, current: float) -> SensorReading:
        return SensorReading(
            timestamp=self.start + timedelta(minutes=minute),
            vibration_mps2=vibration,
            temperature_c=temperature,
            current_ampere=current,
        )

    def test_critical_rule_is_immediate(self) -> None:
        self.monitor.process(self.reading(0, 2.0, 40.0, 5.0))
        result = self.monitor.process(self.reading(1, 4.5, 40.0, 5.0))
        self.assertEqual(result.raw_status, Status.CRITICAL)
        self.assertEqual(result.effective_status, Status.CRITICAL)
        self.assertEqual(result.worst_sensor, "vibration_mps2")

    def test_warning_requires_persistent_votes_after_normal_start(self) -> None:
        self.monitor.process(self.reading(0, 2.0, 40.0, 5.0))
        result_1 = self.monitor.process(self.reading(1, 3.2, 40.0, 5.0))
        result_2 = self.monitor.process(self.reading(2, 3.2, 40.0, 5.0))
        result_3 = self.monitor.process(self.reading(3, 3.2, 40.0, 5.0))
        self.assertEqual(result_1.effective_status, Status.NORMAL)
        self.assertEqual(result_2.effective_status, Status.NORMAL)
        self.assertEqual(result_3.effective_status, Status.WARNING)

    def test_worst_sensor_uses_raw_critical_condition_not_smoothed_severity(self) -> None:
        self.monitor.process(self.reading(0, 2.0, 40.0, 5.0))
        result = self.monitor.process(self.reading(1, 3.7, 81.0, 5.0))
        self.assertEqual(result.raw_status, Status.CRITICAL)
        self.assertEqual(result.worst_sensor, "temperature_c")

    def test_reset_candidate_suppresses_all_machine_forecasts(self) -> None:
        lifecycle = replace(
            self.config.lifecycle,
            minimum_degraded_minutes=1,
            minimum_lifecycle_hours=0,
            reset_cooldown_hours=0,
        )
        monitoring = replace(
            self.config.monitoring,
            warning_votes_required=1,
            critical_recovery_readings=1,
        )
        config = replace(self.config, lifecycle=lifecycle, monitoring=monitoring)
        monitor = ConditionMonitor(config, models_root=ROOT / "models")
        monitor.process(SensorReading(self.start, 4.8, 90.0, 9.0))
        monitor.process(SensorReading(self.start + timedelta(minutes=1), 4.8, 90.0, 9.0))
        result = monitor.process(SensorReading(self.start + timedelta(minutes=2), 1.2, 33.0, 4.2))
        self.assertEqual(result.lifecycle_state, "RESET_CANDIDATE")
        self.assertIsNone(result.statistical_warning_forecast.estimated_hours)
        self.assertIsNone(result.statistical_critical_forecast.estimated_hours)
        self.assertIsNone(result.final_time_to_warning_hours)
        self.assertIsNone(result.final_time_to_critical_hours)
        self.assertEqual(result.forecast_confidence, "UNAVAILABLE")
        self.assertEqual(result.forecast_reason, "Reset confirmation is in progress.")
        self.assertTrue(result.forecast_withheld)
        self.assertEqual(result.withholding_reasons, ("reset_suppression",))

    def test_batched_offline_selection_matches_single_row_runtime_policy(self) -> None:
        direct_monitor = ConditionMonitor(self.config, models_root=ROOT / "models")
        base_monitor = ConditionMonitor(
            self.config, models_root=ROOT / "models", enable_ml=False
        )
        reading = self.reading(0, 2.0, 40.0, 5.0)
        direct = direct_monitor.process(reading)
        base = base_monitor.process(reading)
        forecaster = MLForecaster(self.config, base_monitor.features.names, ROOT / "models")
        batched = base_monitor.apply_ml_forecast(
            base, forecaster.predict_many([base.features])[0]
        )
        self.assertEqual(direct.final_time_to_warning_hours, batched.final_time_to_warning_hours)
        self.assertEqual(direct.final_time_to_critical_hours, batched.final_time_to_critical_hours)
        self.assertEqual(direct.forecast_confidence, batched.forecast_confidence)
        self.assertEqual(direct.withholding_reasons, batched.withholding_reasons)


if __name__ == "__main__":
    unittest.main()
