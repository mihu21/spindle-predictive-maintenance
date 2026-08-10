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
from spindle_monitor.kalman import ConstantVelocityKalmanFilter
from spindle_monitor.models import MLForecast, SensorReading, StatisticalForecast, Status
from spindle_monitor.monitor import ConditionMonitor
from spindle_monitor.smoothing import SignalSmoother


class SmoothingAndForecastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "config" / "thresholds.json")
        self.start = datetime(2026, 1, 1)

    def test_rolling_median_resists_single_spike(self) -> None:
        smoother = SignalSmoother(self.config)
        output = None
        for vibration in [1.0, 1.1, 10.0, 1.2, 1.1]:
            output = smoother.update(
                {
                    "vibration_mps2": vibration,
                    "temperature_c": 30.0,
                    "current_ampere": 4.0,
                },
                60.0,
            )
        assert output is not None
        self.assertAlmostEqual(output["vibration_mps2"].rolling_median, 1.1)
        self.assertLess(output["vibration_mps2"].fast_ewma, 2.0)

    def test_ewma_update_uses_configured_alpha(self) -> None:
        smoothing = replace(
            self.config.smoothing,
            rolling_median_window=1,
            fast_ewma_alpha=0.5,
            slow_ewma_alpha=0.25,
        )
        config = replace(self.config, smoothing=smoothing)
        smoother = SignalSmoother(config)
        smoother.update(
            {"vibration_mps2": 1.0, "temperature_c": 30.0, "current_ampere": 4.0},
            60.0,
        )
        output = smoother.update(
            {"vibration_mps2": 3.0, "temperature_c": 30.0, "current_ampere": 4.0},
            60.0,
        )
        self.assertAlmostEqual(output["vibration_mps2"].fast_ewma, 2.0)
        self.assertAlmostEqual(output["vibration_mps2"].slow_ewma, 1.5)

    def test_kalman_estimates_positive_rising_rate(self) -> None:
        kalman = ConstantVelocityKalmanFilter(5e-5, 0.01)
        rate = 0.0
        for index in range(40):
            _, rate = kalman.update(1.0 + index * 0.1, 60.0)
        self.assertGreater(rate * 3600.0, 0.0)

    def test_critical_status_forces_final_forecast_zero(self) -> None:
        monitor = ConditionMonitor(self.config, models_root=ROOT / "models")
        result = monitor.process(
            SensorReading(self.start, 4.5, 40.0, 5.0)
        )
        self.assertEqual(result.raw_status, Status.CRITICAL)
        self.assertEqual(result.final_time_to_warning_hours, 0.0)
        self.assertEqual(result.final_time_to_critical_hours, 0.0)
        self.assertEqual(result.final_forecast_hours, 0.0)

    def _decision(self, ml_hours: float, statistical_hours: float, target: str = "critical"):
        test_config = replace(
            self.config,
            ml=replace(
                self.config.ml,
                warning_disagreement_absolute_hours=1.0,
                critical_disagreement_absolute_hours=2.0,
                disagreement_relative_threshold=0.75,
            ),
        )
        monitor = ConditionMonitor(test_config, models_root=ROOT / "models")
        statistical = StatisticalForecast(
            statistical_hours, "vibration_mps2", "high", "baseline"
        )
        ml = MLForecast(
            time_to_warning_hours=ml_hours if target == "warning" else None,
            time_to_critical_hours=ml_hours if target == "critical" else None,
            maturity_stage="deployed",
            confidence="medium",
            reason="production model",
        )
        return monitor._combine_forecasts(Status.NORMAL, statistical, ml, target)

    def test_near_zero_difference_does_not_trigger_disagreement(self) -> None:
        decision = self._decision(0.0, 0.01, "warning")
        self.assertFalse(decision.strong_disagreement)
        self.assertEqual(decision.primary_hours, 0.0)
        self.assertAlmostEqual(decision.absolute_disagreement_hours, 0.01)
        self.assertAlmostEqual(decision.relative_disagreement, 1.0)

    def test_small_absolute_large_relative_difference_is_not_strong(self) -> None:
        decision = self._decision(0.1, 0.5, "warning")
        self.assertGreater(decision.relative_disagreement, 0.75)
        self.assertFalse(decision.strong_disagreement)

    def test_large_absolute_small_relative_difference_is_not_strong(self) -> None:
        decision = self._decision(100.0, 102.5, "warning")
        self.assertGreater(decision.absolute_disagreement_hours, 1.0)
        self.assertLess(decision.relative_disagreement, 0.75)
        self.assertFalse(decision.strong_disagreement)

    def test_strong_disagreement_withholds_actionable_forecast(self) -> None:
        decision = self._decision(100.0, 10.0)
        self.assertIsNone(decision.primary_hours)
        self.assertIsNone(decision.conservative_alert_hours)
        self.assertEqual(decision.confidence, "low")
        self.assertTrue(decision.strong_disagreement)
        self.assertTrue(decision.withheld)
        self.assertIn("disagreement", decision.reason.lower())

    def test_earlier_ml_with_severe_disagreement_is_withheld(self) -> None:
        decision = self._decision(4.0, 20.0)
        self.assertIsNone(decision.primary_hours)
        self.assertTrue(decision.withheld)

    def test_out_of_distribution_ml_withholds_actionable_forecast(self) -> None:
        monitor = ConditionMonitor(self.config, models_root=ROOT / "models")
        statistical = StatisticalForecast(8.0, "vibration_mps2", "high", "baseline")
        ml = MLForecast(
            time_to_critical_hours=4.0,
            maturity_stage="deployed",
            confidence="low",
            reason="outside",
            outside_training_distribution=True,
            outside_feature_fraction=0.2,
        )
        decision = monitor._combine_forecasts(Status.NORMAL, statistical, ml)
        self.assertIsNone(decision.primary_hours)
        self.assertEqual(decision.primary_source, "unavailable")
        self.assertTrue(decision.withheld)

    def test_warning_and_critical_are_independent_threshold_forecasts(self) -> None:
        monitor = ConditionMonitor(self.config, models_root=ROOT / "models")
        result = None
        for minute in range(80):
            result = monitor.process(
                SensorReading(
                    self.start + timedelta(minutes=minute),
                    1.5 + minute * 0.004,
                    38.0 + minute * 0.04,
                    4.4 + minute * 0.003,
                )
            )
        assert result is not None
        self.assertIsNotNone(result.statistical_warning_forecast)
        self.assertIsNotNone(result.statistical_critical_forecast)
        self.assertLessEqual(
            result.statistical_warning_forecast.estimated_hours,
            result.statistical_critical_forecast.estimated_hours,
        )

    def test_warning_status_forces_warning_forecast_to_zero_only(self) -> None:
        monitor = ConditionMonitor(self.config, models_root=ROOT / "models")
        result = monitor.process(SensorReading(self.start, 3.2, 40.0, 5.0))
        self.assertEqual(result.raw_status, Status.WARNING)
        self.assertEqual(result.final_time_to_warning_hours, 0.0)
        self.assertNotEqual(result.final_time_to_critical_hours, 0.0)


if __name__ == "__main__":
    unittest.main()
