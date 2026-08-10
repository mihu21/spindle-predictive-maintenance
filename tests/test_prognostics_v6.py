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
from spindle_monitor.models import MLForecast, PrognosticForecast, SensorReading, Status
from spindle_monitor.monitor import ConditionMonitor


class PrognosticsV6Tests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_config(ROOT / "config" / "thresholds.json")
        prognostics = replace(
            config.prognostics,
            minimum_history_minutes=60.0,
            minimum_trend_windows=1,
            minimum_samples_per_trend_window=30,
            minimum_window_coverage_fraction=0.80,
            minimum_window_history_fraction=0.80,
            high_confidence_history_minutes=180.0,
            high_confidence_minimum_trend_windows=2,
        )
        self.config = replace(config, prognostics=prognostics)
        self.start = datetime(2026, 1, 1)

    def _run(self, generator, minutes: int = 240):
        monitor = ConditionMonitor(self.config, enable_ml=False)
        result = None
        for minute in range(minutes):
            vibration, temperature, current = generator(minute)
            result = monitor.process(
                SensorReading(
                    self.start + timedelta(minutes=minute),
                    vibration,
                    temperature,
                    current,
                )
            )
        assert result is not None
        return monitor, result

    def test_flat_healthy_history_has_negligible_crossing_risk(self) -> None:
        _, result = self._run(lambda _: (1.2, 35.0, 4.3))
        forecast = result.prognostic_forecast
        self.assertIn(forecast.confidence, {"medium", "high"})
        self.assertLess(forecast.probability_warning[12], 0.05)
        self.assertLess(forecast.probability_critical[24], 0.05)
        self.assertIsNone(forecast.time_to_warning_hours)
        self.assertIsNone(forecast.time_to_critical_hours)
        self.assertEqual(result.warning_primary_source, "probabilistic_degradation")
        self.assertEqual(result.maintenance_urgency, "NORMAL_MONITORING")

    def test_persistent_rise_produces_monotonic_probabilities_and_warning_before_critical(self) -> None:
        _, result = self._run(
            lambda minute: (
                1.2 + minute * 0.004,
                35.0 + minute * 0.03,
                4.3 + minute * 0.002,
            )
        )
        forecast = result.prognostic_forecast
        warning = [forecast.probability_warning[h] for h in (6, 12, 24)]
        critical = [forecast.probability_critical[h] for h in (6, 12, 24)]
        self.assertEqual(warning, sorted(warning))
        self.assertEqual(critical, sorted(critical))
        self.assertIsNotNone(forecast.time_to_warning_hours)
        self.assertIsNotNone(forecast.time_to_critical_hours)
        self.assertLess(forecast.time_to_warning_hours, forecast.time_to_critical_hours)
        self.assertEqual(result.model_stage, "model_based_prognostics")
        self.assertTrue(result.model_version.startswith("probabilistic_degradation_v6"))

    def test_raw_critical_still_forces_immediate_safety_output(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        result = monitor.process(SensorReading(self.start, 4.5, 40.0, 5.0))
        self.assertEqual(result.raw_status, Status.CRITICAL)
        self.assertEqual(result.final_time_to_warning_hours, 0.0)
        self.assertEqual(result.final_time_to_critical_hours, 0.0)
        self.assertEqual(result.maintenance_urgency, "IMMEDIATE_MAINTENANCE_REQUIRED")

    def test_warning_probability_uses_shorter_activation_persistence(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        forecast = PrognosticForecast(
            probability_warning={6: 0.1, 12: 0.9, 24: 0.9},
            probability_critical={6: 0.1, 12: 0.1, 24: 0.1},
            confidence="high",
            selected_thresholds={"probability_warning_12h": 0.5},
            withheld=False,
        )
        first = monitor._persistent_prognostic_crossings(self.start, forecast)
        before = monitor._persistent_prognostic_crossings(
            self.start + timedelta(minutes=9), forecast
        )
        after = monitor._persistent_prognostic_crossings(
            self.start + timedelta(minutes=10), forecast
        )
        self.assertEqual(first, ())
        self.assertEqual(before, ())
        self.assertEqual(after, ("probability_warning_12h",))

    def test_critical_probability_keeps_30_minute_activation_persistence(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        forecast = PrognosticForecast(
            probability_warning={6: 0.1, 12: 0.1, 24: 0.1},
            probability_critical={6: 0.1, 12: 0.1, 24: 0.9},
            confidence="high",
            selected_thresholds={"probability_critical_24h": 0.5},
            withheld=False,
        )
        self.assertEqual(monitor._persistent_prognostic_crossings(self.start, forecast), ())
        self.assertEqual(
            monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=29), forecast),
            (),
        )
        self.assertEqual(
            monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=30), forecast),
            ("probability_critical_24h",),
        )

    def test_brief_warning_spike_below_10_minutes_does_not_activate(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        def warning(value: float) -> PrognosticForecast:
            return PrognosticForecast(
                probability_warning={6: 0.1, 12: value, 24: value},
                probability_critical={6: 0.1, 12: 0.1, 24: 0.1},
                confidence="high",
                selected_thresholds={"probability_warning_12h": 0.5},
                withheld=False,
            )
        self.assertEqual(monitor._persistent_prognostic_crossings(self.start, warning(0.9)), ())
        self.assertEqual(monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=9), warning(0.9)), ())
        self.assertEqual(monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=10), warning(0.2)), ())
        self.assertEqual(monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=20), warning(0.9)), ())


    def test_probability_release_requires_sustained_low_probability(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        high = PrognosticForecast(
            probability_warning={6: 0.1, 12: 0.9, 24: 0.9},
            probability_critical={6: 0.1, 12: 0.1, 24: 0.1},
            confidence="high",
            selected_thresholds={"probability_warning_12h": 0.5},
            withheld=False,
        )
        low = PrognosticForecast(
            probability_warning={6: 0.1, 12: 0.2, 24: 0.2},
            probability_critical={6: 0.1, 12: 0.1, 24: 0.1},
            confidence="high",
            selected_thresholds={"probability_warning_12h": 0.5},
            withheld=False,
        )
        monitor._persistent_prognostic_crossings(self.start, high)
        active = monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=30), high)
        self.assertEqual(active, ("probability_warning_12h",))
        brief_dip = monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=45), low)
        self.assertEqual(brief_dip, ("probability_warning_12h",))
        still_active = monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=104), low)
        self.assertEqual(still_active, ("probability_warning_12h",))
        released = monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=105), low)
        self.assertEqual(released, ())

    def test_probability_release_dip_cancels_when_probability_recovers(self) -> None:
        monitor = ConditionMonitor(self.config, enable_ml=False)
        def forecast(value: float) -> PrognosticForecast:
            return PrognosticForecast(
                probability_warning={6: 0.1, 12: value, 24: value},
                probability_critical={6: 0.1, 12: 0.1, 24: 0.1},
                confidence="high",
                selected_thresholds={"probability_warning_12h": 0.5},
                withheld=False,
            )
        monitor._persistent_prognostic_crossings(self.start, forecast(0.9))
        self.assertEqual(monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=30), forecast(0.9)), ("probability_warning_12h",))
        self.assertEqual(monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=45), forecast(0.2)), ("probability_warning_12h",))
        self.assertEqual(monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=60), forecast(0.4)), ("probability_warning_12h",))
        self.assertEqual(monitor._persistent_prognostic_crossings(self.start + timedelta(minutes=130), forecast(0.4)), ("probability_warning_12h",))

    def test_legacy_ml_attachment_is_audit_only_in_v6(self) -> None:
        monitor, result = self._run(
            lambda minute: (
                1.2 + minute * 0.004,
                35.0 + minute * 0.03,
                4.3 + minute * 0.002,
            )
        )
        original_warning = result.final_time_to_warning_hours
        original_critical = result.final_time_to_critical_hours
        attached = monitor.apply_ml_forecast(
            result,
            MLForecast(
                time_to_warning_hours=999.0,
                time_to_critical_hours=999.0,
                maturity_stage="deployed",
                confidence="high",
                reason="legacy research model",
            ),
        )
        self.assertEqual(attached.final_time_to_warning_hours, original_warning)
        self.assertEqual(attached.final_time_to_critical_hours, original_critical)
        self.assertEqual(attached.ml_forecast.time_to_warning_hours, 999.0)
        self.assertEqual(attached.warning_primary_source, "probabilistic_degradation")


if __name__ == "__main__":
    unittest.main()
