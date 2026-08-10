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
from spindle_monitor.lifecycle_detector import LifecycleDetector
from spindle_monitor.models import Status


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        base = load_config(ROOT / "config" / "thresholds.json")
        lifecycle = replace(
            base.lifecycle,
            minimum_degraded_minutes=2,
            normal_confirmation_minutes=2,
            medium_confirmation_multiplier=1.5,
            pre_reset_window_minutes=5,
            post_reset_window_minutes=5,
            minimum_lifecycle_hours=0,
            reset_cooldown_hours=0,
            minimum_sensor_drop_fraction=0.10,
            minimum_overall_severity_drop=0.20,
            healthy_baseline_tolerance=0.5,
            minimum_critical_duration_before_reset_minutes=0,
        )
        self.config = replace(base, lifecycle=lifecycle)
        self.start = datetime(2026, 1, 1)

    def update(self, detector, minute, status, values, severity):
        return detector.update(
            timestamp=self.start + timedelta(minutes=minute),
            status=status,
            smoothed_values=values,
            maximum_severity=severity,
        )

    def test_temporary_warning_to_normal_is_rejected(self) -> None:
        detector = LifecycleDetector(self.config)
        high = {"vibration_mps2": 3.5, "temperature_c": 70.0, "current_ampere": 7.0}
        low = {"vibration_mps2": 1.2, "temperature_c": 33.0, "current_ampere": 4.2}
        for minute in range(3):
            self.update(detector, minute, Status.WARNING, high, 0.8)
        self.update(detector, 3, Status.NORMAL, low, 0.1)
        snapshot = self.update(detector, 4, Status.WARNING, high, 0.8)
        self.assertEqual(snapshot.reset_confidence, "REJECTED")
        self.assertEqual(len(detector.completed), 0)

    def test_critical_to_sustained_normal_creates_high_confidence_reset(self) -> None:
        detector = LifecycleDetector(self.config)
        high = {"vibration_mps2": 4.8, "temperature_c": 90.0, "current_ampere": 9.0}
        low = {"vibration_mps2": 1.2, "temperature_c": 33.0, "current_ampere": 4.2}
        for minute in range(4):
            self.update(detector, minute, Status.CRITICAL, high, 1.2)
        confirmed = None
        for minute in range(4, 8):
            snapshot = self.update(detector, minute, Status.NORMAL, low, 0.1)
            if snapshot.completed_lifecycle is not None:
                confirmed = snapshot
        assert confirmed is not None
        self.assertEqual(confirmed.reset_confidence, "HIGH")
        self.assertIsNotNone(confirmed.completed_lifecycle)
        self.assertEqual(confirmed.lifecycle_id, "lifecycle_0002")

    def test_one_sensor_improvement_does_not_create_reset(self) -> None:
        detector = LifecycleDetector(self.config)
        high = {"vibration_mps2": 4.8, "temperature_c": 90.0, "current_ampere": 9.0}
        partial = {"vibration_mps2": 1.2, "temperature_c": 88.0, "current_ampere": 8.8}
        for minute in range(4):
            self.update(detector, minute, Status.CRITICAL, high, 1.2)
        for minute in range(4, 10):
            snapshot = self.update(detector, minute, Status.NORMAL, partial, 0.9)
        self.assertEqual(len(detector.completed), 0)
        self.assertIn(snapshot.lifecycle_state, {"RESET_CANDIDATE", "DEGRADING"})

    def test_warning_only_recovery_can_create_medium_confidence_reset(self) -> None:
        detector = LifecycleDetector(self.config)
        high = {"vibration_mps2": 3.7, "temperature_c": 72.0, "current_ampere": 7.2}
        low = {"vibration_mps2": 1.2, "temperature_c": 33.0, "current_ampere": 4.2}
        for minute in range(4):
            self.update(detector, minute, Status.WARNING, high, 0.85)
        confirmed = None
        for minute in range(4, 10):
            snapshot = self.update(detector, minute, Status.NORMAL, low, 0.1)
            if snapshot.completed_lifecycle is not None:
                confirmed = snapshot
        assert confirmed is not None
        self.assertEqual(confirmed.reset_confidence, "MEDIUM")

    def test_minimum_lifecycle_duration_blocks_early_reset(self) -> None:
        detector = LifecycleDetector(
            replace(
                self.config,
                lifecycle=replace(self.config.lifecycle, minimum_lifecycle_hours=1.0),
            )
        )
        high = {"vibration_mps2": 4.8, "temperature_c": 90.0, "current_ampere": 9.0}
        low = {"vibration_mps2": 1.2, "temperature_c": 33.0, "current_ampere": 4.2}
        for minute in range(4):
            self.update(detector, minute, Status.CRITICAL, high, 1.2)
        for minute in range(4, 12):
            self.update(detector, minute, Status.NORMAL, low, 0.1)
        self.assertEqual(len(detector.completed), 0)


if __name__ == "__main__":
    unittest.main()
