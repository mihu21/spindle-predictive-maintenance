from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.config import load_config
from spindle_monitor.models import Status
from spindle_monitor.rules import severity_for_value, status_for_value


class RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "config" / "thresholds.json")

    def test_dataset_matching_boundary_policy(self) -> None:
        sensor = self.config.sensors["vibration_mps2"]
        self.assertEqual(status_for_value(3.0, sensor, self.config), Status.NORMAL)
        self.assertEqual(status_for_value(3.01, sensor, self.config), Status.WARNING)
        self.assertEqual(status_for_value(4.0, sensor, self.config), Status.WARNING)
        self.assertEqual(status_for_value(4.01, sensor, self.config), Status.CRITICAL)

    def test_severity_hits_named_bands(self) -> None:
        sensor = self.config.sensors["temperature_c"]
        self.assertAlmostEqual(severity_for_value(sensor.healthy_baseline, sensor), 0.0)
        self.assertAlmostEqual(severity_for_value(sensor.warning, sensor), 0.6)
        self.assertAlmostEqual(severity_for_value(sensor.critical, sensor), 1.0)


if __name__ == "__main__":
    unittest.main()
