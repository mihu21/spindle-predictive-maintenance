from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.cli import _count_csv_rows, build_parser


class CliDefaultTests(unittest.TestCase):
    def test_replay_defaults_to_complete_supplied_dataset(self) -> None:
        args = build_parser().parse_args(["replay"])
        self.assertIsNone(args.limit)
        self.assertTrue(args.input.endswith("spindle_predictive_maintenance_10000_unlabeled.csv"))
        self.assertEqual(args.csv, "output/replay_results.csv")
        self.assertIsNone(args.detailed_csv)
        self.assertFalse(args.verbose)
        self.assertEqual(args.progress_every, 1000)

    def test_simulation_defaults_to_reference_dataset_row_count(self) -> None:
        args = build_parser().parse_args(["simulate"])
        self.assertIsNone(args.samples)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "small.csv"
            fixture.write_text("timestamp,vibration_mps2,temperature_c,current_ampere,health_status\n2026-01-01,1,30,4,NORMAL\n", encoding="utf-8")
            self.assertEqual(_count_csv_rows(fixture), 1)
        self.assertEqual(args.csv, "output/monitor_results.csv")


if __name__ == "__main__":
    unittest.main()
