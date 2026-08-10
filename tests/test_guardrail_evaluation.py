from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.guardrail_evaluation import GuardrailThresholds, evaluate_thresholds


class GuardrailEvaluationTests(unittest.TestCase):
    def row(
        self,
        lifecycle: str,
        *,
        ood: bool = False,
        warning_statistical: float | None = 5.0,
        critical_statistical: float | None = 8.0,
    ):
        return {
            "lifecycle_id": lifecycle,
            "timestamp": "2026-01-01T00:00:00",
            "out_of_distribution": ood,
            "outside_feature_fraction": 0.2 if ood else 0.0,
            "warning": {"actual": 4.0, "ml": 4.5, "statistical": warning_statistical},
            "critical": {"actual": 7.0, "ml": 7.5, "statistical": critical_statistical},
        }

    def test_report_contains_required_metrics_and_per_lifecycle_results(self) -> None:
        report = evaluate_thresholds(
            [self.row("one"), self.row("two")],
            GuardrailThresholds(1.0, 2.0, 0.75),
        )
        for target in ("warning", "critical"):
            self.assertIn("forecast_coverage_pct", report[target])
            self.assertIn("mae_hours", report[target])
            self.assertIn("bias_hours", report[target])
            self.assertIn("p90_absolute_error_hours", report[target])
            self.assertIn("low_confidence_pct", report[target])
            self.assertIn("withheld_pct", report[target])
            self.assertIn("unsafe_late_prediction_rate_pct", report[target])
            self.assertEqual(set(report[target]["per_lifecycle"]), {"one", "two"})

    def test_out_of_distribution_withholds_actionable_forecast(self) -> None:
        report = evaluate_thresholds(
            [self.row("one", ood=True)], GuardrailThresholds(1.0, 2.0, 0.75)
        )
        self.assertEqual(report["warning"]["forecast_coverage_pct"], 0.0)
        self.assertEqual(report["critical"]["forecast_coverage_pct"], 0.0)
        self.assertEqual(report["row_withheld_pct"], 100.0)

    def test_out_of_distribution_without_fallback_is_withheld(self) -> None:
        report = evaluate_thresholds(
            [self.row("one", ood=True, warning_statistical=None, critical_statistical=None)],
            GuardrailThresholds(1.0, 2.0, 0.75),
        )
        self.assertEqual(report["warning"]["forecast_coverage_pct"], 0.0)
        self.assertEqual(report["critical"]["forecast_coverage_pct"], 0.0)
        self.assertEqual(report["row_withheld_pct"], 100.0)

    def test_target_support_violation_uses_complete_fallback_schema(self) -> None:
        row = self.row("one")
        row["warning"]["beyond_target_support"] = True
        row["warning"]["physically_invalid"] = True
        report = evaluate_thresholds(
            [row], GuardrailThresholds(1.0, 2.0, 0.75)
        )
        self.assertEqual(report["warning"]["forecast_coverage_pct"], 0.0)
        self.assertEqual(report["row_withheld_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
