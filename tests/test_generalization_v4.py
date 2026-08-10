from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_generalization_compact import (
    _build_compact_audit,
    fast_per_target_metrics,
    validate_compact_audit,
)
from evaluate_monitoring import per_target_metrics
from spindle_monitor.retraining import (
    CalibratedFeatureClassifier,
    ConstantProbabilityClassifier,
    probability_hard_example_multipliers,
    probability_sample_weights,
    probability_feature_indices,
    probability_residual_feature_pairs,
    probability_standardized_residual_specs,
    probability_trend_snr_specs,
    split_probability_validation_lifecycles,
)


TARGETS = [
    f"probability_{kind}_{hours}h"
    for kind in ("warning", "critical")
    for hours in (6, 12, 24)
]


class GeneralizationV4Tests(unittest.TestCase):
    def _metric_fixture(self) -> tuple[pd.DataFrame, pd.Series]:
        count = 3600
        timestamps = pd.date_range("2026-01-01", periods=count, freq="min")
        actual = pd.Series("NORMAL", index=np.arange(count), dtype=str)
        actual.iloc[1800:2100] = "WARNING"
        actual.iloc[3000:3300] = "CRITICAL"
        thresholds = {target: 0.5 for target in TARGETS}
        eligibility = {target: True for target in TARGETS}
        probabilities: dict[str, np.ndarray] = {}
        row = np.arange(count)
        for target in TARGETS:
            kind = "warning" if "warning" in target else "critical"
            hours = int(target.rsplit("_", 1)[1][:-1])
            onset = 1800 if kind == "warning" else 3000
            horizon_rows = hours * 60
            start = onset - horizon_rows
            p = np.full(count, 0.08, dtype=float)
            within = (row >= start) & (row < onset)
            p[within] = 0.75
            # A deterministic false-alert episode during otherwise NORMAL time.
            p[420:451] = 0.70
            # Reconciliation changes a small region but does not alter schema.
            probabilities[target] = p
        crossings: list[str] = []
        frame = pd.DataFrame({
            "timestamp": timestamps.astype(str),
            "lifecycle_id": "life_001",
            "status": actual,
            "raw_status": actual,
            "raw_safety_status": actual,
            "stabilized_status": actual,
            "event_status": actual,
            "selected_probability_thresholds": json.dumps(thresholds, sort_keys=True),
            "probability_target_eligibility": json.dumps(eligibility, sort_keys=True),
            "loaded_model_targets": json.dumps(TARGETS),
            "recommendation_actionable": False,
        })
        crossing_rows: list[str] = []
        for position in range(count):
            values = [target for target in TARGETS if probabilities[target][position] >= 0.5]
            crossing_rows.append(json.dumps(values))
        frame["probability_threshold_crossings"] = crossing_rows
        for target, p in probabilities.items():
            frame[target] = p
            frame[f"raw_{target}"] = p
            reconciled = p.copy()
            reconciled[1000:1011] = np.clip(reconciled[1000:1011] + 0.1, 0, 1)
            frame[f"reconciled_{target}"] = reconciled
        return frame, actual.str.upper()

    def _assert_equivalent(self, left, right, path="root") -> None:
        if isinstance(left, dict):
            self.assertIsInstance(right, dict, path)
            self.assertEqual(set(left), set(right), path)
            for key in left:
                self._assert_equivalent(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, list):
            self.assertIsInstance(right, list, path)
            self.assertEqual(len(left), len(right), path)
            for index, (a, b) in enumerate(zip(left, right)):
                self._assert_equivalent(a, b, f"{path}[{index}]")
            return
        if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
            if left is None or right is None:
                self.assertIs(left, right, path)
            elif math.isnan(float(left)) and math.isnan(float(right)):
                return
            else:
                self.assertAlmostEqual(float(left), float(right), places=12, msg=path)
            return
        self.assertEqual(left, right, path)

    def test_fast_compact_metrics_match_legacy_semantics(self) -> None:
        frame, actual = self._metric_fixture()
        trusted = pd.Series(False, index=frame.index)
        expected = per_target_metrics(frame, actual, trusted, 10.0)
        actual_fast = fast_per_target_metrics(frame, actual, trusted, 10.0)
        self._assert_equivalent(expected, actual_fast)

    def test_compact_audit_validates_distinct_payloads_and_types(self) -> None:
        frame, _ = self._metric_fixture()
        audit = _build_compact_audit(frame)
        result = validate_compact_audit(frame, audit)
        self.assertTrue(result["validated"])
        self.assertEqual(result["selected_threshold_payload_count"], 1)
        broken = frame.iloc[:3].copy()
        broken.loc[1, "probability_target_eligibility"] = json.dumps({TARGETS[0]: "yes"})
        with self.assertRaisesRegex(ValueError, "invalid probability target eligibility"):
            validate_compact_audit(broken)


    def test_validation_calibration_and_threshold_lifecycles_are_disjoint(self) -> None:
        ids = [
            "positive_a", "positive_a", "positive_b", "positive_b",
            "negative_a", "negative_a", "negative_b", "negative_b",
        ]
        labels = np.asarray([0, 1, 0, 1, 0, 0, 0, 0], dtype=int)
        calibration, selection, calibration_ids = split_probability_validation_lifecycles(
            labels, ids, minimum_lifecycles_for_calibration=4
        )
        calibration_rows = {value for value, keep in zip(ids, calibration) if keep}
        selection_rows = {value for value, keep in zip(ids, selection) if keep}
        self.assertEqual(calibration_rows, calibration_ids)
        self.assertFalse(calibration_rows & selection_rows)
        self.assertEqual(set(np.unique(labels[calibration])), {0, 1})
        self.assertEqual(set(np.unique(labels[selection])), {0, 1})

    def test_small_validation_cohort_is_not_split_again_for_calibration(self) -> None:
        ids = ["positive_a", "positive_a", "positive_b", "negative_a", "negative_b"]
        labels = np.asarray([1, 0, 1, 0, 0], dtype=int)
        calibration, selection, calibration_ids = split_probability_validation_lifecycles(labels, ids)
        self.assertFalse(np.any(calibration))
        self.assertTrue(np.all(selection))
        self.assertEqual(calibration_ids, set())

    def test_probability_feature_policy_uses_directional_change_not_absolute_level(self) -> None:
        names = [
            "vibration_mps2__raw",
            "vibration_mps2__rolling_median",
            "vibration_mps2__normalized_severity",
            "vibration_mps2__w60m__std",
            "vibration_mps2__kalman_rate_per_hour",
            "vibration_mps2__fast_slow_difference",
            "vibration_mps2__w60m__slope_per_hour",
            "vibration_mps2__w60m__change",
            "vibration_mps2__w60m__slope_acceleration_per_hour2",
            "elapsed_lifecycle_hours",
        ]
        selected = [names[index] for index in probability_feature_indices(names)]
        self.assertEqual(selected, names[4:9])
        self.assertNotIn("vibration_mps2__raw", selected)
        self.assertNotIn("vibration_mps2__normalized_severity", selected)
        self.assertNotIn("elapsed_lifecycle_hours", selected)

    def test_relative_state_features_are_derived_from_local_baselines(self) -> None:
        names = [
            "vibration_mps2__kalman_level",
            "vibration_mps2__slow_ewma",
            "vibration_mps2__w15m__mean",
            "vibration_mps2__w60m__mean",
            "vibration_mps2__w180m__mean",
            "vibration_mps2__w360m__mean",
            "vibration_mps2__w360m__median",
            "vibration_mps2__w720m__median",
            "vibration_mps2__w1440m__median",
        ]
        pairs = probability_residual_feature_pairs(names)
        self.assertGreaterEqual(len(pairs), 6)
        self.assertIn((0, 8), pairs)
        self.assertIn((3, 6), pairs)
        self.assertTrue(all(left != right for left, right in pairs))

    def test_early_positive_weighting_redistributes_positive_mass(self) -> None:
        labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=int)
        ids = ["a", "a", "a", "b", "b", "b"]
        leads = np.asarray([12.0, 6.0, 1.0, np.nan, np.nan, np.nan])
        base = probability_sample_weights(labels, ids)
        weighted = probability_sample_weights(labels, ids, leads, 12.0, 2.0)
        self.assertGreater(weighted[0], weighted[2])
        self.assertAlmostEqual(float(np.sum(base[labels == 1])), float(np.sum(weighted[labels == 1])))
        self.assertAlmostEqual(float(np.sum(base[labels == 0])), float(np.sum(weighted[labels == 0])))

    def test_partial_class_balance_is_less_alarm_biased_than_full_balance(self) -> None:
        labels = np.asarray([1, 0, 0, 0, 0], dtype=int)
        ids = ["a", "a", "a", "a", "a"]
        full = probability_sample_weights(labels, ids, positive_class_balance_strength=1.0)
        partial = probability_sample_weights(labels, ids, positive_class_balance_strength=0.35)
        self.assertGreater(full[0], partial[0])

    def test_boundary_negatives_receive_extra_weight(self) -> None:
        labels = np.asarray([1, 0, 0, 0], dtype=int)
        ids = ["a", "a", "a", "a"]
        leads = np.asarray([6.0, 13.0, 30.0, np.nan])
        weighted = probability_sample_weights(
            labels, ids, leads, 12.0, 1.0, 0.0, 2.0
        )
        # 13h is just outside the 12h positive horizon and should be emphasized;
        # 30h and healthy/no-event negatives should not.
        self.assertGreater(weighted[1], weighted[2])
        self.assertGreater(weighted[1], weighted[3])

    def test_normalized_residual_and_trend_snr_features_are_finite_and_clipped(self) -> None:
        names = [
            "vibration_mps2__kalman_level",
            "vibration_mps2__slow_ewma",
            "vibration_mps2__w15m__mean",
            "vibration_mps2__w60m__mean",
            "vibration_mps2__w180m__mean",
            "vibration_mps2__w360m__mean",
            "vibration_mps2__w360m__median",
            "vibration_mps2__w360m__std",
            "vibration_mps2__w720m__median",
            "vibration_mps2__w720m__std",
            "vibration_mps2__w1440m__median",
            "vibration_mps2__w1440m__std",
            "vibration_mps2__w60m__slope_per_hour",
            "vibration_mps2__w180m__slope_per_hour",
            "vibration_mps2__w360m__slope_per_hour",
            "vibration_mps2__w720m__slope_per_hour",
            "vibration_mps2__w1440m__slope_per_hour",
        ]
        training = np.asarray([
            [8.0, 7.8, 7.9, 7.8, 7.7, 7.6, 7.5, .20, 7.4, .30, 7.2, .40, .08, .06, .04, .03, .02],
            [8.1, 7.9, 8.0, 7.9, 7.8, 7.7, 7.6, .25, 7.5, .35, 7.3, .45, .09, .07, .05, .04, .03],
        ], dtype=float)
        standardized = probability_standardized_residual_specs(names, training)
        trend = probability_trend_snr_specs(names, training)
        self.assertGreaterEqual(len(standardized), 6)
        self.assertEqual(len(trend), 3)
        wrapper = CalibratedFeatureClassifier(
            ConstantProbabilityClassifier(.2),
            (12,),
            standardized_residual_specs=standardized,
            trend_snr_specs=trend,
        )
        transformed = wrapper.transform(training)
        derived = transformed[:, 1:]
        self.assertTrue(np.all(np.isfinite(derived)))
        self.assertLessEqual(float(np.max(np.abs(derived))), 8.0)

    def test_training_oof_hard_example_mining_weights_both_error_sides(self) -> None:
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        probabilities = np.asarray([.05, .10, .80, .90, .05, .15, .70, .80], dtype=float)
        leads = np.asarray([np.nan, np.nan, np.nan, np.nan, 12.0, 10.0, 8.0, 7.0])
        multipliers, report = probability_hard_example_multipliers(
            labels,
            probabilities,
            leads,
            12.0,
            .5,
            2.0,
            .5,
            1.5,
        )
        self.assertEqual(report["hard_negative_count"], 2)
        self.assertEqual(report["hard_early_positive_count"], 2)
        self.assertGreater(multipliers[2], multipliers[0])
        self.assertGreater(multipliers[4], multipliers[7])


if __name__ == "__main__":
    unittest.main()
