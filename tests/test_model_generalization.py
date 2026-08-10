from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.config import load_config
from spindle_monitor.realistic_generator import DURATION_BANDS_HOURS, _duration_plan, generate_realistic_mock
from spindle_monitor.retraining import (
    CalibratedFeatureClassifier,
    LifecycleExamples,
    probability_feature_indices,
    probability_residual_feature_pairs,
    predictive_feature_indices,
    split_lifecycles,
    validate_training_input,
)


class FirstColumnProbability:
    def predict_proba(self, rows):
        probability = np.clip(np.asarray(rows)[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - probability, probability))


class ModelGeneralizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "config" / "thresholds.json")

    def _example(self, event_hour: float | None) -> LifecycleExamples:
        start = datetime(2026, 1, 1)
        timestamps = [start + timedelta(hours=value) for value in (0, 6, 12, 18, 24, 30)]
        event = start + timedelta(hours=event_hour) if event_hour is not None else None
        return LifecycleExamples(
            lifecycle_id="life", timestamps=timestamps,
            feature_rows=[[float(index)] for index in range(len(timestamps))],
            baseline_warning=[None] * len(timestamps), baseline_critical=[None] * len(timestamps),
            first_warning_timestamp=event, first_critical_timestamp=event,
        )

    def test_event_free_lifecycle_contributes_probability_negatives(self) -> None:
        rows, labels, timestamps = self._example(None).probability_rows("warning", 6)
        self.assertEqual(len(rows), len(timestamps))
        self.assertEqual(labels.tolist(), [0] * len(rows))

    def test_horizon_labels_use_elapsed_time_and_exclude_onset_and_post_event(self) -> None:
        example = self._example(24)
        expected = {6: [0, 0, 0, 1], 12: [0, 0, 1, 1], 24: [1, 1, 1, 1]}
        for horizon, labels_expected in expected.items():
            _, labels, timestamps = example.probability_rows("warning", horizon)
            self.assertEqual(labels.tolist(), labels_expected)
            self.assertTrue(all(value < example.first_warning_timestamp for value in timestamps))

    def test_absolute_age_is_audit_only_and_identical_sensor_rows_match(self) -> None:
        names = ["sensor_signal", "elapsed_lifecycle_hours"]
        indices = predictive_feature_indices(names)
        self.assertEqual(indices, (0,))
        model = CalibratedFeatureClassifier(FirstColumnProbability(), indices)
        probabilities = model.predict_proba(np.asarray([[.2, 10.0], [.2, 1000.0]]))[:, 1]
        np.testing.assert_allclose(probabilities, [.2, .2])

    def test_probability_feature_policy_excludes_level_variability_and_age_proxies(self) -> None:
        names = [
            "vibration_mps2__raw", "vibration_mps2__w60m__median",
            "vibration_mps2__w360m__std", "vibration_mps2__w60m__slope_per_hour",
            "vibration_mps2__fast_slow_difference", "elapsed_lifecycle_hours",
            "vibration_mps2__w60m__sample_count",
        ]
        selected = [names[index] for index in probability_feature_indices(names)]
        self.assertEqual(
            selected,
            ["vibration_mps2__w60m__slope_per_hour", "vibration_mps2__fast_slow_difference"],
        )

    def test_relative_state_transform_is_invariant_to_common_level_offset(self) -> None:
        names = [
            "vibration_mps2__kalman_rate_per_hour",
            "vibration_mps2__kalman_level",
            "vibration_mps2__w1440m__median",
        ]
        direct = probability_feature_indices(names)
        residuals = probability_residual_feature_pairs(names)
        model = CalibratedFeatureClassifier(FirstColumnProbability(), direct, residual_pairs=residuals)
        transformed = model.transform(np.asarray([
            [0.2, 5.0, 4.0],
            [0.2, 8.0, 7.0],
        ]))
        np.testing.assert_allclose(transformed[0], transformed[1])

    def test_lifecycle_split_is_disjoint(self) -> None:
        examples = []
        for index in range(12):
            example = self._example(24 if index % 2 else None)
            example.lifecycle_id = f"life_{index:02d}"
            examples.append(example)
        training, validation, test = split_lifecycles(examples, self.config)
        groups = [{value.lifecycle_id for value in group} for group in (training, validation, test)]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_duration_plan_contains_short_through_1000_hour_support(self) -> None:
        durations = _duration_plan(len(DURATION_BANDS_HOURS), np.random.default_rng(42))
        self.assertTrue(any(value < 72 for value in durations))
        self.assertTrue(any(500 <= value < 750 for value in durations))
        self.assertTrue(any(value >= 1000 for value in durations))

    def test_generator_covers_high_load_healthy_and_outcome_independent_regimes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            profile.write_text(json.dumps({
                "start_timestamp": "2026-01-01T00:00:00",
                "sampling_interval_seconds": {"median": 60},
                "sensor_statistics": {},
            }), encoding="utf-8")
            output = root / "generated.csv"
            sidecar = root / "generated_metadata.json"
            short_bands = ((.25, .30), (.30, .35), (.35, .40), (.40, .45))
            with patch("spindle_monitor.realistic_generator.DURATION_BANDS_HOURS", short_bands):
                metadata = generate_realistic_mock(
                    profile, 8, output, sidecar, self.config, seed=17,
                )
            for lifecycle_kind in ("long_healthy", "degradation"):
                cohort = [
                    value for value in metadata["lifecycles"]
                    if value["lifecycle_kind"] == lifecycle_kind
                ]
                regimes = {value["operating_regime"] for value in cohort}
                self.assertEqual(regimes, {"low_load", "medium_load", "high_load", "variable_mixed_load"})
                self.assertEqual(
                    {value["duration_band_index"] for value in cohort},
                    set(range(len(short_bands))),
                )
            rows = __import__("pandas").read_csv(output)
            high_healthy = next(
                value for value in metadata["lifecycles"]
                if value["lifecycle_kind"] == "long_healthy" and value["operating_regime"] == "high_load"
            )
            timestamps = __import__("pandas").to_datetime(rows["timestamp"])
            mask = (
                timestamps.ge(high_healthy["start_timestamp"])
                & timestamps.lt(high_healthy["end_timestamp"])
            )
            selected = rows.loc[mask]
            self.assertGreater(float(selected["vibration_mps2"].median()), 2.2)
            self.assertGreater(float(selected["temperature_c"].median()), 48.0)
            self.assertGreater(float(selected["current_ampere"].median()), 5.2)
            self.assertLessEqual(float(selected["vibration_mps2"].max()), 3.0)
            self.assertLessEqual(float(selected["temperature_c"].max()), 60.0)
            self.assertLessEqual(float(selected["current_ampere"].max()), 6.0)

    def test_external_stress_and_acceptance_role_are_rejected_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "spindle_fp_rate_stress_test_100000.csv"
            external.write_text("x\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "External acceptance"):
                validate_training_input(external)
            candidate = root / "held_out.csv"
            candidate.write_text("x\n1\n", encoding="utf-8")
            candidate.with_name("held_out_metadata.json").write_text(
                json.dumps({"dataset_role": "acceptance"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Held-out acceptance"):
                validate_training_input(candidate)


if __name__ == "__main__":
    unittest.main()
