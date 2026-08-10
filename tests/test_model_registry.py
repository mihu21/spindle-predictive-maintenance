from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.config import load_config
from spindle_monitor.model_registry import ModelRegistry
from spindle_monitor.retraining import ConstantProbabilityClassifier, evaluate_candidate


class ModelRegistryTests(unittest.TestCase):
    def test_candidate_is_not_promoted_below_lifecycle_minimum(self) -> None:
        config = load_config(ROOT / "config" / "thresholds.json")
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            registry.save_candidate(
                {},
                {
                    "model_version": "test",
                    "usable_lifecycles": 3,
                    "targets": {
                        "time_to_critical": {
                            "validation_metrics": {"mae_hours": 1.0}
                        }
                    },
                    "baseline_metrics": {
                        "time_to_critical": {"mae_hours": 2.0}
                    },
                },
            )
            report = evaluate_candidate(config, directory, promote=True)
            self.assertFalse(report["promoted"])
            self.assertFalse(report["lifecycle_count_eligible"])
            self.assertTrue((Path(directory) / "registry_audit.jsonl").exists())

    def test_candidate_is_not_promoted_when_worse_than_baseline(self) -> None:
        config = load_config(ROOT / "config" / "thresholds.json")
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(directory)
            registry.save_candidate(
                {},
                {
                    "model_version": "worse",
                    "usable_lifecycles": 10,
                    "targets": {
                        "time_to_critical": {
                            "validation_metrics": {"mae_hours": 3.0}
                        }
                    },
                    "baseline_metrics": {
                        "time_to_critical": {"mae_hours": 2.0}
                    },
                },
            )
            report = evaluate_candidate(config, directory, promote=True)
            self.assertFalse(report["promoted"])
            self.assertFalse(report["targets"]["time_to_critical"]["better_than_baseline"])

    def test_constant_probability_model_preserves_one_class_horizon_output(self) -> None:
        model = ConstantProbabilityClassifier(1.0)
        probabilities = model.predict_proba(np.zeros((3, 2)))
        self.assertEqual(probabilities.shape, (3, 2))
        self.assertTrue(np.allclose(probabilities[:, 1], 1.0))


if __name__ == "__main__":
    unittest.main()
