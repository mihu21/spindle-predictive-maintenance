from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_monitoring import _future_event_labels, per_target_metrics
from spindle_monitor.forecast_policy import (
    THRESHOLD_SCHEMA_VERSION,
    normalize_metadata_contract,
    production_eligibility,
    runtime_probability_policy,
)
from spindle_monitor.ml_forecaster import reconcile_horizon_probabilities
from spindle_monitor.policy_contract import (
    CONTRACT,
    FORECAST_POLICY_CONTRACT_VERSION,
    MODEL_METADATA_SCHEMA_VERSION,
)


PROBABILITY_TARGETS = list(CONTRACT.probability_targets)


def canonical_metadata() -> dict:
    required = set(CONTRACT.required_targets)
    evidence = {}
    thresholds = {}
    for target in PROBABILITY_TARGETS:
        evidence[target] = {
            "target_required": target in required,
            "target_eligible": True,
            "target_ineligibility_reasons": [],
            "selected_threshold": 0.4,
            "validation_fp_rate": 0.1,
            "validation_fn_rate": 0.05,
            "validation_fn_ceiling": 0.1,
            "validation_fn_ceiling_met": True,
            "test_fp_rate": 0.1,
            "test_fn_rate": 0.05,
            "test_fn_ceiling": 0.1,
            "test_fn_ceiling_met": True,
            "baseline_passed": True,
            "calibration_passed": True,
        }
        thresholds[target] = {"selected_threshold": 0.4, "eligible": True, "reason": None}
    return {
        "model_metadata_schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "forecast_policy_contract_version": FORECAST_POLICY_CONTRACT_VERSION,
        "model_version": "v3-test",
        "maturity_stage": "deployed",
        "model_stage": "plant_production",
        "production_eligible": True,
        "data_domain": "plant",
        "required_model_targets": list(CONTRACT.required_targets),
        "probability_thresholds": {"schema_version": THRESHOLD_SCHEMA_VERSION, "targets": thresholds},
        "probability_targets": evidence,
    }


class ForecastContractV3Tests(unittest.TestCase):
    def test_optional_failure_is_isolated_from_required_targets(self) -> None:
        metadata = canonical_metadata()
        optional = "probability_critical_6h"
        metadata["probability_targets"][optional].update({
            "target_eligible": False,
            "target_ineligibility_reasons": ["test_fn_ceiling_not_met"],
        })
        metadata["probability_thresholds"]["targets"][optional].update({
            "eligible": False,
            "reason": "test_fn_ceiling_not_met",
        })
        passed, reasons, thresholds, eligible = production_eligibility(
            metadata, (6, 12, 24), loaded_targets=set(PROBABILITY_TARGETS)
        )
        self.assertTrue(passed, reasons)
        self.assertFalse(eligible[optional])
        self.assertEqual(thresholds[optional], 0.4)
        self.assertTrue(eligible["probability_warning_12h"])

    def test_missing_required_target_still_blocks(self) -> None:
        metadata = canonical_metadata()
        loaded = set(PROBABILITY_TARGETS) - {"probability_critical_24h"}
        passed, reasons, _, eligible = production_eligibility(metadata, (6, 12, 24), loaded_targets=loaded)
        self.assertFalse(passed)
        self.assertFalse(eligible["probability_critical_24h"])
        self.assertIn("missing_required_model:probability_critical_24h", reasons)

    def test_legacy_migration_is_deterministic_and_does_not_invent_fn_ceiling(self) -> None:
        legacy = canonical_metadata()
        legacy.pop("model_metadata_schema_version")
        legacy.pop("forecast_policy_contract_version")
        legacy.pop("probability_thresholds")
        for record in legacy["probability_targets"].values():
            threshold = record.pop("selected_threshold")
            record["validation_metrics"] = {"classification_threshold": threshold, "false_positive_rate": 0.1, "false_negative_rate": 0.05, "sample_count": 20, "observed_event_rate": 0.5, "calibration_error": 0.05}
            record["test_metrics"] = {"classification_threshold": threshold, "false_positive_rate": 0.1, "false_negative_rate": 0.05, "sample_count": 20, "observed_event_rate": 0.5, "calibration_error": 0.05}
            for key in ("validation_fn_ceiling", "test_fn_ceiling", "validation_fn_ceiling_met", "test_fn_ceiling_met"):
                record.pop(key)
        first = normalize_metadata_contract(legacy, (6, 12, 24))
        second = normalize_metadata_contract(legacy, (6, 12, 24))
        self.assertEqual(first, second)
        self.assertEqual(first["probability_thresholds"]["targets"]["probability_warning_12h"]["selected_threshold"], 0.4)
        _, eligible, _ = runtime_probability_policy(first, (6, 12, 24))
        self.assertFalse(eligible["probability_warning_12h"])
        self.assertIsNone(first["probability_targets"]["probability_warning_12h"].get("validation_fn_ceiling"))

    def test_incompatible_metadata_schema_is_rejected(self) -> None:
        metadata = canonical_metadata(); metadata["model_metadata_schema_version"] = "999"
        normalized = normalize_metadata_contract(metadata, (6, 12, 24))
        self.assertIn("unsupported_model_metadata_schema_version", normalized["metadata_invalid"])

    def test_thresholds_and_evidence_round_trip_as_numeric_json(self) -> None:
        metadata = canonical_metadata()
        restored = json.loads(json.dumps(metadata))
        value = restored["probability_thresholds"]["targets"]["probability_warning_12h"]["selected_threshold"]
        self.assertIsInstance(value, float)
        self.assertEqual(restored["probability_targets"]["probability_warning_12h"]["validation_fn_rate"], 0.05)

    def test_cross_horizon_reconciliation_preserves_raw_input(self) -> None:
        raw = {"warning": {6: 0.7, 12: 0.2, 24: 0.8}, "critical": {6: 0.1, 12: 0.3, 24: 0.2}}
        reconciled, changed = reconcile_horizon_probabilities(raw, (6, 12, 24))
        self.assertEqual(raw["warning"][12], 0.2)
        self.assertEqual([reconciled["warning"][h] for h in (6, 12, 24)], [0.7, 0.7, 0.8])
        self.assertEqual([reconciled["critical"][h] for h in (6, 12, 24)], [0.1, 0.3, 0.3])
        self.assertEqual(set(changed), {"probability_warning_12h", "probability_critical_24h"})

    def test_elapsed_time_labels_do_not_assume_row_count(self) -> None:
        frame = pd.DataFrame({
            "timestamp": ["2026-01-01T00:00:00", "2026-01-01T05:00:00", "2026-01-01T07:00:00"],
            "lifecycle_id": ["L1"] * 3,
        })
        actual = pd.Series(["NORMAL", "NORMAL", "WARNING"])
        labels, _ = _future_event_labels(frame, actual, "warning", 6, 360)
        self.assertEqual(labels.tolist(), [False, True, False])

    def test_withheld_predictions_are_operational_misses_not_dropped(self) -> None:
        rows = []
        for hour, status in ((0, "NORMAL"), (1, "NORMAL"), (2, "WARNING")):
            row = {"timestamp": f"2026-01-01T0{hour}:00:00", "lifecycle_id": "L1", "status": status,
                   "selected_probability_thresholds": json.dumps({target: 0.5 for target in PROBABILITY_TARGETS}),
                   "probability_target_eligibility": json.dumps({target: True for target in PROBABILITY_TARGETS}),
                   "probability_threshold_crossings": "[]", "loaded_model_targets": json.dumps(PROBABILITY_TARGETS)}
            row.update({target: 0.9 for target in PROBABILITY_TARGETS})
            rows.append(row)
        frame = pd.DataFrame(rows); actual = frame.status; trusted = pd.Series(False, index=frame.index)
        report = per_target_metrics(frame, actual, trusted, 61)["probability_warning_6h"]
        self.assertEqual(report["model_only_reconciled"]["fn"], 0)
        self.assertEqual(report["strict_operational"]["fn"], 2)


if __name__ == "__main__":
    unittest.main()
