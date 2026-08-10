from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spindle_monitor.forecast_policy import production_eligibility, validate_probability_target_evidence, mandatory_production_targets
from spindle_monitor.retraining import _select_probability_threshold
from spindle_monitor.ml_forecaster import MLForecaster
from spindle_monitor.config import load_config
from spindle_monitor.model_registry import ModelRegistry
from spindle_monitor.retraining import ConstantProbabilityClassifier


class ForecastPolicyHardeningTests(unittest.TestCase):
    def metadata(self) -> dict:
        targets = {f"probability_{kind}_{hour}h": {"selected_threshold": .5, "eligible": True, "reason": None}
                   for kind in ("warning", "critical") for hour in (6, 12, 24)}
        evidence = {name: {"target_required": True, "target_eligible": True, "target_ineligibility_reasons": [], "selected_threshold": .5,
            "validation_fp_rate": .1, "validation_fn_rate": .1, "validation_fn_ceiling": .1, "validation_fn_ceiling_met": True,
            "test_fp_rate": .1, "test_fn_rate": .1, "test_fn_ceiling": .1, "test_fn_ceiling_met": True,
            "baseline_passed": True, "calibration_passed": True} for name in targets}
        names = ["time_to_warning", "time_to_critical", *targets]
        eta = {name: {"target_required": True, "target_eligible": True, "target_ineligibility_reasons": [], "model_loaded": True,
                      "model_schema_compatible": True, "feature_schema_compatible": True, "physical_validation_passed": True,
                      "baseline_passed": True, "support_min_hours": 0.0, "support_max_hours": 100.0}
               for name in ("time_to_warning", "time_to_critical")}
        return {"model_version": "valid", "maturity_stage": "deployed", "model_stage": "plant_production",
                "production_eligible": True, "data_domain": "plant", "required_model_targets": names,
                "probability_thresholds": {"schema_version": "1.0", "targets": targets}, "probability_targets": evidence,
                "eta_target_evidence": eta}

    def test_complete_artifacts_pass_and_missing_required_fails_closed(self) -> None:
        meta = self.metadata(); loaded = set(meta["required_model_targets"])
        ok, reasons, _, _ = production_eligibility(meta, (6, 12, 24), loaded_targets=loaded)
        self.assertTrue(ok); self.assertFalse(reasons)
        loaded.remove("probability_critical_24h")
        ok, reasons, _, _ = production_eligibility(meta, (6, 12, 24), loaded_targets=loaded)
        self.assertFalse(ok); self.assertIn("missing_required_model:probability_critical_24h", reasons)

    def test_metadata_cannot_remove_mandatory_probability_targets(self) -> None:
        meta = self.metadata(); meta["required_model_targets"] = ["time_to_warning", "time_to_critical"]
        ok, reasons, _, _ = production_eligibility(meta, (6, 12, 24), loaded_targets=set(meta["required_model_targets"]))
        self.assertFalse(ok); self.assertIn("mandatory_target_missing_from_metadata:probability_critical_24h", reasons)

    def test_fn_ceiling_boundary_is_inclusive_and_excess_rejected(self) -> None:
        labels = np.array([1] * 10 + [0] * 10)
        exact = _select_probability_threshold(labels, np.array([.9] * 9 + [.1] + [.1] * 10), .5, .1)
        self.assertTrue(exact["eligible"])
        fail = _select_probability_threshold(labels, np.array([.0] * 20), .5, .1)
        self.assertFalse(fail["eligible"]); self.assertEqual(fail["reason"], "fn_ceiling_not_met")

    def test_research_fp_ceiling_prevents_alarm_everywhere_threshold(self) -> None:
        labels = np.array([1] * 10 + [0] * 10)
        probabilities = np.array([.9] * 9 + [.4] + [.9] * 10)
        result = _select_probability_threshold(labels, probabilities, .5, .1, maximum_fp_rate=.15)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "no_threshold_satisfies_joint_fn_fp_constraints")
        self.assertTrue(result["fn_constraint_feasible"])
        self.assertFalse(result["joint_constraint_feasible"])
        self.assertEqual(result["maximum_false_positive_rate"], .15)

    def test_infeasible_joint_constraints_use_balanced_fallback_not_alarm_everywhere(self) -> None:
        labels = np.array([1] * 10 + [0] * 10)
        # A threshold near .2 has zero FN but 100% FP.  A threshold near .6
        # accepts some FN but sharply reduces FP.  The diagnostic fallback must
        # choose the smaller normalized joint-policy violation, not zero-FN at
        # any cost.
        probabilities = np.array([.95, .9, .85, .8, .75, .7, .65, .6, .55, .2] + [.55] * 10)
        result = _select_probability_threshold(labels, probabilities, .5, .1, maximum_fp_rate=.15)
        self.assertFalse(result["eligible"])
        self.assertGreater(result["selected_threshold"], .2)
        self.assertLess(result["best_achievable_metrics"]["false_positive_rate"], 1.0)

    def test_no_loaded_models_retains_complete_audit(self) -> None:
        meta = self.metadata()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "production").mkdir()
            import json
            (root / "production" / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            config = load_config(ROOT / "config" / "thresholds.json")
            forecaster = MLForecaster(config, ["x"], root)
            result = forecaster.predict({"x": 1.0})
            self.assertFalse(result.production_policy_passed)
            self.assertEqual(set(mandatory_production_targets((6,12,24))), set(result.missing_required_model_targets))
            self.assertIn("missing_required_model:probability_warning_12h", result.policy_withholding_reasons)

    def test_mandatory_target_set_is_code_defined(self) -> None:
        targets=mandatory_production_targets((6,12,24)); self.assertEqual(len(targets),2); self.assertEqual(set(targets), {"probability_warning_12h","probability_critical_24h"})

    def test_numeric_fn_contradiction_and_boundary(self) -> None:
        record={"target_required":True,"selected_threshold":.5,"validation_fp_rate":.2,"validation_fn_rate":.1,"validation_fn_ceiling":.1,"validation_fn_ceiling_met":True,
            "test_fp_rate":.2,"test_fn_rate":.1,"test_fn_ceiling":.1,"test_fn_ceiling_met":True,"baseline_passed":True,"calibration_passed":True,
            "target_eligible":True,"target_ineligibility_reasons":[]}
        self.assertEqual(validate_probability_target_evidence("p",record,.5),())
        record["test_fn_rate"]=.100001
        reasons=validate_probability_target_evidence("p",record,.5); self.assertIn("test_fn_numeric_contradiction:p",reasons); self.assertIn("test_fn_ceiling_not_met:p",reasons)

    def test_runtime_ceiling_rejects_looser_model_policy(self) -> None:
        record={"target_required":True,"target_eligible":True,"target_ineligibility_reasons":[],"selected_threshold":.5,
            "validation_fp_rate":.1,"validation_fn_rate":.15,"validation_fn_ceiling":.2,"validation_fn_ceiling_met":True,
            "test_fp_rate":.1,"test_fn_rate":.15,"test_fn_ceiling":.2,"test_fn_ceiling_met":True,"baseline_passed":True,"calibration_passed":True}
        reasons=validate_probability_target_evidence("p",record,.5,.1)
        self.assertIn("model_fn_ceiling_exceeds_runtime_policy:p",reasons); self.assertIn("validation_fn_ceiling_not_met:p",reasons); self.assertIn("test_fn_ceiling_not_met:p",reasons)

    def test_mandatory_target_required_flag_is_strict_boolean(self) -> None:
        record={"target_required":False,"target_eligible":True,"target_ineligibility_reasons":[],"selected_threshold":.5,
            "validation_fp_rate":.1,"validation_fn_rate":.05,"validation_fn_ceiling":.1,"validation_fn_ceiling_met":True,
            "test_fp_rate":.1,"test_fn_rate":.05,"test_fn_ceiling":.1,"test_fn_ceiling_met":True,"baseline_passed":True,"calibration_passed":True}
        self.assertIn("mandatory_target_not_marked_required:p",validate_probability_target_evidence("p",record,.5,.1))
        record["target_required"]="true"; self.assertIn("required_target_flag_invalid:p",validate_probability_target_evidence("p",record,.5,.1))

    def test_threshold_boundaries_zero_and_one_are_invalid(self) -> None:
        base={"target_required":True,"target_eligible":True,"target_ineligibility_reasons":[],"validation_fp_rate":.1,"validation_fn_rate":.05,"validation_fn_ceiling":.1,"validation_fn_ceiling_met":True,
            "test_fp_rate":.1,"test_fn_rate":.05,"test_fn_ceiling":.1,"test_fn_ceiling_met":True,"baseline_passed":True,"calibration_passed":True}
        for value in (0.0,1.0):
            record={**base,"selected_threshold":value}; self.assertIn("threshold_at_invalid_boundary:p",validate_probability_target_evidence("p",record,value,.1))

    def test_registry_rejects_looser_plant_fn_policy(self) -> None:
        meta=self.metadata(); meta.update({"data_domain":"plant","synthetic_data_only":False})
        for record in meta["probability_targets"].values():
            record.update({"validation_fn_rate":.15,"validation_fn_ceiling":.2,"validation_fn_ceiling_met":True,"test_fn_rate":.15,"test_fn_ceiling":.2,"test_fn_ceiling_met":True})
        with tempfile.TemporaryDirectory() as directory:
            registry=ModelRegistry(directory); registry.save_candidate({name:ConstantProbabilityClassifier(.5) for name in meta["required_model_targets"]},meta)
            with self.assertRaisesRegex(ValueError,"model_fn_ceiling_exceeds_runtime_policy"):
                registry.promote_targets(meta["required_model_targets"],maximum_fn_rate=.1)


if __name__ == "__main__": unittest.main()
