from __future__ import annotations
import json, sys, unittest
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from evaluate_monitoring import trusted_evidence, horizon_metrics, false_alert_metrics, unsafe_episodes, evaluate, target_horizon_coverage, timestamp_validation, per_target_metrics
from evaluate_generalization_compact import load_compact_frames, validate_compact_audit
from spindle_monitor.forecast_policy import ordered_reasons, primary_reason
from spindle_monitor.policy_contract import CONTRACT, FORECAST_POLICY_CONTRACT_VERSION, FORECAST_POLICY_VERSION

PROBS=[f"probability_{kind}_{h}h" for kind in ("warning","critical") for h in (6,12,24)]
REQUIRED=list(CONTRACT.required_targets)

def audit_row(timestamp="2026-01-02T00:00:00", lifecycle="L1", status="NORMAL", actionable=True):
    evidence={target:{"target_required":target in REQUIRED,"target_eligible":True,"target_ineligibility_reasons":[],"selected_threshold":.5,
        "validation_fp_rate":.1,"validation_fn_rate":.1,"validation_fn_ceiling":.1,"validation_fn_ceiling_met":True,
        "test_fp_rate":.1,"test_fn_rate":.1,"test_fn_ceiling":.1,"test_fn_ceiling_met":True,"baseline_passed":True,"calibration_passed":True} for target in PROBS}
    eta={target:{"target_required":target in REQUIRED,"target_eligible":True,"target_ineligibility_reasons":[],"model_loaded":True,"model_schema_compatible":True,"feature_schema_compatible":True,"support_min_hours":0.0,"support_max_hours":100.0,"physical_validation_passed":True,"baseline_passed":True} for target in ("time_to_warning","time_to_critical")}
    return {"timestamp":timestamp,"lifecycle_id":lifecycle,"status":status,"source_label":status,
        "model_version":"v1","model_maturity_stage":"deployed","model_stage":"plant_production","model_data_domain":"plant",
        "production_eligible":True,"production_policy_passed":True,"production_policy_reasons_json":"[]",
        "model_revoked":False,"model_superseded":False,"model_corrupted":False,"model_metadata_valid":True,"model_artifacts_valid":True,
        "model_schema_compatible":True,"model_feature_schema_compatible":True,"model_load_failures_json":"{}",
        "required_model_targets":json.dumps(REQUIRED),"loaded_model_targets":json.dumps(["time_to_warning","time_to_critical",*PROBS]),"missing_required_model_targets":"[]",
        "configured_mandatory_targets":json.dumps(REQUIRED),"metadata_required_targets":json.dumps(REQUIRED),"effective_required_targets":json.dumps(REQUIRED),"mandatory_targets_missing_from_metadata":"[]",
        "runtime_probability_fn_ceiling":.1,"model_recorded_probability_fn_ceiling":json.dumps({x:.1 for x in PROBS}),"effective_probability_fn_ceiling":json.dumps({x:.1 for x in PROBS}),
        "selected_probability_thresholds":json.dumps({x:.5 for x in PROBS}),"probability_target_eligibility":json.dumps({x:True for x in PROBS}),
        "probability_target_ineligibility_reasons":json.dumps({x:[] for x in PROBS}),"probability_target_evidence":json.dumps(evidence),
        "physical_validation_passed":True,"physical_validation_reasons_json":"[]","feature_ready":True,"feature_readiness_reasons_json":"[]",
        "ood_detected":False,"target_support_valid":True,"disagreement_detected":False,"forecast_withheld":False,"withholding_reasons_json":"[]",
        "recommendation_source":"ml" if actionable else "unavailable","recommendation_actionable":actionable,
        "recommendation_trigger_targets":"[]",
        "probability_threshold_crossings":"[]","recommendation_trigger_probability_targets":"[]","recommendation_trigger_eta_targets":"[]","recommendation_trigger_rules":"[]",
        "eta_target_evidence":json.dumps(eta),"urgency_policy_version":"1.0","urgency_policy_parameters":json.dumps({"critical_probability_6h_target":"probability_critical_6h","critical_probability_24h_target":"probability_critical_24h","warning_probability_12h_target":"probability_warning_12h","critical_eta_soon_hours":6.0,"critical_eta_plan_hours":24.0,"warning_eta_plan_hours":12.0}),"forecast_policy_contract_version":FORECAST_POLICY_CONTRACT_VERSION,
        "runtime_validation_fn_ceiling":.1,"runtime_test_fn_ceiling":.1,"model_recorded_validation_fn_ceilings":json.dumps({x:.1 for x in PROBS}),"model_recorded_test_fn_ceilings":json.dumps({x:.1 for x in PROBS}),"effective_validation_fn_ceilings":json.dumps({x:.1 for x in PROBS}),"effective_test_fn_ceilings":json.dumps({x:.1 for x in PROBS}),
        "forecast_policy_version":FORECAST_POLICY_VERSION,"probability_threshold_schema_version":"1.0"}

class EvaluatorHardeningTests(unittest.TestCase):
    def test_fully_consistent_row_is_trusted(self):
        trusted,available,failures=trusted_evidence(pd.DataFrame([audit_row()])); self.assertTrue(available); self.assertTrue(trusted.iloc[0]); self.assertEqual(failures["malformed_audit_row_count"],0)

    def test_revoked_superseded_and_corrupted_each_fail_closed(self):
        for field in ("model_revoked","model_superseded","model_corrupted"):
            row=audit_row(); row[field]=True; trusted,_,failures=trusted_evidence(pd.DataFrame([row])); self.assertFalse(trusted.iloc[0]); self.assertIn(field,failures)

    def test_policy_success_with_reasons_is_rejected(self):
        row=audit_row(); row["production_policy_reasons_json"]='["production_ineligible"]'
        trusted,_,failures=trusted_evidence(pd.DataFrame([row])); self.assertFalse(trusted.iloc[0]); self.assertIn("production_policy_contradiction",failures)

    def test_missing_loaded_target_and_load_failure_are_rejected(self):
        row=audit_row(); row["loaded_model_targets"]=json.dumps(REQUIRED[:-1]); row["model_load_failures_json"]=json.dumps({REQUIRED[-1]:"model_load_failed"})
        trusted,_,failures=trusted_evidence(pd.DataFrame([row])); self.assertFalse(trusted.iloc[0]); self.assertIn("required_model_artifact_evidence_failed",failures)

    def test_target_fn_gate_contradiction_is_rejected(self):
        target=REQUIRED[0]; row=audit_row(); evidence=json.loads(row["probability_target_evidence"]); evidence[target]["test_fn_ceiling_met"]=False; row["probability_target_evidence"]=json.dumps(evidence)
        trusted,_,failures=trusted_evidence(pd.DataFrame([row])); self.assertFalse(trusted.iloc[0]); self.assertIn(f"test_fn_ceiling_failed:{target}",failures)

    def test_reason_priority_is_safety_order_not_alphabetical(self):
        reasons=ordered_reasons(["sampling_coverage_too_low","missing_required_model:x","model_corrupted","zzz"])
        self.assertEqual(reasons[0],"model_corrupted"); self.assertEqual(primary_reason(reasons),"model_corrupted"); self.assertIn("zzz",reasons)

    def _horizon_frame(self,prediction_hour:int,onset_hour:int=24,event="WARNING"):
        rows=[]
        for hour in range(onset_hour+2):
            label=event if hour>=onset_hour else "NORMAL"; timestamp=(datetime(2026,1,1)+timedelta(hours=hour)).isoformat()
            row=audit_row(timestamp,status=label,actionable=(hour==prediction_hour))
            if hour!=prediction_hour: row["recommendation_source"]="manufacturer" if hour>=onset_hour else "unavailable"
            rows.append(row)
        frame=pd.DataFrame(rows); actual=frame.source_label; trusted,_,_=trusted_evidence(frame); return frame,actual,trusted

    def test_prediction_four_hours_before_onset_succeeds_all_horizons(self):
        frame,actual,trusted=self._horizon_frame(20); result=horizon_metrics(frame,actual,trusted,"WARNING",maximum_gap_minutes=61)
        self.assertEqual([result[x]["predicted_unsafe_episodes"] for x in ("6h","12h","24h")],[1,1,1])

    def test_prediction_eight_hours_before_fails_six_but_passes_twelve_and_twentyfour(self):
        frame,actual,trusted=self._horizon_frame(16); result=horizon_metrics(frame,actual,trusted,"WARNING",maximum_gap_minutes=61)
        self.assertEqual([result[x]["predicted_unsafe_episodes"] for x in ("6h","12h","24h")],[0,1,1])

    def test_prediction_at_onset_does_not_count(self):
        frame,actual,trusted=self._horizon_frame(24); result=horizon_metrics(frame,actual,trusted,"WARNING",maximum_gap_minutes=61)
        self.assertEqual(result["24h"]["predicted_unsafe_episodes"],0)

    def test_insufficient_history_episode_is_not_a_miss(self):
        frame,actual,trusted=self._horizon_frame(2,onset_hour=4); result=horizon_metrics(frame,actual,trusted,"WARNING",maximum_gap_minutes=61)
        self.assertEqual(result["6h"]["eligible_unsafe_episodes"],0); self.assertEqual(result["6h"]["insufficient_history_episode_count"],1)

    def test_false_alert_episodes_split_at_lifecycle(self):
        rows=[audit_row("2026-01-01T00:00:00","L1"),audit_row("2026-01-01T00:01:00","L1"),audit_row("2026-01-01T00:02:00","L2")]
        frame=pd.DataFrame(rows); trusted,_,_=trusted_evidence(frame); result=false_alert_metrics(frame,frame.source_label,trusted)
        self.assertEqual(result["false_alert_rows"],3); self.assertEqual(result["false_alert_episodes"],2)

    def test_critical_to_warning_is_not_warning_onset(self):
        rows=[audit_row("2026-01-01T00:00:00",status="CRITICAL",actionable=False),audit_row("2026-01-01T00:01:00",status="WARNING",actionable=False)]
        frame=pd.DataFrame(rows); self.assertEqual(unsafe_episodes(frame,frame.source_label,"WARNING"),[])

    def test_warning_to_critical_is_critical_escalation_onset(self):
        rows=[audit_row("2026-01-01T00:00:00",status="WARNING",actionable=False),audit_row("2026-01-01T00:01:00",status="CRITICAL",actionable=False)]
        frame=pd.DataFrame(rows); self.assertEqual(len(unsafe_episodes(frame,frame.source_label,"CRITICAL")),1)

    def test_gap_ended_warning_episode_is_censored(self):
        rows=[audit_row("2026-01-01T00:00:00",status="NORMAL",actionable=False),audit_row("2026-01-01T00:01:00",status="WARNING",actionable=False),
              audit_row("2026-01-01T00:02:00",status="WARNING",actionable=False),audit_row("2026-01-01T01:00:00",status="NORMAL",actionable=False)]
        episode=unsafe_episodes(pd.DataFrame(rows),pd.Series(["NORMAL","WARNING","WARNING","NORMAL"]),"WARNING")[0]
        self.assertTrue(episode["censored"]); self.assertEqual(episode["censoring_reason"],"timestamp_gap")

    def test_return_to_normal_confirms_warning_closure(self):
        rows=[audit_row("2026-01-01T00:00:00",status="NORMAL",actionable=False),audit_row("2026-01-01T00:01:00",status="WARNING",actionable=False),audit_row("2026-01-01T00:02:00",status="NORMAL",actionable=False)]
        episode=unsafe_episodes(pd.DataFrame(rows),pd.Series(["NORMAL","WARNING","NORMAL"]),"WARNING")[0]
        self.assertFalse(episode["censored"]); self.assertEqual(episode["censoring_reason"],"")

    def test_numeric_fn_boolean_contradiction_is_untrusted(self):
        target=REQUIRED[0]; row=audit_row(); evidence=json.loads(row["probability_target_evidence"]); evidence[target]["validation_fn_rate"]=.2; row["probability_target_evidence"]=json.dumps(evidence)
        trusted,_,failures=trusted_evidence(pd.DataFrame([row])); self.assertFalse(trusted.iloc[0]); self.assertIn(f"validation_fn_numeric_contradiction:{target}",failures)

    def test_threshold_mapping_mismatch_is_untrusted(self):
        target=REQUIRED[0]; row=audit_row(); evidence=json.loads(row["probability_target_evidence"]); evidence[target]["selected_threshold"]=.4; row["probability_target_evidence"]=json.dumps(evidence)
        trusted,_,failures=trusted_evidence(pd.DataFrame([row])); self.assertFalse(trusted.iloc[0]); self.assertIn(f"threshold_evidence_mismatch:{target}",failures)

    def test_current_status_rates_and_target_coverage_are_reported(self):
        rows=[audit_row("2026-01-01T00:00:00",status="NORMAL"),audit_row("2026-01-01T00:01:00",status="WARNING",actionable=False)]
        rows[0]["status"]="WARNING"; rows[1]["status"]="NORMAL"; frame=pd.DataFrame(rows); report=evaluate(frame,frame.source_label)
        self.assertEqual(report["current_status"]["current_status_fp_rate"],1.0); self.assertEqual(report["current_status"]["current_status_fn_rate"],1.0)
        self.assertIn("probability_warning_6h",report["coverage_by_target_and_horizon"])

    def test_required_list_contradiction_and_invalid_reason_mapping_fail(self):
        row=audit_row(); row["required_model_targets"]="[]"; trusted,_,failures=trusted_evidence(pd.DataFrame([row])); self.assertFalse(trusted.iloc[0]); self.assertIn("required_model_targets_mismatch",failures)
        row=audit_row(); mapping=json.loads(row["probability_target_ineligibility_reasons"]); mapping[REQUIRED[0]]=""; row["probability_target_ineligibility_reasons"]=json.dumps(mapping)
        trusted,_,failures=trusted_evidence(pd.DataFrame([row])); self.assertFalse(trusted.iloc[0]); self.assertIn("invalid_target_reason_mapping",failures)

    def test_invalid_timestamp_breaks_history_and_prevents_24h_credit(self):
        frame,actual,trusted=self._horizon_frame(20); frame.loc[10,"timestamp"]="not-a-time"
        result=horizon_metrics(frame,actual,trusted,"WARNING",maximum_gap_minutes=61)
        self.assertEqual(result["24h"]["eligible_unsafe_episodes"],0); self.assertEqual(result["24h"]["episodes"][0]["horizon_ineligibility_reason"],"invalid_timestamp")
        counts=timestamp_validation(frame,61)[3]; self.assertEqual(counts["invalid_timestamp_rows"],1)

    def test_duplicate_and_non_monotonic_timestamps_break_continuity(self):
        rows=[audit_row("2026-01-01T00:00:00"),audit_row("2026-01-01T00:00:00"),audit_row("2025-12-31T23:59:00")]
        counts=timestamp_validation(pd.DataFrame(rows),10)[3]; self.assertEqual(counts["duplicate_timestamp_rows"],1); self.assertEqual(counts["non_monotonic_timestamp_rows"],1)

    def test_target_specific_trigger_attribution(self):
        row=audit_row(); row.update({target:.1 for target in PROBS}); row[PROBS[0]]=.6; row["recommendation_trigger_targets"]=json.dumps([PROBS[0]]); row["probability_threshold_crossings"]=json.dumps([PROBS[0]]); row["recommendation_trigger_probability_targets"]=json.dumps([PROBS[0]]); row["recommendation_trigger_rules"]=json.dumps(["warning_probability_12h"])
        frame=pd.DataFrame([row]); trusted,_,_=trusted_evidence(frame); coverage=target_horizon_coverage(frame,trusted)
        self.assertEqual(coverage[PROBS[0]]["target_triggered_actionable_rows"],0); self.assertEqual(coverage[PROBS[1]]["target_triggered_actionable_rows"],0)

    def test_streaming_compact_evaluator_matches_in_memory_model_metrics(self):
        with __import__("tempfile").TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for hour in range(30):
                status = "WARNING" if hour >= 24 else "NORMAL"
                row = audit_row(
                    (datetime(2026, 1, 1) + timedelta(hours=hour)).isoformat(),
                    status=status, actionable=False,
                )
                row.update({target: (.8 if 18 <= hour < 24 else .1) for target in PROBS})
                row.update({
                    "raw_status": status, "raw_safety_status": status,
                    "stabilized_status": status, "event_status": status,
                })
                rows.append(row)
            frame = pd.DataFrame(rows)
            labels = pd.DataFrame({"timestamp": frame["timestamp"], "health_status": frame["source_label"]})
            replay_path = root / "replay.csv"; labels_path = root / "labels.csv"
            frame.to_csv(replay_path, index=False); labels.to_csv(labels_path, index=False)
            compact, actual = load_compact_frames(replay_path, labels_path, chunksize=7)
            validation = validate_compact_audit(compact)
            self.assertTrue(validation["validated"])
            expected = per_target_metrics(
                frame, frame["source_label"], pd.Series(False, index=frame.index), 61.0
            )
            observed = per_target_metrics(
                compact, actual, pd.Series(False, index=compact.index), 61.0
            )
            self.assertEqual(observed, expected)

if __name__=="__main__": unittest.main()
