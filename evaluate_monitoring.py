#!/usr/bin/env python3
"""Fail-closed status, deployment-audit, and pre-event horizon evaluation."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from spindle_monitor.policy_contract import CONTRACT, FORECAST_POLICY_CONTRACT_VERSION, FORECAST_POLICY_VERSION, URGENCY_POLICY_VERSION, URGENCY_POLICY_PARAMETERS

STATUSES = ("NORMAL", "WARNING", "CRITICAL")
ACTIONABLE = {"PLAN_INSPECTION", "MAINTENANCE_RECOMMENDED_SOON", "PLAN_MAINTENANCE"}
POLICY_VERSION = FORECAST_POLICY_VERSION
THRESHOLD_SCHEMA_VERSION = "1.0"
MANDATORY_TARGETS = CONTRACT.mandatory_targets

def label_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    for name in ([requested] if requested else []) + ["source_label", "health_status"]:
        if name and name in frame and name not in {"status", "effective_status", "raw_status"}:
            if frame[name].astype(str).str.upper().isin(STATUSES).all(): return name
    raise ValueError("no valid ground-truth label column (expected source_label or health_status); predictions cannot be labels")

def _bool(value: Any) -> bool:
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)) and value in (0, 1): return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}: return True
    if text in {"false", "0"}: return False
    raise ValueError(f"invalid boolean {value!r}")

def _json(value: Any, kind: type) -> Any:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, kind): raise ValueError(f"expected {kind.__name__}")
    return parsed

def _json_column(values: pd.Series, kind: type, default: Any) -> list[Any]:
    """Parse repeated audit JSON once per distinct serialized value."""
    cache: dict[Any, Any] = {}
    output: list[Any] = []
    for value in values:
        key = value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
        if key not in cache:
            try: cache[key] = _json(value, kind)
            except Exception: cache[key] = default() if callable(default) else default
        output.append(cache[key])
    return output

def _safe_bool(value: Any, default: bool = False) -> bool:
    try: return _bool(value)
    except (ValueError, TypeError): return default

def episodes(mask: pd.Series, frame: pd.DataFrame, maximum_gap_minutes: float = 10.0) -> int:
    times = pd.to_datetime(frame["timestamp"], errors="coerce")
    lifecycle = frame.get("lifecycle_id", pd.Series("default", index=frame.index)).fillna("unknown")
    split = lifecycle.ne(lifecycle.shift()) | times.diff().dt.total_seconds().gt(maximum_gap_minutes * 60).fillna(False)
    return int((mask & (~mask.shift(fill_value=False) | split)).sum())

AUDIT_COLUMNS = {
    "model_version", "model_maturity_stage", "model_stage", "model_data_domain", "production_eligible",
    "production_policy_passed", "production_policy_reasons_json", "model_revoked", "model_superseded",
    "model_corrupted", "model_metadata_valid", "model_artifacts_valid", "model_schema_compatible",
    "model_feature_schema_compatible", "model_load_failures_json", "required_model_targets",
    "loaded_model_targets", "missing_required_model_targets", "selected_probability_thresholds",
    "configured_mandatory_targets", "metadata_required_targets", "effective_required_targets", "mandatory_targets_missing_from_metadata",
    "probability_target_eligibility", "probability_target_ineligibility_reasons", "probability_target_evidence",
    "physical_validation_passed", "physical_validation_reasons_json", "feature_ready",
    "feature_readiness_reasons_json", "ood_detected", "target_support_valid", "disagreement_detected",
    "forecast_withheld", "withholding_reasons_json", "recommendation_source", "recommendation_actionable",
    "forecast_policy_version", "probability_threshold_schema_version",
    "runtime_probability_fn_ceiling", "model_recorded_probability_fn_ceiling", "effective_probability_fn_ceiling",
    "recommendation_trigger_targets",
    "eta_target_evidence", "urgency_policy_version", "urgency_policy_parameters", "forecast_policy_contract_version",
    "probability_threshold_crossings", "recommendation_trigger_probability_targets", "recommendation_trigger_eta_targets", "recommendation_trigger_rules",
    "runtime_validation_fn_ceiling", "runtime_test_fn_ceiling", "model_recorded_validation_fn_ceilings", "model_recorded_test_fn_ceilings", "effective_validation_fn_ceilings", "effective_test_fn_ceilings",
}

def trusted_evidence(frame: pd.DataFrame) -> tuple[pd.Series, bool, dict[str, int]]:
    missing_columns = sorted(AUDIT_COLUMNS - set(frame.columns))
    if missing_columns:
        return pd.Series(False, index=frame.index), False, {
            "missing_audit_evidence_row_count": len(frame), "missing_audit_columns": len(missing_columns)
        }
    counts: dict[str, int] = {}; values: list[bool] = []
    def fail(key: str): counts[key] = counts.get(key, 0) + 1
    for _, row in frame.iterrows():
        failures: set[str] = set()
        def need(condition: bool, reason: str):
            if not condition: failures.add(reason)
        try:
            required = _json(row.required_model_targets, list); loaded = _json(row.loaded_model_targets, list)
            configured = _json(row.configured_mandatory_targets, list); metadata_required = _json(row.metadata_required_targets, list)
            effective = _json(row.effective_required_targets, list); mandatory_missing = _json(row.mandatory_targets_missing_from_metadata, list)
            absent = _json(row.missing_required_model_targets, list); loads = _json(row.model_load_failures_json, dict)
            thresholds = _json(row.selected_probability_thresholds, dict); eligibility = _json(row.probability_target_eligibility, dict)
            target_reasons = _json(row.probability_target_ineligibility_reasons, dict); evidence = _json(row.probability_target_evidence, dict)
            policy_reasons = _json(row.production_policy_reasons_json, list); withholding = _json(row.withholding_reasons_json, list)
            physical = _json(row.physical_validation_reasons_json, list); feature = _json(row.feature_readiness_reasons_json, list)
            recorded_ceilings=_json(row.model_recorded_probability_fn_ceiling,dict); effective_ceilings=_json(row.effective_probability_fn_ceiling,dict)
            trigger_targets=_json(row.recommendation_trigger_targets,list); runtime_ceiling=float(row.runtime_probability_fn_ceiling)
            eta_evidence=_json(row.eta_target_evidence,dict); urgency_parameters=_json(row.urgency_policy_parameters,dict)
            probability_triggers=_json(row.recommendation_trigger_probability_targets,list); eta_triggers=_json(row.recommendation_trigger_eta_targets,list); trigger_rules=_json(row.recommendation_trigger_rules,list); crossings=_json(row.probability_threshold_crossings,list)
            validation_recorded=_json(row.model_recorded_validation_fn_ceilings,dict); test_recorded=_json(row.model_recorded_test_fn_ceilings,dict); validation_effective=_json(row.effective_validation_fn_ceilings,dict); test_effective=_json(row.effective_test_fn_ceilings,dict)
            bools = {name: _bool(row[name]) for name in (
                "production_eligible", "production_policy_passed", "model_revoked", "model_superseded", "model_corrupted",
                "model_metadata_valid", "model_artifacts_valid", "model_schema_compatible", "model_feature_schema_compatible",
                "physical_validation_passed", "feature_ready", "ood_detected", "target_support_valid", "disagreement_detected",
                "forecast_withheld", "recommendation_actionable")}
        except Exception:
            fail("malformed_audit_row_count"); values.append(False); continue
        need(bool(str(row.model_version).strip()), "missing_model_version")
        need(row.model_maturity_stage == "deployed", "model_not_deployed")
        need(row.model_stage in {"production", "plant_production"}, "model_stage_not_allowed")
        need(row.model_data_domain == "plant", "data_domain_not_allowed")
        need(bools["production_eligible"], "production_ineligible")
        need(bools["production_policy_passed"] and not policy_reasons, "production_policy_contradiction")
        need(not bools["model_revoked"], "model_revoked"); need(not bools["model_superseded"], "model_superseded")
        need(not bools["model_corrupted"], "model_corrupted")
        need(bools["model_metadata_valid"], "model_metadata_invalid"); need(bools["model_artifacts_valid"], "model_artifacts_invalid")
        need(bools["model_schema_compatible"], "model_schema_incompatible"); need(bools["model_feature_schema_compatible"], "feature_schema_mismatch")
        need(set(required).issubset(set(loaded)) and not absent and not loads, "required_model_artifact_evidence_failed")
        need(set(configured) == set(MANDATORY_TARGETS), "configured_mandatory_targets_mismatch")
        need(math.isfinite(runtime_ceiling) and 0<=runtime_ceiling<=1,"invalid_runtime_fn_ceiling")
        need(set(effective) == set(MANDATORY_TARGETS) | set(metadata_required), "effective_required_targets_mismatch")
        need(set(required)==set(effective),"required_model_targets_mismatch")
        need(set(absent)==set(effective)-set(loaded),"missing_required_targets_mismatch")
        need(set(mandatory_missing)==set(configured)-set(metadata_required),"mandatory_metadata_gap_mismatch")
        need(not mandatory_missing and set(MANDATORY_TARGETS).issubset(set(metadata_required)), "mandatory_targets_missing_from_metadata")
        need(set(effective).issubset(set(loaded)), "mandatory_or_required_target_not_loaded")
        need(str(row.forecast_policy_version) == POLICY_VERSION, "unsupported_forecast_policy_version")
        need(str(row.forecast_policy_contract_version) == FORECAST_POLICY_CONTRACT_VERSION, "unsupported_policy_contract_version")
        need(str(row.urgency_policy_version) == URGENCY_POLICY_VERSION and urgency_parameters == URGENCY_POLICY_PARAMETERS, "unsupported_urgency_policy")
        for target in CONTRACT.mandatory_eta_targets:
            record=eta_evidence.get(target)
            need(isinstance(record,dict),f"eta_target_evidence_missing:{target}")
            if isinstance(record,dict):
                need(record.get("target_required") is True,f"eta_target_not_required:{target}"); need(record.get("target_eligible") is True,f"eta_target_ineligible:{target}")
                need(record.get("model_loaded") is True and target in loaded,f"eta_model_not_loaded:{target}")
                need(record.get("model_schema_compatible") is True and record.get("feature_schema_compatible") is True,f"eta_schema_incompatible:{target}")
                need(isinstance(record.get("target_ineligibility_reasons"),list) and not record.get("target_ineligibility_reasons"),f"eta_evidence_malformed:{target}")
                lo,hi=record.get("support_min_hours"),record.get("support_max_hours"); need(isinstance(lo,(int,float)) and isinstance(hi,(int,float)) and not isinstance(lo,bool) and not isinstance(hi,bool) and math.isfinite(lo) and math.isfinite(hi) and lo>=0 and hi>=lo,f"eta_support_invalid:{target}")
                need(record.get("physical_validation_passed") is True and record.get("baseline_passed") is True,f"eta_evidence_malformed:{target}")
        need(str(row.probability_threshold_schema_version) == THRESHOLD_SCHEMA_VERSION, "unsupported_threshold_schema_version")
        for target in [str(x) for x in effective if str(x).startswith("probability_")]:
            value = thresholds.get(target); record = evidence.get(target)
            need(target in eligibility and isinstance(eligibility.get(target),bool), "invalid_target_eligibility_mapping")
            need(target in target_reasons and isinstance(target_reasons.get(target),list) and all(isinstance(x,str) for x in target_reasons.get(target,[])), "invalid_target_reason_mapping")
            need(isinstance(record, dict), f"target_evidence_missing:{target}")
            need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 < value < 1, f"threshold_invalid:{target}")
            if isinstance(record, dict):
                reasons = record.get("target_ineligibility_reasons")
                need(isinstance(reasons,list) and all(isinstance(x,str) for x in reasons),"invalid_target_reason_mapping")
                need(eligibility.get(target) is True and target_reasons.get(target)==[] and record.get("target_eligible") is True and reasons == [], f"target_ineligible:{target}")
                need(isinstance(record.get("target_required"),bool) and record.get("target_required") is True, f"target_not_declared_required:{target}")
                need(record.get("validation_fn_ceiling_met") is True, f"validation_fn_ceiling_failed:{target}")
                need(record.get("test_fn_ceiling_met") is True, f"test_fn_ceiling_failed:{target}")
                numeric_names=("validation_fp_rate","validation_fn_rate","validation_fn_ceiling","test_fp_rate","test_fn_rate","test_fn_ceiling")
                need(all(isinstance(record.get(n),(int,float)) and not isinstance(record.get(n),bool) and math.isfinite(float(record[n])) and 0<=float(record[n])<=1 for n in numeric_names),f"invalid_probability_metric:{target}")
                if all(isinstance(record.get(n),(int,float)) and not isinstance(record.get(n),bool) and math.isfinite(float(record[n])) for n in ("validation_fn_rate","validation_fn_ceiling")):
                    computed=float(record["validation_fn_rate"])<=float(record["validation_fn_ceiling"])
                    need(record.get("validation_fn_ceiling_met") is computed,f"validation_fn_numeric_contradiction:{target}")
                    effective_ceiling=min(runtime_ceiling,float(record["validation_fn_ceiling"])); need(float(record["validation_fn_rate"])<=effective_ceiling,f"validation_fn_ceiling_not_met:{target}")
                    need(float(record["validation_fn_ceiling"])<=runtime_ceiling,f"model_fn_ceiling_exceeds_runtime_policy:{target}")
                if all(isinstance(record.get(n),(int,float)) and not isinstance(record.get(n),bool) and math.isfinite(float(record[n])) for n in ("test_fn_rate","test_fn_ceiling")):
                    computed=float(record["test_fn_rate"])<=float(record["test_fn_ceiling"])
                    need(record.get("test_fn_ceiling_met") is computed,f"test_fn_numeric_contradiction:{target}")
                    effective_ceiling=min(runtime_ceiling,float(record["test_fn_ceiling"])); need(float(record["test_fn_rate"])<=effective_ceiling,f"test_fn_ceiling_not_met:{target}")
                    need(float(record["test_fn_ceiling"])<=runtime_ceiling,f"model_fn_ceiling_exceeds_runtime_policy:{target}")
                need(record.get("baseline_passed") is True,f"baseline_not_passed:{target}"); need(record.get("calibration_passed") is True,f"calibration_not_passed:{target}")
                record_threshold=record.get("selected_threshold")
                need(isinstance(record_threshold,(int,float)) and not isinstance(record_threshold,bool) and math.isfinite(float(record_threshold)) and abs(float(record_threshold)-float(value))<=1e-12,f"threshold_evidence_mismatch:{target}")
                def valid_probability(value: Any) -> bool:
                    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(float(value)) and 0<=float(value)<=1
                validation_metrics_valid=all(valid_probability(record.get(n)) for n in ("validation_fn_rate","validation_fn_ceiling"))
                test_metrics_valid=all(valid_probability(record.get(n)) for n in ("test_fn_rate","test_fn_ceiling"))
                runtime_validation_valid=valid_probability(row.runtime_validation_fn_ceiling); runtime_test_valid=valid_probability(row.runtime_test_fn_ceiling)
                need(runtime_validation_valid,"invalid_runtime_validation_fn_ceiling"); need(runtime_test_valid,"invalid_runtime_test_fn_ceiling")
                if validation_metrics_valid and runtime_validation_valid:
                    validation_ceiling=float(record["validation_fn_ceiling"]); runtime_validation=float(row.runtime_validation_fn_ceiling)
                    expected_validation=min(runtime_validation,validation_ceiling)
                    need(recorded_ceilings.get(target)==record.get("validation_fn_ceiling"),f"recorded_fn_ceiling_mismatch:{target}")
                    need(effective_ceilings.get(target)==min(runtime_ceiling,validation_ceiling),f"effective_fn_ceiling_mismatch:{target}")
                    need(validation_recorded.get(target)==record.get("validation_fn_ceiling"), f"validation_ceiling_mapping_mismatch:{target}")
                    need(validation_effective.get(target)==expected_validation, f"validation_ceiling_mapping_mismatch:{target}")
                    need(float(record["validation_fn_rate"])<=expected_validation, f"validation_fn_ceiling_not_met:{target}")
                if test_metrics_valid and runtime_test_valid:
                    test_ceiling=float(record["test_fn_ceiling"]); runtime_test=float(row.runtime_test_fn_ceiling)
                    expected_test=min(runtime_test,test_ceiling)
                    need(test_recorded.get(target)==record.get("test_fn_ceiling"), f"test_ceiling_mapping_mismatch:{target}")
                    need(test_effective.get(target)==expected_test, f"test_ceiling_mapping_mismatch:{target}")
                    need(float(record["test_fn_rate"])<=expected_test, f"test_fn_ceiling_not_met:{target}")
        need(bools["physical_validation_passed"] and not physical, "physical_validation_failed")
        need(bools["feature_ready"] and not feature, "feature_readiness_failed")
        need(not bools["ood_detected"] and bools["target_support_valid"] and not bools["disagreement_detected"], "runtime_guardrail_failed")
        if not bools["forecast_withheld"]:
            need(not withholding, "withholding_contradiction")
        if bools["recommendation_actionable"]:
            need(row.recommendation_source == "ml", "recommendation_not_actionable_ml")
        need(all(isinstance(x,str) for x in required+loaded+absent+configured+metadata_required+effective+mandatory_missing+trigger_targets),"invalid_target_list_schema")
        need(all(isinstance(x,str) for x in policy_reasons+withholding+physical+feature),"invalid_reason_array")
        for target in trigger_targets:
            need(target in set(probability_triggers) | set(eta_triggers), f"trigger_target_union_mismatch")
        need(set(trigger_targets) == set(probability_triggers) | set(eta_triggers), "trigger_target_union_mismatch")
        expected_probability:set[str]=set(); expected_eta:set[str]=set(); expected_rules:set[str]=set()
        def probability_crosses(target: str) -> bool:
            value=row.get(target); threshold=thresholds.get(target)
            return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(float(value)) and isinstance(threshold,(int,float)) and not isinstance(threshold,bool) and 0<float(threshold)<1 and float(value)>=float(threshold) and eligibility.get(target) is True
        if str(row.get("raw_status", row.get("status", ""))).upper() == "NORMAL" and bools["production_policy_passed"]:
            for target in thresholds:
                if probability_crosses(target): expected_probability.add(target)
        need(set(crossings)==expected_probability, "threshold_crossing_mismatch")
        if bools["recommendation_actionable"] and not bools["forecast_withheld"]:
            if probability_crosses(urgency_parameters["critical_probability_6h_target"]): expected_probability={urgency_parameters["critical_probability_6h_target"]}; expected_rules={"critical_probability_6h"}
            elif isinstance(row.get("ml_time_to_critical_hours"),(int,float)) and float(row["ml_time_to_critical_hours"])<=float(urgency_parameters["critical_eta_soon_hours"]): expected_probability=set(); expected_eta={"time_to_critical"}; expected_rules={"critical_eta_soon"}
            elif probability_crosses(urgency_parameters["critical_probability_24h_target"]): expected_probability={urgency_parameters["critical_probability_24h_target"]}; expected_rules={"critical_probability_24h"}
            elif probability_crosses(urgency_parameters["warning_probability_12h_target"]): expected_probability={urgency_parameters["warning_probability_12h_target"]}; expected_rules={"warning_probability_12h"}
            elif isinstance(row.get("ml_time_to_warning_hours"),(int,float)) and float(row["ml_time_to_warning_hours"])<=float(urgency_parameters["warning_eta_plan_hours"]): expected_probability=set(); expected_eta={"time_to_warning"}; expected_rules={"warning_eta_plan"}
        need(set(probability_triggers)==expected_probability, "missing_expected_probability_trigger" if expected_probability-set(probability_triggers) else "unexpected_probability_trigger")
        need(set(eta_triggers)==expected_eta, "missing_expected_eta_trigger" if expected_eta-set(eta_triggers) else "unexpected_eta_trigger")
        need(set(trigger_rules)==expected_rules, "trigger_rule_mismatch")
        for target in probability_triggers:
            need(target in CONTRACT.probability_targets and probability_crosses(target), f"invalid_probability_trigger_target:{target}")
        for target in eta_triggers:
            eta_value=row.get(f"ml_{target}_hours")
            need(target in CONTRACT.eta_targets and isinstance(eta_value,(int,float)) and math.isfinite(float(eta_value)) and float(eta_value)>=0, f"invalid_eta_trigger_target:{target}")
        for reason in failures: fail(reason)
        values.append(not failures)
    counts.setdefault("malformed_audit_row_count", 0); counts.setdefault("missing_audit_evidence_row_count", 0)
    return pd.Series(values, index=frame.index), True, dict(sorted(counts.items()))

def timestamp_validation(frame: pd.DataFrame, maximum_gap_minutes: float = 10.0) -> tuple[pd.Series,pd.Series,pd.Series,dict[str,int]]:
    raw=frame.get("timestamp",pd.Series(None,index=frame.index)); times=pd.to_datetime(raw,errors="coerce")
    lifecycle=frame.get("lifecycle_id",pd.Series("default",index=frame.index)).fillna("unknown")
    segments=[]; reasons=[]; segment=0; counts={"invalid_timestamp_rows":0,"missing_timestamp_rows":0,"duplicate_timestamp_rows":0,"non_monotonic_timestamp_rows":0,"segments_broken_by_timestamp_errors":0}
    previous_valid=False; previous_time=None; previous_invalid_reason=""
    for position in range(len(frame)):
        value=raw.iloc[position]; current=times.iloc[position]; reason=""; split=position==0
        if pd.isna(current):
            reason="missing_timestamp" if value is None or str(value).strip()=="" or str(value).lower()=="nan" else "invalid_timestamp"
            counts[f"{reason}_rows"]+=1; split=True; previous_valid=False; previous_invalid_reason=reason
        elif not previous_valid and position>0:
            reason=previous_invalid_reason or "invalid_timestamp"; split=True; previous_valid=True
        elif previous_valid:
            delta=(current-previous_time).total_seconds()
            if lifecycle.iloc[position]!=lifecycle.iloc[position-1]: reason="lifecycle_boundary"; split=True
            elif delta==0: reason="duplicate_timestamp"; counts["duplicate_timestamp_rows"]+=1; split=True
            elif delta<0: reason="non_monotonic_timestamp"; counts["non_monotonic_timestamp_rows"]+=1; split=True
            elif delta>maximum_gap_minutes*60: reason="timestamp_gap"; split=True
        if split and position>0: segment+=1
        if reason in {"missing_timestamp","invalid_timestamp","duplicate_timestamp","non_monotonic_timestamp"}: counts["segments_broken_by_timestamp_errors"]+=1
        segments.append(segment); reasons.append(reason)
        if not pd.isna(current): previous_time=current; previous_valid=True; previous_invalid_reason=""
    return times,pd.Series(segments,index=frame.index),pd.Series(reasons,index=frame.index),counts

def _segments(frame: pd.DataFrame, maximum_gap_minutes: float) -> pd.Series:
    return timestamp_validation(frame,maximum_gap_minutes)[1]

def unsafe_episodes(frame: pd.DataFrame, actual: pd.Series, episode_type: str, maximum_gap_minutes: float = 10.0) -> list[dict[str, Any]]:
    times = pd.to_datetime(frame.timestamp, errors="coerce"); lifecycle = frame.get("lifecycle_id", pd.Series("default", index=frame.index)).fillna("unknown")
    segment = _segments(frame, maximum_gap_minutes)
    previous=actual.shift(); same_segment=segment.eq(segment.shift())
    if episode_type=="WARNING": starts=actual.eq("WARNING") & previous.eq("NORMAL") & same_segment; match=actual.eq("WARNING")
    elif episode_type=="CRITICAL": starts=actual.eq("CRITICAL") & previous.isin(["NORMAL","WARNING"]) & same_segment; match=actual.eq("CRITICAL")
    else: starts=actual.isin(["WARNING","CRITICAL"]) & previous.eq("NORMAL") & same_segment; match=actual.isin(["WARNING","CRITICAL"])
    result = []
    for number, start in enumerate(frame.index[starts], 1):
        position = frame.index.get_loc(start); end_position = position
        while end_position + 1 < len(frame) and bool(match.iloc[end_position + 1]) and segment.iloc[end_position + 1] == segment.iloc[position]: end_position += 1
        censored=False; censoring_reason=""
        if end_position==len(frame)-1: censored=True; censoring_reason="end_of_file"
        elif segment.iloc[end_position+1]!=segment.iloc[position]:
            censored=True
            censoring_reason="lifecycle_boundary" if lifecycle.iloc[end_position+1]!=lifecycle.iloc[position] else "timestamp_gap"
        result.append({"episode_id": f"{episode_type.lower()}_{number:04d}", "lifecycle_id": str(lifecycle.iloc[position]),
            "episode_type": episode_type, "onset_timestamp": times.iloc[position], "end_timestamp": times.iloc[end_position],
            "censored": censored, "censoring_reason": censoring_reason, "segment_id": int(segment.iloc[position]), "onset_position": position})
    return result

def horizon_metrics(frame: pd.DataFrame, actual: pd.Series, trusted: pd.Series, episode_type: str, horizons=(6,12,24), maximum_gap_minutes: float = 10.0) -> dict[str, Any]:
    times,segment,segment_reasons,_ = timestamp_validation(frame,maximum_gap_minutes); lifecycle = frame.get("lifecycle_id", pd.Series("default", index=frame.index)).fillna("unknown")
    actionable = frame.get("recommendation_actionable", pd.Series(False,index=frame.index)).map(_safe_bool)
    source = frame.get("recommendation_source", pd.Series("",index=frame.index)); candidates = trusted & actionable & source.eq("ml")
    candidates &= actual.isin(["NORMAL","WARNING"]) if episode_type=="CRITICAL" else actual.eq("NORMAL")
    episodes_found = unsafe_episodes(frame, actual, episode_type, maximum_gap_minutes); output = {}
    for horizon in horizons:
        leads=[]; eligible=predicted=insufficient=0; details=[]
        for episode in episodes_found:
            pos=episode["onset_position"]; onset=episode["onset_timestamp"]; seg=episode["segment_id"]
            segment_positions = [i for i in range(pos + 1) if segment.iloc[i] == seg]
            start_time = times.iloc[min(segment_positions)] if segment_positions else onset
            history=(onset-start_time).total_seconds()/3600 if not pd.isna(onset) else 0.0
            if pd.isna(onset) or history < horizon:
                first=min(segment_positions) if segment_positions else pos
                ineligible_reason="insufficient_history"
                if first>0:
                    ineligible_reason=segment_reasons.iloc[first] or ("lifecycle_boundary" if lifecycle.iloc[first]!=lifecycle.iloc[first-1] else "timestamp_gap")
                insufficient += 1; details.append({"episode_id":episode["episode_id"],"history_available_hours":history,"horizon_eligible":False,"horizon_ineligibility_reason":ineligible_reason,"segment_validation_reasons":([ineligible_reason] if ineligible_reason else []),"censored":episode["censored"],"censoring_reason":episode["censoring_reason"]}); continue
            eligible += 1
            mask = candidates & lifecycle.eq(episode["lifecycle_id"]) & segment.eq(seg) & times.ge(onset-pd.Timedelta(hours=horizon)) & times.lt(onset)
            matches = times[mask]
            lead=None
            if len(matches):
                predicted += 1; chosen=matches.min(); lead=(onset-chosen).total_seconds()/3600; leads.append(lead)
            details.append({"episode_id":episode["episode_id"],"history_available_hours":history,"horizon_eligible":True,"horizon_ineligibility_reason":"","segment_validation_reasons":[],"predicted":lead is not None,"lead_time_hours":lead,"censored":episode["censored"],"censoring_reason":episode["censoring_reason"]})
        output[f"{horizon}h"]={"eligible_unsafe_episodes":eligible,"predicted_unsafe_episodes":predicted,"missed_unsafe_episodes":eligible-predicted,
            "insufficient_history_episode_count":insufficient,"episode_recall":(predicted/eligible if eligible else None),"episode_miss_rate":((eligible-predicted)/eligible if eligible else None),
            "median_lead_time_hours":(median(leads) if leads else None),"minimum_lead_time_hours":(min(leads) if leads else None),"maximum_lead_time_hours":(max(leads) if leads else None),"episodes":details}
    return output

def false_alert_metrics(frame: pd.DataFrame, actual: pd.Series, trusted: pd.Series, maximum_gap_minutes: float = 10.0) -> dict[str, Any]:
    normal=actual.eq("NORMAL"); actionable=frame.get("recommendation_actionable",pd.Series(False,index=frame.index)).map(_safe_bool)
    source=frame.get("recommendation_source",pd.Series("",index=frame.index)); mask=normal & trusted & actionable & source.eq("ml")
    times=pd.to_datetime(frame.timestamp,errors="coerce"); segment=_segments(frame,maximum_gap_minutes)
    starts=mask & (~mask.shift(fill_value=False)|segment.ne(segment.shift())); durations=[]
    for start in frame.index[starts]:
        pos=frame.index.get_loc(start); end=pos
        while end+1<len(frame) and bool(mask.iloc[end+1]) and segment.iloc[end+1]==segment.iloc[pos]: end+=1
        if not pd.isna(times.iloc[end]) and not pd.isna(times.iloc[pos]): durations.append(max((times.iloc[end]-times.iloc[pos]).total_seconds()/3600,0))
    # Same validated segments are used for normal periods and false alerts.
    normal_starts = normal & (~normal.shift(fill_value=False) | segment.ne(segment.shift()))
    normal_periods=int(normal_starts.sum())
    normal_with_alert=0
    for pos in frame.index[normal_starts]:
        i=frame.index.get_loc(pos); end=i
        while end+1<len(frame) and bool(normal.iloc[end+1]) and segment.iloc[end+1]==segment.iloc[i]: end+=1
        if bool(mask.iloc[i:end+1].any()): normal_with_alert += 1
    exposure=0.0
    for seg in sorted(segment.unique()):
        indexes=[i for i in range(len(frame)) if segment.iloc[i]==seg and bool(normal.iloc[i])]
        if len(indexes)>1 and not pd.isna(times.iloc[indexes[0]]) and not pd.isna(times.iloc[indexes[-1]]): exposure += max((times.iloc[indexes[-1]]-times.iloc[indexes[0]]).total_seconds()/3600,0)
    return {"normal_rows_evaluated":int(normal.sum()),"false_alert_rows":int(mask.sum()),"false_alert_row_rate":(float(mask.sum()/normal.sum()) if normal.any() else None),
        "normal_periods_evaluated":normal_periods,"normal_periods_with_false_alert":normal_with_alert,"normal_period_false_alert_rate":(normal_with_alert/normal_periods if normal_periods else None),"false_alert_episodes":int(starts.sum()),"false_alert_episode_rate_per_100_normal_hours":(100*int(starts.sum())/exposure if exposure else None),
        "median_false_alert_episode_duration_hours":(median(durations) if durations else None),"maximum_false_alert_episode_duration_hours":(max(durations) if durations else None),"duration_available_episode_count":len(durations)}

def target_horizon_coverage(frame: pd.DataFrame, trusted: pd.Series) -> dict[str, Any]:
    output={}
    for target in [f"probability_{kind}_{h}h" for kind in ("warning","critical") for h in (6,12,24)]:
        values=pd.to_numeric(frame.get(target,pd.Series(float("nan"),index=frame.index)),errors="coerce")
        valid=values.map(lambda x: math.isfinite(x) and 0<=x<=1)
        above=[]; missing=[]; ineligible=[]; trigger_declared=[]
        for index,row in frame.iterrows():
            try:
                thresholds=_json(row.get("selected_probability_thresholds","{}"),dict); loaded=_json(row.get("loaded_model_targets","[]"),list); eligibility=_json(row.get("probability_target_eligibility","{}"),dict)
                triggers=_json(row.get("recommendation_trigger_targets","[]"),list)
                above.append(bool(valid.loc[index] and target in thresholds and isinstance(thresholds[target],(int,float)) and 0<float(thresholds[target])<1 and values.loc[index]>=thresholds[target])); missing.append(target not in loaded); ineligible.append(eligibility.get(target) is not True); trigger_declared.append(target in triggers)
            except Exception: above.append(False); missing.append(True); ineligible.append(True); trigger_declared.append(False)
        withheld=frame.get("forecast_withheld",pd.Series(True,index=frame.index)).map(lambda x: _safe_bool(x, True))
        actionable=frame.get("recommendation_actionable",pd.Series(False,index=frame.index)).map(_safe_bool)
        target_triggered=trusted & actionable & valid & pd.Series(above,index=frame.index) & ~pd.Series(ineligible,index=frame.index) & pd.Series(trigger_declared,index=frame.index)
        any_action=trusted & actionable
        output[target]={"rows_with_valid_probability":int(valid.sum()),"rows_above_selected_threshold":int(sum(above)),"target_triggered_actionable_rows":int(target_triggered.sum()),
            "rows_with_any_trusted_actionable_recommendation":int(any_action.sum()),"withheld_rows":int(withheld.sum()),"missing_model_rows":int(sum(missing)),"ineligible_target_rows":int(sum(ineligible)),
            "probability_coverage_rate":float(valid.mean()),"target_trigger_rate":float(target_triggered.mean())}
    return output

def _binary_metrics(labels: pd.Series, predicted: pd.Series, probabilities: pd.Series | None = None) -> dict[str, Any]:
    labels = labels.astype(bool); predicted = predicted.astype(bool)
    tp=int((labels&predicted).sum()); tn=int((~labels&~predicted).sum())
    fp=int((~labels&predicted).sum()); fn=int((labels&~predicted).sum())
    precision=tp/(tp+fp) if tp+fp else None; recall=tp/(tp+fn) if tp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    result={"tp":tp,"tn":tn,"fp":fp,"fn":fn,"positive_rows":int(labels.sum()),"negative_rows":int((~labels).sum()),
        "fn_rate":fn/(tp+fn) if tp+fn else None,"fp_rate":fp/(tn+fp) if tn+fp else None,
        "precision":precision,"recall":recall,"specificity":specificity,
        "f1":(2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None)}
    if probabilities is not None:
        probability=pd.to_numeric(probabilities,errors="coerce"); available=probability.notna()&np.isfinite(probability)&probability.between(0,1)
        result["probability_coverage"]=float(available.mean()); result["probability_rows"]=int(available.sum())
        if available.any():
            y=labels[available].astype(int); p=probability[available].clip(1e-9,1-1e-9)
            both=y.nunique()==2
            result.update({"brier_score":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),
                "pr_auc":float(average_precision_score(y,p)) if both else None,"roc_auc":float(roc_auc_score(y,p)) if both else None,
                "calibration_error":float(sum(float(mask.mean())*abs(float(y[mask].mean())-float(p[mask].mean())) for low in np.linspace(0,1,10,endpoint=False)
                    if (mask:=((p>=low)&(p<(low+.1) if low<.9 else p<=1))).any()))})
    return result

def _future_event_labels(frame: pd.DataFrame, actual: pd.Series, kind: str, hours: int, maximum_gap_minutes: float) -> tuple[pd.Series,list[dict[str,Any]]]:
    times,segments,_,_=timestamp_validation(frame,maximum_gap_minutes)
    lifecycle=frame.get("lifecycle_id",pd.Series("default",index=frame.index)).fillna("unknown")
    events=unsafe_episodes(frame,actual,"WARNING" if kind=="warning" else "CRITICAL",maximum_gap_minutes)
    labels=pd.Series(False,index=frame.index)
    for event in events:
        onset=event["onset_timestamp"]
        mask=(segments.eq(event["segment_id"])&lifecycle.eq(event["lifecycle_id"])
              &times.ge(onset-pd.Timedelta(hours=hours))&times.lt(onset))
        labels |= mask
    return labels,events

def _false_episode_detail(mask: pd.Series, frame: pd.DataFrame, maximum_gap_minutes: float) -> dict[str,Any]:
    times,segments,_,_=timestamp_validation(frame,maximum_gap_minutes); lifecycle=frame.get("lifecycle_id",pd.Series("default",index=frame.index)).fillna("unknown")
    starts=mask&(~mask.shift(fill_value=False)|segments.ne(segments.shift())); durations=[]; crosses=0
    for index in frame.index[starts]:
        pos=frame.index.get_loc(index); end=pos
        while end+1<len(frame) and bool(mask.iloc[end+1]) and segments.iloc[end+1]==segments.iloc[pos]: end+=1
        if pd.notna(times.iloc[pos]) and pd.notna(times.iloc[end]): durations.append(max(0.0,(times.iloc[end]-times.iloc[pos]).total_seconds()/3600))
        if lifecycle.iloc[pos]!=lifecycle.iloc[end]: crosses+=1
    normal_hours=0.0
    for segment_id in segments.unique():
        positions=np.flatnonzero(segments.to_numpy()==segment_id)
        if len(positions)>1 and pd.notna(times.iloc[positions[0]]) and pd.notna(times.iloc[positions[-1]]): normal_hours += max(0.0,(times.iloc[positions[-1]]-times.iloc[positions[0]]).total_seconds()/3600)
    return {"false_positive_rows":int(mask.sum()),"false_alert_episodes":int(starts.sum()),
        "false_alert_episode_rate_per_operating_day":(int(starts.sum())/(normal_hours/24) if normal_hours else None),
        "median_false_alert_duration_hours":median(durations) if durations else None,"maximum_false_alert_duration_hours":max(durations) if durations else None,
        "normal_time_under_false_alert":(sum(durations)/normal_hours if normal_hours else None),"alerts_crossing_lifecycle_boundaries":crosses}

def _target_event_metrics(frame:pd.DataFrame, events:list[dict[str,Any]], predicted:pd.Series, hours:int, maximum_gap_minutes:float)->dict[str,Any]:
    times,segments,_,_=timestamp_validation(frame,maximum_gap_minutes); lifecycle=frame.get("lifecycle_id",pd.Series("default",index=frame.index)).fillna("unknown")
    leads=[]; missed=0; detail=[]
    for event in events:
        onset=event["onset_timestamp"]; mask=(predicted&segments.eq(event["segment_id"])&lifecycle.eq(event["lifecycle_id"])
            &times.ge(onset-pd.Timedelta(hours=hours))&times.lt(onset)); matches=times[mask]
        lead=None
        if len(matches): lead=float((onset-matches.min()).total_seconds()/3600); leads.append(lead)
        else: missed+=1
        detail.append({"episode_id":event["episode_id"],"onset_timestamp":onset.isoformat() if pd.notna(onset) else None,"warned_before_onset":lead is not None,"first_alert_lead_time_hours":lead})
    return {"event_onsets":len(events),"warned_before_onset":len(events)-missed,"missed":missed,"event_fn_rate":(missed/len(events) if events else None),
        "median_lead_time_hours":median(leads) if leads else None,"minimum_lead_time_hours":min(leads) if leads else None,"maximum_lead_time_hours":max(leads) if leads else None,
        **{f"warned_at_least_{lead}h_early_rate":(sum(x>=lead for x in leads)/len(events) if events else None) for lead in (1,3,6,12,24)},"episodes":detail}

def status_layer_metrics(frame:pd.DataFrame,actual:pd.Series)->dict[str,Any]:
    result={}
    for column in ("status","raw_status","raw_safety_status","stabilized_status","event_status"):
        if column not in frame: continue
        predicted=frame[column].astype(str).str.upper(); normal=actual.eq("NORMAL"); unsafe=actual.isin(["WARNING","CRITICAL"])
        result[column]={"actual_label_source":"external_ground_truth","predicted_column":column,"fp_count":int((normal&predicted.isin(["WARNING","CRITICAL"])).sum()),
            "fp_denominator":int(normal.sum()),"fp_rate":float((normal&predicted.isin(["WARNING","CRITICAL"])).sum()/normal.sum()) if normal.any() else None,
            "fn_count":int((unsafe&predicted.eq("NORMAL")).sum()),"fn_denominator":int(unsafe.sum()),"fn_rate":float((unsafe&predicted.eq("NORMAL")).sum()/unsafe.sum()) if unsafe.any() else None}
    actionable=frame.get("recommendation_actionable",pd.Series(False,index=frame.index)).map(_safe_bool)
    normal=actual.eq("NORMAL"); unsafe=actual.isin(["WARNING","CRITICAL"])
    result["recommendation_actionability"]={"fp_count":int((normal&actionable).sum()),"fp_denominator":int(normal.sum()),"fp_rate":float((normal&actionable).sum()/normal.sum()) if normal.any() else None,
        "fn_count":int((unsafe&~actionable).sum()),"fn_denominator":int(unsafe.sum()),"fn_rate":float((unsafe&~actionable).sum()/unsafe.sum()) if unsafe.any() else None,
        "interpretation":"predictive recommendation only; manufacturer unsafe rows are handled by safety status and are not expected to require a predictive recommendation"}
    return result

def per_target_metrics(frame:pd.DataFrame,actual:pd.Series,trusted:pd.Series,maximum_gap_minutes:float)->dict[str,Any]:
    threshold_maps=_json_column(frame.get("selected_probability_thresholds",pd.Series("{}",index=frame.index)),dict,dict)
    eligibility_maps=_json_column(frame.get("probability_target_eligibility",pd.Series("{}",index=frame.index)),dict,dict)
    crossing_sets=[set(value) for value in _json_column(frame.get("probability_threshold_crossings",pd.Series("[]",index=frame.index)),list,list)]
    loaded_sets=[set(value) for value in _json_column(frame.get("loaded_model_targets",pd.Series("[]",index=frame.index)),list,list)]
    output={}
    for kind in ("warning","critical"):
        for hours in (6,12,24):
            target=f"probability_{kind}_{hours}h"; labels,events=_future_event_labels(frame,actual,kind,hours,maximum_gap_minutes)
            reconciled=pd.to_numeric(frame.get(f"reconciled_{target}",frame.get(target,pd.Series(np.nan,index=frame.index))),errors="coerce")
            raw=pd.to_numeric(frame.get(f"raw_{target}",reconciled),errors="coerce")
            threshold=pd.Series([value.get(target) for value in threshold_maps],index=frame.index,dtype="float64")
            valid_raw=raw.notna()&np.isfinite(raw)&raw.between(0,1)&threshold.notna()&threshold.between(0,1,inclusive="neither")
            valid_reconciled=reconciled.notna()&np.isfinite(reconciled)&reconciled.between(0,1)&threshold.notna()&threshold.between(0,1,inclusive="neither")
            model_raw=valid_raw&(raw>=threshold); model_reconciled=valid_reconciled&(reconciled>=threshold)
            eligible=pd.Series([value.get(target) is True for value in eligibility_maps],index=frame.index)
            loaded=pd.Series([target in value for value in loaded_sets],index=frame.index)
            strict=pd.Series([target in value for value in crossing_sets],index=frame.index)&trusted&eligible&loaded
            normal_negative=~labels&actual.eq("NORMAL")
            output[target]={"target":target,"horizon_hours":hours,"actual_label_source":f"next {kind.upper()} onset within elapsed {hours}h in same lifecycle/continuous segment",
                "threshold_source":"serialized validation-selected per-target threshold","model_only_raw":_binary_metrics(labels,model_raw,raw),
                "model_only_reconciled":_binary_metrics(labels,model_reconciled,reconciled),"strict_operational":_binary_metrics(labels,strict,reconciled),
                "event_level_model_only":_target_event_metrics(frame,events,model_reconciled,hours,maximum_gap_minutes),
                "event_level_strict_operational":_target_event_metrics(frame,events,strict,hours,maximum_gap_minutes),
                "false_alerts_model_only":_false_episode_detail(model_reconciled&normal_negative,frame,maximum_gap_minutes),
                "false_alerts_strict_operational":_false_episode_detail(strict&normal_negative,frame,maximum_gap_minutes),
                "availability":{"raw_probability_rows":int(valid_raw.sum()),"reconciled_probability_rows":int(valid_reconciled.sum()),"loaded_rows":int(loaded.sum()),"eligible_rows":int(eligible.sum()),
                    "strict_crossing_rows":int(strict.sum()),"withheld_or_ineligible_positive_rows":int((labels&~strict).sum())},
                "reconciliation":{"changed_probability_rows":int((valid_raw&valid_reconciled&~np.isclose(raw,reconciled)).sum()),
                    "changed_classification_rows":int((model_raw!=model_reconciled).sum())}}
    return output

def fp_root_cause_breakdown(frame:pd.DataFrame,actual:pd.Series)->dict[str,Any]:
    normal=actual.eq("NORMAL"); actionable=frame.get("recommendation_actionable",pd.Series(False,index=frame.index)).map(_safe_bool); fp=normal&actionable
    result={"normal_rows":int(normal.sum()),"actionable_false_positive_rows":int(fp.sum())}
    for column in ("maintenance_urgency","recommendation_source","lifecycle_state","quality_status","anomaly_type","model_stage","model_maturity_stage"):
        if column in frame: result[f"by_{column}"]={str(k):int(v) for k,v in frame.loc[fp,column].fillna("missing").value_counts().items()}
    for column in ("forecast_withheld","ood_detected","disagreement_detected","anomaly_is_active","target_support_violation","withheld_due_to_reset_suppression"):
        if column in frame: result[f"{column}_rows"]=int((fp&frame[column].map(_safe_bool)).sum())
    trigger_counts={}; reason_counts={}
    for index in frame.index[fp]:
        for column,destination in (("recommendation_trigger_targets",trigger_counts),("recommendation_trigger_rules",reason_counts)):
            try:
                for value in _json(frame.at[index,column],list): destination[str(value)]=destination.get(str(value),0)+1
            except Exception: destination["malformed_or_missing"] = destination.get("malformed_or_missing",0)+1
    lifecycle=frame.get("lifecycle_id",pd.Series("default",index=frame.index)); first=lifecycle.ne(lifecycle.shift())
    crossings=frame.get("probability_threshold_crossings",pd.Series("[]",index=frame.index)).map(lambda value: bool(_json(value,list)) if isinstance(value,(str,list)) else False)
    result.update({"by_forecast_target":dict(sorted(trigger_counts.items())),"by_recommendation_rule":dict(sorted(reason_counts.items())),
        "actionable_on_lifecycle_first_row":int((fp&first).sum()),"possible_stale_or_carried_forward_rows":int((fp&~crossings).sum())})
    return result

def report_csv_rows(report:dict[str,Any])->pd.DataFrame:
    rows=[]
    for target,detail in report.get("per_probability_target",{}).items():
        for mode in ("model_only_raw","model_only_reconciled","strict_operational"):
            rows.append({"scope":"probability_target_row","target":target,"mode":mode,**detail[mode]})
        for mode in ("event_level_model_only","event_level_strict_operational"):
            rows.append({"scope":"probability_target_event","target":target,"mode":mode,**{k:v for k,v in detail[mode].items() if k!="episodes"}})
        for mode in ("false_alerts_model_only","false_alerts_strict_operational"):
            rows.append({"scope":"probability_target_false_alert","target":target,"mode":mode,**detail[mode]})
    for layer,detail in report.get("status_layers",{}).items(): rows.append({"scope":"status_layer","target":layer,"mode":"classification",**detail})
    return pd.DataFrame(rows)

def evaluate(frame: pd.DataFrame, actual: pd.Series, maximum_gap_minutes: float = 10.0) -> dict[str, Any]:
    predicted=frame.status.astype(str).str.upper(); normal=actual.eq("NORMAL"); unsafe=actual.isin(["WARNING","CRITICAL"])
    trusted, available, failures=trusted_evidence(frame)
    reason_counts: dict[str,int]={}
    for value in frame.get("withholding_reasons_json",pd.Series("[]",index=frame.index)):
        try:
            for reason in _json(value,list): reason_counts[str(reason)]=reason_counts.get(str(reason),0)+1
        except Exception: reason_counts["malformed_withholding_reasons"] = reason_counts.get("malformed_withholding_reasons",0)+1
    lifecycle=frame.get("lifecycle_id",pd.Series("default",index=frame.index)).fillna("unknown")
    lifecycle_coverage={str(name):{"rows":int(len(group)),"trusted_rows":int(trusted.loc[group.index].sum()),"trusted_coverage":float(trusted.loc[group.index].mean())}
        for name,group in frame.groupby(lifecycle)}
    fp_count=int((normal & predicted.isin(["WARNING","CRITICAL"])).sum()); fn_count=int((unsafe & predicted.eq("NORMAL")).sum())
    _,_,_,timestamp_counts=timestamp_validation(frame,maximum_gap_minutes)
    warning_horizons=horizon_metrics(frame,actual,trusted,"WARNING",maximum_gap_minutes=maximum_gap_minutes)
    critical_horizons=horizon_metrics(frame,actual,trusted,"CRITICAL",maximum_gap_minutes=maximum_gap_minutes)
    combined_horizons=horizon_metrics(frame,actual,trusted,"UNSAFE",maximum_gap_minutes=maximum_gap_minutes)
    timestamp_counts["horizon_ineligible_due_to_timestamp_errors"]=sum(1 for collection in (warning_horizons,critical_horizons,combined_horizons) for metric in collection.values() for episode in metric["episodes"] if episode["horizon_ineligibility_reason"] in {"missing_timestamp","invalid_timestamp","duplicate_timestamp","non_monotonic_timestamp"})
    targets=per_target_metrics(frame,actual,trusted,maximum_gap_minutes)
    actionable=frame.get("recommendation_actionable",pd.Series(False,index=frame.index)).map(_safe_bool)
    withheld=frame.get("forecast_withheld",pd.Series(True,index=frame.index)).map(lambda x:_safe_bool(x,True))
    return {"policy_version":POLICY_VERSION,"metric_definitions":{"actual_label_source":"caller-selected external source_label/health_status or explicit --all-normal assertion","predicted_status_column":"status",
        "denominators":"FP uses actual NORMAL; FN uses actual WARNING/CRITICAL; per-target labels use a future same-lifecycle event onset within elapsed time",
        "model_only":"raw/reconciled model probability is evaluated whenever numeric; operational withholding is not applied",
        "strict_operational":"missing, withheld, untrusted, unloaded, or ineligible target produces no alert and therefore counts as a miss on positive rows",
        "horizon_match":"earliest trusted actionable ML row in [onset-H,onset), same lifecycle and continuous segment",
        "horizon_eligibility":"continuous labelled history from segment start to onset is at least H","false_alert_episode":"contiguous actionable rows during actual NORMAL, split by lifecycle/gap/non-actionable row",
        "event_onset":"WARNING is NORMAL->WARNING; CRITICAL is NORMAL/WARNING->CRITICAL; alert at or after onset receives no advance-warning credit","withheld_predictions":"reported separately and treated as negative in strict operational target metrics"},
        "status_layers":status_layer_metrics(frame,actual),
        "current_status":{"current_status_fp_count":fp_count,"current_status_fp_denominator":int(normal.sum()),"current_status_fp_rate":(fp_count/int(normal.sum()) if normal.any() else None),
        "current_status_fn_count":fn_count,"current_status_fn_denominator":int(unsafe.sum()),"current_status_fn_rate":(fn_count/int(unsafe.sum()) if unsafe.any() else None)},
        "trusted_coverage":{"trusted_rows":int(trusted.sum()),"untrusted_rows":int((~trusted).sum()),"trusted_forecast_coverage":(float(trusted.mean()) if available else None),"trusted_coverage_available":available,
            "trusted_evidence_failure_counts":failures,"trusted_evidence_failure_reasons":sorted(failures)},
        "false_alerts":false_alert_metrics(frame,actual,trusted,maximum_gap_minutes),
        "warning_onset_by_horizon":warning_horizons,
        "critical_onset_by_horizon":critical_horizons,
        "combined_unsafe_onset_by_horizon":combined_horizons,
        "coverage_by_withholding_reason":dict(sorted(reason_counts.items())),"coverage_by_lifecycle":lifecycle_coverage,
        "coverage_by_target_and_horizon":target_horizon_coverage(frame,trusted),"per_probability_target":targets,
        "operational_metrics":{"rows":len(frame),"forecast_availability_rate":float(pd.concat([pd.to_numeric(frame.get(t,pd.Series(np.nan,index=frame.index)),errors="coerce").notna() for t in CONTRACT.probability_targets],axis=1).any(axis=1).mean()),
            "withheld_rows":int(withheld.sum()),"withheld_rate":float(withheld.mean()),"actionable_rows":int(actionable.sum()),"actionable_rate":float(actionable.mean()),
            "ood_withholding_rows":int((withheld&frame.get("ood_detected",pd.Series(False,index=frame.index)).map(_safe_bool)).sum()),
            "disagreement_withholding_rows":int((withheld&frame.get("disagreement_detected",pd.Series(False,index=frame.index)).map(_safe_bool)).sum()),
            "reset_suppression_rows":int(frame.get("withheld_due_to_reset_suppression",pd.Series(False,index=frame.index)).map(_safe_bool).sum()),
            "recommendation_source_distribution":{str(k):int(v) for k,v in frame.get("recommendation_source",pd.Series("missing",index=frame.index)).value_counts().items()}},
        "fp_root_cause_breakdown":fp_root_cause_breakdown(frame,actual),"timestamp_and_continuity_failures":timestamp_counts}

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--replay",required=True); p.add_argument("--label-column"); p.add_argument("--all-normal",action="store_true")
    p.add_argument("--labels-file",help="Separate source CSV containing ground truth; row count and timestamps must align exactly")
    p.add_argument("--labels-column",help="Ground-truth column in --labels-file (default: source_label/health_status auto-detection)")
    p.add_argument("--maximum-gap-minutes",type=float,default=10.0); p.add_argument("--json-output"); p.add_argument("--csv-output"); p.add_argument("--fail-on-malformed-audit",action="store_true"); args=p.parse_args()
    if args.all_normal and args.labels_file: p.error("--all-normal and --labels-file are mutually exclusive")
    frame=pd.read_csv(Path(args.replay), low_memory=False)
    alignment={"method":"same_replay_row","row_count_match":True,"timestamp_alignment":"not_separately_checked"}
    if args.all_normal:
        selected_label="all_rows_asserted_normal"; actual=pd.Series("NORMAL",index=frame.index)
    elif args.labels_file:
        labels_frame=pd.read_csv(Path(args.labels_file),low_memory=False)
        selected_column=label_column(labels_frame,args.labels_column); selected_label=f"{args.labels_file}:{selected_column}"
        if len(labels_frame)!=len(frame): raise ValueError(f"label/replay row-count mismatch: {len(labels_frame)} != {len(frame)}")
        replay_times=pd.to_datetime(frame.get("timestamp"),errors="coerce"); label_times=pd.to_datetime(labels_frame.get("timestamp"),errors="coerce")
        if replay_times.isna().any() or label_times.isna().any() or not replay_times.reset_index(drop=True).equals(label_times.reset_index(drop=True)):
            raise ValueError("label/replay timestamp alignment failed")
        actual=labels_frame[selected_column].astype(str).str.upper().reset_index(drop=True)
        alignment={"method":"separate_labels_file_exact_row_and_timestamp_join","row_count_match":True,"timestamp_alignment":"exact","labels_file":str(Path(args.labels_file)),"labels_column":selected_column}
    else:
        selected_label=label_column(frame,args.label_column); actual=frame[selected_label].astype(str).str.upper()
    report=evaluate(frame,actual,args.maximum_gap_minutes)
    report["evaluation_input"]={"replay":str(Path(args.replay)),"actual_label_source":selected_label,"predicted_status_column":"status","row_count":len(frame),"invalid_actual_label_rows":int((~actual.isin(STATUSES)).sum()),"withheld_predictions_count_as_operational_negative":True,"alignment":alignment}
    for title,key in (("Current-status classification","current_status"),("Predictive trusted coverage","trusted_coverage"),("Predictive false-alert rows and episodes","false_alerts"),
                      ("WARNING onset prediction by horizon","warning_onset_by_horizon"),("CRITICAL onset prediction by horizon","critical_onset_by_horizon"),
                      ("Combined unsafe onset prediction by horizon","combined_unsafe_onset_by_horizon")):
        print(f"\n{title}\n"+json.dumps(report[key],indent=2,sort_keys=True))
    print("\nAudit evidence failures\n"+json.dumps(report["trusted_coverage"]["trusted_evidence_failure_counts"],indent=2,sort_keys=True))
    print("\nCoverage by withholding reason\n"+json.dumps(report["coverage_by_withholding_reason"],indent=2,sort_keys=True))
    print("\nCoverage by lifecycle\n"+json.dumps(report["coverage_by_lifecycle"],indent=2,sort_keys=True))
    print("\nCoverage by target and horizon\n"+json.dumps(report["coverage_by_target_and_horizon"],indent=2,sort_keys=True))
    print("\nTimestamp and continuity failures\n"+json.dumps(report["timestamp_and_continuity_failures"],indent=2,sort_keys=True))
    if args.json_output: Path(args.json_output).write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    if args.csv_output: report_csv_rows(report).to_csv(Path(args.csv_output),index=False)
    return 2 if args.fail_on_malformed_audit and report["trusted_coverage"]["trusted_evidence_failure_counts"].get("malformed_audit_row_count",0) else 0
if __name__=="__main__": raise SystemExit(main())
