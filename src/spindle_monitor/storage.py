from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .lifecycle_store import lifecycle_to_row
from .models import LifecycleRecord, MonitorResult, ValidationResult
from .anomaly import invalid_anomaly_result
from .forecast_policy import primary_reason
from .policy_contract import FORECAST_POLICY_VERSION, FORECAST_POLICY_CONTRACT_VERSION, URGENCY_POLICY_VERSION, URGENCY_POLICY_PARAMETERS


def _rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)

def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))



def _best_critical_forecast(result: MonitorResult):
    """Backward-compatible access to the earliest per-sensor trend forecast."""
    available = []
    for assessment in result.assessments.values():
        forecast = assessment.forecast
        if forecast is not None and forecast.critical_eta_seconds is not None:
            available.append((forecast.critical_eta_seconds, assessment, forecast))
    if not available:
        return None
    _, assessment, forecast = min(available, key=lambda item: item[0])
    return assessment, forecast


def _best_forecast(result: MonitorResult):
    available = []
    for assessment in result.assessments.values():
        forecast = assessment.forecast
        if forecast is not None and forecast.eta_seconds is not None:
            available.append((forecast.eta_seconds, assessment, forecast))
    if available:
        _, assessment, forecast = min(available, key=lambda item: item[0])
        return assessment, forecast
    assessment = result.assessments[result.worst_sensor]
    return assessment, assessment.forecast

def result_to_flat_row(
    result: MonitorResult,
    source_label: str | None = None,
) -> dict[str, Any]:
    assessments = result.assessments
    warning_statistical = result.statistical_warning_forecast
    critical_statistical = result.statistical_critical_forecast
    ml = result.ml_forecast
    prognostic = result.prognostic_forecast
    maturity_prefixes = [
        name.removesuffix("__coverage_fraction")
        for name in result.features
        if name.endswith("__w1440m__coverage_fraction")
    ]
    coverage_24h = [result.features[f"{prefix}__coverage_fraction"] for prefix in maturity_prefixes]
    mature_24h = [result.features[f"{prefix}__window_mature"] for prefix in maturity_prefixes]
    samples_24h = [result.features[f"{prefix}__sample_count"] for prefix in maturity_prefixes]
    gaps_24h = [result.features[f"{prefix}__gap_or_discontinuity"] for prefix in maturity_prefixes]
    return {
        "timestamp": result.timestamp.isoformat(),
        "row_provenance": "interpolated" if result.interpolated else "original",
        "interpolated": result.interpolated,
        "data_domain": result.data_domain,
        "dataset_hash": result.dataset_hash,
        "generator_version": result.generator_version,
        "model_stage": result.model_stage,
        "model_version": result.model_version,
        "vibration_mps2": _rounded(assessments["vibration_mps2"].raw_value),
        "temperature_c": _rounded(assessments["temperature_c"].raw_value),
        "current_ampere": _rounded(assessments["current_ampere"].raw_value),
        "status": result.raw_status.name,
        "raw_status": result.raw_status.name,
        "stabilized_status": result.effective_status.name,
        "event_status": result.event_status.name,
        "status_reason": " | ".join(result.reasons),
        "worst_sensor": result.worst_sensor,
        "lifecycle_id": result.lifecycle_id,
        "lifecycle_state": result.lifecycle_state,
        "lifecycle_censored": result.lifecycle_state.startswith("open_") or result.lifecycle_state == "censored",
        "elapsed_lifecycle_hours": _rounded(result.elapsed_lifecycle_hours, 4),
        "statistical_time_to_warning_hours": _rounded(
            warning_statistical.estimated_hours if warning_statistical else None, 3
        ),
        "statistical_warning_sensor": (
            warning_statistical.forecast_sensor if warning_statistical else None
        ),
        "prognostic_time_to_warning_hours": _rounded(prognostic.time_to_warning_hours, 3),
        "prognostic_warning_earliest_hours": _rounded(prognostic.warning_earliest_hours, 3),
        "prognostic_warning_latest_hours": _rounded(prognostic.warning_latest_hours, 3),
        "prognostic_warning_sensor": prognostic.warning_forecast_sensor,
        "ml_time_to_warning_hours": _rounded(ml.time_to_warning_hours, 3),
        "prediction_warning_hours_raw": _rounded(prognostic.time_to_warning_hours, 3),
        "prediction_warning_target": "first_confirmed_warning_timestamp",
        "statistical_time_to_critical_hours": _rounded(
            critical_statistical.estimated_hours if critical_statistical else None, 3
        ),
        "statistical_critical_sensor": (
            critical_statistical.forecast_sensor if critical_statistical else None
        ),
        "prognostic_time_to_critical_hours": _rounded(prognostic.time_to_critical_hours, 3),
        "prognostic_critical_earliest_hours": _rounded(prognostic.critical_earliest_hours, 3),
        "prognostic_critical_latest_hours": _rounded(prognostic.critical_latest_hours, 3),
        "prognostic_critical_sensor": prognostic.critical_forecast_sensor,
        "ml_time_to_critical_hours": _rounded(ml.time_to_critical_hours, 3),
        "prediction_critical_hours_raw": _rounded(prognostic.time_to_critical_hours, 3),
        "prediction_critical_target": "first_confirmed_critical_timestamp",
        "warning_absolute_disagreement_hours": _rounded(
            result.warning_absolute_disagreement_hours, 3
        ),
        "warning_relative_disagreement": _rounded(
            result.warning_relative_disagreement, 4
        ),
        "critical_absolute_disagreement_hours": _rounded(
            result.critical_absolute_disagreement_hours, 3
        ),
        "critical_relative_disagreement": _rounded(
            result.critical_relative_disagreement, 4
        ),
        "disagreement_trigger_targets": "|".join(result.disagreement_trigger_targets),
        "ml_outside_training_distribution": result.ml_outside_training_distribution,
        "ml_outside_feature_fraction": _rounded(result.ml_outside_feature_fraction, 4),
        "probability_warning_6h": _rounded(prognostic.probability_warning.get(6), 4),
        "probability_warning_12h": _rounded(prognostic.probability_warning.get(12), 4),
        "probability_warning_24h": _rounded(prognostic.probability_warning.get(24), 4),
        "probability_critical_6h": _rounded(prognostic.probability_critical.get(6), 4),
        "probability_critical_12h": _rounded(prognostic.probability_critical.get(12), 4),
        "probability_critical_24h": _rounded(prognostic.probability_critical.get(24), 4),
        "ml_probability_warning_6h": _rounded(ml.probability_warning.get(6), 4),
        "ml_probability_warning_12h": _rounded(ml.probability_warning.get(12), 4),
        "ml_probability_warning_24h": _rounded(ml.probability_warning.get(24), 4),
        "ml_probability_critical_6h": _rounded(ml.probability_critical.get(6), 4),
        "ml_probability_critical_12h": _rounded(ml.probability_critical.get(12), 4),
        "ml_probability_critical_24h": _rounded(ml.probability_critical.get(24), 4),
        "raw_probability_warning_6h": _rounded(ml.raw_probability_warning.get(6), 6),
        "raw_probability_warning_12h": _rounded(ml.raw_probability_warning.get(12), 6),
        "raw_probability_warning_24h": _rounded(ml.raw_probability_warning.get(24), 6),
        "raw_probability_critical_6h": _rounded(ml.raw_probability_critical.get(6), 6),
        "raw_probability_critical_12h": _rounded(ml.raw_probability_critical.get(12), 6),
        "raw_probability_critical_24h": _rounded(ml.raw_probability_critical.get(24), 6),
        "reconciled_probability_warning_6h": _rounded(ml.reconciled_probability_warning.get(6), 6),
        "reconciled_probability_warning_12h": _rounded(ml.reconciled_probability_warning.get(12), 6),
        "reconciled_probability_warning_24h": _rounded(ml.reconciled_probability_warning.get(24), 6),
        "reconciled_probability_critical_6h": _rounded(ml.reconciled_probability_critical.get(6), 6),
        "reconciled_probability_critical_12h": _rounded(ml.reconciled_probability_critical.get(12), 6),
        "reconciled_probability_critical_24h": _rounded(ml.reconciled_probability_critical.get(24), 6),
        "probability_reconciled": ml.probability_reconciled,
        "probability_reconciliation_targets": _stable_json(list(ml.probability_reconciliation_targets)),
        "probability_reconciliation_reason": ml.probability_reconciliation_reason,
        "primary_time_to_warning_hours": _rounded(result.final_time_to_warning_hours, 3),
        "primary_time_to_critical_hours": _rounded(result.final_time_to_critical_hours, 3),
        "conservative_alert_time_to_warning_hours": _rounded(
            result.conservative_alert_time_to_warning_hours, 3
        ),
        "conservative_alert_time_to_critical_hours": _rounded(
            result.conservative_alert_time_to_critical_hours, 3
        ),
        "warning_primary_source": result.warning_primary_source,
        "critical_primary_source": result.critical_primary_source,
        "final_time_to_warning_hours": _rounded(result.final_time_to_warning_hours, 3),
        "final_time_to_critical_hours": _rounded(result.final_time_to_critical_hours, 3),
        "forecast_confidence": result.forecast_confidence,
        "prediction_confidence": prognostic.confidence,
        "prognostic_confidence": prognostic.confidence,
        "prognostic_method_version": prognostic.method_version,
        "prognostic_history_hours": _rounded(prognostic.history_hours, 3),
        "prognostic_baseline_ready": prognostic.baseline_ready,
        "health_deviation_score": _rounded(prognostic.health_deviation_score, 4),
        "prognostic_selected_probability_thresholds": _stable_json(prognostic.selected_thresholds),
        "prognostic_probability_threshold_crossings": _stable_json(list(prognostic.probability_threshold_crossings)),
        "prognostic_withheld": prognostic.withheld,
        "prognostic_withholding_reasons": _stable_json(list(prognostic.withholding_reasons)),
        "feature_available": result.feature_available,
        "available_history_duration_seconds": _rounded(result.available_history_duration_seconds, 3),
        "window_24h_coverage_fraction": _rounded(min(coverage_24h), 4) if coverage_24h else None,
        "window_24h_mature": bool(mature_24h and all(value >= 0.5 for value in mature_24h)),
        "window_24h_sample_count": int(min(samples_24h)) if samples_24h else 0,
        "window_24h_gap_or_discontinuity": bool(any(value >= 0.5 for value in gaps_24h)),
        "source_sampling_interval_seconds": _rounded(result.source_sampling_interval_seconds, 3),
        "effective_resampling_interval_seconds": _rounded(result.effective_resampling_interval_seconds, 3),
        "sampling_interval_out_of_distribution": ml.sampling_interval_out_of_distribution,
        "target_support_violation": result.target_support_violation,
        "training_support_warning": (
            "outside_training_target_or_duration_support" if result.target_support_violation else ""
        ),
        "physically_invalid_prediction": any(ml.physically_invalid_targets.values()),
        "guardrail_result": result.guardrail_result,
        "refusal_or_fallback_reason": result.refusal_or_fallback_reason,
        "forecast_withheld": result.forecast_withheld,
        "withholding_reasons": "|".join(result.withholding_reasons),
        "forecast_policy_version": FORECAST_POLICY_VERSION,
        "model_maturity_stage": result.model_stage,
        "model_data_domain": result.data_domain,
        "forecast_method": prognostic.method_version,
        "production_eligible": ml.production_eligible,
        "production_policy_passed": ml.production_policy_passed,
        "production_policy_reasons": _stable_json(list(ml.production_policy_reasons)),
        "production_policy_reasons_json": _stable_json(list(ml.production_policy_reasons)),
        "required_model_targets": _stable_json(list(ml.required_model_targets)),
        "configured_mandatory_targets": _stable_json(list(ml.configured_mandatory_targets)),
        "metadata_required_targets": _stable_json(list(ml.metadata_required_targets)),
        "effective_required_targets": _stable_json(list(ml.effective_required_targets)),
        "mandatory_targets_missing_from_metadata": _stable_json(list(ml.mandatory_targets_missing_from_metadata)),
        "runtime_probability_fn_ceiling": ml.runtime_probability_fn_ceiling,
        "model_recorded_probability_fn_ceiling": _stable_json(ml.model_recorded_probability_fn_ceiling),
        "effective_probability_fn_ceiling": _stable_json(ml.effective_probability_fn_ceiling),
        "runtime_validation_fn_ceiling": ml.runtime_validation_fn_ceiling,
        "runtime_test_fn_ceiling": ml.runtime_test_fn_ceiling,
        "model_recorded_validation_fn_ceilings": _stable_json(ml.model_recorded_validation_fn_ceilings),
        "model_recorded_test_fn_ceilings": _stable_json(ml.model_recorded_test_fn_ceilings),
        "effective_validation_fn_ceilings": _stable_json(ml.effective_validation_fn_ceilings),
        "effective_test_fn_ceilings": _stable_json(ml.effective_test_fn_ceilings),
        "loaded_model_targets": _stable_json(list(ml.loaded_model_targets)),
        "missing_required_model_targets": _stable_json(list(ml.missing_required_model_targets)),
        "probability_threshold_schema_version": ml.probability_threshold_schema_version,
        "model_metadata_schema_version": ml.model_metadata_schema_version,
        "model_forecast_policy_contract_version": ml.model_forecast_policy_contract_version,
        "selected_probability_thresholds": _stable_json(ml.selected_thresholds),
        "probability_target_eligibility": _stable_json(ml.target_eligibility),
        "probability_target_ineligibility_reasons": _stable_json(ml.target_ineligibility_reasons),
        "physical_validation_passed": ml.physical_validation_passed,
        "physical_validation_reasons": _stable_json(list(ml.physical_validation_reasons)),
        "physical_validation_reasons_json": _stable_json(list(ml.physical_validation_reasons)),
        "feature_ready": result.feature_available and not ml.feature_readiness_reasons,
        "feature_readiness_reasons": _stable_json(list(ml.feature_readiness_reasons)),
        "feature_readiness_reasons_json": _stable_json(list(ml.feature_readiness_reasons)),
        "ood_detected": ml.outside_training_distribution,
        "target_support_valid": not result.target_support_violation,
        "disagreement_detected": bool(result.disagreement_trigger_targets),
        "recommendation_source": result.recommendation_source,
        "recommendation_actionable": result.recommendation_actionable,
        "recommendation_advisory": result.maintenance_urgency in {"PLAN_INSPECTION", "PLAN_MAINTENANCE", "MAINTENANCE_RECOMMENDED_SOON"},
        "recommendation_reason": result.forecast_reason,
        "recommendation_trigger_targets": _stable_json(list(result.recommendation_trigger_targets)),
        "probability_threshold_crossings": _stable_json(list(result.probability_threshold_crossings)),
        "recommendation_trigger_probability_targets": _stable_json(list(result.recommendation_trigger_probability_targets)),
        "recommendation_trigger_eta_targets": _stable_json(list(result.recommendation_trigger_eta_targets)),
        "recommendation_trigger_rules": _stable_json(list(result.recommendation_trigger_rules)),
        "urgency_policy_version": URGENCY_POLICY_VERSION,
        "urgency_policy_parameters": _stable_json(URGENCY_POLICY_PARAMETERS),
        "forecast_policy_contract_version": FORECAST_POLICY_CONTRACT_VERSION,
        "eta_target_evidence": _stable_json(ml.eta_target_evidence),
        "model_revoked": ml.revoked,
        "model_superseded": ml.superseded,
        "model_corrupted": ml.corrupted,
        "model_metadata_valid": ml.metadata_valid,
        "model_artifacts_valid": ml.artifacts_valid,
        "model_schema_compatible": ml.schema_compatible,
        "model_feature_schema_compatible": ml.feature_schema_compatible,
        "model_load_failures": _stable_json(ml.model_load_failures),
        "model_load_failures_json": _stable_json(ml.model_load_failures),
        "probability_target_evidence": _stable_json(ml.probability_target_evidence),
        "withholding_reasons_json": _stable_json(list(result.withholding_reasons)),
        "primary_withholding_reason": primary_reason(result.withholding_reasons),
        "withheld_due_to_disagreement": "disagreement" in result.withholding_reasons,
        "withheld_due_to_missing_ml": any(
            "missing_ml" in reason or "missing_deployed_ml" in reason
            for reason in result.withholding_reasons
        ),
        "withheld_due_to_reset_suppression": (
            "reset_suppression" in result.withholding_reasons
        ),
        "withheld_due_to_distribution_check": (
            "distribution_check" in result.withholding_reasons
        ),
        "maintenance_urgency": result.maintenance_urgency,
        "forecast_reason": result.forecast_reason,
        "quality_status": result.anomaly.quality_status.value,
        "anomaly_type": "|".join(value.value for value in result.anomaly.anomaly_type),
        "anomaly_severity": result.anomaly.anomaly_severity,
        "anomaly_confidence": _rounded(result.anomaly.anomaly_confidence, 4),
        "suspected_origin": result.anomaly.suspected_origin,
        "affected_sensors": "|".join(result.anomaly.affected_sensors),
        "first_observed_timestamp": (
            result.anomaly.first_observed_timestamp.isoformat()
            if result.anomaly.first_observed_timestamp else ""
        ),
        "confirmed_timestamp": (
            result.anomaly.confirmed_timestamp.isoformat()
            if result.anomaly.confirmed_timestamp else ""
        ),
        "resolved_timestamp": (
            result.anomaly.resolved_timestamp.isoformat()
            if result.anomaly.resolved_timestamp else ""
        ),
        "anomaly_is_active": result.anomaly.is_active,
        "raw_safety_status": result.anomaly.raw_safety_status,
        "safety_action": result.anomaly.safety_action,
        "model_action": result.anomaly.model_action.value,
        "feature_history_action": result.anomaly.feature_history_action.value,
        "use_for_features": result.anomaly.use_for_features,
        "use_for_prediction": result.anomaly.use_for_prediction,
        "use_for_retraining": result.anomaly.use_for_retraining,
        "confidence_multiplier": _rounded(result.anomaly.confidence_multiplier, 4),
        "supporting_evidence_json": json.dumps(result.anomaly.supporting_evidence),
        "anomaly_configuration_snapshot_json": json.dumps(
            result.anomaly.configuration_snapshot, sort_keys=True
        ),
        "detector_version": result.anomaly.detector_version,
        "causal_decision": result.anomaly.causal_decision,
        "final_offline_classification": result.anomaly.final_offline_classification,
        "training_eligible": result.anomaly.training_eligible,
        "training_exclusion_reason": result.anomaly.training_exclusion_reason,
        "requires_human_review": result.anomaly.requires_human_review,
        "decision_timestamp": (
            result.anomaly.decision_timestamp.isoformat()
            if result.anomaly.decision_timestamp else ""
        ),
        "suspicion_timestamp": (
            result.anomaly.suspicion_timestamp.isoformat()
            if result.anomaly.suspicion_timestamp else ""
        ),
        "confirmation_source": result.anomaly.confirmation_source,
        "confirmation_evidence": result.anomaly.confirmation_evidence,
        "confirming_actor": result.anomaly.confirming_actor,
        "expected_sampling_interval_seconds": _rounded(
            result.anomaly.expected_sampling_interval_seconds, 3
        ),
        "anomaly_source_sampling_interval_seconds": _rounded(
            result.anomaly.source_sampling_interval_seconds, 3
        ),
        "estimated_missing_sample_count": result.anomaly.estimated_missing_sample_count,
        "gap_duration_seconds": _rounded(result.anomaly.gap_duration_seconds, 3),
        "interpolation_enabled": result.anomaly.interpolation_enabled,
        "interpolation_occurred": result.anomaly.interpolation_occurred,
        "confidence_penalty_reason": result.anomaly.confidence_penalty_reason,
        "configured_confidence_multiplier": _rounded(
            result.anomaly.configured_confidence_multiplier, 4
        ),
        "minimum_confidence_floor": _rounded(result.anomaly.minimum_confidence_floor, 4),
        "correlation_evidence_json": json.dumps(
            result.anomaly.correlation_evidence, sort_keys=True
        ),
    }


def result_to_detailed_row(
    result: MonitorResult,
    source_label: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = result_to_flat_row(result, source_label)
    row.update(
        {
            "source_label": source_label or "",
            "status_matches_label": (
                (source_label or "").strip().upper() == result.raw_status.name
                if source_label else ""
            ),
            "effective_status": result.effective_status.name,
            "maximum_severity": result.maximum_severity,
            "weighted_severity": result.weighted_severity,
            "estimated_degradation": result.estimated_degradation,
            "degradation_rate_per_hour": result.degradation_rate_per_hour,
            "reset_confidence": result.reset_confidence,
            "reset_reason": result.reset_reason,
            "ml_maturity_stage": result.ml_forecast.maturity_stage,
            "ml_model_version": result.ml_forecast.model_version or "",
            "prognostic_method_version": result.prognostic_forecast.method_version,
            "prognostic_confidence": result.prognostic_forecast.confidence,
            "prognostic_health_deviation_score": result.prognostic_forecast.health_deviation_score,
            "prognostic_baseline_ready": result.prognostic_forecast.baseline_ready,
            "prognostic_sensor_evidence_json": json.dumps(result.prognostic_forecast.sensor_evidence, sort_keys=True),
            "prognostic_per_sensor_eta_json": json.dumps(result.prognostic_forecast.per_sensor_eta, sort_keys=True),
            "prognostic_per_sensor_probability_warning_json": json.dumps(result.prognostic_forecast.per_sensor_probability_warning, sort_keys=True),
            "prognostic_per_sensor_probability_critical_json": json.dumps(result.prognostic_forecast.per_sensor_probability_critical, sort_keys=True),
            "data_domain": result.data_domain,
            "raw_status": result.raw_status.name,
            "stabilized_status": result.effective_status.name,
            "event_status": result.event_status.name,
            "lifecycle_censored": result.lifecycle_state.startswith("open_") or result.lifecycle_state == "censored",
            "sampling_gap_seconds": result.sampling_gap_seconds,
            "interpolated": result.interpolated,
            "feature_available": result.feature_available,
            "target_support_violation": result.target_support_violation,
            "sampling_interval_out_of_distribution": result.ml_forecast.sampling_interval_out_of_distribution,
            "prediction_confidence": result.ml_forecast.prediction_confidence,
            "training_target_max_hours": json.dumps(result.ml_forecast.training_target_max_hours, sort_keys=True),
            "training_target_min_hours": json.dumps(result.ml_forecast.training_target_min_hours, sort_keys=True),
            "target_support_violations": json.dumps(result.ml_forecast.target_support_violations, sort_keys=True),
            "physically_invalid_targets": json.dumps(result.ml_forecast.physically_invalid_targets, sort_keys=True),
        }
    )
    for target, forecast in (
        ("warning", result.statistical_warning_forecast),
        ("critical", result.statistical_critical_forecast),
    ):
        if forecast:
            row[f"statistical_{target}_forecast_sensor"] = forecast.forecast_sensor or ""
            row[f"statistical_{target}_forecast_reason"] = forecast.reason
            for key, value in forecast.per_sensor_hours.items():
                row[f"{key}__statistical_{target}_hours"] = value

    for key, assessment in result.assessments.items():
        prefix = key
        row.update(
            {
                f"{prefix}__raw": assessment.raw_value,
                f"{prefix}__rolling_median": assessment.rolling_median,
                f"{prefix}__fast_ewma": assessment.fast_ewma,
                f"{prefix}__slow_ewma": assessment.slow_ewma,
                f"{prefix}__kalman_level": assessment.kalman_level,
                f"{prefix}__kalman_rate_per_hour": assessment.kalman_rate_per_hour,
                f"{prefix}__raw_status": assessment.raw_status.name,
                f"{prefix}__smoothed_status": assessment.smoothed_status.name,
                f"{prefix}__severity": assessment.severity,
                f"{prefix}__critical_margin_percent": assessment.critical_margin_percent,
            }
        )
        forecast = assessment.forecast
        if forecast:
            row.update(
                {
                    f"{prefix}__trend_forecast_target": forecast.target_status,
                    f"{prefix}__trend_forecast_eta_hours": (
                        None if forecast.eta_seconds is None else forecast.eta_seconds / 3600.0
                    ),
                    f"{prefix}__trend_critical_eta_hours": (
                        None if forecast.critical_eta_seconds is None else forecast.critical_eta_seconds / 3600.0
                    ),
                    f"{prefix}__trend_r_squared": forecast.r_squared,
                    f"{prefix}__trend_slope_per_hour": forecast.slope_per_hour,
                    f"{prefix}__trend_confidence": forecast.confidence,
                    f"{prefix}__trend_reason": forecast.explanation,
                }
            )
    row.update({f"feature__{key}": value for key, value in result.features.items()})
    return row


class CSVResultStore:
    def __init__(self, path: str | Path, *, detailed: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.file = self.path.open("w", encoding="utf-8-sig", newline="")
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot overwrite {self.path}. Close it in Excel or another program, "
                "or choose another filename."
            ) from exc
        self.detailed = detailed
        self.writer: csv.DictWriter | None = None
        self.row_count = 0

    def save(self, result: MonitorResult, source_label: str | None = None) -> None:
        row = (
            result_to_detailed_row(result, source_label)
            if self.detailed else result_to_flat_row(result, source_label)
        )
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.row_count += 1
        if self.row_count % 100 == 0:
            self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()


class InvalidCSVStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.file = self.path.open("w", encoding="utf-8-sig", newline="")
        except PermissionError as exc:
            raise PermissionError(f"Cannot overwrite {self.path}; close it in Excel.") from exc
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "row_number", "validation_reason", "timestamp", "vibration_mps2",
                "temperature_c", "current_ampere", "health_status",
                "quality_status", "anomaly_type", "model_action",
                "training_eligible", "training_exclusion_reason",
                "requires_human_review", "supporting_evidence_json",
            ],
        )
        self.writer.writeheader()
        self.row_count = 0

    def save(self, validation: ValidationResult) -> None:
        row = validation.raw_row or {}
        anomaly = invalid_anomaly_result(validation)
        self.writer.writerow(
            {
                "row_number": validation.row_number,
                "validation_reason": validation.reason,
                "timestamp": row.get("timestamp", ""),
                "vibration_mps2": row.get("vibration_mps2", ""),
                "temperature_c": row.get("temperature_c", ""),
                "current_ampere": row.get("current_ampere", ""),
                "health_status": row.get("health_status", ""),
                "quality_status": anomaly.quality_status.value,
                "anomaly_type": "|".join(value.value for value in anomaly.anomaly_type),
                "model_action": anomaly.model_action.value,
                "training_eligible": anomaly.training_eligible,
                "training_exclusion_reason": anomaly.training_exclusion_reason,
                "requires_human_review": anomaly.requires_human_review,
                "supporting_evidence_json": json.dumps(anomaly.supporting_evidence),
            }
        )
        self.row_count += 1

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


class SQLiteResultStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.pending = 0
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_label TEXT,
                status TEXT NOT NULL,
                lifecycle_id TEXT NOT NULL,
                lifecycle_state TEXT NOT NULL,
                final_forecast_hours REAL,
                forecast_confidence TEXT,
                details_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON readings(timestamp);
            CREATE INDEX IF NOT EXISTS idx_readings_lifecycle ON readings(lifecycle_id);
            CREATE TABLE IF NOT EXISTS lifecycles (
                lifecycle_id TEXT PRIMARY KEY,
                start_timestamp TEXT NOT NULL,
                end_timestamp TEXT NOT NULL,
                duration_hours REAL NOT NULL,
                highest_status TEXT NOT NULL,
                critical_reached INTEGER NOT NULL,
                reset_confidence TEXT NOT NULL,
                reset_reason TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_versions (
                model_version TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                training_timestamp TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                target_name TEXT NOT NULL,
                stage TEXT NOT NULL CHECK(stage IN ('candidate', 'production', 'archived')),
                promotion_timestamp TEXT,
                training_lifecycle_ids_json TEXT NOT NULL,
                validation_lifecycle_ids_json TEXT NOT NULL,
                test_lifecycle_ids_json TEXT NOT NULL,
                validation_metrics_json TEXT,
                test_metrics_json TEXT,
                baseline_metrics_json TEXT,
                feature_schema_json TEXT NOT NULL,
                synthetic_data_only INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_model_records_target_stage
                ON model_records(target_name, stage);
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                target TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS anomaly_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                row_number INTEGER,
                quality_status TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                raw_safety_status TEXT NOT NULL,
                model_action TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                affected_sensors TEXT,
                training_eligible INTEGER NOT NULL,
                details_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_anomaly_events_timestamp ON anomaly_events(timestamp);
            CREATE TABLE IF NOT EXISTS anomaly_state (
                sensor TEXT PRIMARY KEY,
                current_anomaly_type TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                active INTEGER NOT NULL,
                first_observed_timestamp TEXT,
                latest_update_timestamp TEXT NOT NULL,
                resolution_timestamp TEXT,
                safety_action TEXT NOT NULL,
                model_action TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS anomaly_intervals (
                interval_id TEXT PRIMARY KEY,
                anomaly_type TEXT NOT NULL,
                affected_sensors TEXT NOT NULL,
                first_observed_timestamp TEXT NOT NULL,
                decision_or_confirmation_timestamp TEXT,
                end_or_resolution_timestamp TEXT,
                duration_seconds REAL NOT NULL,
                maximum_severity TEXT NOT NULL,
                maximum_severity_rank INTEGER NOT NULL,
                maximum_confidence REAL NOT NULL,
                safety_actions_json TEXT NOT NULL,
                model_actions_json TEXT NOT NULL,
                active INTEGER NOT NULL,
                final_offline_classification TEXT NOT NULL,
                event_row_count INTEGER NOT NULL,
                supporting_evidence_json TEXT NOT NULL,
                latest_update_timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_anomaly_intervals_active
                ON anomaly_intervals(active, first_observed_timestamp);
            """
        )
        self._ensure_columns("readings", {
            "data_domain": "TEXT NOT NULL DEFAULT 'accelerated_mock'",
            "raw_status": "TEXT", "stabilized_status": "TEXT", "event_status": "TEXT",
            "lifecycle_censored": "INTEGER NOT NULL DEFAULT 0",
            "sampling_gap_seconds": "REAL", "interpolated": "INTEGER NOT NULL DEFAULT 0",
            "feature_available": "INTEGER NOT NULL DEFAULT 1",
            "target_support_violation": "INTEGER NOT NULL DEFAULT 0",
            "feature_ood": "INTEGER NOT NULL DEFAULT 0", "sampling_ood": "INTEGER NOT NULL DEFAULT 0",
            "primary_forecast": "REAL", "conservative_alert": "REAL",
            "withholding_reason": "TEXT", "model_version": "TEXT", "target_source": "TEXT",
            "dataset_hash": "TEXT", "generator_version": "TEXT", "model_stage": "TEXT",
            "source_sampling_interval_seconds": "REAL",
            "effective_resampling_interval_seconds": "REAL",
            "available_history_duration_seconds": "REAL",
            "guardrail_result": "TEXT", "refusal_or_fallback_reason": "TEXT",
            "quality_status": "TEXT", "anomaly_type": "TEXT", "anomaly_severity": "TEXT",
            "anomaly_confidence": "REAL", "suspected_origin": "TEXT",
            "affected_sensors": "TEXT", "safety_action": "TEXT", "model_action": "TEXT",
            "confidence_multiplier": "REAL", "training_eligible": "INTEGER NOT NULL DEFAULT 1",
            "training_exclusion_reason": "TEXT", "requires_human_review": "INTEGER NOT NULL DEFAULT 0",
            "feature_history_action": "TEXT",
            "estimated_missing_sample_count": "INTEGER NOT NULL DEFAULT 0",
            "gap_duration_seconds": "REAL",
        })
        self._ensure_columns("model_records", {
            "data_domain": "TEXT", "model_stage": "TEXT", "dataset_hash": "TEXT",
            "generator_version": "TEXT", "production_eligible": "INTEGER NOT NULL DEFAULT 0",
        })
        self.connection.commit()

    def _ensure_columns(self, table: str, definitions: dict[str, str]) -> None:
        existing = {row[1] for row in self.connection.execute(f"PRAGMA table_info({table})")}
        for name, definition in definitions.items():
            if name not in existing:
                self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def clear_monitoring_data(self) -> None:
        """Start a clean replay without deleting model/training audit history."""
        self.connection.execute("DELETE FROM readings")
        self.connection.execute("DELETE FROM lifecycles")
        self.connection.execute("DELETE FROM anomaly_events")
        self.connection.execute("DELETE FROM anomaly_state")
        self.connection.execute("DELETE FROM anomaly_intervals")
        self.connection.commit()

    def save(self, result: MonitorResult, source_label: str | None = None) -> None:
        self.connection.execute(
            """
            INSERT INTO readings (
                timestamp, source_label, status, lifecycle_id, lifecycle_state,
                final_forecast_hours, forecast_confidence, details_json,
                data_domain, raw_status, stabilized_status, event_status,
                lifecycle_censored, sampling_gap_seconds, interpolated, feature_available,
                target_support_violation, feature_ood, sampling_ood, primary_forecast,
                conservative_alert, withholding_reason, model_version, target_source
                , dataset_hash, generator_version, model_stage,
                source_sampling_interval_seconds, effective_resampling_interval_seconds,
                available_history_duration_seconds, guardrail_result, refusal_or_fallback_reason
                , quality_status, anomaly_type, anomaly_severity, anomaly_confidence,
                suspected_origin, affected_sensors, safety_action, model_action,
                confidence_multiplier, training_eligible, training_exclusion_reason,
                requires_human_review, feature_history_action,
                estimated_missing_sample_count, gap_duration_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.timestamp.isoformat(), source_label, result.raw_status.name,
                result.lifecycle_id, result.lifecycle_state, result.final_forecast_hours,
                result.forecast_confidence,
                json.dumps(
                    {
                        **result_to_flat_row(result, source_label),
                        "effective_status": result.effective_status.name,
                        "reset_confidence": result.reset_confidence,
                        "reset_reason": result.reset_reason,
                        "sensor_diagnostics": {
                            key: {
                                "rolling_median": assessment.rolling_median,
                                "fast_ewma": assessment.fast_ewma,
                                "slow_ewma": assessment.slow_ewma,
                                "kalman_level": assessment.kalman_level,
                                "kalman_rate_per_hour": assessment.kalman_rate_per_hour,
                            }
                            for key, assessment in result.assessments.items()
                        },
                    },
                    separators=(",", ":"), sort_keys=True,
                ),
                result.data_domain, result.raw_status.name, result.effective_status.name,
                result.event_status.name,
                int(result.lifecycle_state.startswith("open_") or result.lifecycle_state == "censored"),
                result.sampling_gap_seconds, int(result.interpolated), int(result.feature_available),
                int(result.target_support_violation), 0,
                0, result.final_forecast_hours,
                result.conservative_alert_time_to_critical_hours,
                "|".join(result.withholding_reasons), result.prognostic_forecast.method_version,
                result.critical_primary_source,
                result.dataset_hash, result.generator_version, result.model_stage,
                result.source_sampling_interval_seconds,
                result.effective_resampling_interval_seconds,
                result.available_history_duration_seconds,
                result.guardrail_result, result.refusal_or_fallback_reason,
                result.anomaly.quality_status.value,
                "|".join(value.value for value in result.anomaly.anomaly_type),
                result.anomaly.anomaly_severity, result.anomaly.anomaly_confidence,
                result.anomaly.suspected_origin, "|".join(result.anomaly.affected_sensors),
                result.anomaly.safety_action, result.anomaly.model_action.value,
                result.anomaly.confidence_multiplier, int(result.anomaly.training_eligible),
                result.anomaly.training_exclusion_reason,
                int(result.anomaly.requires_human_review),
                result.anomaly.feature_history_action.value,
                result.anomaly.estimated_missing_sample_count,
                result.anomaly.gap_duration_seconds,
            ),
        )
        if any(value.value != "NONE" for value in result.anomaly.anomaly_type):
            self._save_anomaly_event(result.anomaly, timestamp=result.timestamp.isoformat())
        self._maintain_anomaly_state_and_intervals(result)
        if result.completed_lifecycle:
            self.save_lifecycle(result.completed_lifecycle)
        self.pending += 1
        if self.pending >= 100:
            self.connection.commit()
            self.pending = 0

    def _save_anomaly_event(self, anomaly, *, timestamp: str | None, row_number: int | None = None) -> None:
        payload = anomaly.to_dict()
        self.connection.execute(
            """
            INSERT INTO anomaly_events (
                timestamp, row_number, quality_status, anomaly_type, raw_safety_status,
                model_action, is_active, affected_sensors, training_eligible, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp, row_number, anomaly.quality_status.value,
                "|".join(value.value for value in anomaly.anomaly_type),
                anomaly.raw_safety_status, anomaly.model_action.value, int(anomaly.is_active),
                "|".join(anomaly.affected_sensors), int(anomaly.training_eligible),
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
            ),
        )

    def save_invalid(self, validation: ValidationResult) -> None:
        anomaly = invalid_anomaly_result(validation)
        timestamp = (validation.raw_row or {}).get("timestamp") or None
        self._save_anomaly_event(anomaly, timestamp=timestamp, row_number=validation.row_number)
        self.pending += 1

    @staticmethod
    def _primary_anomaly_type(anomaly) -> str:
        auxiliary = {"NONE", "ORIGIN_UNCERTAIN", "SENSOR_DISAGREEMENT"}
        return next(
            (value.value for value in anomaly.anomaly_type if value.value not in auxiliary),
            next((value.value for value in anomaly.anomaly_type if value.value != "NONE"), "NONE"),
        )

    def _next_interval_id(self) -> str:
        row = self.connection.execute(
            "SELECT interval_id FROM anomaly_intervals ORDER BY interval_id DESC LIMIT 1"
        ).fetchone()
        number = int(row[0].rsplit("_", 1)[-1]) + 1 if row else 1
        return f"anomaly_interval_{number:08d}"

    def _resolve_interval(self, interval_id: str, timestamp: str, final: str | None = None) -> None:
        row = self.connection.execute(
            "SELECT first_observed_timestamp, final_offline_classification FROM anomaly_intervals WHERE interval_id = ?",
            (interval_id,),
        ).fetchone()
        if not row:
            return
        duration = max(0.0, (
            datetime.fromisoformat(timestamp) - datetime.fromisoformat(row[0])
        ).total_seconds())
        self.connection.execute(
            """
            UPDATE anomaly_intervals
            SET active = 0, end_or_resolution_timestamp = ?, duration_seconds = ?,
                final_offline_classification = ?, latest_update_timestamp = ?
            WHERE interval_id = ?
            """,
            (timestamp, duration, final or row[1], timestamp, interval_id),
        )

    def _maintain_anomaly_state_and_intervals(self, result: MonitorResult) -> None:
        anomaly = result.anomaly
        timestamp = result.timestamp.isoformat()
        affected = set(anomaly.affected_sensors)
        active_affected = affected if anomaly.is_active else set()
        anomaly_type = "|".join(value.value for value in anomaly.anomaly_type)

        # Current state is maintained on every reading, including normal rows.
        for sensor in result.assessments:
            existing = self.connection.execute(
                "SELECT active, first_observed_timestamp FROM anomaly_state WHERE sensor = ?",
                (sensor,),
            ).fetchone()
            if sensor in active_affected:
                first = (
                    anomaly.first_observed_timestamp.isoformat()
                    if anomaly.first_observed_timestamp else timestamp
                )
                payload = anomaly.to_dict()
                self.connection.execute(
                    """
                    INSERT INTO anomaly_state (
                        sensor, current_anomaly_type, quality_status, active,
                        first_observed_timestamp, latest_update_timestamp,
                        resolution_timestamp, safety_action, model_action, details_json
                    ) VALUES (?, ?, ?, 1, ?, ?, NULL, ?, ?, ?)
                    ON CONFLICT(sensor) DO UPDATE SET
                        current_anomaly_type=excluded.current_anomaly_type,
                        quality_status=excluded.quality_status, active=1,
                        first_observed_timestamp=excluded.first_observed_timestamp,
                        latest_update_timestamp=excluded.latest_update_timestamp,
                        resolution_timestamp=NULL, safety_action=excluded.safety_action,
                        model_action=excluded.model_action, details_json=excluded.details_json
                    """,
                    (
                        sensor, anomaly_type, anomaly.quality_status.value, first, timestamp,
                        anomaly.safety_action, anomaly.model_action.value,
                        json.dumps(payload, separators=(",", ":"), sort_keys=True),
                    ),
                )
            else:
                resolution = timestamp if existing and existing[0] else None
                self.connection.execute(
                    """
                    INSERT INTO anomaly_state (
                        sensor, current_anomaly_type, quality_status, active,
                        first_observed_timestamp, latest_update_timestamp,
                        resolution_timestamp, safety_action, model_action, details_json
                    ) VALUES (?, 'NONE', 'NORMAL', 0, NULL, ?, ?, ?, 'USE_NORMALLY', ?)
                    ON CONFLICT(sensor) DO UPDATE SET
                        current_anomaly_type='NONE', quality_status='NORMAL', active=0,
                        first_observed_timestamp=NULL,
                        latest_update_timestamp=excluded.latest_update_timestamp,
                        resolution_timestamp=COALESCE(excluded.resolution_timestamp, anomaly_state.resolution_timestamp),
                        safety_action=excluded.safety_action, model_action='USE_NORMALLY',
                        details_json=excluded.details_json
                    """,
                    (
                        sensor, timestamp, resolution, anomaly.safety_action,
                        json.dumps({
                            "sensor": sensor, "current_anomaly_type": "NONE", "active": False,
                            "latest_update_timestamp": timestamp,
                            "resolution_timestamp": resolution,
                        }, separators=(",", ":"), sort_keys=True),
                    ),
                )

        # Resolve intervals no longer represented by the current active decision.
        active_rows = list(self.connection.execute(
            "SELECT interval_id, anomaly_type, affected_sensors FROM anomaly_intervals WHERE active = 1"
        ))
        primary = self._primary_anomaly_type(anomaly)
        sensor_key = "|".join(sorted(affected))
        first = (
            anomaly.first_observed_timestamp.isoformat()
            if anomaly.first_observed_timestamp else timestamp
        )
        matching_interval = None
        for interval_id, stored_type, stored_sensors in active_rows:
            stored_set = set(filter(None, stored_sensors.split("|")))
            same_episode = (
                anomaly.is_active and stored_type == primary
                and stored_set == affected
            )
            resolution_episode = (
                not anomaly.is_active and stored_type == primary
                and stored_set == affected
            )
            if same_episode or resolution_episode:
                matching_interval = interval_id
            elif stored_set & (set(result.assessments) - active_affected):
                self._resolve_interval(interval_id, timestamp)

        if primary == "NONE" or not affected:
            return

        severity_rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        evidence = list(dict.fromkeys(anomaly.supporting_evidence))
        if anomaly.is_active and matching_interval:
            row = self.connection.execute(
                """
                SELECT maximum_severity_rank, maximum_confidence, safety_actions_json,
                       model_actions_json, event_row_count, supporting_evidence_json,
                       first_observed_timestamp
                FROM anomaly_intervals WHERE interval_id = ?
                """,
                (matching_interval,),
            ).fetchone()
            safety_actions = list(dict.fromkeys(json.loads(row[2]) + [anomaly.safety_action]))
            model_actions = list(dict.fromkeys(json.loads(row[3]) + [anomaly.model_action.value]))
            evidence = list(dict.fromkeys(json.loads(row[5]) + evidence))
            duration = max(0.0, (
                result.timestamp - datetime.fromisoformat(row[6])
            ).total_seconds())
            maximum_rank = max(row[0], severity_rank.get(anomaly.anomaly_severity, 0))
            rank_name = next(name for name, rank in severity_rank.items() if rank == maximum_rank)
            self.connection.execute(
                """
                UPDATE anomaly_intervals SET duration_seconds=?, maximum_severity=?,
                    maximum_severity_rank=?, maximum_confidence=?, safety_actions_json=?,
                    model_actions_json=?, event_row_count=?, supporting_evidence_json=?,
                    latest_update_timestamp=?
                WHERE interval_id=?
                """,
                (
                    duration, rank_name, maximum_rank, max(row[1], anomaly.anomaly_confidence),
                    json.dumps(safety_actions), json.dumps(model_actions), row[4] + 1,
                    json.dumps(evidence), timestamp, matching_interval,
                ),
            )
            return

        if not anomaly.is_active:
            # A resolution row closes the matching logical episode when available.
            candidate = ((matching_interval,) if matching_interval else self.connection.execute(
                """
                SELECT interval_id FROM anomaly_intervals
                WHERE anomaly_type=? AND affected_sensors=? AND active=1
                ORDER BY first_observed_timestamp DESC LIMIT 1
                """,
                (primary, sensor_key),
            ).fetchone())
            if candidate:
                existing = self.connection.execute(
                    """
                    SELECT safety_actions_json, model_actions_json,
                           supporting_evidence_json, maximum_severity_rank
                    FROM anomaly_intervals WHERE interval_id=?
                    """,
                    (candidate[0],),
                ).fetchone()
                safety_actions = list(dict.fromkeys(
                    json.loads(existing[0]) + [anomaly.safety_action]
                ))
                model_actions = list(dict.fromkeys(
                    json.loads(existing[1]) + [anomaly.model_action.value]
                ))
                evidence = list(dict.fromkeys(
                    json.loads(existing[2]) + list(anomaly.supporting_evidence)
                ))
                rank = max(existing[3], severity_rank.get(anomaly.anomaly_severity, 0))
                rank_name = next(name for name, value in severity_rank.items() if value == rank)
                self.connection.execute(
                    """
                    UPDATE anomaly_intervals SET event_row_count=event_row_count+1,
                        maximum_confidence=MAX(maximum_confidence, ?),
                        final_offline_classification=?, safety_actions_json=?,
                        model_actions_json=?, supporting_evidence_json=?,
                        maximum_severity=?, maximum_severity_rank=?
                    WHERE interval_id=?
                    """,
                    (
                        anomaly.anomaly_confidence, anomaly.final_offline_classification,
                        json.dumps(safety_actions), json.dumps(model_actions), json.dumps(evidence),
                        rank_name, rank, candidate[0],
                    ),
                )
                self._resolve_interval(candidate[0], timestamp, anomaly.final_offline_classification)
                return

        interval_id = self._next_interval_id()
        end = anomaly.resolved_timestamp.isoformat() if anomaly.resolved_timestamp else None
        duration = max(0.0, (
            (anomaly.resolved_timestamp or result.timestamp)
            - (anomaly.first_observed_timestamp or result.timestamp)
        ).total_seconds())
        rank = severity_rank.get(anomaly.anomaly_severity, 0)
        self.connection.execute(
            """
            INSERT INTO anomaly_intervals (
                interval_id, anomaly_type, affected_sensors, first_observed_timestamp,
                decision_or_confirmation_timestamp, end_or_resolution_timestamp,
                duration_seconds, maximum_severity, maximum_severity_rank,
                maximum_confidence, safety_actions_json, model_actions_json, active,
                final_offline_classification, event_row_count,
                supporting_evidence_json, latest_update_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                interval_id, primary, sensor_key, first,
                (anomaly.confirmed_timestamp or anomaly.suspicion_timestamp
                 or anomaly.decision_timestamp or result.timestamp).isoformat(),
                end, duration, anomaly.anomaly_severity, rank,
                anomaly.anomaly_confidence, json.dumps([anomaly.safety_action]),
                json.dumps([anomaly.model_action.value]), int(anomaly.is_active),
                anomaly.final_offline_classification, json.dumps(evidence), timestamp,
            ),
        )

    def save_lifecycle(self, record: LifecycleRecord) -> None:
        row = lifecycle_to_row(record)
        self.connection.execute(
            """
            INSERT OR REPLACE INTO lifecycles (
                lifecycle_id, start_timestamp, end_timestamp, duration_hours,
                highest_status, critical_reached, reset_confidence, reset_reason,
                details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.lifecycle_id, record.start_timestamp.isoformat(),
                record.end_timestamp.isoformat(), record.duration_hours,
                record.highest_status, int(record.critical_reached),
                record.reset_confidence, record.reset_reason,
                json.dumps(row, separators=(",", ":")),
            ),
        )

    def save_model_metadata(self, metadata: dict[str, Any], stage: str) -> None:
        version = str(metadata.get("model_version", "unknown"))
        training_timestamp = str(metadata.get("training_timestamp", ""))
        self.connection.execute(
            """
            INSERT OR REPLACE INTO model_versions (
                model_version, stage, training_timestamp, metadata_json
            ) VALUES (?, ?, ?, ?)
            """,
            (version, stage, training_timestamp, json.dumps(metadata, separators=(",", ":"))),
        )
        self.connection.commit()

    def archive_production_model_records(
        self, target_names: list[str], promotion_timestamp: str
    ) -> None:
        for target_name in target_names:
            self.connection.execute(
                """
                UPDATE model_records
                SET stage = 'archived', promotion_timestamp = COALESCE(promotion_timestamp, ?)
                WHERE target_name = ? AND stage = 'production'
                """,
                (promotion_timestamp, target_name),
            )
        self.connection.commit()

    def save_model_target_records(
        self,
        metadata: dict[str, Any],
        stage: str,
        target_names: list[str] | None = None,
    ) -> None:
        if stage not in {"candidate", "production", "archived"}:
            raise ValueError(f"Unsupported model stage: {stage}")
        groups = {
            **metadata.get("targets", {}),
            **metadata.get("probability_targets", {}),
        }
        selected = target_names if target_names is not None else list(groups)
        for target_name in selected:
            target = groups.get(target_name)
            if not isinstance(target, dict):
                continue
            version = str(target.get("model_version", metadata.get("model_version", "unknown")))
            validation_metrics = target.get("validation_metrics")
            test_metrics = target.get("test_metrics")
            baseline_metrics = target.get("baseline_metrics")
            audit = {
                "model_version": version,
                "target_name": target_name,
                "stage": stage,
                "promotion_timestamp": target.get(
                    "promotion_timestamp", metadata.get("last_promotion_timestamp")
                ),
                "training_lifecycle_ids": target.get(
                    "training_lifecycle_ids", metadata.get("training_lifecycle_ids", [])
                ),
                "validation_lifecycle_ids": target.get(
                    "validation_lifecycle_ids", metadata.get("validation_lifecycle_ids", [])
                ),
                "test_lifecycle_ids": target.get(
                    "test_lifecycle_ids", metadata.get("test_lifecycle_ids", [])
                ),
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "baseline_metrics": baseline_metrics,
                "feature_schema": metadata.get("feature_names", []),
                "synthetic_data_only": bool(metadata.get("synthetic_data_only", False)),
                "data_domain": metadata.get("data_domain", "legacy_unclassified"),
                "model_stage": metadata.get("model_stage", stage),
                "dataset_hash": (metadata.get("dataset_hashes") or [None])[0],
                "generator_version": metadata.get("generator_version"),
                "production_eligible": bool(metadata.get("production_eligible", False)),
            }
            self.connection.execute(
                """
                INSERT INTO model_records (
                    model_version, target_name, stage, promotion_timestamp,
                    training_lifecycle_ids_json, validation_lifecycle_ids_json,
                    test_lifecycle_ids_json, validation_metrics_json,
                    test_metrics_json, baseline_metrics_json, feature_schema_json,
                    synthetic_data_only, metadata_json, data_domain, model_stage,
                    dataset_hash, generator_version, production_eligible
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version, target_name, stage, audit["promotion_timestamp"],
                    json.dumps(audit["training_lifecycle_ids"]),
                    json.dumps(audit["validation_lifecycle_ids"]),
                    json.dumps(audit["test_lifecycle_ids"]),
                    json.dumps(validation_metrics), json.dumps(test_metrics),
                    json.dumps(baseline_metrics), json.dumps(audit["feature_schema"]),
                    int(audit["synthetic_data_only"]),
                    json.dumps(audit, separators=(",", ":")),
                    audit["data_domain"], audit["model_stage"], audit["dataset_hash"],
                    audit["generator_version"], int(audit["production_eligible"]),
                ),
            )
        self.connection.commit()

    def save_training_run(self, timestamp: str, status: str, details: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO training_runs (timestamp, status, details_json) VALUES (?, ?, ?)",
            (timestamp, status, json.dumps(details, separators=(",", ":"))),
        )
        self.connection.commit()

    def save_validation_result(self, timestamp: str, target: str, metrics: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO validation_results (timestamp, target, metrics_json) VALUES (?, ?, ?)",
            (timestamp, target, json.dumps(metrics, separators=(",", ":"))),
        )
        self.connection.commit()

    def close(self) -> None:
        if self.pending:
            self.connection.commit()
        self.connection.close()
