"""One runtime policy for trained probability thresholds and ML eligibility."""
from __future__ import annotations

import math
import copy
from typing import Any
from dataclasses import dataclass

from .policy_contract import (
    CONTRACT,
    MODEL_METADATA_SCHEMA_VERSION,
    FORECAST_POLICY_CONTRACT_VERSION,
    FORECAST_POLICY_VERSION,
)


THRESHOLD_SCHEMA_VERSION = "1.0"

REASON_PRIORITY = (
    "model_corrupted", "model_load_failed:", "missing_required_model:",
    "metadata_invalid:", "production_ineligible", "model_revoked",
    "model_superseded", "invalid_probability", "invalid_eta",
    "inconsistent_warning_critical_eta", "ood_detected",
    "target_support_violation", "severe_disagreement",
    "feature_schema_mismatch", "insufficient_sample_count",
    "required_window_immature", "sampling_coverage_too_low",
)

def reason_priority(reason: str) -> tuple[int, str]:
    for index, prefix in enumerate(REASON_PRIORITY):
        if reason == prefix or reason.startswith(prefix):
            return index, reason
    return len(REASON_PRIORITY), reason

def ordered_reasons(reasons: Any) -> tuple[str, ...]:
    """Deduplicate all reasons and order by explicit safety priority."""
    return tuple(sorted({str(reason) for reason in reasons if reason}, key=reason_priority))

def primary_reason(reasons: Any, default: str = "none") -> str:
    ordered = ordered_reasons(reasons)
    return ordered[0] if ordered else default


def probability_target_names(horizons: tuple[int, ...] | list[int]) -> list[str]:
    return [f"probability_{kind}_{hours}h" for kind in ("warning", "critical") for hours in horizons]

def mandatory_production_targets(
    horizons: tuple[int, ...] | list[int],
    configured: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    """Return trusted-code requirements, never a model-controlled list."""
    allowed = {"time_to_warning", "time_to_critical", *probability_target_names(horizons)}
    requested = CONTRACT.required_targets if configured is None else tuple(map(str, configured))
    return tuple(target for target in requested if target in allowed)

def validate_eta_target_evidence(target: str, evidence: Any, *, loaded: bool) -> tuple[str, ...]:
    """Fail closed on ETA evidence without inventing a regression quality gate."""
    if not isinstance(evidence, dict):
        return (f"eta_target_evidence_missing:{target}",)
    reasons: list[str] = []
    if evidence.get("target_required") is not True:
        reasons.append(f"mandatory_target_not_marked_required:{target}")
    if evidence.get("target_eligible") is not True:
        reasons.append(f"required_target_ineligible:{target}")
    if evidence.get("model_loaded") is not loaded:
        reasons.append(f"eta_model_loaded_evidence_mismatch:{target}")
    if evidence.get("model_schema_compatible") is not True or evidence.get("feature_schema_compatible") is not True:
        reasons.append(f"eta_schema_incompatible:{target}")
    if evidence.get("physical_validation_passed") is not True or evidence.get("baseline_passed") is not True:
        reasons.append(f"eta_validation_not_passed:{target}")
    target_reasons = evidence.get("target_ineligibility_reasons")
    if not isinstance(target_reasons, list) or target_reasons:
        reasons.append(f"eta_target_reasons_invalid:{target}")
    for field in ("support_min_hours", "support_max_hours"):
        value = evidence.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            reasons.append(f"eta_support_invalid:{target}")
    if not reasons and float(evidence["support_max_hours"]) < float(evidence["support_min_hours"]):
        reasons.append(f"eta_support_invalid:{target}")
    return ordered_reasons(reasons)

@dataclass(frozen=True)
class PromotionPolicyResult:
    passed: bool
    reasons: tuple[str, ...]
    effective_required_targets: tuple[str, ...]
    missing_required_targets: tuple[str, ...]
    runtime_fn_ceiling: float | None

def validate_model_for_production_promotion(metadata: dict[str, Any] | None, available_targets: set[str], *, maximum_fn_rate: float | None, mandatory_targets: tuple[str, ...] | None = None) -> PromotionPolicyResult:
    """The sole registry acceptance gate; metadata can only tighten policy."""
    metadata = metadata or {}
    mandatory_targets = mandatory_targets or CONTRACT.mandatory_targets
    failures: list[str] = []
    if maximum_fn_rate is None: failures.append("runtime_fn_ceiling_missing")
    elif not _finite_rate(maximum_fn_rate) or float(maximum_fn_rate) <= 0: failures.append("runtime_fn_ceiling_invalid")
    if not mandatory_targets: failures.append("mandatory_target_policy_missing")
    if metadata.get("data_domain") != "plant": failures.append("data_domain_not_allowed")
    for flag, reason in (("revoked", "model_revoked"), ("superseded", "model_superseded"), ("corrupted", "model_corrupted")):
        if metadata.get(flag): failures.append(reason)
    declared = metadata.get("required_model_targets")
    if not isinstance(declared, list): failures.append("mandatory_target_policy_missing")
    declared_set = set(map(str, declared or []))
    effective = tuple(sorted(set(mandatory_targets) | declared_set))
    missing = tuple(sorted(set(mandatory_targets) - declared_set))
    failures.extend(f"mandatory_target_missing_from_metadata:{target}" for target in missing)
    for target in effective:
        if target not in available_targets:
            failures.append(f"missing_required_model:{target}")
    if maximum_fn_rate is not None and _finite_rate(maximum_fn_rate) and float(maximum_fn_rate) > 0:
        threshold_records = metadata.get("probability_thresholds", {}).get("targets", {}) if isinstance(metadata.get("probability_thresholds"), dict) else {}
        for target in mandatory_targets:
            if target.startswith("probability_"):
                record = threshold_records.get(target, {})
                selected = record.get("selected_threshold") if isinstance(record, dict) else None
                failures.extend(validate_probability_target_evidence(target, metadata.get("probability_targets", {}).get(target), selected, float(maximum_fn_rate)))
            else:
                failures.extend(validate_eta_target_evidence(target, metadata.get("eta_target_evidence", {}).get(target), loaded=target in available_targets))
    return PromotionPolicyResult(not failures, ordered_reasons(failures), effective, missing, float(maximum_fn_rate) if _finite_rate(maximum_fn_rate) else None)

def effective_required_targets(
    metadata: dict[str, Any],
    horizons: tuple[int, ...] | list[int],
    configured: tuple[str, ...] | list[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    mandatory = mandatory_production_targets(horizons, configured)
    declared_value = metadata.get("required_model_targets")
    declared = tuple(sorted({str(x) for x in declared_value})) if isinstance(declared_value, list) else ()
    effective = tuple(sorted(set(mandatory) | set(declared)))
    missing = tuple(target for target in mandatory if target not in declared)
    return declared, effective, missing

def _finite_rate(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0 <= float(value) <= 1

def _valid_threshold(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0 < float(value) < 1

def validate_probability_target_evidence(
    target: str,
    evidence: Any,
    selected_threshold: Any,
    runtime_fn_ceiling: float = 0.10,
    *,
    required: bool = True,
) -> tuple[str, ...]:
    """Independently recompute FN gates and validate every production metric."""
    if not isinstance(evidence, dict): return (f"required_target_evidence_missing:{target}",)
    reasons: list[str] = []
    threshold = evidence.get("selected_threshold", evidence.get("validation_threshold_selected"))
    if not _valid_threshold(threshold): reasons.append(f"threshold_at_invalid_boundary:{target}" if _finite_rate(threshold) else f"invalid_threshold:{target}")
    if not _valid_threshold(selected_threshold): reasons.append(f"threshold_at_invalid_boundary:{target}" if _finite_rate(selected_threshold) else f"invalid_threshold:{target}")
    if _valid_threshold(threshold) and _valid_threshold(selected_threshold) and abs(float(threshold)-float(selected_threshold)) > 1e-12:
        reasons.append(f"threshold_evidence_mismatch:{target}")
    numeric = ("validation_fp_rate","validation_fn_rate","validation_fn_ceiling","test_fp_rate","test_fn_rate","test_fn_ceiling")
    if any(not _finite_rate(evidence.get(name)) for name in numeric): reasons.append(f"invalid_probability_metric:{target}")
    if all(_finite_rate(evidence.get(name)) for name in ("validation_fn_rate","validation_fn_ceiling")):
        recorded_ceiling=float(evidence["validation_fn_ceiling"]); recorded=float(evidence["validation_fn_rate"])<=recorded_ceiling
        effective=min(float(runtime_fn_ceiling),recorded_ceiling)
        if recorded_ceiling>float(runtime_fn_ceiling): reasons.append(f"model_fn_ceiling_exceeds_runtime_policy:{target}")
        if evidence.get("validation_fn_ceiling_met") is not recorded: reasons.append(f"validation_fn_numeric_contradiction:{target}")
        if float(evidence["validation_fn_rate"])>effective: reasons.append(f"validation_fn_ceiling_not_met:{target}")
    if all(_finite_rate(evidence.get(name)) for name in ("test_fn_rate","test_fn_ceiling")):
        recorded_ceiling=float(evidence["test_fn_ceiling"]); recorded=float(evidence["test_fn_rate"])<=recorded_ceiling
        effective=min(float(runtime_fn_ceiling),recorded_ceiling)
        if recorded_ceiling>float(runtime_fn_ceiling): reasons.append(f"model_fn_ceiling_exceeds_runtime_policy:{target}")
        if evidence.get("test_fn_ceiling_met") is not recorded: reasons.append(f"test_fn_numeric_contradiction:{target}")
        if float(evidence["test_fn_rate"])>effective: reasons.append(f"test_fn_ceiling_not_met:{target}")
    if evidence.get("baseline_passed") is not True: reasons.append(f"baseline_not_passed:{target}")
    if evidence.get("calibration_passed") is not True: reasons.append(f"calibration_not_passed:{target}")
    target_reasons = evidence.get("target_ineligibility_reasons")
    if evidence.get("target_eligible") is not True: reasons.append(f"target_ineligible:{target}")
    if "target_required" not in evidence: reasons.append(f"required_target_flag_missing:{target}")
    elif not isinstance(evidence.get("target_required"), bool): reasons.append(f"required_target_flag_invalid:{target}")
    elif required and evidence.get("target_required") is not True: reasons.append(f"mandatory_target_not_marked_required:{target}")
    elif not required and evidence.get("target_required") is not False: reasons.append(f"optional_target_marked_required:{target}")
    if not isinstance(target_reasons, list): reasons.append(f"required_target_evidence_missing:{target}")
    elif target_reasons: reasons.append(f"eligible_target_has_rejection_reasons:{target}")
    return ordered_reasons(reasons)


def threshold_record(metadata: dict[str, Any], target: str) -> dict[str, Any] | None:
    """Read the versioned record; legacy metadata is intentionally ineligible."""
    container = metadata.get("probability_thresholds")
    if not isinstance(container, dict) or str(container.get("schema_version")) != THRESHOLD_SCHEMA_VERSION:
        return None
    record = container.get("targets", {}).get(target)
    return record if isinstance(record, dict) else None


def _legacy_count(metrics: Any, positive: bool) -> int | None:
    if not isinstance(metrics, dict):
        return None
    count = metrics.get("sample_count")
    rate = metrics.get("observed_event_rate")
    if not isinstance(count, int) or not _finite_rate(rate):
        return None
    positives = round(count * float(rate))
    if abs(positives - count * float(rate)) > 1e-6:
        return None
    return int(positives if positive else count - positives)


def normalize_metadata_contract(
    metadata: dict[str, Any] | None,
    horizons: tuple[int, ...] | list[int],
    *,
    required_targets: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Return canonical metadata or a fail-closed invalid marker.

    The only supported migration is the repository's unversioned legacy
    layout.  Values are copied only when the legacy paths agree; absent FN
    ceilings or other evidence remain absent and therefore ineligible.
    """
    if not isinstance(metadata, dict):
        return {"metadata_invalid": "not_an_object"}
    result = copy.deepcopy(metadata)
    version = result.get("model_metadata_schema_version")
    required = set(mandatory_production_targets(horizons, required_targets))
    if version is not None and str(version) != MODEL_METADATA_SCHEMA_VERSION:
        result["metadata_invalid"] = f"unsupported_model_metadata_schema_version:{version}"
        return result
    legacy = version is None
    if legacy:
        result["migrated_from_model_metadata_schema_version"] = "legacy_unversioned"
        result["model_metadata_schema_version"] = MODEL_METADATA_SCHEMA_VERSION
    contract_version = result.get("forecast_policy_contract_version")
    if contract_version not in (None, FORECAST_POLICY_CONTRACT_VERSION):
        result["metadata_invalid"] = f"unsupported_forecast_policy_contract_version:{contract_version}"
        return result
    result["forecast_policy_contract_version"] = FORECAST_POLICY_CONTRACT_VERSION
    result["forecast_policy_version"] = FORECAST_POLICY_VERSION
    result["forecast_policy_contract"] = {
        **CONTRACT.to_dict(),
        "required_targets": sorted(required),
    }
    if legacy:
        result["required_model_targets"] = sorted(required)
    elif not isinstance(result.get("required_model_targets"), list):
        result["metadata_invalid"] = "required_model_targets_missing_or_invalid"

    probability_targets = result.get("probability_targets")
    if not isinstance(probability_targets, dict):
        probability_targets = {}
        result["probability_targets"] = probability_targets
    threshold_targets: dict[str, Any] = {}
    existing_thresholds = result.get("probability_thresholds")
    if isinstance(existing_thresholds, dict) and str(existing_thresholds.get("schema_version")) == THRESHOLD_SCHEMA_VERSION:
        target_values = existing_thresholds.get("targets")
        if isinstance(target_values, dict):
            threshold_targets = copy.deepcopy(target_values)

    for target in probability_target_names(horizons):
        record = probability_targets.get(target)
        if not isinstance(record, dict):
            continue
        validation = record.get("validation_metrics") if isinstance(record.get("validation_metrics"), dict) else {}
        test = record.get("test_metrics") if isinstance(record.get("test_metrics"), dict) else {}
        validation_threshold = record.get("selected_threshold", record.get("validation_threshold_selected", validation.get("classification_threshold")))
        test_threshold = test.get("classification_threshold")
        threshold = validation_threshold if _valid_threshold(validation_threshold) and (
            test_threshold is None or (_valid_threshold(test_threshold) and abs(float(validation_threshold) - float(test_threshold)) <= 1e-12)
        ) else None
        aliases = {
            "selected_threshold": threshold,
            "validation_threshold_selected": threshold,
            "validation_fp_rate": validation.get("false_positive_rate"),
            "validation_fn_rate": validation.get("false_negative_rate"),
            "test_fp_rate": test.get("false_positive_rate"),
            "test_fn_rate": test.get("false_negative_rate"),
            "baseline_passed": record.get("baseline_passed", record.get("beats_constant_rate_baseline")),
            "calibration_passed": record.get("calibration_passed", (
                validation.get("calibration_error") is not None
                and test.get("calibration_error") is not None
                and _finite_rate(record.get("maximum_calibration_error"))
                and float(validation["calibration_error"]) <= float(record["maximum_calibration_error"])
                and float(test["calibration_error"]) <= float(record["maximum_calibration_error"])
            )),
            "target_required": target in required,
            "validation_sample_count": validation.get("sample_count"),
            "test_sample_count": test.get("sample_count"),
            "validation_positive_count": _legacy_count(validation, True),
            "validation_negative_count": _legacy_count(validation, False),
            "test_positive_count": _legacy_count(test, True),
            "test_negative_count": _legacy_count(test, False),
        }
        for name, value in aliases.items():
            if legacy and name == "target_required":
                record[name] = value
            else:
                record.setdefault(name, value)
        reasons = record.get("target_ineligibility_reasons")
        if not isinstance(reasons, list):
            reasons = [] if record.get("eligible") is True else [str(record.get("reason") or "legacy_target_ineligible")]
            record["target_ineligibility_reasons"] = reasons
        record.setdefault("target_eligible", bool(record.get("eligible") is True and not reasons))
        threshold_targets[target] = {
            "selected_threshold": threshold,
            "eligible": bool(record.get("target_eligible") is True and threshold is not None),
            "reason": (None if record.get("target_eligible") is True and threshold is not None else (reasons[0] if reasons else "missing_selected_threshold")),
            "threshold_selected_on": record.get("threshold_selected_on"),
        }
    result["probability_thresholds"] = {"schema_version": THRESHOLD_SCHEMA_VERSION, "targets": threshold_targets}
    return result


def runtime_probability_policy(metadata: dict[str, Any] | None, horizons: tuple[int, ...]) -> tuple[dict[str, float | None], dict[str, bool], tuple[str, ...]]:
    metadata = metadata or {}
    thresholds: dict[str, float | None] = {}
    eligible: dict[str, bool] = {}
    reasons: set[str] = set()
    for target in probability_target_names(horizons):
        record = threshold_record(metadata, target)
        value = record.get("selected_threshold") if record else None
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0 < float(value) < 1
        evidence = metadata.get("probability_targets", {}).get(target)
        target_required = bool(isinstance(evidence, dict) and evidence.get("target_required") is True)
        evidence_reasons = validate_probability_target_evidence(
            target, evidence, value, required=target_required
        )
        target_ok = bool(record and record.get("eligible") is True and valid and not evidence_reasons)
        thresholds[target] = float(value) if valid else None
        eligible[target] = target_ok
        if not target_ok:
            reasons.update(evidence_reasons or (str(record.get("reason") if record else "missing_selected_threshold"),))
    return thresholds, eligible, ordered_reasons(reasons)


def production_eligibility(metadata: dict[str, Any] | None, horizons: tuple[int, ...], *, deployment_domain: str = "plant", loaded_targets: set[str] | None = None, load_failures: dict[str, str] | None = None, maximum_fn_rate: float = 0.10, mandatory_targets: tuple[str, ...] | None = None) -> tuple[bool, tuple[str, ...], dict[str, float | None], dict[str, bool]]:
    """Central, fail-closed live eligibility check. FN comparison is <= at training."""
    metadata = metadata or {}
    thresholds, targets, _ = runtime_probability_policy(metadata, horizons)
    failures: set[str] = set()
    if metadata.get("maturity_stage") != "deployed": failures.add("model_not_deployed")
    if metadata.get("model_stage") not in {"plant_production", "production"}: failures.add("model_stage_not_allowed")
    if metadata.get("production_eligible") is not True: failures.add("production_ineligible")
    if metadata.get("data_domain") != deployment_domain: failures.add("data_domain_not_allowed")
    if metadata.get("revoked"): failures.add("model_revoked")
    if metadata.get("superseded"): failures.add("model_superseded")
    if metadata.get("corrupted"): failures.add("model_corrupted")
    if metadata.get("metadata_invalid"): failures.add(f"metadata_invalid:{metadata['metadata_invalid']}")
    if not metadata.get("model_version"): failures.add("missing_model_version")
    # New metadata declares required artifacts explicitly. For pre-schema
    # metadata, every configured probability target and both ETA models are
    # required; absence must never be mistaken for successful loading.
    declared, effective, mandatory_missing = effective_required_targets(metadata, horizons, mandatory_targets)
    required = set(effective)
    for target in mandatory_missing: failures.add(f"mandatory_target_missing_from_metadata:{target}")
    loaded_targets = loaded_targets if loaded_targets is not None else set()
    for target in sorted(required):
        if target not in loaded_targets:
            failures.add((load_failures or {}).get(target, f"missing_required_model:{target}"))
    for target in probability_target_names(horizons):
        required_target = target in required
        if target not in metadata.get("probability_targets", {}) and required_target: failures.add(f"required_probability_target_missing:{target}")
        record = metadata.get("probability_targets", {}).get(target)
        selected = thresholds.get(target)
        target_failures = validate_probability_target_evidence(target, record, selected, maximum_fn_rate, required=required_target)
        if required_target: failures.update(target_failures)
        if target not in loaded_targets: targets[target] = False
    for target in ("time_to_warning", "time_to_critical"):
        if target in required:
            failures.update(validate_eta_target_evidence(target, metadata.get("eta_target_evidence", {}).get(target), loaded=target in loaded_targets))
    return not failures, ordered_reasons(failures), thresholds, targets
