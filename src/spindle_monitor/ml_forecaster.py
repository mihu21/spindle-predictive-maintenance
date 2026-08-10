from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Some Windows/container combinations do not expose a physical-core count.
# Joblib explicitly supports this override and otherwise emits a warning on
# first parallel tree prediction despite being able to use logical cores.
_logical_cpus = max(os.cpu_count() or 1, 1)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(_logical_cpus - 1, 1)))

from .config import ProjectConfig
from .model_registry import ModelRegistry
from .models import MLForecast
from .forecast_policy import (
    production_eligibility,
    ordered_reasons,
    mandatory_production_targets,
    effective_required_targets,
    validate_probability_target_evidence,
)


def reconcile_horizon_probabilities(
    probabilities: dict[str, dict[int, float | None]],
    horizons: tuple[int, ...],
    target_eligibility: dict[str, bool] | None = None,
) -> tuple[dict[str, dict[int, float | None]], tuple[str, ...]]:
    """Enforce nested-horizon monotonicity while preserving caller-owned raw values."""
    reconciled = {condition: dict(values) for condition, values in probabilities.items()}
    changed: list[str] = []
    for condition in ("warning", "critical"):
        running: float | None = None
        for hours in sorted(horizons):
            target = f"probability_{condition}_{hours}h"
            if target_eligibility is not None and target_eligibility.get(target) is not True:
                continue
            value = reconciled.get(condition, {}).get(hours)
            if value is None or not np.isfinite(value):
                continue
            adjusted = float(value) if running is None else max(float(value), running)
            if adjusted != float(value):
                changed.append(target)
                reconciled[condition][hours] = adjusted
            running = adjusted
    return reconciled, tuple(changed)


class MLForecaster:
    """Load compatible production regressors and calibrated horizon classifiers."""

    def __init__(
        self,
        config: ProjectConfig,
        feature_names: list[str],
        models_root: str | Path,
    ) -> None:
        self.config = config
        self.feature_names = list(feature_names)
        configured_required = tuple(config.ml.required_eta_targets + config.ml.required_probability_targets)
        self.registry = ModelRegistry(
            models_root,
            maximum_probability_false_negative_rate=config.ml.maximum_probability_false_negative_rate,
            mandatory_targets=configured_required,
        )
        self.models: dict[str, object] = {}
        self.metadata: dict | None = None
        self.target_versions: dict[str, str] = {}
        self.load_failures: dict[str, str] = {}
        self._load()

    @property
    def _targets(self) -> list[str]:
        targets = ["time_to_warning", "time_to_critical"]
        for condition in ("warning", "critical"):
            targets.extend(
                f"probability_{condition}_{hours}h"
                for hours in self.config.ml.forecast_horizons_hours
            )
        return targets

    def _load(self) -> None:
        stage = "candidate" if self.registry.root.name.lower() in {"mock", "realistic"} else "production"
        try:
            self.metadata = self.registry.load_metadata(stage)
        except Exception:
            self.metadata = {"metadata_invalid": "unreadable"}
        if self.metadata is not None and not isinstance(self.metadata, dict):
            self.metadata = {"metadata_invalid": "not_an_object"}
        for target in self._targets:
            try:
                loaded = self.registry.load_model(
                    target, stage=stage, expected_features=self.feature_names,
                    expected_schema_version=self.config.ml.schema_version,
                    expected_threshold_config_version=self.config.threshold_config_version,
                    expected_lifecycle_config_version=self.config.lifecycle_config_version,
                )
            except Exception:
                self.load_failures[target] = f"model_load_failed:{target}"
                continue
            if loaded is not None:
                model, metadata = loaded
                declared = metadata.get("artifact_targets", {}).get(target, target)
                if declared != target:
                    self.load_failures[target] = f"loaded_target_mismatch:{target}"
                    continue
                self.models[target] = model
                self.metadata = metadata
                group = "probability_targets" if target.startswith("probability_") else "targets"
                self.target_versions[target] = str(
                    metadata.get(group, {}).get(target, {}).get(
                        "model_version", metadata.get("model_version", "unknown")
                    )
                )

    def predict(self, features: dict[str, float]) -> MLForecast:
        return self.predict_many([features])[0]

    def predict_many(self, feature_rows: list[dict[str, float]]) -> list[MLForecast]:
        """Vectorized offline inference; row ordering and causal features are preserved."""
        if not feature_rows:
            return []
        if self.metadata is None:
            return [MLForecast(policy_withholding_reasons=("metadata_invalid:missing",),
                production_policy_reasons=("metadata_invalid:missing",), reason="Model metadata is missing.") for _ in feature_rows]
        try:
            rows = np.asarray(
                [[float(features[name]) for name in self.feature_names] for features in feature_rows],
                dtype=float,
            )
        except KeyError as exc:
            return [
                MLForecast(
                    reason=f"Runtime feature schema is missing {exc.args[0]!r}; ML was not used."
                ) for _ in feature_rows
            ]

        distribution = self.metadata.get("feature_distribution", {})
        lower = distribution.get("lower_quantile_01")
        upper = distribution.get("upper_quantile_99")
        outside_distribution = np.zeros(len(rows), dtype=bool)
        outside_feature_fraction = np.zeros(len(rows), dtype=float)
        if isinstance(lower, list) and isinstance(upper, list) and len(lower) == rows.shape[1] == len(upper):
            lower_array = np.asarray(lower, dtype=float)
            upper_array = np.asarray(upper, dtype=float)
            span = np.maximum(upper_array - lower_array, 1e-9)
            margin = self.config.ml.feature_distribution_margin_fraction * span
            outside = (rows < lower_array - margin) | (rows > upper_array + margin)
            outside_feature_fraction = np.mean(outside, axis=1)
            outside_distribution = (
                outside_feature_fraction
                > self.config.ml.feature_distribution_max_outside_fraction
            )

        predictions: dict[str, np.ndarray] = {}
        probabilities: dict[str, np.ndarray] = {}
        for target, model in self.models.items():
            if target.startswith("time_to_"):
                # Preserve estimator output exactly. Support violations are flagged below;
                # forecasts are never silently clipped to the observed target range.
                predictions[target] = np.asarray(model.predict(rows), dtype=float)
                continue
            if hasattr(model, "predict_proba"):
                probability = np.asarray(model.predict_proba(rows)[:, 1], dtype=float)
            else:
                probability = np.asarray(model.predict(rows), dtype=float)
            # Do not repair malformed estimator output: it is a safety failure.
            probabilities[target] = probability

        maturity = str(self.metadata.get("maturity_stage", "experimental"))
        policy_ok, policy_reasons, selected_thresholds, target_eligibility = production_eligibility(
            self.metadata, self.config.ml.forecast_horizons_hours,
            loaded_targets=set(self.models), load_failures=self.load_failures,
            maximum_fn_rate=self.config.ml.maximum_probability_false_negative_rate,
            mandatory_targets=tuple(self.config.ml.required_eta_targets + self.config.ml.required_probability_targets),
        )
        compatibility_reasons = []
        if str(self.metadata.get("input_schema_version")) != str(self.config.ml.schema_version): compatibility_reasons.append("model_schema_incompatible")
        if self.metadata.get("feature_names") != self.feature_names: compatibility_reasons.append("feature_schema_mismatch")
        if compatibility_reasons:
            policy_ok = False
            policy_reasons = ordered_reasons(policy_reasons + tuple(compatibility_reasons))
        configured_required = tuple(self.config.ml.required_eta_targets + self.config.ml.required_probability_targets)
        target_reasons = {}
        for target in target_eligibility:
            record = self.metadata.get("probability_targets", {}).get(target, {})
            reasons = list(record.get("target_ineligibility_reasons", [])) if isinstance(record, dict) and isinstance(record.get("target_ineligibility_reasons"), list) else []
            reasons.extend(validate_probability_target_evidence(
                target, record, selected_thresholds.get(target),
                self.config.ml.maximum_probability_false_negative_rate,
                required=target in configured_required,
            ))
            if target not in self.models:
                reasons.append(self.load_failures.get(target, f"model_unavailable:{target}"))
            target_reasons[target] = ordered_reasons(reasons)
            target_eligibility[target] = bool(target_eligibility[target] and not target_reasons[target])
        output: list[MLForecast] = []
        for index in range(len(rows)):
            is_outside = bool(outside_distribution[index])
            target_maxima: dict[str, float | None] = {}
            target_minima: dict[str, float | None] = {}
            target_violations: dict[str, bool] = {}
            physically_invalid: dict[str, bool] = {}
            beyond_support = False
            target_support = self.metadata.get("target_support", {})
            for target in ("time_to_warning", "time_to_critical"):
                minimum = target_support.get(target, {}).get("minimum")
                maximum = target_support.get(target, {}).get("maximum")
                target_minima[target] = float(minimum) if minimum is not None else None
                target_maxima[target] = float(maximum) if maximum is not None else None
                prediction = float(predictions[target][index]) if target in predictions else None
                violation = bool(
                    prediction is not None and (
                        (minimum is not None and prediction < float(minimum))
                        or (maximum is not None and prediction > float(maximum))
                    )
                )
                target_violations[target] = violation
                physically_invalid[target] = bool(prediction is not None and (not np.isfinite(prediction) or prediction < 0.0))
                beyond_support |= violation
            duration_max = self.metadata.get("lifecycle_duration_distribution_hours", {}).get("maximum")
            elapsed_name = "elapsed_lifecycle_hours"
            if duration_max is not None and elapsed_name in self.feature_names:
                beyond_support |= bool(rows[index, self.feature_names.index(elapsed_name)] > float(duration_max))
            sampling_ood = False
            sampling_support = self.metadata.get("sampling_interval_distribution_seconds", {})
            maximum_sampling = sampling_support.get("maximum")
            gap_names = [i for i, name in enumerate(self.feature_names) if name.endswith("sampling_gap_max_seconds")]
            if maximum_sampling is not None and gap_names:
                sampling_ood = bool(np.max(rows[index, gap_names]) > float(maximum_sampling) * 1.5)
            is_outside = bool(is_outside or sampling_ood)
            confidence = "low" if is_outside or beyond_support else ("medium" if maturity == "deployed" else "experimental")
            reason = (
                "One or more predictions are outside observed training target support."
                if any(target_violations.values())
                else "Current features are outside the candidate training distribution."
                if is_outside
                else "Production ML forecasts generated from the saved lifecycle-trained feature schema."
                if maturity == "deployed"
                else "ML models are available but remain experimental until deployment criteria are met."
            )
            horizon_probabilities = {
                condition: {
                    hours: (
                        None
                        if f"probability_{condition}_{hours}h" not in probabilities
                        else float(probabilities[f"probability_{condition}_{hours}h"][index])
                    )
                    for hours in self.config.ml.forecast_horizons_hours
                }
                for condition in ("warning", "critical")
            }
            raw_horizon_probabilities = {
                condition: dict(values) for condition, values in horizon_probabilities.items()
            }
            horizon_probabilities, reconciliation_targets = reconcile_horizon_probabilities(
                horizon_probabilities, self.config.ml.forecast_horizons_hours, target_eligibility
            )
            invalid_probability = any(
                value is not None and (not np.isfinite(value) or value < 0.0 or value > 1.0)
                for values in horizon_probabilities.values() for value in values.values()
            )
            invalid_eta = any(physically_invalid.values())
            eta_inconsistent = (
                predictions.get("time_to_warning") is not None and predictions.get("time_to_critical") is not None
                and np.isfinite(predictions["time_to_warning"][index]) and np.isfinite(predictions["time_to_critical"][index])
                and predictions["time_to_warning"][index] > predictions["time_to_critical"][index]
            )
            physical_reasons = tuple(reason for reason, failed in (
                ("invalid_probability", invalid_probability), ("invalid_eta", invalid_eta),
                ("inconsistent_warning_critical_eta", eta_inconsistent),
            ) if failed)
            metadata_reason = (f"metadata_invalid:{self.metadata['metadata_invalid']}",) if self.metadata.get("metadata_invalid") else ()
            all_reasons = ordered_reasons(policy_reasons + physical_reasons + metadata_reason)
            metadata_required, required, mandatory_missing = effective_required_targets(
                self.metadata, self.config.ml.forecast_horizons_hours, configured_required
            )
            missing = tuple(target for target in required if target not in self.models)
            output.append(MLForecast(
                time_to_warning_hours=(
                    None if "time_to_warning" not in predictions
                    else float(predictions["time_to_warning"][index])
                ),
                time_to_critical_hours=(
                    None if "time_to_critical" not in predictions
                    else float(predictions["time_to_critical"][index])
                ),
                probability_warning=horizon_probabilities["warning"],
                probability_critical=horizon_probabilities["critical"],
                raw_probability_warning=raw_horizon_probabilities["warning"],
                raw_probability_critical=raw_horizon_probabilities["critical"],
                reconciled_probability_warning=dict(horizon_probabilities["warning"]),
                reconciled_probability_critical=dict(horizon_probabilities["critical"]),
                probability_reconciled=bool(reconciliation_targets),
                probability_reconciliation_targets=reconciliation_targets,
                probability_reconciliation_reason=("cumulative_max_enforces_nested_horizons" if reconciliation_targets else "none"),
                maturity_stage=maturity,
                model_version=", ".join(sorted(set(self.target_versions.values()))) or str(self.metadata.get("model_version", "")),
                confidence=(confidence if policy_ok and not physical_reasons else "low"),
                reason=(reason if not all_reasons else reason + " Withheld: " + ", ".join(all_reasons)),
                outside_training_distribution=is_outside,
                outside_feature_fraction=float(outside_feature_fraction[index]),
                beyond_training_duration_support=beyond_support,
                training_target_max_hours=target_maxima,
                training_target_min_hours=target_minima,
                target_support_violations=target_violations,
                physically_invalid_targets=physically_invalid,
                sampling_interval_out_of_distribution=sampling_ood,
                prediction_confidence=confidence,
                selected_thresholds=selected_thresholds,
                target_eligibility=target_eligibility,
                target_ineligibility_reasons=target_reasons,
                policy_withholding_reasons=all_reasons,
                physical_validation_passed=not physical_reasons,
                physical_validation_reasons=physical_reasons,
                production_policy_passed=policy_ok,
                production_policy_reasons=policy_reasons,
                required_model_targets=required,
                loaded_model_targets=tuple(sorted(self.models)),
                missing_required_model_targets=missing,
                probability_threshold_schema_version=(self.metadata.get("probability_thresholds") or {}).get("schema_version"),
                model_metadata_schema_version=self.metadata.get("model_metadata_schema_version"),
                model_forecast_policy_contract_version=self.metadata.get("forecast_policy_contract_version"),
                model_stage=str(self.metadata.get("model_stage", "unavailable")),
                data_domain=str(self.metadata.get("data_domain", "unknown")),
                production_eligible=bool(self.metadata.get("production_eligible", False)),
                revoked=bool(self.metadata.get("revoked", False)),
                superseded=bool(self.metadata.get("superseded", False)),
                corrupted=bool(self.metadata.get("corrupted", False)),
                metadata_valid=not bool(self.metadata.get("metadata_invalid")),
                artifacts_valid=not missing and not self.load_failures,
                schema_compatible=(str(self.metadata.get("input_schema_version")) == str(self.config.ml.schema_version)),
                feature_schema_compatible=(self.metadata.get("feature_names") == self.feature_names),
                model_load_failures=dict(sorted(self.load_failures.items())),
                probability_target_evidence={target: {
                    "target_required": target in required,
                    "target_eligible": bool(target_eligibility.get(target, False)),
                    "target_ineligibility_reasons": list(target_reasons.get(target, ())),
                    "selected_threshold": selected_thresholds.get(target),
                    "validation_fp_rate": self.metadata.get("probability_targets", {}).get(target, {}).get("validation_fp_rate"),
                    "validation_fn_rate": self.metadata.get("probability_targets", {}).get(target, {}).get("validation_fn_rate"),
                    "validation_fn_ceiling": self.metadata.get("probability_targets", {}).get(target, {}).get("validation_fn_ceiling"),
                    "validation_fn_ceiling_met": self.metadata.get("probability_targets", {}).get(target, {}).get("validation_fn_ceiling_met"),
                    "test_fp_rate": self.metadata.get("probability_targets", {}).get(target, {}).get("test_fp_rate"),
                    "test_fn_rate": self.metadata.get("probability_targets", {}).get(target, {}).get("test_fn_rate"),
                    "test_fn_ceiling": self.metadata.get("probability_targets", {}).get(target, {}).get("test_fn_ceiling"),
                    "test_fn_ceiling_met": self.metadata.get("probability_targets", {}).get(target, {}).get("test_fn_ceiling_met"),
                    "baseline_passed": self.metadata.get("probability_targets", {}).get(target, {}).get("baseline_passed"),
                    "calibration_passed": self.metadata.get("probability_targets", {}).get(target, {}).get("calibration_passed"),
                } for target in target_eligibility},
                eta_target_evidence={
                    target: {
                        key: value for key, value in {
                            "target": target,
                            "target_required": record.get("target_required", False),
                            "target_eligible": record.get("target_eligible", False),
                            "target_ineligibility_reasons": record.get("target_ineligibility_reasons", []),
                            "model_loaded": target in self.models,
                            "model_schema_compatible": record.get("model_schema_compatible", False),
                            "feature_schema_compatible": record.get("feature_schema_compatible", False),
                            "physical_validation_passed": record.get("physical_validation_passed", False),
                            "baseline_passed": record.get("baseline_passed", False),
                            "support_min_hours": record.get("support_min_hours"),
                            "support_max_hours": record.get("support_max_hours"),
                            "training_sample_count": record.get("training_sample_count"),
                            "validation_sample_count": record.get("validation_sample_count"),
                            "test_sample_count": record.get("test_sample_count"),
                            "data_domain": record.get("data_domain"),
                        }.items()
                    }
                    for target, record in self.metadata.get("eta_target_evidence", {}).items()
                    if isinstance(record, dict)
                },
                configured_mandatory_targets=mandatory_production_targets(self.config.ml.forecast_horizons_hours, configured_required),
                metadata_required_targets=metadata_required,
                effective_required_targets=required,
                mandatory_targets_missing_from_metadata=mandatory_missing,
                runtime_probability_fn_ceiling=self.config.ml.maximum_probability_false_negative_rate,
                model_recorded_probability_fn_ceiling={target:self.metadata.get("probability_targets",{}).get(target,{}).get("validation_fn_ceiling") for target in target_eligibility},
                effective_probability_fn_ceiling={target:(min(self.config.ml.maximum_probability_false_negative_rate,float(value)) if isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) else None)
                    for target,value in ((name,self.metadata.get("probability_targets",{}).get(name,{}).get("validation_fn_ceiling")) for name in target_eligibility)},
                runtime_validation_fn_ceiling=self.config.ml.maximum_probability_false_negative_rate,
                runtime_test_fn_ceiling=self.config.ml.maximum_probability_false_negative_rate,
                model_recorded_validation_fn_ceilings={target:self.metadata.get("probability_targets",{}).get(target,{}).get("validation_fn_ceiling") for target in target_eligibility},
                model_recorded_test_fn_ceilings={target:self.metadata.get("probability_targets",{}).get(target,{}).get("test_fn_ceiling") for target in target_eligibility},
                effective_validation_fn_ceilings={target:(min(self.config.ml.maximum_probability_false_negative_rate,float(value)) if isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) else None) for target,value in ((n,self.metadata.get("probability_targets",{}).get(n,{}).get("validation_fn_ceiling")) for n in target_eligibility)},
                effective_test_fn_ceilings={target:(min(self.config.ml.maximum_probability_false_negative_rate,float(value)) if isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) else None) for target,value in ((n,self.metadata.get("probability_targets",{}).get(n,{}).get("test_fn_ceiling")) for n in target_eligibility)},
            ))
        return output
