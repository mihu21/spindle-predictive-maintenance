from __future__ import annotations

from collections import deque
from dataclasses import replace, dataclass
from datetime import datetime
from pathlib import Path

from .config import ProjectConfig
from .anomaly import AnomalyDetector
from .feature_engineering import FeatureGenerator
from .kalman import DegradationKalmanFilter
from .lifecycle_detector import LifecycleDetector
from .ml_forecaster import MLForecaster
from .prognostics import ProbabilisticDegradationForecaster, PROGNOSTIC_METHOD_VERSION
from .models import (
    ForecastDecision,
    MLForecast,
    PrognosticForecast,
    MonitorResult,
    SensorAssessment,
    SensorReading,
    StatisticalForecast,
    Status,
    ModelAction,
    QualityStatus,
    FeatureHistoryAction,
)
from .offline import OfflineLifecyclePlan
from .rules import critical_margin_percent, health_from_severity, severity_for_value, status_for_value, validate_value
from .smoothing import SignalSmoother, SmoothedSignal
from .statistical_forecast import (
    sensor_kalman_threshold_forecast,
    statistical_time_to_critical,
    statistical_time_to_warning,
)
from .status_policy import StabilizedStatusPolicy, TimeBasedEventPolicy
from .forecast_policy import ordered_reasons, primary_reason

@dataclass(frozen=True)
class UrgencyDecision:
    urgency: str
    actionable: bool
    recommendation_source: str = "unavailable"
    probability_trigger_targets: tuple[str, ...] = ()
    eta_trigger_targets: tuple[str, ...] = ()
    trigger_rules: tuple[str, ...] = ()


class ConditionMonitor:
    def __init__(
        self,
        config: ProjectConfig,
        *,
        models_root: str | Path | None = None,
        lifecycle_plan: OfflineLifecyclePlan | None = None,
        enable_ml: bool = True,
        data_domain: str = "accelerated_mock",
        enable_features: bool = True,
        dataset_hash: str = "",
        generator_version: str = "",
    ) -> None:
        self.config = config
        self.data_domain = data_domain
        self.enable_features = enable_features
        self.dataset_hash = dataset_hash
        self.generator_version = generator_version
        self.timestamps: deque[datetime] = deque(maxlen=config.monitoring.history_size)
        self.raw_history: dict[str, deque[float]] = {
            key: deque(maxlen=config.monitoring.history_size) for key in config.sensors
        }
        self.last_timestamp: datetime | None = None
        self.status_policy = StabilizedStatusPolicy(config.monitoring)
        self.event_policy = TimeBasedEventPolicy(config.lifecycle)
        self.smoother = SignalSmoother(config)
        self.lifecycle = LifecycleDetector(config)
        self.lifecycle_plan = lifecycle_plan
        self.features = FeatureGenerator(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.degradation_kalman = self._new_degradation_filter()
        self.prognostics = ProbabilisticDegradationForecaster(config)
        self.prognostic_probability_state: dict[str, datetime] = {}
        self.prognostic_probability_active: set[str] = set()
        self.prognostic_probability_release_state: dict[str, datetime] = {}
        if models_root is None:
            models_root = Path(__file__).resolve().parents[2] / "models"
        self.ml = (
            MLForecaster(config, self.features.names, models_root) if enable_ml else None
        )
        self.audit_model_metadata = self.ml.metadata if self.ml and self.ml.metadata else {}

    def _new_degradation_filter(self) -> DegradationKalmanFilter:
        return DegradationKalmanFilter(
            self.config.monitoring.kalman_process_variance,
            self.config.monitoring.kalman_measurement_variance,
        )

    def _assess(
        self, values: dict[str, float], signals: dict[str, SmoothedSignal]
    ) -> tuple[dict[str, SensorAssessment], dict[str, Status], Status, list[str]]:
        assessments: dict[str, SensorAssessment] = {}
        sensor_statuses: dict[str, Status] = {}
        reasons: list[str] = []
        raw_status = Status.NORMAL
        for key, sensor in self.config.sensors.items():
            raw_value = values[key]
            signal = signals[key]
            sensor_raw_status = status_for_value(raw_value, sensor, self.config)
            sensor_smoothed_status = status_for_value(signal.kalman_level, sensor, self.config)
            sensor_statuses[key] = sensor_raw_status
            raw_status = max(raw_status, sensor_raw_status)
            assessments[key] = SensorAssessment(
                sensor=key,
                display_name=sensor.display_name,
                unit=sensor.unit,
                raw_value=raw_value,
                smoothed_value=signal.kalman_level,
                raw_status=sensor_raw_status,
                smoothed_status=sensor_smoothed_status,
                severity=severity_for_value(signal.kalman_level, sensor),
                health_percent=health_from_severity(severity_for_value(signal.kalman_level, sensor)),
                critical_margin_percent=critical_margin_percent(raw_value, sensor),
                forecast=sensor_kalman_threshold_forecast(
                    self.config, key, signal, sensor_raw_status, len(self.timestamps)
                ),
                rolling_median=signal.rolling_median,
                fast_ewma=signal.fast_ewma,
                slow_ewma=signal.slow_ewma,
                kalman_level=signal.kalman_level,
                kalman_rate_per_hour=signal.kalman_rate_per_hour,
            )
            if sensor_raw_status != Status.NORMAL:
                threshold = sensor.critical if sensor_raw_status == Status.CRITICAL else sensor.warning
                reasons.append(
                    f"{sensor.display_name} is {sensor_raw_status.name}: {raw_value:.3f} "
                    f"{sensor.unit} exceeds {threshold:.3f} {sensor.unit}."
                )
        return assessments, sensor_statuses, raw_status, reasons

    def _worst_sensor(self, assessments: dict[str, SensorAssessment]) -> str:
        """Choose the physical worst sensor, prioritising raw safety status.

        Smoothed severity is useful for trends but must not hide the raw sensor
        that actually triggered a WARNING or CRITICAL manufacturer condition.
        """
        overall_status = max(assessment.raw_status for assessment in assessments.values())
        status_candidates = [
            key for key, assessment in assessments.items()
            if assessment.raw_status == overall_status
        ]
        return max(
            status_candidates,
            key=lambda key: severity_for_value(
                assessments[key].raw_value, self.config.sensors[key]
            ),
        )

    def _minimum_final_confidence_allowed(self, confidence: str) -> bool:
        rank = {"low": 0, "medium": 1, "high": 2, "reached": 3}
        return rank.get(confidence, -1) >= rank[self.config.ml.minimum_final_confidence]

    @staticmethod
    def _probability_trigger_targets(ml: MLForecast) -> tuple[str, ...]:
        values={**{f"probability_warning_{h}h":v for h,v in ml.probability_warning.items()},
                **{f"probability_critical_{h}h":v for h,v in ml.probability_critical.items()}}
        return tuple(sorted(target for target,value in values.items()
            if ml.target_eligibility.get(target) is True and value is not None
            and ml.selected_thresholds.get(target) is not None
            and value >= ml.selected_thresholds[target]))

    def _combine_forecasts(
        self,
        raw_status: Status,
        statistical: StatisticalForecast,
        ml: MLForecast,
        target: str = "critical",
    ) -> ForecastDecision:
        threshold_status = Status.WARNING if target == "warning" else Status.CRITICAL
        if raw_status >= threshold_status:
            return ForecastDecision(
                0.0,
                0.0,
                "reached",
                "manufacturer_threshold",
                f"Manufacturer {target} status forces time-to-{target} to zero.",
            )

        statistical_hours = statistical.estimated_hours
        ml_hours = getattr(ml, f"time_to_{target}_hours")
        # A raw value can remain in the audit record, but no policy failure may
        # turn it into a primary estimate or an urgency recommendation.
        if ml.policy_withholding_reasons:
            return ForecastDecision(None, statistical_hours, "low", "unavailable",
                f"ML {target} forecast withheld by production/physical policy: " + ", ".join(ml.policy_withholding_reasons),
                withheld=True, withholding_reason=primary_reason(ml.policy_withholding_reasons),
                withholding_reasons=ml.policy_withholding_reasons)
        # Support is a property of the prediction, independent of whether the
        # model is presently approved for primary use.  Never turn an
        # out-of-support ML value into an actionable statistical fallback.
        if ml.target_support_violations.get(f"time_to_{target}", False):
            physical = ml.physically_invalid_targets.get(f"time_to_{target}", False)
            return ForecastDecision(
                None, statistical_hours, "low", "unavailable",
                f"ML {target} output is outside observed training target support"
                + (" and is physically invalid/non-actionable." if physical else "."),
                withheld=True, withholding_reason="target_support",
            )
        deployed_ml = ml.maturity_stage == "deployed" and ml_hours is not None
        if deployed_ml:
            if ml.outside_training_distribution:
                reason = (
                    f"ML {target} output is outside the training feature distribution "
                    f"({ml.outside_feature_fraction:.1%} of features outside the envelope)."
                )
                return ForecastDecision(
                    None,
                    statistical_hours,
                    "low",
                    "unavailable",
                    reason + " No sufficiently confident statistical fallback is available.",
                    withheld=True,
                    withholding_reason="distribution_check",
                )

            confidence = ml.confidence
            reason = f"ML primary {target} forecast: {ml.reason}"
            absolute_disagreement = relative_disagreement = None
            strong_disagreement = False
            conservative_alert = None
            if statistical_hours is not None:
                denominator = max(abs(ml_hours), abs(statistical_hours), 1e-6)
                absolute_disagreement = abs(ml_hours - statistical_hours)
                relative_disagreement = absolute_disagreement / denominator
                absolute_threshold = getattr(
                    self.config.ml, f"{target}_disagreement_absolute_hours"
                )
                strong_disagreement = bool(
                    absolute_disagreement > absolute_threshold
                    and relative_disagreement > self.config.ml.disagreement_relative_threshold
                )
                if strong_disagreement:
                    confidence = "low"
                    reason += (
                        f" Strong disagreement with statistical {target} forecast "
                        f"({ml_hours:.2f} h ML vs {statistical_hours:.2f} h statistical); "
                        f"absolute={absolute_disagreement:.2f} h, "
                        f"relative={relative_disagreement:.3f}. Neither estimate is "
                        "actionable until the disagreement is resolved."
                    )
                    if self.config.ml.withhold_on_severe_disagreement:
                        return ForecastDecision(
                            None, None, confidence, "unavailable", reason,
                            absolute_disagreement_hours=absolute_disagreement,
                            relative_disagreement=relative_disagreement,
                            strong_disagreement=True, withheld=True,
                            withholding_reason="disagreement",
                        )
            return ForecastDecision(
                ml_hours,
                conservative_alert,
                confidence,
                "ml",
                reason,
                absolute_disagreement_hours=absolute_disagreement,
                relative_disagreement=relative_disagreement,
                strong_disagreement=strong_disagreement,
            )

        if statistical_hours is not None:
            reason = statistical.reason
            if ml_hours is not None:
                reason += (
                    f" Experimental ML {target} estimate={ml_hours:.2f} h was not deployed; "
                    "the statistical baseline is retained for monitoring diagnostics."
                )
            if (not self.config.ml.statistical_fallback_actionable
                    or not self._minimum_final_confidence_allowed(statistical.confidence)):
                return ForecastDecision(
                    None,
                    statistical_hours,
                    statistical.confidence,
                    "statistical_monitoring",
                    reason + " Statistical estimate retained for diagnostics only; no approved actionable ML forecast is available.",
                    withheld=True,
                    withholding_reason=(
                        "missing_ml" if ml_hours is None else "statistical_confidence"
                    ),
                )
            return ForecastDecision(
                statistical_hours,
                statistical_hours,
                statistical.confidence,
                "statistical",
                reason,
            )

        if ml_hours is not None:
            return ForecastDecision(
                None,
                None,
                "experimental",
                "unavailable",
                f"Experimental ML {target} estimate={ml_hours:.2f} h is shown but not used because "
                "deployment criteria have not been met.",
                withheld=True,
                withholding_reason="missing_deployed_ml",
            )
        return ForecastDecision(
            None,
            None,
            statistical.confidence,
            "unavailable",
            statistical.reason,
            withheld=True,
            withholding_reason="missing_ml",
        )

    def _combine_prognostic_forecast(
        self,
        raw_status: Status,
        statistical: StatisticalForecast,
        forecast: PrognosticForecast,
        target: str,
    ) -> ForecastDecision:
        """Use the v6 model-based forecast as primary; legacy statistical ETA is audit-only."""
        threshold_status = Status.WARNING if target == "warning" else Status.CRITICAL
        if raw_status >= threshold_status:
            return ForecastDecision(
                0.0, 0.0, "reached", "manufacturer_threshold",
                f"Manufacturer {target} status forces time-to-{target} to zero.",
            )
        eta = getattr(forecast, f"time_to_{target}_hours")
        earliest = getattr(forecast, f"{target}_earliest_hours")
        statistical_hours = statistical.estimated_hours
        if forecast.withheld:
            return ForecastDecision(
                None, statistical_hours, forecast.confidence, "unavailable", forecast.reason,
                withheld=True,
                withholding_reason=primary_reason(forecast.withholding_reasons or ("prognostic_withheld",)),
                withholding_reasons=forecast.withholding_reasons or ("prognostic_withheld",),
            )
        absolute = relative = None
        if eta is not None and statistical_hours is not None:
            absolute = abs(eta - statistical_hours)
            relative = absolute / max(abs(eta), abs(statistical_hours), 1e-6)
        if forecast.confidence not in {"medium", "high", "reached"}:
            return ForecastDecision(
                None, earliest, forecast.confidence, "probabilistic_monitoring",
                forecast.reason + " The model-based trend is retained for monitoring but is not confident enough for ETA action.",
                absolute_disagreement_hours=absolute, relative_disagreement=relative,
                withheld=True, withholding_reason="prognostic_confidence",
                withholding_reasons=("prognostic_confidence",),
            )
        if eta is None:
            return ForecastDecision(
                None, earliest, forecast.confidence, "probabilistic_degradation",
                forecast.reason + f" Median {target} threshold crossing is beyond the configured horizon or the persistent slope is non-rising.",
                absolute_disagreement_hours=absolute, relative_disagreement=relative,
                withheld=False,
            )
        reason = forecast.reason
        if statistical_hours is not None:
            reason += (
                f" Legacy Kalman diagnostic={statistical_hours:.2f} h; v6 model-based ETA={eta:.2f} h. "
                "This disagreement is recorded but does not veto v6 because the legacy Kalman rate is no longer a primary policy input."
            )
        return ForecastDecision(
            eta, earliest, forecast.confidence, "probabilistic_degradation", reason,
            absolute_disagreement_hours=absolute, relative_disagreement=relative,
            strong_disagreement=False, withheld=False,
        )

    def _apply_status_constraints(self, raw_status: Status, forecast: MLForecast) -> MLForecast:
        warning = {hours: forecast.probability_warning.get(hours) for hours in self.config.ml.forecast_horizons_hours}
        critical = {hours: forecast.probability_critical.get(hours) for hours in self.config.ml.forecast_horizons_hours}
        warning_time = forecast.time_to_warning_hours
        critical_time = forecast.time_to_critical_hours
        if raw_status >= Status.WARNING:
            warning_time = 0.0
            warning = {hours: 1.0 for hours in warning}
        if raw_status >= Status.CRITICAL:
            critical_time = 0.0
            critical = {hours: 1.0 for hours in critical}
        return replace(
            forecast,
            time_to_warning_hours=warning_time,
            time_to_critical_hours=critical_time,
            probability_warning=warning,
            probability_critical=critical,
        )

    @staticmethod
    def _enforce_forecast_consistency(
        raw_status: Status,
        warning: ForecastDecision,
        critical: ForecastDecision,
    ) -> tuple[ForecastDecision, ForecastDecision]:
        if (
            raw_status == Status.NORMAL
            and warning.primary_hours is not None
            and critical.primary_hours is not None
            and warning.primary_hours > critical.primary_hours
        ):
            consistency_reason = (
                "Warning and critical primary forecasts are logically inconsistent; "
                "both primary forecasts were withheld without swapping or averaging."
            )
            return (
                replace(
                    warning,
                    primary_hours=None,
                    confidence="low",
                    reason=warning.reason + " " + consistency_reason,
                    withheld=True,
                    withholding_reason="logical_consistency",
                ),
                replace(
                    critical,
                    primary_hours=None,
                    confidence="low",
                    reason=critical.reason + " " + consistency_reason,
                    withheld=True,
                    withholding_reason="logical_consistency",
                ),
            )
        return warning, critical

    @staticmethod
    def _suppressed_statistical(target: str, reason: str) -> StatisticalForecast:
        return StatisticalForecast(None, None, "suppressed", reason, {}, target)

    def apply_ml_forecast(
        self, result: MonitorResult, forecast: MLForecast
    ) -> MonitorResult:
        """Apply the normal runtime policy to a causally generated offline row."""
        if self.config.prognostics.enabled:
            # v6 keeps supervised synthetic ML for research comparison only.
            # Attaching it must never replace the model-based prognostic output.
            return replace(result, ml_forecast=forecast)
        if result.lifecycle_state == "RESET_CANDIDATE" or result.withholding_reasons == (
            "reset_suppression",
        ):
            return result
        # Keep batched/offline application causally identical to process(): a
        # feature row with an immature required window is diagnostics-only.
        if any(value < 0.5 for name, value in result.features.items() if name.endswith("__window_mature")):
            return replace(result, ml_forecast=forecast, forecast_withheld=True,
                withholding_reasons=ordered_reasons(result.withholding_reasons + forecast.policy_withholding_reasons))
        warning_statistical = result.statistical_warning_forecast or self._suppressed_statistical(
            "warning", "Statistical warning forecast is unavailable."
        )
        critical_statistical = result.statistical_critical_forecast or self._suppressed_statistical(
            "critical", "Statistical critical forecast is unavailable."
        )
        ml_forecast = self._apply_status_constraints(result.raw_status, forecast)
        ml_forecast = replace(ml_forecast, recommendation_trigger_targets=(self._probability_trigger_targets(ml_forecast) if result.raw_status == Status.NORMAL and ml_forecast.production_policy_passed else ()))
        warning = self._combine_forecasts(
            result.raw_status, warning_statistical, ml_forecast, "warning"
        )
        critical = self._combine_forecasts(
            result.raw_status, critical_statistical, ml_forecast, "critical"
        )
        warning, critical = self._enforce_forecast_consistency(
            result.raw_status, warning, critical
        )
        confidence_rank = {
            "reached": 4, "high": 3, "medium": 2, "low": 1, "experimental": 0,
            "insufficient_history": 0, "no_rising_trend": 0,
            "beyond_horizon": 0, "unavailable": 0,
        }
        confidence = min(
            (warning.confidence, critical.confidence),
            key=lambda value: confidence_rank.get(value, 0),
        )
        final_warning = warning.primary_hours
        final_critical = critical.primary_hours
        conservative_warning = warning.conservative_alert_hours
        conservative_critical = critical.conservative_alert_hours
        if result.raw_status >= Status.WARNING:
            final_warning = conservative_warning = 0.0
        if result.raw_status >= Status.CRITICAL:
            final_warning = final_critical = 0.0
            conservative_warning = conservative_critical = 0.0
        withholding_reasons = ordered_reasons({
            reason for decision in (warning, critical) for reason in
            (decision.withholding_reasons or (decision.withholding_reason,))
            if decision.withheld and reason != "none"
        })
        reason = f"Warning: {warning.reason} Critical: {critical.reason}"
        urgency_decision = self._urgency(
            result.raw_status, final_warning, final_critical, ml_forecast, confidence,
            primary_sources=(warning.primary_source, critical.primary_source),
            forecast_withheld=bool(warning.withheld or critical.withheld),
            feature_available=result.feature_available,
        )
        urgency = urgency_decision.urgency
        ml_forecast = replace(ml_forecast, probability_threshold_crossings=(self._probability_trigger_targets(ml_forecast) if result.raw_status == Status.NORMAL and ml_forecast.production_policy_passed else ()), recommendation_trigger_probability_targets=urgency_decision.probability_trigger_targets, recommendation_trigger_eta_targets=urgency_decision.eta_trigger_targets, recommendation_trigger_rules=urgency_decision.trigger_rules, recommendation_trigger_targets=tuple(sorted(set(urgency_decision.probability_trigger_targets + urgency_decision.eta_trigger_targets))))
        guardrail_result = (
            "withheld" if warning.withheld or critical.withheld
            else "fallback" if "statistical_fallback" in {warning.primary_source, critical.primary_source}
            else "passed"
        )
        metadata = self.audit_model_metadata or (
            self.ml.metadata if self.ml and self.ml.metadata else {}
        )
        updated = replace(
            result,
            ml_forecast=ml_forecast,
            final_time_to_warning_hours=final_warning,
            final_time_to_critical_hours=final_critical,
            conservative_alert_time_to_warning_hours=conservative_warning,
            conservative_alert_time_to_critical_hours=conservative_critical,
            warning_primary_source=warning.primary_source,
            critical_primary_source=critical.primary_source,
            warning_absolute_disagreement_hours=warning.absolute_disagreement_hours,
            warning_relative_disagreement=warning.relative_disagreement,
            critical_absolute_disagreement_hours=critical.absolute_disagreement_hours,
            critical_relative_disagreement=critical.relative_disagreement,
            disagreement_trigger_targets=tuple(
                target for target, decision in (("warning", warning), ("critical", critical))
                if decision.strong_disagreement
            ),
            ml_outside_training_distribution=ml_forecast.outside_training_distribution,
            ml_outside_feature_fraction=ml_forecast.outside_feature_fraction,
            forecast_withheld=bool(warning.withheld or critical.withheld),
            withholding_reasons=withholding_reasons,
            recommendation_trigger_targets=ml_forecast.recommendation_trigger_targets,
            probability_threshold_crossings=ml_forecast.probability_threshold_crossings,
            recommendation_trigger_probability_targets=ml_forecast.recommendation_trigger_probability_targets,
            recommendation_trigger_eta_targets=ml_forecast.recommendation_trigger_eta_targets,
            recommendation_trigger_rules=ml_forecast.recommendation_trigger_rules,
            recommendation_source=urgency_decision.recommendation_source,
            recommendation_actionable=urgency_decision.actionable,
            final_forecast_hours=final_critical,
            forecast_confidence=confidence.upper(),
            forecast_reason=reason,
            maintenance_urgency=urgency,
            target_support_violation=bool(
                ml_forecast.beyond_training_duration_support
                or any(ml_forecast.target_support_violations.values())
            ),
            model_stage=str(metadata.get("model_stage", result.model_stage)),
            model_version=str(ml_forecast.model_version or result.model_version),
            guardrail_result=guardrail_result,
            refusal_or_fallback_reason=(reason if guardrail_result != "passed" else ""),
        )
        return self._apply_anomaly_policy(updated)

    def _apply_anomaly_policy(self, result: MonitorResult) -> MonitorResult:
        """Apply model usability without modifying manufacturer safety outputs."""
        anomaly = result.anomaly
        if anomaly.model_action == ModelAction.USE_NORMALLY:
            return result
        reason_code = f"anomaly:{anomaly.model_action.value.lower()}"
        reason = (
            f"Anomaly policy {anomaly.model_action.value}: "
            + "; ".join(anomaly.supporting_evidence or (anomaly.causal_decision,))
        )
        if not anomaly.use_for_prediction:
            warning = None
            critical = None
            conservative_warning = None
            conservative_critical = None
            # Immediate manufacturer threshold behavior is never weakened.
            if result.raw_status >= Status.WARNING:
                warning = conservative_warning = 0.0
            if result.raw_status >= Status.CRITICAL:
                warning = critical = conservative_warning = conservative_critical = 0.0
            return replace(
                result,
                final_time_to_warning_hours=warning,
                final_time_to_critical_hours=critical,
                conservative_alert_time_to_warning_hours=conservative_warning,
                conservative_alert_time_to_critical_hours=conservative_critical,
                final_forecast_hours=critical,
                forecast_withheld=True,
                withholding_reasons=ordered_reasons(result.withholding_reasons + (reason_code,)),
                forecast_confidence="UNAVAILABLE",
                forecast_reason=reason,
                maintenance_urgency=(
                    result.maintenance_urgency if result.raw_status >= Status.WARNING
                    else "MONITOR CLOSELY — INSUFFICIENT FORECAST CONFIDENCE"
                ),
                recommendation_trigger_targets=(),
                probability_threshold_crossings=(),
                recommendation_actionable=False,
                recommendation_trigger_probability_targets=(),
                recommendation_trigger_eta_targets=(),
                recommendation_trigger_rules=(),
                ml_forecast=replace(result.ml_forecast, recommendation_trigger_targets=(), recommendation_trigger_probability_targets=(), recommendation_trigger_eta_targets=(), recommendation_trigger_rules=()),
                guardrail_result="withheld",
                refusal_or_fallback_reason=reason,
            )
        confidence = result.forecast_confidence
        reduced_confidence = anomaly.confidence_multiplier < 1.0
        if reduced_confidence:
            confidence = "LOW"

        # A physically valid, origin-uncertain machine/regime change is allowed
        # to retain prognostic values for situational awareness.  It must not,
        # however, turn a low-confidence forecast into an actionable maintenance
        # recommendation while manufacturer safety remains NORMAL.
        recommendation_actionable = result.recommendation_actionable
        maintenance_urgency = result.maintenance_urgency
        recommendation_targets = result.recommendation_trigger_targets
        recommendation_probability_targets = result.recommendation_trigger_probability_targets
        recommendation_eta_targets = result.recommendation_trigger_eta_targets
        recommendation_rules = result.recommendation_trigger_rules
        ml_forecast = result.ml_forecast
        if reduced_confidence and result.raw_status == Status.NORMAL:
            recommendation_actionable = False
            maintenance_urgency = "MONITOR CLOSELY — INSUFFICIENT FORECAST CONFIDENCE"
            recommendation_targets = ()
            recommendation_probability_targets = ()
            recommendation_eta_targets = ()
            recommendation_rules = ()
            ml_forecast = replace(
                result.ml_forecast,
                recommendation_trigger_targets=(),
                recommendation_trigger_probability_targets=(),
                recommendation_trigger_eta_targets=(),
                recommendation_trigger_rules=(),
            )

        return replace(
            result,
            forecast_confidence=confidence,
            forecast_reason=f"{result.forecast_reason} {reason}".strip(),
            refusal_or_fallback_reason=reason,
            guardrail_result="qualified",
            maintenance_urgency=maintenance_urgency,
            recommendation_actionable=recommendation_actionable,
            recommendation_trigger_targets=recommendation_targets,
            recommendation_trigger_probability_targets=recommendation_probability_targets,
            recommendation_trigger_eta_targets=recommendation_eta_targets,
            recommendation_trigger_rules=recommendation_rules,
            ml_forecast=ml_forecast,
        )

    def _reset_processing_state(
        self, reading: SensorReading, values: dict[str, float], raw_status: Status,
        *, commit_feature_history: bool = True,
    ) -> dict[str, SmoothedSignal]:
        """Seed clean trend, feature, and status state after a confirmed reset."""
        if commit_feature_history:
            signals = self.smoother.reset(values)
            assert signals is not None
        else:
            self.smoother.reset()
            signals = self.smoother.preview(values, 1.0)
        self.timestamps.clear()
        if commit_feature_history:
            self.timestamps.append(reading.timestamp)
        self.raw_history = {
            key: deque(
                [values[key]] if commit_feature_history else [],
                maxlen=self.config.monitoring.history_size,
            )
            for key in self.config.sensors
        }
        self.status_policy.reset(raw_status)
        self.event_policy.reset()
        self.features.reset()
        self.anomaly_detector.reset()
        self.degradation_kalman = self._new_degradation_filter()
        self.prognostic_probability_state.clear()
        self.prognostic_probability_active.clear()
        self.prognostic_probability_release_state.clear()
        return signals

    def _clear_processing_state(self) -> None:
        """Clear every stateful historical component at a known offline boundary."""
        self.smoother.reset()
        self.timestamps.clear()
        self.raw_history = {
            key: deque(maxlen=self.config.monitoring.history_size)
            for key in self.config.sensors
        }
        self.status_policy.reset()
        self.event_policy.reset()
        self.features.reset()
        self.anomaly_detector.reset()
        self.degradation_kalman = self._new_degradation_filter()
        self.prognostic_probability_state.clear()
        self.prognostic_probability_active.clear()
        self.prognostic_probability_release_state.clear()

    def _urgency(
        self,
        raw_status: Status,
        final_warning: float | None,
        final_critical: float | None,
        ml: MLForecast,
        confidence: str,
        *,
        primary_sources: tuple[str, str] = ("unavailable", "unavailable"),
        forecast_withheld: bool = False,
        feature_available: bool = True,
    ) -> UrgencyDecision:
        if raw_status == Status.CRITICAL:
            return UrgencyDecision("IMMEDIATE_MAINTENANCE_REQUIRED", False, "manufacturer")
        if raw_status == Status.WARNING:
            return UrgencyDecision("MONITOR CLOSELY — WARNING PRESENT", False, "manufacturer")
        # Future recommendations require an approved, supported ML forecast.
        # Statistical estimates are diagnostic-only unless explicitly enabled.
        trusted = (
            ml.production_policy_passed
            and feature_available
            and not ml.outside_training_distribution
            and ml.physical_validation_passed
            and self._minimum_final_confidence_allowed(ml.prediction_confidence.lower())
        )
        if not trusted:
            return UrgencyDecision("MONITOR CLOSELY — INSUFFICIENT FORECAST CONFIDENCE", False)
        critical_6 = ml.probability_critical.get(6)
        critical_24 = ml.probability_critical.get(24)
        warning_12 = ml.probability_warning.get(12)
        # Trained per-target thresholds, never configuration defaults, decide
        # whether a probability target has crossed its action boundary.
        critical_6_threshold = ml.selected_thresholds.get("probability_critical_6h")
        critical_24_threshold = ml.selected_thresholds.get("probability_critical_24h")
        warning_12_threshold = ml.selected_thresholds.get("probability_warning_12h")
        def crosses(target: str, value: float | None, threshold: float | None) -> bool:
            return bool(
                ml.target_eligibility.get(target) is True
                and value is not None
                and threshold is not None
                and value >= threshold
            )

        if crosses("probability_critical_6h", critical_6, critical_6_threshold):
            return UrgencyDecision("MAINTENANCE_RECOMMENDED_SOON", True, "ml", ("probability_critical_6h",), (), ("critical_probability_6h",))
        critical_eta_ok = ml.eta_target_evidence.get("time_to_critical", {}).get("target_eligible") is True
        warning_eta_ok = ml.eta_target_evidence.get("time_to_warning", {}).get("target_eligible") is True
        eta_actionable = not forecast_withheld and "ml" in primary_sources
        if eta_actionable and critical_eta_ok and final_critical is not None and final_critical <= 6.0:
            return UrgencyDecision("MAINTENANCE_RECOMMENDED_SOON", True, "ml", (), ("time_to_critical",), ("critical_eta_soon",))
        if crosses("probability_critical_24h", critical_24, critical_24_threshold):
            return UrgencyDecision("PLAN_MAINTENANCE", True, "ml", ("probability_critical_24h",), (), ("critical_probability_24h",))
        if crosses("probability_warning_12h", warning_12, warning_12_threshold):
            return UrgencyDecision("PLAN_INSPECTION", True, "ml", ("probability_warning_12h",), (), ("warning_probability_12h",))
        if eta_actionable and warning_eta_ok and final_warning is not None and final_warning <= 12.0:
            return UrgencyDecision("PLAN_INSPECTION", True, "ml", (), ("time_to_warning",), ("warning_eta_plan",))
        if raw_status == Status.WARNING:
            return UrgencyDecision("MONITOR CLOSELY — WARNING PRESENT", False)
        if confidence.lower() in {"low", "experimental", "suppressed", "unavailable"}:
            return UrgencyDecision("MONITOR CLOSELY — INSUFFICIENT FORECAST CONFIDENCE", False)
        return UrgencyDecision("NORMAL_MONITORING", False)

    def _persistent_prognostic_crossings(
        self, timestamp: datetime, forecast: PrognosticForecast
    ) -> tuple[str, ...]:
        """Return operational crossings with asymmetric activation/release hysteresis.

        Activation keeps the existing above-threshold persistence. Once active, a
        target is released only after the probability remains below a lower
        boundary for the configured release-persistence period. This prevents
        short probability dips from fragmenting one condition into many alerts.
        """
        if forecast.withheld:
            self.prognostic_probability_state.clear()
            self.prognostic_probability_active.clear()
            self.prognostic_probability_release_state.clear()
            return ()

        active: list[str] = []
        default_persistence = self.config.prognostics.probability_persistence_minutes
        release_fraction = self.config.prognostics.probability_release_fraction
        release_persistence = self.config.prognostics.probability_release_persistence_minutes
        configured_targets = set(forecast.selected_thresholds)

        for target in set(self.prognostic_probability_state) - configured_targets:
            self.prognostic_probability_state.pop(target, None)
            self.prognostic_probability_release_state.pop(target, None)
            self.prognostic_probability_active.discard(target)

        for target, threshold in forecast.selected_thresholds.items():
            value = forecast.probability_target_value(target)
            if value is None:
                self.prognostic_probability_state.pop(target, None)
                self.prognostic_probability_release_state.pop(target, None)
                self.prognostic_probability_active.discard(target)
                continue

            if target in self.prognostic_probability_active:
                release_boundary = threshold * release_fraction
                if value < release_boundary:
                    release_started = self.prognostic_probability_release_state.setdefault(target, timestamp)
                    release_elapsed = max(0.0, (timestamp - release_started).total_seconds() / 60.0)
                    if release_elapsed >= release_persistence:
                        self.prognostic_probability_active.discard(target)
                        self.prognostic_probability_release_state.pop(target, None)
                        self.prognostic_probability_state.pop(target, None)
                        continue
                else:
                    self.prognostic_probability_release_state.pop(target, None)
                active.append(target)
                continue

            if value >= threshold:
                if target.startswith("probability_warning_"):
                    persistence = self.config.prognostics.warning_probability_persistence_minutes
                elif target.startswith("probability_critical_"):
                    persistence = self.config.prognostics.critical_probability_persistence_minutes
                else:
                    persistence = default_persistence
                started = self.prognostic_probability_state.setdefault(target, timestamp)
                elapsed_minutes = max(0.0, (timestamp - started).total_seconds() / 60.0)
                if elapsed_minutes >= persistence:
                    self.prognostic_probability_active.add(target)
                    self.prognostic_probability_release_state.pop(target, None)
                    active.append(target)
            else:
                self.prognostic_probability_state.pop(target, None)
                self.prognostic_probability_release_state.pop(target, None)

        return tuple(sorted(active))

    def _urgency_prognostic(
        self,
        raw_status: Status,
        final_warning: float | None,
        final_critical: float | None,
        forecast: PrognosticForecast,
        confidence: str,
        *,
        persistent_crossings: tuple[str, ...],
        forecast_withheld: bool,
        feature_available: bool,
    ) -> UrgencyDecision:
        if raw_status == Status.CRITICAL:
            return UrgencyDecision("IMMEDIATE_MAINTENANCE_REQUIRED", False, "manufacturer")
        if raw_status == Status.WARNING:
            return UrgencyDecision("MONITOR CLOSELY — WARNING PRESENT", False, "manufacturer")
        if not self.config.prognostics.advisory_recommendations_enabled:
            return UrgencyDecision("NORMAL_MONITORING", False, "probabilistic_degradation")
        if forecast_withheld or not feature_available or confidence.lower() not in {"medium", "high", "reached"}:
            return UrgencyDecision("MONITOR CLOSELY — INSUFFICIENT FORECAST CONFIDENCE", False, "probabilistic_degradation")
        def crosses(target: str) -> bool:
            return target in persistent_crossings
        actionable = bool(self.config.prognostics.automated_action_enabled)
        if crosses("probability_critical_6h"):
            return UrgencyDecision("MAINTENANCE_RECOMMENDED_SOON", actionable, "probabilistic_degradation", ("probability_critical_6h",), (), ("critical_probability_6h",))
        critical_6 = forecast.probability_critical.get(6)
        if final_critical is not None and final_critical <= 6.0 and critical_6 is not None and critical_6 >= self.config.prognostics.eta_action_min_probability:
            return UrgencyDecision("MAINTENANCE_RECOMMENDED_SOON", actionable, "probabilistic_degradation", (), ("time_to_critical",), ("critical_eta_soon",))
        if crosses("probability_critical_24h"):
            return UrgencyDecision("PLAN_MAINTENANCE", actionable, "probabilistic_degradation", ("probability_critical_24h",), (), ("critical_probability_24h",))
        if crosses("probability_warning_12h"):
            return UrgencyDecision("PLAN_INSPECTION", actionable, "probabilistic_degradation", ("probability_warning_12h",), (), ("warning_probability_12h",))
        warning_12 = forecast.probability_warning.get(12)
        if final_warning is not None and final_warning <= 12.0 and warning_12 is not None and warning_12 >= self.config.prognostics.eta_action_min_probability:
            return UrgencyDecision("PLAN_INSPECTION", actionable, "probabilistic_degradation", (), ("time_to_warning",), ("warning_eta_plan",))
        return UrgencyDecision("NORMAL_MONITORING", False, "probabilistic_degradation")

    def process(
        self,
        reading: SensorReading,
        *,
        compute_features: bool = True,
        interpolated: bool = False,
        source_sampling_interval_seconds: float = 0.0,
        effective_resampling_interval_seconds: float = 0.0,
        input_features_available: bool = True,
    ) -> MonitorResult:
        first_observation = self.last_timestamp is None
        values = {
            key: validate_value(reading.values()[key], sensor)
            for key, sensor in self.config.sensors.items()
        }
        if self.last_timestamp is not None and reading.timestamp <= self.last_timestamp:
            raise ValueError("ConditionMonitor requires strictly increasing timestamps")
        planned_boundary = bool(
            self.lifecycle_plan and self.lifecycle_plan.is_boundary(reading.timestamp)
        )
        if planned_boundary:
            self._clear_processing_state()
        dt_seconds = 1.0 if self.last_timestamp is None or planned_boundary else max(
            (reading.timestamp - self.last_timestamp).total_seconds(), 1e-6
        )
        self.last_timestamp = reading.timestamp
        raw_status = max(
            status_for_value(values[key], sensor, self.config)
            for key, sensor in self.config.sensors.items()
        )
        anomaly = self.anomaly_detector.detect(
            reading,
            raw_status,
            interpolated=interpolated,
            source_sampling_interval_seconds=(source_sampling_interval_seconds or None),
            input_features_available=input_features_available,
        )
        commit_feature_history = (
            anomaly.feature_history_action == FeatureHistoryAction.COMMIT_TO_FEATURE_HISTORY
        )
        signals = (
            self.smoother.update(values, dt_seconds)
            if commit_feature_history
            else self.smoother.preview(values, dt_seconds)
        )
        if commit_feature_history:
            self.timestamps.append(reading.timestamp)
            for key in self.config.sensors:
                self.raw_history[key].append(values[key])

        assessments, sensor_statuses, raw_status, reasons = self._assess(values, signals)
        effective_status = self.status_policy.update(raw_status)
        event_status = self.event_policy.update(reading.timestamp, raw_status)
        if event_status >= Status.CRITICAL:
            anomaly = replace(
                anomaly,
                quality_status=QualityStatus.CONFIRMED_MACHINE_CRITICAL,
                confirmed_timestamp=anomaly.confirmed_timestamp or reading.timestamp,
            )
        maximum_severity = max(assessment.severity for assessment in assessments.values())
        total_weight = sum(sensor.weight for sensor in self.config.sensors.values())
        weighted_severity = sum(
            assessments[key].severity * self.config.sensors[key].weight for key in assessments
        ) / total_weight
        estimated_degradation, degradation_rate = self.degradation_kalman.update(
            maximum_severity, dt_seconds
        )

        if self.lifecycle_plan is None:
            lifecycle = self.lifecycle.update(
                timestamp=reading.timestamp,
                status=effective_status,
                event_status=event_status,
                smoothed_values={key: signals[key].kalman_level for key in signals},
                maximum_severity=maximum_severity,
            )
        else:
            lifecycle = self.lifecycle_plan.snapshot(reading.timestamp)
        reset_confirmed = lifecycle.completed_lifecycle is not None
        if reset_confirmed and self.lifecycle_plan is None:
            signals = self._reset_processing_state(
                reading, values, raw_status,
                commit_feature_history=commit_feature_history,
            )
            assessments, sensor_statuses, raw_status, reasons = self._assess(values, signals)
            effective_status = raw_status
            event_status = raw_status
            maximum_severity = max(assessment.severity for assessment in assessments.values())
            weighted_severity = sum(
                assessments[key].severity * self.config.sensors[key].weight for key in assessments
            ) / total_weight
            estimated_degradation, degradation_rate = self.degradation_kalman.update(maximum_severity, 1.0)

        feature_values = (
            self.features.update(
                timestamp=reading.timestamp, raw_values=values, smoothed=signals,
                sensor_statuses=sensor_statuses, overall_status=effective_status,
                elapsed_lifecycle_hours=lifecycle.elapsed_lifecycle_hours,
                compute=compute_features and anomaly.use_for_features,
                history_action=anomaly.feature_history_action,
            )
            if self.enable_features else {}
        )

        available_history = max(
            (value for name, value in feature_values.items()
             if name.endswith("__available_history_duration_seconds")),
            default=0.0,
        )
        feature_available = input_features_available and anomaly.use_for_features and all(
            value >= 0.5 for name, value in feature_values.items()
            if name.endswith("__feature_available")
        )

        suppression_reason = ""
        if lifecycle.lifecycle_state == "RESET_CANDIDATE":
            suppression_reason = "Reset confirmation is in progress."
        elif reset_confirmed:
            suppression_reason = "Reset confirmation is complete; clean lifecycle history is being established."

        # Legacy supervised ML remains available only as a research/audit output.
        runtime_ml = self.ml.predict(feature_values) if self.ml is not None else MLForecast()
        readiness_reasons: list[str] = []
        if not input_features_available or not anomaly.use_for_features:
            readiness_reasons.append("invalid_sensor_state")
        if readiness_reasons:
            runtime_ml = replace(
                runtime_ml,
                policy_withholding_reasons=ordered_reasons(
                    runtime_ml.policy_withholding_reasons + tuple(readiness_reasons)
                ),
                feature_readiness_reasons=ordered_reasons(readiness_reasons),
            )
        ml_forecast = self._apply_status_constraints(raw_status, runtime_ml)

        if suppression_reason:
            warning_statistical = self._suppressed_statistical("warning", suppression_reason)
            critical_statistical = self._suppressed_statistical("critical", suppression_reason)
            prognostic_forecast = PrognosticForecast(
                probability_warning={hours: None for hours in self.config.prognostics.forecast_horizons_hours},
                probability_critical={hours: None for hours in self.config.prognostics.forecast_horizons_hours},
                confidence="unavailable",
                reason=suppression_reason,
                method_version=PROGNOSTIC_METHOD_VERSION,
                withheld=True,
                withholding_reasons=("reset_suppression",),
            )
            final_warning = final_critical = None
            conservative_warning = conservative_critical = None
            warning_source = critical_source = "suppressed"
            warning_absolute = warning_relative = None
            critical_absolute = critical_relative = None
            disagreement_targets: tuple[str, ...] = ()
            forecast_withheld = True
            withholding_reasons = ("reset_suppression",)
            forecast_confidence = "UNAVAILABLE"
            forecast_reason = suppression_reason
        else:
            warning_statistical = statistical_time_to_warning(
                self.config, signals, sensor_statuses, len(self.timestamps)
            )
            critical_statistical = statistical_time_to_critical(
                self.config, signals, sensor_statuses, len(self.timestamps)
            )
            prognostic_forecast = self.prognostics.predict(
                values=values,
                signals=signals,
                features=feature_values,
                sensor_statuses=sensor_statuses,
                raw_status=raw_status,
                anomaly=anomaly,
                available_history_seconds=float(available_history),
            )
            warning_decision = self._combine_prognostic_forecast(
                raw_status, warning_statistical, prognostic_forecast, "warning"
            )
            critical_decision = self._combine_prognostic_forecast(
                raw_status, critical_statistical, prognostic_forecast, "critical"
            )
            warning_decision, critical_decision = self._enforce_forecast_consistency(
                raw_status, warning_decision, critical_decision
            )
            final_warning = warning_decision.primary_hours
            final_critical = critical_decision.primary_hours
            conservative_warning = warning_decision.conservative_alert_hours
            conservative_critical = critical_decision.conservative_alert_hours
            warning_source = warning_decision.primary_source
            critical_source = critical_decision.primary_source
            warning_absolute = warning_decision.absolute_disagreement_hours
            warning_relative = warning_decision.relative_disagreement
            critical_absolute = critical_decision.absolute_disagreement_hours
            critical_relative = critical_decision.relative_disagreement
            disagreement_targets = ()  # Legacy statistical disagreement is diagnostic-only in v6.
            withholding_reasons = ordered_reasons({
                reason for decision in (warning_decision, critical_decision) for reason in
                (decision.withholding_reasons or (decision.withholding_reason,))
                if decision.withheld and reason != "none"
            })
            forecast_withheld = bool(warning_decision.withheld or critical_decision.withheld)
            confidence_rank = {
                "reached": 4, "high": 3, "medium": 2, "low": 1,
                "insufficient_history": 0, "unavailable": 0,
            }
            forecast_confidence = min(
                (warning_decision.confidence, critical_decision.confidence),
                key=lambda value: confidence_rank.get(value, 0),
            )
            forecast_reason = f"Warning: {warning_decision.reason} Critical: {critical_decision.reason}"

        # Raw manufacturer thresholds remain the final safety authority.
        if raw_status >= Status.WARNING and lifecycle.lifecycle_state != "RESET_CANDIDATE":
            final_warning = 0.0
            conservative_warning = 0.0
        if raw_status >= Status.CRITICAL:
            final_warning = 0.0
            final_critical = 0.0
            conservative_warning = 0.0
            conservative_critical = 0.0

        if not reasons:
            reasons.append("All raw sensor readings are within configured manufacturer limits.")
        if effective_status != raw_status:
            reasons.append(
                f"Operational status is stabilized at {effective_status.name}; "
                f"the immediate raw status is {raw_status.name}."
            )
        if lifecycle.reset_confidence:
            reasons.append(f"Inferred lifecycle reset {lifecycle.reset_confidence}: {lifecycle.reset_reason}")

        worst_key = self._worst_sensor(assessments)
        persistent_probability_crossings = self._persistent_prognostic_crossings(
            reading.timestamp, prognostic_forecast
        )
        urgency_decision = self._urgency_prognostic(
            raw_status, final_warning, final_critical, prognostic_forecast, forecast_confidence,
            persistent_crossings=persistent_probability_crossings,
            forecast_withheld=forecast_withheld,
            feature_available=feature_available,
        )
        urgency = urgency_decision.urgency
        recommendation_targets = tuple(sorted(set(
            urgency_decision.probability_trigger_targets + urgency_decision.eta_trigger_targets
        )))
        forecast_confidence = forecast_confidence.upper()
        guardrail_result = "withheld" if forecast_withheld else "passed"
        result = MonitorResult(
            timestamp=reading.timestamp,
            raw_status=raw_status,
            effective_status=effective_status,
            health_percent=health_from_severity(estimated_degradation),
            worst_sensor=worst_key,
            worst_sensor_display=self.config.sensors[worst_key].display_name,
            maximum_severity=maximum_severity,
            weighted_severity=weighted_severity,
            estimated_degradation=estimated_degradation,
            degradation_rate_per_hour=degradation_rate * 3600.0,
            assessments=assessments,
            reasons=reasons,
            lifecycle_id=lifecycle.lifecycle_id,
            lifecycle_state=lifecycle.lifecycle_state,
            elapsed_lifecycle_hours=lifecycle.elapsed_lifecycle_hours,
            reset_confidence=lifecycle.reset_confidence,
            reset_reason=lifecycle.reset_reason,
            statistical_forecast=critical_statistical,
            statistical_warning_forecast=warning_statistical,
            statistical_critical_forecast=critical_statistical,
            prognostic_forecast=prognostic_forecast,
            ml_forecast=ml_forecast,
            final_time_to_warning_hours=final_warning,
            final_time_to_critical_hours=final_critical,
            conservative_alert_time_to_warning_hours=conservative_warning,
            conservative_alert_time_to_critical_hours=conservative_critical,
            warning_primary_source=warning_source,
            critical_primary_source=critical_source,
            warning_absolute_disagreement_hours=warning_absolute,
            warning_relative_disagreement=warning_relative,
            critical_absolute_disagreement_hours=critical_absolute,
            critical_relative_disagreement=critical_relative,
            disagreement_trigger_targets=disagreement_targets,
            ml_outside_training_distribution=False,
            ml_outside_feature_fraction=0.0,
            forecast_withheld=forecast_withheld,
            withholding_reasons=withholding_reasons,
            recommendation_trigger_targets=recommendation_targets,
            probability_threshold_crossings=persistent_probability_crossings,
            recommendation_trigger_probability_targets=urgency_decision.probability_trigger_targets,
            recommendation_trigger_eta_targets=urgency_decision.eta_trigger_targets,
            recommendation_trigger_rules=urgency_decision.trigger_rules,
            recommendation_source=urgency_decision.recommendation_source,
            recommendation_actionable=urgency_decision.actionable,
            final_forecast_hours=final_critical,
            forecast_confidence=forecast_confidence,
            forecast_reason=forecast_reason,
            maintenance_urgency=urgency,
            features=feature_values,
            completed_lifecycle=lifecycle.completed_lifecycle,
            event_status=event_status,
            sampling_gap_seconds=float(
                source_sampling_interval_seconds
                if source_sampling_interval_seconds > 0
                else 0.0 if first_observation else dt_seconds
            ),
            interpolated=interpolated,
            feature_available=feature_available,
            target_support_violation=False,
            data_domain=self.data_domain,
            source_sampling_interval_seconds=(
                float(source_sampling_interval_seconds)
                if source_sampling_interval_seconds > 0
                else None if first_observation else float(dt_seconds)
            ),
            effective_resampling_interval_seconds=float(
                effective_resampling_interval_seconds or self.config.target_sampling_interval_seconds
            ),
            available_history_duration_seconds=float(available_history),
            dataset_hash=self.dataset_hash,
            generator_version=self.generator_version,
            model_stage="model_based_prognostics",
            model_version=prognostic_forecast.method_version,
            guardrail_result=guardrail_result,
            refusal_or_fallback_reason=(forecast_reason if guardrail_result != "passed" else ""),
            anomaly=anomaly,
        )
        return self._apply_anomaly_policy(result)
