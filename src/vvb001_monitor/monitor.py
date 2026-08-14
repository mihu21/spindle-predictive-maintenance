from __future__ import annotations

from .features import FeatureEngine
from .models import ProcessedReading, SourceRecord, ValidationResult
from .predictor import VVB001Predictor
from .rul import VVB001RULEstimator
from .rul_ml import VVB001LearnedRULEstimator
from .sensor_quality import SensorQualityGuard
from .validation import VVB001Validator


class VVB001Monitor:
    def __init__(
        self,
        validator: VVB001Validator,
        features: FeatureEngine,
        predictor: VVB001Predictor | None = None,
        sensor_quality_guard: SensorQualityGuard | None = None,
        rul_estimator: VVB001RULEstimator | None = None,
        enable_rul: bool = True,
    ) -> None:
        self.validator = validator
        self.features = features
        self.predictor = predictor
        self.sensor_quality_guard = sensor_quality_guard
        if not enable_rul:
            self.rul_estimator = None
        elif rul_estimator is not None:
            self.rul_estimator = rul_estimator
        elif predictor is not None and getattr(predictor, "rul_model_artifact", None):
            self.rul_estimator = VVB001LearnedRULEstimator(predictor.rul_model_artifact)
        elif predictor is not None and getattr(predictor.model, "score_boundaries", None):
            # Backward compatibility for pre-v2 model bundles. Newly trained bundles use the
            # supervised learned RUL estimator above.
            warning_boundary, critical_boundary = predictor.model.score_boundaries
            self.rul_estimator = VVB001RULEstimator(warning_boundary, critical_boundary)
        else:
            self.rul_estimator = None

    def reset_machine(self, machine_key: str) -> None:
        self.features.reset_machine(machine_key)
        if self.predictor is not None:
            self.predictor.reset_machine(machine_key)
        if self.sensor_quality_guard is not None:
            self.sensor_quality_guard.reset_machine(machine_key)
        if self.rul_estimator is not None:
            self.rul_estimator.reset_machine(machine_key)

    def process(self, record: SourceRecord) -> tuple[ValidationResult, ProcessedReading | None]:
        validation = self.validator.validate_record(record)
        if not validation.valid or validation.reading is None:
            return validation, None
        reading = validation.reading
        if self.sensor_quality_guard is not None:
            quality = self.sensor_quality_guard.process(reading)
            feature_reading = quality.feature_reading
        else:
            quality = None
            feature_reading = reading
        feature_values = self.features.process(feature_reading)
        prediction = (
            self.predictor.predict(
                feature_values,
                hold_state=bool(quality and quality.hold_prediction_state),
                state_action=quality.prediction_state_action if quality else "LIVE",
                allow_thermal_hysteresis_bypass=not bool(quality and quality.held_sensors),
            )
            if self.predictor is not None
            else None
        )
        rul = None
        if prediction is not None and self.rul_estimator is not None:
            rul = self.rul_estimator.update(
                machine_key=reading.machine_key,
                timestamp=reading.timestamp,
                degradation_score=prediction.degradation_score,
                predicted_status=prediction.status,
                features=feature_values,
                trusted=not prediction.state_held,
                state_reset=bool(feature_values.get("state_reset")),
                raw_safety_override=bool(quality and quality.extreme_raw_override),
            )
        return validation, ProcessedReading(
            reading=reading,
            quality_status=validation.status,
            quality_reasons=validation.reasons,
            features=feature_values,
            predicted_status=prediction.status if prediction else None,
            prediction_probabilities=prediction.probabilities if prediction else None,
            degradation_score=prediction.degradation_score if prediction else None,
            prediction_policy_reason=prediction.policy_reason if prediction else None,
            prediction_state_source=prediction.state_source if prediction else "NOT_EVALUATED",
            prediction_state_held=prediction.state_held if prediction else False,
            sensor_quality_status=quality.status if quality else "NOT_EVALUATED",
            sensor_quality_reasons=quality.reasons if quality else (),
            sensor_quality_held_sensors=quality.held_sensors if quality else (),
            sensor_quality_extreme_raw_override=quality.extreme_raw_override if quality else False,
            estimated_hours_to_warning=rul.estimated_hours_to_warning if rul else None,
            warning_rul_lower_hours=rul.warning_lower_hours if rul else None,
            warning_rul_upper_hours=rul.warning_upper_hours if rul else None,
            estimated_hours_to_critical=rul.estimated_hours_to_critical if rul else None,
            critical_rul_lower_hours=rul.critical_lower_hours if rul else None,
            critical_rul_upper_hours=rul.critical_upper_hours if rul else None,
            rul_reliability=rul.reliability if rul else "UNAVAILABLE",
            rul_reason=rul.reason if rul else None,
            rul_trend_score_per_hour=rul.trend_score_per_hour if rul else None,
            rul_trend_r2=rul.trend_r2 if rul else None,
            rul_history_hours=rul.history_hours if rul else 0.0,
            rul_trusted_points=rul.trusted_points if rul else 0,
            rul_state_source=rul.state_source if rul else "NOT_EVALUATED",
            rul_method=rul.method if rul else "unavailable",
            rul_calibration_method=rul.calibration_method if rul else None,
            rul_calibration_bucket=rul.calibration_bucket if rul else None,
            rul_forecastability_state=rul.forecastability_state if rul else "RUL_UNAVAILABLE",
            rul_forecastability_score=rul.forecastability_score if rul else None,
            rul_serviceable_intent=rul.serviceable_intent if rul else False,
            rul_hard_eligible=rul.hard_eligible if rul else False,
            rul_selector_active=rul.selector_active if rul else False,
            rul_withholding_reason_code=rul.withholding_reason_code if rul else None,
            rul_withholding_reasons=rul.withholding_reasons if rul else (),
            rul_support_distance=rul.support_distance if rul else None,
            rul_neighbor_dispersion_hours=rul.neighbor_dispersion_hours if rul else None,
            rul_model_disagreement_hours=rul.model_disagreement_hours if rul else None,
            warning_rul_forecastability_state=rul.warning_forecastability_state if rul else "RUL_UNAVAILABLE",
            warning_rul_forecastability_score=rul.warning_forecastability_score if rul else None,
            warning_rul_serviceable_intent=rul.warning_serviceable_intent if rul else False,
            warning_rul_hard_eligible=rul.warning_hard_eligible if rul else False,
            warning_rul_selector_active=rul.warning_selector_active if rul else False,
            warning_rul_withholding_reason_code=rul.warning_withholding_reason_code if rul else None,
            warning_rul_withholding_reasons=rul.warning_withholding_reasons if rul else (),
            warning_rul_raw_point_hours=rul.warning_raw_point_hours if rul else None,
            warning_rul_corrected_point_hours=rul.warning_corrected_point_hours if rul else None,
            warning_rul_calibration_stratum=rul.warning_calibration_stratum if rul else None,
            critical_rul_forecastability_state=rul.critical_forecastability_state if rul else "RUL_UNAVAILABLE",
            critical_rul_forecastability_score=rul.critical_forecastability_score if rul else None,
            critical_rul_serviceable_intent=rul.critical_serviceable_intent if rul else False,
            critical_rul_hard_eligible=rul.critical_hard_eligible if rul else False,
            critical_rul_selector_active=rul.critical_selector_active if rul else False,
            critical_rul_withholding_reason_code=rul.critical_withholding_reason_code if rul else None,
            critical_rul_withholding_reasons=rul.critical_withholding_reasons if rul else (),
            critical_rul_raw_point_hours=rul.critical_raw_point_hours if rul else None,
            critical_rul_corrected_point_hours=rul.critical_corrected_point_hours if rul else None,
            critical_rul_calibration_stratum=rul.critical_calibration_stratum if rul else None,
        )
