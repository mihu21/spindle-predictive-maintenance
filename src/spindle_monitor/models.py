from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any


class Status(IntEnum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2

    @classmethod
    def from_text(cls, value: str) -> "Status":
        return cls[value.strip().upper()]


class QualityStatus(str, Enum):
    NORMAL = "NORMAL"
    VALID_WITH_OBSERVATION = "VALID_WITH_OBSERVATION"
    SUSPECTED_SENSOR_ANOMALY = "SUSPECTED_SENSOR_ANOMALY"
    INVALID_SENSOR_DATA = "INVALID_SENSOR_DATA"
    POSSIBLE_MACHINE_EVENT = "POSSIBLE_MACHINE_EVENT"
    CONFIRMED_MACHINE_CRITICAL = "CONFIRMED_MACHINE_CRITICAL"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


class AnomalyType(str, Enum):
    NONE = "NONE"
    MALFORMED_VALUE = "MALFORMED_VALUE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    OUT_OF_ORDER_TIMESTAMP = "OUT_OF_ORDER_TIMESTAMP"
    SHORT_DATA_GAP = "SHORT_DATA_GAP"
    LONG_DATA_GAP = "LONG_DATA_GAP"
    SINGLE_SAMPLE_SPIKE = "SINGLE_SAMPLE_SPIKE"
    PERSISTENT_STEP_CHANGE = "PERSISTENT_STEP_CHANGE"
    POSSIBLE_STUCK_SENSOR = "POSSIBLE_STUCK_SENSOR"
    STUCK_SENSOR_SUSPECTED = "STUCK_SENSOR_SUSPECTED"
    STUCK_SENSOR_CONFIRMED = "STUCK_SENSOR_CONFIRMED"
    EXCESSIVE_NOISE = "EXCESSIVE_NOISE"
    CLIPPING_SUSPECTED = "CLIPPING_SUSPECTED"
    SENSOR_DISAGREEMENT = "SENSOR_DISAGREEMENT"
    CORRELATED_ABRUPT_CHANGE = "CORRELATED_ABRUPT_CHANGE"
    DRIFT_SUSPECTED = "DRIFT_SUSPECTED"
    ORIGIN_UNCERTAIN = "ORIGIN_UNCERTAIN"


class ModelAction(str, Enum):
    USE_NORMALLY = "USE_NORMALLY"
    USE_WITH_REDUCED_CONFIDENCE = "USE_WITH_REDUCED_CONFIDENCE"
    EXCLUDE_CURRENT_READING_FROM_TREND = "EXCLUDE_CURRENT_READING_FROM_TREND"
    HOLD_FOR_CONFIRMATION = "HOLD_FOR_CONFIRMATION"
    SUSPEND_PREDICTION = "SUSPEND_PREDICTION"
    REFUSE_PREDICTION = "REFUSE_PREDICTION"
    EXCLUDE_FROM_RETRAINING = "EXCLUDE_FROM_RETRAINING"


class FeatureHistoryAction(str, Enum):
    COMMIT_TO_FEATURE_HISTORY = "COMMIT_TO_FEATURE_HISTORY"
    HOLD_OUTSIDE_FEATURE_HISTORY = "HOLD_OUTSIDE_FEATURE_HISTORY"
    DISCARD_FROM_FEATURE_HISTORY = "DISCARD_FROM_FEATURE_HISTORY"
    COMMIT_AFTER_CONFIRMATION = "COMMIT_AFTER_CONFIRMATION"


@dataclass(frozen=True)
class AnomalyResult:
    quality_status: QualityStatus = QualityStatus.NORMAL
    anomaly_type: tuple[AnomalyType, ...] = (AnomalyType.NONE,)
    anomaly_severity: str = "NONE"
    anomaly_confidence: float = 0.0
    suspected_origin: str = "none"
    affected_sensors: tuple[str, ...] = ()
    first_observed_timestamp: datetime | None = None
    confirmed_timestamp: datetime | None = None
    resolved_timestamp: datetime | None = None
    is_active: bool = False
    raw_safety_status: str = "NORMAL"
    safety_action: str = "CONTINUE_MANUFACTURER_MONITORING"
    model_action: ModelAction = ModelAction.USE_NORMALLY
    feature_history_action: FeatureHistoryAction = FeatureHistoryAction.COMMIT_TO_FEATURE_HISTORY
    use_for_features: bool = True
    use_for_prediction: bool = True
    use_for_retraining: bool = True
    confidence_multiplier: float = 1.0
    supporting_evidence: tuple[str, ...] = ()
    configuration_snapshot: dict[str, Any] = field(default_factory=dict)
    detector_version: str = "1.0.0"
    causal_decision: str = "No anomaly evidence in current or past observations."
    final_offline_classification: str = "PENDING_OR_SAME_AS_CAUSAL"
    training_eligible: bool = True
    training_exclusion_reason: str = ""
    requires_human_review: bool = False
    decision_timestamp: datetime | None = None
    suspicion_timestamp: datetime | None = None
    confirmation_source: str = ""
    confirmation_evidence: str = ""
    confirming_actor: str = ""
    expected_sampling_interval_seconds: float | None = None
    source_sampling_interval_seconds: float | None = None
    estimated_missing_sample_count: int = 0
    gap_duration_seconds: float = 0.0
    interpolation_enabled: bool = False
    interpolation_occurred: bool = False
    confidence_penalty_reason: str = ""
    configured_confidence_multiplier: float = 1.0
    minimum_confidence_floor: float = 0.0
    correlation_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quality_status"] = self.quality_status.value
        data["anomaly_type"] = [value.value for value in self.anomaly_type]
        data["model_action"] = self.model_action.value
        data["feature_history_action"] = self.feature_history_action.value
        for name in (
            "first_observed_timestamp", "confirmed_timestamp", "resolved_timestamp",
            "decision_timestamp", "suspicion_timestamp",
        ):
            value = data[name]
            data[name] = value.isoformat() if value is not None else None
        return data


@dataclass(frozen=True)
class SensorReading:
    timestamp: datetime
    vibration_mps2: float
    temperature_c: float
    current_ampere: float

    def values(self) -> dict[str, float]:
        return {
            "vibration_mps2": self.vibration_mps2,
            "temperature_c": self.temperature_c,
            "current_ampere": self.current_ampere,
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str = ""
    row_number: int | None = None
    raw_row: dict[str, str] | None = None
    reading: SensorReading | None = None
    source_label: str | None = None
    interpolated: bool = False
    source_sampling_interval_seconds: float | None = None
    effective_resampling_interval_seconds: float = 0.0
    features_available: bool = True
    raw_safety_status: str = "UNKNOWN"


@dataclass(frozen=True)
class ThresholdForecast:
    sensor: str
    target_status: str
    target_value: float
    eta_seconds: float | None
    earliest_seconds: float | None
    latest_seconds: float | None
    slope_per_hour: float | None
    r_squared: float | None
    confidence: str
    explanation: str
    critical_eta_seconds: float | None = None
    critical_earliest_seconds: float | None = None
    critical_latest_seconds: float | None = None


@dataclass(frozen=True)
class StatisticalForecast:
    estimated_hours: float | None
    forecast_sensor: str | None
    confidence: str
    reason: str
    per_sensor_hours: dict[str, float | None] = field(default_factory=dict)
    target: str = "critical"


@dataclass(frozen=True)
class PrognosticForecast:
    time_to_warning_hours: float | None = None
    time_to_critical_hours: float | None = None
    warning_earliest_hours: float | None = None
    warning_latest_hours: float | None = None
    critical_earliest_hours: float | None = None
    critical_latest_hours: float | None = None
    warning_forecast_sensor: str | None = None
    critical_forecast_sensor: str | None = None
    probability_warning: dict[int, float | None] = field(default_factory=dict)
    probability_critical: dict[int, float | None] = field(default_factory=dict)
    per_sensor_probability_warning: dict[str, dict[int, float]] = field(default_factory=dict)
    per_sensor_probability_critical: dict[str, dict[int, float]] = field(default_factory=dict)
    per_sensor_eta: dict[str, dict[str, float | None]] = field(default_factory=dict)
    sensor_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    health_deviation_score: float = 0.0
    baseline_ready: bool = False
    confidence: str = "unavailable"
    reason: str = "Probabilistic degradation forecast is unavailable."
    method_version: str = "probabilistic_degradation_v6.0"
    history_hours: float = 0.0
    selected_thresholds: dict[str, float] = field(default_factory=dict)
    probability_threshold_crossings: tuple[str, ...] = ()
    withheld: bool = True
    withholding_reasons: tuple[str, ...] = ()

    def probability_target_value(self, target: str) -> float | None:
        if target.startswith("probability_warning_") and target.endswith("h"):
            hours = int(target.rsplit("_", 1)[1][:-1])
            return self.probability_warning.get(hours)
        if target.startswith("probability_critical_") and target.endswith("h"):
            hours = int(target.rsplit("_", 1)[1][:-1])
            return self.probability_critical.get(hours)
        return None


@dataclass(frozen=True)
class MLForecast:
    time_to_warning_hours: float | None = None
    time_to_critical_hours: float | None = None
    probability_warning: dict[int, float | None] = field(default_factory=dict)
    probability_critical: dict[int, float | None] = field(default_factory=dict)
    raw_probability_warning: dict[int, float | None] = field(default_factory=dict)
    raw_probability_critical: dict[int, float | None] = field(default_factory=dict)
    reconciled_probability_warning: dict[int, float | None] = field(default_factory=dict)
    reconciled_probability_critical: dict[int, float | None] = field(default_factory=dict)
    probability_reconciled: bool = False
    probability_reconciliation_targets: tuple[str, ...] = ()
    probability_reconciliation_reason: str = "none"
    maturity_stage: str = "unavailable"
    model_version: str | None = None
    confidence: str = "unavailable"
    reason: str = "No compatible production ML model is available."
    outside_training_distribution: bool = False
    outside_feature_fraction: float = 0.0
    beyond_training_duration_support: bool = False
    training_target_max_hours: dict[str, float | None] = field(default_factory=dict)
    training_target_min_hours: dict[str, float | None] = field(default_factory=dict)
    target_support_violations: dict[str, bool] = field(default_factory=dict)
    physically_invalid_targets: dict[str, bool] = field(default_factory=dict)
    sampling_interval_out_of_distribution: bool = False
    prediction_confidence: str = "unavailable"
    # These fields are deliberately carried with the raw prediction so that a
    # replay can explain why a number was displayed but not allowed to act.
    selected_thresholds: dict[str, float | None] = field(default_factory=dict)
    target_eligibility: dict[str, bool] = field(default_factory=dict)
    target_ineligibility_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    policy_withholding_reasons: tuple[str, ...] = ()
    physical_validation_passed: bool = True
    physical_validation_reasons: tuple[str, ...] = ()
    production_policy_passed: bool = False
    production_policy_reasons: tuple[str, ...] = ()
    required_model_targets: tuple[str, ...] = ()
    loaded_model_targets: tuple[str, ...] = ()
    missing_required_model_targets: tuple[str, ...] = ()
    probability_threshold_schema_version: str | None = None
    model_metadata_schema_version: str | None = None
    model_forecast_policy_contract_version: str | None = None
    feature_readiness_reasons: tuple[str, ...] = ()
    model_stage: str = "unavailable"
    data_domain: str = "unknown"
    production_eligible: bool = False
    revoked: bool = False
    superseded: bool = False
    corrupted: bool = False
    metadata_valid: bool = False
    artifacts_valid: bool = False
    schema_compatible: bool = False
    feature_schema_compatible: bool = False
    model_load_failures: dict[str, str] = field(default_factory=dict)
    probability_target_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    configured_mandatory_targets: tuple[str, ...] = ()
    metadata_required_targets: tuple[str, ...] = ()
    effective_required_targets: tuple[str, ...] = ()
    mandatory_targets_missing_from_metadata: tuple[str, ...] = ()
    runtime_probability_fn_ceiling: float | None = None
    model_recorded_probability_fn_ceiling: dict[str, float | None] = field(default_factory=dict)
    effective_probability_fn_ceiling: dict[str, float | None] = field(default_factory=dict)
    runtime_validation_fn_ceiling: float | None = None
    runtime_test_fn_ceiling: float | None = None
    model_recorded_validation_fn_ceilings: dict[str, float | None] = field(default_factory=dict)
    model_recorded_test_fn_ceilings: dict[str, float | None] = field(default_factory=dict)
    effective_validation_fn_ceilings: dict[str, float | None] = field(default_factory=dict)
    effective_test_fn_ceilings: dict[str, float | None] = field(default_factory=dict)
    eta_target_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    probability_threshold_crossings: tuple[str, ...] = ()
    recommendation_trigger_probability_targets: tuple[str, ...] = ()
    recommendation_trigger_eta_targets: tuple[str, ...] = ()
    recommendation_trigger_rules: tuple[str, ...] = ()
    recommendation_source: str = "unavailable"
    recommendation_trigger_targets: tuple[str, ...] = ()  # legacy union


@dataclass(frozen=True)
class ForecastDecision:
    """Auditable selection of one target forecast without conflating its roles."""

    primary_hours: float | None
    conservative_alert_hours: float | None
    confidence: str
    primary_source: str
    reason: str
    absolute_disagreement_hours: float | None = None
    relative_disagreement: float | None = None
    strong_disagreement: bool = False
    withheld: bool = False
    withholding_reason: str = "none"
    withholding_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensorAssessment:
    sensor: str
    display_name: str
    unit: str
    raw_value: float
    smoothed_value: float
    raw_status: Status
    smoothed_status: Status
    severity: float
    health_percent: float
    critical_margin_percent: float
    forecast: ThresholdForecast | None = None
    rolling_median: float | None = None
    fast_ewma: float | None = None
    slow_ewma: float | None = None
    kalman_level: float | None = None
    kalman_rate_per_hour: float | None = None


@dataclass(frozen=True)
class LifecycleRecord:
    lifecycle_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    duration_hours: float
    highest_status: str
    first_warning_timestamp: datetime | None
    first_critical_timestamp: datetime | None
    critical_reached: bool
    inferred_reset_timestamp: datetime
    reset_confidence: str
    reset_reason: str
    pre_reset_values: dict[str, float]
    post_reset_values: dict[str, float]
    lifecycle_state: str = "completed_with_reset"
    censored: bool = False
    first_raw_warning_timestamp: datetime | None = None
    first_confirmed_warning_timestamp: datetime | None = None
    first_raw_critical_timestamp: datetime | None = None
    first_confirmed_critical_timestamp: datetime | None = None
    operating_regime: str = "unknown"
    degradation_family: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "start_timestamp",
            "end_timestamp",
            "first_warning_timestamp",
            "first_critical_timestamp",
            "first_raw_warning_timestamp",
            "first_confirmed_warning_timestamp",
            "first_raw_critical_timestamp",
            "first_confirmed_critical_timestamp",
            "inferred_reset_timestamp",
        ):
            value = data[key]
            data[key] = value.isoformat() if value is not None else None
        return data


@dataclass(frozen=True)
class LifecycleSnapshot:
    lifecycle_id: str
    lifecycle_state: str
    elapsed_lifecycle_hours: float
    reset_confidence: str = ""
    reset_reason: str = ""
    completed_lifecycle: LifecycleRecord | None = None


@dataclass(frozen=True)
class MonitorResult:
    timestamp: datetime
    raw_status: Status
    effective_status: Status
    health_percent: float
    worst_sensor: str
    worst_sensor_display: str
    maximum_severity: float
    weighted_severity: float
    estimated_degradation: float
    degradation_rate_per_hour: float
    assessments: dict[str, SensorAssessment]
    reasons: list[str] = field(default_factory=list)
    lifecycle_id: str = "lifecycle_0001"
    lifecycle_state: str = "HEALTHY"
    elapsed_lifecycle_hours: float = 0.0
    reset_confidence: str = ""
    reset_reason: str = ""
    statistical_forecast: StatisticalForecast | None = None
    statistical_warning_forecast: StatisticalForecast | None = None
    statistical_critical_forecast: StatisticalForecast | None = None
    prognostic_forecast: PrognosticForecast = PrognosticForecast()
    ml_forecast: MLForecast = MLForecast()
    final_time_to_warning_hours: float | None = None
    final_time_to_critical_hours: float | None = None
    conservative_alert_time_to_warning_hours: float | None = None
    conservative_alert_time_to_critical_hours: float | None = None
    warning_primary_source: str = "unavailable"
    critical_primary_source: str = "unavailable"
    warning_absolute_disagreement_hours: float | None = None
    warning_relative_disagreement: float | None = None
    critical_absolute_disagreement_hours: float | None = None
    critical_relative_disagreement: float | None = None
    disagreement_trigger_targets: tuple[str, ...] = ()
    ml_outside_training_distribution: bool = False
    ml_outside_feature_fraction: float = 0.0
    forecast_withheld: bool = False
    withholding_reasons: tuple[str, ...] = ()
    recommendation_trigger_targets: tuple[str, ...] = ()
    probability_threshold_crossings: tuple[str, ...] = ()
    recommendation_trigger_probability_targets: tuple[str, ...] = ()
    recommendation_trigger_eta_targets: tuple[str, ...] = ()
    recommendation_trigger_rules: tuple[str, ...] = ()
    recommendation_source: str = "unavailable"
    recommendation_actionable: bool = False
    final_forecast_hours: float | None = None
    forecast_confidence: str = "unavailable"
    forecast_reason: str = ""
    maintenance_urgency: str = "NORMAL_MONITORING"
    features: dict[str, float] = field(default_factory=dict)
    completed_lifecycle: LifecycleRecord | None = None
    event_status: Status = Status.NORMAL
    data_domain: str = "accelerated_mock"
    sampling_gap_seconds: float = 0.0
    interpolated: bool = False
    feature_available: bool = True
    target_support_violation: bool = False
    source_sampling_interval_seconds: float | None = None
    effective_resampling_interval_seconds: float = 0.0
    available_history_duration_seconds: float = 0.0
    dataset_hash: str = ""
    generator_version: str = ""
    model_stage: str = "unavailable"
    model_version: str = ""
    guardrail_result: str = ""
    refusal_or_fallback_reason: str = ""
    anomaly: AnomalyResult = field(default_factory=AnomalyResult)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["raw_status"] = self.raw_status.name
        data["effective_status"] = self.effective_status.name
        data["event_status"] = self.event_status.name
        data["anomaly"] = self.anomaly.to_dict()
        for assessment in data["assessments"].values():
            assessment["raw_status"] = Status(assessment["raw_status"]).name
            assessment["smoothed_status"] = Status(assessment["smoothed_status"]).name
        if data.get("completed_lifecycle"):
            completed = self.completed_lifecycle
            assert completed is not None
            data["completed_lifecycle"] = completed.to_dict()
        return data
