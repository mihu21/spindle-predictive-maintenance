from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SensorConfig:
    key: str
    display_name: str
    unit: str
    healthy_baseline: float
    warning: float
    critical: float
    valid_min: float
    valid_max: float
    severity_cap_value: float
    weight: float

    def validate(self) -> None:
        if not self.valid_min <= self.healthy_baseline < self.warning < self.critical:
            raise ValueError(
                f"Invalid threshold order for {self.key}: expected "
                "valid_min <= healthy_baseline < warning < critical"
            )
        if self.severity_cap_value <= self.critical:
            raise ValueError(f"severity_cap_value must exceed critical for {self.key}")
        if self.valid_max < self.severity_cap_value:
            raise ValueError(f"valid_max must cover severity_cap_value for {self.key}")
        if self.weight < 0:
            raise ValueError(f"weight cannot be negative for {self.key}")


@dataclass(frozen=True)
class MonitoringConfig:
    ewma_alpha: float
    history_size: int
    forecast_window: int
    minimum_forecast_points: int
    minimum_forecast_r_squared: float
    forecast_horizon_hours: float
    forecast_step_minutes: float
    warning_vote_window: int
    warning_votes_required: int
    normal_recovery_readings: int
    critical_recovery_readings: int
    kalman_process_variance: float
    kalman_measurement_variance: float


@dataclass(frozen=True)
class SmoothingConfig:
    rolling_median_window: int = 5
    fast_ewma_alpha: float = 0.35
    slow_ewma_alpha: float = 0.10
    kalman_process_variance: float = 0.00005
    kalman_measurement_variance: float = 0.01
    maximum_jump_multiplier: float = 4.0

    def validate(self) -> None:
        if self.rolling_median_window < 1:
            raise ValueError("rolling_median_window must be at least 1")
        for name, value in (
            ("fast_ewma_alpha", self.fast_ewma_alpha),
            ("slow_ewma_alpha", self.slow_ewma_alpha),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.fast_ewma_alpha < self.slow_ewma_alpha:
            raise ValueError("fast_ewma_alpha must be at least slow_ewma_alpha")
        if self.maximum_jump_multiplier <= 0:
            raise ValueError("maximum_jump_multiplier must be positive")


@dataclass(frozen=True)
class PrognosticsConfig:
    enabled: bool = True
    method: str = "probabilistic_degradation_v6"
    forecast_horizons_hours: tuple[int, ...] = (6, 12, 24)
    trend_windows_minutes: tuple[int, ...] = (60, 180, 360, 720, 1440)
    minimum_history_minutes: float = 180.0
    minimum_trend_windows: int = 2
    minimum_samples_per_trend_window: int = 30
    minimum_window_coverage_fraction: float = 0.80
    minimum_window_history_fraction: float = 0.90
    healthy_baseline_ewma_alpha: float = 0.002
    minimum_healthy_baseline_samples: int = 180
    initial_baseline_std_fraction: float = 0.05
    minimum_baseline_std_fraction: float = 0.02
    baseline_update_max_z: float = 2.5
    baseline_warmup_max_z: float = 8.0
    minimum_slope_uncertainty_fraction_per_hour: float = 0.001
    single_window_relative_uncertainty: float = 0.50
    slope_snr_for_full_strength: float = 2.0
    shrinkage_uncertainty_multiplier: float = 0.50
    eta_interval_normal_z: float = 1.2815515655446004
    high_confidence_history_minutes: float = 720.0
    high_confidence_minimum_trend_windows: int = 4
    high_confidence_minimum_consistency: float = 0.75
    medium_confidence_minimum_consistency: float = 0.55
    health_deviation_z_cap: float = 8.0
    warning_12h_action_probability: float = 0.50
    critical_6h_action_probability: float = 0.50
    critical_24h_action_probability: float = 0.50
    eta_action_min_probability: float = 0.50
    probability_persistence_minutes: float = 30.0
    warning_probability_persistence_minutes: float = 10.0
    critical_probability_persistence_minutes: float = 30.0
    probability_release_fraction: float = 0.70
    probability_release_persistence_minutes: float = 60.0
    advisory_recommendations_enabled: bool = True
    automated_action_enabled: bool = False
    schema_version: str = "1.0"

    def validate(self) -> None:
        if self.method != "probabilistic_degradation_v6":
            raise ValueError("unsupported prognostics method")
        if not self.forecast_horizons_hours or any(value <= 0 for value in self.forecast_horizons_hours):
            raise ValueError("prognostics forecast_horizons_hours must contain positive values")
        if tuple(sorted(set(self.forecast_horizons_hours))) != tuple(self.forecast_horizons_hours):
            raise ValueError("prognostics forecast_horizons_hours must be unique and increasing")
        if not self.trend_windows_minutes or any(value <= 0 for value in self.trend_windows_minutes):
            raise ValueError("trend_windows_minutes must contain positive values")
        if tuple(sorted(set(self.trend_windows_minutes))) != tuple(self.trend_windows_minutes):
            raise ValueError("trend_windows_minutes must be unique and increasing")
        if self.minimum_history_minutes <= 0:
            raise ValueError("minimum_history_minutes must be positive")
        if self.minimum_trend_windows < 1:
            raise ValueError("minimum_trend_windows must be at least 1")
        if self.high_confidence_minimum_trend_windows < self.minimum_trend_windows:
            raise ValueError("high_confidence_minimum_trend_windows cannot be below minimum_trend_windows")
        if self.minimum_samples_per_trend_window < 3:
            raise ValueError("minimum_samples_per_trend_window must be at least 3")
        for name, value in (
            ("minimum_window_coverage_fraction", self.minimum_window_coverage_fraction),
            ("minimum_window_history_fraction", self.minimum_window_history_fraction),
            ("healthy_baseline_ewma_alpha", self.healthy_baseline_ewma_alpha),
            ("high_confidence_minimum_consistency", self.high_confidence_minimum_consistency),
            ("medium_confidence_minimum_consistency", self.medium_confidence_minimum_consistency),
            ("warning_12h_action_probability", self.warning_12h_action_probability),
            ("critical_6h_action_probability", self.critical_6h_action_probability),
            ("critical_24h_action_probability", self.critical_24h_action_probability),
            ("eta_action_min_probability", self.eta_action_min_probability),
            ("probability_release_fraction", self.probability_release_fraction),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.minimum_healthy_baseline_samples < 1:
            raise ValueError("minimum_healthy_baseline_samples must be positive")
        for name, value in (
            ("initial_baseline_std_fraction", self.initial_baseline_std_fraction),
            ("minimum_baseline_std_fraction", self.minimum_baseline_std_fraction),
            ("minimum_slope_uncertainty_fraction_per_hour", self.minimum_slope_uncertainty_fraction_per_hour),
            ("single_window_relative_uncertainty", self.single_window_relative_uncertainty),
            ("slope_snr_for_full_strength", self.slope_snr_for_full_strength),
            ("shrinkage_uncertainty_multiplier", self.shrinkage_uncertainty_multiplier),
            ("eta_interval_normal_z", self.eta_interval_normal_z),
            ("health_deviation_z_cap", self.health_deviation_z_cap),
            ("baseline_update_max_z", self.baseline_update_max_z),
            ("baseline_warmup_max_z", self.baseline_warmup_max_z),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.probability_persistence_minutes < 0:
            raise ValueError("probability_persistence_minutes cannot be negative")
        if self.warning_probability_persistence_minutes < 0:
            raise ValueError("warning_probability_persistence_minutes cannot be negative")
        if self.critical_probability_persistence_minutes < 0:
            raise ValueError("critical_probability_persistence_minutes cannot be negative")
        if self.probability_release_persistence_minutes < 0:
            raise ValueError("probability_release_persistence_minutes cannot be negative")
        if self.baseline_warmup_max_z < self.baseline_update_max_z:
            raise ValueError("baseline_warmup_max_z cannot be below baseline_update_max_z")
        if self.high_confidence_history_minutes < self.minimum_history_minutes:
            raise ValueError("high_confidence_history_minutes cannot be below minimum_history_minutes")
        if self.medium_confidence_minimum_consistency > self.high_confidence_minimum_consistency:
            raise ValueError("medium confidence consistency cannot exceed high confidence consistency")


@dataclass(frozen=True)
class LifecycleConfig:
    minimum_degraded_minutes: float = 30.0
    normal_confirmation_minutes: float = 60.0
    medium_confirmation_multiplier: float = 1.5
    pre_reset_window_minutes: float = 60.0
    post_reset_window_minutes: float = 60.0
    minimum_lifecycle_hours: float = 12.0
    reset_cooldown_hours: float = 2.0
    minimum_improved_sensor_count: int = 2
    minimum_sensor_drop_fraction: float = 0.20
    minimum_overall_severity_drop: float = 0.30
    healthy_baseline_tolerance: float = 0.35
    warning_confirmation_minutes: float = 30.0
    critical_confirmation_minutes: float = 30.0
    normal_reset_confirmation_minutes: float = 60.0
    minimum_critical_duration_before_reset_minutes: float = 30.0
    maximum_confirmation_gap_minutes: float = 5.0

    def validate(self) -> None:
        positive = {
            "minimum_degraded_minutes": self.minimum_degraded_minutes,
            "normal_confirmation_minutes": self.normal_confirmation_minutes,
            "pre_reset_window_minutes": self.pre_reset_window_minutes,
            "post_reset_window_minutes": self.post_reset_window_minutes,
            "minimum_lifecycle_hours": self.minimum_lifecycle_hours,
            "reset_cooldown_hours": self.reset_cooldown_hours,
            "medium_confirmation_multiplier": self.medium_confirmation_multiplier,
            "warning_confirmation_minutes": self.warning_confirmation_minutes,
            "critical_confirmation_minutes": self.critical_confirmation_minutes,
            "normal_reset_confirmation_minutes": self.normal_reset_confirmation_minutes,
            "minimum_critical_duration_before_reset_minutes": self.minimum_critical_duration_before_reset_minutes,
            "maximum_confirmation_gap_minutes": self.maximum_confirmation_gap_minutes,
        }
        for name, value in positive.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.minimum_improved_sensor_count < 1:
            raise ValueError("minimum_improved_sensor_count must be at least 1")
        if not 0 <= self.minimum_sensor_drop_fraction <= 1:
            raise ValueError("minimum_sensor_drop_fraction must be in [0, 1]")
        if not 0 <= self.minimum_overall_severity_drop <= 1.5:
            raise ValueError("minimum_overall_severity_drop must be in [0, 1.5]")


@dataclass(frozen=True)
class MLConfig:
    model_type: str = "HistGradientBoostingRegressor"
    minimum_candidate_lifecycles: int = 3
    minimum_deployment_lifecycles: int = 10
    meaningful_validation_lifecycles: int = 20
    minimum_completed_lifecycles_for_training: int = 10
    minimum_completed_lifecycles_for_promotion: int = 20
    minimum_validation_lifecycles: int = 4
    minimum_test_lifecycles: int = 4
    minimum_samples_per_lifecycle: int = 12
    maximum_training_rows_per_lifecycle: int = 500
    random_state: int = 42
    split_random_seed: int = 42
    sampling_random_seed: int = 42
    training_fraction: float = 2.0 / 3.0
    validation_fraction: float = 1.0 / 6.0
    test_fraction: float = 1.0 / 6.0
    sample_weighting_enabled: bool = True
    minimum_probability_positive_rate: float = 0.05
    maximum_probability_positive_rate: float = 0.95
    probability_classification_threshold: float = 0.5
    maximum_probability_false_negative_rate: float = 0.10
    probability_validation_fn_safety_factor: float = 0.90
    maximum_probability_validation_false_positive_rate: float = 0.15
    probability_early_positive_weight_multiplier: float = 1.35
    probability_positive_class_balance_strength: float = 0.35
    probability_boundary_negative_weight_multiplier: float = 1.75
    probability_hard_negative_mining_enabled: bool = True
    probability_hard_negative_quantile: float = 0.85
    probability_hard_negative_weight_multiplier: float = 2.25
    probability_hard_early_positive_quantile: float = 0.20
    probability_hard_early_positive_weight_multiplier: float = 1.25
    probability_hard_example_cv_folds: int = 3
    minimum_validation_lifecycles_for_calibration: int = 8
    lifecycle_split_candidate_attempts: int = 128
    maximum_probability_calibration_error: float = 0.10
    max_iter: int = 250
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.1
    warning_disagreement_absolute_hours: float = 9.615345
    critical_disagreement_absolute_hours: float = 5.233922
    disagreement_relative_threshold: float = 0.991534
    feature_distribution_margin_fraction: float = 0.10
    feature_distribution_max_outside_fraction: float = 0.10
    forecast_horizons_hours: tuple[int, ...] = (6, 12, 24)
    required_probability_targets: tuple[str, ...] = (
        "probability_warning_12h",
        "probability_critical_24h",
    )
    required_eta_targets: tuple[str, ...] = ()
    minimum_final_confidence: str = "medium"
    critical_soon_probability: float = 0.70
    critical_plan_probability: float = 0.60
    warning_plan_probability: float = 0.70
    statistical_fallback_actionable: bool = False
    withhold_on_severe_disagreement: bool = True
    schema_version: str = "3.0"

    def validate(self) -> None:
        if self.model_type != "HistGradientBoostingRegressor":
            raise ValueError("Only HistGradientBoostingRegressor is currently supported")
        if self.minimum_candidate_lifecycles < 2:
            raise ValueError("minimum_candidate_lifecycles must be at least 2")
        if self.minimum_deployment_lifecycles < self.minimum_candidate_lifecycles:
            raise ValueError("minimum_deployment_lifecycles cannot be below candidate minimum")
        if self.maximum_training_rows_per_lifecycle < self.minimum_samples_per_lifecycle:
            raise ValueError("maximum_training_rows_per_lifecycle is too small")
        fractions = self.training_fraction + self.validation_fraction + self.test_fraction
        if abs(fractions - 1.0) > 1e-9 or min(
            self.training_fraction, self.validation_fraction, self.test_fraction
        ) <= 0:
            raise ValueError("training/validation/test fractions must be positive and sum to 1")
        if not 0 <= self.minimum_probability_positive_rate < self.maximum_probability_positive_rate <= 1:
            raise ValueError("probability event-rate limits must satisfy 0 <= minimum < maximum <= 1")
        if not 0 < self.probability_classification_threshold < 1:
            raise ValueError("probability_classification_threshold must be in (0, 1)")
        if not 0 <= self.maximum_probability_false_negative_rate <= 1:
            raise ValueError("maximum_probability_false_negative_rate must be in [0, 1]")
        if not 0 < self.probability_validation_fn_safety_factor <= 1:
            raise ValueError("probability_validation_fn_safety_factor must be in (0, 1]")
        if not 0 <= self.maximum_probability_validation_false_positive_rate <= 1:
            raise ValueError("maximum_probability_validation_false_positive_rate must be in [0, 1]")
        if self.probability_early_positive_weight_multiplier < 1:
            raise ValueError("probability_early_positive_weight_multiplier must be at least 1")
        if not 0 <= self.probability_positive_class_balance_strength <= 1:
            raise ValueError("probability_positive_class_balance_strength must be in [0, 1]")
        if self.probability_boundary_negative_weight_multiplier < 1:
            raise ValueError("probability_boundary_negative_weight_multiplier must be at least 1")
        if not 0 < self.probability_hard_negative_quantile < 1:
            raise ValueError("probability_hard_negative_quantile must be in (0, 1)")
        if self.probability_hard_negative_weight_multiplier < 1:
            raise ValueError("probability_hard_negative_weight_multiplier must be at least 1")
        if not 0 < self.probability_hard_early_positive_quantile < 1:
            raise ValueError("probability_hard_early_positive_quantile must be in (0, 1)")
        if self.probability_hard_early_positive_weight_multiplier < 1:
            raise ValueError("probability_hard_early_positive_weight_multiplier must be at least 1")
        if self.probability_hard_example_cv_folds < 2:
            raise ValueError("probability_hard_example_cv_folds must be at least 2")
        if self.minimum_validation_lifecycles_for_calibration < 4:
            raise ValueError("minimum_validation_lifecycles_for_calibration must be at least 4")
        if self.lifecycle_split_candidate_attempts < 1:
            raise ValueError("lifecycle_split_candidate_attempts must be positive")
        if not 0 <= self.maximum_probability_calibration_error <= 1:
            raise ValueError("maximum_probability_calibration_error must be in [0, 1]")
        if self.warning_disagreement_absolute_hours < 0:
            raise ValueError("warning_disagreement_absolute_hours cannot be negative")
        if self.critical_disagreement_absolute_hours < 0:
            raise ValueError("critical_disagreement_absolute_hours cannot be negative")
        if self.disagreement_relative_threshold < 0:
            raise ValueError("disagreement_relative_threshold cannot be negative")
        if self.feature_distribution_margin_fraction < 0:
            raise ValueError("feature_distribution_margin_fraction cannot be negative")
        if not 0 <= self.feature_distribution_max_outside_fraction <= 1:
            raise ValueError("feature_distribution_max_outside_fraction must be in [0, 1]")
        if not self.forecast_horizons_hours or any(value <= 0 for value in self.forecast_horizons_hours):
            raise ValueError("forecast_horizons_hours must contain positive values")
        if tuple(sorted(set(self.forecast_horizons_hours))) != tuple(self.forecast_horizons_hours):
            raise ValueError("forecast_horizons_hours must be unique and increasing")
        configured_probability_targets = {
            f"probability_{kind}_{hours}h"
            for kind in ("warning", "critical")
            for hours in self.forecast_horizons_hours
        }
        unknown_probability = set(self.required_probability_targets) - configured_probability_targets
        if unknown_probability:
            raise ValueError(f"unknown required_probability_targets: {sorted(unknown_probability)}")
        unknown_eta = set(self.required_eta_targets) - {"time_to_warning", "time_to_critical"}
        if unknown_eta:
            raise ValueError(f"unknown required_eta_targets: {sorted(unknown_eta)}")
        if len(set(self.required_probability_targets)) != len(self.required_probability_targets):
            raise ValueError("required_probability_targets must be unique")
        if len(set(self.required_eta_targets)) != len(self.required_eta_targets):
            raise ValueError("required_eta_targets must be unique")
        for name, value in (
            ("minimum_completed_lifecycles_for_training", self.minimum_completed_lifecycles_for_training),
            ("minimum_completed_lifecycles_for_promotion", self.minimum_completed_lifecycles_for_promotion),
            ("minimum_validation_lifecycles", self.minimum_validation_lifecycles),
            ("minimum_test_lifecycles", self.minimum_test_lifecycles),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.minimum_final_confidence not in {"medium", "high"}:
            raise ValueError("minimum_final_confidence must be medium or high")
        for name, value in (
            ("critical_soon_probability", self.critical_soon_probability),
            ("critical_plan_probability", self.critical_plan_probability),
            ("warning_plan_probability", self.warning_plan_probability),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class AnomalySensorConfig:
    measurement_resolution: float
    decimal_precision: int
    expected_response_delay_seconds: float
    physical_minimum: float
    physical_maximum: float
    stuck_observation_seconds: float
    stuck_suspicion_seconds: float
    exact_raw_values_available: bool = True
    related_sensors: tuple[str, ...] = ()
    meaningful_context_change: float = 0.0

    def validate(self, key: str) -> None:
        if self.measurement_resolution <= 0:
            raise ValueError(f"measurement_resolution must be positive for {key}")
        if self.decimal_precision < 0:
            raise ValueError(f"decimal_precision cannot be negative for {key}")
        if self.physical_minimum >= self.physical_maximum:
            raise ValueError(f"physical bounds are invalid for {key}")
        if min(self.expected_response_delay_seconds, self.stuck_observation_seconds,
               self.stuck_suspicion_seconds) < 0:
            raise ValueError(f"anomaly durations cannot be negative for {key}")


@dataclass(frozen=True)
class AnomalyConfig:
    enabled: bool = True
    detector_version: str = "1.1.0"
    threshold_origin: str = "Conservative engineering assumption; requires plant calibration"
    spike_observation_seconds: float = 180.0
    spike_recovery_tolerance_multiplier: float = 3.0
    spike_delta_fraction_of_threshold_span: float = 0.25
    persistent_change_seconds: float = 180.0
    noise_window_samples: int = 20
    noise_multiplier: float = 8.0
    clipping_boundary_tolerance_multiplier: float = 0.5
    clipping_minimum_samples: int = 3
    multi_sensor_correlation_window_seconds: float = 300.0
    long_gap_seconds: float = 300.0
    sampling_interval_tolerance_seconds: float = 5.0
    recovery_valid_samples: int = 5
    minimum_contextual_evidence_count: int = 2
    drift_window_samples: int = 120
    drift_minimum_span_fraction: float = 0.35
    confidence_penalties: dict[str, float] = field(default_factory=lambda: {
        "observation": 1.0, "suspected": 0.65, "noise": 0.75, "short_gap": 0.9,
        "long_gap": 0.0, "machine_event": 0.55, "invalid": 0.0,
    })
    minimum_confidence_floor: float = 0.0
    sensors: dict[str, AnomalySensorConfig] = field(default_factory=dict)

    def validate(self, configured_sensors: set[str]) -> None:
        if self.spike_observation_seconds <= 0 or self.persistent_change_seconds <= 0:
            raise ValueError("anomaly observation durations must be positive")
        if self.noise_window_samples < 5 or self.clipping_minimum_samples < 2:
            raise ValueError("anomaly sample windows are too small")
        if self.recovery_valid_samples < 1 or self.minimum_contextual_evidence_count < 1:
            raise ValueError("anomaly evidence counts must be positive")
        if self.sampling_interval_tolerance_seconds < 0:
            raise ValueError("sampling_interval_tolerance_seconds cannot be negative")
        if not 0 <= self.minimum_confidence_floor <= 1:
            raise ValueError("minimum_confidence_floor must be in [0, 1]")
        if any(not 0 <= value <= 1 for value in self.confidence_penalties.values()):
            raise ValueError("confidence penalties must be in [0, 1]")
        if set(self.sensors) != configured_sensors:
            raise ValueError("anomaly sensor metadata must match configured sensors")
        for key, sensor in self.sensors.items():
            sensor.validate(key)
            unknown = set(sensor.related_sensors) - configured_sensors
            if unknown:
                raise ValueError(f"unknown related sensors for {key}: {sorted(unknown)}")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    threshold_source: str
    threshold_comparison: str
    monitoring: MonitoringConfig
    sensors: dict[str, SensorConfig]
    smoothing: SmoothingConfig = SmoothingConfig()
    lifecycle: LifecycleConfig = LifecycleConfig()
    prognostics: PrognosticsConfig = PrognosticsConfig()
    ml: MLConfig = MLConfig()
    threshold_config_version: str = "1"
    lifecycle_config_version: str = "1"
    feature_windows_minutes: tuple[int, ...] = (5, 15, 30, 60, 180, 360, 720, 1440)
    target_sampling_interval_seconds: int = 60
    interpolation_enabled: bool = False
    maximum_interpolation_gap_minutes: float = 5.0
    large_gap_policy: str = "mark_unavailable"
    duration_bucket_boundaries_hours: tuple[float, ...] = (72.0, 168.0, 336.0)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)

    def validate(self) -> None:
        if self.threshold_comparison not in {"greater_than", "greater_or_equal"}:
            raise ValueError("threshold_comparison must be greater_than or greater_or_equal")
        if not 0 < self.monitoring.ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.monitoring.warning_votes_required > self.monitoring.warning_vote_window:
            raise ValueError("warning_votes_required cannot exceed warning_vote_window")
        if self.monitoring.minimum_forecast_points > self.monitoring.forecast_window:
            raise ValueError("minimum_forecast_points cannot exceed forecast_window")
        if self.monitoring.history_size < self.monitoring.forecast_window:
            raise ValueError("history_size must be at least forecast_window")
        if self.monitoring.forecast_horizon_hours <= 0:
            raise ValueError("forecast_horizon_hours must be positive")
        if self.monitoring.forecast_step_minutes <= 0:
            raise ValueError("forecast_step_minutes must be positive")
        if not self.sensors:
            raise ValueError("At least one sensor must be configured")
        for sensor in self.sensors.values():
            sensor.validate()
        if sum(sensor.weight for sensor in self.sensors.values()) <= 0:
            raise ValueError("At least one sensor weight must be positive")
        self.smoothing.validate()
        self.lifecycle.validate()
        self.prognostics.validate()
        self.ml.validate()
        if self.anomaly.sensors:
            self.anomaly.validate(set(self.sensors))
        if not self.feature_windows_minutes or any(value <= 0 for value in self.feature_windows_minutes):
            raise ValueError("feature_windows_minutes must contain positive durations")
        if self.target_sampling_interval_seconds <= 0:
            raise ValueError("target_sampling_interval_seconds must be positive")
        if self.large_gap_policy not in {"mark_unavailable", "reject"}:
            raise ValueError("large_gap_policy must be mark_unavailable or reject")
        if tuple(sorted(self.duration_bucket_boundaries_hours)) != self.duration_bucket_boundaries_hours:
            raise ValueError("duration_bucket_boundaries_hours must be increasing")


def _load_optional(path: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(defaults)
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    return {**defaults, **loaded}


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = json.load(file)

    project = raw["project"]
    monitoring_raw = dict(raw["monitoring"])
    monitoring_raw.setdefault("forecast_horizon_hours", 240.0)
    monitoring_raw.setdefault("forecast_step_minutes", 10.0)
    monitoring = MonitoringConfig(**monitoring_raw)
    sensors = {
        key: SensorConfig(key=key, **sensor_data)
        for key, sensor_data in raw["sensors"].items()
    }

    smoothing = SmoothingConfig(**_load_optional(path.parent / "smoothing.json", {}))
    lifecycle = LifecycleConfig(**_load_optional(path.parent / "lifecycle.json", {}))
    prognostics_raw = _load_optional(path.parent / "prognostics.json", {})
    for key in ("forecast_horizons_hours", "trend_windows_minutes"):
        if key in prognostics_raw:
            prognostics_raw[key] = tuple(prognostics_raw[key])
    prognostics = PrognosticsConfig(**prognostics_raw)
    ml_raw = _load_optional(path.parent / "ml.json", {})
    if "forecast_horizons_hours" in ml_raw:
        ml_raw["forecast_horizons_hours"] = tuple(ml_raw["forecast_horizons_hours"])
    for key in ("required_probability_targets", "required_eta_targets"):
        if key in ml_raw:
            ml_raw[key] = tuple(ml_raw[key])
    ml = MLConfig(**ml_raw)
    data_raw = _load_optional(path.parent / "data.json", {})
    if "feature_windows_minutes" in data_raw:
        data_raw["feature_windows_minutes"] = tuple(data_raw["feature_windows_minutes"])
    if "duration_bucket_boundaries_hours" in data_raw:
        data_raw["duration_bucket_boundaries_hours"] = tuple(data_raw["duration_bucket_boundaries_hours"])
    anomaly_raw = _load_optional(path.parent / "anomaly.json", {"enabled": False, "sensors": {}})
    anomaly_sensors = {}
    for key, value in anomaly_raw.pop("sensors", {}).items():
        value = dict(value)
        value["related_sensors"] = tuple(value.get("related_sensors", ()))
        anomaly_sensors[key] = AnomalySensorConfig(**value)
    anomaly = AnomalyConfig(sensors=anomaly_sensors, **anomaly_raw)

    config = ProjectConfig(
        name=project["name"],
        threshold_source=project["threshold_source"],
        threshold_comparison=project.get("threshold_comparison", "greater_than"),
        monitoring=monitoring,
        sensors=sensors,
        smoothing=smoothing,
        lifecycle=lifecycle,
        prognostics=prognostics,
        ml=ml,
        anomaly=anomaly,
        threshold_config_version=str(project.get("config_version", "1")),
        lifecycle_config_version=str(project.get("lifecycle_config_version", "1")),
        **data_raw,
    )
    config.validate()
    return config
