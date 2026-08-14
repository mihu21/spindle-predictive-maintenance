from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class VVB001Reading:
    source_id: int
    timestamp: datetime
    line_sel: str
    machine_id: str
    vrms: float
    arms: float
    apeak: float
    crest: float
    temp: float

    @property
    def machine_key(self) -> str:
        return f"{self.line_sel}::{self.machine_id}"

    def sensor_values(self) -> dict[str, float]:
        return {
            "vrms": self.vrms,
            "arms": self.arms,
            "apeak": self.apeak,
            "crest": self.crest,
            "temp": self.temp,
        }


@dataclass(frozen=True)
class SourceRecord:
    source_id: int
    raw: dict[str, Any]
    reading: VVB001Reading | None
    parse_error: str = ""


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    status: str
    reasons: tuple[str, ...]
    reading: VVB001Reading | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class ProcessedReading:
    reading: VVB001Reading
    quality_status: str
    quality_reasons: tuple[str, ...]
    features: dict[str, float | int | str | None]
    predicted_status: str | None = None
    prediction_probabilities: dict[str, float] | None = None
    degradation_score: float | None = None
    prediction_policy_reason: str | None = None
    prediction_state_source: str = "NOT_EVALUATED"
    prediction_state_held: bool = False
    sensor_quality_status: str = "GOOD"
    sensor_quality_reasons: tuple[str, ...] = ()
    sensor_quality_held_sensors: tuple[str, ...] = ()
    sensor_quality_extreme_raw_override: bool = False
    estimated_hours_to_warning: float | None = None
    warning_rul_lower_hours: float | None = None
    warning_rul_upper_hours: float | None = None
    estimated_hours_to_critical: float | None = None
    critical_rul_lower_hours: float | None = None
    critical_rul_upper_hours: float | None = None
    rul_reliability: str = "UNAVAILABLE"
    rul_reason: str | None = None
    rul_trend_score_per_hour: float | None = None
    rul_trend_r2: float | None = None
    rul_history_hours: float = 0.0
    rul_trusted_points: int = 0
    rul_state_source: str = "NOT_EVALUATED"
    rul_method: str = "unavailable"
    rul_calibration_method: str | None = None
    rul_calibration_bucket: str | None = None
    rul_forecastability_state: str = "RUL_UNAVAILABLE"
    rul_forecastability_score: float | None = None
    rul_serviceable_intent: bool = False
    rul_hard_eligible: bool = False
    rul_selector_active: bool = False
    rul_withholding_reason_code: str | None = None
    rul_withholding_reasons: tuple[str, ...] = ()
    rul_support_distance: float | None = None
    rul_neighbor_dispersion_hours: float | None = None
    rul_model_disagreement_hours: float | None = None
    warning_rul_forecastability_state: str = "RUL_UNAVAILABLE"
    warning_rul_forecastability_score: float | None = None
    warning_rul_serviceable_intent: bool = False
    warning_rul_hard_eligible: bool = False
    warning_rul_selector_active: bool = False
    warning_rul_withholding_reason_code: str | None = None
    warning_rul_withholding_reasons: tuple[str, ...] = ()
    warning_rul_raw_point_hours: float | None = None
    warning_rul_corrected_point_hours: float | None = None
    warning_rul_calibration_stratum: str | None = None
    critical_rul_forecastability_state: str = "RUL_UNAVAILABLE"
    critical_rul_forecastability_score: float | None = None
    critical_rul_serviceable_intent: bool = False
    critical_rul_hard_eligible: bool = False
    critical_rul_selector_active: bool = False
    critical_rul_withholding_reason_code: str | None = None
    critical_rul_withholding_reasons: tuple[str, ...] = ()
    critical_rul_raw_point_hours: float | None = None
    critical_rul_corrected_point_hours: float | None = None
    critical_rul_calibration_stratum: str | None = None
