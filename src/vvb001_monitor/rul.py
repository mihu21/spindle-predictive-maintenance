from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class RULPrediction:
    """Causal time-to-threshold estimate derived from trusted degradation-score history.

    This is deliberately separate from the learned regime classifier.  It extrapolates the
    recent *trusted* degradation-score trajectory to the model's learned WARNING/CRITICAL score
    boundaries.  It does not interpret degradation_score as a percentage of life consumed.
    """

    estimated_hours_to_warning: float | None
    warning_lower_hours: float | None
    warning_upper_hours: float | None
    estimated_hours_to_critical: float | None
    critical_lower_hours: float | None
    critical_upper_hours: float | None
    reliability: str
    reason: str
    trend_score_per_hour: float | None
    trend_r2: float | None
    history_hours: float
    trusted_points: int
    state_source: str = "LIVE_TREND"
    method: str = "legacy_causal_trend_extrapolation"
    calibration_method: str | None = None
    calibration_bucket: str | None = None
    forecastability_state: str = "RUL_ACTIVE"
    forecastability_score: float | None = None
    serviceable_intent: bool = False
    hard_eligible: bool = False
    selector_active: bool = False
    withholding_reason_code: str | None = None
    withholding_reasons: tuple[str, ...] = ()
    support_distance: float | None = None
    neighbor_dispersion_hours: float | None = None
    model_disagreement_hours: float | None = None
    warning_forecastability_state: str = "RUL_UNAVAILABLE"
    warning_forecastability_score: float | None = None
    warning_serviceable_intent: bool = False
    warning_hard_eligible: bool = False
    warning_selector_active: bool = False
    warning_withholding_reason_code: str | None = None
    warning_withholding_reasons: tuple[str, ...] = ()
    warning_raw_point_hours: float | None = None
    warning_corrected_point_hours: float | None = None
    warning_calibration_stratum: str | None = None
    critical_forecastability_state: str = "RUL_UNAVAILABLE"
    critical_forecastability_score: float | None = None
    critical_serviceable_intent: bool = False
    critical_hard_eligible: bool = False
    critical_selector_active: bool = False
    critical_withholding_reason_code: str | None = None
    critical_withholding_reasons: tuple[str, ...] = ()
    critical_raw_point_hours: float | None = None
    critical_corrected_point_hours: float | None = None
    critical_calibration_stratum: str | None = None


class VVB001RULEstimator:
    """Online, causal RUL estimator for WARNING and CRITICAL threshold crossing.

    The estimator only learns from trusted runtime predictions.  Sensor-quality quarantined rows
    are excluded, and their displayed RUL is held at the last trusted estimate.  A statistically
    meaningful upward score trend is required before an ETA is emitted; otherwise the estimate is
    explicitly unavailable.

    These ETAs are trend extrapolations and remain provisional until validated against real plant
    lifecycle/maintenance outcomes.
    """

    def __init__(
        self,
        warning_boundary: float,
        critical_boundary: float,
        *,
        trend_window_hours: float = 12.0,
        min_history_hours: float = 2.0,
        min_points: int = 10,
        min_slope_per_hour: float = 0.001,
        max_forecast_hours: float = 720.0,
        confidence_z: float = 1.96,
    ) -> None:
        warning_boundary = float(warning_boundary)
        critical_boundary = float(critical_boundary)
        if not (0.0 <= warning_boundary < critical_boundary <= 1.0):
            raise ValueError("RUL boundaries must satisfy 0 <= warning < critical <= 1")
        if trend_window_hours <= 0 or min_history_hours <= 0:
            raise ValueError("RUL history windows must be positive")
        if min_points < 3:
            raise ValueError("RUL min_points must be at least 3")
        if min_slope_per_hour <= 0 or max_forecast_hours <= 0:
            raise ValueError("RUL slope/horizon configuration must be positive")

        self.warning_boundary = warning_boundary
        self.critical_boundary = critical_boundary
        self.trend_window_hours = float(trend_window_hours)
        self.min_history_hours = float(min_history_hours)
        self.min_points = int(min_points)
        self.min_slope_per_hour = float(min_slope_per_hour)
        self.max_forecast_hours = float(max_forecast_hours)
        self.confidence_z = float(confidence_z)

        self._history: dict[str, deque[tuple[datetime, float]]] = defaultdict(deque)
        self._last_prediction: dict[str, RULPrediction] = {}

    def reset_machine(self, machine_key: str) -> None:
        self._history.pop(machine_key, None)
        self._last_prediction.pop(machine_key, None)

    def _prune(self, machine_key: str, timestamp: datetime) -> None:
        history = self._history[machine_key]
        cutoff_seconds = self.trend_window_hours * 3600.0
        while history and (timestamp - history[0][0]).total_seconds() > cutoff_seconds:
            history.popleft()

    @staticmethod
    def _finite(value: float | None) -> bool:
        return value is not None and math.isfinite(float(value))

    def _unavailable(
        self,
        reason: str,
        *,
        history_hours: float,
        trusted_points: int,
        slope: float | None = None,
        r2: float | None = None,
        state_source: str = "LIVE_TREND",
    ) -> RULPrediction:
        return RULPrediction(
            estimated_hours_to_warning=None,
            warning_lower_hours=None,
            warning_upper_hours=None,
            estimated_hours_to_critical=None,
            critical_lower_hours=None,
            critical_upper_hours=None,
            reliability="UNAVAILABLE",
            reason=reason,
            trend_score_per_hour=slope,
            trend_r2=r2,
            history_hours=float(history_hours),
            trusted_points=int(trusted_points),
            state_source=state_source,
        )

    def _eta_interval(
        self,
        current_score: float,
        boundary: float,
        slope: float,
        slope_se: float | None,
    ) -> tuple[float | None, float | None, float | None]:
        if current_score >= boundary:
            return 0.0, 0.0, 0.0
        delta = boundary - current_score
        eta = delta / slope
        if eta < 0 or eta > self.max_forecast_hours:
            return None, None, None

        if slope_se is None or not math.isfinite(slope_se):
            return float(eta), None, None
        slope_hi = slope + self.confidence_z * slope_se
        slope_lo = slope - self.confidence_z * slope_se
        lower = delta / slope_hi if slope_hi > self.min_slope_per_hour else None
        upper = delta / slope_lo if slope_lo > self.min_slope_per_hour else None
        if lower is not None and lower > self.max_forecast_hours:
            lower = None
        if upper is not None and upper > self.max_forecast_hours:
            upper = None
        return (
            float(eta),
            float(max(0.0, lower)) if lower is not None else None,
            float(max(0.0, upper)) if upper is not None else None,
        )

    def update(
        self,
        *,
        machine_key: str,
        timestamp: datetime,
        degradation_score: float,
        predicted_status: str,
        features: dict[str, object] | None = None,
        trusted: bool = True,
        state_reset: bool = False,
        raw_safety_override: bool = False,
    ) -> RULPrediction:
        if state_reset:
            self.reset_machine(machine_key)

        # During sensor-quality quarantine, do not contaminate the trend history.  Display the
        # previous trusted RUL rather than recomputing from provisional telemetry.
        if not trusted:
            previous = self._last_prediction.get(machine_key)
            if previous is None:
                return self._unavailable(
                    "sensor_quality_quarantine_no_trusted_rul_yet",
                    history_hours=0.0,
                    trusted_points=0,
                    state_source="HELD_NO_TRUSTED_RUL",
                )
            return RULPrediction(
                **{
                    **previous.__dict__,
                    "reason": "sensor_quality_quarantine_held_last_trusted_rul",
                    "state_source": "HELD_LAST_TRUSTED_RUL",
                }
            )

        score = float(degradation_score)
        if not math.isfinite(score):
            result = self._unavailable(
                "non_finite_degradation_score",
                history_hours=0.0,
                trusted_points=0,
            )
            self._last_prediction[machine_key] = result
            return result

        history = self._history[machine_key]
        if history and timestamp <= history[-1][0]:
            # Warm/replay paths should be ordered.  Ignore duplicate/out-of-order timestamps rather
            # than allowing future information or zero-time duplicates into the trend fit.
            if timestamp == history[-1][0]:
                history[-1] = (timestamp, score)
            else:
                result = self._unavailable(
                    "out_of_order_timestamp_not_used_for_rul",
                    history_hours=(history[-1][0] - history[0][0]).total_seconds() / 3600.0 if len(history) > 1 else 0.0,
                    trusted_points=len(history),
                )
                self._last_prediction[machine_key] = result
                return result
        else:
            history.append((timestamp, score))
        self._prune(machine_key, timestamp)
        history = self._history[machine_key]

        n = len(history)
        history_hours = (history[-1][0] - history[0][0]).total_seconds() / 3600.0 if n > 1 else 0.0

        # Once the operational state is already CRITICAL, remaining time to CRITICAL is zero even
        # if a trend fit is not yet available.  Raw-safety override is explicitly identified.
        if predicted_status == "CRITICAL":
            result = RULPrediction(
                estimated_hours_to_warning=0.0,
                warning_lower_hours=0.0,
                warning_upper_hours=0.0,
                estimated_hours_to_critical=0.0,
                critical_lower_hours=0.0,
                critical_upper_hours=0.0,
                reliability="SAFETY_OVERRIDE" if raw_safety_override else "CURRENTLY_CRITICAL",
                reason="raw_safety_override_currently_critical" if raw_safety_override else "current_status_is_critical",
                trend_score_per_hour=None,
                trend_r2=None,
                history_hours=history_hours,
                trusted_points=n,
                state_source="RAW_SAFETY_OVERRIDE" if raw_safety_override else "CURRENT_STATUS",
            )
            self._last_prediction[machine_key] = result
            return result

        if n < self.min_points:
            result = self._unavailable(
                "insufficient_trusted_points",
                history_hours=history_hours,
                trusted_points=n,
            )
            self._last_prediction[machine_key] = result
            return result
        if history_hours < self.min_history_hours:
            result = self._unavailable(
                "insufficient_trusted_history_duration",
                history_hours=history_hours,
                trusted_points=n,
            )
            self._last_prediction[machine_key] = result
            return result

        t0 = history[0][0]
        x = np.asarray([(ts - t0).total_seconds() / 3600.0 for ts, _ in history], dtype=float)
        y = np.asarray([value for _, value in history], dtype=float)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        sxx = float(np.sum((x - x_mean) ** 2))
        if sxx <= 1e-12:
            result = self._unavailable(
                "insufficient_time_spread_for_trend",
                history_hours=history_hours,
                trusted_points=n,
            )
            self._last_prediction[machine_key] = result
            return result

        slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
        intercept = y_mean - slope * x_mean
        fitted = intercept + slope * x
        residuals = y - fitted
        sse = float(np.sum(residuals**2))
        sst = float(np.sum((y - y_mean) ** 2))
        r2 = 1.0 - sse / sst if sst > 1e-12 else 0.0
        slope_se = math.sqrt(max(0.0, sse / max(1, n - 2)) / sxx) if n > 2 else None

        if slope < self.min_slope_per_hour:
            result = self._unavailable(
                "no_meaningful_upward_degradation_trend",
                history_hours=history_hours,
                trusted_points=n,
                slope=slope,
                r2=r2,
            )
            self._last_prediction[machine_key] = result
            return result

        # Require at least modest evidence that the positive slope is not just score noise.
        slope_signal = slope / slope_se if slope_se is not None and slope_se > 1e-12 else float("inf")
        if slope_signal < 1.0 and r2 < 0.20:
            result = self._unavailable(
                "upward_trend_too_uncertain_for_rul",
                history_hours=history_hours,
                trusted_points=n,
                slope=slope,
                r2=r2,
            )
            self._last_prediction[machine_key] = result
            return result

        current_score = score
        warning_eta, warning_lo, warning_hi = self._eta_interval(
            current_score, self.warning_boundary, slope, slope_se
        )
        critical_eta, critical_lo, critical_hi = self._eta_interval(
            current_score, self.critical_boundary, slope, slope_se
        )

        # Current WARNING means the warning threshold has already been operationally crossed,
        # irrespective of small score/hysteresis differences.
        if predicted_status == "WARNING":
            warning_eta = warning_lo = warning_hi = 0.0

        if critical_eta is None and predicted_status != "CRITICAL":
            reason = "critical_crossing_beyond_forecast_horizon_or_uncertain"
        elif warning_eta is None and predicted_status == "NORMAL":
            reason = "warning_crossing_beyond_forecast_horizon_or_uncertain"
        else:
            reason = "trusted_degradation_trend_extrapolation_not_plant_validated"

        if slope_signal >= 2.0 and r2 >= 0.55 and history_hours >= 6.0 and n >= max(24, self.min_points):
            reliability = "HIGH"
        elif slope_signal >= 1.5 and r2 >= 0.30:
            reliability = "MEDIUM"
        else:
            reliability = "LOW"

        result = RULPrediction(
            estimated_hours_to_warning=warning_eta,
            warning_lower_hours=warning_lo,
            warning_upper_hours=warning_hi,
            estimated_hours_to_critical=critical_eta,
            critical_lower_hours=critical_lo,
            critical_upper_hours=critical_hi,
            reliability=reliability,
            reason=reason,
            trend_score_per_hour=slope,
            trend_r2=float(r2),
            history_hours=history_hours,
            trusted_points=n,
            state_source="LIVE_TREND",
        )
        self._last_prediction[machine_key] = result
        return result
