from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import NormalDist, median
from typing import Any

from .config import ProjectConfig
from .models import AnomalyResult, PrognosticForecast, Status
from .smoothing import SmoothedSignal


PROGNOSTIC_METHOD_VERSION = "probabilistic_degradation_v6.0"
_NORMAL = NormalDist()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if isfinite(value) else default


def _weighted_median(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    ordered = sorted((float(value), max(0.0, float(weight))) for value, weight in values)
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return float(median(value for value, _ in ordered))
    threshold = total / 2.0
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= threshold:
            return value
    return ordered[-1][0]


def _mad(values: list[float], center: float) -> float:
    if len(values) < 2:
        return 0.0
    return 1.4826 * float(median(abs(float(value) - center) for value in values))


@dataclass
class _AdaptiveBaseline:
    mean: float
    variance: float
    samples: int = 0

    def update(self, value: float, alpha: float, variance_floor: float) -> None:
        value = float(value)
        if self.samples <= 0:
            delta = value - self.mean
            self.mean += alpha * delta
            self.variance = max(variance_floor, (1.0 - alpha) * self.variance + alpha * delta * delta)
            self.samples = 1
            return
        previous_mean = self.mean
        self.mean = (1.0 - alpha) * self.mean + alpha * value
        residual = value - previous_mean
        self.variance = max(
            variance_floor,
            (1.0 - alpha) * self.variance + alpha * residual * residual,
        )
        self.samples += 1

    @property
    def std(self) -> float:
        return sqrt(max(self.variance, 0.0))


class ProbabilisticDegradationForecaster:
    """Training-free probabilistic prognostics for scarce-failure-data assets.

    The model treats each configured manufacturer threshold as a first-passage
    target.  It estimates a robust persistent degradation rate from multiple
    causal time windows and models *rate uncertainty* rather than learning a
    classifier from synthetic failure labels.

    If S is the uncertain degradation rate, then a threshold at distance d is
    reached by horizon h when S >= d/h.  S is represented by a Normal
    distribution centered on the robust multi-timescale slope.  This produces
    monotonic horizon probabilities and an interpretable ETA interval without
    expensive row-by-row Monte Carlo simulation.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.settings = config.prognostics
        self.baselines: dict[str, _AdaptiveBaseline] = {}
        for key, sensor in config.sensors.items():
            span = max(sensor.warning - sensor.healthy_baseline, 1e-6)
            initial_std = max(span * self.settings.initial_baseline_std_fraction, 1e-6)
            self.baselines[key] = _AdaptiveBaseline(
                mean=float(sensor.healthy_baseline),
                variance=initial_std * initial_std,
                samples=0,
            )

    def reset_transient_state(self) -> None:
        """Lifecycle resets do not erase the learned healthy-machine baseline."""
        return None

    def update_healthy_baseline(
        self,
        values: dict[str, float],
        *,
        raw_status: Status,
        anomaly: AnomalyResult,
    ) -> None:
        if raw_status != Status.NORMAL:
            return
        if not anomaly.use_for_features or anomaly.requires_human_review:
            return
        if anomaly.anomaly_severity.upper() in {"HIGH", "CRITICAL"}:
            return
        alpha = self.settings.healthy_baseline_ewma_alpha
        for key, sensor in self.config.sensors.items():
            span = max(sensor.warning - sensor.healthy_baseline, 1e-6)
            floor_std = max(span * self.settings.minimum_baseline_std_fraction, 1e-6)
            baseline = self.baselines[key]
            gate_z = (
                self.settings.baseline_warmup_max_z
                if baseline.samples < self.settings.minimum_healthy_baseline_samples
                else self.settings.baseline_update_max_z
            )
            deviation_z = abs(values[key] - baseline.mean) / max(baseline.std, floor_std, 1e-9)
            # Do not let persistent degradation redefine the healthy reference.
            if deviation_z > gate_z:
                continue
            baseline.update(values[key], alpha, floor_std * floor_std)

    def _window_slope_evidence(
        self,
        sensor_key: str,
        features: dict[str, float],
    ) -> tuple[list[tuple[float, float]], list[dict[str, float]]]:
        weighted: list[tuple[float, float]] = []
        evidence: list[dict[str, float]] = []
        for minutes in self.settings.trend_windows_minutes:
            prefix = f"{sensor_key}__w{minutes}m"
            samples = _finite(features.get(f"{prefix}__sample_count"), 0.0)
            coverage = _finite(features.get(f"{prefix}__coverage_fraction"), 0.0)
            gap = _finite(features.get(f"{prefix}__gap_or_discontinuity"), 1.0)
            history_seconds = _finite(
                features.get(f"{prefix}__available_history_duration_seconds"), 0.0
            )
            required_seconds = max(60.0, minutes * 60.0 - self.config.target_sampling_interval_seconds)
            if samples < self.settings.minimum_samples_per_trend_window:
                continue
            if coverage < self.settings.minimum_window_coverage_fraction:
                continue
            if gap >= 0.5:
                continue
            if history_seconds < required_seconds * self.settings.minimum_window_history_fraction:
                continue
            slope = _finite(features.get(f"{prefix}__slope_per_hour"), 0.0)
            # Longer windows carry more evidence, but sqrt weighting prevents a
            # 24h window from completely suppressing a genuine recent trend.
            weight = sqrt(max(minutes, 1.0) / 60.0)
            weighted.append((slope, weight))
            evidence.append(
                {
                    "window_minutes": float(minutes),
                    "slope_per_hour": slope,
                    "weight": weight,
                    "coverage_fraction": coverage,
                    "sample_count": samples,
                }
            )
        return weighted, evidence

    def _sensor_state(
        self,
        sensor_key: str,
        signal: SmoothedSignal,
        features: dict[str, float],
    ) -> dict[str, Any]:
        sensor = self.config.sensors[sensor_key]
        weighted_slopes, window_evidence = self._window_slope_evidence(sensor_key, features)
        raw_slopes = [value for value, _ in weighted_slopes]
        robust_slope = _weighted_median(weighted_slopes)
        span = max(sensor.critical - sensor.healthy_baseline, 1e-6)
        slope_mad = _mad(raw_slopes, robust_slope)
        uncertainty_floor = span * self.settings.minimum_slope_uncertainty_fraction_per_hour
        if len(raw_slopes) <= 1:
            slope_sigma = max(
                uncertainty_floor,
                abs(robust_slope) * self.settings.single_window_relative_uncertainty,
            )
        else:
            slope_sigma = max(uncertainty_floor, slope_mad)

        sign_reference = 1.0 if robust_slope >= 0 else -1.0
        total_weight = sum(weight for _, weight in weighted_slopes)
        agreeing_weight = sum(
            weight for slope, weight in weighted_slopes
            if slope == 0.0 or slope * sign_reference >= 0.0
        )
        consistency = agreeing_weight / total_weight if total_weight > 0 else 0.0

        # Shrink inconsistent/noisy slopes toward zero.  This is deliberately
        # symmetric: downward recovery trends are retained as downward trends.
        snr = abs(robust_slope) / max(slope_sigma, 1e-12)
        evidence_strength = min(1.0, snr / self.settings.slope_snr_for_full_strength)
        shrinkage = consistency * evidence_strength
        effective_slope = robust_slope * shrinkage
        effective_sigma = max(
            slope_sigma,
            abs(robust_slope - effective_slope) * self.settings.shrinkage_uncertainty_multiplier,
        )

        baseline = self.baselines[sensor_key]
        baseline_std = max(baseline.std, span * self.settings.minimum_baseline_std_fraction)
        level = float(signal.slow_ewma)
        signed_z = (level - baseline.mean) / max(baseline_std, 1e-9)
        positive_z = max(0.0, signed_z)

        return {
            "sensor": sensor_key,
            "level": level,
            "raw_robust_slope_per_hour": robust_slope,
            "effective_slope_per_hour": effective_slope,
            "slope_sigma_per_hour": effective_sigma,
            "slope_consistency": consistency,
            "slope_snr": snr,
            "window_count": len(weighted_slopes),
            "window_evidence": window_evidence,
            "healthy_baseline": baseline.mean,
            "healthy_baseline_std": baseline_std,
            "healthy_baseline_samples": baseline.samples,
            "health_deviation_z": signed_z,
            "positive_health_deviation_z": positive_z,
        }

    @staticmethod
    def _crossing_probability(
        *,
        level: float,
        threshold: float,
        slope_mean: float,
        slope_sigma: float,
        horizon_hours: float,
    ) -> float:
        if level >= threshold:
            return 1.0
        if horizon_hours <= 0:
            return 0.0
        required_slope = (threshold - level) / horizon_hours
        if slope_sigma <= 1e-12:
            return float(slope_mean >= required_slope)
        z = (required_slope - slope_mean) / slope_sigma
        return _clamp(1.0 - _NORMAL.cdf(z), 0.0, 1.0)

    @staticmethod
    def _eta_interval(
        *,
        level: float,
        threshold: float,
        slope_mean: float,
        slope_sigma: float,
        max_horizon_hours: float,
        interval_z: float,
    ) -> tuple[float | None, float | None, float | None]:
        if level >= threshold:
            return 0.0, 0.0, 0.0
        distance = threshold - level
        if slope_mean <= 1e-12:
            median_eta = None
        else:
            median_eta = distance / slope_mean
            if median_eta > max_horizon_hours:
                median_eta = None
        upper_rate = slope_mean + interval_z * slope_sigma
        lower_rate = slope_mean - interval_z * slope_sigma
        earliest = distance / upper_rate if upper_rate > 1e-12 else None
        latest = distance / lower_rate if lower_rate > 1e-12 else None
        if earliest is not None and earliest > max_horizon_hours:
            earliest = None
        if latest is not None and latest > max_horizon_hours:
            latest = None
        return median_eta, earliest, latest

    def predict(
        self,
        *,
        values: dict[str, float],
        signals: dict[str, SmoothedSignal],
        features: dict[str, float],
        sensor_statuses: dict[str, Status],
        raw_status: Status,
        anomaly: AnomalyResult,
        available_history_seconds: float,
    ) -> PrognosticForecast:
        self.update_healthy_baseline(values, raw_status=raw_status, anomaly=anomaly)
        horizons = tuple(self.settings.forecast_horizons_hours)
        empty_prob = {hours: None for hours in horizons}

        if not self.settings.enabled:
            return PrognosticForecast(
                probability_warning=empty_prob,
                probability_critical=empty_prob.copy(),
                confidence="unavailable",
                reason="Probabilistic degradation prognostics are disabled by configuration.",
                method_version=PROGNOSTIC_METHOD_VERSION,
                withheld=True,
                withholding_reasons=("prognostics_disabled",),
            )
        if not anomaly.use_for_prediction:
            return PrognosticForecast(
                probability_warning=empty_prob,
                probability_critical=empty_prob.copy(),
                confidence="unavailable",
                reason="Prediction withheld by the sensor-integrity anomaly policy.",
                method_version=PROGNOSTIC_METHOD_VERSION,
                withheld=True,
                withholding_reasons=("sensor_integrity_anomaly",),
            )

        history_minutes = available_history_seconds / 60.0
        if history_minutes < self.settings.minimum_history_minutes:
            return PrognosticForecast(
                probability_warning=empty_prob,
                probability_critical=empty_prob.copy(),
                confidence="insufficient_history",
                reason=(
                    f"Need at least {self.settings.minimum_history_minutes:g} minutes of trusted "
                    "history before model-based threshold-crossing probabilities are reported."
                ),
                method_version=PROGNOSTIC_METHOD_VERSION,
                history_hours=history_minutes / 60.0,
                withheld=True,
                withholding_reasons=("insufficient_prognostic_history",),
            )

        states = {
            key: self._sensor_state(key, signals[key], features)
            for key in self.config.sensors
        }
        sufficiently_supported = [
            state for state in states.values()
            if state["window_count"] >= self.settings.minimum_trend_windows
        ]
        if not sufficiently_supported:
            return PrognosticForecast(
                probability_warning=empty_prob,
                probability_critical=empty_prob.copy(),
                confidence="insufficient_history",
                reason="No sensor has enough mature, gap-free trend windows for a stable degradation-rate estimate.",
                method_version=PROGNOSTIC_METHOD_VERSION,
                history_hours=history_minutes / 60.0,
                sensor_evidence=states,
                withheld=True,
                withholding_reasons=("insufficient_trend_support",),
            )

        per_sensor_warning: dict[str, dict[int, float]] = {}
        per_sensor_critical: dict[str, dict[int, float]] = {}
        per_sensor_eta: dict[str, dict[str, float | None]] = {}
        max_horizon = max(float(value) for value in horizons)
        interval_z = self.settings.eta_interval_normal_z

        for key, state in states.items():
            sensor = self.config.sensors[key]
            warning_prob: dict[int, float] = {}
            critical_prob: dict[int, float] = {}
            for hours in horizons:
                warning_prob[hours] = self._crossing_probability(
                    level=state["level"], threshold=sensor.warning,
                    slope_mean=state["effective_slope_per_hour"],
                    slope_sigma=state["slope_sigma_per_hour"], horizon_hours=float(hours),
                )
                critical_prob[hours] = self._crossing_probability(
                    level=state["level"], threshold=sensor.critical,
                    slope_mean=state["effective_slope_per_hour"],
                    slope_sigma=state["slope_sigma_per_hour"], horizon_hours=float(hours),
                )
            # Explicitly enforce monotonicity against floating-point edge cases.
            running = 0.0
            for hours in horizons:
                running = max(running, warning_prob[hours]); warning_prob[hours] = running
            running = 0.0
            for hours in horizons:
                running = max(running, critical_prob[hours]); critical_prob[hours] = running
            per_sensor_warning[key] = warning_prob
            per_sensor_critical[key] = critical_prob
            w_eta, w_early, w_late = self._eta_interval(
                level=state["level"], threshold=sensor.warning,
                slope_mean=state["effective_slope_per_hour"],
                slope_sigma=state["slope_sigma_per_hour"], max_horizon_hours=max_horizon,
                interval_z=interval_z,
            )
            c_eta, c_early, c_late = self._eta_interval(
                level=state["level"], threshold=sensor.critical,
                slope_mean=state["effective_slope_per_hour"],
                slope_sigma=state["slope_sigma_per_hour"], max_horizon_hours=max_horizon,
                interval_z=interval_z,
            )
            per_sensor_eta[key] = {
                "warning_eta_hours": w_eta,
                "warning_earliest_hours": w_early,
                "warning_latest_hours": w_late,
                "critical_eta_hours": c_eta,
                "critical_earliest_hours": c_early,
                "critical_latest_hours": c_late,
            }

        # Manufacturer status is a max-over-sensors rule.  Use the maximum
        # per-sensor crossing probability rather than an independence union,
        # which would overstate risk when sensors are correlated.
        probability_warning = {
            hours: max(per_sensor_warning[key][hours] for key in per_sensor_warning)
            for hours in horizons
        }
        probability_critical = {
            hours: max(per_sensor_critical[key][hours] for key in per_sensor_critical)
            for hours in horizons
        }

        warning_candidates = [
            (eta["warning_eta_hours"], key) for key, eta in per_sensor_eta.items()
            if eta["warning_eta_hours"] is not None
        ]
        critical_candidates = [
            (eta["critical_eta_hours"], key) for key, eta in per_sensor_eta.items()
            if eta["critical_eta_hours"] is not None
        ]
        warning_eta, warning_sensor = min(warning_candidates) if warning_candidates else (None, None)
        critical_eta, critical_sensor = min(critical_candidates) if critical_candidates else (None, None)
        warning_interval = per_sensor_eta.get(warning_sensor or "", {})
        critical_interval = per_sensor_eta.get(critical_sensor or "", {})

        # Raw manufacturer thresholds remain absolute and immediate.
        if raw_status >= Status.WARNING:
            warning_eta = 0.0
            probability_warning = {hours: 1.0 for hours in horizons}
            warning_sensor = max(
                (key for key in self.config.sensors if sensor_statuses[key] >= Status.WARNING),
                key=lambda key: values[key] / max(self.config.sensors[key].warning, 1e-9),
                default=warning_sensor,
            )
        if raw_status >= Status.CRITICAL:
            critical_eta = 0.0
            probability_critical = {hours: 1.0 for hours in horizons}
            critical_sensor = max(
                (key for key in self.config.sensors if sensor_statuses[key] >= Status.CRITICAL),
                key=lambda key: values[key] / max(self.config.sensors[key].critical, 1e-9),
                default=critical_sensor,
            )

        weighted_z_sq = 0.0
        total_weight = 0.0
        for key, state in states.items():
            weight = max(self.config.sensors[key].weight, 0.0)
            weighted_z_sq += weight * min(
                state["positive_health_deviation_z"], self.settings.health_deviation_z_cap
            ) ** 2
            total_weight += weight
        health_deviation_score = sqrt(weighted_z_sq / total_weight) if total_weight > 0 else 0.0

        support = [state for state in states.values() if state["window_count"] >= self.settings.minimum_trend_windows]
        median_consistency = float(median(state["slope_consistency"] for state in support))
        minimum_windows = min(state["window_count"] for state in support)
        if (
            history_minutes >= self.settings.high_confidence_history_minutes
            and minimum_windows >= self.settings.high_confidence_minimum_trend_windows
            and median_consistency >= self.settings.high_confidence_minimum_consistency
        ):
            confidence = "high"
        elif median_consistency >= self.settings.medium_confidence_minimum_consistency:
            confidence = "medium"
        else:
            confidence = "low"

        baseline_ready = all(
            baseline.samples >= self.settings.minimum_healthy_baseline_samples
            for baseline in self.baselines.values()
        )
        action_thresholds = {
            "probability_warning_12h": self.settings.warning_12h_action_probability,
            "probability_critical_6h": self.settings.critical_6h_action_probability,
            "probability_critical_24h": self.settings.critical_24h_action_probability,
        }
        crossings: list[str] = []
        for target, threshold in action_thresholds.items():
            kind = "warning" if "warning" in target else "critical"
            hours = int(target.rsplit("_", 1)[1][:-1])
            value = (probability_warning if kind == "warning" else probability_critical).get(hours)
            if value is not None and value >= threshold:
                crossings.append(target)

        reason = (
            f"Robust multi-timescale degradation model using {minimum_windows}+ supported trend windows; "
            f"median trend consistency={median_consistency:.2f}; health-deviation score={health_deviation_score:.2f}."
        )
        return PrognosticForecast(
            time_to_warning_hours=warning_eta,
            time_to_critical_hours=critical_eta,
            warning_earliest_hours=warning_interval.get("warning_earliest_hours") if warning_eta is not None else None,
            warning_latest_hours=warning_interval.get("warning_latest_hours") if warning_eta is not None else None,
            critical_earliest_hours=critical_interval.get("critical_earliest_hours") if critical_eta is not None else None,
            critical_latest_hours=critical_interval.get("critical_latest_hours") if critical_eta is not None else None,
            warning_forecast_sensor=warning_sensor,
            critical_forecast_sensor=critical_sensor,
            probability_warning=probability_warning,
            probability_critical=probability_critical,
            per_sensor_probability_warning=per_sensor_warning,
            per_sensor_probability_critical=per_sensor_critical,
            per_sensor_eta=per_sensor_eta,
            sensor_evidence=states,
            health_deviation_score=health_deviation_score,
            baseline_ready=baseline_ready,
            confidence=confidence,
            reason=reason,
            method_version=PROGNOSTIC_METHOD_VERSION,
            history_hours=history_minutes / 60.0,
            selected_thresholds=action_thresholds,
            probability_threshold_crossings=tuple(sorted(crossings)),
            withheld=False,
            withholding_reasons=(),
        )
