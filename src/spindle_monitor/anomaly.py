from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from statistics import median

from .config import ProjectConfig
from .models import (
    AnomalyResult,
    AnomalyType,
    FeatureHistoryAction,
    ModelAction,
    QualityStatus,
    SensorReading,
    Status,
    ValidationResult,
)


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    return median([abs(value - center) for value in values])


def _validation_type(reason: str) -> AnomalyType:
    lowered = reason.lower()
    if "duplicate timestamp" in lowered:
        return AnomalyType.DUPLICATE_TIMESTAMP
    if "earlier than" in lowered or "out of order" in lowered:
        return AnomalyType.OUT_OF_ORDER_TIMESTAMP
    if "outside valid range" in lowered or "jump limit" in lowered:
        return AnomalyType.OUT_OF_RANGE
    return AnomalyType.MALFORMED_VALUE


def invalid_anomaly_result(validation: ValidationResult) -> AnomalyResult:
    """Map authoritative input-validation failure to the common audit schema."""
    anomaly_type = _validation_type(validation.reason)
    timestamp = validation.reading.timestamp if validation.reading else None
    raw_status = validation.raw_safety_status
    safety_action = (
        "IMMEDIATE_MANUFACTURER_CRITICAL_ACTION"
        if raw_status == "CRITICAL"
        else "PRESERVE_MANUFACTURER_WARNING_ACTION"
        if raw_status == "WARNING"
        else "PRESERVE_ANY_PARSEABLE_RAW_MANUFACTURER_ALERT; DO_NOT_ASSUME_SAFE"
    )
    return AnomalyResult(
        quality_status=QualityStatus.INVALID_SENSOR_DATA,
        anomaly_type=(anomaly_type,),
        anomaly_severity="HIGH",
        anomaly_confidence=1.0,
        suspected_origin="input_or_sensor_data",
        first_observed_timestamp=timestamp,
        is_active=True,
        raw_safety_status=raw_status,
        safety_action=safety_action,
        model_action=ModelAction.REFUSE_PREDICTION,
        use_for_features=False,
        use_for_prediction=False,
        use_for_retraining=False,
        confidence_multiplier=0.0,
        supporting_evidence=(validation.reason,),
        causal_decision="Authoritative input validation rejected the row before feature calculation.",
        final_offline_classification=anomaly_type.value,
        training_eligible=False,
        training_exclusion_reason=f"invalid_sensor_data:{anomaly_type.value}",
        requires_human_review=True,
    )


@dataclass
class _PendingChange:
    first_timestamp: datetime
    baseline: float
    first_value: float
    latest_abrupt_timestamp: datetime
    recovery_observed_timestamp: datetime | None = None
    persistent: bool = False


@dataclass
class _ExternalConfirmation:
    timestamp: datetime
    evidence: str
    source: str
    actor: str


class AnomalyDetector:
    """Conservative, causal anomaly observer independent of manufacturer safety.

    Decisions use only values already received. Constancy never raises a stuck
    observation without a related operating-context response.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.settings = config.anomaly
        self.history: dict[str, deque[tuple[datetime, float]]] = {
            key: deque(maxlen=max(config.monitoring.history_size, self.settings.drift_window_samples * 2))
            for key in config.sensors
        }
        self.last_timestamp: datetime | None = None
        self.last_values: dict[str, float] = {}
        self.constant_start: dict[str, datetime | None] = {key: None for key in config.sensors}
        self.constant_value: dict[str, float | None] = {key: None for key in config.sensors}
        self.context_events: dict[str, list[datetime]] = {key: [] for key in config.sensors}
        self.longest_normal_constant_seconds: dict[str, float] = {key: 0.0 for key in config.sensors}
        self.clip_counts: dict[str, int] = {key: 0 for key in config.sensors}
        self.pending: dict[str, _PendingChange] = {}
        self.recovery_remaining = 0
        self.external_confirmations: dict[str, _ExternalConfirmation] = {}

    def reset(self) -> None:
        self.__init__(self.config)

    def confirm_stuck_sensor(
        self, sensor: str, timestamp: datetime, evidence: str, *,
        source: str = "external_diagnostic_or_human", actor: str = "",
    ) -> None:
        if sensor not in self.config.sensors or not evidence.strip():
            raise ValueError("External stuck-sensor confirmation requires a configured sensor and evidence")
        self.external_confirmations[sensor] = _ExternalConfirmation(
            timestamp, evidence.strip(), source.strip(), actor.strip()
        )

    def _snapshot(self) -> dict[str, object]:
        return {
            "threshold_origin": self.settings.threshold_origin,
            "spike_observation_seconds": self.settings.spike_observation_seconds,
            "persistent_change_seconds": self.settings.persistent_change_seconds,
            "long_gap_seconds": self.settings.long_gap_seconds,
            "sampling_interval_tolerance_seconds": self.settings.sampling_interval_tolerance_seconds,
            "minimum_contextual_evidence_count": self.settings.minimum_contextual_evidence_count,
            "multi_sensor_correlation_window_seconds": self.settings.multi_sensor_correlation_window_seconds,
            "confidence_penalties": dict(self.settings.confidence_penalties),
            "minimum_confidence_floor": self.settings.minimum_confidence_floor,
        }

    def _penalty(self, key: str) -> float:
        return max(
            self.settings.minimum_confidence_floor,
            float(self.settings.confidence_penalties.get(key, 1.0)),
        )

    @staticmethod
    def _safety(status: Status) -> tuple[str, QualityStatus]:
        if status == Status.CRITICAL:
            return "IMMEDIATE_MANUFACTURER_CRITICAL_ACTION", QualityStatus.CONFIRMED_MACHINE_CRITICAL
        if status == Status.WARNING:
            return "PRESERVE_MANUFACTURER_WARNING_ACTION", QualityStatus.VALID_WITH_OBSERVATION
        return "CONTINUE_MANUFACTURER_MONITORING", QualityStatus.NORMAL

    def _result(
        self, *, timestamp: datetime, raw_status: Status,
        quality: QualityStatus, types: list[AnomalyType], severity: str,
        confidence: float, origin: str, sensors: list[str], action: ModelAction,
        evidence: list[str], first: datetime | None = None, active: bool = True,
        final: str = "PENDING_OR_SAME_AS_CAUSAL", confidence_multiplier: float = 1.0,
        training_eligible: bool = True, exclusion: str = "", review: bool = False,
        causal: str = "", suspicion: datetime | None = None,
        confirmed: datetime | None = None, resolved: datetime | None = None,
        confirmation_source: str = "", confirmation_evidence: str = "",
        confirming_actor: str = "", penalty_reason: str = "",
        expected_interval: float | None = None, source_interval: float | None = None,
        missing_samples: int = 0, gap_duration: float = 0.0,
        interpolation_occurred: bool = False,
        correlation_evidence: dict[str, object] | None = None,
    ) -> AnomalyResult:
        safety_action, _ = self._safety(raw_status)
        use_features = action not in {
            ModelAction.EXCLUDE_CURRENT_READING_FROM_TREND,
            ModelAction.HOLD_FOR_CONFIRMATION,
            ModelAction.SUSPEND_PREDICTION,
            ModelAction.REFUSE_PREDICTION,
        }
        use_prediction = action not in {
            ModelAction.HOLD_FOR_CONFIRMATION,
            ModelAction.SUSPEND_PREDICTION,
            ModelAction.REFUSE_PREDICTION,
        }
        if action == ModelAction.HOLD_FOR_CONFIRMATION:
            feature_history_action = FeatureHistoryAction.HOLD_OUTSIDE_FEATURE_HISTORY
        elif not use_features:
            feature_history_action = FeatureHistoryAction.DISCARD_FROM_FEATURE_HISTORY
        else:
            feature_history_action = FeatureHistoryAction.COMMIT_TO_FEATURE_HISTORY
        return AnomalyResult(
            quality_status=quality,
            anomaly_type=tuple(dict.fromkeys(types)) or (AnomalyType.NONE,),
            anomaly_severity=severity,
            anomaly_confidence=max(0.0, min(1.0, confidence)),
            suspected_origin=origin,
            affected_sensors=tuple(sorted(set(sensors))),
            first_observed_timestamp=first or timestamp,
            confirmed_timestamp=confirmed,
            resolved_timestamp=resolved or (timestamp if not active and types else None),
            is_active=active,
            raw_safety_status=raw_status.name,
            safety_action=safety_action,
            model_action=action,
            feature_history_action=feature_history_action,
            use_for_features=use_features,
            use_for_prediction=use_prediction,
            use_for_retraining=training_eligible,
            confidence_multiplier=confidence_multiplier,
            supporting_evidence=tuple(evidence),
            configuration_snapshot=self._snapshot(),
            detector_version=self.settings.detector_version,
            causal_decision=causal or "Decision used the current and prior observations only.",
            final_offline_classification=final,
            training_eligible=training_eligible,
            training_exclusion_reason=exclusion,
            requires_human_review=review,
            decision_timestamp=timestamp,
            suspicion_timestamp=suspicion,
            confirmation_source=confirmation_source,
            confirmation_evidence=confirmation_evidence,
            confirming_actor=confirming_actor,
            expected_sampling_interval_seconds=expected_interval,
            source_sampling_interval_seconds=source_interval,
            estimated_missing_sample_count=missing_samples,
            gap_duration_seconds=gap_duration,
            interpolation_enabled=self.config.interpolation_enabled,
            interpolation_occurred=interpolation_occurred,
            confidence_penalty_reason=penalty_reason,
            configured_confidence_multiplier=confidence_multiplier,
            minimum_confidence_floor=self.settings.minimum_confidence_floor,
            correlation_evidence=dict(correlation_evidence or {}),
        )

    def _abrupt_threshold(self, sensor: str) -> float:
        metadata = self.settings.sensors[sensor]
        limits = self.config.sensors[sensor]
        return max(
            metadata.measurement_resolution * 8.0,
            (limits.critical - limits.healthy_baseline)
            * self.settings.spike_delta_fraction_of_threshold_span,
        )

    def _update_context_and_constancy(self, timestamp: datetime, values: dict[str, float]) -> None:
        for sensor, value in values.items():
            metadata = self.settings.sensors[sensor]
            previous = self.constant_value[sensor]
            if previous is None:
                self.constant_start[sensor] = timestamp
                self.constant_value[sensor] = value
            elif abs(value - previous) > metadata.measurement_resolution * 0.5:
                start = self.constant_start[sensor]
                if start is not None:
                    self.longest_normal_constant_seconds[sensor] = max(
                        self.longest_normal_constant_seconds[sensor],
                        (timestamp - start).total_seconds(),
                    )
                self.constant_start[sensor] = timestamp
                self.constant_value[sensor] = value
                self.context_events[sensor].clear()

        if self.last_values:
            changed = {
                key for key, value in values.items()
                if abs(value - self.last_values[key])
                >= max(self.settings.sensors[key].meaningful_context_change,
                       self.settings.sensors[key].measurement_resolution * 5.0)
            }
            for sensor, metadata in self.settings.sensors.items():
                if changed.intersection(metadata.related_sensors):
                    self.context_events[sensor].append(timestamp)

    def _stuck_evidence(self, timestamp: datetime, values: dict[str, float]) -> tuple[list[str], list[str], list[str]]:
        possible: list[str] = []
        suspected: list[str] = []
        evidence: list[str] = []
        for sensor, value in values.items():
            metadata = self.settings.sensors[sensor]
            start = self.constant_start[sensor]
            if start is None or self.constant_value[sensor] is None:
                continue
            duration = (timestamp - start).total_seconds()
            context = [event for event in self.context_events[sensor] if event >= start]
            evidence_duration_multiplier = 1.0 if metadata.exact_raw_values_available else 2.0
            unusual = duration > max(
                metadata.stuck_observation_seconds * evidence_duration_multiplier,
                self.longest_normal_constant_seconds[sensor] * 1.5,
            )
            if unusual and context:
                possible.append(sensor)
                evidence.append(
                    f"{sensor} repeated exact raw value {value!r} for {duration:.0f}s while related context changed"
                )
                if not metadata.exact_raw_values_available:
                    evidence[-1] = (
                        f"{sensor} repeated a precision-limited value ({metadata.decimal_precision} decimals, "
                        f"resolution {metadata.measurement_resolution:g}); repetition evidence is weakened"
                    )
            last_context = context[-1] if context else None
            response_elapsed = (timestamp - last_context).total_seconds() if last_context else 0.0
            if (
                unusual
                and duration >= metadata.stuck_suspicion_seconds * evidence_duration_multiplier
                and len(context) >= self.settings.minimum_contextual_evidence_count + int(not metadata.exact_raw_values_available)
                and response_elapsed >= metadata.expected_response_delay_seconds
            ):
                suspected.append(sensor)
        return possible, suspected, evidence

    def _clipping(self, values: dict[str, float]) -> tuple[list[str], list[str]]:
        affected: list[str] = []
        evidence: list[str] = []
        for sensor, value in values.items():
            metadata = self.settings.sensors[sensor]
            tolerance = metadata.measurement_resolution * self.settings.clipping_boundary_tolerance_multiplier
            at_boundary = (
                abs(value - metadata.physical_minimum) <= tolerance
                or abs(value - metadata.physical_maximum) <= tolerance
            )
            self.clip_counts[sensor] = self.clip_counts[sensor] + 1 if at_boundary else 0
            if self.clip_counts[sensor] >= self.settings.clipping_minimum_samples:
                affected.append(sensor)
                evidence.append(f"{sensor} repeated at configured physical/instrument boundary")
        return affected, evidence

    def _noise(self, sensor: str, values: list[float]) -> tuple[bool, str]:
        window = self.settings.noise_window_samples
        if len(values) < window * 2:
            return False, ""
        recent = values[-window:]
        prior = values[-window * 2:-window]
        # Residuals from the endpoint trend avoid calling a monotonic rise noise.
        def residual_mad(series: list[float]) -> float:
            slope = (series[-1] - series[0]) / max(1, len(series) - 1)
            residuals = [value - (series[0] + slope * index) for index, value in enumerate(series)]
            return _mad(residuals)
        recent_noise = residual_mad(recent)
        baseline_noise = max(residual_mad(prior), self.settings.sensors[sensor].measurement_resolution)
        sign_changes = sum(
            1 for left, middle, right in zip(recent, recent[1:], recent[2:])
            if (middle - left) * (right - middle) < 0
        )
        noisy = recent_noise > baseline_noise * self.settings.noise_multiplier and sign_changes >= window // 4
        return noisy, f"{sensor} robust residual MAD rose to {recent_noise:.6g} from {baseline_noise:.6g}"

    def detect(
        self, reading: SensorReading, raw_status: Status, *, interpolated: bool = False,
        source_sampling_interval_seconds: float | None = None,
        input_features_available: bool = True,
    ) -> AnomalyResult:
        timestamp = reading.timestamp
        values = reading.values()
        if not self.settings.enabled:
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.NORMAL,
                types=[], severity="NONE", confidence=0.0, origin="none", sensors=[],
                action=ModelAction.USE_NORMALLY, evidence=["Anomaly handling disabled by configuration."],
                active=False, causal="Detector disabled; existing replay behavior is preserved.",
            )
            self.last_timestamp, self.last_values = timestamp, values
            for key, value in values.items():
                self.history[key].append((timestamp, value))
            return result

        gap = (
            float(source_sampling_interval_seconds)
            if source_sampling_interval_seconds is not None
            else 0.0 if self.last_timestamp is None else (timestamp - self.last_timestamp).total_seconds()
        )
        self._update_context_and_constancy(timestamp, values)

        # External diagnostic or human evidence is the only confirmation path.
        confirmed = [
            sensor for sensor, item in self.external_confirmations.items()
            if item.timestamp <= timestamp
        ]
        if confirmed:
            confirmations = [self.external_confirmations[sensor] for sensor in confirmed]
            evidence = [item.evidence for item in confirmations]
            confirmation_time = min(item.timestamp for item in confirmations)
            first = min(
                (self.constant_start[sensor] or confirmation_time for sensor in confirmed),
                default=confirmation_time,
            )
            result = self._result(
                timestamp=timestamp, raw_status=raw_status,
                quality=QualityStatus.SUSPECTED_SENSOR_ANOMALY,
                types=[AnomalyType.STUCK_SENSOR_CONFIRMED], severity="CRITICAL", confidence=1.0,
                origin="externally_confirmed_sensor_fault", sensors=confirmed,
                action=ModelAction.SUSPEND_PREDICTION, evidence=evidence,
                first=first, confirmed=confirmation_time,
                confirmation_source="|".join(sorted({item.source for item in confirmations})),
                confirmation_evidence=" | ".join(evidence),
                confirming_actor="|".join(sorted({item.actor for item in confirmations if item.actor})),
                confidence_multiplier=self._penalty("invalid"), penalty_reason="confirmed_sensor_fault",
                training_eligible=False,
                exclusion="confirmed_sensor_fault", review=True,
                causal="External diagnostic or human-confirmed evidence established sensor failure.",
                final=AnomalyType.STUCK_SENSOR_CONFIRMED.value,
            )
            return self._finish(timestamp, values, result)

        expected_interval = float(self.config.target_sampling_interval_seconds)
        missing_samples = max(0, int(round(gap / expected_interval)) - 1) if gap > 0 else 0
        if gap >= self.settings.long_gap_seconds or not input_features_available:
            self.recovery_remaining = self.settings.recovery_valid_samples
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.DATA_UNAVAILABLE,
                types=[AnomalyType.LONG_DATA_GAP], severity="HIGH", confidence=1.0,
                origin="communication_or_data_availability", sensors=list(values),
                action=ModelAction.SUSPEND_PREDICTION,
                evidence=[f"source gap {gap:.0f}s exceeds {self.settings.long_gap_seconds:.0f}s"],
                first=self.last_timestamp or timestamp,
                confidence_multiplier=self._penalty("long_gap"), penalty_reason="long_data_gap",
                expected_interval=expected_interval, source_interval=gap,
                missing_samples=missing_samples, gap_duration=gap,
                interpolation_occurred=interpolated, training_eligible=False,
                exclusion="long_data_gap_or_contaminated_recovery", review=True,
                causal="Long outage preserves lifecycle state but suspends the operational prediction.",
            )
            return self._finish(timestamp, values, result)

        if self.recovery_remaining > 0:
            self.recovery_remaining -= 1
            action = ModelAction.SUSPEND_PREDICTION if self.recovery_remaining else ModelAction.USE_WITH_REDUCED_CONFIDENCE
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.VALID_WITH_OBSERVATION,
                types=[AnomalyType.LONG_DATA_GAP], severity="MEDIUM", confidence=1.0,
                origin="post_gap_recovery", sensors=list(values), action=action,
                evidence=[f"{self.recovery_remaining} additional valid recovery sample(s) required"],
                confidence_multiplier=(self._penalty("suspected") if self.recovery_remaining == 0 else self._penalty("long_gap")),
                penalty_reason="post_long_gap_recovery",
                training_eligible=False, exclusion="post_long_gap_recovery", review=False,
                causal="Confidence is not fully restored immediately after a long outage.",
            )
            return self._finish(timestamp, values, result)

        short_gap = (
            missing_samples > 0
            and gap > expected_interval + self.settings.sampling_interval_tolerance_seconds
            and gap < self.settings.long_gap_seconds
        )
        if interpolated or short_gap:
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.VALID_WITH_OBSERVATION,
                types=[AnomalyType.SHORT_DATA_GAP], severity="LOW", confidence=1.0,
                origin=("safe_causal_interpolation" if interpolated else "source_cadence_gap"),
                sensors=list(values),
                action=ModelAction.USE_WITH_REDUCED_CONFIDENCE,
                evidence=[
                    f"source interval={gap:.0f}s; expected={expected_interval:.0f}s; "
                    f"estimated missing samples={missing_samples}; interpolation enabled="
                    f"{self.config.interpolation_enabled}; occurred={interpolated}"
                ],
                first=self.last_timestamp or timestamp,
                confidence_multiplier=self._penalty("short_gap"), penalty_reason="short_data_gap",
                expected_interval=expected_interval, source_interval=gap,
                missing_samples=missing_samples, gap_duration=gap,
                interpolation_occurred=interpolated,
                training_eligible=not interpolated,
                exclusion=("interpolated_short_gap_pending_policy_review" if interpolated else ""),
                review=False,
                causal=(
                    "Existing safe resampler generated this normal-to-normal row."
                    if interpolated else
                    "Missing cadence samples were detected without fabricating rows because interpolation was disabled."
                ),
            )
            return self._finish(timestamp, values, result)

        clipping, clip_evidence = self._clipping(values)
        if clipping:
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.SUSPECTED_SENSOR_ANOMALY,
                types=[AnomalyType.CLIPPING_SUSPECTED], severity="MEDIUM", confidence=0.8,
                origin="sensor_or_instrument_boundary", sensors=clipping,
                action=ModelAction.USE_WITH_REDUCED_CONFIDENCE, evidence=clip_evidence,
                confidence_multiplier=self._penalty("suspected"), penalty_reason="clipping_suspected",
                training_eligible=False,
                exclusion="clipping_suspected", review=True,
                causal="Repeated boundary values are retained; no values beyond the limit are reconstructed.",
            )
            return self._finish(timestamp, values, result)

        possible_stuck, suspected_stuck, stuck_evidence = self._stuck_evidence(timestamp, values)
        if suspected_stuck:
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.SUSPECTED_SENSOR_ANOMALY,
                types=[AnomalyType.STUCK_SENSOR_SUSPECTED], severity="MEDIUM", confidence=0.75,
                origin="sensor_response_uncertain", sensors=suspected_stuck,
                action=ModelAction.USE_WITH_REDUCED_CONFIDENCE, evidence=stuck_evidence,
                first=min(
                    (self.constant_start[sensor] or timestamp for sensor in suspected_stuck),
                    default=timestamp,
                ),
                suspicion=timestamp,
                confidence_multiplier=self._penalty("suspected"), penalty_reason="stuck_sensor_suspected",
                training_eligible=False,
                exclusion="stuck_sensor_suspected_pending_review", review=True,
                causal="Constancy persisted through multiple context changes and the physical response delay.",
            )
            return self._finish(timestamp, values, result)
        if possible_stuck:
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.VALID_WITH_OBSERVATION,
                types=[AnomalyType.POSSIBLE_STUCK_SENSOR], severity="LOW", confidence=0.4,
                origin="uncertain", sensors=possible_stuck, action=ModelAction.USE_NORMALLY,
                evidence=stuck_evidence,
                first=min(
                    (self.constant_start[sensor] or timestamp for sensor in possible_stuck),
                    default=timestamp,
                ),
                suspicion=timestamp,
                confidence_multiplier=self._penalty("observation"), penalty_reason="possible_stuck_observation",
                training_eligible=True,
                causal="Exact repetition plus one context change starts observation; constancy alone is insufficient.",
            )
            return self._finish(timestamp, values, result)

        prior_values = {key: [value for _, value in self.history[key]] for key in values}
        abrupt: list[str] = []
        evidence: list[str] = []
        resolved: list[str] = []
        resolved_persistent: list[str] = []
        persistent: list[str] = []
        for sensor, value in values.items():
            prior = prior_values[sensor]
            baseline = median(prior[-10:]) if len(prior) >= 5 else value
            threshold = self._abrupt_threshold(sensor)
            pending = self.pending.get(sensor)
            if pending is not None:
                recovery_tolerance = max(
                    self.settings.sensors[sensor].measurement_resolution
                    * self.settings.spike_recovery_tolerance_multiplier,
                    threshold * 0.25,
                )
                elapsed = (timestamp - pending.first_timestamp).total_seconds()
                if abs(value - pending.baseline) <= recovery_tolerance:
                    pending.recovery_observed_timestamp = pending.recovery_observed_timestamp or timestamp
                    if pending.persistent:
                        resolved_persistent.append(sensor)
                        evidence.append(f"{sensor} persistent change returned to its pre-change baseline")
                    elif elapsed >= self.settings.spike_observation_seconds:
                        resolved.append(sensor)
                        evidence.append(f"{sensor} returned toward its causal pre-change baseline after observation window")
                    else:
                        abrupt.append(sensor)
                        evidence.append(
                            f"{sensor} returned toward baseline but remains held until "
                            f"{self.settings.spike_observation_seconds:.0f}s causal observation completes"
                        )
                elif (timestamp - pending.first_timestamp).total_seconds() >= self.settings.persistent_change_seconds:
                    pending.persistent = True
                    persistent.append(sensor)
                    evidence.append(f"{sensor} abrupt change persisted beyond confirmation window")
                else:
                    abrupt.append(sensor)
                pending.latest_abrupt_timestamp = timestamp
            elif len(prior) >= 5 and abs(value - baseline) >= threshold:
                self.pending[sensor] = _PendingChange(timestamp, baseline, value, timestamp)
                abrupt.append(sensor)
                evidence.append(f"{sensor} changed {value - baseline:+.6g} from recent median {baseline:.6g}")

        correlation_candidates = sorted(set(abrupt + persistent))
        correlation_times = {
            sensor: self.pending[sensor].first_timestamp
            for sensor in correlation_candidates if sensor in self.pending
        }
        coherent: list[str] = []
        if len(correlation_times) >= 2:
            ordered_times = sorted(correlation_times.values())
            separation = (ordered_times[-1] - ordered_times[0]).total_seconds()
            if separation <= self.settings.multi_sensor_correlation_window_seconds:
                coherent = sorted(correlation_times)
        if len(coherent) >= 2:
            first_times = {sensor: correlation_times[sensor].isoformat() for sensor in coherent}
            latest_times = {
                sensor: self.pending[sensor].latest_abrupt_timestamp.isoformat()
                for sensor in coherent
            }
            separation = (
                max(correlation_times.values()) - min(correlation_times.values())
            ).total_seconds()
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.POSSIBLE_MACHINE_EVENT,
                types=[AnomalyType.CORRELATED_ABRUPT_CHANGE, AnomalyType.PERSISTENT_STEP_CHANGE],
                severity="HIGH", confidence=0.8, origin="possible_abrupt_machine_event",
                sensors=coherent, action=ModelAction.USE_WITH_REDUCED_CONFIDENCE, evidence=evidence,
                first=min(correlation_times.values()),
                confidence_multiplier=self._penalty("machine_event"), penalty_reason="correlated_abrupt_machine_event",
                correlation_evidence={
                    "first_abrupt_timestamp_by_sensor": first_times,
                    "latest_abrupt_timestamp_by_sensor": latest_times,
                    "included_sensors": coherent,
                    "maximum_timestamp_separation_seconds": separation,
                    "configured_correlation_window_seconds": self.settings.multi_sensor_correlation_window_seconds,
                },
                training_eligible=False,
                exclusion="possible_abrupt_machine_event", review=True,
                causal=(
                    "Multiple sensors changed coherently and remain physically valid. Treat this as a "
                    "possible machine/regime event: preserve the forecast for situational awareness, "
                    "downgrade confidence, and keep the row out of retraining until reviewed."
                ),
            )
            return self._finish(timestamp, values, result)
        if persistent:
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.SUSPECTED_SENSOR_ANOMALY,
                types=[AnomalyType.PERSISTENT_STEP_CHANGE, AnomalyType.SENSOR_DISAGREEMENT,
                       AnomalyType.ORIGIN_UNCERTAIN], severity="HIGH", confidence=0.7,
                origin="machine_or_sensor_or_operating_regime", sensors=persistent,
                action=ModelAction.USE_WITH_REDUCED_CONFIDENCE, evidence=evidence,
                first=min(self.pending[sensor].first_timestamp for sensor in persistent),
                suspicion=timestamp,
                confidence_multiplier=self._penalty("machine_event"), penalty_reason="persistent_change_origin_uncertain",
                training_eligible=False,
                exclusion="persistent_change_origin_uncertain", review=True,
                causal=(
                    "A persistent physically valid single-channel change can be real degradation or an "
                    "operating-regime shift. It is not a confirmed sensor fault, so forecasting continues "
                    "at reduced confidence while the row remains excluded from retraining pending review."
                ),
            )
            return self._finish(timestamp, values, result)
        if abrupt:
            first = min(self.pending[sensor].first_timestamp for sensor in abrupt)
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.SUSPECTED_SENSOR_ANOMALY,
                types=[AnomalyType.SINGLE_SAMPLE_SPIKE, AnomalyType.SENSOR_DISAGREEMENT,
                       AnomalyType.ORIGIN_UNCERTAIN], severity="MEDIUM", confidence=0.45,
                origin="uncertain", sensors=abrupt, action=ModelAction.HOLD_FOR_CONFIRMATION,
                evidence=evidence, first=first,
                confidence_multiplier=self._penalty("suspected"), penalty_reason="abrupt_reading_pending_confirmation",
                training_eligible=False, exclusion="abrupt_reading_pending_confirmation", review=True,
                causal="First suspicious reading is held using current/past data only; no future sample was consulted.",
            )
            return self._finish(timestamp, values, result)
        if resolved:
            first = min(self.pending[sensor].first_timestamp for sensor in resolved)
            for sensor in resolved:
                del self.pending[sensor]
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.VALID_WITH_OBSERVATION,
                types=[AnomalyType.SINGLE_SAMPLE_SPIKE], severity="LOW", confidence=0.8,
                origin="transient_origin_uncertain", sensors=resolved,
                action=ModelAction.USE_WITH_REDUCED_CONFIDENCE, evidence=evidence,
                active=False, first=first, resolved=timestamp,
                final="RESOLVED_TRANSIENT_SPIKE",
                confidence_multiplier=self._penalty("short_gap"), penalty_reason="resolved_transient_history_contamination_guard",
                training_eligible=False, exclusion="resolved_transient_spike",
                causal="A later causal reading resolved the prior pending event; the original decision remains audited.",
            )
            return self._finish(timestamp, values, result)
        if resolved_persistent:
            first = min(self.pending[sensor].first_timestamp for sensor in resolved_persistent)
            for sensor in resolved_persistent:
                del self.pending[sensor]
            result = self._result(
                timestamp=timestamp, raw_status=raw_status,
                quality=QualityStatus.VALID_WITH_OBSERVATION,
                types=[AnomalyType.PERSISTENT_STEP_CHANGE], severity="MEDIUM", confidence=0.8,
                origin="resolved_machine_sensor_or_regime_change", sensors=resolved_persistent,
                action=ModelAction.USE_WITH_REDUCED_CONFIDENCE, evidence=evidence,
                active=False, first=first, resolved=timestamp,
                final="RESOLVED_PERSISTENT_CHANGE",
                confidence_multiplier=self._penalty("suspected"),
                penalty_reason="resolved_persistent_change_recovery",
                training_eligible=False, exclusion="persistent_change_episode",
                causal="The persistent episode resolved; excluded feature-history decisions were not backfilled.",
            )
            return self._finish(timestamp, values, result)

        noisy: list[str] = []
        noise_evidence: list[str] = []
        for sensor in values:
            flagged, item = self._noise(sensor, prior_values[sensor] + [values[sensor]])
            if flagged:
                noisy.append(sensor)
                noise_evidence.append(item)
        if noisy:
            result = self._result(
                timestamp=timestamp, raw_status=raw_status, quality=QualityStatus.SUSPECTED_SENSOR_ANOMALY,
                types=[AnomalyType.EXCESSIVE_NOISE], severity="MEDIUM", confidence=0.7,
                origin="sensor_or_process_variability", sensors=noisy,
                action=ModelAction.USE_WITH_REDUCED_CONFIDENCE, evidence=noise_evidence,
                confidence_multiplier=self._penalty("noise"), penalty_reason="excessive_noise",
                training_eligible=False, exclusion="excessive_noise", review=True,
                causal="Robust detrended variability rose materially; raw values and alerts are retained.",
            )
            return self._finish(timestamp, values, result)

        drifted: list[str] = []
        drift_evidence: list[str] = []
        window = self.settings.drift_window_samples
        for sensor in values:
            series = (prior_values[sensor] + [values[sensor]])[-window:]
            if len(series) < window:
                continue
            differences = [right - left for left, right in zip(series, series[1:])]
            nonzero = [value for value in differences if abs(value) > self.settings.sensors[sensor].measurement_resolution * 0.5]
            if not nonzero:
                continue
            same_direction = max(
                sum(value > 0 for value in nonzero), sum(value < 0 for value in nonzero)
            ) / len(nonzero)
            limits = self.config.sensors[sensor]
            required_span = (
                limits.critical - limits.healthy_baseline
            ) * self.settings.drift_minimum_span_fraction
            if same_direction >= 0.85 and abs(series[-1] - series[0]) >= required_span:
                drifted.append(sensor)
                drift_evidence.append(
                    f"{sensor} moved gradually {series[-1] - series[0]:+.6g} over {window} samples"
                )
        if drifted:
            result = self._result(
                timestamp=timestamp, raw_status=raw_status,
                quality=QualityStatus.SUSPECTED_SENSOR_ANOMALY,
                types=[AnomalyType.DRIFT_SUSPECTED], severity="MEDIUM", confidence=0.55,
                origin="sensor_drift_or_real_degradation", sensors=drifted,
                action=ModelAction.USE_WITH_REDUCED_CONFIDENCE, evidence=drift_evidence,
                confidence_multiplier=self._penalty("suspected"), penalty_reason="drift_suspected",
                training_eligible=False,
                exclusion="drift_suspected_pending_calibration_review", review=True,
                causal="Conservative drift report only; no correction or lifecycle change was applied.",
            )
            return self._finish(timestamp, values, result)

        safety_action, _ = self._safety(raw_status)
        quality = QualityStatus.NORMAL
        result = AnomalyResult(
            quality_status=quality,
            raw_safety_status=raw_status.name,
            safety_action=safety_action,
            configuration_snapshot=self._snapshot(),
            detector_version=self.settings.detector_version,
            confirmed_timestamp=timestamp if raw_status == Status.CRITICAL else None,
            causal_decision="No configured anomaly evidence; constant or low-variance values alone remain valid.",
        )
        return self._finish(timestamp, values, result)

    def _finish(self, timestamp: datetime, values: dict[str, float], result: AnomalyResult) -> AnomalyResult:
        self.last_timestamp = timestamp
        self.last_values = dict(values)
        for key, value in values.items():
            self.history[key].append((timestamp, value))
        return result
