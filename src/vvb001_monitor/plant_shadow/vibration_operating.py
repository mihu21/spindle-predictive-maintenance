from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from typing import Deque

from ..sensor_quality import SensorQualityGuard
from .contracts import (
    OperatingState,
    OperatingStateSource,
    PlantObservation,
    require_aware,
)


VIBRATION_OPERATING_POLICY_VERSION = "vibration_operating_policy_v1"


@dataclass(frozen=True)
class VibrationOperatingConfig:
    short_window_seconds: float = 1_800.0
    calibration_history_seconds: float = 86_400.0
    calibration_min_samples: int = 60
    calibration_min_span_seconds: float = 14_400.0
    labeled_min_samples_per_class: int = 20
    activation_seconds: float = 120.0
    activation_confidence: float = 0.95
    minimum_energy_ratio: float = 1.8
    maximum_window_cv: float = 0.45
    maximum_impulse_ratio: float = 2.5
    maximum_crest_consistency_error: float = 0.35

    def __post_init__(self) -> None:
        if self.short_window_seconds <= 0 or self.calibration_history_seconds <= 0:
            raise ValueError("vibration operating windows must be positive")
        if self.calibration_min_samples < 10 or self.labeled_min_samples_per_class < 3:
            raise ValueError("vibration operating calibration support is too small")
        if not 0.5 <= self.activation_confidence <= 1.0:
            raise ValueError("activation confidence must be within [0.5,1]")


@dataclass(frozen=True)
class VibrationInferenceDecision:
    classification: str
    confidence: float
    reason_code: str
    calibration_state: str
    sample_count: int
    history_span_seconds: float
    window_energy: float | None
    window_variability: float | None
    production_similarity: float | None
    novelty_score: float | None
    persistent_running_seconds: float
    quiet_center: float | None
    running_center: float | None
    separation: float | None
    calibration_artifact_sha256: str | None
    policy_version: str = VIBRATION_OPERATING_POLICY_VERSION

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Sample:
    timestamp: datetime
    energy: float
    log_energy: float
    crest: float
    consistency_error: float
    temp: float


@dataclass
class _Calibration:
    quiet_center: float
    running_center: float
    quiet_scale: float
    running_scale: float
    source: str
    artifact_sha256: str


@dataclass
class _MachineState:
    samples: Deque[_Sample] = field(default_factory=deque)
    calibration_values: Deque[tuple[datetime, float]] = field(default_factory=deque)
    labeled_quiet: Deque[float] = field(default_factory=deque)
    labeled_running: Deque[float] = field(default_factory=deque)
    calibration: _Calibration | None = None
    candidate_since: datetime | None = None
    last_timestamp: datetime | None = None


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _mad(values: list[float], center: float) -> float:
    return max(0.03, float(statistics.median(abs(value - center) for value in values)))


def _artifact_hash(values: dict[str, object]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class VibrationOperatingDetector:
    """Causal, machine-specific eligibility gate for sources without equipment-state logs.

    It intentionally recognizes only a persistent production-like regime. Quiet, impulsive,
    unfamiliar, or uncalibrated vibration remains UNKNOWN and cannot advance the RUL runtime.
    """

    def __init__(self, config: VibrationOperatingConfig | None = None) -> None:
        self.config = config or VibrationOperatingConfig()
        self._state: dict[str, _MachineState] = {}

    def reset_machine(self, machine_uid: str) -> None:
        self._state.pop(machine_uid, None)

    def observe(self, observation: PlantObservation) -> tuple[PlantObservation, VibrationInferenceDecision]:
        timestamp = require_aware(observation.event_timestamp, "event_timestamp")
        state = self._state.setdefault(observation.uid, _MachineState())
        if state.last_timestamp is not None and timestamp < state.last_timestamp:
            raise ValueError("vibration operating detector received a non-causal timestamp")
        state.last_timestamp = timestamp

        energy = math.hypot(max(0.0, observation.vrms), max(0.0, observation.arms))
        log_energy = math.log1p(energy)
        recomputed_crest = observation.apeak / max(observation.arms, 1e-9)
        consistency = abs(observation.crest - recomputed_crest) / max(
            abs(observation.crest), abs(recomputed_crest), 1e-9
        )
        sample = _Sample(timestamp, energy, log_energy, observation.crest, consistency, observation.temp)
        state.samples.append(sample)
        history_cutoff = timestamp - timedelta(seconds=self.config.calibration_history_seconds)
        while state.samples and state.samples[0].timestamp < history_cutoff:
            state.samples.popleft()
        while state.calibration_values and state.calibration_values[0][0] < history_cutoff:
            state.calibration_values.popleft()

        window_cutoff = timestamp - timedelta(seconds=self.config.short_window_seconds)
        window = [item for item in state.samples if item.timestamp >= window_cutoff]
        window_energy: float | None = None
        window_cv: float | None = None
        impulse_ratio: float | None = None
        median_consistency: float | None = None
        if len(window) >= 3:
            energies = [item.energy for item in window]
            window_energy = _median(energies)
            window_cv = statistics.pstdev(energies) / max(window_energy, 1e-9)
            impulse_ratio = max(energies) / max(window_energy, 1e-9)
            median_consistency = _median([item.consistency_error for item in window])
            if (
                window_cv <= self.config.maximum_window_cv
                and impulse_ratio <= self.config.maximum_impulse_ratio
                and median_consistency <= self.config.maximum_crest_consistency_error
            ):
                state.calibration_values.append((timestamp, math.log1p(window_energy)))

        authoritative = observation.operating_state_source not in {
            OperatingStateSource.UNAVAILABLE,
            OperatingStateSource.VIBRATION_INFERENCE,
        }
        if authoritative:
            if observation.operating_state == OperatingState.RUNNING and observation.confirmed_running:
                state.labeled_running.append(log_energy)
            elif observation.operating_state in {OperatingState.OFF, OperatingState.IDLE}:
                state.labeled_quiet.append(log_energy)
            self._trim_labels(state)

        if state.calibration is None:
            state.calibration = self._fit_calibration(state)

        first_timestamp = state.samples[0].timestamp if state.samples else timestamp
        span = max(0.0, (timestamp - first_timestamp).total_seconds())
        if observation.operating_context_block_reason is not None:
            state.candidate_since = None
            decision = self._decision(
                state,
                classification="AUTHORITATIVE_BLOCKED",
                confidence=0.0,
                reason=observation.operating_context_block_reason,
                history_span=span,
                window_energy=window_energy,
                window_cv=window_cv,
            )
            return replace(observation, vibration_inference=decision.payload()), decision
        if authoritative:
            state.candidate_since = None
            decision = self._decision(
                state,
                classification=f"AUTHORITATIVE_{observation.operating_state.value}",
                confidence=observation.operating_state_confidence,
                reason="AUTHORITATIVE_OPERATING_CONTEXT",
                history_span=span,
                window_energy=window_energy,
                window_cv=window_cv,
            )
            return replace(observation, vibration_inference=decision.payload()), decision

        calibration = state.calibration
        if calibration is None:
            state.candidate_since = None
            return self._unknown(observation, self._decision(
                state,
                classification="CALIBRATING",
                confidence=0.0,
                reason="VIBRATION_CALIBRATION_INCOMPLETE",
                history_span=span,
                window_energy=window_energy,
                window_cv=window_cv,
            ))

        if window_energy is None or window_cv is None or impulse_ratio is None or median_consistency is None:
            state.candidate_since = None
            return self._unknown(observation, self._decision(
                state, "UNKNOWN", 0.0, "VIBRATION_WINDOW_INCOMPLETE", span,
                window_energy, window_cv,
            ))

        median_log = math.log1p(window_energy)
        threshold = (calibration.quiet_center + calibration.running_center) / 2.0
        running_band = max(0.35, 6.0 * calibration.running_scale)
        distance = abs(median_log - calibration.running_center)
        similarity = math.exp(-0.5 * (distance / running_band) ** 2)
        novelty = distance / running_band
        progress = max(0.0, min(1.0, (median_log - threshold) / max(
            calibration.running_center - threshold, 1e-9
        )))
        confidence = max(0.0, min(0.999, 0.90 + 0.099 * progress * similarity))
        extreme = SensorQualityGuard.is_extreme_raw(observation.to_legacy_reading(0))
        quality_ok = (
            not extreme
            and window_cv <= self.config.maximum_window_cv
            and impulse_ratio <= self.config.maximum_impulse_ratio
            and median_consistency <= self.config.maximum_crest_consistency_error
            and novelty <= 1.0
        )
        candidate = median_log > threshold and confidence >= self.config.activation_confidence and quality_ok
        if candidate:
            if state.candidate_since is None:
                state.candidate_since = timestamp
            persistent = max(0.0, (timestamp - state.candidate_since).total_seconds())
        else:
            state.candidate_since = None
            persistent = 0.0

        if candidate and persistent >= self.config.activation_seconds:
            decision = self._decision(
                state, "RUNNING_CONFIRMED", confidence, "VIBRATION_PRODUCTION_CONFIRMED",
                span, window_energy, window_cv, similarity, novelty, persistent,
            )
            inferred = replace(
                observation,
                operating_state=OperatingState.RUNNING,
                operating_state_source=OperatingStateSource.VIBRATION_INFERENCE,
                operating_state_confidence=confidence,
                vibration_inference=decision.payload(),
            )
            return inferred, decision

        if extreme or impulse_ratio > self.config.maximum_impulse_ratio:
            classification, reason = "UNKNOWN_IMPULSIVE", "VIBRATION_IMPULSIVE_OR_EXTREME"
        elif median_log <= threshold:
            classification, reason = "QUIET_NOT_RUNNING", "VIBRATION_QUIET_NOT_RUNNING"
        elif not quality_ok:
            classification, reason = "UNKNOWN_NOVEL", "VIBRATION_PATTERN_UNFAMILIAR"
        else:
            classification, reason = "PRODUCTION_PENDING", "VIBRATION_RUNNING_PERSISTENCE_PENDING"
        decision = self._decision(
            state, classification, confidence if candidate else 0.0, reason,
            span, window_energy, window_cv, similarity, novelty, persistent,
        )
        return self._unknown(observation, decision)

    def _unknown(
        self,
        observation: PlantObservation,
        decision: VibrationInferenceDecision,
    ) -> tuple[PlantObservation, VibrationInferenceDecision]:
        return replace(
            observation,
            operating_state=OperatingState.UNKNOWN,
            operating_state_source=OperatingStateSource.VIBRATION_INFERENCE,
            operating_state_confidence=0.0,
            maintenance_event_id=None,
            vibration_inference=decision.payload(),
        ), decision

    def _trim_labels(self, state: _MachineState) -> None:
        maximum = max(500, self.config.calibration_min_samples * 3)
        while len(state.labeled_quiet) > maximum:
            state.labeled_quiet.popleft()
        while len(state.labeled_running) > maximum:
            state.labeled_running.popleft()

    def _fit_calibration(self, state: _MachineState) -> _Calibration | None:
        support = self.config.labeled_min_samples_per_class
        if len(state.labeled_quiet) >= support and len(state.labeled_running) >= support:
            quiet_values = list(state.labeled_quiet)
            running_values = list(state.labeled_running)
            return self._build_calibration(quiet_values, running_values, "COMMISSIONED_LABELS")

        values = list(state.calibration_values)
        if len(values) < self.config.calibration_min_samples:
            return None
        if (values[-1][0] - values[0][0]).total_seconds() < self.config.calibration_min_span_seconds:
            return None
        points = [value for _, value in values]
        ordered = sorted(points)
        upper_index = max(1, int(len(ordered) * 0.95))
        points = ordered[:upper_index]
        low = ordered[max(0, int(len(ordered) * 0.25) - 1)]
        high = ordered[min(len(ordered) - 1, int(len(ordered) * 0.75))]
        for _ in range(30):
            lower_group = [value for value in points if abs(value - low) <= abs(value - high)]
            upper_group = [value for value in points if abs(value - low) > abs(value - high)]
            if not lower_group or not upper_group:
                return None
            next_low, next_high = _median(lower_group), _median(upper_group)
            if abs(next_low - low) + abs(next_high - high) < 1e-9:
                break
            low, high = next_low, next_high
        minimum_group = max(12, int(len(points) * 0.10))
        if len(lower_group) < minimum_group or len(upper_group) < minimum_group:
            return None
        return self._build_calibration(lower_group, upper_group, "UNSUPERVISED_BIMODAL")

    def _build_calibration(
        self,
        quiet_values: list[float],
        running_values: list[float],
        source: str,
    ) -> _Calibration | None:
        quiet_center, running_center = _median(quiet_values), _median(running_values)
        if running_center <= quiet_center:
            return None
        if math.expm1(running_center) / max(math.expm1(quiet_center), 1e-9) < self.config.minimum_energy_ratio:
            return None
        quiet_scale = _mad(quiet_values, quiet_center)
        running_scale = _mad(running_values, running_center)
        separation = running_center - quiet_center
        if separation < max(math.log(self.config.minimum_energy_ratio), 4.0 * quiet_scale):
            return None
        values = {
            "policy_version": VIBRATION_OPERATING_POLICY_VERSION,
            "quiet_center": quiet_center,
            "running_center": running_center,
            "quiet_scale": quiet_scale,
            "running_scale": running_scale,
            "source": source,
        }
        return _Calibration(
            quiet_center, running_center, quiet_scale, running_scale, source,
            _artifact_hash(values),
        )

    def _decision(
        self,
        state: _MachineState,
        classification: str,
        confidence: float,
        reason: str,
        history_span: float,
        window_energy: float | None,
        window_cv: float | None,
        similarity: float | None = None,
        novelty: float | None = None,
        persistent: float = 0.0,
    ) -> VibrationInferenceDecision:
        calibration = state.calibration
        return VibrationInferenceDecision(
            classification=classification,
            confidence=float(confidence),
            reason_code=reason,
            calibration_state=(
                calibration.source if calibration is not None else "CALIBRATING"
            ),
            sample_count=len(state.calibration_values),
            history_span_seconds=history_span,
            window_energy=window_energy,
            window_variability=window_cv,
            production_similarity=similarity,
            novelty_score=novelty,
            persistent_running_seconds=persistent,
            quiet_center=calibration.quiet_center if calibration else None,
            running_center=calibration.running_center if calibration else None,
            separation=(
                calibration.running_center - calibration.quiet_center if calibration else None
            ),
            calibration_artifact_sha256=calibration.artifact_sha256 if calibration else None,
        )
