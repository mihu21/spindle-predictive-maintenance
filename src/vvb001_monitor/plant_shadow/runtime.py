from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..config import SensorConfig
from ..features import FeatureEngine
from ..models import ProcessedReading, SourceRecord
from ..monitor import VVB001Monitor
from ..predictor import VVB001Predictor
from ..sensor_quality import SensorQualityGuard
from ..validation import VVB001Validator
from .contracts import ForecastState, ObservationDisposition, PlantObservation, utc_now
from .manifest import sha256_file
from .operating_context import OperatingContextClock, OperatingContextDecision
from .storage import ProcessingCommit
from .vibration_operating import VibrationOperatingConfig, VibrationOperatingDetector


class FrozenSourceRuntime:
    """One stateful frozen monitor instance owned by exactly one source."""

    def __init__(
        self,
        source_key: str,
        model_path: str | Path,
        sensor_config: SensorConfig,
        *,
        deployment_id: str,
        runtime_manifest_id: str,
        vibration_operating_config: VibrationOperatingConfig | None = None,
    ) -> None:
        self.source_key = source_key
        self.model_path = Path(model_path)
        self.model_sha256 = sha256_file(self.model_path)
        self.sensor_config = sensor_config
        self.deployment_id = deployment_id
        self.runtime_manifest_id = runtime_manifest_id
        self.vibration_operating_config = vibration_operating_config
        self.monitor = self._new_monitor()
        self.operating_clock = OperatingContextClock()
        self.vibration_detector = VibrationOperatingDetector(vibration_operating_config)
        self.raw_validator = VVB001Validator(self.sensor_config)
        self._lifecycle_by_machine: dict[str, str] = {}

    def _new_monitor(self) -> VVB001Monitor:
        predictor = VVB001Predictor(self.model_path, self.sensor_config)
        return VVB001Monitor(
            VVB001Validator(self.sensor_config),
            FeatureEngine(self.sensor_config),
            predictor,
            SensorQualityGuard(self.sensor_config),
        )

    @staticmethod
    def _context_events(decision: OperatingContextDecision) -> tuple[dict[str, object], ...]:
        if not decision.state_transition:
            return ()
        return ({
            "event_type": "OPERATING_STATE_CHANGED",
            "severity": "INFO" if decision.operating_state.value != "UNKNOWN" else "WARNING",
            "message": (
                f"Operating state changed from {decision.previous_operating_state} "
                f"to {decision.operating_state.value}."
            ),
            "evidence": {
                "previous_operating_state": (
                    decision.previous_operating_state.value
                    if decision.previous_operating_state is not None
                    else None
                ),
                "operating_state": decision.operating_state.value,
                "operating_state_source": decision.operating_state_source,
                "operating_state_confidence": decision.operating_state_confidence,
                "reason_code": decision.reason_code,
            },
        },)

    def _held_prediction_payload(
        self,
        observation: PlantObservation,
        ingestion_id: int,
        decision: OperatingContextDecision,
    ) -> tuple[str, tuple[str, ...], dict[str, object] | None]:
        reading = observation.to_legacy_reading(ingestion_id)
        record = SourceRecord(ingestion_id, dict(observation.raw_payload), reading)
        validation = self.raw_validator.validate_record(record)
        if not validation.valid or validation.reading is None:
            return validation.status, validation.reasons, None
        extreme = SensorQualityGuard.is_extreme_raw(validation.reading)
        manufacturer_state = "CRITICAL" if extreme else None
        reason = decision.reason_code
        return validation.status, validation.reasons, {
            "deployment_id": self.deployment_id,
            "prediction_timestamp": utc_now().isoformat(),
            "health_state_model": None,
            "health_state_manufacturer": manufacturer_state,
            "health_state_source": (
                "RAW_SAFETY_OVERRIDE_OPERATING_GATE" if extreme else "OPERATING_CONTEXT_GATE"
            ),
            "connectivity_state": "ONLINE",
            "forecast_state": ForecastState.PAUSED,
            "warning_point_hours": None,
            "warning_lower_hours": None,
            "warning_upper_hours": None,
            "warning_serviceable": False,
            "warning_withhold_reason": reason,
            "critical_point_hours": None,
            "critical_lower_hours": None,
            "critical_upper_hours": None,
            "critical_serviceable": False,
            "critical_withhold_reason": reason,
            "model_version": "v2.7",
            "model_artifact_sha256": self.model_sha256,
            "runtime_manifest_id": self.runtime_manifest_id,
            "inference_latency_ms": 0.0,
            "model_state_reset_gap": False,
            "quality_status": validation.status,
            "quality_reasons": list(validation.reasons),
            "sensor_quality_status": "RAW_SAFETY_OVERRIDE" if extreme else "NOT_EVALUATED_PAUSED",
            "sensor_quality_reasons": (
                ["extreme raw telemetry remained immediately visible while prognostics were paused"]
                if extreme else []
            ),
            "features": {},
            "rul_reliability": "UNAVAILABLE",
            "rul_method": "paused_operating_context",
            "warning_forecastability_state": "RUL_UNAVAILABLE",
            "critical_forecastability_state": "RUL_UNAVAILABLE",
            "rul_time_basis": "OPERATING_HOURS",
            "vibration_operating_inference": observation.vibration_inference,
            **decision.payload(),
        }

    def process(
        self,
        observation: PlantObservation,
        ingestion_id: int,
        lifecycle_id: str | None,
    ) -> ProcessingCommit:
        if observation.source_key != self.source_key:
            raise ValueError("source runtime isolation violation")
        if observation.vibration_inference is None:
            observation, _ = self.vibration_detector.observe(observation)
        if lifecycle_id is not None:
            previous_lifecycle = self._lifecycle_by_machine.get(observation.uid)
            if previous_lifecycle is not None and previous_lifecycle != lifecycle_id:
                self.monitor.reset_machine(observation.to_legacy_reading(ingestion_id).machine_key)
                self.operating_clock.reset_machine(observation.uid)
            self._lifecycle_by_machine[observation.uid] = lifecycle_id
        decision = self.operating_clock.observe(observation)
        events = self._context_events(decision)
        if not decision.admitted_to_runtime:
            validation_status, reasons, payload = self._held_prediction_payload(
                observation,
                ingestion_id,
                decision,
            )
            return ProcessingCommit(
                ObservationDisposition.HELD_OPERATING_CONTEXT if payload is not None else ObservationDisposition.INVALID,
                validation_status,
                reasons,
                prediction=payload,
                lifecycle_events=events,
                operating_context=decision.payload(),
                vibration_inference=observation.vibration_inference,
            )
        reading = observation.to_legacy_reading(
            ingestion_id,
            effective_timestamp=decision.effective_operating_timestamp,
        )
        record = SourceRecord(ingestion_id, dict(observation.raw_payload), reading)
        started = time.perf_counter()
        validation, processed = self.monitor.process(record)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if processed is None:
            return ProcessingCommit(
                ObservationDisposition.INVALID,
                validation.status,
                validation.reasons,
                lifecycle_events=events,
                operating_context=decision.payload(),
                vibration_inference=observation.vibration_inference,
            )
        payload = self._prediction_payload(
            processed, latency_ms, decision, observation.vibration_inference
        )
        return ProcessingCommit(
            ObservationDisposition.PROCESSED,
            validation.status,
            validation.reasons,
            prediction=payload,
            lifecycle_events=events,
            operating_context=decision.payload(),
            vibration_inference=observation.vibration_inference,
        )

    def _prediction_payload(
        self,
        item: ProcessedReading,
        latency_ms: float,
        decision: OperatingContextDecision,
        vibration_inference: dict[str, object] | None,
    ) -> dict[str, object]:
        warning_available = bool(item.warning_rul_serviceable_intent and item.estimated_hours_to_warning is not None)
        critical_available = bool(item.critical_rul_serviceable_intent and item.estimated_hours_to_critical is not None)
        if warning_available or critical_available:
            forecast_state = ForecastState.AVAILABLE
        elif "history" in (item.rul_reason or "").lower():
            forecast_state = ForecastState.INITIALIZING
        else:
            forecast_state = ForecastState.WITHHELD
        manufacturer_state = item.predicted_status if (
            item.sensor_quality_extreme_raw_override
            or "RAW" in item.prediction_state_source.upper()
            or "MANUFACTURER" in item.prediction_state_source.upper()
        ) else None
        return {
            "deployment_id": self.deployment_id,
            "prediction_timestamp": utc_now().isoformat(),
            "health_state_model": item.predicted_status,
            "health_state_manufacturer": manufacturer_state,
            "health_state_source": item.prediction_state_source,
            "connectivity_state": "ONLINE",
            "forecast_state": forecast_state,
            "warning_point_hours": item.estimated_hours_to_warning if warning_available else None,
            "warning_lower_hours": item.warning_rul_lower_hours if warning_available else None,
            "warning_upper_hours": item.warning_rul_upper_hours if warning_available else None,
            "warning_serviceable": warning_available,
            "warning_withhold_reason": item.warning_rul_withholding_reason_code or item.rul_reason,
            "critical_point_hours": item.estimated_hours_to_critical if critical_available else None,
            "critical_lower_hours": item.critical_rul_lower_hours if critical_available else None,
            "critical_upper_hours": item.critical_rul_upper_hours if critical_available else None,
            "critical_serviceable": critical_available,
            "critical_withhold_reason": item.critical_rul_withholding_reason_code or item.rul_reason,
            "model_version": "v2.7",
            "model_artifact_sha256": self.model_sha256,
            "runtime_manifest_id": self.runtime_manifest_id,
            "inference_latency_ms": latency_ms,
            "model_state_reset_gap": bool(item.features.get("state_reset")),
            "quality_status": item.quality_status,
            "quality_reasons": list(item.quality_reasons),
            "sensor_quality_status": item.sensor_quality_status,
            "sensor_quality_reasons": list(item.sensor_quality_reasons),
            "features": item.features,
            "rul_reliability": item.rul_reliability,
            "rul_method": item.rul_method,
            "warning_forecastability_state": item.warning_rul_forecastability_state,
            "critical_forecastability_state": item.critical_rul_forecastability_state,
            "rul_time_basis": "OPERATING_HOURS",
            "vibration_operating_inference": vibration_inference,
            **decision.payload(),
        }

    def resolve_operating_context(self, observation: PlantObservation) -> PlantObservation:
        inferred, _ = self.vibration_detector.observe(observation)
        return inferred

    def rebuild(self, observations: Iterable[tuple[int, PlantObservation]]) -> None:
        candidate = FrozenSourceRuntime(
            self.source_key,
            self.model_path,
            self.sensor_config,
            deployment_id=self.deployment_id,
            runtime_manifest_id=self.runtime_manifest_id,
            vibration_operating_config=self.vibration_operating_config,
        )
        for ingestion_id, observation in observations:
            if observation.source_key != self.source_key:
                raise ValueError("cannot replay another source into this runtime")
            resolved = candidate.resolve_operating_context(observation)
            if not observation.runtime_replay_eligible:
                continue
            candidate.process(resolved, ingestion_id, None)
        self.monitor = candidate.monitor
        self.operating_clock = candidate.operating_clock
        self.vibration_detector = candidate.vibration_detector
        self.raw_validator = candidate.raw_validator
        self._lifecycle_by_machine = candidate._lifecycle_by_machine


class FrozenRuntimeRouter:
    def __init__(
        self,
        model_path: str | Path,
        sensor_config: SensorConfig,
        *,
        deployment_id: str,
        runtime_manifest_id: str,
        vibration_operating_config: VibrationOperatingConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.sensor_config = sensor_config
        self.deployment_id = deployment_id
        self.runtime_manifest_id = runtime_manifest_id
        self.vibration_operating_config = vibration_operating_config
        self._runtimes: dict[str, FrozenSourceRuntime] = {}

    def runtime(self, source_key: str) -> FrozenSourceRuntime:
        if source_key not in self._runtimes:
            self._runtimes[source_key] = FrozenSourceRuntime(
                source_key,
                self.model_path,
                self.sensor_config,
                deployment_id=self.deployment_id,
                runtime_manifest_id=self.runtime_manifest_id,
                vibration_operating_config=self.vibration_operating_config,
            )
        return self._runtimes[source_key]

    def process(self, observation: PlantObservation, ingestion_id: int, lifecycle_id: str | None) -> ProcessingCommit:
        return self.runtime(observation.source_key).process(observation, ingestion_id, lifecycle_id)

    def resolve_operating_context(self, observation: PlantObservation) -> PlantObservation:
        return self.runtime(observation.source_key).resolve_operating_context(observation)

    def rebuild_source(self, source_key: str, observations: Iterable[tuple[int, PlantObservation]]) -> None:
        self.runtime(source_key).rebuild(observations)

    @property
    def active_sources(self) -> tuple[str, ...]:
        return tuple(sorted(self._runtimes))
