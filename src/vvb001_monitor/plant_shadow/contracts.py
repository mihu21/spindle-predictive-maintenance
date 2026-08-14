from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from ..models import VVB001Reading


SCHEMA_VERSION = "plant_shadow_schema_v3_vibration_operating_inference"
LIFECYCLE_POLICY_VERSION = "plant_lifecycle_policy_v1"
TRUTH_POLICY_VERSION = "plant_target_truth_policy_v1"
SUPPORT_POLICY_VERSION = "plant_support_policy_v1"
FEATURE_CONTRACT_VERSION = "vvb001_causal_features_v2_7"
OPERATING_CONTEXT_POLICY_VERSION = "plant_operating_context_policy_v1"
MIN_CONFIRMED_RUNNING_CONFIDENCE = 0.90


class ActivationLevel(StrEnum):
    PROFILE_ONLY = "PROFILE_ONLY"
    SHADOW_MONITORING = "SHADOW_MONITORING"
    PLANT_EVIDENCE_ELIGIBLE = "PLANT_EVIDENCE_ELIGIBLE"


class ObservationDisposition(StrEnum):
    PROCESSED = "PROCESSED"
    INVALID = "INVALID"
    DUPLICATE = "DUPLICATE"
    LATE_QUARANTINED = "LATE_QUARANTINED"
    ERROR = "ERROR"
    HELD_OPERATING_CONTEXT = "HELD_OPERATING_CONTEXT"


class OperatingState(StrEnum):
    RUNNING = "RUNNING"
    IDLE = "IDLE"
    OFF = "OFF"
    MAINTENANCE = "MAINTENANCE"
    UNKNOWN = "UNKNOWN"


class OperatingStateSource(StrEnum):
    PLC = "PLC"
    CMMS = "CMMS"
    OPERATOR = "OPERATOR"
    DATABASE = "DATABASE"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
    VIBRATION_INFERENCE = "VIBRATION_INFERENCE"
    UNAVAILABLE = "UNAVAILABLE"


class ManagerState(StrEnum):
    UNINITIALIZED = "UNINITIALIZED"
    ACTIVE = "ACTIVE"
    RESET_CANDIDATE = "RESET_CANDIDATE"
    CLOSURE_PENDING = "CLOSURE_PENDING"
    QUARANTINED = "QUARANTINED"


class LifecycleStatus(StrEnum):
    ACTIVE = "ACTIVE"
    CLOSED_CONFIRMED = "CLOSED_CONFIRMED"
    CLOSED_INFERRED = "CLOSED_INFERRED"
    CENSORED = "CENSORED"
    QUARANTINED = "QUARANTINED"


class EndpointClass(StrEnum):
    WARNING_ONSET = "WARNING_ONSET"
    CRITICAL_ONSET = "CRITICAL_ONSET"
    PREVENTIVE_MAINTENANCE = "PREVENTIVE_MAINTENANCE"
    COMPONENT_REPLACEMENT = "COMPONENT_REPLACEMENT"
    PHYSICAL_FAILURE = "PHYSICAL_FAILURE"
    MACHINE_REPLACEMENT = "MACHINE_REPLACEMENT"
    SENSOR_REPLACEMENT = "SENSOR_REPLACEMENT"
    RESTART = "RESTART"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class EndpointPrecision(StrEnum):
    EXACT_TIMESTAMP = "EXACT_TIMESTAMP"
    BOUNDED_INTERVAL = "BOUNDED_INTERVAL"
    DATE_ONLY = "DATE_ONLY"
    UNKNOWN = "UNKNOWN"


class TargetName(StrEnum):
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class TruthEligibility(StrEnum):
    ELIGIBLE_EXACT = "ELIGIBLE_EXACT"
    ELIGIBLE_INTERVAL = "ELIGIBLE_INTERVAL"
    CENSORED = "CENSORED"
    INELIGIBLE = "INELIGIBLE"


class CensoringKind(StrEnum):
    NOT_CENSORED = "NOT_CENSORED"
    LEFT_CENSORED = "LEFT_CENSORED"
    RIGHT_CENSORED = "RIGHT_CENSORED"
    INTERVAL_CENSORED = "INTERVAL_CENSORED"
    UNKNOWN = "UNKNOWN"


class ForecastState(StrEnum):
    AVAILABLE = "AVAILABLE"
    WITHHELD = "WITHHELD"
    INITIALIZING = "INITIALIZING"
    ERROR = "ERROR"
    PAUSED = "PAUSED"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def machine_uid(source_key: str, line_sel: str, machine_id: str) -> str:
    values = (source_key.strip(), line_sel.strip(), machine_id.strip())
    if not all(values):
        raise ValueError("source_key, line_sel, and machine_id must be non-empty")
    return "::".join(values)


@dataclass(frozen=True)
class PlantObservation:
    source_key: str
    source_row_id: str
    event_timestamp: datetime
    line_sel: str
    machine_id: str
    vrms: float
    arms: float
    apeak: float
    crest: float
    temp: float
    observed_at: datetime = field(default_factory=utc_now)
    raw_payload: dict[str, Any] = field(default_factory=dict)
    operating_state: OperatingState = OperatingState.UNKNOWN
    operating_state_source: OperatingStateSource = OperatingStateSource.UNAVAILABLE
    operating_state_confidence: float = 0.0
    maintenance_event_id: str | None = None
    operating_context_block_reason: str | None = field(default=None, repr=False, compare=False)
    vibration_inference: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    runtime_replay_eligible: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.source_key.strip():
            raise ValueError("source_key must be non-empty")
        if not self.source_row_id.strip():
            raise ValueError("source_row_id must be a durable non-empty identity")
        if not self.line_sel.strip() or not self.machine_id.strip():
            raise ValueError("line_sel and machine_id must be non-empty")
        require_aware(self.event_timestamp, "event_timestamp")
        require_aware(self.observed_at, "observed_at")
        for name in ("vrms", "arms", "apeak", "crest", "temp"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "operating_state", OperatingState(str(self.operating_state)))
        object.__setattr__(
            self,
            "operating_state_source",
            OperatingStateSource(str(self.operating_state_source)),
        )
        confidence = float(self.operating_state_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("operating_state_confidence must be finite and within [0,1]")
        object.__setattr__(self, "operating_state_confidence", confidence)
        if self.maintenance_event_id is not None and not str(self.maintenance_event_id).strip():
            raise ValueError("maintenance_event_id must be non-empty when supplied")

    @property
    def uid(self) -> str:
        return machine_uid(self.source_key, self.line_sel, self.machine_id)

    @property
    def order_key(self) -> tuple[str, str]:
        return (require_aware(self.event_timestamp, "event_timestamp").isoformat(), self.source_row_id)

    def to_legacy_reading(
        self,
        ingestion_id: int,
        *,
        effective_timestamp: datetime | None = None,
    ) -> VVB001Reading:
        return VVB001Reading(
            source_id=int(ingestion_id),
            timestamp=require_aware(effective_timestamp or self.event_timestamp, "effective_timestamp"),
            line_sel=self.line_sel,
            machine_id=self.machine_id,
            vrms=float(self.vrms),
            arms=float(self.arms),
            apeak=float(self.apeak),
            crest=float(self.crest),
            temp=float(self.temp),
        )

    @property
    def confirmed_running(self) -> bool:
        return (
            self.operating_state == OperatingState.RUNNING
            and self.operating_state_source != OperatingStateSource.UNAVAILABLE
            and self.operating_state_confidence >= MIN_CONFIRMED_RUNNING_CONFIDENCE
        )


@dataclass(frozen=True)
class SourceDefinition:
    source_key: str
    display_name: str
    secret_reference: str
    schema_name: str
    table_name: str
    timestamp_column: str
    row_id_column: str | None
    tie_breaker_column: str | None = None
    activation_level: ActivationLevel = ActivationLevel.PROFILE_ONLY
    enabled: bool = False

    def validate_activation(self) -> None:
        if not self.source_key.strip() or not self.display_name.strip():
            raise ValueError("source_key and display_name must be non-empty")
        if self.activation_level != ActivationLevel.PROFILE_ONLY:
            identity = (self.row_id_column or self.tie_breaker_column or "").strip()
            if not identity:
                raise ValueError("SOURCE_IDENTITY_UNSAFE: continuous sources require a durable row identity")
            if identity.lower() == "ctid":
                raise ValueError("SOURCE_IDENTITY_UNSAFE: PostgreSQL ctid is not durable")


@dataclass(frozen=True)
class EndpointDecision:
    target: TargetName
    eligibility: TruthEligibility
    censoring: CensoringKind
    reason_code: str


def default_endpoint_decisions(endpoint_class: EndpointClass, precision: EndpointPrecision) -> tuple[EndpointDecision, ...]:
    exact = precision == EndpointPrecision.EXACT_TIMESTAMP
    interval = precision == EndpointPrecision.BOUNDED_INTERVAL

    def onset(target: TargetName) -> EndpointDecision:
        if exact:
            return EndpointDecision(target, TruthEligibility.ELIGIBLE_EXACT, CensoringKind.NOT_CENSORED, "INDEPENDENT_ONSET")
        if interval:
            return EndpointDecision(target, TruthEligibility.ELIGIBLE_INTERVAL, CensoringKind.INTERVAL_CENSORED, "INDEPENDENT_ONSET_INTERVAL")
        return EndpointDecision(target, TruthEligibility.INELIGIBLE, CensoringKind.UNKNOWN, "ENDPOINT_PRECISION_INSUFFICIENT")

    ineligible_warning = EndpointDecision(TargetName.WARNING, TruthEligibility.INELIGIBLE, CensoringKind.UNKNOWN, "NO_INDEPENDENT_WARNING_ONSET")
    ineligible_critical = EndpointDecision(TargetName.CRITICAL, TruthEligibility.INELIGIBLE, CensoringKind.UNKNOWN, "NO_INDEPENDENT_CRITICAL_ONSET")

    if endpoint_class == EndpointClass.WARNING_ONSET:
        return (onset(TargetName.WARNING), ineligible_critical)
    if endpoint_class == EndpointClass.CRITICAL_ONSET:
        return (ineligible_warning, onset(TargetName.CRITICAL))
    if endpoint_class in {EndpointClass.PREVENTIVE_MAINTENANCE, EndpointClass.COMPONENT_REPLACEMENT}:
        return (
            ineligible_warning,
            EndpointDecision(TargetName.CRITICAL, TruthEligibility.CENSORED, CensoringKind.RIGHT_CENSORED, "INTERVENTION_BEFORE_INDEPENDENT_CRITICAL"),
        )
    return (ineligible_warning, ineligible_critical)
