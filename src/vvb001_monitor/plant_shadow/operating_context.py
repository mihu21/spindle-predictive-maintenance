from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from .contracts import (
    MIN_CONFIRMED_RUNNING_CONFIDENCE,
    OPERATING_CONTEXT_POLICY_VERSION,
    OperatingState,
    PlantObservation,
    require_aware,
)


@dataclass(frozen=True)
class OperatingContextDecision:
    operating_state: OperatingState
    operating_state_source: str
    operating_state_confidence: float
    admitted_to_runtime: bool
    reason_code: str
    cumulative_operating_seconds: float
    effective_operating_timestamp: datetime | None
    maintenance_event_id: str | None
    state_transition: bool
    previous_operating_state: OperatingState | None
    policy_version: str = OPERATING_CONTEXT_POLICY_VERSION

    def payload(self) -> dict[str, object]:
        value = asdict(self)
        value["operating_state"] = self.operating_state.value
        value["previous_operating_state"] = (
            self.previous_operating_state.value if self.previous_operating_state else None
        )
        value["effective_operating_timestamp"] = (
            self.effective_operating_timestamp.isoformat()
            if self.effective_operating_timestamp is not None
            else None
        )
        return value


@dataclass
class _MachineOperatingClock:
    anchor_timestamp: datetime
    last_event_timestamp: datetime
    last_state: OperatingState
    last_confirmed_running: bool
    cumulative_operating_seconds: float = 0.0
    last_model_operating_seconds: float | None = None


class OperatingContextClock:
    """Causal per-machine clock that excludes non-running wall time from model exposure."""

    def __init__(self, minimum_running_confidence: float = MIN_CONFIRMED_RUNNING_CONFIDENCE) -> None:
        if not 0.0 <= minimum_running_confidence <= 1.0:
            raise ValueError("minimum_running_confidence must be within [0,1]")
        self.minimum_running_confidence = float(minimum_running_confidence)
        self._state: dict[str, _MachineOperatingClock] = {}

    def reset_machine(self, machine_uid: str) -> None:
        self._state.pop(machine_uid, None)

    def observe(self, observation: PlantObservation) -> OperatingContextDecision:
        timestamp = require_aware(observation.event_timestamp, "event_timestamp")
        confirmed_running = (
            observation.operating_state == OperatingState.RUNNING
            and observation.confirmed_running
            and observation.operating_state_confidence >= self.minimum_running_confidence
        )
        state = self._state.get(observation.uid)
        previous_state = state.last_state if state else None
        transition = previous_state is not None and previous_state != observation.operating_state

        if state is None:
            state = _MachineOperatingClock(
                anchor_timestamp=timestamp,
                last_event_timestamp=timestamp,
                last_state=observation.operating_state,
                last_confirmed_running=confirmed_running,
            )
            self._state[observation.uid] = state
        else:
            elapsed = (timestamp - state.last_event_timestamp).total_seconds()
            if elapsed < 0:
                raise ValueError("operating-context clock received a non-causal timestamp")
            if state.last_confirmed_running and confirmed_running:
                state.cumulative_operating_seconds += elapsed
            state.last_event_timestamp = timestamp
            state.last_state = observation.operating_state
            state.last_confirmed_running = confirmed_running

        effective = state.anchor_timestamp + timedelta(seconds=state.cumulative_operating_seconds)
        admitted = False
        if confirmed_running:
            admitted = (
                state.last_model_operating_seconds is None
                or state.cumulative_operating_seconds > state.last_model_operating_seconds
            )
            if admitted:
                state.last_model_operating_seconds = state.cumulative_operating_seconds

        if admitted:
            reason = "CONFIRMED_RUNNING"
        elif observation.operating_state == OperatingState.UNKNOWN:
            reason = "OPERATING_STATE_UNKNOWN"
        elif observation.operating_state == OperatingState.RUNNING and not confirmed_running:
            reason = "OPERATING_STATE_LOW_CONFIDENCE"
        elif confirmed_running and previous_state != OperatingState.RUNNING:
            reason = "OPERATING_RESUME_WARMUP"
        elif confirmed_running:
            reason = "NO_OPERATING_TIME_ADVANCE"
        else:
            reason = f"OPERATING_STATE_{observation.operating_state.value}"

        return OperatingContextDecision(
            operating_state=observation.operating_state,
            operating_state_source=observation.operating_state_source.value,
            operating_state_confidence=observation.operating_state_confidence,
            admitted_to_runtime=admitted,
            reason_code=reason,
            cumulative_operating_seconds=state.cumulative_operating_seconds,
            effective_operating_timestamp=effective if confirmed_running else None,
            maintenance_event_id=observation.maintenance_event_id,
            state_transition=transition,
            previous_operating_state=previous_state,
        )
