from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from .config import SensorConfig
from .models import VVB001Reading


@dataclass(frozen=True)
class SensorQualityDecision:
    status: str
    reasons: tuple[str, ...]
    feature_reading: VVB001Reading
    held_sensors: tuple[str, ...] = ()
    extreme_raw_override: bool = False
    hold_prediction_state: bool = False
    prediction_state_action: str = "LIVE"


@dataclass
class _MachineQualityState:
    last_raw: VVB001Reading | None = None
    last_trusted: VVB001Reading | None = None
    vibration_candidate: VVB001Reading | None = None
    vibration_candidate_at: datetime | None = None
    vibration_candidate_confirmations: int = 0
    temp_bias_level: float | None = None
    temp_bias_started_at: datetime | None = None
    temp_bias_rapid_confirmations: int = 0
    temp_bias_rapid_direction: int = 0
    stuck_counts: dict[str, int] | None = None


class SensorQualityGuard:
    """Conservative causal guard against plausible sensor faults.

    The guard never uses simulator truth or model outputs. Suspect values are prevented from
    contaminating rolling features by substituting the last trusted value for the affected
    sensor(s). Abrupt uncorroborated vibration changes require persistence confirmation;
    physically corroborated changes are released as machine behavior.

    Extremely large raw readings bypass suppression. These are not manufacturer machine-health
    thresholds; they are a fail-safe near the VVB001 measurement envelope so the quality layer
    cannot hide extreme telemetry.
    """

    VERSION = "sensor_quality_guard_v3_checkpointed_bias"
    TEMP_STEP_C = 7.0
    TEMP_PLATEAU_C = 2.5
    TEMP_RETURN_C = 4.0
    TEMP_BIAS_MAX_HOLD_HOURS = 12.0  # audit threshold; stable isolated bias remains quarantined beyond it
    TEMP_BIAS_DRIFT_MAX_C_PER_H = 5.0
    TEMP_BIAS_RAPID_RELEASE_CONFIRM_ROWS = 3
    VIB_STEP_VRMS = 6.0
    VIB_STEP_ARMS = 12.0
    VIB_STEP_APEAK = 45.0
    TEMP_CORROBORATION_C = 2.5
    PERSIST_VRMS_TOL = 3.5
    PERSIST_ARMS_TOL = 7.0
    PERSIST_APEAK_TOL = 28.0
    STUCK_EPS = {"vrms": 1e-5, "arms": 1e-5, "crest": 1e-5, "temp": 1e-5}
    STUCK_CONFIRM_ROWS = 4

    # Near-envelope values bypass suppression. Measurement ranges themselves still belong to
    # VVB001Validator and remain authoritative for validity.
    EXTREME_VRMS = 38.0
    EXTREME_ARMS = 165.0
    EXTREME_APEAK = 430.0
    EXTREME_TEMP = 74.0

    def __init__(self, config: SensorConfig) -> None:
        self.config = config
        self._state: dict[str, _MachineQualityState] = {}

    def reset_machine(self, machine_key: str) -> None:
        self._state.pop(machine_key, None)

    @staticmethod
    def _with_values(reading: VVB001Reading, **values: float) -> VVB001Reading:
        return replace(reading, **values)

    @staticmethod
    def _copy_sensor_values(source: VVB001Reading, target: VVB001Reading, sensors: tuple[str, ...]) -> VVB001Reading:
        values = {name: getattr(source, name) for name in sensors}
        return replace(target, **values)

    @classmethod
    def is_extreme_raw(cls, r: VVB001Reading) -> bool:
        return (
            r.vrms >= cls.EXTREME_VRMS
            or r.arms >= cls.EXTREME_ARMS
            or r.apeak >= cls.EXTREME_APEAK
            or r.temp >= cls.EXTREME_TEMP
        )

    def _extreme_override(self, r: VVB001Reading) -> bool:
        return self.is_extreme_raw(r)

    def _dropout_sensors(self, current: VVB001Reading, previous: VVB001Reading) -> tuple[str, ...]:
        held: list[str] = []
        if current.vrms <= 0.08 and previous.vrms >= 0.5:
            held.append("vrms")
        if current.arms <= 0.08 and previous.arms >= 0.5:
            held.extend(["arms", "apeak", "crest"])
        if current.apeak <= 0.10 and previous.apeak >= 1.0:
            held.append("apeak")
        if current.temp <= -15.0 and previous.temp >= 5.0:
            held.append("temp")
        return tuple(dict.fromkeys(held))

    def process(self, reading: VVB001Reading) -> SensorQualityDecision:
        key = reading.machine_key
        state = self._state.setdefault(key, _MachineQualityState(stuck_counts={}))
        previous_raw = state.last_raw
        trusted = state.last_trusted
        if previous_raw is None or trusted is None:
            state.last_raw = reading
            state.last_trusted = reading
            return SensorQualityDecision("GOOD", (), reading, prediction_state_action="LIVE")

        dt_h = (reading.timestamp - previous_raw.timestamp).total_seconds() / 3600.0
        if dt_h <= 0 or dt_h > self.config.history_hours:
            self.reset_machine(key)
            self._state[key] = _MachineQualityState(last_raw=reading, last_trusted=reading, stuck_counts={})
            return SensorQualityDecision(
                "GOOD_RESET",
                ("quality state reset after non-causal/long timestamp gap",),
                reading,
                prediction_state_action="RESET",
            )

        state.last_raw = reading

        if self._extreme_override(reading):
            state.last_trusted = reading
            state.vibration_candidate = None
            state.vibration_candidate_at = None
            state.vibration_candidate_confirmations = 0
            state.temp_bias_level = None
            state.temp_bias_started_at = None
            state.temp_bias_rapid_confirmations = 0
            state.temp_bias_rapid_direction = 0
            return SensorQualityDecision(
                "EXTREME_RAW_OVERRIDE",
                ("extreme raw telemetry bypassed sensor-fault suppression",),
                reading,
                extreme_raw_override=True,
                prediction_state_action="RAW_SAFETY_OVERRIDE",
            )

        dropout = self._dropout_sensors(reading, trusted)
        if dropout:
            sanitized = self._copy_sensor_values(trusted, reading, dropout)
            # Update trusted values only for unaffected channels.
            unaffected = tuple(name for name in ("vrms", "arms", "apeak", "crest", "temp") if name not in dropout)
            state.last_trusted = self._copy_sensor_values(reading, trusted, unaffected)
            return SensorQualityDecision(
                "SUSPECT_DROPOUT",
                tuple(f"{name} abrupt near-floor dropout held from feature state" for name in dropout),
                sanitized,
                dropout,
                hold_prediction_state=True,
                prediction_state_action="BEGIN_OR_HOLD_QUARANTINE",
            )

        # Continue a previously identified temperature step/bias. Keep the pre-shift temperature
        # in feature state while allowing other sensors to evolve normally. A slow isolated drift
        # of the biased level remains quarantined; only recovery, physical corroboration, a rapid
        # post-step thermal movement, or an extreme raw override releases it.
        if state.temp_bias_level is not None and state.temp_bias_started_at is not None:
            elapsed = (reading.timestamp - state.temp_bias_started_at).total_seconds() / 3600.0
            returned = abs(reading.temp - trusted.temp) <= self.TEMP_RETURN_C
            plateau = abs(reading.temp - state.temp_bias_level) <= self.TEMP_PLATEAU_C
            corroborated = (
                abs(reading.vrms - trusted.vrms) >= self.VIB_STEP_VRMS
                or abs(reading.arms - trusted.arms) >= self.VIB_STEP_ARMS
            )
            signed_temp_rate_per_h = (reading.temp - previous_raw.temp) / max(dt_h, 1e-9)
            temp_rate_per_h = abs(signed_temp_rate_per_h)
            slow_isolated_drift = (
                not corroborated
                and abs(reading.temp - trusted.temp) > self.TEMP_RETURN_C
                and temp_rate_per_h <= self.TEMP_BIAS_DRIFT_MAX_C_PER_H
            )
            if returned:
                state.temp_bias_level = None
                state.temp_bias_started_at = None
                state.temp_bias_rapid_confirmations = 0
                state.temp_bias_rapid_direction = 0
                state.last_trusted = reading
                return SensorQualityDecision(
                    "GOOD_RECOVERED_BIAS",
                    ("temperature returned to the pre-bias trusted range",),
                    reading,
                    prediction_state_action="ROLLBACK_AND_RELEASE",
                )
            if plateau or slow_isolated_drift:
                state.temp_bias_rapid_confirmations = 0
                state.temp_bias_rapid_direction = 0
                if not plateau:
                    # Follow the biased level only for quality-state comparison. The trusted
                    # temperature used by the feature engine remains the pre-bias value.
                    state.temp_bias_level = reading.temp
                sanitized = replace(reading, temp=trusted.temp)
                state.last_trusted = replace(
                    trusted,
                    source_id=reading.source_id,
                    timestamp=reading.timestamp,
                    line_sel=reading.line_sel,
                    machine_id=reading.machine_id,
                    vrms=reading.vrms,
                    arms=reading.arms,
                    apeak=reading.apeak,
                    crest=reading.crest,
                )
                long_hold = elapsed > self.TEMP_BIAS_MAX_HOLD_HOURS
                if slow_isolated_drift and not plateau:
                    quality_status = "SUSPECT_BIAS_DRIFT_LONG" if long_hold else "SUSPECT_BIAS_DRIFT"
                    reason = f"isolated temperature bias drift remained quarantined for {elapsed:.2f}h"
                else:
                    quality_status = "SUSPECT_BIAS_LONG" if long_hold else "SUSPECT_BIAS"
                    reason = f"temperature step remained on a stable isolated plateau for {elapsed:.2f}h"
                return SensorQualityDecision(
                    quality_status,
                    (reason,),
                    sanitized,
                    ("temp",),
                    hold_prediction_state=True,
                    prediction_state_action="BEGIN_OR_HOLD_QUARANTINE",
                )

            # A single noisy fast movement must not terminate a known isolated bias episode.
            # Require several consecutive fast moves in the same direction before treating the
            # temperature change as a genuine thermal machine shift. This protects against slow
            # sensor drift whose per-sample jitter can briefly exceed the drift-rate threshold.
            if not corroborated and abs(reading.temp - trusted.temp) > self.TEMP_RETURN_C:
                direction = 1 if signed_temp_rate_per_h > 0 else (-1 if signed_temp_rate_per_h < 0 else 0)
                if direction and direction == state.temp_bias_rapid_direction:
                    state.temp_bias_rapid_confirmations += 1
                else:
                    state.temp_bias_rapid_direction = direction
                    state.temp_bias_rapid_confirmations = 1 if direction else 0

                if state.temp_bias_rapid_confirmations < self.TEMP_BIAS_RAPID_RELEASE_CONFIRM_ROWS:
                    state.temp_bias_level = reading.temp
                    sanitized = replace(reading, temp=trusted.temp)
                    state.last_trusted = replace(
                        trusted,
                        source_id=reading.source_id,
                        timestamp=reading.timestamp,
                        line_sel=reading.line_sel,
                        machine_id=reading.machine_id,
                        vrms=reading.vrms,
                        arms=reading.arms,
                        apeak=reading.apeak,
                        crest=reading.crest,
                    )
                    return SensorQualityDecision(
                        "SUSPECT_BIAS_RAPID_CHECK",
                        (
                            "isolated temperature movement exceeded the drift-rate limit but remains quarantined "
                            f"pending {self.TEMP_BIAS_RAPID_RELEASE_CONFIRM_ROWS} consecutive directional confirmations",
                        ),
                        sanitized,
                        ("temp",),
                        hold_prediction_state=True,
                        prediction_state_action="BEGIN_OR_HOLD_QUARANTINE",
                    )

                state.temp_bias_level = None
                state.temp_bias_started_at = None
                state.temp_bias_rapid_confirmations = 0
                state.temp_bias_rapid_direction = 0
                state.last_trusted = reading
                return SensorQualityDecision(
                    "CONFIRMED_THERMAL_MACHINE_SHIFT",
                    ("sustained rapid isolated temperature movement confirmed as machine behavior",),
                    reading,
                    prediction_state_action="COMMIT_AND_RELEASE",
                )

            # Cross-sensor corroboration is strong evidence that the temperature change belongs to
            # the machine rather than an isolated sensor bias.
            state.temp_bias_level = None
            state.temp_bias_started_at = None
            state.temp_bias_rapid_confirmations = 0
            state.temp_bias_rapid_direction = 0
            state.last_trusted = reading
            return SensorQualityDecision(
                "CONFIRMED_CORROBORATED_MACHINE_SHIFT",
                ("temperature change gained vibration/acceleration corroboration",),
                reading,
                prediction_state_action="COMMIT_AND_RELEASE",
            )

        temp_jump = abs(reading.temp - previous_raw.temp) >= self.TEMP_STEP_C
        temp_isolated = (
            abs(reading.vrms - previous_raw.vrms) < self.VIB_STEP_VRMS
            and abs(reading.arms - previous_raw.arms) < self.VIB_STEP_ARMS
        )
        if temp_jump and temp_isolated:
            state.temp_bias_level = reading.temp
            state.temp_bias_started_at = reading.timestamp
            state.temp_bias_rapid_confirmations = 0
            state.temp_bias_rapid_direction = 0
            sanitized = replace(reading, temp=trusted.temp)
            state.last_trusted = replace(
                trusted,
                source_id=reading.source_id,
                timestamp=reading.timestamp,
                line_sel=reading.line_sel,
                machine_id=reading.machine_id,
                vrms=reading.vrms,
                arms=reading.arms,
                apeak=reading.apeak,
                crest=reading.crest,
            )
            return SensorQualityDecision(
                "SUSPECT_BIAS",
                ("abrupt isolated temperature step held pending physical corroboration",),
                sanitized,
                ("temp",),
                hold_prediction_state=True,
                prediction_state_action="BEGIN_OR_HOLD_QUARANTINE",
            )

        # One-sample confirmation for abrupt vibration/acceleration changes. This catches narrow
        # spikes without permanently smoothing them into the model. If the new level persists on
        # the next sample, release it as a genuine machine shift (sudden faults stay detectable).
        candidate = state.vibration_candidate
        if candidate is not None:
            persistent = (
                abs(reading.vrms - candidate.vrms) <= self.PERSIST_VRMS_TOL
                and abs(reading.arms - candidate.arms) <= self.PERSIST_ARMS_TOL
                and abs(reading.apeak - candidate.apeak) <= self.PERSIST_APEAK_TOL
                and (
                    abs(reading.vrms - trusted.vrms) >= self.VIB_STEP_VRMS * 0.65
                    or abs(reading.arms - trusted.arms) >= self.VIB_STEP_ARMS * 0.65
                )
            )
            if persistent:
                state.vibration_candidate_confirmations += 1
                if state.vibration_candidate_confirmations < 3:
                    held = ("vrms", "arms", "apeak", "crest")
                    sanitized = self._copy_sensor_values(trusted, reading, held)
                    state.last_trusted = replace(
                        trusted,
                        source_id=reading.source_id,
                        timestamp=reading.timestamp,
                        line_sel=reading.line_sel,
                        machine_id=reading.machine_id,
                        temp=reading.temp,
                    )
                    return SensorQualityDecision(
                        "SUSPECT_SPIKE_PERSISTENCE_CHECK",
                        ("abrupt vibration level persisted once; held for a second confirmation sample",),
                        sanitized,
                        held,
                        hold_prediction_state=True,
                        prediction_state_action="BEGIN_OR_HOLD_QUARANTINE",
                    )
                state.vibration_candidate = None
                state.vibration_candidate_at = None
                state.vibration_candidate_confirmations = 0
                state.last_trusted = reading
                return SensorQualityDecision(
                    "CONFIRMED_ABRUPT_MACHINE_SHIFT",
                    ("abrupt uncorroborated vibration change persisted across three confirmation samples",),
                    reading,
                    prediction_state_action="COMMIT_AND_RELEASE",
                )
            recovered_to_trusted = (
                abs(reading.vrms - trusted.vrms) < self.VIB_STEP_VRMS * 0.65
                and abs(reading.arms - trusted.arms) < self.VIB_STEP_ARMS * 0.65
                and abs(reading.apeak - trusted.apeak) < self.VIB_STEP_APEAK * 0.65
            )
            if recovered_to_trusted:
                state.vibration_candidate = None
                state.vibration_candidate_at = None
                state.vibration_candidate_confirmations = 0
                state.last_trusted = reading
                return SensorQualityDecision(
                    "GOOD_RECOVERED_SPIKE",
                    ("abrupt candidate returned near the pre-spike trusted baseline",),
                    reading,
                    prediction_state_action="ROLLBACK_AND_RELEASE",
                )
            # The abrupt level changed shape but is still far from trusted baseline. Treat it as
            # the same untrusted burst instead of redefining the abnormal value as healthy.
            state.vibration_candidate = reading
            state.vibration_candidate_at = reading.timestamp
            state.vibration_candidate_confirmations = 0
            held = ("vrms", "arms", "apeak", "crest")
            sanitized = self._copy_sensor_values(trusted, reading, held)
            state.last_trusted = replace(
                trusted,
                source_id=reading.source_id,
                timestamp=reading.timestamp,
                line_sel=reading.line_sel,
                machine_id=reading.machine_id,
                temp=reading.temp,
            )
            return SensorQualityDecision(
                "SUSPECT_SPIKE_VARIANT",
                ("abrupt candidate changed amplitude but remained far from trusted baseline",),
                sanitized,
                held,
                hold_prediction_state=True,
                prediction_state_action="BEGIN_OR_HOLD_QUARANTINE",
            )

        abrupt_vibration = (
            abs(reading.vrms - previous_raw.vrms) >= self.VIB_STEP_VRMS
            or abs(reading.arms - previous_raw.arms) >= self.VIB_STEP_ARMS
            or abs(reading.apeak - previous_raw.apeak) >= self.VIB_STEP_APEAK
        )
        weak_temp_corroboration = abs(reading.temp - previous_raw.temp) < self.TEMP_CORROBORATION_C
        if abrupt_vibration and weak_temp_corroboration:
            state.vibration_candidate = reading
            state.vibration_candidate_at = reading.timestamp
            state.vibration_candidate_confirmations = 0
            held = ("vrms", "arms", "apeak", "crest")
            sanitized = self._copy_sensor_values(trusted, reading, held)
            # Temperature remains trusted/updated because it was not implicated in the spike.
            state.last_trusted = replace(
                trusted,
                source_id=reading.source_id,
                timestamp=reading.timestamp,
                line_sel=reading.line_sel,
                machine_id=reading.machine_id,
                temp=reading.temp,
            )
            return SensorQualityDecision(
                "SUSPECT_SPIKE",
                ("abrupt vibration/acceleration jump held for one-sample persistence confirmation",),
                sanitized,
                held,
                hold_prediction_state=True,
                prediction_state_action="BEGIN_OR_HOLD_QUARANTINE",
            )

        # Audit likely stuck channels. This does not suppress otherwise plausible telemetry because
        # the stress/plant context may genuinely be steady; it merely surfaces quality suspicion.
        stuck_counts = state.stuck_counts if state.stuck_counts is not None else {}
        stuck: list[str] = []
        for name in ("vrms", "arms", "crest", "temp"):
            count = stuck_counts.get(name, 0)
            if abs(getattr(reading, name) - getattr(previous_raw, name)) <= self.STUCK_EPS[name]:
                count += 1
            else:
                count = 0
            stuck_counts[name] = count
            if count >= self.STUCK_CONFIRM_ROWS:
                stuck.append(name)
        state.stuck_counts = stuck_counts
        state.last_trusted = reading
        if stuck:
            return SensorQualityDecision(
                "SUSPECT_STUCK",
                tuple(f"{name} unchanged across repeated samples" for name in stuck),
                reading,
                tuple(stuck),
                prediction_state_action="LIVE",
            )
        return SensorQualityDecision("GOOD", (), reading, prediction_state_action="LIVE")
