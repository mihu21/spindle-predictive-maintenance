from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import SensorConfig


@dataclass(frozen=True)
class Prediction:
    status: str
    probabilities: dict[str, float]
    degradation_score: float
    policy_reason: str | None = None
    state_held: bool = False
    state_source: str = "LIVE_MODEL"


def sensor_contract(config: SensorConfig) -> dict[str, Any]:
    return {
        "acceleration_unit": config.acceleration_unit,
        "history_hours": config.history_hours,
        "feature_windows_minutes": list(config.feature_windows_minutes),
        "ewma_alpha": config.ewma_alpha,
        "baseline_hours": config.baseline_hours,
        "baseline_min_points": config.baseline_min_points,
        "baseline_std_floor_fraction": config.baseline_std_floor_fraction,
    }


class VVB001Predictor:
    """Stateful regime predictor with conservative runtime policy guards.

    Sensor-quality quarantine can hold the predictor state so provisional sensor faults cannot
    advance smoothing/hysteresis. Separately, trusted sustained thermal evidence may bypass only
    the *upward hysteresis margin* once the learned degradation score has already reached the
    model's calibrated CRITICAL boundary. This never lowers the learned boundary globally.
    """

    # Thermal runaway evidence is deliberately conservative and feature-derived only. The raw
    # temperature must be substantially above the frozen per-machine baseline, the long window
    # must still be rising, and the model score itself must already be at its learned CRITICAL
    # boundary. This is a hysteresis bypass, not a new independent threshold classifier.
    THERMAL_MIN_DELTA_BASELINE_C = 20.0
    THERMAL_MIN_360M_TREND_C_PER_H = 0.75
    THERMAL_MIN_Z_BASELINE = 10.0
    THERMAL_MIN_RAW_C = 55.0

    def __init__(self, path: str | Path, sensor_config: SensorConfig) -> None:
        bundle = joblib.load(path)
        if bundle.get("sensor_contract") != sensor_contract(sensor_config):
            raise RuntimeError(
                "Model feature contract does not match the runtime sensor configuration. "
                "Use the same acceleration unit, history window, feature windows, EWMA and baseline settings used for training."
            )
        self.model = bundle["model"]
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.classes: list[str] = list(bundle["classes"])
        self.metadata = dict(bundle.get("metadata") or {})
        self.rul_model_artifact = bundle.get("rul_model")
        self.smoothing_tau_hours = float(self.metadata.get("prediction_smoothing_tau_hours", 0.25))
        self.hysteresis_margin = float(self.metadata.get("prediction_hysteresis_margin", 0.04))
        self._smoothed_score: dict[str, float] = {}
        self._status: dict[str, str] = {}
        self._pre_suspect_checkpoint: dict[str, tuple[float, str]] = {}

    def reset_machine(self, machine_key: str) -> None:
        self._smoothed_score.pop(machine_key, None)
        self._status.pop(machine_key, None)
        self._pre_suspect_checkpoint.pop(machine_key, None)

    def _capture_checkpoint(self, machine_key: str) -> tuple[float, str] | None:
        existing = self._pre_suspect_checkpoint.get(machine_key)
        if existing is not None:
            return existing
        score = self._smoothed_score.get(machine_key)
        status = self._status.get(machine_key)
        if score is None or status is None:
            return None
        checkpoint = (float(score), str(status))
        self._pre_suspect_checkpoint[machine_key] = checkpoint
        return checkpoint

    def _rollback_checkpoint(self, machine_key: str) -> tuple[float, str] | None:
        checkpoint = self._pre_suspect_checkpoint.pop(machine_key, None)
        if checkpoint is not None:
            self._smoothed_score[machine_key] = checkpoint[0]
            self._status[machine_key] = checkpoint[1]
        return checkpoint

    @staticmethod
    def _finite_number(value: object) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def _thermal_hysteresis_bypass(self, features: dict[str, float | int | str | None], score: float) -> bool:
        boundaries = getattr(self.model, "score_boundaries", None)
        if not boundaries or len(boundaries) != 2:
            return False
        critical_boundary = float(boundaries[1])
        if score < critical_boundary:
            return False
        if int(features.get("baseline_ready") or 0) != 1:
            return False
        temp_raw = self._finite_number(features.get("temp_raw"))
        temp_delta = self._finite_number(features.get("temp_delta_baseline"))
        temp_trend_360 = self._finite_number(features.get("temp_360m_trend_per_hour"))
        temp_z = self._finite_number(features.get("temp_z_baseline"))
        return (
            temp_raw is not None
            and temp_delta is not None
            and temp_trend_360 is not None
            and temp_z is not None
            and temp_raw >= self.THERMAL_MIN_RAW_C
            and temp_delta >= self.THERMAL_MIN_DELTA_BASELINE_C
            and temp_trend_360 >= self.THERMAL_MIN_360M_TREND_C_PER_H
            and temp_z >= self.THERMAL_MIN_Z_BASELINE
        )

    def predict(
        self,
        features: dict[str, float | int | str | None],
        *,
        hold_state: bool = False,
        state_action: str = "LIVE",
        allow_thermal_hysteresis_bypass: bool = True,
    ) -> Prediction:
        row: list[float] = []
        for name in self.feature_names:
            value = features.get(name)
            row.append(float(value) if isinstance(value, (int, float)) and value is not None else np.nan)
        X = np.asarray([row], dtype=float)
        probs = self.model.predict_proba(X)[0]
        raw_score = float(self.model.degradation_score(X)[0])
        probability_map = {str(c): float(p) for c, p in zip(self.model.classes_, probs)}

        machine_key = str(features.get("machine_key") or "")
        if int(features.get("state_reset") or 0):
            self.reset_machine(machine_key)

        if state_action == "RESET":
            self.reset_machine(machine_key)
        elif state_action == "RAW_SAFETY_OVERRIDE":
            self._pre_suspect_checkpoint.pop(machine_key, None)
        elif state_action == "COMMIT_AND_RELEASE":
            self._pre_suspect_checkpoint.pop(machine_key, None)
        elif state_action == "ROLLBACK_AND_RELEASE":
            self._rollback_checkpoint(machine_key)
        elif state_action == "LIVE" and machine_key in self._pre_suspect_checkpoint:
            # A one-row quarantine such as a dropout can end on the next ordinary GOOD row.
            # Resume from the last explicitly trusted checkpoint rather than from any provisional
            # state that may have been exposed during the episode.
            self._rollback_checkpoint(machine_key)

        previous_score = self._smoothed_score.get(machine_key)
        previous_status = self._status.get(machine_key)

        # Quarantined sensor rows are provisional. Capture the state *before* the first suspect
        # sample and always serve that checkpoint for the whole episode. This is stronger than
        # merely freezing the current state: if a suspect episode spans multiple quality labels or
        # temporary detector ambiguity, the operational prediction still comes from the explicitly
        # trusted pre-suspect state.
        if hold_state:
            checkpoint = self._capture_checkpoint(machine_key)
            if checkpoint is not None:
                checkpoint_score, checkpoint_status = checkpoint
                return Prediction(
                    status=checkpoint_status,
                    probabilities=probability_map,
                    degradation_score=float(checkpoint_score),
                    policy_reason="sensor_quality_quarantine_pre_suspect_checkpoint",
                    state_held=True,
                    state_source="PRE_SUSPECT_CHECKPOINT",
                )
        if hold_state and previous_score is not None and previous_status is not None:
            return Prediction(
                status=previous_status,
                probabilities=probability_map,
                degradation_score=float(previous_score),
                policy_reason="sensor_quality_quarantine_held_prediction_state",
                state_held=True,
                state_source="CURRENT_TRUSTED_STATE_NO_CHECKPOINT",
            )

        elapsed_h = features.get("elapsed_since_previous_hours")
        if previous_score is None or not isinstance(elapsed_h, (int, float)) or elapsed_h is None:
            smoothed = raw_score
        else:
            dt_h = max(0.0, float(elapsed_h))
            alpha = 1.0 - math.exp(-dt_h / max(self.smoothing_tau_hours, 1e-6)) if dt_h > 0 else 0.0
            smoothed = alpha * raw_score + (1.0 - alpha) * previous_score

        status = str(
            self.model.status_from_score(
                smoothed,
                previous_status=previous_status,
                hysteresis=self.hysteresis_margin,
            )
        )
        policy_reason = None
        if (
            allow_thermal_hysteresis_bypass
            and previous_status == "WARNING"
            and status == "WARNING"
            and self._thermal_hysteresis_bypass(features, smoothed)
        ):
            status = "CRITICAL"
            policy_reason = "trusted_sustained_thermal_evidence_bypassed_upward_hysteresis_only"

        self._smoothed_score[machine_key] = smoothed
        self._status[machine_key] = status
        return Prediction(
            status=status,
            probabilities=probability_map,
            degradation_score=float(smoothed),
            policy_reason=policy_reason,
            state_held=False,
            state_source=(
                "RAW_SAFETY_OVERRIDE"
                if state_action == "RAW_SAFETY_OVERRIDE"
                else "LIVE_AFTER_CHECKPOINT_ROLLBACK"
                if state_action == "ROLLBACK_AND_RELEASE"
                else "LIVE_MODEL"
            ),
        )
