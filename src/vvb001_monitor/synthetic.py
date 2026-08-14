from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


FAULT_MODES = (
    "bearing",
    "unbalance",
    "lubrication",
    "looseness",
    "thermal",
    "sudden_impact",
    "mixed",
)


@dataclass(frozen=True)
class SyntheticConfig:
    lifecycles: int = 50
    cadence_seconds: int = 60
    seed: int = 42
    machines: int = 6
    line_sel: str = "LINE_1"
    start_time: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    identity_prefix: str = "mock"
    duration_min_hours: float = 36.0
    duration_max_hours: float = 96.0
    causal_hazard_coupling: bool = False
    duty_cycled_operating_context: bool = False
    vibration_inference_fixture: bool = False

    def validate(self) -> None:
        if self.lifecycles < 3:
            raise ValueError("lifecycles must be at least 3")
        if self.cadence_seconds < 10:
            raise ValueError("cadence_seconds must be at least 10")
        if self.machines < 2:
            raise ValueError("machines must be at least 2")
        if not self.line_sel.strip():
            raise ValueError("line_sel cannot be empty")
        if not self.identity_prefix.strip():
            raise ValueError("identity_prefix cannot be empty")
        if self.duration_min_hours < 12.0 or self.duration_max_hours <= self.duration_min_hours:
            raise ValueError("duration range must be ordered and at least 12 hours")
        if self.vibration_inference_fixture and not self.duty_cycled_operating_context:
            raise ValueError(
                "vibration inference fixtures require duty_cycled_operating_context"
            )


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _machine_profiles(rng: random.Random, count: int) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for idx in range(1, count + 1):
        # Persistent machine-to-machine healthy differences. Some machines are naturally
        # higher-vibration than others so the learner must use relative/baseline features.
        profiles[f"SPINDLE_{idx:02d}"] = {
            "vrms": rng.uniform(0.7, 3.8),
            "arms": rng.uniform(0.6, 3.5),
            "crest": rng.uniform(2.0, 3.8),
            "temp": rng.uniform(30.0, 48.0),
            "load_amp": rng.uniform(0.03, 0.16),
            "noise_scale": rng.uniform(0.8, 1.35),
        }
    return profiles


def _damage_curve(progress: float, onset: float, exponent: float) -> float:
    if progress <= onset:
        return 0.0
    normalized = (progress - onset) / max(1e-9, 1.0 - onset)
    return _clip(normalized ** exponent, 0.0, 1.0)


def _fault_effects(mode: str, damage: float, late_damage: float) -> tuple[float, float, float, float, float]:
    """Return additive/multiplicative fault effects for vrms, arms, crest, temp and impact."""
    if mode == "bearing":
        return 4.0 * damage, 13.0 * damage, 4.5 * math.sin(math.pi * damage), 7.0 * damage, 18.0 * damage
    if mode == "unbalance":
        return 11.0 * damage, 4.0 * damage, 0.8 * damage, 3.0 * damage, 5.0 * damage
    if mode == "lubrication":
        return 3.0 * damage, 10.0 * damage, 2.0 * damage, 16.0 * damage, 10.0 * damage
    if mode == "looseness":
        return 8.0 * damage, 8.0 * damage, 2.5 * damage, 5.0 * damage, 14.0 * damage
    if mode == "thermal":
        return 2.0 * damage, 3.0 * damage, 0.5 * damage, 24.0 * damage, 3.0 * damage
    if mode == "sudden_impact":
        jump = 1.0 if late_damage > 0 else 0.0
        return 2.0 * damage, 5.0 * damage + 5.0 * jump, 5.0 * jump, 4.0 * damage, 30.0 * jump
    # mixed
    return 7.0 * damage, 11.0 * damage, 2.5 * math.sin(math.pi * damage), 11.0 * damage, 15.0 * damage


def generate_mock_csv(path: str | Path, config: SyntheticConfig) -> int:
    """Generate diverse, *unlabelled-for-training* VVB001 degradation lifecycles.

    The file intentionally has no NORMAL/WARNING/CRITICAL training label. `latent_damage_score`
    and `fault_mode` are simulator-only audit fields and are ignored by the learner. The model
    discovers three regimes from the feature distribution and lifecycle structure.
    """
    config.validate()
    rng = random.Random(config.seed)
    profiles = _machine_profiles(rng, config.machines)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "timestamp", "line_sel", "machine_id", "vrms", "arms", "apeak",
        "crest", "temp", "lifecycle_id", "lifecycle_progress", "latent_damage_score",
        "fault_mode",
        "operating_state", "operating_state_source", "operating_state_confidence",
        "maintenance_event_id", "operating_elapsed_hours",
        "vibration_inference_fixture", "synthetic_true_operating_state",
    ]
    source_id = 1
    current_start = config.start_time
    rows = 0

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for lifecycle_index in range(config.lifecycles):
            profile_key = f"SPINDLE_{1 + lifecycle_index % config.machines:02d}"
            machine = profile_key if config.identity_prefix == "mock" else f"{config.identity_prefix}_{profile_key}"
            profile = profiles[profile_key]
            lifecycle_id = f"{config.identity_prefix}_lifecycle_{lifecycle_index + 1:04d}"
            mode = FAULT_MODES[lifecycle_index % len(FAULT_MODES)] if lifecycle_index < len(FAULT_MODES) else rng.choice(FAULT_MODES)
            # v2.2 development batches can couple an unobserved wear rate to both eventual
            # degradation timing and a weak *causal* precursor visible from the beginning. The
            # ordinary generator keeps its historical behavior unless explicitly enabled.
            wear_rate = rng.uniform(0.65, 1.45) if config.causal_hazard_coupling else 1.0
            nominal_duration_h = rng.uniform(config.duration_min_hours, config.duration_max_hours)
            duration_h = nominal_duration_h / wear_rate if config.causal_hazard_coupling else nominal_duration_h
            steps = max(2, int(duration_h * 3600 / config.cadence_seconds))
            onset = rng.uniform(0.18, 0.58)
            if config.causal_hazard_coupling:
                onset = _clip(onset - 0.12 * (wear_rate - 1.0), 0.10, 0.70)
            exponent = rng.uniform(1.15, 2.8)
            sudden_point = rng.uniform(0.64, 0.88)
            cycle_period_h = rng.uniform(0.5, 3.5)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            cycle_scale = profile["load_amp"]
            noise = profile["noise_scale"]
            wall_delay_seconds = 0
            interruption_index = 0
            interruption_every = max(12, int(4 * 3600 / config.cadence_seconds))

            for i in range(steps):
                progress = i / (steps - 1)
                ts = current_start + timedelta(
                    seconds=i * config.cadence_seconds + wall_delay_seconds
                )
                elapsed_h = i * config.cadence_seconds / 3600.0
                load_wave = math.sin(2.0 * math.pi * elapsed_h / cycle_period_h + phase)
                load = 1.0 + cycle_scale * load_wave + rng.gauss(0.0, 0.015)

                damage = _damage_curve(progress, onset, exponent)
                # Accumulated operating stress is inference-time observable because it depends
                # only on elapsed time and the current lifecycle's wear environment. It is not
                # computed from eventual duration, onset, or future readings.
                precursor = (
                    max(0.0, wear_rate - 0.55) * elapsed_h / 144.0
                    if config.causal_hazard_coupling
                    else 0.0
                )
                # Sudden-impact mode has a real discontinuity; other modes can have small shocks.
                late_damage = 1.0 if mode == "sudden_impact" and progress >= sudden_point else 0.0
                vrms_eff, arms_eff, crest_eff, temp_eff, impact_eff = _fault_effects(mode, damage, late_damage)

                transient = 0.0
                if rng.random() < 0.0015:
                    transient = rng.uniform(0.5, 2.5)  # healthy/operating transient, not necessarily degradation

                vrms = profile["vrms"] * load + 0.65 * precursor + vrms_eff + transient + rng.gauss(0, 0.12 * noise + 0.10 * damage)
                arms = profile["arms"] * (0.96 + 0.08 * load) + 0.9 * precursor + arms_eff + 0.5 * transient + rng.gauss(0, 0.22 * noise + 0.18 * damage)
                # Crest may rise in intermediate bearing/impact development and soften late as RMS rises.
                crest = profile["crest"] + crest_eff + rng.gauss(0, 0.10 * noise)
                if mode in {"bearing", "mixed"} and damage > 0.75:
                    crest -= (damage - 0.75) * rng.uniform(2.0, 4.0)
                crest = _clip(crest, 1.05, 30.0)
                apeak = arms * crest + impact_eff + rng.gauss(0, 0.30 * noise + 0.50 * damage)
                temp = profile["temp"] + 1.4 * (load - 1.0) + 2.5 * precursor + temp_eff + rng.gauss(0, 0.28 * noise)

                vrms = _clip(vrms, 0.0, 44.5)
                arms = _clip(arms, 0.02, 180.0)
                apeak = _clip(max(arms, apeak), arms, 470.0)
                # Store a crest value physically consistent with apeak/arms, as the VVB001 process
                # values are related. Small mismatch is intentionally retained via apeak noise.
                crest = _clip(apeak / max(arms, 1e-9), 1.0, 49.0)
                temp = _clip(temp, -29.0, 79.0)

                # Continuous hidden simulator truth for evaluation only. It is never a training label.
                latent = _clip(0.82 * damage + 0.18 * late_damage, 0.0, 1.0)
                writer.writerow({
                    "id": source_id,
                    "timestamp": ts.isoformat(),
                    "line_sel": config.line_sel,
                    "machine_id": machine,
                    "vrms": f"{vrms:.8f}",
                    "arms": f"{arms:.8f}",
                    "apeak": f"{apeak:.8f}",
                    "crest": f"{crest:.8f}",
                    "temp": f"{temp:.8f}",
                    "lifecycle_id": lifecycle_id,
                    "lifecycle_progress": f"{progress:.8f}",
                    "latent_damage_score": f"{latent:.8f}",
                    "fault_mode": mode,
                    "operating_state": "" if config.vibration_inference_fixture else "RUNNING",
                    "operating_state_source": "" if config.vibration_inference_fixture else "SYNTHETIC_FIXTURE",
                    "operating_state_confidence": "" if config.vibration_inference_fixture else "1.00000000",
                    "maintenance_event_id": "",
                    "operating_elapsed_hours": f"{elapsed_h:.8f}",
                    "vibration_inference_fixture": "1" if config.vibration_inference_fixture else "0",
                    "synthetic_true_operating_state": "RUNNING",
                })
                source_id += 1
                rows += 1

                if (
                    config.duty_cycled_operating_context
                    and i > 0
                    and i < steps - 1
                    and i % interruption_every == 0
                ):
                    state_name, duration_rows = (
                        ("IDLE", 2),
                        ("OFF", 4),
                        ("MAINTENANCE", 3),
                        ("UNKNOWN", 1),
                    )[(lifecycle_index + interruption_index) % 4]
                    maintenance_id = (
                        f"synthetic-maintenance-{lifecycle_index + 1:04d}-{interruption_index + 1:03d}"
                        if state_name == "MAINTENANCE" else ""
                    )
                    for pause_index in range(duration_rows):
                        pause_ts = ts + timedelta(
                            seconds=(pause_index + 1) * config.cadence_seconds
                        )
                        if state_name == "OFF":
                            pause_vrms = rng.uniform(0.01, 0.08)
                            pause_arms = rng.uniform(0.01, 0.07)
                            pause_temp = max(20.0, profile["temp"] - rng.uniform(5.0, 12.0))
                        elif state_name == "IDLE":
                            pause_vrms = max(0.05, profile["vrms"] * rng.uniform(0.08, 0.20))
                            pause_arms = max(0.05, profile["arms"] * rng.uniform(0.08, 0.18))
                            pause_temp = profile["temp"] - rng.uniform(1.0, 4.0)
                        elif state_name == "MAINTENANCE":
                            # Deliberately disruptive tool vibration. These values must be stored
                            # but never admitted to degradation features or RUL state.
                            pause_vrms = rng.uniform(7.0, 18.0)
                            pause_arms = rng.uniform(12.0, 35.0)
                            pause_temp = profile["temp"] + rng.uniform(-2.0, 5.0)
                        else:
                            pause_vrms = rng.uniform(0.1, 8.0)
                            pause_arms = rng.uniform(0.1, 15.0)
                            pause_temp = profile["temp"] + rng.uniform(-4.0, 4.0)
                        pause_crest = rng.uniform(2.0, 4.5)
                        pause_apeak = pause_arms * pause_crest
                        writer.writerow({
                            "id": source_id,
                            "timestamp": pause_ts.isoformat(),
                            "line_sel": config.line_sel,
                            "machine_id": machine,
                            "vrms": f"{_clip(pause_vrms, 0.0, 44.5):.8f}",
                            "arms": f"{_clip(pause_arms, 0.02, 180.0):.8f}",
                            "apeak": f"{_clip(pause_apeak, pause_arms, 470.0):.8f}",
                            "crest": f"{pause_crest:.8f}",
                            "temp": f"{_clip(pause_temp, -29.0, 79.0):.8f}",
                            "lifecycle_id": lifecycle_id,
                            "lifecycle_progress": f"{progress:.8f}",
                            "latent_damage_score": f"{latent:.8f}",
                            "fault_mode": mode,
                            "operating_state": "" if config.vibration_inference_fixture else state_name,
                            "operating_state_source": (
                                "" if config.vibration_inference_fixture else
                                ("UNAVAILABLE" if state_name == "UNKNOWN" else "SYNTHETIC_FIXTURE")
                            ),
                            "operating_state_confidence": (
                                "" if config.vibration_inference_fixture else
                                ("0.00000000" if state_name == "UNKNOWN" else "1.00000000")
                            ),
                            "maintenance_event_id": "" if config.vibration_inference_fixture else maintenance_id,
                            "operating_elapsed_hours": f"{elapsed_h:.8f}",
                            "vibration_inference_fixture": "1" if config.vibration_inference_fixture else "0",
                            "synthetic_true_operating_state": state_name,
                        })
                        source_id += 1
                        rows += 1
                    wall_delay_seconds += duration_rows * config.cadence_seconds
                    interruption_index += 1

            # A gap longer than the feature-history horizon represents maintenance/new lifecycle.
            current_start += timedelta(
                hours=duration_h + rng.uniform(25.0, 36.0),
                seconds=wall_delay_seconds,
            )
    return rows
