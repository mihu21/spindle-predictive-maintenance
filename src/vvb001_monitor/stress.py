from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import SensorConfig
from .features import FeatureEngine
from .models import SourceRecord, VVB001Reading
from .monitor import VVB001Monitor
from .predictor import VVB001Predictor, sensor_contract
from .sensor_quality import SensorQualityGuard
from .validation import VVB001Validator


STRESS_SUITE_VERSION = "stress_v1"  # backwards-compatible alias
STRESS_GENERATOR_VERSION = "independent_adversarial_v1"
STRESS_V1_SEED = 20260811
STRESS_V2_SEED = 20260817
STRESS_V2_GENERATOR_VERSION = "independent_adversarial_v2"
STRESS_V3_SEED = 20260829
STRESS_V3_GENERATOR_VERSION = "independent_adversarial_v3"
STRESS_V4_SEED = 20260907
STRESS_V4_GENERATOR_VERSION = "independent_adversarial_v4"


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    kind: str
    fault_mode: str
    duration_hours: float
    description: str


SCENARIOS: tuple[StressScenario, ...] = (
    StressScenario(
        "high_safe_load",
        "healthy",
        "none",
        72.0,
        "Sustained high-but-safe operating load after a clean baseline; machine truth remains NORMAL.",
    ),
    StressScenario(
        "load_step_recovery",
        "healthy",
        "none",
        60.0,
        "Large temporary operating-load step followed by full recovery; machine truth remains NORMAL.",
    ),
    StressScenario(
        "repeated_sensor_spikes",
        "sensor_fault_healthy",
        "sensor_fault",
        48.0,
        "Repeated short vibration/acceleration spikes from a healthy machine.",
    ),
    StressScenario(
        "stuck_sensor",
        "sensor_fault_healthy",
        "sensor_fault",
        54.0,
        "One vibration channel becomes temporarily stuck while the healthy machine continues operating.",
    ),
    StressScenario(
        "temperature_bias_shift",
        "sensor_fault_healthy",
        "sensor_fault",
        54.0,
        "Temporary temperature sensor bias on an otherwise healthy machine.",
    ),
    StressScenario(
        "noise_burst",
        "sensor_fault_healthy",
        "sensor_fault",
        48.0,
        "Temporary high-noise measurement burst without real machine degradation.",
    ),
    StressScenario(
        "very_slow_degradation",
        "fault",
        "slow_mixed",
        120.0,
        "Long slow degradation with a late nonlinear tail unlike the bootstrap generator curve.",
    ),
    StressScenario(
        "very_fast_degradation",
        "fault",
        "fast_mixed",
        42.0,
        "Short lifecycle with a fast nonlinear ramp from healthy to severe degradation.",
    ),
    StressScenario(
        "sudden_mechanical_fault",
        "abrupt_fault",
        "sudden_mechanical",
        48.0,
        "Healthy operation followed by an abrupt real mechanical jump directly into CRITICAL truth.",
    ),
    StressScenario(
        "thermal_runaway",
        "fault",
        "thermal",
        60.0,
        "Temperature-dominant runaway with only weak vibration response.",
    ),
    StressScenario(
        "vibration_dominant_fault",
        "fault",
        "vibration",
        60.0,
        "Vibration/acceleration-dominant degradation with little thermal change.",
    ),
    StressScenario(
        "intermittent_fault",
        "fault",
        "intermittent",
        72.0,
        "Degradation appears in repeated bursts before becoming persistent late in the lifecycle.",
    ),
    StressScenario(
        "partial_recovery",
        "fault",
        "mixed_recovery",
        84.0,
        "Degradation rises, partially recovers without maintenance, then worsens to CRITICAL.",
    ),
    StressScenario(
        "compound_fault",
        "fault",
        "compound",
        72.0,
        "Two mechanisms overlap: vibration degradation first, thermal degradation later.",
    ),
)


# stress_v2 was the independent suite after the first sensor-quality remediation. It has now
# been consumed for diagnosis and is retained as a regression suite.
SCENARIOS_V2: tuple[StressScenario, ...] = tuple(
    StressScenario(
        scenario.scenario_id,
        scenario.kind,
        scenario.fault_mode,
        scenario.duration_hours * (0.93 if index % 2 == 0 else 1.08),
        scenario.description + " stress_v2 duration/profile variant.",
    )
    for index, scenario in enumerate(SCENARIOS)
) + (
    StressScenario(
        "sensor_dropout",
        "sensor_fault_healthy",
        "sensor_fault",
        50.0,
        "Intermittent acceleration-channel near-floor dropouts on a physically healthy machine.",
    ),
    StressScenario(
        "drifting_temperature_bias",
        "sensor_fault_healthy",
        "sensor_fault",
        66.0,
        "A temperature offset appears abruptly and then drifts slowly while other channels remain healthy.",
    ),
)


# stress_v3 was the independent suite after the thermal/bias remediation. It has now been
# consumed for diagnosis and is retained as a regression suite.
SCENARIOS_V3: tuple[StressScenario, ...] = tuple(
    StressScenario(
        scenario.scenario_id,
        scenario.kind,
        scenario.fault_mode,
        scenario.duration_hours * (1.06 if index % 3 == 0 else (0.91 if index % 3 == 1 else 1.13)),
        scenario.description.replace(" stress_v2 duration/profile variant.", "")
        + " stress_v3 independent duration/profile variant.",
    )
    for index, scenario in enumerate(SCENARIOS_V2)
)


# stress_v4 was the untouched independent acceptance suite after the v3 remediation. The user has
# now run it successfully, so it is retained as consumed regression evidence. The later RUL-only
# addition does not change the classifier/status path that v4 exercised.
SCENARIOS_V4: tuple[StressScenario, ...] = tuple(
    StressScenario(
        scenario.scenario_id,
        scenario.kind,
        scenario.fault_mode,
        scenario.duration_hours * (0.96 if index % 4 == 0 else (1.09 if index % 4 == 1 else (1.17 if index % 4 == 2 else 0.88))),
        scenario.description.replace(" stress_v3 independent duration/profile variant.", "")
        + " stress_v4 independent duration/profile variant.",
    )
    for index, scenario in enumerate(SCENARIOS_V3)
)


STRESS_SPECS: dict[str, dict[str, Any]] = {
    "stress_v1": {
        "seed": STRESS_V1_SEED,
        "generator_version": STRESS_GENERATOR_VERSION,
        "scenarios": SCENARIOS,
        "evidence_role": "development_regression_consumed_v1",
    },
    "stress_v2": {
        "seed": STRESS_V2_SEED,
        "generator_version": STRESS_V2_GENERATOR_VERSION,
        "scenarios": SCENARIOS_V2,
        "evidence_role": "development_regression_consumed_v2",
    },
    "stress_v3": {
        "seed": STRESS_V3_SEED,
        "generator_version": STRESS_V3_GENERATOR_VERSION,
        "scenarios": SCENARIOS_V3,
        "evidence_role": "development_regression_consumed_v3",
    },
    "stress_v4": {
        "seed": STRESS_V4_SEED,
        "generator_version": STRESS_V4_GENERATOR_VERSION,
        "scenarios": SCENARIOS_V4,
        "evidence_role": "development_regression_consumed_v4",
    },
}


@dataclass(frozen=True)
class StressSuiteConfig:
    cadence_seconds: int = 600
    replicates: int = 2
    machines: int = 6
    line_sel: str = "LINE_1"
    suite_version: str = "stress_v1"
    seed: int | None = None
    start_time: datetime = datetime(2026, 6, 1, tzinfo=timezone.utc)

    def validate(self) -> None:
        if self.cadence_seconds < 10:
            raise ValueError("cadence_seconds must be at least 10")
        if self.replicates < 1:
            raise ValueError("replicates must be at least 1")
        if self.machines < 2:
            raise ValueError("machines must be at least 2")
        if not self.line_sel.strip():
            raise ValueError("line_sel cannot be empty")
        if self.suite_version not in STRESS_SPECS:
            raise ValueError(f"unsupported stress suite version: {self.suite_version!r}")
        expected_seed = int(STRESS_SPECS[self.suite_version]["seed"])
        if self.seed is not None and int(self.seed) != expected_seed:
            raise ValueError(
                f"{self.suite_version} uses locked seed {expected_seed}; create a new suite version instead of changing its seed"
            )


@dataclass(frozen=True)
class StressCriteria:
    max_overall_fp_rate: float = 0.15
    max_overall_fn_rate: float = 0.35
    min_overall_critical_recall: float = 0.35
    max_healthy_alert_rate: float = 0.20
    max_sensor_fault_alert_rate: float = 0.25
    max_fault_fn_rate: float = 0.40
    min_fault_critical_recall: float = 0.30
    min_sudden_critical_recall: float = 0.50

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_profiles(rng: random.Random, count: int) -> dict[str, dict[str, float]]:
    # Deliberately uses a different baseline family from synthetic.py. Values are still inside
    # VVB001 measurement ranges and include machine-to-machine offsets.
    profiles: dict[str, dict[str, float]] = {}
    for idx in range(1, count + 1):
        profiles[f"STRESS_SPINDLE_{idx:02d}"] = {
            "vrms": rng.uniform(1.0, 4.2),
            "arms": rng.uniform(0.9, 4.4),
            "crest": rng.uniform(2.2, 4.2),
            "temp": rng.uniform(31.0, 46.0),
            "phase": rng.uniform(0.0, 2.0 * math.pi),
        }
    return profiles


def _smoothstep(x: float) -> float:
    x = _clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ramp(progress: float, onset: float, end: float = 1.0, power: float = 1.0) -> float:
    if progress <= onset:
        return 0.0
    x = (progress - onset) / max(1e-9, end - onset)
    return _clip(x, 0.0, 1.0) ** power


def _scenario_truth(scenario_id: str, progress: float) -> tuple[float, bool]:
    """Return hidden physical damage and whether a sensor fault is active.

    This function is intentionally independent from the bootstrap generator's _damage_curve and
    _fault_effects helpers. The returned values are written only to the separate ground-truth file.
    """
    sensor_fault = False
    damage = 0.0
    if scenario_id in {"high_safe_load", "load_step_recovery"}:
        damage = 0.0
    elif scenario_id == "repeated_sensor_spikes":
        sensor_fault = progress >= 0.25
    elif scenario_id == "stuck_sensor":
        sensor_fault = 0.42 <= progress <= 0.72
    elif scenario_id == "temperature_bias_shift":
        sensor_fault = 0.40 <= progress <= 0.72
    elif scenario_id == "noise_burst":
        sensor_fault = 0.38 <= progress <= 0.58
    elif scenario_id == "sensor_dropout":
        sensor_fault = 0.30 <= progress <= 0.62
    elif scenario_id == "drifting_temperature_bias":
        sensor_fault = 0.35 <= progress <= 0.74
    elif scenario_id == "very_slow_degradation":
        damage = _smoothstep(_ramp(progress, 0.12, power=1.0)) ** 1.7
    elif scenario_id == "very_fast_degradation":
        damage = _ramp(progress, 0.48, power=0.55)
    elif scenario_id == "sudden_mechanical_fault":
        damage = 0.97 if progress >= 0.62 else 0.0
    elif scenario_id == "thermal_runaway":
        damage = _ramp(progress, 0.34, power=1.65)
    elif scenario_id == "vibration_dominant_fault":
        damage = _smoothstep(_ramp(progress, 0.28, power=0.9))
    elif scenario_id == "intermittent_fault":
        base = _ramp(progress, 0.22, power=0.9)
        if progress < 0.78:
            pulse = max(0.0, math.sin((progress - 0.20) * 12.0 * math.pi))
            damage = base * (0.20 + 0.65 * pulse)
        else:
            damage = _clip(0.55 + 1.8 * (progress - 0.78), 0.0, 1.0)
    elif scenario_id == "partial_recovery":
        if progress < 0.48:
            damage = 0.62 * _ramp(progress, 0.18, end=0.48, power=0.9)
        elif progress < 0.68:
            damage = 0.62 - 0.28 * _ramp(progress, 0.48, end=0.68, power=1.0)
        else:
            damage = 0.34 + 0.66 * _ramp(progress, 0.68, power=0.8)
    elif scenario_id == "compound_fault":
        first = 0.55 * _ramp(progress, 0.22, power=1.0)
        second = 0.55 * _ramp(progress, 0.58, power=0.8)
        damage = _clip(first + second, 0.0, 1.0)
    else:
        raise KeyError(scenario_id)
    return _clip(damage, 0.0, 1.0), sensor_fault


def _true_state(damage: float) -> str:
    if damage >= 0.75:
        return "CRITICAL"
    if damage >= 0.35:
        return "WARNING"
    return "NORMAL"


def _scenario_sensor_values(
    scenario_id: str,
    progress: float,
    elapsed_h: float,
    profile: dict[str, float],
    damage: float,
    sensor_fault: bool,
    rng: random.Random,
    *,
    stuck_values: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, float] | None]:
    # Operating variation uses mixed frequencies/random walk-like perturbations instead of the
    # bootstrap generator's single load sine. A clean baseline exists at the beginning of every
    # lifecycle so relative features can establish a causal reference before the stress event.
    wave = 0.08 * math.sin(2 * math.pi * elapsed_h / 1.7 + profile["phase"])
    wave += 0.045 * math.sin(2 * math.pi * elapsed_h / 5.3 + 0.4 * profile["phase"])
    load = 1.0 + wave
    if scenario_id == "high_safe_load" and progress >= 0.30:
        load += 0.32 + 0.05 * math.sin(2 * math.pi * elapsed_h / 0.8)
    elif scenario_id == "load_step_recovery" and 0.34 <= progress <= 0.66:
        load += 0.42

    vrms = profile["vrms"] * load
    arms = profile["arms"] * (1.0 + 0.35 * (load - 1.0))
    temp = profile["temp"] + 8.0 * max(0.0, load - 1.0)
    crest = profile["crest"]

    # Distinct scenario-specific physical responses. These are intentionally not the same
    # coefficients or curves used by synthetic.py.
    if scenario_id in {"very_slow_degradation", "very_fast_degradation", "partial_recovery"}:
        vrms += 9.0 * damage ** 1.25
        arms += 14.0 * damage ** 1.05
        temp += 13.0 * damage ** 1.35
        crest += 1.7 * math.sin(math.pi * damage)
    elif scenario_id == "sudden_mechanical_fault":
        vrms += 15.0 * damage
        arms += 24.0 * damage
        temp += 4.0 * damage
        crest += 4.5 * damage
    elif scenario_id == "thermal_runaway":
        temp += 31.0 * damage ** 1.1
        vrms += 2.4 * damage
        arms += 3.8 * damage
        crest += 0.4 * damage
    elif scenario_id == "vibration_dominant_fault":
        vrms += 17.0 * damage
        arms += 21.0 * damage ** 1.1
        temp += 2.2 * damage
        crest += 3.2 * math.sin(math.pi * min(1.0, damage + 0.1))
    elif scenario_id == "intermittent_fault":
        vrms += 12.0 * damage
        arms += 18.0 * damage
        temp += 7.0 * damage
        crest += 2.0 * damage
    elif scenario_id == "compound_fault":
        vib = _ramp(progress, 0.22, power=1.0)
        thermal = _ramp(progress, 0.58, power=0.8)
        vrms += 10.0 * vib
        arms += 15.0 * vib
        temp += 24.0 * thermal
        crest += 2.4 * math.sin(math.pi * vib)

    noise_scale = 1.0
    if scenario_id == "noise_burst" and sensor_fault:
        noise_scale = 7.5

    vrms += rng.gauss(0.0, 0.10 * noise_scale)
    arms += rng.gauss(0.0, 0.16 * noise_scale)
    temp += rng.gauss(0.0, 0.22 * noise_scale)
    crest += rng.gauss(0.0, 0.08 * noise_scale)

    if scenario_id == "repeated_sensor_spikes" and progress >= 0.25:
        # Deterministic narrow spikes every ~8% lifecycle, offset by replicate noise.
        phase = (progress * 12.5) % 1.0
        if phase < 0.055:
            vrms += rng.uniform(9.0, 18.0)
            arms += rng.uniform(16.0, 42.0)
            crest += rng.uniform(2.0, 6.0)
    elif scenario_id == "temperature_bias_shift" and sensor_fault:
        temp += 12.0
    elif scenario_id == "drifting_temperature_bias" and sensor_fault:
        active_progress = _clip((progress - 0.35) / max(1e-9, 0.74 - 0.35), 0.0, 1.0)
        temp += 8.0 + 5.0 * active_progress
    elif scenario_id == "sensor_dropout" and sensor_fault:
        dropout_phase = (progress * 19.0) % 1.0
        if dropout_phase < 0.30:
            arms = 0.05
            crest = max(1.05, crest)
    elif scenario_id == "stuck_sensor" and sensor_fault:
        if stuck_values is None:
            stuck_values = {"vrms": vrms, "arms": arms, "crest": crest, "temp": temp}
        vrms = stuck_values["vrms"]
        arms = stuck_values["arms"]
        crest = stuck_values["crest"]
    elif scenario_id == "stuck_sensor" and not sensor_fault:
        stuck_values = None

    vrms = _clip(vrms, 0.05, 44.0)
    arms = _clip(arms, 0.05, 180.0)
    crest = _clip(crest, 1.05, 25.0)
    # Make apeak physically consistent with acceleration RMS and crest, then add an independent
    # impact component for real fault scenarios.
    impact = 0.0
    if damage > 0:
        impact = 8.0 * damage
        if scenario_id in {"sudden_mechanical_fault", "vibration_dominant_fault"}:
            impact += 18.0 * damage
    apeak = arms * crest + impact + rng.gauss(0.0, 0.25 * noise_scale)
    apeak = _clip(max(arms, apeak), arms, 470.0)
    crest = _clip(apeak / max(arms, 1e-9), 1.0, 49.0)
    temp = _clip(temp, -20.0, 78.0)
    return {
        "vrms": vrms,
        "arms": arms,
        "apeak": apeak,
        "crest": crest,
        "temp": temp,
    }, stuck_values


def generate_stress_suite(
    output_dir: str | Path,
    config: StressSuiteConfig | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = config or StressSuiteConfig()
    config.validate()
    spec = STRESS_SPECS[config.suite_version]
    locked_seed = int(spec["seed"])
    scenarios: tuple[StressScenario, ...] = tuple(spec["scenarios"])

    root = Path(output_dir)
    suite_dir = root if root.name == config.suite_version else root / config.suite_version
    suite_dir.mkdir(parents=True, exist_ok=True)
    sensor_path = suite_dir / f"spindle_{config.suite_version}.csv"
    truth_path = suite_dir / f"spindle_{config.suite_version}_ground_truth.csv"
    manifest_path = suite_dir / f"spindle_{config.suite_version}_manifest.json"
    if not overwrite and any(path.exists() for path in (sensor_path, truth_path, manifest_path)):
        raise FileExistsError(
            f"Frozen stress suite already exists in {suite_dir}. Delete it explicitly or use --overwrite only when intentionally rebuilding the same deterministic suite."
        )

    rng = random.Random(locked_seed)
    profiles = _base_profiles(rng, config.machines)
    sensor_fields = [
        "id", "timestamp", "line_sel", "machine_id", "vrms", "arms", "apeak", "crest", "temp",
        "lifecycle_id", "lifecycle_progress",
    ]
    truth_fields = [
        "id", "lifecycle_id", "scenario_id", "scenario_kind", "fault_mode", "true_damage",
        "true_state", "is_sensor_fault",
    ]

    row_count = 0
    source_id = 1
    current_start = config.start_time
    lifecycle_records: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = {scenario.scenario_id: 0 for scenario in scenarios}

    with sensor_path.open("w", newline="", encoding="utf-8") as sensor_handle, truth_path.open(
        "w", newline="", encoding="utf-8"
    ) as truth_handle:
        sensor_writer = csv.DictWriter(sensor_handle, fieldnames=sensor_fields)
        truth_writer = csv.DictWriter(truth_handle, fieldnames=truth_fields)
        sensor_writer.writeheader()
        truth_writer.writeheader()

        lifecycle_index = 0
        for replicate in range(1, config.replicates + 1):
            for scenario in scenarios:
                lifecycle_index += 1
                machine = f"STRESS_SPINDLE_{1 + (lifecycle_index - 1) % config.machines:02d}"
                profile = profiles[machine]
                lifecycle_id = f"stress_{scenario.scenario_id}_{replicate:02d}"
                duration_h = scenario.duration_hours * rng.uniform(0.92, 1.08)
                steps = max(20, int(round(duration_h * 3600 / config.cadence_seconds)))
                stuck_values: dict[str, float] | None = None
                state_counts = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
                sensor_fault_rows = 0

                for i in range(steps):
                    progress = i / max(1, steps - 1)
                    elapsed_h = i * config.cadence_seconds / 3600.0
                    timestamp = current_start + timedelta(seconds=i * config.cadence_seconds)
                    damage, sensor_fault = _scenario_truth(scenario.scenario_id, progress)
                    truth_state = _true_state(damage)
                    values, stuck_values = _scenario_sensor_values(
                        scenario.scenario_id,
                        progress,
                        elapsed_h,
                        profile,
                        damage,
                        sensor_fault,
                        rng,
                        stuck_values=stuck_values,
                    )
                    sensor_writer.writerow(
                        {
                            "id": source_id,
                            "timestamp": timestamp.isoformat(),
                            "line_sel": config.line_sel,
                            "machine_id": machine,
                            "vrms": f"{values['vrms']:.8f}",
                            "arms": f"{values['arms']:.8f}",
                            "apeak": f"{values['apeak']:.8f}",
                            "crest": f"{values['crest']:.8f}",
                            "temp": f"{values['temp']:.8f}",
                            "lifecycle_id": lifecycle_id,
                            "lifecycle_progress": f"{progress:.8f}",
                        }
                    )
                    truth_writer.writerow(
                        {
                            "id": source_id,
                            "lifecycle_id": lifecycle_id,
                            "scenario_id": scenario.scenario_id,
                            "scenario_kind": scenario.kind,
                            "fault_mode": scenario.fault_mode,
                            "true_damage": f"{damage:.8f}",
                            "true_state": truth_state,
                            "is_sensor_fault": "1" if sensor_fault else "0",
                        }
                    )
                    source_id += 1
                    row_count += 1
                    state_counts[truth_state] += 1
                    sensor_fault_rows += int(sensor_fault)

                lifecycle_records.append(
                    {
                        "lifecycle_id": lifecycle_id,
                        "scenario_id": scenario.scenario_id,
                        "scenario_kind": scenario.kind,
                        "fault_mode": scenario.fault_mode,
                        "replicate": replicate,
                        "machine_id": machine,
                        "duration_hours": duration_h,
                        "rows": steps,
                        "true_state_counts": state_counts,
                        "sensor_fault_rows": sensor_fault_rows,
                    }
                )
                scenario_counts[scenario.scenario_id] += 1
                # Reset all rolling state between lifecycles during evaluation.
                current_start += timedelta(hours=duration_h + 30.0 + rng.uniform(0.0, 4.0))

    manifest: dict[str, Any] = {
        "suite_version": config.suite_version,
        "generator_version": str(spec["generator_version"]),
        "seed": locked_seed,
        "evidence_role": str(spec["evidence_role"]),
        "frozen": True,
        "safe_for_training": False,
        "independent_from_training_generator": True,
        "cadence_seconds": config.cadence_seconds,
        "replicates": config.replicates,
        "machines": config.machines,
        "line_sel": config.line_sel,
        "row_count": row_count,
        "lifecycle_count": len(lifecycle_records),
        "scenario_count": len(scenarios),
        "sensor_csv": sensor_path.name,
        "ground_truth_csv": truth_path.name,
        "sensor_csv_sha256": _sha256(sensor_path),
        "ground_truth_csv_sha256": _sha256(truth_path),
        "sensor_csv_contains_ground_truth": False,
        "scenario_definitions": [asdict(scenario) for scenario in scenarios],
        "scenario_lifecycle_counts": scenario_counts,
        "lifecycles": lifecycle_records,
        "truth_state_definition": {
            "NORMAL": "hidden physical damage < 0.35",
            "WARNING": "0.35 <= hidden physical damage < 0.75",
            "CRITICAL": "hidden physical damage >= 0.75",
            "note": "These are stress-simulator truth states, not manufacturer VVB001 limits and not plant labels.",
        },
        "usage_contract": (
            [
                f"{config.suite_version} has been consumed as development evidence and is now regression-only.",
                f"Do not claim {config.suite_version} as independent acceptance evidence after remediation.",
                "Ground-truth fields are stored separately and are never passed to feature engineering or model inference.",
            ]
            if str(spec["evidence_role"]).startswith("development_regression")
            else [
                f"Do not train, calibrate, or select thresholds on {config.suite_version} before its first acceptance run.",
                f"If {config.suite_version} failures are inspected and used to modify the model, this suite becomes development evidence; create a new frozen suite version for the next independent acceptance claim.",
                "Ground-truth fields are stored separately and are never passed to feature engineering or model inference.",
            ]
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "suite_dir": str(suite_dir.resolve()),
        "sensor_csv": str(sensor_path.resolve()),
        "ground_truth_csv": str(truth_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "rows": row_count,
        "lifecycles": len(lifecycle_records),
        "scenarios": len(scenarios),
        "cadence_seconds": config.cadence_seconds,
    }


def _load_and_verify_suite(suite_dir: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    root = Path(suite_dir)
    manifest_candidates = sorted(root.glob("spindle_stress_v*_manifest.json"))
    if len(manifest_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one stress manifest in {root}; found {len(manifest_candidates)}. "
            "Run 'python main.py generate-stress --suite-version stress_v4' (or the intended suite version) if needed."
        )
    manifest_path = manifest_candidates[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("suite_version") not in STRESS_SPECS:
        raise ValueError(f"Unsupported stress suite version: {manifest.get('suite_version')!r}")
    sensor_path = root / str(manifest["sensor_csv"])
    truth_path = root / str(manifest["ground_truth_csv"])
    if not sensor_path.exists() or not truth_path.exists():
        raise FileNotFoundError("Frozen stress suite is incomplete")
    sensor_hash = _sha256(sensor_path)
    truth_hash = _sha256(truth_path)
    if sensor_hash != manifest.get("sensor_csv_sha256"):
        raise RuntimeError("Frozen stress sensor CSV hash mismatch; refuse evaluation of modified stress data")
    if truth_hash != manifest.get("ground_truth_csv_sha256"):
        raise RuntimeError("Frozen stress ground-truth CSV hash mismatch; refuse evaluation of modified stress truth")
    return sensor_path, truth_path, manifest


def _parse_stress_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _runtime_predictions(
    model_path: str | Path,
    sensor_path: Path,
    sensor_config: SensorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Replay stress sensor rows through the same guarded runtime inference path used live."""
    predictor = VVB001Predictor(model_path, sensor_config)
    monitor = VVB001Monitor(
        VVB001Validator(sensor_config),
        FeatureEngine(sensor_config),
        predictor,
        SensorQualityGuard(sensor_config),
        enable_rul=False,
    )
    groups: list[str] = []
    timestamps: list[datetime] = []
    predicted: list[str] = []
    scores: list[float] = []
    quality_statuses: list[str] = []
    quality_held_counts: list[int] = []
    sensor_ids: list[int] = []
    previous_lifecycle_by_machine: dict[str, str] = {}

    with sensor_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            reading = VVB001Reading(
                source_id=int(raw["id"]),
                timestamp=_parse_stress_timestamp(raw["timestamp"]),
                line_sel=raw["line_sel"],
                machine_id=raw["machine_id"],
                vrms=float(raw["vrms"]),
                arms=float(raw["arms"]),
                apeak=float(raw["apeak"]),
                crest=float(raw["crest"]),
                temp=float(raw["temp"]),
            )
            lifecycle = raw["lifecycle_id"]
            prior = previous_lifecycle_by_machine.get(reading.machine_key)
            if prior is not None and prior != lifecycle:
                monitor.reset_machine(reading.machine_key)
            previous_lifecycle_by_machine[reading.machine_key] = lifecycle
            validation, item = monitor.process(SourceRecord(reading.source_id, dict(raw), reading))
            if item is None or item.predicted_status is None or item.degradation_score is None:
                raise RuntimeError(f"Stress row {reading.source_id} could not be scored: {validation.reasons}")
            sensor_ids.append(reading.source_id)
            groups.append(lifecycle)
            timestamps.append(reading.timestamp)
            predicted.append(item.predicted_status)
            scores.append(float(item.degradation_score))
            quality_statuses.append(item.sensor_quality_status)
            quality_held_counts.append(len(item.sensor_quality_held_sensors))

    return (
        np.asarray(groups, dtype=object),
        np.asarray(timestamps, dtype=object),
        np.asarray(predicted, dtype=object),
        np.asarray(scores, dtype=float),
        np.asarray(quality_statuses, dtype=object),
        np.asarray(quality_held_counts, dtype=int),
        sensor_ids,
    )

def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual_degraded = actual != "NORMAL"
    predicted_degraded = predicted != "NORMAL"
    tp = int(np.sum(actual_degraded & predicted_degraded))
    tn = int(np.sum(~actual_degraded & ~predicted_degraded))
    fp = int(np.sum(~actual_degraded & predicted_degraded))
    fn = int(np.sum(actual_degraded & ~predicted_degraded))
    warning_mask = actual == "WARNING"
    critical_mask = actual == "CRITICAL"
    return {
        "rows": int(len(actual)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "false_positive_rate": _safe_rate(fp, fp + tn),
        "false_negative_rate": _safe_rate(fn, fn + tp),
        "warning_operational_recall": float(np.mean(predicted[warning_mask] != "NORMAL")) if np.any(warning_mask) else None,
        "critical_exact_recall": float(np.mean(predicted[critical_mask] == "CRITICAL")) if np.any(critical_mask) else None,
        "critical_operational_recall": float(np.mean(predicted[critical_mask] != "NORMAL")) if np.any(critical_mask) else None,
        "normal_alert_rate": float(np.mean(predicted[actual == "NORMAL"] != "NORMAL")) if np.any(actual == "NORMAL") else None,
        "actual_counts": {status: int(np.sum(actual == status)) for status in ("NORMAL", "WARNING", "CRITICAL")},
        "predicted_counts": {status: int(np.sum(predicted == status)) for status in ("NORMAL", "WARNING", "CRITICAL")},
    }


def _first_onset_hours(
    lifecycle_mask: np.ndarray,
    timestamps: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    target: str,
) -> float | None:
    positions = np.flatnonzero(lifecycle_mask)
    if target == "WARNING":
        actual_positions = [pos for pos in positions if actual[pos] in {"WARNING", "CRITICAL"}]
        predicted_positions = [pos for pos in positions if predicted[pos] in {"WARNING", "CRITICAL"}]
    else:
        actual_positions = [pos for pos in positions if actual[pos] == "CRITICAL"]
        predicted_positions = [pos for pos in positions if predicted[pos] == "CRITICAL"]
    if not actual_positions or not predicted_positions:
        return None
    # Positive = model alerted before the hidden truth onset; negative = late detection.
    return float((timestamps[actual_positions[0]] - timestamps[predicted_positions[0]]).total_seconds() / 3600.0)


def _transition_reversal_rate(groups: np.ndarray, predicted: np.ndarray) -> float:
    severity = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    reversals = 0
    transitions = 0
    for lifecycle in sorted(set(groups.tolist())):
        seq = np.asarray([severity[str(value)] for value in predicted[groups == lifecycle]], dtype=int)
        diffs = np.diff(seq)
        reversals += int(np.sum(diffs < 0))
        transitions += int(np.sum(diffs != 0))
    return float(reversals / transitions) if transitions else 0.0


def _scenario_checks(kind: str, metrics: dict[str, Any], criteria: StressCriteria) -> dict[str, bool]:
    if kind == "healthy":
        rate = metrics.get("normal_alert_rate")
        return {"healthy_alert_rate": rate is not None and float(rate) <= criteria.max_healthy_alert_rate}
    if kind == "sensor_fault_healthy":
        rate = metrics.get("normal_alert_rate")
        return {"sensor_fault_alert_rate": rate is not None and float(rate) <= criteria.max_sensor_fault_alert_rate}
    if kind == "abrupt_fault":
        recall = metrics.get("critical_exact_recall")
        return {"sudden_critical_recall": recall is not None and float(recall) >= criteria.min_sudden_critical_recall}
    fn_rate = metrics.get("false_negative_rate")
    critical_recall = metrics.get("critical_exact_recall")
    return {
        "fault_false_negative_rate": fn_rate is not None and float(fn_rate) <= criteria.max_fault_fn_rate,
        "fault_critical_recall": critical_recall is not None and float(critical_recall) >= criteria.min_fault_critical_recall,
    }


def evaluate_stress_suite(
    model_path: str | Path,
    suite_dir: str | Path,
    report_path: str | Path,
    sensor_config: SensorConfig,
    *,
    criteria: StressCriteria | None = None,
) -> dict[str, Any]:
    criteria = criteria or StressCriteria()
    criteria.validate()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    sensor_path, truth_path, manifest = _load_and_verify_suite(suite_dir)
    bundle = joblib.load(model_path)
    if bundle.get("sensor_contract") != sensor_contract(sensor_config):
        raise RuntimeError("Model feature contract does not match the stress evaluator sensor configuration")
    expected_cadence = (bundle.get("metadata") or {}).get("training_cadence_seconds")
    suite_cadence = int(manifest["cadence_seconds"])
    cadence_warning = None
    if expected_cadence is not None and abs(float(expected_cadence) - suite_cadence) > 1e-6:
        cadence_warning = (
            f"Model training cadence is {expected_cadence}s but frozen stress suite cadence is {suite_cadence}s. "
            "Feature windows are time-based, but cadence mismatch should be interpreted cautiously."
        )

    groups, timestamps, pred, score, quality_status, quality_held_counts, sensor_ids = _runtime_predictions(
        model_path, sensor_path, sensor_config
    )

    truth_rows: list[dict[str, str]] = []
    with truth_path.open("r", encoding="utf-8", newline="") as handle:
        truth_rows = list(csv.DictReader(handle))
    truth_ids = [int(row["id"]) for row in truth_rows]
    if sensor_ids != truth_ids or len(truth_rows) != len(pred):
        raise RuntimeError("Stress sensor rows and hidden truth rows are not exactly aligned")

    actual = np.asarray([row["true_state"] for row in truth_rows], dtype=object)
    scenarios = np.asarray([row["scenario_id"] for row in truth_rows], dtype=object)
    scenario_kinds = np.asarray([row["scenario_kind"] for row in truth_rows], dtype=object)
    truth_damage = np.asarray([float(row["true_damage"]) for row in truth_rows], dtype=float)
    sensor_fault = np.asarray([row["is_sensor_fault"] == "1" for row in truth_rows], dtype=bool)

    metadata = dict(bundle.get("metadata") or {})

    overall = _classification_metrics(actual, pred)
    overall["sensor_quality_status_counts"] = {
        str(status): int(np.sum(quality_status == status)) for status in sorted(set(quality_status.tolist()))
    }
    overall["sensor_quality_held_row_fraction"] = float(np.mean(quality_held_counts > 0))
    overall["transition_reversal_rate"] = _transition_reversal_rate(groups, pred)
    overall["mean_degradation_score"] = float(np.mean(score))
    sensor_fault_normal = sensor_fault & (actual == "NORMAL")
    overall["sensor_fault_false_alert_rate"] = (
        float(np.mean(pred[sensor_fault_normal] != "NORMAL")) if np.any(sensor_fault_normal) else None
    )

    per_scenario: dict[str, Any] = {}
    scenario_definition_map = {item["scenario_id"]: item for item in manifest["scenario_definitions"]}
    for scenario_id in sorted(set(scenarios.tolist())):
        mask = scenarios == scenario_id
        kind_values = sorted(set(scenario_kinds[mask].tolist()))
        if len(kind_values) != 1:
            raise RuntimeError(f"Scenario {scenario_id} has inconsistent kinds")
        kind = str(kind_values[0])
        metrics = _classification_metrics(actual[mask], pred[mask])
        metrics["mean_degradation_score"] = float(np.mean(score[mask]))
        metrics["mean_true_damage"] = float(np.mean(truth_damage[mask]))
        metrics["sensor_fault_rows"] = int(np.sum(sensor_fault[mask]))
        metrics["sensor_quality_status_counts"] = {
            str(status): int(np.sum(quality_status[mask] == status))
            for status in sorted(set(quality_status[mask].tolist()))
        }
        metrics["sensor_quality_held_row_fraction"] = float(np.mean(quality_held_counts[mask] > 0))
        metrics["transition_reversal_rate"] = _transition_reversal_rate(groups[mask], pred[mask])
        lifecycle_results: list[dict[str, Any]] = []
        for lifecycle in sorted(set(groups[mask].tolist())):
            lifecycle_mask = groups == lifecycle
            lifecycle_results.append(
                {
                    "lifecycle_id": str(lifecycle),
                    "warning_detection_lead_hours": _first_onset_hours(lifecycle_mask, timestamps, actual, pred, "WARNING"),
                    "critical_detection_lead_hours": _first_onset_hours(lifecycle_mask, timestamps, actual, pred, "CRITICAL"),
                    "rows": int(np.sum(lifecycle_mask)),
                }
            )
        checks = _scenario_checks(kind, metrics, criteria)
        per_scenario[scenario_id] = {
            "scenario_kind": kind,
            "fault_mode": scenario_definition_map[scenario_id]["fault_mode"],
            "description": scenario_definition_map[scenario_id]["description"],
            "metrics": metrics,
            "lifecycles": lifecycle_results,
            "checks": checks,
            "passed": bool(all(checks.values())),
        }

    aggregate_checks = {
        "overall_false_positive_rate": overall["false_positive_rate"] is not None
        and float(overall["false_positive_rate"]) <= criteria.max_overall_fp_rate,
        "overall_false_negative_rate": overall["false_negative_rate"] is not None
        and float(overall["false_negative_rate"]) <= criteria.max_overall_fn_rate,
        "overall_critical_recall": overall["critical_exact_recall"] is not None
        and float(overall["critical_exact_recall"]) >= criteria.min_overall_critical_recall,
    }
    passed_scenarios = sum(1 for value in per_scenario.values() if value["passed"])
    report: dict[str, Any] = {
        "evaluation_type": (
            "frozen_independent_synthetic_stress"
            if manifest.get("evidence_role") == "frozen_independent_acceptance_unseen"
            else "development_regression_synthetic_stress"
        ),
        "suite_version": manifest["suite_version"],
        "evidence_role": manifest.get("evidence_role"),
        "independent_acceptance_eligible": manifest.get("evidence_role") == "frozen_independent_acceptance_unseen",
        "generator_version": manifest["generator_version"],
        "suite_integrity_verified": True,
        "independent_from_training_generator": True,
        "bootstrap_only": True,
        "model_path": str(model_path.resolve()),
        "model_metadata": metadata,
        "suite_manifest": manifest,
        "criteria": asdict(criteria),
        "cadence_warning": cadence_warning,
        "overall_metrics": overall,
        "aggregate_checks": aggregate_checks,
        "per_scenario": per_scenario,
        "passed_scenarios": passed_scenarios,
        "failed_scenarios": len(per_scenario) - passed_scenarios,
        "overall_pass": bool(all(aggregate_checks.values()) and passed_scenarios == len(per_scenario)),
        "interpretation": (
            (
                "PASS means the fixed model met the configured untouched adversarial synthetic acceptance gates without retraining or threshold fitting. "
                "It is still not evidence of real plant accuracy or safety."
            )
            if manifest.get("evidence_role") == "frozen_independent_acceptance_unseen"
            else (
                "PASS means the fixed model meets this consumed regression suite after remediation. "
                "This suite is development evidence only and must not be presented as independent acceptance."
            )
        ),
        "limitations": [
            f"{manifest['suite_version']} is synthetic and encodes deliberately adversarial assumptions rather than measured plant failure physics.",
            "The model receives only sensor-derived features; hidden true_state, true_damage, scenario_id, and sensor-fault flags are evaluator-only.",
            "No model fitting, regime re-ordering, calibration, or threshold selection occurs during stress evaluation.",
            "Any stress suite used to change the model becomes development evidence; use a newly frozen version for the next independent acceptance run.",
            "Real FP/FN and maintenance lead-time claims require plant maintenance/failure labels.",
        ],
    }
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
