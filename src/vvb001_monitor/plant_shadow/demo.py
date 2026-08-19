from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .contracts import (
    EndpointClass,
    EndpointPrecision,
    OperatingState,
    OperatingStateSource,
    PlantObservation,
    machine_uid,
    require_aware,
)
from .manifest import sha256_file
from .runtime import FrozenRuntimeRouter
from .service import PlantShadowService
from .storage import EvidenceStore


DEMO_GENERATOR_VERSION = "plant_shadow_local_demo_v1"
DEMO_SOURCE_KEY = "local-demo-vvb001"
DEFAULT_DEMO_DATABASE = "output/plant_shadow_demo/plant_shadow_demo.db"
DEFAULT_START_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
MINIMUM_SCENARIO_MACHINES = 6


@dataclass(frozen=True)
class DemoConfig:
    machines: int = 8
    hours: float = 72.0
    seed: int = 42
    start_time: datetime = DEFAULT_START_TIME
    cadence_minutes: int = 10
    line_sel: str = "DEMO_LINE"
    actor: str = "local-demo-generator"

    def validate(self) -> None:
        if self.machines < MINIMUM_SCENARIO_MACHINES:
            raise ValueError(
                f"machines must be at least {MINIMUM_SCENARIO_MACHINES} to cover the required demo scenarios"
            )
        if self.machines > 50:
            raise ValueError("machines must not exceed 50")
        if self.hours <= 0:
            raise ValueError("hours must be positive")
        if self.cadence_minutes < 1 or self.cadence_minutes > 60:
            raise ValueError("cadence_minutes must be within [1,60]")
        if not self.line_sel.strip() or not self.actor.strip():
            raise ValueError("line_sel and actor must be non-empty")
        require_aware(self.start_time, "start_time")
        if self.steps < 12:
            raise ValueError("hours and cadence must generate at least 12 observations per machine")

    @property
    def steps(self) -> int:
        return max(1, int(round(self.hours * 60.0 / self.cadence_minutes)))


@dataclass(frozen=True)
class Scenario:
    machine_id: str
    name: str
    description: str


@dataclass(frozen=True)
class _Sample:
    state: OperatingState
    source: OperatingStateSource
    confidence: float
    vrms: float
    arms: float
    apeak: float
    crest: float
    temp: float
    maintenance_event_id: str | None = None
    phase: str = "RUNNING"


BASE_SCENARIOS = (
    Scenario("DEMO_HEALTHY", "HEALTHY_RUNNING", "Healthy confirmed RUNNING history with low degradation."),
    Scenario(
        "DEMO_DEGRADING",
        "PROGRESSIVE_DEGRADATION_SENSOR_FAULT",
        "Progressive physical degradation plus a bounded dropout/recovery episode.",
    ),
    Scenario("DEMO_IDLE", "RUNNING_IDLE_RUNNING", "Confirmed RUNNING pauses in IDLE and later resumes."),
    Scenario("DEMO_OFF", "RUNNING_OFF_RUNNING", "Confirmed RUNNING pauses in OFF and later resumes."),
    Scenario(
        "DEMO_UNKNOWN",
        "UNKNOWN_FAIL_CLOSED",
        "Missing operating evidence leaves the final interval UNKNOWN and not admitted.",
    ),
    Scenario(
        "DEMO_MAINTENANCE",
        "MAINTENANCE_CONFIRMED_REPLACEMENT",
        "Degradation enters MAINTENANCE, closes on confirmed replacement, then starts a new lifecycle.",
    ),
    Scenario(
        "DEMO_SENSOR_FAULT",
        "SENSOR_QUALITY_FAULT",
        "Dedicated dropout and stuck-sensor evidence followed by recovery.",
    ),
    Scenario(
        "DEMO_WITHHELD",
        "FORECAST_SUPPORT_GATING",
        "Valid but atypical RUNNING telemetry exercises the frozen model support gates.",
    ),
)


def _scenario(index: int) -> Scenario:
    if index < len(BASE_SCENARIOS):
        return BASE_SCENARIOS[index]
    suffix = index - len(BASE_SCENARIOS) + 1
    return Scenario(
        f"DEMO_HEALTHY_{suffix:02d}",
        "HEALTHY_RUNNING_REPLICA",
        "Additional deterministic healthy RUNNING machine with an independent profile.",
    )


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _running_values(
    rng: random.Random,
    *,
    index: int,
    progress: float,
    damage: float = 0.0,
    atypical: bool = False,
) -> tuple[float, float, float, float, float]:
    if atypical:
        base_vrms, base_arms, base_temp = 25.0, 72.0, 63.0
        load = 1.0 + 0.015 * math.sin(progress * 18.0 * math.pi)
        vrms = base_vrms * load + rng.gauss(0.0, 0.08)
        arms = base_arms * (2.0 - load) + rng.gauss(0.0, 0.18)
        crest = 4.25 + rng.gauss(0.0, 0.015)
        apeak = arms * crest
        temp = base_temp + 0.8 * math.sin(progress * 5.0 * math.pi) + rng.gauss(0.0, 0.05)
        return vrms, arms, apeak, crest, temp

    base_vrms = 1.20 + 0.18 * (index % 5)
    base_arms = 1.65 + 0.22 * (index % 4)
    base_temp = 35.0 + 1.2 * (index % 4)
    wave = math.sin(progress * 16.0 * math.pi + index * 0.7)
    load = 1.0 + 0.055 * wave
    vrms = base_vrms * load + 7.5 * damage + rng.gauss(0.0, 0.035 + 0.05 * damage)
    arms = base_arms * (0.98 + 0.04 * load) + 13.0 * damage + rng.gauss(0.0, 0.06 + 0.08 * damage)
    crest = 2.65 + 1.6 * math.sin(math.pi * damage) + rng.gauss(0.0, 0.025)
    if damage > 0.82:
        crest -= 1.2 * (damage - 0.82)
    crest = _clip(crest, 1.05, 12.0)
    apeak = arms * crest + rng.gauss(0.0, 0.08 + 0.15 * damage)
    apeak = max(arms, apeak)
    crest = apeak / max(arms, 1e-9)
    temp = base_temp + 13.0 * damage + 0.6 * wave + rng.gauss(0.0, 0.10)
    return (
        _clip(vrms, 0.0, 44.5),
        _clip(arms, 0.02, 180.0),
        _clip(apeak, arms, 470.0),
        _clip(crest, 1.0, 49.0),
        _clip(temp, -29.0, 79.0),
    )


def _sample_for(
    scenario: Scenario,
    *,
    index: int,
    step: int,
    steps: int,
    rng: random.Random,
) -> _Sample:
    progress = step / max(1, steps - 1)
    state = OperatingState.RUNNING
    source = OperatingStateSource.SYNTHETIC_FIXTURE
    confidence = 1.0
    phase = "RUNNING"
    maintenance_event_id: str | None = None
    damage = 0.0
    atypical = scenario.name == "FORECAST_SUPPORT_GATING"

    if scenario.name == "PROGRESSIVE_DEGRADATION_SENSOR_FAULT":
        damage = _clip((progress - 0.22) / 0.78, 0.0, 1.0) ** 1.55
    elif scenario.name == "MAINTENANCE_CONFIRMED_REPLACEMENT":
        if progress < 0.55:
            damage = _clip(progress / 0.55, 0.0, 1.0) ** 1.4
        elif progress < 0.68:
            state = OperatingState.MAINTENANCE
            phase = "MAINTENANCE"
            maintenance_event_id = "DEMO-CONFIRMED-REPLACEMENT-001"
            damage = 0.85
        else:
            phase = "POST_REPLACEMENT_RUNNING"
            damage = 0.02 * ((progress - 0.68) / 0.32)
    elif scenario.name == "RUNNING_IDLE_RUNNING" and 0.48 <= progress < 0.68:
        state = OperatingState.IDLE
        phase = "IDLE"
    elif scenario.name == "RUNNING_OFF_RUNNING" and 0.48 <= progress < 0.68:
        state = OperatingState.OFF
        phase = "OFF"
    elif scenario.name == "UNKNOWN_FAIL_CLOSED" and progress >= 0.72:
        state = OperatingState.UNKNOWN
        source = OperatingStateSource.UNAVAILABLE
        confidence = 0.0
        phase = "UNKNOWN_MISSING_EVIDENCE"

    vrms, arms, apeak, crest, temp = _running_values(
        rng,
        index=index,
        progress=progress,
        damage=damage,
        atypical=atypical,
    )

    if state == OperatingState.IDLE:
        vrms, arms, temp = 0.28 + rng.random() * 0.04, 0.24 + rng.random() * 0.04, 32.0 + rng.random()
        crest = 2.4 + rng.random() * 0.2
        apeak = arms * crest
    elif state == OperatingState.OFF:
        vrms, arms, temp = 0.03 + rng.random() * 0.02, 0.025 + rng.random() * 0.02, 27.0 + rng.random()
        crest = 2.0 + rng.random() * 0.2
        apeak = arms * crest
    elif state == OperatingState.MAINTENANCE:
        vrms, arms, temp = 9.0 + rng.random() * 4.0, 15.0 + rng.random() * 8.0, 40.0 + rng.random() * 2.0
        crest = 2.8 + rng.random() * 0.5
        apeak = arms * crest

    # Fault inputs stay valid raw evidence; the existing SensorQualityGuard decides quarantine.
    fault_profile = scenario.name in {"PROGRESSIVE_DEGRADATION_SENSOR_FAULT", "SENSOR_QUALITY_FAULT"}
    if fault_profile and 0.42 <= progress < 0.46:
        vrms, arms, apeak, crest = 0.02, 0.02, 0.04, 2.0
        phase = "SENSOR_DROPOUT"
    elif scenario.name == "SENSOR_QUALITY_FAULT" and 0.56 <= progress < 0.64:
        vrms, arms, temp = 2.345678, 3.456789, 41.234567
        crest = 2.75
        apeak = arms * crest
        phase = "SENSOR_STUCK"

    return _Sample(
        state,
        source,
        confidence,
        vrms,
        arms,
        apeak,
        crest,
        temp,
        maintenance_event_id,
        phase,
    )


def _run_id(config: DemoConfig) -> str:
    payload = json.dumps(
        {
            "generator_version": DEMO_GENERATOR_VERSION,
            "machines": config.machines,
            "hours": config.hours,
            "seed": config.seed,
            "start_time": require_aware(config.start_time, "start_time").isoformat(),
            "cadence_minutes": config.cadence_minutes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "demo_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            candidate.unlink()


def _normal_database_path(root: Path) -> Path:
    return (root / "output/plant_shadow/plant_shadow.db").resolve()


def _summary(store: EvidenceStore, run_id: str) -> dict[str, Any]:
    rows = store.db.execute(
        """
        SELECT m.machine_uid,m.machine_id,
            GROUP_CONCAT(DISTINCT s.scenario_name) AS scenarios,
            (SELECT o.operating_state FROM operating_context_decisions o
             WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1) AS final_operating_state,
            (SELECT o.admitted_to_runtime FROM operating_context_decisions o
             WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1) AS runtime_admitted,
            (SELECT o.reason_code FROM operating_context_decisions o
             WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1) AS operating_reason,
            (SELECT p.forecast_state FROM prediction_attempts p
             WHERE p.machine_uid=m.machine_uid ORDER BY p.ingestion_id DESC LIMIT 1) AS forecast_state,
            (SELECT p.warning_withhold_reason FROM prediction_attempts p
             WHERE p.machine_uid=m.machine_uid ORDER BY p.ingestion_id DESC LIMIT 1) AS warning_reason,
            (SELECT l.lifecycle_id FROM lifecycle_records l
             WHERE l.machine_uid=m.machine_uid ORDER BY l.sequence_number DESC LIMIT 1) AS lifecycle_id,
            (SELECT l.sequence_number FROM lifecycle_records l
             WHERE l.machine_uid=m.machine_uid ORDER BY l.sequence_number DESC LIMIT 1) AS lifecycle_sequence,
            (SELECT l.status FROM lifecycle_records l
             WHERE l.machine_uid=m.machine_uid ORDER BY l.sequence_number DESC LIMIT 1) AS lifecycle_status
        FROM machine_registry m
        LEFT JOIN demo_machine_scenarios s ON s.run_id=? AND s.machine_uid=m.machine_uid
        GROUP BY m.machine_uid,m.machine_id ORDER BY m.machine_id
        """,
        (run_id,),
    ).fetchall()
    counts = {
        "observations": int(store.db.execute("SELECT COUNT(*) FROM raw_observations").fetchone()[0]),
        "predictions": int(store.db.execute("SELECT COUNT(*) FROM prediction_attempts").fetchone()[0]),
        "withheld_predictions": int(
            store.db.execute("SELECT COUNT(*) FROM prediction_attempts WHERE forecast_state='WITHHELD'").fetchone()[0]
        ),
        "paused_predictions": int(
            store.db.execute("SELECT COUNT(*) FROM prediction_attempts WHERE forecast_state='PAUSED'").fetchone()[0]
        ),
        "lifecycles": int(store.db.execute("SELECT COUNT(*) FROM lifecycle_records").fetchone()[0]),
    }
    return {"counts": counts, "machines": [dict(row) for row in rows]}


def generate_demo_database(
    database: str | Path,
    *,
    config: DemoConfig,
    app_config_path: str | Path = "config/vvb001.json",
    model_path: str | Path = "models/rul_v2_7_full_cadence.joblib",
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild an isolated SQLite demo by exercising the production shadow runtime path."""
    config.validate()
    root = Path(repository_root or Path.cwd()).resolve()
    target = Path(database)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if target == _normal_database_path(root):
        raise ValueError("demo generation refuses to use the normal plant-shadow database path")

    model = Path(model_path)
    if not model.is_absolute():
        model = root / model
    app_config = Path(app_config_path)
    if not app_config.is_absolute():
        app_config = root / app_config
    if not model.is_file():
        raise FileNotFoundError(model)
    if not app_config.is_file():
        raise FileNotFoundError(app_config)

    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.with_name(target.name + ".building")
    _remove_sqlite_files(building)
    run_id = _run_id(config)
    model_sha256 = sha256_file(model)
    scenarios = [_scenario(index) for index in range(config.machines)]
    machine_uids = {
        item.machine_id: machine_uid(DEMO_SOURCE_KEY, config.line_sel, item.machine_id)
        for item in scenarios
    }

    try:
        with EvidenceStore(building) as store:
            store.record_demo_run(
                {
                    "run_id": run_id,
                    "seed": config.seed,
                    "requested_machines": config.machines,
                    "hours": config.hours,
                    "cadence_minutes": config.cadence_minutes,
                    "start_time": require_aware(config.start_time, "start_time").isoformat(),
                    "generator_version": DEMO_GENERATOR_VERSION,
                    "generated_at": require_aware(config.start_time, "start_time").isoformat(),
                    "model_artifact_sha256": model_sha256,
                    "metadata": {
                        "evidence_kind": "DETERMINISTIC_SYNTHETIC_PLANT_SHADOW",
                        "postgresql_accessed": False,
                        "credentials_required": False,
                        "scenario_labels_are_model_inputs": False,
                    },
                }
            )
            store.save_source_config(
                DEMO_SOURCE_KEY,
                {
                    "source_kind": "DETERMINISTIC_LOCAL_DEMO",
                    "row_id_kind": "integer",
                    "read_path": "IN_PROCESS_GENERATOR",
                    "postgresql": False,
                    "credentials_required": False,
                    "generator_version": DEMO_GENERATOR_VERSION,
                },
                actor=config.actor,
                enabled=False,
            )
            for item in scenarios:
                store.record_demo_machine_scenario(
                    run_id=run_id,
                    machine_uid=machine_uids[item.machine_id],
                    machine_id=item.machine_id,
                    scenario_name=item.name,
                    description=item.description,
                    metadata={"synthetic": True},
                )
            # With six machines these behaviors are deliberately combined into production-path
            # histories; larger demos also receive their dedicated profiles above.
            if config.machines == MINIMUM_SCENARIO_MACHINES:
                store.record_demo_machine_scenario(
                    run_id=run_id,
                    machine_uid=machine_uids["DEMO_DEGRADING"],
                    machine_id="DEMO_DEGRADING",
                    scenario_name="SENSOR_QUALITY_FAULT",
                    description="The degrading profile contains a bounded dropout and recovery episode.",
                    metadata={"synthetic": True, "combined_profile": True},
                )
                store.record_demo_machine_scenario(
                    run_id=run_id,
                    machine_uid=machine_uids["DEMO_HEALTHY"],
                    machine_id="DEMO_HEALTHY",
                    scenario_name="FORECAST_SUPPORT_GATING",
                    description="Early healthy history exposes genuine initialization/support gating decisions.",
                    metadata={"synthetic": True, "combined_profile": True},
                )

            router = FrozenRuntimeRouter(
                model,
                AppConfig.load(app_config).sensor,
                deployment_id="vvb001-local-demo-v1",
                runtime_manifest_id="vvb001-local-demo-runtime-v1",
            )
            service = PlantShadowService(store, router)
            row_id = 0
            replacement_closed = False
            maintenance_lifecycle_id: str | None = None
            rngs = {
                item.machine_id: random.Random(config.seed * 1009 + index * 9176)
                for index, item in enumerate(scenarios)
            }
            latest_timestamp = config.start_time
            for step in range(config.steps):
                timestamp = require_aware(config.start_time, "start_time") + timedelta(
                    minutes=step * config.cadence_minutes
                )
                latest_timestamp = timestamp
                for index, item in enumerate(scenarios):
                    row_id += 1
                    sample = _sample_for(
                        item,
                        index=index,
                        step=step,
                        steps=config.steps,
                        rng=rngs[item.machine_id],
                    )
                    if (
                        item.machine_id == "DEMO_MAINTENANCE"
                        and sample.phase == "POST_REPLACEMENT_RUNNING"
                        and not replacement_closed
                    ):
                        if maintenance_lifecycle_id is None:
                            raise RuntimeError("maintenance replacement scenario has no active lifecycle to close")
                        evidence_id = store.add_endpoint_evidence(
                            source_system="LOCAL_DEMO_GENERATOR",
                            external_event_id=f"{run_id}:replacement",
                            machine_uid=machine_uids[item.machine_id],
                            lifecycle_id=maintenance_lifecycle_id,
                            precision=EndpointPrecision.EXACT_TIMESTAMP,
                            event_time_lower=timestamp,
                            event_time_upper=timestamp,
                            details={
                                "demo": True,
                                "confirmed_component_replacement": True,
                                "maintenance_event_id": "DEMO-CONFIRMED-REPLACEMENT-001",
                            },
                            actor=config.actor,
                        )
                        store.confirm_endpoint_classification(
                            evidence_id,
                            EndpointClass.COMPONENT_REPLACEMENT,
                            actor=config.actor,
                        )
                        replacement_closed = True
                    observation = PlantObservation(
                        source_key=DEMO_SOURCE_KEY,
                        source_row_id=str(row_id),
                        event_timestamp=timestamp,
                        observed_at=timestamp,
                        line_sel=config.line_sel,
                        machine_id=item.machine_id,
                        vrms=sample.vrms,
                        arms=sample.arms,
                        apeak=sample.apeak,
                        crest=sample.crest,
                        temp=sample.temp,
                        operating_state=sample.state,
                        operating_state_source=sample.source,
                        operating_state_confidence=sample.confidence,
                        maintenance_event_id=sample.maintenance_event_id,
                        operating_context_block_reason=(
                            "DEMO_OPERATING_EVIDENCE_MISSING"
                            if sample.phase == "UNKNOWN_MISSING_EVIDENCE"
                            else None
                        ),
                        raw_payload={
                            "demo": True,
                            "demo_run_id": run_id,
                            "scenario_name": item.name,
                            "scenario_phase": sample.phase,
                            "generator_version": DEMO_GENERATOR_VERSION,
                            "seed": config.seed,
                            "synthetic_row_number": row_id,
                        },
                    )
                    result = service.process(observation)
                    if item.machine_id == "DEMO_MAINTENANCE" and sample.phase == "MAINTENANCE":
                        maintenance_lifecycle_id = result.lifecycle_id

            if not replacement_closed:
                raise RuntimeError("demo duration did not exercise confirmed replacement lifecycle closure")
            store.record_source_health(
                DEMO_SOURCE_KEY,
                "OFFLINE_DEMO_COMPLETE",
                latest_source_timestamp=latest_timestamp.isoformat(),
                message="Deterministic in-process generator; PostgreSQL was not accessed.",
            )
            summary = _summary(store, run_id)
            summary.update(
                {
                    "run_id": run_id,
                    "environment_mode": "DEMO",
                    "database": str(target),
                    "seed": config.seed,
                    "machines_requested": config.machines,
                    "hours": config.hours,
                    "cadence_minutes": config.cadence_minutes,
                    "start_time": require_aware(config.start_time, "start_time").isoformat(),
                    "generator_version": DEMO_GENERATOR_VERSION,
                    "model_artifact_sha256": model_sha256,
                    "postgresql_accessed": False,
                    "credentials_required": False,
                    "plant_production_authorized": False,
                }
            )

        os.replace(building, target)
        _remove_sqlite_files(Path(str(target) + ".building"))
        return summary
    except Exception:
        _remove_sqlite_files(building)
        raise
