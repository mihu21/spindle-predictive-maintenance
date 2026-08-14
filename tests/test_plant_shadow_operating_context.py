from __future__ import annotations

from datetime import datetime, timedelta, timezone
import csv

import pytest
from fastapi.testclient import TestClient

from vvb001_monitor.config import AppConfig
from vvb001_monitor.plant_shadow.contracts import (
    ForecastState,
    EndpointClass,
    EndpointPrecision,
    ObservationDisposition,
    OperatingState,
    OperatingStateSource,
    PlantObservation,
)
from vvb001_monitor.plant_shadow.operating_context import OperatingContextClock
from vvb001_monitor.plant_shadow.api import create_app
from vvb001_monitor.plant_shadow.runtime import FrozenRuntimeRouter
from vvb001_monitor.plant_shadow.service import PlantShadowService
from vvb001_monitor.plant_shadow.storage import EvidenceStore
from vvb001_monitor.synthetic import SyntheticConfig, generate_mock_csv
from vvb001_monitor.training import build_training_matrix


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def observation(
    row_id: int,
    hour: float,
    state: OperatingState,
    *,
    confidence: float = 1.0,
    extreme: bool = False,
) -> PlantObservation:
    source = (
        OperatingStateSource.UNAVAILABLE
        if state == OperatingState.UNKNOWN else OperatingStateSource.SYNTHETIC_FIXTURE
    )
    return PlantObservation(
        source_key="source-a",
        source_row_id=str(row_id),
        event_timestamp=BASE + timedelta(hours=hour),
        line_sel="LINE",
        machine_id="M1",
        vrms=40.0 if extreme else 2.0,
        arms=5.0,
        apeak=14.0,
        crest=2.8,
        temp=32.0,
        operating_state=state,
        operating_state_source=source,
        operating_state_confidence=confidence,
        maintenance_event_id="work-1" if state == OperatingState.MAINTENANCE else None,
    )


def test_operating_clock_excludes_off_idle_maintenance_and_unknown_wall_time():
    clock = OperatingContextClock()
    decisions = [
        clock.observe(observation(1, 0, OperatingState.RUNNING)),
        clock.observe(observation(2, 1, OperatingState.RUNNING)),
        clock.observe(observation(3, 2, OperatingState.IDLE)),
        clock.observe(observation(4, 3, OperatingState.OFF)),
        clock.observe(observation(5, 4, OperatingState.MAINTENANCE)),
        clock.observe(observation(6, 8, OperatingState.UNKNOWN, confidence=0.0)),
        clock.observe(observation(7, 10, OperatingState.RUNNING)),
        clock.observe(observation(8, 11, OperatingState.RUNNING)),
    ]

    assert [decision.admitted_to_runtime for decision in decisions] == [
        True, True, False, False, False, False, False, True
    ]
    assert decisions[6].reason_code == "OPERATING_RESUME_WARMUP"
    assert decisions[-1].cumulative_operating_seconds == 7200.0
    assert decisions[-1].effective_operating_timestamp == BASE + timedelta(hours=2)


def test_unknown_and_low_confidence_running_fail_closed():
    clock = OperatingContextClock()
    unknown = clock.observe(observation(1, 0, OperatingState.UNKNOWN, confidence=0.0))
    low = clock.observe(observation(2, 1, OperatingState.RUNNING, confidence=0.5))
    assert unknown.reason_code == "OPERATING_STATE_UNKNOWN"
    assert low.reason_code == "OPERATING_STATE_LOW_CONFIDENCE"
    assert not unknown.admitted_to_runtime
    assert not low.admitted_to_runtime


def test_service_retains_off_row_clears_rul_and_preserves_extreme_raw_safety(tmp_path):
    database = tmp_path / "shadow.db"
    sensor = AppConfig.load("config/vvb001.json").sensor
    router = FrozenRuntimeRouter(
        "models/rul_v2_7_full_cadence.joblib",
        sensor,
        deployment_id="operating-context-test",
        runtime_manifest_id="operating-context-test",
    )
    with EvidenceStore(database) as store:
        service = PlantShadowService(store, router)
        held = service.process(observation(1, 0, OperatingState.OFF, extreme=True))
        assert held.disposition == ObservationDisposition.HELD_OPERATING_CONTEXT
        assert held.lifecycle_id is None
        prediction = store.db.execute(
            "SELECT * FROM prediction_attempts WHERE ingestion_id=?", (held.ingestion_id,)
        ).fetchone()
        assert prediction["forecast_state"] == ForecastState.PAUSED
        assert prediction["health_state_manufacturer"] == "CRITICAL"
        assert prediction["warning_point_hours"] is None
        assert prediction["critical_point_hours"] is None
        assert store.db.execute("SELECT COUNT(*) FROM raw_observations").fetchone()[0] == 1
        assert store.db.execute("SELECT COUNT(*) FROM lifecycle_records").fetchone()[0] == 0

        running = service.process(observation(2, 1, OperatingState.RUNNING))
        assert running.lifecycle_id is not None
        assert store.db.execute("SELECT COUNT(*) FROM lifecycle_records").fetchone()[0] == 1
        context = store.db.execute(
            "SELECT operating_state,admitted_to_runtime FROM operating_context_decisions "
            "WHERE ingestion_id=?",
            (running.ingestion_id,),
        ).fetchone()
        assert tuple(context) == ("RUNNING", 1)
    client = TestClient(create_app(database))
    machines = client.get("/api/v1/machines").json()["items"]
    assert machines[0]["operating_state"] == "RUNNING"
    assert machines[0]["cumulative_operating_hours"] == 0.0
    sensors = client.get("/api/v1/machines/source-a%3A%3ALINE%3A%3AM1/sensors").json()["items"]
    assert [row["operating_state"] for row in sensors] == ["OFF", "RUNNING"]
    assert [row["admitted_to_runtime"] for row in sensors] == [0, 1]


def test_runtime_rebuild_replays_held_context_without_contaminating_operating_clock(tmp_path):
    sensor = AppConfig.load("config/vvb001.json").sensor
    database = tmp_path / "shadow.db"
    with EvidenceStore(database) as store:
        router = FrozenRuntimeRouter(
            "models/rul_v2_7_full_cadence.joblib",
            sensor,
            deployment_id="rebuild-test",
            runtime_manifest_id="rebuild-test",
        )
        service = PlantShadowService(store, router)
        service.process(observation(1, 0, OperatingState.RUNNING))
        service.process(observation(2, 1, OperatingState.RUNNING))
        service.process(observation(3, 2, OperatingState.MAINTENANCE))
        service.process(observation(4, 8, OperatingState.RUNNING))
        router.rebuild_source("source-a", store.committed_observations("source-a"))
        resumed = service.process(observation(5, 9, OperatingState.RUNNING))
        context = store.db.execute(
            "SELECT admitted_to_runtime,cumulative_operating_seconds FROM operating_context_decisions "
            "WHERE ingestion_id=?",
            (resumed.ingestion_id,),
        ).fetchone()
        assert tuple(context) == (1, 7200.0)


def test_exact_truth_uses_operating_hours_when_endpoint_context_is_available(tmp_path):
    sensor = AppConfig.load("config/vvb001.json").sensor
    with EvidenceStore(tmp_path / "shadow.db") as store:
        router = FrozenRuntimeRouter(
            "models/rul_v2_7_full_cadence.joblib",
            sensor,
            deployment_id="truth-test",
            runtime_manifest_id="truth-test",
        )
        service = PlantShadowService(store, router)
        first = service.process(observation(1, 0, OperatingState.RUNNING))
        service.process(observation(2, 1, OperatingState.RUNNING))
        service.process(observation(3, 8, OperatingState.OFF))
        evidence = store.add_endpoint_evidence(
            source_system="synthetic-independent",
            external_event_id="warning-operating-time",
            machine_uid="source-a::LINE::M1",
            lifecycle_id=str(first.lifecycle_id),
            precision=EndpointPrecision.EXACT_TIMESTAMP,
            event_time_lower=BASE + timedelta(hours=8),
            event_time_upper=BASE + timedelta(hours=8),
            details={"fixture": True},
            actor="test",
        )
        store.confirm_endpoint_classification(evidence, EndpointClass.WARNING_ONSET, actor="test")
        truth = store.rows(
            "SELECT r.source_row_id,t.true_hours FROM target_truth t "
            "JOIN prediction_attempts p ON p.prediction_id=t.prediction_id "
            "JOIN raw_observations r ON r.ingestion_id=p.ingestion_id "
            "WHERE t.target='WARNING' ORDER BY r.ingestion_id"
        )
        assert [(row["source_row_id"], row["true_hours"]) for row in truth] == [
            ("1", 1.0), ("2", 0.0), ("3", 0.0)
        ]


def test_confirmed_component_replacement_starts_clean_operating_lifecycle(tmp_path):
    sensor = AppConfig.load("config/vvb001.json").sensor
    with EvidenceStore(tmp_path / "shadow.db") as store:
        router = FrozenRuntimeRouter(
            "models/rul_v2_7_full_cadence.joblib",
            sensor,
            deployment_id="replacement-test",
            runtime_manifest_id="replacement-test",
        )
        service = PlantShadowService(store, router)
        first = service.process(observation(1, 0, OperatingState.RUNNING))
        service.process(observation(2, 1, OperatingState.RUNNING))
        evidence = store.add_endpoint_evidence(
            source_system="cmms",
            external_event_id="component-replacement",
            machine_uid="source-a::LINE::M1",
            lifecycle_id=str(first.lifecycle_id),
            precision=EndpointPrecision.EXACT_TIMESTAMP,
            event_time_lower=BASE + timedelta(hours=2),
            event_time_upper=BASE + timedelta(hours=2),
            details={"component": "spindle"},
            actor="test",
        )
        store.confirm_endpoint_classification(evidence, EndpointClass.COMPONENT_REPLACEMENT, actor="test")
        off = service.process(observation(3, 2, OperatingState.OFF))
        assert off.lifecycle_id is None
        replacement = service.process(observation(4, 8, OperatingState.RUNNING))
        assert replacement.lifecycle_id != first.lifecycle_id
        context = store.db.execute(
            "SELECT cumulative_operating_seconds,admitted_to_runtime "
            "FROM operating_context_decisions WHERE ingestion_id=?",
            (replacement.ingestion_id,),
        ).fetchone()
        assert tuple(context) == (0.0, 1)


def test_duty_cycled_mock_covers_all_states_and_is_refused_for_training(tmp_path):
    path = tmp_path / "duty_cycle.csv"
    rows = generate_mock_csv(
        path,
        SyntheticConfig(
            lifecycles=4,
            machines=2,
            cadence_seconds=1800,
            seed=812,
            duration_min_hours=12.0,
            duration_max_hours=13.0,
            duty_cycled_operating_context=True,
        ),
    )
    with path.open("r", encoding="utf-8", newline="") as stream:
        generated = list(csv.DictReader(stream))
    assert rows == len(generated)
    assert {row["operating_state"] for row in generated} == {
        "RUNNING", "IDLE", "OFF", "MAINTENANCE", "UNKNOWN"
    }
    maintenance = [row for row in generated if row["operating_state"] == "MAINTENANCE"]
    assert maintenance
    assert all(row["maintenance_event_id"] for row in maintenance)
    with pytest.raises(ValueError, match="runtime tests, not training data"):
        build_training_matrix(path, AppConfig.load("config/vvb001.json").sensor)


def test_vibration_inference_mock_hides_state_labels_and_is_refused_for_training(tmp_path):
    path = tmp_path / "vibration_inference.csv"
    generate_mock_csv(
        path,
        SyntheticConfig(
            lifecycles=3,
            machines=2,
            cadence_seconds=300,
            seed=913,
            duration_min_hours=14.0,
            duration_max_hours=15.0,
            duty_cycled_operating_context=True,
            vibration_inference_fixture=True,
        ),
    )
    with path.open("r", encoding="utf-8", newline="") as stream:
        generated = list(csv.DictReader(stream))
    assert generated
    assert {row["vibration_inference_fixture"] for row in generated} == {"1"}
    assert {row["operating_state"] for row in generated} == {""}
    assert {row["operating_state_source"] for row in generated} == {""}
    assert {row["synthetic_true_operating_state"] for row in generated} == {
        "RUNNING", "IDLE", "OFF", "MAINTENANCE", "UNKNOWN"
    }
    with pytest.raises(ValueError, match="Vibration operating-inference fixtures"):
        build_training_matrix(path, AppConfig.load("config/vvb001.json").sensor)


def test_local_operating_evidence_resolves_missing_source_state_and_conflicts_fail_closed(tmp_path):
    with EvidenceStore(tmp_path / "shadow.db") as store:
        store.add_operating_state_evidence(
            source_system="plc",
            external_event_id="run-shift-1",
            machine_uid="source-a::LINE::M1",
            operating_state=OperatingState.RUNNING,
            operating_state_source=OperatingStateSource.PLC,
            confidence=0.99,
            effective_from=BASE,
            effective_to=BASE + timedelta(hours=4),
            maintenance_event_id=None,
            details={},
            actor="test",
        )
        raw_unknown = observation(1, 1, OperatingState.UNKNOWN, confidence=0.0)
        resolved = store.resolve_operating_context(raw_unknown)
        assert resolved.operating_state == OperatingState.RUNNING
        assert resolved.operating_state_source == OperatingStateSource.PLC
        assert resolved.operating_state_confidence == 0.99

        store.add_operating_state_evidence(
            source_system="cmms",
            external_event_id="maintenance-conflict",
            machine_uid="source-a::LINE::M1",
            operating_state=OperatingState.MAINTENANCE,
            operating_state_source=OperatingStateSource.CMMS,
            confidence=1.0,
            effective_from=BASE + timedelta(minutes=30),
            effective_to=BASE + timedelta(hours=2),
            maintenance_event_id="work-conflict",
            details={},
            actor="test",
        )
        conflict = store.resolve_operating_context(raw_unknown)
        assert conflict.operating_state == OperatingState.UNKNOWN
        assert conflict.operating_state_source == OperatingStateSource.UNAVAILABLE
        assert conflict.operating_state_confidence == 0.0
        assert conflict.operating_context_block_reason == "CONTRADICTORY_LOCAL_OPERATING_EVIDENCE"


def test_v1_ledger_upgrade_backfills_unknown_context_without_admitting_history(tmp_path):
    path = tmp_path / "legacy-shadow.db"
    with EvidenceStore(path) as store:
        store.db.execute(
            "INSERT INTO raw_observations("
            "source_key,source_row_id,machine_uid,event_timestamp,observed_at,line_sel,machine_id,"
            "vrms,arms,apeak,crest,temp,raw_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "source-a", "legacy-1", "source-a::LINE::M1", BASE.isoformat(), BASE.isoformat(),
                "LINE", "M1", 2.0, 5.0, 14.0, 2.8, 32.0, "{}",
            ),
        )
        store.db.execute("DROP TABLE operating_context_decisions")
        store.db.execute(
            "UPDATE schema_info SET schema_version='plant_shadow_schema_v1' WHERE singleton=1"
        )

    with EvidenceStore(path) as upgraded:
        row = upgraded.db.execute(
            "SELECT operating_state,operating_state_source,operating_state_confidence,"
            "admitted_to_runtime,reason_code FROM operating_context_decisions"
        ).fetchone()
        assert tuple(row) == (
            "UNKNOWN", "UNAVAILABLE", 0.0, 0, "LEGACY_CONTEXT_UNAVAILABLE"
        )
        version = upgraded.db.execute(
            "SELECT schema_version FROM schema_info WHERE singleton=1"
        ).fetchone()[0]
        assert version == "plant_shadow_schema_v3_vibration_operating_inference"
