from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from vvb001_monitor.plant_shadow.api import create_app
from vvb001_monitor.plant_shadow.commands import DEFAULT_DATABASE
from vvb001_monitor.plant_shadow.demo import DemoConfig, generate_demo_database
from vvb001_monitor.plant_shadow.service import PlantShadowService
from vvb001_monitor.plant_shadow.storage import EvidenceStore


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/rul_v2_7_full_cadence.joblib"
CONFIG = ROOT / "config/vvb001.json"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_core(path: Path) -> dict[str, list[tuple[object, ...]]]:
    db = sqlite3.connect(path)
    try:
        return {
            "raw": db.execute(
                "SELECT source_row_id,machine_uid,event_timestamp,vrms,arms,apeak,crest,temp,raw_json "
                "FROM raw_observations ORDER BY ingestion_id"
            ).fetchall(),
            "dispositions": db.execute(
                "SELECT r.source_row_id,d.disposition,d.validation_status,d.reasons_json "
                "FROM observation_dispositions d JOIN raw_observations r USING(ingestion_id) "
                "ORDER BY d.ingestion_id"
            ).fetchall(),
            "context": db.execute(
                "SELECT r.source_row_id,o.operating_state,o.operating_state_source,"
                "o.operating_state_confidence,o.admitted_to_runtime,o.reason_code,"
                "o.cumulative_operating_seconds,o.effective_operating_timestamp,o.maintenance_event_id "
                "FROM operating_context_decisions o JOIN raw_observations r USING(ingestion_id) "
                "ORDER BY o.ingestion_id"
            ).fetchall(),
            "predictions": db.execute(
                "SELECT r.source_row_id,p.health_state_model,p.health_state_manufacturer,"
                "p.health_state_source,p.forecast_state,p.warning_point_hours,p.warning_serviceable,"
                "p.warning_withhold_reason,p.critical_point_hours,p.critical_serviceable,"
                "p.critical_withhold_reason,p.model_version,p.model_artifact_sha256 "
                "FROM prediction_attempts p JOIN raw_observations r USING(ingestion_id) "
                "ORDER BY p.ingestion_id"
            ).fetchall(),
            "lifecycles": db.execute(
                "SELECT machine_uid,sequence_number,status,observed_start_timestamp,end_timestamp,"
                "left_censored,closure_reason,closure_confidence FROM lifecycle_records "
                "ORDER BY machine_uid,sequence_number"
            ).fetchall(),
            "scenarios": db.execute(
                "SELECT machine_id,scenario_name,description,metadata_json "
                "FROM demo_machine_scenarios ORDER BY machine_id,scenario_name"
            ).fetchall(),
        }
    finally:
        db.close()


@pytest.fixture(scope="module")
def demo_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("plant-shadow-demo")
    normal = root / "output/plant_shadow/plant_shadow.db"
    normal.parent.mkdir(parents=True)
    normal.write_bytes(b"normal-plant-shadow-sentinel")
    first = root / "output/plant_shadow_demo/recreated.db"
    snapshot = root / "output/plant_shadow_demo/first_snapshot.db"
    config = DemoConfig(machines=6, hours=4, seed=42, start_time=START, cadence_minutes=10)
    with (
        patch.dict(os.environ, {"VVB001_POSTGRES_DSN": ""}),
        patch(
            "vvb001_monitor.plant_shadow.commands.PlantPostgresSource",
            side_effect=AssertionError("demo invoked PostgreSQL command source"),
        ),
        patch(
            "vvb001_monitor.plant_shadow.source.PlantPostgresSource",
            side_effect=AssertionError("demo invoked PostgreSQL source"),
        ),
    ):
        generate_demo_database(
            first,
            config=config,
            app_config_path=CONFIG,
            model_path=MODEL,
            repository_root=root,
        )
        shutil.copyfile(first, snapshot)
        generate_demo_database(
            first,
            config=config,
            app_config_path=CONFIG,
            model_path=MODEL,
            repository_root=root,
        )
    return snapshot, first, normal


def test_demo_is_deterministic_and_does_not_touch_normal_database(demo_pair: tuple[Path, Path, Path]) -> None:
    first, second, normal = demo_pair
    assert _normalized_core(first) == _normalized_core(second)
    assert normal.read_bytes() == b"normal-plant-shadow-sentinel"
    assert _sha256(normal) == hashlib.sha256(b"normal-plant-shadow-sentinel").hexdigest()


def test_demo_refuses_the_normal_database_path(tmp_path: Path) -> None:
    normal = tmp_path / DEFAULT_DATABASE
    with pytest.raises(ValueError, match="refuses"):
        generate_demo_database(
            normal,
            config=DemoConfig(machines=6, hours=2, seed=1, start_time=START),
            app_config_path=CONFIG,
            model_path=MODEL,
            repository_root=tmp_path,
        )


def test_demo_requires_no_dsn_and_does_not_invoke_postgres(
    demo_pair: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _, _ = demo_pair
    monkeypatch.delenv("VVB001_POSTGRES_DSN", raising=False)
    with sqlite3.connect(first) as db:
        run = db.execute("SELECT metadata_json FROM demo_runs").fetchone()[0]
        health = db.execute("SELECT connection_state,message FROM source_health_events").fetchone()
    assert '"postgresql_accessed":false' in run
    assert health[0] == "OFFLINE_DEMO_COMPLETE"
    assert "PostgreSQL was not accessed" in health[1]


def _uid(machine_id: str) -> str:
    return f"local-demo-vvb001::DEMO_LINE::{machine_id}"


def test_running_idle_off_unknown_and_machine_isolation(demo_pair: tuple[Path, Path, Path]) -> None:
    first, _, _ = demo_pair
    db = sqlite3.connect(first)
    db.row_factory = sqlite3.Row
    try:
        healthy_latest = db.execute(
            "SELECT admitted_to_runtime,cumulative_operating_seconds FROM operating_context_decisions "
            "WHERE machine_uid=? ORDER BY ingestion_id DESC LIMIT 1",
            (_uid("DEMO_HEALTHY"),),
        ).fetchone()
        assert healthy_latest["admitted_to_runtime"] == 1
        assert healthy_latest["cumulative_operating_seconds"] > 0

        for machine_id, paused_state in (("DEMO_IDLE", "IDLE"), ("DEMO_OFF", "OFF")):
            rows = db.execute(
                "SELECT o.admitted_to_runtime,o.cumulative_operating_seconds,r.event_timestamp "
                "FROM operating_context_decisions o JOIN raw_observations r USING(ingestion_id) "
                "WHERE o.machine_uid=? AND o.operating_state=? ORDER BY o.ingestion_id",
                (_uid(machine_id), paused_state),
            ).fetchall()
            assert rows
            assert {row["admitted_to_runtime"] for row in rows} == {0}
            assert len({row["cumulative_operating_seconds"] for row in rows}) == 1
            latest = db.execute(
                "SELECT operating_state,admitted_to_runtime,cumulative_operating_seconds "
                "FROM operating_context_decisions WHERE machine_uid=? ORDER BY ingestion_id DESC LIMIT 1",
                (_uid(machine_id),),
            ).fetchone()
            assert latest["operating_state"] == "RUNNING"
            assert latest["admitted_to_runtime"] == 1
            assert latest["cumulative_operating_seconds"] > rows[-1]["cumulative_operating_seconds"]

        unknown = db.execute(
            "SELECT o.*,p.forecast_state,p.warning_withhold_reason FROM operating_context_decisions o "
            "JOIN prediction_attempts p USING(ingestion_id) WHERE o.machine_uid=? "
            "ORDER BY o.ingestion_id DESC LIMIT 1",
            (_uid("DEMO_UNKNOWN"),),
        ).fetchone()
        assert unknown["operating_state"] == "UNKNOWN"
        assert unknown["operating_state_source"] == "UNAVAILABLE"
        assert unknown["admitted_to_runtime"] == 0
        assert unknown["reason_code"] == "OPERATING_STATE_UNKNOWN"
        assert unknown["forecast_state"] == "PAUSED"
        assert unknown["warning_withhold_reason"] == "OPERATING_STATE_UNKNOWN"

        idle_times = db.execute(
            "SELECT MIN(r.event_timestamp),MAX(r.event_timestamp) FROM raw_observations r "
            "JOIN operating_context_decisions o USING(ingestion_id) "
            "WHERE o.machine_uid=? AND o.operating_state='IDLE'",
            (_uid("DEMO_IDLE"),),
        ).fetchone()
        healthy_during_idle = db.execute(
            "SELECT MIN(o.cumulative_operating_seconds),MAX(o.cumulative_operating_seconds) "
            "FROM operating_context_decisions o JOIN raw_observations r USING(ingestion_id) "
            "WHERE o.machine_uid=? AND r.event_timestamp BETWEEN ? AND ?",
            (_uid("DEMO_HEALTHY"), idle_times[0], idle_times[1]),
        ).fetchone()
        assert healthy_during_idle[1] > healthy_during_idle[0]
    finally:
        db.close()


def test_maintenance_replacement_and_sensor_quality_use_existing_rules(
    demo_pair: tuple[Path, Path, Path]
) -> None:
    first, _, _ = demo_pair
    db = sqlite3.connect(first)
    try:
        maintenance_context = db.execute(
            "SELECT COUNT(*),SUM(admitted_to_runtime) FROM operating_context_decisions "
            "WHERE machine_uid=? AND operating_state='MAINTENANCE'",
            (_uid("DEMO_MAINTENANCE"),),
        ).fetchone()
        assert maintenance_context[0] > 0
        assert maintenance_context[1] == 0
        lifecycles = db.execute(
            "SELECT sequence_number,status,closure_reason FROM lifecycle_records "
            "WHERE machine_uid=? ORDER BY sequence_number",
            (_uid("DEMO_MAINTENANCE"),),
        ).fetchall()
        assert lifecycles == [
            (1, "CLOSED_CONFIRMED", "COMPONENT_REPLACEMENT"),
            (2, "ACTIVE", None),
        ]
        assert db.execute(
            "SELECT COUNT(*) FROM endpoint_evidence WHERE machine_uid=?",
            (_uid("DEMO_MAINTENANCE"),),
        ).fetchone()[0] == 1

        statuses = {
            row[0]
            for row in db.execute(
                "SELECT json_extract(payload_json,'$.sensor_quality_status') "
                "FROM prediction_attempts WHERE machine_uid IN (?,?)",
                (_uid("DEMO_DEGRADING"), _uid("DEMO_SENSOR_FAULT")),
            )
        }
        assert statuses & {"SUSPECT_DROPOUT", "SUSPECT_STUCK"}
        assert db.execute(
            "SELECT COUNT(*) FROM raw_observations WHERE machine_uid=? "
            "AND json_extract(raw_json,'$.scenario_phase')='SENSOR_DROPOUT'",
            (_uid("DEMO_DEGRADING"),),
        ).fetchone()[0] > 0
    finally:
        db.close()


def test_idempotency_and_same_timestamp_ordering(demo_pair: tuple[Path, Path, Path]) -> None:
    first, _, _ = demo_pair
    with EvidenceStore(first) as store:
        original_count = store.db.execute("SELECT COUNT(*) FROM raw_observations").fetchone()[0]
        observation = next(store.committed_observations("local-demo-vvb001"))[1]

        class ExplodingRouter:
            def resolve_operating_context(self, *_: object) -> None:
                raise AssertionError("duplicate must not enter runtime")

            def process(self, *_: object) -> None:
                raise AssertionError("duplicate must not enter runtime")

        result = PlantShadowService(store, ExplodingRouter()).process(observation)  # type: ignore[arg-type]
        assert result.duplicate is True
        assert store.db.execute("SELECT COUNT(*) FROM raw_observations").fetchone()[0] == original_count
        late = store.db.execute(
            "SELECT COUNT(*) FROM observation_dispositions WHERE disposition='LATE_QUARANTINED'"
        ).fetchone()[0]
        same_timestamp_rows = store.db.execute(
            "SELECT COUNT(*) FROM raw_observations GROUP BY event_timestamp HAVING COUNT(*)>1"
        ).fetchall()
        watermark = store.db.execute(
            "SELECT CAST(source_row_id AS INTEGER) FROM source_watermarks WHERE source_key='local-demo-vvb001'"
        ).fetchone()[0]
        assert late == 0
        assert same_timestamp_rows
        assert watermark == original_count


def test_demo_api_exposes_real_evidence_and_environment(demo_pair: tuple[Path, Path, Path]) -> None:
    first, _, _ = demo_pair
    client = TestClient(create_app(first, model_path=MODEL))
    overview = client.get("/api/v1/overview")
    assert overview.status_code == 200
    assert overview.json()["environment_mode"] == "DEMO"
    assert overview.json()["machines"] == 6
    assert overview.json()["plant_production_authorized"] is False

    machines = client.get("/api/v1/machines").json()["items"]
    assert len(machines) == 6
    assert {item["operating_state"] for item in machines} >= {"RUNNING", "UNKNOWN"}
    healthy_uid = _uid("DEMO_HEALTHY")
    detail = client.get(f"/api/v1/machines/{healthy_uid}").json()
    assert detail["latest_prediction"]["model_version"] == "v2.7"
    assert detail["latest_operating_context"]["admitted_to_runtime"] == 1
    assert detail["demo_scenarios"]
    assert client.get(f"/api/v1/machines/{healthy_uid}/sensors").json()["items"]
    assert client.get(f"/api/v1/machines/{healthy_uid}/operating-context").json()["items"]
    assert client.get("/api/v1/lifecycles").json()["items"]
    assert client.get("/api/v1/model").json()["environment_mode"] == "DEMO"
    health = client.get("/api/v1/system/health").json()
    assert health["environment_mode"] == "DEMO"
    assert health["plant_production_authorized"] is False
    demo = client.get("/api/v1/demo").json()
    assert demo["demo_run"]["seed"] == 42
    assert demo["scenarios"]
    evaluation = client.get("/api/v1/plant-evaluation").json()
    assert evaluation["validation_domain"] == "local_demo_synthetic"
    assert evaluation["plant_validation_eligible"] is False
