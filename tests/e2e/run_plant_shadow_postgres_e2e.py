from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import psycopg
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo

from vvb001_monitor.config import AppConfig
from vvb001_monitor.plant_shadow.api import create_app
from vvb001_monitor.plant_shadow.contracts import (
    ActivationLevel,
    EndpointClass,
    EndpointPrecision,
    ObservationDisposition,
)
from vvb001_monitor.plant_shadow.evaluation import evaluate_plant
from vvb001_monitor.plant_shadow.golden import verify_golden_replay
from vvb001_monitor.plant_shadow.manifest import sha256_file, verify_manifest
from vvb001_monitor.plant_shadow.native_postgres import (
    dsn_for_database,
    find_native_postgres_binaries,
    temporary_native_postgres,
    validate_dedicated_database_name,
    validate_loopback_dsn,
)
from vvb001_monitor.plant_shadow.runtime import FrozenRuntimeRouter
from vvb001_monitor.plant_shadow.service import PlantShadowService
from vvb001_monitor.plant_shadow.source import PlantColumnMap, PlantPostgresConfig, PlantPostgresSource
from vvb001_monitor.plant_shadow.storage import EvidenceStore


MODEL = ROOT / "models/rul_v2_7_full_cadence.joblib"
MANIFEST = ROOT / "output/plant_shadow/plant_shadow_runtime_manifest.json"
GOLDEN = ROOT / "config/plant_shadow_golden_replay.json"
APP_CONFIG = ROOT / "config/vvb001.json"
REPORT = ROOT / "output/plant_shadow/postgres_e2e_report.json"
EXPECTED_MODEL_SHA256 = "ecad8f4f704129a3f0456c3c8dd47aabeb94ea6dfbdfd4c264313aea46076b9a"
DEFAULT_ADMIN_ENV = "VVB001_E2E_ADMIN_DSN"
READER_PASSWORD = "vvb001-e2e-reader-only"
CONNECT_TIMEOUT_SECONDS = 10
STATEMENT_TIMEOUT_MILLISECONDS = 30_000
HOST = "127.0.0.1"


def _stage(message: str) -> None:
    print(f"[E2E] {message}", flush=True)


def _connect(dsn: str, *, autocommit: bool = True):
    return psycopg.connect(
        dsn,
        autocommit=autocommit,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        options=(
            f"-c statement_timeout={STATEMENT_TIMEOUT_MILLISECONDS} "
            "-c lock_timeout=10000"
        ),
    )


def _create_database_and_reader(admin_dsn: str, database_name: str, reader_role: str) -> tuple[str, str]:
    validate_dedicated_database_name(database_name)
    validate_loopback_dsn(admin_dsn)
    database_created = False
    role_created = False
    with _connect(admin_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (database_name,))
            if cursor.fetchone() is not None:
                raise RuntimeError(
                    f"dedicated database {database_name!r} already exists; "
                    "choose another name or remove it intentionally"
                )
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (reader_role,))
            if cursor.fetchone() is not None:
                raise RuntimeError(f"dedicated reader role {reader_role!r} already exists")
            try:
                cursor.execute(
                    sql.SQL(
                        "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB "
                        "NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                    ).format(sql.Identifier(reader_role), sql.Literal(READER_PASSWORD))
                )
                role_created = True
                cursor.execute(
                    sql.SQL("ALTER ROLE {} SET default_transaction_read_only=on").format(
                        sql.Identifier(reader_role)
                    )
                )
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
                database_created = True
                cursor.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(database_name)))
                cursor.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                        sql.Identifier(database_name), sql.Identifier(reader_role)
                    )
                )
            except Exception:
                if database_created:
                    with suppress(Exception):
                        cursor.execute(
                            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                                sql.Identifier(database_name)
                            )
                        )
                if role_created:
                    with suppress(Exception):
                        cursor.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(reader_role)))
                raise
    fixture_dsn = dsn_for_database(admin_dsn, database_name)
    reader_dsn = dsn_for_database(admin_dsn, database_name, user=reader_role, password=READER_PASSWORD)
    return fixture_dsn, reader_dsn


def _drop_database_and_reader(admin_dsn: str, database_name: str, reader_role: str) -> None:
    validate_dedicated_database_name(database_name)
    validate_loopback_dsn(admin_dsn)
    with _connect(admin_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(database_name)))
            cursor.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(reader_role)))


@contextmanager
def _temporary_native_cluster(bin_dir: str | None, port: int) -> Iterator[str]:
    binaries = find_native_postgres_binaries(bin_dir)
    if binaries is None:
        raise RuntimeError(
            "No native PostgreSQL server binaries were found. Install PostgreSQL locally or set "
            f"{DEFAULT_ADMIN_ENV} to a loopback admin DSN for an existing local server."
        )
    _stage("Creating temporary PostgreSQL cluster...")
    with tempfile.TemporaryDirectory(prefix="vvb001-native-pg-") as temp:
        root = Path(temp)
        with temporary_native_postgres(
            binaries,
            root=root,
            host=HOST,
            port=port,
            admin_user="vvb001_e2e_admin",
            admin_password="vvb001-e2e-admin",
            progress=_stage,
        ) as (data, _log, owned_pid):
            dsn = f"postgresql://vvb001_e2e_admin:vvb001-e2e-admin@{HOST}:{port}/postgres"
            with _connect(dsn) as connection:
                actual_data = Path(connection.execute("SHOW data_directory").fetchone()[0]).resolve()
            if actual_data != data.resolve():
                raise RuntimeError(
                    "temporary PostgreSQL identity mismatch: connected server does not own the "
                    f"E2E data directory (owned PID {owned_pid})"
                )
            yield dsn


@contextmanager
def _fixture_database(args: argparse.Namespace) -> Iterator[tuple[str, str, str]]:
    supplied = os.getenv(args.admin_dsn_env, "").strip()
    cluster_context = None
    if supplied:
        validate_loopback_dsn(supplied)
        admin_dsn = supplied
        mode = "existing_loopback_server"
    else:
        cluster_context = _temporary_native_cluster(args.pg_bin_dir, args.port)
        admin_dsn = cluster_context.__enter__()
        mode = "disposable_native_cluster"
    database_name = validate_dedicated_database_name(args.database_name)
    reader_role = f"{database_name}_reader"
    if len(reader_role) > 63:
        raise ValueError("derived reader role exceeds PostgreSQL identifier length")
    created = False
    try:
        _stage("Creating dedicated E2E database and SELECT-only reader...")
        fixture_dsn, reader_dsn = _create_database_and_reader(admin_dsn, database_name, reader_role)
        created = True
        yield fixture_dsn, reader_dsn, mode
    finally:
        try:
            if created and not args.keep_database:
                _stage("Dropping dedicated E2E database and reader...")
                _drop_database_and_reader(admin_dsn, database_name, reader_role)
        finally:
            if cluster_context is not None:
                cluster_context.__exit__(*sys.exc_info())


def _provision(fixture_dsn: str) -> None:
    with _connect(fixture_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE SCHEMA source_a;
                CREATE SCHEMA source_b;
                CREATE TABLE source_a.vvb001_readings (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    line_sel TEXT NOT NULL,
                    machine_id TEXT NOT NULL,
                    vrms DOUBLE PRECISION NOT NULL,
                    arms DOUBLE PRECISION NOT NULL,
                    apeak DOUBLE PRECISION NOT NULL,
                    crest DOUBLE PRECISION NOT NULL,
                    temp DOUBLE PRECISION NOT NULL,
                    operating_state TEXT NOT NULL,
                    operating_state_source TEXT NOT NULL,
                    operating_state_confidence DOUBLE PRECISION NOT NULL,
                    maintenance_event_id TEXT
                );
                CREATE TABLE source_b.vvb001_readings
                    (LIKE source_a.vvb001_readings INCLUDING DEFAULTS);
                ALTER TABLE source_b.vvb001_readings ADD PRIMARY KEY (id);
                CREATE INDEX source_a_vvb001_order_idx
                    ON source_a.vvb001_readings(timestamp,id);
                CREATE INDEX source_b_vvb001_order_idx
                    ON source_b.vvb001_readings(timestamp,id);
                """
            )
            cursor.execute("SELECT rolname FROM pg_roles WHERE rolname=current_database() || '_reader'")
            reader_role = cursor.fetchone()[0]
            cursor.execute(
                sql.SQL("GRANT USAGE ON SCHEMA source_a,source_b TO {}").format(sql.Identifier(reader_role))
            )
            cursor.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")
            cursor.execute(
                sql.SQL("GRANT SELECT ON source_a.vvb001_readings,source_b.vvb001_readings TO {}").format(
                    sql.Identifier(reader_role)
                )
            )
            cursor.execute(
                sql.SQL(
                    "REVOKE INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER "
                    "ON source_a.vvb001_readings,source_b.vvb001_readings FROM {}"
                ).format(sql.Identifier(reader_role))
            )
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    with _connect(fixture_dsn) as connection:
        with connection.cursor() as cursor:
            for schema_name, degrading in (("source_a", True), ("source_b", False)):
                insert = sql.SQL(
                    "INSERT INTO {}.vvb001_readings "
                    "(id,timestamp,line_sel,machine_id,vrms,arms,apeak,crest,temp,"
                    "operating_state,operating_state_source,operating_state_confidence,maintenance_event_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                ).format(sql.Identifier(schema_name))
                for index in range(36):
                    progress = index / 35.0
                    # Include the integer 9 -> 10 boundary plus recurring equal timestamps so
                    # PostgreSQL and the SQLite watermark must agree on typed ID tie-breaking.
                    if index == 9:
                        minute = 40
                    else:
                        minute = index * 5 if index % 6 else max(0, (index - 1) * 5)
                    arms = 4.8 + ((2.0 * progress) if degrading else (0.1 * progress))
                    crest = 2.55 + ((0.35 * progress) if degrading else 0.02)
                    cursor.execute(
                        insert,
                        (
                            index + 1, start + timedelta(minutes=minute), "LINE_SHARED", "MACHINE_01",
                            2.1 + ((3.2 * progress) if degrading else (0.15 * progress)),
                            arms, arms * crest, crest,
                            31.0 + ((8.0 * progress) if degrading else (0.5 * progress)),
                            "RUNNING", "SYNTHETIC_FIXTURE", 1.0, None,
                        ),
                    )


def _config(source_key: str, schema_name: str, env_name: str) -> PlantPostgresConfig:
    return PlantPostgresConfig(
        source_key=source_key,
        display_name=f"Native fixture {source_key}",
        dsn_env=env_name,
        schema=schema_name,
        columns=PlantColumnMap(
            operating_state="operating_state",
            operating_state_source="operating_state_source",
            operating_state_confidence="operating_state_confidence",
            maintenance_event_id="maintenance_event_id",
        ),
        activation_level=ActivationLevel.SHADOW_MONITORING,
        row_id_kind="integer",
        batch_size=1000,
        poll_seconds=0,
        lateness_seconds=600,
    )


def _watermark(store: EvidenceStore, source_key: str) -> tuple[datetime, str] | None:
    row = store.db.execute(
        "SELECT event_timestamp,source_row_id FROM source_watermarks WHERE source_key=?", (source_key,)
    ).fetchone()
    return (datetime.fromisoformat(str(row[0])), str(row[1])) if row else None


def _ingest(config: PlantPostgresConfig, store: EvidenceStore, service: PlantShadowService) -> dict[str, Any]:
    started = time.perf_counter()
    with PlantPostgresSource(config) as source:
        validation = source.validate_schema()
        observations = source.fetch_after(_watermark(store, config.source_key))
        bootstrap_window = source.bootstrap_window
    results = [service.process(observation) for observation in observations]
    elapsed = time.perf_counter() - started
    return {
        "source_key": config.source_key,
        "schema_validation": validation,
        "bootstrap_window": bootstrap_window,
        "fetched": len(observations),
        "committed": sum(not item.duplicate for item in results),
        "duplicates": sum(item.duplicate for item in results),
        "late_quarantined": sum(item.disposition == ObservationDisposition.LATE_QUARANTINED for item in results),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(observations) / elapsed if elapsed else None,
    }


def _assert_reader_is_select_only(reader_dsn: str) -> dict[str, str]:
    attempts = {
        "fixture_table_insert": (
            "INSERT INTO source_a.vvb001_readings "
            "(id,timestamp,line_sel,machine_id,vrms,arms,apeak,crest,temp) "
            "VALUES (999999,NOW(),'L','M',1,1,2,2,30)"
        ),
        "temporary_table_create": "CREATE TEMP TABLE forbidden_temp(value INTEGER)",
        "public_table_create": "CREATE TABLE public.forbidden_table(value INTEGER)",
    }
    rejections: dict[str, str] = {}
    with _connect(reader_dsn) as connection:
        with connection.cursor() as cursor:
            # Prove the grants are sufficient on their own; the production adapter additionally
            # enforces a read-only session and verifies it after connecting.
            cursor.execute("SET default_transaction_read_only=off")
            for name, statement in attempts.items():
                try:
                    cursor.execute(statement)
                except psycopg.errors.InsufficientPrivilege as exc:
                    rejections[name] = type(exc).__name__
                else:
                    raise AssertionError(f"production adapter credential unexpectedly passed {name}")
    assert set(rejections) == set(attempts)
    return rejections


def _insert_post_restart_rows(fixture_dsn: str) -> None:
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    with _connect(fixture_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO source_a.vvb001_readings VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (37, start + timedelta(minutes=180), "LINE_SHARED", "MACHINE_01", 0.4, 0.5, 1.4, 2.8, 35.0,
                 "OFF", "PLC", 1.0, None),
            )
            cursor.execute(
                "INSERT INTO source_a.vvb001_readings VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (38, start + timedelta(minutes=185), "LINE_SHARED", "MACHINE_01", 14.0, 25.0, 85.0, 3.4, 39.0,
                 "MAINTENANCE", "CMMS", 1.0, "work-order-e2e-001"),
            )
            cursor.execute(
                "INSERT INTO source_a.vvb001_readings VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (39, start + timedelta(minutes=190), "LINE_SHARED", "MACHINE_01", 5.5, 7.0, 20.3, 2.9, 40.0,
                 "RUNNING", "PLC", 1.0, None),
            )
            cursor.execute(
                "INSERT INTO source_a.vvb001_readings VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (40, start + timedelta(minutes=195), "LINE_SHARED", "MACHINE_01", 5.55, 7.05, 20.5, 20.5 / 7.05, 40.1,
                 "RUNNING", "PLC", 1.0, None),
            )
            cursor.execute(
                "INSERT INTO source_b.vvb001_readings VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (37, start + timedelta(hours=28), "LINE_SHARED", "MACHINE_01", 2.3, 4.9, 12.6, 12.6 / 4.9, 31.7,
                 "RUNNING", "PLC", 1.0, None),
            )


def _insert_late_row(fixture_dsn: str) -> None:
    """Insert inside the lateness lookback only after the committed watermark reaches minute 195."""
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    with _connect(fixture_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO source_a.vvb001_readings VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    1001, start + timedelta(minutes=190), "LINE_SHARED", "MACHINE_01",
                    5.4, 6.9, 19.8, 19.8 / 6.9, 39.8, "RUNNING", "PLC", 1.0, None,
                ),
            )


def _exercise_source_failure_isolation(
    fixture_dsn: str,
    configs: tuple[PlantPostgresConfig, PlantPostgresConfig],
    store: EvidenceStore,
    service: PlantShadowService,
) -> dict[str, Any]:
    with _connect(fixture_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE source_a.vvb001_readings RENAME TO unavailable_readings")
            cursor.execute(
                "INSERT INTO source_b.vvb001_readings VALUES "
                "(%s,NOW(),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (38, "LINE_SHARED", "MACHINE_01", 2.35, 4.92, 12.7, 12.7 / 4.92, 31.8,
                 "RUNNING", "PLC", 1.0, None),
            )
    error = None
    try:
        _ingest(configs[0], store, service)
    except Exception as exc:
        error = type(exc).__name__
    finally:
        with _connect(fixture_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("ALTER TABLE source_a.unavailable_readings RENAME TO vvb001_readings")
    if error is None:
        raise AssertionError("source A schema failure was not detected")
    source_b = _ingest(configs[1], store, service)
    assert source_b["committed"] >= 1
    return {"source_a_error": error, "source_b_continued": source_b}


def _exercise_runtime_rebuild(
    fixture_dsn: str,
    config: PlantPostgresConfig,
    store: EvidenceStore,
    service: PlantShadowService,
    router: FrozenRuntimeRouter,
) -> dict[str, Any]:
    with _connect(fixture_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO source_a.vvb001_readings VALUES "
                "(%s,NOW() + INTERVAL '1 minute',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (2001, "LINE_SHARED", "MACHINE_01", 5.6, 7.1, 20.6, 20.6 / 7.1, 40.2,
                 "RUNNING", "PLC", 1.0, None),
            )
    with PlantPostgresSource(config) as source:
        candidates = source.fetch_after(_watermark(store, config.source_key))
    observation = next(item for item in candidates if item.source_row_id == "2001")
    original_audit = store._audit
    rebuild_count = 0
    original_rebuild = router.rebuild_source

    def counted_rebuild(source_key, observations):
        nonlocal rebuild_count
        rebuild_count += 1
        return original_rebuild(source_key, observations)

    def fail_audit(*_args, **_kwargs):
        raise RuntimeError("injected local commit failure")

    router.rebuild_source = counted_rebuild  # type: ignore[method-assign]
    store._audit = fail_audit  # type: ignore[method-assign]
    try:
        try:
            service.process(observation)
        except RuntimeError as exc:
            assert "injected local commit failure" in str(exc)
        else:
            raise AssertionError("injected commit failure did not fail")
    finally:
        store._audit = original_audit  # type: ignore[method-assign]
        router.rebuild_source = original_rebuild  # type: ignore[method-assign]
    assert rebuild_count == 1
    assert store.db.execute(
        "SELECT COUNT(*) FROM raw_observations WHERE source_key=? AND source_row_id='2001'",
        (config.source_key,),
    ).fetchone()[0] == 0
    retry = service.process(observation)
    assert retry.duplicate is False
    return {"rebuild_count": rebuild_count, "retry_disposition": retry.disposition}


def _endpoint_semantics(store: EvidenceStore) -> dict[str, Any]:
    machine_a = "e2e-source-a::LINE_SHARED::MACHINE_01"
    machine_b = "e2e-source-b::LINE_SHARED::MACHINE_01"
    lifecycle_a = store.db.execute(
        "SELECT lifecycle_id FROM lifecycle_records WHERE machine_uid=? AND status='ACTIVE'", (machine_a,)
    ).fetchone()[0]
    lifecycle_b = store.db.execute(
        "SELECT lifecycle_id FROM lifecycle_records WHERE machine_uid=? AND status='ACTIVE'", (machine_b,)
    ).fetchone()[0]
    last_a = datetime.fromisoformat(store.db.execute(
        "SELECT event_timestamp FROM source_watermarks WHERE source_key='e2e-source-a'"
    ).fetchone()[0])
    last_b = datetime.fromisoformat(store.db.execute(
        "SELECT event_timestamp FROM source_watermarks WHERE source_key='e2e-source-b'"
    ).fetchone()[0])

    def evidence(external_id: str, machine: str, lifecycle: str, at: datetime) -> str:
        return store.add_endpoint_evidence(
            source_system="native-e2e-independent",
            external_event_id=external_id,
            machine_uid=machine,
            lifecycle_id=lifecycle,
            precision=EndpointPrecision.EXACT_TIMESTAMP,
            event_time_lower=at,
            event_time_upper=at,
            details={"fixture": True},
            actor="native-e2e-harness",
        )

    warning_result = store.confirm_endpoint_classification(
        evidence("warning-a", machine_a, lifecycle_a, last_a + timedelta(hours=2)),
        EndpointClass.WARNING_ONSET,
        actor="native-e2e-harness",
    )
    maintenance_result = store.confirm_endpoint_classification(
        evidence("maintenance-a", machine_a, lifecycle_a, last_a + timedelta(hours=4)),
        EndpointClass.PREVENTIVE_MAINTENANCE,
        actor="native-e2e-harness",
    )
    critical_result = store.confirm_endpoint_classification(
        evidence("critical-b", machine_b, lifecycle_b, last_b + timedelta(hours=3)),
        EndpointClass.CRITICAL_ONSET,
        actor="native-e2e-harness",
    )
    assert store.db.execute(
        "SELECT COUNT(*) FROM target_truth WHERE target='CRITICAL' AND prediction_id IN "
        "(SELECT prediction_id FROM prediction_attempts WHERE lifecycle_id=?)", (lifecycle_a,)
    ).fetchone()[0] == 0
    assert store.db.execute(
        "SELECT COUNT(*) FROM target_censoring c JOIN target_truth_eligibility e "
        "ON e.eligibility_id=c.eligibility_id WHERE e.lifecycle_id=? AND e.target='CRITICAL' "
        "AND c.censoring='RIGHT_CENSORED'", (lifecycle_a,)
    ).fetchone()[0] == 1
    assert critical_result["truth_rows_created"] > 0
    return {"warning_a": warning_result, "maintenance_a": maintenance_result, "critical_b": critical_result}


def _api_checks(database: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with TestClient(create_app(database, manifest_path=MANIFEST, model_path=MODEL)) as client:
        for path in ("/api/v1/overview", "/api/v1/machines", "/api/v1/lifecycles", "/api/v1/plant-evaluation", "/api/v1/system/health"):
            started = time.perf_counter()
            response = client.get(path)
            latency_ms = (time.perf_counter() - started) * 1000.0
            assert response.status_code == 200, (path, response.status_code, response.text)
            result[path] = {"latency_ms": latency_ms, "payload": response.json()}
    assert result["/api/v1/overview"]["payload"]["machines"] == 2
    assert result["/api/v1/overview"]["payload"]["plant_production_authorized"] is False
    return result


def _scenario(fixture_dsn: str, reader_dsn: str, mode: str, database: Path) -> dict[str, Any]:
    _stage("Verifying frozen runtime identity and read-only reader...")
    manifest = verify_manifest(ROOT, MANIFEST)
    sensor = AppConfig.load(APP_CONFIG).sensor
    report: dict[str, Any] = {
        "status": "RUNNING",
        "fixture_mode": mode,
        "plant_source_used": False,
        "model_sha256": sha256_file(MODEL),
        "golden_replay": verify_golden_replay(GOLDEN, MODEL, sensor),
        "manifest_deployment_id": manifest["deployment_id"],
        "reader_write_rejection": _assert_reader_is_select_only(reader_dsn),
    }
    assert report["model_sha256"] == EXPECTED_MODEL_SHA256
    os.environ["VVB001_E2E_SOURCE_A_DSN"] = reader_dsn
    os.environ["VVB001_E2E_SOURCE_B_DSN"] = reader_dsn
    configs = (
        _config("e2e-source-a", "source_a", "VVB001_E2E_SOURCE_A_DSN"),
        _config("e2e-source-b", "source_b", "VVB001_E2E_SOURCE_B_DSN"),
    )
    persisted_watermarks: dict[str, tuple[str, str]] = {}
    _stage("Running initial two-source plant-shadow ingestion...")
    with EvidenceStore(database) as store:
        for config in configs:
            store.save_source_config(config.source_key, config.redacted_dict(), actor="native-e2e", enabled=True)
        router = FrozenRuntimeRouter(
            MODEL, sensor, deployment_id=str(manifest["deployment_id"]), runtime_manifest_id=str(manifest["deployment_id"])
        )
        service = PlantShadowService(store, router)
        report["initial_ingestion"] = [_ingest(config, store, service) for config in configs]
        assert all(
            item["bootstrap_window"] is not None
            and item["bootstrap_window"]["mode"] == "SOURCE_TAIL"
            and item["bootstrap_window"]["lookback_hours"] == 24.0
            for item in report["initial_ingestion"]
        )
        assert router.active_sources == ("e2e-source-a", "e2e-source-b")
        assert store.overview()["machines"] == 2
        assert store.overview()["predictions"] == 72
        assert {row["machine_uid"] for row in store.rows("SELECT machine_uid FROM machine_registry")} == {
            "e2e-source-a::LINE_SHARED::MACHINE_01", "e2e-source-b::LINE_SHARED::MACHINE_01"
        }
        # Each ingestion opens and closes the exact production adapter, so this retry also proves
        # clean disconnect/reconnect behavior while SQLite watermarks remain authoritative.
        report["retry_idempotency_and_reconnect"] = [_ingest(config, store, service) for config in configs]
        assert sum(item["duplicates"] for item in report["retry_idempotency_and_reconnect"]) > 0
        assert store.overview()["predictions"] == 72
        persisted_watermarks = {
            config.source_key: tuple(str(value) for value in _watermark(store, config.source_key))
            for config in configs
        }

    _stage("Validating restart, watermark recovery, operating context, and runtime rebuild...")
    # A new SQLite connection and new runtime router emulate a full worker process restart.
    with EvidenceStore(database) as store:
        recovered_watermarks = {
            config.source_key: tuple(str(value) for value in _watermark(store, config.source_key))
            for config in configs
        }
        assert recovered_watermarks == persisted_watermarks
        report["sqlite_watermark_recovery"] = {
            "before_close": persisted_watermarks,
            "after_reopen": recovered_watermarks,
        }
        rebuild_started = time.perf_counter()
        restarted_router = FrozenRuntimeRouter(
            MODEL, sensor, deployment_id=str(manifest["deployment_id"]), runtime_manifest_id=str(manifest["deployment_id"])
        )
        for config in configs:
            restarted_router.rebuild_source(config.source_key, store.committed_observations(config.source_key))
        report["restart_rebuild_seconds"] = time.perf_counter() - rebuild_started
        restarted_service = PlantShadowService(store, restarted_router)
        _insert_post_restart_rows(fixture_dsn)
        report["post_restart_ingestion"] = [_ingest(config, store, restarted_service) for config in configs]
        _insert_late_row(fixture_dsn)
        report["bounded_late_ingestion"] = _ingest(configs[0], store, restarted_service)
        operating_rows = store.rows(
            "SELECT r.source_row_id,o.operating_state,o.admitted_to_runtime,"
            "o.cumulative_operating_seconds,p.forecast_state,p.warning_point_hours,p.critical_point_hours "
            "FROM raw_observations r JOIN operating_context_decisions o ON o.ingestion_id=r.ingestion_id "
            "LEFT JOIN prediction_attempts p ON p.ingestion_id=r.ingestion_id "
            "WHERE r.source_key='e2e-source-a' AND r.source_row_id IN ('37','38','39','40') "
            "ORDER BY r.ingestion_id"
        )
        report["operating_context_gate"] = operating_rows
        assert [row["operating_state"] for row in operating_rows] == [
            "OFF", "MAINTENANCE", "RUNNING", "RUNNING"
        ]
        assert [row["admitted_to_runtime"] for row in operating_rows] == [0, 0, 0, 1]
        assert all(
            row["warning_point_hours"] is None and row["critical_point_hours"] is None
            for row in operating_rows[:3]
        )
        assert operating_rows[0]["cumulative_operating_seconds"] == operating_rows[1]["cumulative_operating_seconds"]
        assert operating_rows[1]["cumulative_operating_seconds"] == operating_rows[2]["cumulative_operating_seconds"]
        assert operating_rows[3]["cumulative_operating_seconds"] - operating_rows[2]["cumulative_operating_seconds"] == 300.0
        assert store.db.execute(
            "SELECT COUNT(*) FROM observation_dispositions WHERE disposition='LATE_QUARANTINED'"
        ).fetchone()[0] == 1
        assert store.db.execute(
            "SELECT COUNT(*) FROM lifecycle_events WHERE event_type='MODEL_STATE_RESET_GAP'"
        ).fetchone()[0] == 1
        assert store.db.execute("SELECT COUNT(*) FROM lifecycle_records WHERE status='ACTIVE'").fetchone()[0] == 2

        report["source_failure_isolation"] = _exercise_source_failure_isolation(
            fixture_dsn, configs, store, restarted_service
        )
        report["runtime_commit_rebuild"] = _exercise_runtime_rebuild(
            fixture_dsn, configs[0], store, restarted_service, restarted_router
        )
        report["endpoint_semantics"] = _endpoint_semantics(store)
        report["plant_evaluation"] = evaluate_plant(store.db)
        assert report["plant_evaluation"]["warning"]["status"] == "INSUFFICIENT_EVIDENCE"
        assert report["plant_evaluation"]["critical"]["status"] == "INSUFFICIENT_EVIDENCE"
        report["sqlite_bytes"] = database.stat().st_size
    _stage("Validating lifecycle truth, SQLite evidence, and FastAPI results...")
    report["api"] = _api_checks(database)
    report["final_counts"] = report["api"]["/api/v1/overview"]["payload"]
    report["status"] = "PASS"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run native, disposable, two-source PostgreSQL plant-shadow E2E")
    parser.add_argument("--admin-dsn-env", default=DEFAULT_ADMIN_ENV)
    parser.add_argument("--pg-bin-dir", help="Native PostgreSQL bin directory for disposable initdb mode")
    parser.add_argument("--port", type=int, default=55432, help="Non-default port for disposable initdb mode")
    parser.add_argument(
        "--database-name",
        default=f"vvb001_e2e_{int(time.time())}_{os.getpid()}",
        help="Dedicated database; must begin vvb001_e2e_",
    )
    parser.add_argument("--keep-database", action="store_true", help="Keep an existing-server fixture for inspection")
    args = parser.parse_args()
    validate_dedicated_database_name(args.database_name)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with _fixture_database(args) as (fixture_dsn, reader_dsn, mode):
        _stage("Loading fixture schema and two overlapping source streams...")
        _provision(fixture_dsn)
        with tempfile.TemporaryDirectory(prefix="vvb001-plant-shadow-e2e-") as temp:
            _stage("Starting complete PostgreSQL to plant-shadow to SQLite scenario...")
            report = _scenario(fixture_dsn, reader_dsn, mode, Path(temp) / "plant_shadow.db")
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"Native PostgreSQL E2E report: {REPORT}")
    _stage("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
