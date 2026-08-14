from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .contracts import SCHEMA_VERSION
from .evaluation import evaluate_plant
from .manifest import sha256_file


def _readonly_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    db = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, timeout=10.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=10000")
    return db


def _decode_rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        for key in tuple(value):
            if key.endswith("_json") and isinstance(value[key], str):
                try:
                    value[key[:-5]] = json.loads(value.pop(key))
                except json.JSONDecodeError:
                    pass
        result.append(value)
    return result


def create_app(
    database: str | Path,
    *,
    manifest_path: str | Path | None = None,
    model_path: str | Path = "models/rul_v2_7_full_cadence.joblib",
) -> FastAPI:
    database_path = Path(database)
    manifest_file = Path(manifest_path) if manifest_path else None
    model_file = Path(model_path)
    app = FastAPI(
        title="VVB001 Plant Shadow API",
        version="1.0.0",
        description="Read-only localhost API. Shadow evidence does not authorize plant production.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Accept", "Content-Type"],
    )

    @contextmanager
    def connect() -> Iterator[sqlite3.Connection]:
        try:
            db = _readonly_connection(database_path)
        except FileNotFoundError as exc:
            raise HTTPException(503, "plant shadow database is not initialized") from exc
        try:
            yield db
        finally:
            db.close()

    @app.get("/api/v1/overview")
    def overview() -> dict[str, Any]:
        with connect() as db:
            return {
                "sources": db.execute("SELECT COUNT(DISTINCT source_key) FROM machine_registry").fetchone()[0],
                "machines": db.execute("SELECT COUNT(*) FROM machine_registry").fetchone()[0],
                "active_lifecycles": db.execute("SELECT COUNT(*) FROM lifecycle_records WHERE status='ACTIVE'").fetchone()[0],
                "predictions": db.execute("SELECT COUNT(*) FROM prediction_attempts").fetchone()[0],
                "forecasts_available": db.execute("SELECT COUNT(*) FROM prediction_attempts WHERE forecast_state='AVAILABLE'").fetchone()[0],
                "forecasts_withheld": db.execute("SELECT COUNT(*) FROM prediction_attempts WHERE forecast_state!='AVAILABLE'").fetchone()[0],
                "machines_running": db.execute(
                    "SELECT COUNT(*) FROM machine_registry m WHERE ("
                    "SELECT operating_state FROM operating_context_decisions o "
                    "WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1"
                    ")='RUNNING'"
                ).fetchone()[0],
                "machines_paused": db.execute(
                    "SELECT COUNT(*) FROM machine_registry m WHERE COALESCE(("
                    "SELECT operating_state FROM operating_context_decisions o "
                    "WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1"
                    "),'UNKNOWN')!='RUNNING'"
                ).fetchone()[0],
                "machines_vibration_calibrating": db.execute(
                    "SELECT COUNT(*) FROM machine_registry m WHERE COALESCE(("
                    "SELECT calibration_state FROM vibration_operating_inferences v "
                    "WHERE v.machine_uid=m.machine_uid ORDER BY v.ingestion_id DESC LIMIT 1"
                    "),'CALIBRATING')='CALIBRATING'"
                ).fetchone()[0],
                "machines_vibration_inferred_running": db.execute(
                    "SELECT COUNT(*) FROM machine_registry m WHERE COALESCE(("
                    "SELECT classification FROM vibration_operating_inferences v "
                    "WHERE v.machine_uid=m.machine_uid ORDER BY v.ingestion_id DESC LIMIT 1"
                    "),'')='RUNNING_CONFIRMED'"
                ).fetchone()[0],
                "active_alerts": db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0],
                "validation_domain": "plant_shadow",
                "plant_production_authorized": False,
            }

    @app.get("/api/v1/machines")
    def machines(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                """
                SELECT m.*,
                    (SELECT health_state_model FROM prediction_attempts p WHERE p.machine_uid=m.machine_uid ORDER BY p.prediction_timestamp DESC LIMIT 1) AS health_state_model,
                    (SELECT health_state_manufacturer FROM prediction_attempts p WHERE p.machine_uid=m.machine_uid ORDER BY p.prediction_timestamp DESC LIMIT 1) AS health_state_manufacturer,
                    (SELECT forecast_state FROM prediction_attempts p WHERE p.machine_uid=m.machine_uid ORDER BY p.prediction_timestamp DESC LIMIT 1) AS forecast_state,
                    (SELECT warning_point_hours FROM prediction_attempts p WHERE p.machine_uid=m.machine_uid ORDER BY p.prediction_timestamp DESC LIMIT 1) AS warning_point_hours,
                    (SELECT critical_point_hours FROM prediction_attempts p WHERE p.machine_uid=m.machine_uid ORDER BY p.prediction_timestamp DESC LIMIT 1) AS critical_point_hours
                    ,(SELECT operating_state FROM operating_context_decisions o WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1) AS operating_state
                    ,(SELECT operating_state_source FROM operating_context_decisions o WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1) AS operating_state_source
                    ,(SELECT operating_state_confidence FROM operating_context_decisions o WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1) AS operating_state_confidence
                    ,(SELECT admitted_to_runtime FROM operating_context_decisions o WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1) AS admitted_to_runtime
                    ,(SELECT cumulative_operating_seconds / 3600.0 FROM operating_context_decisions o WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1) AS cumulative_operating_hours
                    ,(SELECT classification FROM vibration_operating_inferences v WHERE v.machine_uid=m.machine_uid ORDER BY v.ingestion_id DESC LIMIT 1) AS vibration_classification
                    ,(SELECT calibration_state FROM vibration_operating_inferences v WHERE v.machine_uid=m.machine_uid ORDER BY v.ingestion_id DESC LIMIT 1) AS vibration_calibration_state
                    ,(SELECT confidence FROM vibration_operating_inferences v WHERE v.machine_uid=m.machine_uid ORDER BY v.ingestion_id DESC LIMIT 1) AS vibration_confidence
                FROM machine_registry m ORDER BY m.machine_uid LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            return {"items": _decode_rows(rows), "limit": limit, "offset": offset}

    @app.get("/api/v1/machines/{machine_uid}")
    def machine(machine_uid: str) -> dict[str, Any]:
        with connect() as db:
            row = db.execute("SELECT * FROM machine_registry WHERE machine_uid=?", (machine_uid,)).fetchone()
            if row is None:
                raise HTTPException(404, "machine not found")
            latest = db.execute(
                "SELECT * FROM prediction_attempts WHERE machine_uid=? ORDER BY prediction_timestamp DESC LIMIT 1",
                (machine_uid,),
            ).fetchone()
            lifecycle = db.execute(
                "SELECT * FROM lifecycle_records WHERE machine_uid=? ORDER BY sequence_number DESC LIMIT 1",
                (machine_uid,),
            ).fetchone()
            operating_context = db.execute(
                "SELECT * FROM operating_context_decisions WHERE machine_uid=? "
                "ORDER BY ingestion_id DESC LIMIT 1",
                (machine_uid,),
            ).fetchone()
            vibration_inference = db.execute(
                "SELECT * FROM vibration_operating_inferences WHERE machine_uid=? "
                "ORDER BY ingestion_id DESC LIMIT 1",
                (machine_uid,),
            ).fetchone()
            return {
                "machine": _decode_rows([row])[0],
                "latest_prediction": _decode_rows([latest])[0] if latest else None,
                "current_lifecycle": _decode_rows([lifecycle])[0] if lifecycle else None,
                "latest_operating_context": (
                    _decode_rows([operating_context])[0] if operating_context else None
                ),
                "latest_vibration_operating_inference": (
                    _decode_rows([vibration_inference])[0] if vibration_inference else None
                ),
            }

    @app.get("/api/v1/machines/{machine_uid}/sensors")
    def sensors(
        machine_uid: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = Query(2000, ge=1, le=10000),
    ) -> dict[str, Any]:
        clauses = ["r.machine_uid=?"]
        params: list[Any] = [machine_uid]
        if start:
            clauses.append("r.event_timestamp>=?")
            params.append(start)
        if end:
            clauses.append("r.event_timestamp<=?")
            params.append(end)
        params.append(limit)
        with connect() as db:
            rows = db.execute(
                "SELECT r.ingestion_id,r.event_timestamp,r.observed_at,r.vrms,r.arms,r.apeak,"
                "r.crest,r.temp,o.operating_state,o.operating_state_source,"
                "o.operating_state_confidence,o.admitted_to_runtime,"
                "o.cumulative_operating_seconds/3600.0 AS cumulative_operating_hours,"
                "o.reason_code AS operating_context_reason,v.classification AS vibration_classification,"
                "v.confidence AS vibration_confidence,v.calibration_state AS vibration_calibration_state,"
                "v.reason_code AS vibration_reason "
                "FROM raw_observations r JOIN operating_context_decisions o "
                "ON o.ingestion_id=r.ingestion_id LEFT JOIN vibration_operating_inferences v "
                "ON v.ingestion_id=r.ingestion_id WHERE " + " AND ".join(clauses)
                + " ORDER BY r.event_timestamp LIMIT ?",
                tuple(params),
            ).fetchall()
            return {"items": _decode_rows(rows), "limit": limit, "downsampled": False}

    @app.get("/api/v1/machines/{machine_uid}/forecasts")
    def forecasts(machine_uid: str, limit: int = Query(1000, ge=1, le=5000)) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                "SELECT p.*,r.event_timestamp,r.observed_at FROM prediction_attempts p "
                "JOIN raw_observations r ON r.ingestion_id=p.ingestion_id "
                "WHERE p.machine_uid=? ORDER BY r.event_timestamp DESC LIMIT ?",
                (machine_uid, limit),
            ).fetchall()
            return {"items": _decode_rows(rows), "limit": limit}

    @app.get("/api/v1/machines/{machine_uid}/operating-context")
    def operating_context(machine_uid: str, limit: int = Query(1000, ge=1, le=5000)) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                "SELECT o.*,r.event_timestamp,r.observed_at FROM operating_context_decisions o "
                "JOIN raw_observations r ON r.ingestion_id=o.ingestion_id "
                "WHERE o.machine_uid=? ORDER BY r.event_timestamp DESC LIMIT ?",
                (machine_uid, limit),
            ).fetchall()
            return {
                "items": _decode_rows(rows),
                "limit": limit,
                "rul_time_basis": "OPERATING_HOURS",
            }

    @app.get("/api/v1/machines/{machine_uid}/vibration-operating")
    def vibration_operating(machine_uid: str, limit: int = Query(1000, ge=1, le=5000)) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                "SELECT v.*,r.event_timestamp,r.observed_at FROM vibration_operating_inferences v "
                "JOIN raw_observations r ON r.ingestion_id=v.ingestion_id "
                "WHERE v.machine_uid=? ORDER BY r.event_timestamp DESC LIMIT ?",
                (machine_uid, limit),
            ).fetchall()
            return {
                "items": _decode_rows(rows),
                "limit": limit,
                "policy_version": "vibration_operating_policy_v1",
                "warning": "Vibration inference does not identify maintenance or exact machine mode.",
            }

    @app.get("/api/v1/vibration-operating")
    def vibration_operating_latest(limit: int = Query(500, ge=1, le=5000)) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                "SELECT v.*,r.event_timestamp FROM vibration_operating_inferences v "
                "JOIN raw_observations r ON r.ingestion_id=v.ingestion_id "
                "WHERE v.ingestion_id IN (SELECT MAX(v2.ingestion_id) "
                "FROM vibration_operating_inferences v2 GROUP BY v2.machine_uid) "
                "ORDER BY v.machine_uid LIMIT ?",
                (limit,),
            ).fetchall()
            return {"items": _decode_rows(rows), "limit": limit}

    @app.get("/api/v1/machines/{machine_uid}/events")
    def machine_events(machine_uid: str, limit: int = Query(500, ge=1, le=5000)) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                "SELECT * FROM lifecycle_events WHERE machine_uid=? ORDER BY event_timestamp DESC LIMIT ?",
                (machine_uid, limit),
            ).fetchall()
            return {"items": _decode_rows(rows), "limit": limit}

    @app.get("/api/v1/lifecycles")
    def lifecycles(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                "SELECT * FROM lifecycle_records ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return {"items": _decode_rows(rows), "limit": limit, "offset": offset}

    @app.get("/api/v1/lifecycles/{lifecycle_id}")
    def lifecycle(lifecycle_id: str) -> dict[str, Any]:
        with connect() as db:
            row = db.execute("SELECT * FROM lifecycle_records WHERE lifecycle_id=?", (lifecycle_id,)).fetchone()
            if row is None:
                raise HTTPException(404, "lifecycle not found")
            events = db.execute(
                "SELECT * FROM lifecycle_events WHERE lifecycle_id=? ORDER BY event_timestamp",
                (lifecycle_id,),
            ).fetchall()
            eligibility = db.execute(
                "SELECT * FROM target_truth_eligibility WHERE lifecycle_id=? ORDER BY decided_at",
                (lifecycle_id,),
            ).fetchall()
            return {"lifecycle": _decode_rows([row])[0], "events": _decode_rows(events), "target_eligibility": _decode_rows(eligibility)}

    @app.get("/api/v1/lifecycles/{lifecycle_id}/evidence")
    def lifecycle_evidence(lifecycle_id: str) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                "SELECT * FROM endpoint_evidence WHERE lifecycle_id=? ORDER BY recorded_at",
                (lifecycle_id,),
            ).fetchall()
            return {"items": _decode_rows(rows)}

    @app.get("/api/v1/alerts")
    def alerts(limit: int = Query(500, ge=1, le=5000)) -> dict[str, Any]:
        with connect() as db:
            return {"items": _decode_rows(db.execute("SELECT * FROM alerts ORDER BY event_timestamp DESC LIMIT ?", (limit,)).fetchall())}

    @app.get("/api/v1/events")
    def events(limit: int = Query(500, ge=1, le=5000)) -> dict[str, Any]:
        with connect() as db:
            return {"items": _decode_rows(db.execute("SELECT * FROM lifecycle_events ORDER BY event_timestamp DESC LIMIT ?", (limit,)).fetchall())}

    @app.get("/api/v1/operating-state-evidence")
    def operating_state_evidence(
        machine_uid: str | None = None,
        limit: int = Query(500, ge=1, le=5000),
    ) -> dict[str, Any]:
        with connect() as db:
            if machine_uid:
                rows = db.execute(
                    "SELECT * FROM operating_state_evidence WHERE machine_uid=? "
                    "ORDER BY effective_from DESC LIMIT ?",
                    (machine_uid, limit),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM operating_state_evidence ORDER BY effective_from DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return {"items": _decode_rows(rows), "limit": limit, "access": "READ_ONLY"}

    @app.get("/api/v1/sources")
    def sources() -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                """
                SELECT s.* FROM source_config_versions s JOIN (
                    SELECT source_key,MAX(version_number) AS version_number
                    FROM source_config_versions GROUP BY source_key
                ) latest ON latest.source_key=s.source_key AND latest.version_number=s.version_number
                ORDER BY s.source_key
                """
            ).fetchall()
            items = _decode_rows(rows)
            for item in items:
                config = dict(item.get("config") or {})
                config.pop("dsn_env", None)
                config.pop("secret_reference", None)
                item["config"] = config
                item["credentials_exposed"] = False
            return {"items": items, "access": "READ_ONLY"}

    @app.get("/api/v1/sources/{source_key}")
    def source(source_key: str) -> dict[str, Any]:
        values = sources()["items"]
        for value in values:
            if value["source_key"] == source_key:
                return value
        raise HTTPException(404, "source not found")

    @app.get("/api/v1/sources/{source_key}/health")
    def source_health(source_key: str, limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                "SELECT * FROM source_health_events WHERE source_key=? ORDER BY recorded_at DESC LIMIT ?",
                (source_key, limit),
            ).fetchall()
            watermark = db.execute("SELECT * FROM source_watermarks WHERE source_key=?", (source_key,)).fetchone()
            return {"items": _decode_rows(rows), "watermark": _decode_rows([watermark])[0] if watermark else None, "access": "READ_ONLY"}

    @app.get("/api/v1/model")
    def model() -> dict[str, Any]:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file and manifest_file.is_file() else None
        return {
            "model_version": "v2.7",
            "model_artifact_sha256": sha256_file(model_file) if model_file.is_file() else None,
            "runtime_manifest": manifest,
            "synthetic_development": "PASS",
            "sealed_synthetic_holdout": "PASS",
            "plant_shadow": "IN_PROGRESS",
            "operating_context_policy": "plant_operating_context_policy_v1",
            "vibration_operating_policy": "vibration_operating_policy_v1",
            "rul_time_basis": "OPERATING_HOURS",
            "plant_validation": "NOT_STARTED",
            "production_authorized": False,
        }

    @app.get("/api/v1/plant-evaluation")
    def plant_evaluation() -> dict[str, Any]:
        with connect() as db:
            return evaluate_plant(db)

    @app.get("/api/v1/drift")
    def drift() -> dict[str, Any]:
        return {
            "synthetic_to_plant_distance": {"status": "INSUFFICIENT_EVIDENCE"},
            "within_plant_drift": {"status": "INSUFFICIENT_EVIDENCE"},
            "automatic_retraining": False,
        }

    @app.get("/api/v1/system/health")
    def system_health() -> dict[str, Any]:
        with connect() as db:
            version = db.execute("SELECT schema_version FROM schema_info WHERE singleton=1").fetchone()[0]
            watermarks = db.execute("SELECT COUNT(*) FROM source_watermarks").fetchone()[0]
        return {
            "status": "OK",
            "database": "READ_ONLY_API_CONNECTED",
            "schema_version": version,
            "expected_schema_version": SCHEMA_VERSION,
            "sources_with_committed_watermark": watermarks,
            "bind_policy": "LOCALHOST_ONLY",
            "plant_production_authorized": False,
        }

    @app.get("/api/v1/audit")
    def audit(limit: int = Query(500, ge=1, le=5000), offset: int = Query(0, ge=0)) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute("SELECT * FROM audit_log ORDER BY recorded_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            return {"items": _decode_rows(rows), "limit": limit, "offset": offset}

    return app
