from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .contracts import (
    LIFECYCLE_POLICY_VERSION,
    SCHEMA_VERSION,
    TRUTH_POLICY_VERSION,
    CensoringKind,
    EndpointClass,
    EndpointPrecision,
    ForecastState,
    LifecycleStatus,
    ManagerState,
    ObservationDisposition,
    OPERATING_CONTEXT_POLICY_VERSION,
    OperatingState,
    OperatingStateSource,
    PlantObservation,
    TargetName,
    TruthEligibility,
    default_endpoint_decisions,
    require_aware,
    utc_now,
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class ProcessingCommit:
    disposition: ObservationDisposition
    validation_status: str
    reasons: tuple[str, ...] = ()
    prediction: dict[str, Any] | None = None
    lifecycle_events: tuple[dict[str, Any], ...] = ()
    operating_context: dict[str, Any] | None = None
    vibration_inference: dict[str, Any] | None = None


@dataclass(frozen=True)
class CommitResult:
    ingestion_id: int
    lifecycle_id: str | None
    disposition: ObservationDisposition
    prediction_id: str | None
    duplicate: bool = False


class EvidenceStore:
    """Single-writer, append-only evidence ledger for plant shadow operation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._create_schema()

    def _create_schema(self) -> None:
        self.db.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS schema_info (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                schema_version TEXT NOT NULL,
                installed_at TEXT NOT NULL
            );
            INSERT INTO schema_info(singleton,schema_version,installed_at)
                VALUES(1,'plant_shadow_schema_v3_vibration_operating_inference',CURRENT_TIMESTAMP)
                ON CONFLICT(singleton) DO UPDATE SET
                    schema_version='plant_shadow_schema_v3_vibration_operating_inference';

            CREATE TABLE IF NOT EXISTS deployment_manifests (
                deployment_id TEXT PRIMARY KEY,
                manifest_json TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_config_versions (
                config_version_id TEXT PRIMARY KEY,
                source_key TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                UNIQUE(source_key,version_number)
            );

            CREATE TABLE IF NOT EXISTS raw_observations (
                ingestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_key TEXT NOT NULL,
                source_row_id TEXT NOT NULL,
                machine_uid TEXT NOT NULL,
                event_timestamp TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                line_sel TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                vrms REAL NOT NULL,
                arms REAL NOT NULL,
                apeak REAL NOT NULL,
                crest REAL NOT NULL,
                temp REAL NOT NULL,
                raw_json TEXT NOT NULL,
                UNIQUE(source_key,source_row_id)
            );
            CREATE INDEX IF NOT EXISTS idx_raw_source_order
                ON raw_observations(source_key,event_timestamp,source_row_id);
            CREATE INDEX IF NOT EXISTS idx_raw_machine_time
                ON raw_observations(machine_uid,event_timestamp);

            CREATE TABLE IF NOT EXISTS observation_dispositions (
                disposition_id TEXT PRIMARY KEY,
                ingestion_id INTEGER NOT NULL UNIQUE REFERENCES raw_observations(ingestion_id),
                disposition TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS operating_context_decisions (
                decision_id TEXT PRIMARY KEY,
                ingestion_id INTEGER NOT NULL UNIQUE REFERENCES raw_observations(ingestion_id),
                machine_uid TEXT NOT NULL,
                operating_state TEXT NOT NULL,
                operating_state_source TEXT NOT NULL,
                operating_state_confidence REAL NOT NULL CHECK(
                    operating_state_confidence >= 0 AND operating_state_confidence <= 1
                ),
                admitted_to_runtime INTEGER NOT NULL,
                reason_code TEXT NOT NULL,
                cumulative_operating_seconds REAL,
                effective_operating_timestamp TEXT,
                maintenance_event_id TEXT,
                state_transition INTEGER NOT NULL,
                previous_operating_state TEXT,
                policy_version TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_operating_context_machine
                ON operating_context_decisions(machine_uid,ingestion_id);

            CREATE TABLE IF NOT EXISTS vibration_operating_inferences (
                inference_id TEXT PRIMARY KEY,
                ingestion_id INTEGER NOT NULL UNIQUE REFERENCES raw_observations(ingestion_id),
                machine_uid TEXT NOT NULL,
                classification TEXT NOT NULL,
                confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                reason_code TEXT NOT NULL,
                calibration_state TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                history_span_seconds REAL NOT NULL,
                window_energy REAL,
                window_variability REAL,
                production_similarity REAL,
                novelty_score REAL,
                persistent_running_seconds REAL NOT NULL,
                quiet_center REAL,
                running_center REAL,
                separation REAL,
                calibration_artifact_sha256 TEXT,
                policy_version TEXT NOT NULL,
                details_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vibration_inference_machine
                ON vibration_operating_inferences(machine_uid,ingestion_id);

            CREATE TABLE IF NOT EXISTS operating_state_evidence (
                evidence_id TEXT PRIMARY KEY,
                source_system TEXT NOT NULL,
                external_event_id TEXT NOT NULL,
                machine_uid TEXT NOT NULL,
                operating_state TEXT NOT NULL,
                operating_state_source TEXT NOT NULL,
                confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                effective_from TEXT NOT NULL,
                effective_to TEXT,
                maintenance_event_id TEXT,
                details_json TEXT NOT NULL,
                actor TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                UNIQUE(source_system,external_event_id)
            );
            CREATE INDEX IF NOT EXISTS idx_operating_evidence_machine_time
                ON operating_state_evidence(machine_uid,effective_from,effective_to);

            CREATE TABLE IF NOT EXISTS source_watermarks (
                source_key TEXT PRIMARY KEY,
                event_timestamp TEXT NOT NULL,
                source_row_id TEXT NOT NULL,
                ingestion_id INTEGER NOT NULL REFERENCES raw_observations(ingestion_id),
                committed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS machine_registry (
                machine_uid TEXT PRIMARY KEY,
                source_key TEXT NOT NULL,
                line_sel TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                latest_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lifecycle_records (
                lifecycle_id TEXT PRIMARY KEY,
                machine_uid TEXT NOT NULL REFERENCES machine_registry(machine_uid),
                sequence_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                observed_start_timestamp TEXT NOT NULL,
                confirmed_start_timestamp TEXT,
                end_timestamp TEXT,
                left_censored INTEGER NOT NULL,
                closure_reason TEXT,
                closure_confidence TEXT,
                lifecycle_policy_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(machine_uid,sequence_number)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_lifecycle
                ON lifecycle_records(machine_uid) WHERE status='ACTIVE';

            CREATE TABLE IF NOT EXISTS lifecycle_manager_state_versions (
                state_version_id TEXT PRIMARY KEY,
                machine_uid TEXT NOT NULL REFERENCES machine_registry(machine_uid),
                lifecycle_id TEXT REFERENCES lifecycle_records(lifecycle_id),
                manager_state TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                effective_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lifecycle_events (
                event_id TEXT PRIMARY KEY,
                machine_uid TEXT NOT NULL REFERENCES machine_registry(machine_uid),
                lifecycle_id TEXT REFERENCES lifecycle_records(lifecycle_id),
                ingestion_id INTEGER REFERENCES raw_observations(ingestion_id),
                event_timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prediction_attempts (
                prediction_id TEXT PRIMARY KEY,
                ingestion_id INTEGER NOT NULL UNIQUE REFERENCES raw_observations(ingestion_id),
                deployment_id TEXT NOT NULL,
                lifecycle_id TEXT REFERENCES lifecycle_records(lifecycle_id),
                machine_uid TEXT NOT NULL,
                prediction_timestamp TEXT NOT NULL,
                health_state_model TEXT,
                health_state_manufacturer TEXT,
                health_state_source TEXT NOT NULL,
                connectivity_state TEXT NOT NULL,
                forecast_state TEXT NOT NULL,
                warning_point_hours REAL,
                warning_lower_hours REAL,
                warning_upper_hours REAL,
                warning_serviceable INTEGER NOT NULL,
                warning_withhold_reason TEXT,
                critical_point_hours REAL,
                critical_lower_hours REAL,
                critical_upper_hours REAL,
                critical_serviceable INTEGER NOT NULL,
                critical_withhold_reason TEXT,
                model_version TEXT NOT NULL,
                model_artifact_sha256 TEXT NOT NULL,
                runtime_manifest_id TEXT NOT NULL,
                inference_latency_ms REAL NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prediction_machine_time
                ON prediction_attempts(machine_uid,prediction_timestamp);

            CREATE TABLE IF NOT EXISTS prediction_supersessions (
                supersession_id TEXT PRIMARY KEY,
                original_prediction_id TEXT NOT NULL REFERENCES prediction_attempts(prediction_id),
                replacement_prediction_id TEXT NOT NULL REFERENCES prediction_attempts(prediction_id),
                reason TEXT NOT NULL,
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(original_prediction_id,replacement_prediction_id)
            );

            CREATE TABLE IF NOT EXISTS endpoint_evidence (
                evidence_id TEXT PRIMARY KEY,
                source_system TEXT NOT NULL,
                external_event_id TEXT NOT NULL,
                machine_uid TEXT NOT NULL REFERENCES machine_registry(machine_uid),
                lifecycle_id TEXT REFERENCES lifecycle_records(lifecycle_id),
                event_time_lower TEXT,
                event_time_upper TEXT,
                precision TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                actor TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                supersedes_id TEXT REFERENCES endpoint_evidence(evidence_id),
                UNIQUE(source_system,external_event_id)
            );

            CREATE TABLE IF NOT EXISTS endpoint_classifications (
                classification_id TEXT PRIMARY KEY,
                evidence_id TEXT NOT NULL REFERENCES endpoint_evidence(evidence_id),
                endpoint_class TEXT NOT NULL,
                confirmed INTEGER NOT NULL,
                actor TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                supersedes_id TEXT REFERENCES endpoint_classifications(classification_id)
            );

            CREATE TABLE IF NOT EXISTS target_truth_eligibility (
                eligibility_id TEXT PRIMARY KEY,
                classification_id TEXT NOT NULL REFERENCES endpoint_classifications(classification_id),
                lifecycle_id TEXT NOT NULL REFERENCES lifecycle_records(lifecycle_id),
                target TEXT NOT NULL,
                eligibility TEXT NOT NULL,
                censoring TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                decided_at TEXT NOT NULL,
                actor TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS target_truth (
                truth_id TEXT PRIMARY KEY,
                eligibility_id TEXT NOT NULL REFERENCES target_truth_eligibility(eligibility_id),
                prediction_id TEXT NOT NULL REFERENCES prediction_attempts(prediction_id),
                target TEXT NOT NULL,
                endpoint_timestamp TEXT NOT NULL,
                true_hours REAL NOT NULL CHECK(true_hours >= 0),
                created_at TEXT NOT NULL,
                UNIQUE(eligibility_id,prediction_id,target)
            );

            CREATE TABLE IF NOT EXISTS target_censoring (
                censoring_id TEXT PRIMARY KEY,
                eligibility_id TEXT NOT NULL REFERENCES target_truth_eligibility(eligibility_id),
                target TEXT NOT NULL,
                censoring TEXT NOT NULL,
                time_lower TEXT,
                time_upper TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS plant_evaluation_locks (
                lock_id TEXT PRIMARY KEY,
                lifecycle_id TEXT NOT NULL REFERENCES lifecycle_records(lifecycle_id),
                target TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role='PLANT_VALIDATION_LOCKED'),
                locked_at TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                UNIQUE(lifecycle_id,target)
            );

            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                source_key TEXT,
                machine_uid TEXT,
                lifecycle_id TEXT,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                message TEXT NOT NULL,
                event_timestamp TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_health_events (
                health_event_id TEXT PRIMARY KEY,
                source_key TEXT NOT NULL,
                connection_state TEXT NOT NULL,
                latest_source_timestamp TEXT,
                query_latency_ms REAL,
                malformed_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0,
                late_rows INTEGER NOT NULL DEFAULT 0,
                error_code TEXT,
                message TEXT,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                source_key TEXT,
                machine_uid TEXT,
                lifecycle_id TEXT,
                actor TEXT NOT NULL,
                details_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );

            -- A v1 ledger did not record machine operating context. Preserve every historical raw
            -- row, but never infer that it was RUNNING during upgrade. The deterministic IDs make
            -- this migration idempotent, and the runtime will replay these rows as paused evidence.
            INSERT INTO operating_context_decisions(
                decision_id,ingestion_id,machine_uid,operating_state,operating_state_source,
                operating_state_confidence,admitted_to_runtime,reason_code,
                cumulative_operating_seconds,effective_operating_timestamp,maintenance_event_id,
                state_transition,previous_operating_state,policy_version,recorded_at
            )
            SELECT
                'operating_context_legacy_' || r.ingestion_id,
                r.ingestion_id,
                r.machine_uid,
                'UNKNOWN',
                'UNAVAILABLE',
                0.0,
                0,
                'LEGACY_CONTEXT_UNAVAILABLE',
                NULL,
                NULL,
                NULL,
                0,
                NULL,
                'plant_operating_context_policy_v1',
                CURRENT_TIMESTAMP
            FROM raw_observations r
            LEFT JOIN operating_context_decisions o ON o.ingestion_id=r.ingestion_id
            WHERE o.ingestion_id IS NULL;
            COMMIT;
            """
        )
        immutable = (
            "source_config_versions",
            "source_health_events",
            "raw_observations",
            "observation_dispositions",
            "operating_context_decisions",
            "vibration_operating_inferences",
            "operating_state_evidence",
            "prediction_attempts",
            "prediction_supersessions",
            "lifecycle_manager_state_versions",
            "lifecycle_events",
            "endpoint_evidence",
            "endpoint_classifications",
            "target_truth_eligibility",
            "target_truth",
            "target_censoring",
            "plant_evaluation_locks",
            "audit_log",
        )
        for table in immutable:
            for operation in ("UPDATE", "DELETE"):
                trigger = f"protect_{table}_{operation.lower()}"
                self.db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {trigger} BEFORE {operation} ON {table} "
                    "BEGIN SELECT RAISE(ABORT, 'append-only evidence'); END"
                )

    def close(self) -> None:
        self.db.close()

    def save_source_config(
        self,
        source_key: str,
        config: dict[str, Any],
        *,
        actor: str,
        enabled: bool = False,
        archived: bool = False,
    ) -> str:
        if not source_key.strip():
            raise ValueError("source_key must be non-empty")
        version_id = _id("sourcecfg")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            version = int(self.db.execute(
                "SELECT COALESCE(MAX(version_number),0)+1 FROM source_config_versions WHERE source_key=?",
                (source_key,),
            ).fetchone()[0])
            self.db.execute(
                "INSERT INTO source_config_versions VALUES(?,?,?,?,?,?,?,?)",
                (version_id, source_key, version, _json(config), int(enabled), int(archived), utc_now().isoformat(), actor),
            )
            self._audit(
                "SOURCE_CONFIG_VERSION_CREATED",
                actor=actor,
                source_key=source_key,
                details={"version_id": version_id, "version": version, "enabled": enabled, "archived": archived},
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return version_id

    def latest_source_configs(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        query = """
            SELECT s.* FROM source_config_versions s
            JOIN (
                SELECT source_key,MAX(version_number) AS version_number
                FROM source_config_versions GROUP BY source_key
            ) latest ON latest.source_key=s.source_key AND latest.version_number=s.version_number
        """
        if not include_archived:
            query += " WHERE s.archived=0"
        query += " ORDER BY s.source_key"
        result = []
        for row in self.db.execute(query):
            value = dict(row)
            value["config"] = json.loads(value.pop("config_json"))
            result.append(value)
        return result

    def _source_row_id_order_value(self, source_key: str, source_row_id: str) -> int | str | uuid.UUID:
        row = self.db.execute(
            "SELECT config_json FROM source_config_versions WHERE source_key=? "
            "ORDER BY version_number DESC LIMIT 1",
            (source_key,),
        ).fetchone()
        row_id_kind = "text"
        if row is not None:
            row_id_kind = str(json.loads(str(row[0])).get("row_id_kind", "text"))
        if row_id_kind == "integer":
            return int(source_row_id)
        if row_id_kind == "uuid":
            return uuid.UUID(source_row_id)
        return source_row_id

    def _at_or_behind_watermark(
        self,
        observation: PlantObservation,
        watermark_timestamp: str,
        watermark_row_id: str,
    ) -> bool:
        incoming_timestamp = observation.order_key[0]
        if incoming_timestamp != watermark_timestamp:
            return incoming_timestamp < watermark_timestamp
        incoming_id = self._source_row_id_order_value(observation.source_key, observation.source_row_id)
        watermark_id = self._source_row_id_order_value(observation.source_key, watermark_row_id)
        return incoming_id <= watermark_id

    def record_source_health(
        self,
        source_key: str,
        connection_state: str,
        *,
        latest_source_timestamp: str | None = None,
        query_latency_ms: float | None = None,
        malformed_rows: int = 0,
        duplicate_rows: int = 0,
        late_rows: int = 0,
        error_code: str | None = None,
        message: str | None = None,
    ) -> str:
        health_event_id = _id("sourcehealth")
        self.db.execute(
            "INSERT INTO source_health_events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (health_event_id, source_key, connection_state, latest_source_timestamp, query_latency_ms, malformed_rows, duplicate_rows, late_rows, error_code, message, utc_now().isoformat()),
        )
        return health_event_id

    def add_operating_state_evidence(
        self,
        *,
        source_system: str,
        external_event_id: str,
        machine_uid: str,
        operating_state: OperatingState,
        operating_state_source: OperatingStateSource,
        confidence: float,
        effective_from: datetime,
        effective_to: datetime | None,
        maintenance_event_id: str | None,
        details: dict[str, Any],
        actor: str,
    ) -> str:
        start = require_aware(effective_from, "effective_from").isoformat()
        end = require_aware(effective_to, "effective_to").isoformat() if effective_to else None
        if end is not None and end < start:
            raise ValueError("operating-state evidence end cannot precede start")
        numeric_confidence = float(confidence)
        if not 0.0 <= numeric_confidence <= 1.0:
            raise ValueError("operating-state evidence confidence must be within [0,1]")
        evidence_id = _id("opevidence")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO operating_state_evidence VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    evidence_id, source_system, external_event_id, machine_uid,
                    operating_state, operating_state_source, numeric_confidence,
                    start, end, maintenance_event_id, _json(details), actor,
                    utc_now().isoformat(),
                ),
            )
            self._audit(
                "OPERATING_STATE_EVIDENCE_ADDED",
                actor=actor,
                machine_uid=machine_uid,
                details={
                    "evidence_id": evidence_id,
                    "operating_state": operating_state,
                    "operating_state_source": operating_state_source,
                    "effective_from": start,
                    "effective_to": end,
                },
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return evidence_id

    def resolve_operating_context(self, observation: PlantObservation) -> PlantObservation:
        timestamp = require_aware(observation.event_timestamp, "event_timestamp").isoformat()
        rows = self.db.execute(
            "SELECT * FROM operating_state_evidence WHERE machine_uid=? AND effective_from<=? "
            "AND (effective_to IS NULL OR effective_to>=?) ORDER BY recorded_at DESC",
            (observation.uid, timestamp, timestamp),
        ).fetchall()
        if not rows:
            return observation
        states = {str(row["operating_state"]) for row in rows}
        overrides = asdict(observation)
        if len(states) != 1:
            overrides.update(
                operating_state=OperatingState.UNKNOWN,
                operating_state_source=OperatingStateSource.UNAVAILABLE,
                operating_state_confidence=0.0,
                maintenance_event_id=None,
                operating_context_block_reason="CONTRADICTORY_LOCAL_OPERATING_EVIDENCE",
            )
            return PlantObservation(**overrides)
        selected = max(rows, key=lambda row: (float(row["confidence"]), str(row["recorded_at"])))
        overrides.update(
            operating_state=OperatingState(str(selected["operating_state"])),
            operating_state_source=OperatingStateSource(str(selected["operating_state_source"])),
            operating_state_confidence=float(selected["confidence"]),
            maintenance_event_id=selected["maintenance_event_id"],
        )
        return PlantObservation(**overrides)

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _active_lifecycle(self, machine_uid: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM lifecycle_records WHERE machine_uid=? AND status='ACTIVE'",
            (machine_uid,),
        ).fetchone()

    def _create_lifecycle(self, observation: PlantObservation, ingestion_id: int) -> str:
        row = self.db.execute(
            "SELECT COALESCE(MAX(sequence_number),0)+1 FROM lifecycle_records WHERE machine_uid=?",
            (observation.uid,),
        ).fetchone()
        sequence = int(row[0])
        lifecycle_id = _id("lc")
        now = utc_now().isoformat()
        event_time = require_aware(observation.event_timestamp, "event_timestamp").isoformat()
        self.db.execute(
            "INSERT INTO lifecycle_records(lifecycle_id,machine_uid,sequence_number,status,"
            "observed_start_timestamp,confirmed_start_timestamp,end_timestamp,left_censored,closure_reason,"
            "closure_confidence,lifecycle_policy_version,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (lifecycle_id, observation.uid, sequence, LifecycleStatus.ACTIVE, event_time, None, None, 1, None, None, LIFECYCLE_POLICY_VERSION, now, now),
        )
        self.db.execute(
            "INSERT INTO lifecycle_manager_state_versions VALUES(?,?,?,?,?,?,?,?)",
            (_id("state"), observation.uid, lifecycle_id, ManagerState.ACTIVE, "FIRST_OBSERVED_ROW_LEFT_CENSORED", LIFECYCLE_POLICY_VERSION, event_time, now),
        )
        self._insert_event(
            observation.uid,
            lifecycle_id,
            ingestion_id,
            event_time,
            "LIFECYCLE_STARTED",
            "INFO",
            "Observed a machine without an independently confirmed start; lifecycle is left-censored.",
            {"left_censored": True},
        )
        return lifecycle_id

    def _insert_event(
        self,
        machine_uid: str,
        lifecycle_id: str | None,
        ingestion_id: int | None,
        event_timestamp: str,
        event_type: str,
        severity: str,
        message: str,
        evidence: dict[str, Any],
    ) -> str:
        event_id = _id("evt")
        self.db.execute(
            "INSERT INTO lifecycle_events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (event_id, machine_uid, lifecycle_id, ingestion_id, event_timestamp, event_type, severity, message, _json(evidence), LIFECYCLE_POLICY_VERSION, utc_now().isoformat()),
        )
        return event_id

    def _audit(
        self,
        event_type: str,
        *,
        actor: str,
        source_key: str | None = None,
        machine_uid: str | None = None,
        lifecycle_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO audit_log VALUES(?,?,?,?,?,?,?,?)",
            (_id("audit"), event_type, source_key, machine_uid, lifecycle_id, actor, _json(details or {}), utc_now().isoformat()),
        )

    def process_observation(
        self,
        observation: PlantObservation,
        processor: Callable[[int, str | None], ProcessingCommit],
        *,
        actor: str = "plant-shadow-worker",
        context_resolver: Callable[[PlantObservation], PlantObservation] | None = None,
    ) -> CommitResult:
        self.db.execute("BEGIN IMMEDIATE")
        runtime_advanced = False
        try:
            existing = self.db.execute(
                "SELECT ingestion_id FROM raw_observations WHERE source_key=? AND source_row_id=?",
                (observation.source_key, observation.source_row_id),
            ).fetchone()
            if existing is not None:
                prediction = self.db.execute(
                    "SELECT prediction_id FROM prediction_attempts WHERE ingestion_id=?",
                    (existing[0],),
                ).fetchone()
                self.db.execute("ROLLBACK")
                return CommitResult(int(existing[0]), None, ObservationDisposition.DUPLICATE, prediction[0] if prediction else None, True)

            cursor = self.db.execute(
                "INSERT INTO raw_observations(source_key,source_row_id,machine_uid,event_timestamp,observed_at,"
                "line_sel,machine_id,vrms,arms,apeak,crest,temp,raw_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    observation.source_key,
                    observation.source_row_id,
                    observation.uid,
                    require_aware(observation.event_timestamp, "event_timestamp").isoformat(),
                    require_aware(observation.observed_at, "observed_at").isoformat(),
                    observation.line_sel,
                    observation.machine_id,
                    observation.vrms,
                    observation.arms,
                    observation.apeak,
                    observation.crest,
                    observation.temp,
                    _json(observation.raw_payload),
                ),
            )
            ingestion_id = int(cursor.lastrowid)
            now = utc_now().isoformat()
            self.db.execute(
                "INSERT INTO machine_registry(machine_uid,source_key,line_sel,machine_id,first_seen_at,latest_seen_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(machine_uid) DO UPDATE SET latest_seen_at=excluded.latest_seen_at",
                (observation.uid, observation.source_key, observation.line_sel, observation.machine_id, now, now),
            )

            watermark = self.db.execute(
                "SELECT event_timestamp,source_row_id FROM source_watermarks WHERE source_key=?",
                (observation.source_key,),
            ).fetchone()
            incoming_order = observation.order_key
            late = watermark is not None and self._at_or_behind_watermark(
                observation,
                str(watermark[0]),
                str(watermark[1]),
            )
            if not late and context_resolver is not None:
                observation = context_resolver(observation)
            active = self._active_lifecycle(observation.uid)
            lifecycle_id = str(active["lifecycle_id"]) if active else (
                self._create_lifecycle(observation, ingestion_id)
                if observation.confirmed_running and not late else None
            )
            if late:
                commit = ProcessingCommit(
                    ObservationDisposition.LATE_QUARANTINED,
                    "NOT_PROCESSED",
                    ("observation is at or behind the committed source watermark",),
                )
            else:
                runtime_advanced = True
                commit = processor(ingestion_id, lifecycle_id)

            context = dict(commit.operating_context or {})
            if not context:
                context = {
                    "operating_state": observation.operating_state.value,
                    "operating_state_source": observation.operating_state_source.value,
                    "operating_state_confidence": observation.operating_state_confidence,
                    "admitted_to_runtime": False,
                    "reason_code": (
                        "LATE_QUARANTINED" if late else "OPERATING_CONTEXT_NOT_RESOLVED"
                    ),
                    "cumulative_operating_seconds": None,
                    "effective_operating_timestamp": None,
                    "maintenance_event_id": observation.maintenance_event_id,
                    "state_transition": False,
                    "previous_operating_state": None,
                    "policy_version": OPERATING_CONTEXT_POLICY_VERSION,
                }
            self.db.execute(
                "INSERT INTO operating_context_decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    _id("opctx"),
                    ingestion_id,
                    observation.uid,
                    str(context["operating_state"]),
                    str(context["operating_state_source"]),
                    float(context["operating_state_confidence"]),
                    int(bool(context["admitted_to_runtime"])),
                    str(context["reason_code"]),
                    context.get("cumulative_operating_seconds"),
                    context.get("effective_operating_timestamp"),
                    context.get("maintenance_event_id"),
                    int(bool(context.get("state_transition"))),
                    context.get("previous_operating_state"),
                    str(context.get("policy_version", OPERATING_CONTEXT_POLICY_VERSION)),
                    now,
                ),
            )

            inference = dict(commit.vibration_inference or {})
            if inference:
                self.db.execute(
                    "INSERT INTO vibration_operating_inferences VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        _id("vibop"), ingestion_id, observation.uid,
                        str(inference["classification"]), float(inference["confidence"]),
                        str(inference["reason_code"]), str(inference["calibration_state"]),
                        int(inference["sample_count"]), float(inference["history_span_seconds"]),
                        inference.get("window_energy"), inference.get("window_variability"),
                        inference.get("production_similarity"), inference.get("novelty_score"),
                        float(inference.get("persistent_running_seconds", 0.0)),
                        inference.get("quiet_center"), inference.get("running_center"),
                        inference.get("separation"), inference.get("calibration_artifact_sha256"),
                        str(inference["policy_version"]), _json(inference), now,
                    ),
                )

            self.db.execute(
                "INSERT INTO observation_dispositions VALUES(?,?,?,?,?,?,?)",
                (_id("disp"), ingestion_id, commit.disposition, commit.validation_status, _json(commit.reasons), LIFECYCLE_POLICY_VERSION, now),
            )
            prediction_id = None
            if commit.prediction is not None:
                prediction_id = self._insert_prediction(ingestion_id, lifecycle_id, observation.uid, commit.prediction)
                if bool(commit.prediction.get("model_state_reset_gap")):
                    self._insert_event(
                        observation.uid,
                        lifecycle_id,
                        ingestion_id,
                        incoming_order[0],
                        "MODEL_STATE_RESET_GAP",
                        "WARNING",
                        "The causal feature runtime reset after a gap; the lifecycle remained active.",
                        {},
                    )
            for event in commit.lifecycle_events:
                self._insert_event(
                    observation.uid,
                    lifecycle_id,
                    ingestion_id,
                    incoming_order[0],
                    str(event["event_type"]),
                    str(event.get("severity", "INFO")),
                    str(event.get("message", "")),
                    dict(event.get("evidence") or {}),
                )
            if not late:
                self.db.execute(
                    "INSERT INTO source_watermarks(source_key,event_timestamp,source_row_id,ingestion_id,committed_at) "
                    "VALUES(?,?,?,?,?) ON CONFLICT(source_key) DO UPDATE SET event_timestamp=excluded.event_timestamp,"
                    "source_row_id=excluded.source_row_id,ingestion_id=excluded.ingestion_id,committed_at=excluded.committed_at",
                    (observation.source_key, incoming_order[0], incoming_order[1], ingestion_id, now),
                )
            self._audit(
                "OBSERVATION_COMMITTED",
                actor=actor,
                source_key=observation.source_key,
                machine_uid=observation.uid,
                lifecycle_id=lifecycle_id,
                details={"ingestion_id": ingestion_id, "disposition": commit.disposition},
            )
            self.db.execute("COMMIT")
            return CommitResult(ingestion_id, lifecycle_id, commit.disposition, prediction_id)
        except Exception as exc:
            self.db.execute("ROLLBACK")
            setattr(exc, "plant_shadow_runtime_advanced", runtime_advanced)
            raise

    def _insert_prediction(
        self,
        ingestion_id: int,
        lifecycle_id: str | None,
        machine_uid: str,
        value: dict[str, Any],
    ) -> str:
        prediction_id = str(value.get("prediction_id") or _id("pred"))
        self.db.execute(
            "INSERT INTO prediction_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                prediction_id,
                ingestion_id,
                str(value["deployment_id"]),
                lifecycle_id,
                machine_uid,
                str(value["prediction_timestamp"]),
                value.get("health_state_model"),
                value.get("health_state_manufacturer"),
                str(value.get("health_state_source", "MODEL")),
                str(value.get("connectivity_state", "ONLINE")),
                str(value.get("forecast_state", ForecastState.WITHHELD)),
                value.get("warning_point_hours"),
                value.get("warning_lower_hours"),
                value.get("warning_upper_hours"),
                int(bool(value.get("warning_serviceable"))),
                value.get("warning_withhold_reason"),
                value.get("critical_point_hours"),
                value.get("critical_lower_hours"),
                value.get("critical_upper_hours"),
                int(bool(value.get("critical_serviceable"))),
                value.get("critical_withhold_reason"),
                str(value.get("model_version", "v2.7")),
                str(value["model_artifact_sha256"]),
                str(value["runtime_manifest_id"]),
                float(value.get("inference_latency_ms", 0.0)),
                _json(value),
                utc_now().isoformat(),
            ),
        )
        return prediction_id

    def committed_observations(self, source_key: str) -> Iterator[tuple[int, PlantObservation]]:
        active_start_rows = self.db.execute(
            "SELECT l.machine_uid,l.observed_start_timestamp FROM lifecycle_records l "
            "JOIN machine_registry m ON m.machine_uid=l.machine_uid "
            "WHERE m.source_key=? AND l.status='ACTIVE'",
            (source_key,),
        ).fetchall()
        active_starts = {
            str(row["machine_uid"]): str(row["observed_start_timestamp"])
            for row in active_start_rows
        }
        rows = self.db.execute(
            "SELECT r.*,o.operating_state,o.operating_state_source,o.operating_state_confidence,"
            "o.maintenance_event_id,v.classification AS vibration_classification," 
            "v.reason_code AS vibration_reason FROM raw_observations r "
            "JOIN observation_dispositions d ON d.ingestion_id=r.ingestion_id "
            "JOIN operating_context_decisions o ON o.ingestion_id=r.ingestion_id "
            "LEFT JOIN vibration_operating_inferences v ON v.ingestion_id=r.ingestion_id "
            "WHERE r.source_key=? AND d.disposition IN "
            "('PROCESSED','INVALID','HELD_OPERATING_CONTEXT') ORDER BY r.ingestion_id",
            (source_key,),
        )
        for row in rows:
            active_start = active_starts.get(str(row["machine_uid"]))
            yield int(row["ingestion_id"]), PlantObservation(
                source_key=str(row["source_key"]),
                source_row_id=str(row["source_row_id"]),
                event_timestamp=datetime.fromisoformat(str(row["event_timestamp"])),
                observed_at=datetime.fromisoformat(str(row["observed_at"])),
                line_sel=str(row["line_sel"]),
                machine_id=str(row["machine_id"]),
                vrms=float(row["vrms"]),
                arms=float(row["arms"]),
                apeak=float(row["apeak"]),
                crest=float(row["crest"]),
                temp=float(row["temp"]),
                raw_payload=json.loads(str(row["raw_json"])),
                operating_state=str(row["operating_state"]),
                operating_state_source=str(row["operating_state_source"]),
                operating_state_confidence=float(row["operating_state_confidence"]),
                maintenance_event_id=row["maintenance_event_id"],
                operating_context_block_reason=(
                    str(row["vibration_reason"])
                    if row["vibration_classification"] == "AUTHORITATIVE_BLOCKED" else None
                ),
                runtime_replay_eligible=(
                    active_start is not None and str(row["event_timestamp"]) >= active_start
                ),
            )

    def add_endpoint_evidence(
        self,
        *,
        source_system: str,
        external_event_id: str,
        machine_uid: str,
        lifecycle_id: str,
        precision: EndpointPrecision,
        event_time_lower: datetime | None,
        event_time_upper: datetime | None,
        details: dict[str, Any],
        actor: str,
        supersedes_id: str | None = None,
    ) -> str:
        lower = require_aware(event_time_lower, "event_time_lower").isoformat() if event_time_lower else None
        upper = require_aware(event_time_upper, "event_time_upper").isoformat() if event_time_upper else None
        if precision == EndpointPrecision.EXACT_TIMESTAMP and (lower is None or lower != upper):
            raise ValueError("EXACT_TIMESTAMP requires equal lower and upper timestamps")
        evidence_id = _id("evidence")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO endpoint_evidence VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (evidence_id, source_system, external_event_id, machine_uid, lifecycle_id, lower, upper, precision, _json(details), actor, utc_now().isoformat(), supersedes_id),
            )
            self._insert_event(machine_uid, lifecycle_id, None, lower or utc_now().isoformat(), "ENDPOINT_EVIDENCE_ATTACHED", "INFO", "Independent endpoint evidence attached.", {"evidence_id": evidence_id, "precision": precision})
            self._audit("ENDPOINT_EVIDENCE_ADDED", actor=actor, machine_uid=machine_uid, lifecycle_id=lifecycle_id, details={"evidence_id": evidence_id})
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return evidence_id

    def confirm_endpoint_classification(
        self,
        evidence_id: str,
        endpoint_class: EndpointClass,
        *,
        actor: str,
    ) -> dict[str, Any]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            evidence = self.db.execute("SELECT * FROM endpoint_evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
            if evidence is None:
                raise KeyError(evidence_id)
            classification_id = _id("class")
            now = utc_now().isoformat()
            self.db.execute(
                "INSERT INTO endpoint_classifications VALUES(?,?,?,?,?,?,?,?)",
                (classification_id, evidence_id, endpoint_class, 1, actor, TRUTH_POLICY_VERSION, now, None),
            )
            precision = EndpointPrecision(str(evidence["precision"]))
            decisions = default_endpoint_decisions(endpoint_class, precision)
            created_truth = 0
            eligibility_ids: list[str] = []
            for decision in decisions:
                eligibility_id = _id("elig")
                eligibility_ids.append(eligibility_id)
                self.db.execute(
                    "INSERT INTO target_truth_eligibility VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (eligibility_id, classification_id, evidence["lifecycle_id"], decision.target, decision.eligibility, decision.censoring, decision.reason_code, TRUTH_POLICY_VERSION, now, actor),
                )
                if decision.eligibility == TruthEligibility.ELIGIBLE_EXACT:
                    endpoint_timestamp = str(evidence["event_time_lower"])
                    created_truth += self._create_exact_truth(
                        eligibility_id,
                        str(evidence["lifecycle_id"]),
                        decision.target,
                        endpoint_timestamp,
                    )
                    self.db.execute(
                        "INSERT INTO plant_evaluation_locks VALUES(?,?,?,?,?,?) ON CONFLICT(lifecycle_id,target) DO NOTHING",
                        (_id("lock"), evidence["lifecycle_id"], decision.target, "PLANT_VALIDATION_LOCKED", now, TRUTH_POLICY_VERSION),
                    )
                elif decision.eligibility in {TruthEligibility.CENSORED, TruthEligibility.ELIGIBLE_INTERVAL}:
                    self.db.execute(
                        "INSERT INTO target_censoring VALUES(?,?,?,?,?,?,?)",
                        (_id("censor"), eligibility_id, decision.target, decision.censoring, evidence["event_time_lower"], evidence["event_time_upper"], now),
                    )
            if endpoint_class in {
                EndpointClass.PREVENTIVE_MAINTENANCE,
                EndpointClass.COMPONENT_REPLACEMENT,
                EndpointClass.MACHINE_REPLACEMENT,
            }:
                self.db.execute(
                    "UPDATE lifecycle_records SET status=?,end_timestamp=?,closure_reason=?,closure_confidence=?,updated_at=? "
                    "WHERE lifecycle_id=? AND status='ACTIVE'",
                    (LifecycleStatus.CLOSED_CONFIRMED, evidence["event_time_lower"], endpoint_class, "CONFIRMED", now, evidence["lifecycle_id"]),
                )
                self.db.execute(
                    "INSERT INTO lifecycle_manager_state_versions VALUES(?,?,?,?,?,?,?,?)",
                    (_id("state"), evidence["machine_uid"], evidence["lifecycle_id"], ManagerState.UNINITIALIZED, "CONFIRMED_ENDPOINT_CLOSURE", LIFECYCLE_POLICY_VERSION, evidence["event_time_lower"] or now, now),
                )
            self._insert_event(str(evidence["machine_uid"]), str(evidence["lifecycle_id"]), None, str(evidence["event_time_lower"] or now), "ENDPOINT_CLASSIFIED", "INFO", "Endpoint evidence classified and target eligibility decided.", {"classification_id": classification_id, "endpoint_class": endpoint_class, "eligibility_ids": eligibility_ids})
            self._audit("ENDPOINT_CLASSIFIED", actor=actor, machine_uid=evidence["machine_uid"], lifecycle_id=evidence["lifecycle_id"], details={"classification_id": classification_id, "endpoint_class": endpoint_class})
            self.db.execute("COMMIT")
            return {"classification_id": classification_id, "eligibility_ids": eligibility_ids, "truth_rows_created": created_truth}
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def _create_exact_truth(self, eligibility_id: str, lifecycle_id: str, target: TargetName, endpoint_timestamp: str) -> int:
        endpoint = datetime.fromisoformat(endpoint_timestamp)
        rows = self.db.execute(
            "SELECT p.prediction_id,p.machine_uid,r.event_timestamp,"
            "o.cumulative_operating_seconds FROM prediction_attempts p "
            "JOIN raw_observations r ON r.ingestion_id=p.ingestion_id "
            "LEFT JOIN operating_context_decisions o ON o.ingestion_id=p.ingestion_id "
            "WHERE p.lifecycle_id=?",
            (lifecycle_id,),
        ).fetchall()
        endpoint_operating_seconds = None
        if rows:
            endpoint_context = self.db.execute(
                "SELECT o.cumulative_operating_seconds FROM operating_context_decisions o "
                "JOIN raw_observations r ON r.ingestion_id=o.ingestion_id "
                "WHERE o.machine_uid=? AND r.event_timestamp>=? "
                "AND o.cumulative_operating_seconds IS NOT NULL "
                "ORDER BY r.event_timestamp,r.ingestion_id LIMIT 1",
                (rows[0]["machine_uid"], endpoint_timestamp),
            ).fetchone()
            if endpoint_context is not None:
                endpoint_operating_seconds = float(endpoint_context[0])
        created = 0
        for row in rows:
            event_time = datetime.fromisoformat(str(row["event_timestamp"]))
            prediction_operating_seconds = row["cumulative_operating_seconds"]
            if endpoint_operating_seconds is not None and prediction_operating_seconds is not None:
                true_hours = (
                    endpoint_operating_seconds - float(prediction_operating_seconds)
                ) / 3600.0
            else:
                # Backward-compatible evidence created before operating-context coverage exists.
                true_hours = (endpoint - event_time).total_seconds() / 3600.0
            if true_hours < 0:
                continue
            self.db.execute(
                "INSERT INTO target_truth VALUES(?,?,?,?,?,?,?)",
                (_id("truth"), eligibility_id, row["prediction_id"], target, endpoint_timestamp, true_hours, utc_now().isoformat()),
            )
            created += 1
        return created

    def rows(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.execute(query, params).fetchall()]

    def overview(self) -> dict[str, Any]:
        machines = int(self.db.execute("SELECT COUNT(*) FROM machine_registry").fetchone()[0])
        sources = int(self.db.execute("SELECT COUNT(DISTINCT source_key) FROM machine_registry").fetchone()[0])
        predictions = int(self.db.execute("SELECT COUNT(*) FROM prediction_attempts").fetchone()[0])
        lifecycles = int(self.db.execute("SELECT COUNT(*) FROM lifecycle_records").fetchone()[0])
        running = int(self.db.execute(
            "SELECT COUNT(*) FROM machine_registry m WHERE ("
            "SELECT operating_state FROM operating_context_decisions o "
            "WHERE o.machine_uid=m.machine_uid ORDER BY o.ingestion_id DESC LIMIT 1"
            ")='RUNNING'"
        ).fetchone()[0])
        return {
            "sources": sources,
            "machines": machines,
            "machines_running": running,
            "predictions": predictions,
            "lifecycles": lifecycles,
            "plant_production_authorized": False,
        }
