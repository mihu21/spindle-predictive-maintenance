from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from .contracts import (
    ActivationLevel,
    OperatingState,
    OperatingStateSource,
    PlantObservation,
    SourceDefinition,
)


class PlantSourceError(RuntimeError):
    pass


def _psycopg():
    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("psycopg is required for PostgreSQL plant shadow") from exc
    return psycopg, sql


@dataclass(frozen=True)
class PlantColumnMap:
    source_row_id: str = "id"
    timestamp: str = "timestamp"
    line_sel: str = "line_sel"
    machine_id: str = "machine_id"
    vrms: str = "vrms"
    arms: str = "arms"
    apeak: str = "apeak"
    crest: str = "crest"
    temp: str = "temp"
    operating_state: str | None = None
    operating_state_source: str | None = None
    operating_state_confidence: str | None = None
    maintenance_event_id: str | None = None

    def validate(self) -> None:
        required = {
            "source_row_id", "timestamp", "line_sel", "machine_id",
            "vrms", "arms", "apeak", "crest", "temp",
        }
        for name, value in self.__dict__.items():
            if name in required and not str(value).strip():
                raise ValueError(f"column {name} must be non-empty")
            if name not in required and value is not None and not str(value).strip():
                raise ValueError(f"optional column {name} must be non-empty when supplied")
        if self.source_row_id.lower() == "ctid":
            raise ValueError("SOURCE_IDENTITY_UNSAFE: PostgreSQL ctid is not durable")


@dataclass(frozen=True)
class PlantPostgresConfig:
    source_key: str
    display_name: str
    dsn_env: str
    schema: str = "public"
    table: str = "vvb001_readings"
    columns: PlantColumnMap = field(default_factory=PlantColumnMap)
    activation_level: ActivationLevel = ActivationLevel.SHADOW_MONITORING
    row_id_kind: str = "integer"
    batch_size: int = 1000
    poll_seconds: float = 5.0
    lateness_seconds: float = 300.0
    initial_lookback_hours: float | None = 24.0
    statement_timeout_seconds: float = 30.0
    require_ordering_index: bool = True

    def validate(self) -> None:
        self.columns.validate()
        if self.batch_size < 1 or self.poll_seconds < 0 or self.lateness_seconds < 0:
            raise ValueError("batch_size must be positive; poll_seconds and lateness_seconds cannot be negative")
        if self.initial_lookback_hours is not None and self.initial_lookback_hours <= 0:
            raise ValueError("initial_lookback_hours must be positive or null for explicit full history")
        if self.statement_timeout_seconds <= 0:
            raise ValueError("statement_timeout_seconds must be positive")
        if self.row_id_kind not in {"integer", "text", "uuid"}:
            raise ValueError("row_id_kind must be integer, text, or uuid")
        SourceDefinition(
            self.source_key,
            self.display_name,
            f"ENV:{self.dsn_env}",
            self.schema,
            self.table,
            self.columns.timestamp,
            self.columns.source_row_id,
            activation_level=self.activation_level,
        ).validate_activation()

    def resolved_dsn(self) -> str:
        value = os.getenv(self.dsn_env, "")
        if not value:
            raise PlantSourceError(f"secret environment variable {self.dsn_env!r} is not set")
        return value

    def redacted_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "display_name": self.display_name,
            "dsn_env": self.dsn_env,
            "schema": self.schema,
            "table": self.table,
            "columns": dict(self.columns.__dict__),
            "activation_level": self.activation_level,
            "row_id_kind": self.row_id_kind,
            "batch_size": self.batch_size,
            "poll_seconds": self.poll_seconds,
            "lateness_seconds": self.lateness_seconds,
            "initial_lookback_hours": self.initial_lookback_hours,
            "statement_timeout_seconds": self.statement_timeout_seconds,
            "require_ordering_index": self.require_ordering_index,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlantPostgresConfig":
        raw = dict(value)
        columns = PlantColumnMap(**dict(raw.pop("columns", {})))
        if "activation_level" in raw:
            raw["activation_level"] = ActivationLevel(str(raw["activation_level"]))
        result = cls(columns=columns, **raw)
        result.validate()
        return result


class PlantPostgresSource:
    def __init__(self, config: PlantPostgresConfig) -> None:
        config.validate()
        self.config = config
        self.connection = None
        self.bootstrap_window: dict[str, str | float] | None = None

    def connect(self) -> None:
        psycopg, _ = _psycopg()
        self.connection = psycopg.connect(self.config.resolved_dsn(), autocommit=True)
        with self.connection.cursor() as cursor:
            cursor.execute("SET default_transaction_read_only = on")
            cursor.execute("SHOW default_transaction_read_only")
            if str(cursor.fetchone()[0]).lower() not in {"on", "true", "1"}:
                raise PlantSourceError("PostgreSQL session is not read-only")
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (f"{int(self.config.statement_timeout_seconds * 1000)}ms",),
            )

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "PlantPostgresSource":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _connection(self):
        if self.connection is None:
            raise PlantSourceError("source is not connected")
        return self.connection

    def validate_schema(self) -> dict[str, Any]:
        connection = self._connection()
        required = {value for value in self.config.columns.__dict__.values() if value is not None}
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT column_name,data_type FROM information_schema.columns "
                "WHERE table_schema=%s AND table_name=%s",
                (self.config.schema, self.config.table),
            )
            columns = {str(name): str(data_type) for name, data_type in cursor.fetchall()}
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_index i "
                "JOIN pg_catalog.pg_class t ON t.oid=i.indrelid "
                "JOIN pg_catalog.pg_namespace n ON n.oid=t.relnamespace "
                "JOIN pg_catalog.pg_attribute a "
                "ON a.attrelid=t.oid AND a.attnum=i.indkey[0] "
                "WHERE n.nspname=%s AND t.relname=%s AND a.attname=%s "
                "AND i.indisunique AND i.indisvalid AND i.indisready "
                "AND i.indnkeyatts=1 AND i.indpred IS NULL AND i.indexprs IS NULL LIMIT 1",
                (self.config.schema, self.config.table, self.config.columns.source_row_id),
            )
            durable_unique_identity = cursor.fetchone() is not None
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_index i "
                "JOIN pg_catalog.pg_class t ON t.oid=i.indrelid "
                "JOIN pg_catalog.pg_namespace n ON n.oid=t.relnamespace "
                "JOIN pg_catalog.pg_attribute first_key "
                "ON first_key.attrelid=t.oid AND first_key.attnum=i.indkey[0] "
                "JOIN pg_catalog.pg_attribute second_key "
                "ON second_key.attrelid=t.oid AND second_key.attnum=i.indkey[1] "
                "WHERE n.nspname=%s AND t.relname=%s "
                "AND first_key.attname=%s AND second_key.attname=%s "
                "AND i.indisvalid AND i.indisready AND i.indnkeyatts>=2 "
                "AND i.indpred IS NULL AND i.indexprs IS NULL LIMIT 1",
                (
                    self.config.schema,
                    self.config.table,
                    self.config.columns.timestamp,
                    self.config.columns.source_row_id,
                ),
            )
            ordering_index = cursor.fetchone() is not None
        missing = sorted(required - set(columns))
        if missing:
            raise PlantSourceError("missing mapped source columns: " + ", ".join(missing))
        if self.config.activation_level != ActivationLevel.PROFILE_ONLY and not durable_unique_identity:
            raise PlantSourceError(
                "SOURCE_IDENTITY_UNSAFE: source row ID must have a PRIMARY KEY or "
                "single-column non-partial UNIQUE index"
            )
        if self.config.require_ordering_index and not ordering_index:
            raise PlantSourceError(
                "SOURCE_ORDERING_INDEX_REQUIRED: large-table shadow ingestion requires an index "
                f"whose first columns are ({self.config.columns.timestamp},"
                f"{self.config.columns.source_row_id})"
            )
        return {
            "status": "VALID",
            "access": "READ_ONLY",
            "columns": columns,
            "unique_identity": True,
            "ordering_index": ordering_index,
        }

    def _bootstrap_lower_bound(self, latest_timestamp: datetime) -> datetime | None:
        hours = self.config.initial_lookback_hours
        if hours is None:
            return None
        if latest_timestamp.tzinfo is None:
            raise PlantSourceError("source maximum timestamp is timezone-naive")
        return latest_timestamp - timedelta(hours=hours)

    def fetch_after(
        self,
        watermark: tuple[datetime, str] | None,
        *,
        batch_size: int | None = None,
    ) -> list[PlantObservation]:
        connection = self._connection()
        _, sql = _psycopg()
        c = self.config.columns
        operating_state = sql.Identifier(c.operating_state) if c.operating_state else sql.SQL("NULL")
        operating_state_source = (
            sql.Identifier(c.operating_state_source)
            if c.operating_state_source else sql.SQL("NULL")
        )
        operating_state_confidence = (
            sql.Identifier(c.operating_state_confidence)
            if c.operating_state_confidence else sql.SQL("NULL")
        )
        maintenance_event_id = (
            sql.Identifier(c.maintenance_event_id)
            if c.maintenance_event_id else sql.SQL("NULL")
        )
        where = sql.SQL("")
        order = sql.SQL("ORDER BY {ts} ASC,{row_id} ASC").format(
            ts=sql.Identifier(c.timestamp),
            row_id=sql.Identifier(c.source_row_id),
        )
        params: list[Any] = []
        if watermark is not None:
            timestamp, source_row_id = watermark
            lower_bound = timestamp - timedelta(seconds=self.config.lateness_seconds)
            where = sql.SQL(
                "WHERE {ts} >= %s "
                "AND ({ts} < %s OR {ts} > %s OR ({ts} = %s AND {row_id} > %s))"
            ).format(
                ts=sql.Identifier(c.timestamp),
                row_id=sql.Identifier(c.source_row_id),
            )
            params.extend([lower_bound, timestamp, timestamp, timestamp, self._typed_row_id(source_row_id)])
            order = sql.SQL(
                "ORDER BY ({ts} > %s OR ({ts} = %s AND {row_id} > %s)) DESC,"
                "{ts} ASC,{row_id} ASC"
            ).format(
                ts=sql.Identifier(c.timestamp),
                row_id=sql.Identifier(c.source_row_id),
            )
            params.extend([timestamp, timestamp, self._typed_row_id(source_row_id)])
        elif self.config.initial_lookback_hours is not None:
            latest_query = sql.SQL("SELECT MAX({ts}) FROM {schema}.{table}").format(
                ts=sql.Identifier(c.timestamp),
                schema=sql.Identifier(self.config.schema),
                table=sql.Identifier(self.config.table),
            )
            with connection.cursor() as cursor:
                cursor.execute(latest_query)
                latest_timestamp = cursor.fetchone()[0]
            if latest_timestamp is None:
                self.bootstrap_window = {
                    "mode": "SOURCE_TAIL",
                    "lookback_hours": float(self.config.initial_lookback_hours),
                    "status": "SOURCE_EMPTY",
                }
                return []
            lower_bound = self._bootstrap_lower_bound(latest_timestamp)
            if lower_bound is None:  # pragma: no cover - guarded by branch
                raise AssertionError("bounded bootstrap unexpectedly produced no lower bound")
            where = sql.SQL("WHERE {ts} >= %s").format(ts=sql.Identifier(c.timestamp))
            params.append(lower_bound)
            self.bootstrap_window = {
                "mode": "SOURCE_TAIL",
                "lookback_hours": float(self.config.initial_lookback_hours),
                "latest_source_timestamp": latest_timestamp.isoformat(),
                "lower_bound_timestamp": lower_bound.isoformat(),
            }
        query = sql.SQL(
            "SELECT {row_id},{ts},{line},{machine},{vrms},{arms},{apeak},{crest},{temp},"
            "{operating_state},{operating_state_source},{operating_state_confidence},"
            "{maintenance_event_id} "
            "FROM {schema}.{table} {where} {order} LIMIT %s"
        ).format(
            row_id=sql.Identifier(c.source_row_id),
            ts=sql.Identifier(c.timestamp),
            line=sql.Identifier(c.line_sel),
            machine=sql.Identifier(c.machine_id),
            vrms=sql.Identifier(c.vrms),
            arms=sql.Identifier(c.arms),
            apeak=sql.Identifier(c.apeak),
            crest=sql.Identifier(c.crest),
            temp=sql.Identifier(c.temp),
            operating_state=operating_state,
            operating_state_source=operating_state_source,
            operating_state_confidence=operating_state_confidence,
            maintenance_event_id=maintenance_event_id,
            schema=sql.Identifier(self.config.schema),
            table=sql.Identifier(self.config.table),
            where=where,
            order=order,
        )
        params.append(int(batch_size or self.config.batch_size))
        with connection.cursor() as cursor:
            cursor.execute(query, tuple(params))
            return [self._observation(row) for row in cursor.fetchall()]

    def _typed_row_id(self, value: str) -> int | str | UUID:
        if self.config.row_id_kind == "integer":
            return int(value)
        if self.config.row_id_kind == "uuid":
            return UUID(value)
        return value

    def _observation(self, row: tuple[Any, ...]) -> PlantObservation:
        keys = (
            "source_row_id", "timestamp", "line_sel", "machine_id", "vrms", "arms",
            "apeak", "crest", "temp", "operating_state", "operating_state_source",
            "operating_state_confidence", "maintenance_event_id",
        )
        raw = dict(zip(keys, row))
        timestamp = raw["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        if not isinstance(timestamp, datetime):
            raise PlantSourceError("source timestamp is not parseable")
        if timestamp.tzinfo is None:
            raise PlantSourceError("source timestamp is timezone-naive; configure an explicit source timezone")
        state_value = raw.get("operating_state")
        if state_value is None:
            operating_state = OperatingState.UNKNOWN
            state_source = OperatingStateSource.UNAVAILABLE
            confidence = 0.0
        else:
            operating_state = OperatingState(str(state_value).strip().upper())
            source_value = raw.get("operating_state_source")
            state_source = (
                OperatingStateSource(str(source_value).strip().upper())
                if source_value is not None
                else OperatingStateSource.DATABASE
            )
            confidence_value = raw.get("operating_state_confidence")
            confidence = float(confidence_value) if confidence_value is not None else 1.0
        raw_payload = {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in raw.items()
        }
        return PlantObservation(
            source_key=self.config.source_key,
            source_row_id=str(raw["source_row_id"]),
            event_timestamp=timestamp.astimezone(timezone.utc),
            line_sel=str(raw["line_sel"]),
            machine_id=str(raw["machine_id"]),
            vrms=float(raw["vrms"]),
            arms=float(raw["arms"]),
            apeak=float(raw["apeak"]),
            crest=float(raw["crest"]),
            temp=float(raw["temp"]),
            raw_payload=raw_payload,
            operating_state=operating_state,
            operating_state_source=state_source,
            operating_state_confidence=confidence,
            maintenance_event_id=(
                str(raw["maintenance_event_id"])
                if raw.get("maintenance_event_id") is not None else None
            ),
        )


def load_source_configs(path: str) -> list[PlantPostgresConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload if isinstance(payload, list) else payload.get("sources", [])
    if not isinstance(values, list):
        raise ValueError("source configuration must contain a list")
    configs = [PlantPostgresConfig.from_dict(dict(item)) for item in values]
    keys = [item.source_key for item in configs]
    if len(keys) != len(set(keys)):
        raise ValueError("source_key values must be unique")
    return configs
