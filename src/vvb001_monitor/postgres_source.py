from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from .config import PostgresConfig
from .models import SourceRecord, VVB001Reading


class PostgresSourceError(RuntimeError):
    pass


def _load_psycopg():
    try:
        import psycopg  # type: ignore
        from psycopg import sql  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise PostgresSourceError(
            "PostgreSQL support requires psycopg. Run: python -m pip install -r requirements.txt"
        ) from exc
    return psycopg, sql


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


class PostgresVVB001Source:
    """Read-only PostgreSQL adapter for VVB001 process values."""

    def __init__(self, config: PostgresConfig) -> None:
        self.config = config
        self.connection = None

    def connect(self) -> None:
        psycopg, _ = _load_psycopg()
        self.connection = psycopg.connect(self.config.resolved_dsn(), autocommit=True)
        with self.connection.cursor() as cursor:
            cursor.execute("SET default_transaction_read_only = on")
            cursor.execute("SHOW default_transaction_read_only")
            value = str(cursor.fetchone()[0]).lower()
            if value not in {"on", "true", "1"}:
                raise PostgresSourceError("Could not enforce PostgreSQL read-only session mode")

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _ensure_connection(self) -> None:
        if self.connection is None or getattr(self.connection, "closed", False):
            self.connect()

    def _select_sql(self, *, where_sql: str, order_sql: str = "ORDER BY {id} ASC", limit: bool = True):
        _, sql = _load_psycopg()
        c = self.config.columns
        query = sql.SQL(
            "SELECT "
            "{id} AS source_id, {timestamp} AS timestamp, {line} AS line_sel, "
            "{machine} AS machine_id, {vrms} AS vrms, {arms} AS arms, "
            "{apeak} AS apeak, {crest} AS crest, {temp} AS temp "
            "FROM {schema}.{table} " + where_sql + " " + order_sql + (" LIMIT %s" if limit else "")
        ).format(
            id=sql.Identifier(c.source_id), timestamp=sql.Identifier(c.timestamp),
            line=sql.Identifier(c.line_sel), machine=sql.Identifier(c.machine_id),
            vrms=sql.Identifier(c.vrms), arms=sql.Identifier(c.arms),
            apeak=sql.Identifier(c.apeak), crest=sql.Identifier(c.crest),
            temp=sql.Identifier(c.temp), schema=sql.Identifier(self.config.schema),
            table=sql.Identifier(self.config.table),
        )
        return query

    @staticmethod
    def _to_record(row: tuple[Any, ...]) -> SourceRecord:
        keys = ("source_id", "timestamp", "line_sel", "machine_id", "vrms", "arms", "apeak", "crest", "temp")
        raw = dict(zip(keys, row))
        source_id = -1
        try:
            source_id = int(raw["source_id"])
            reading = VVB001Reading(
                source_id=source_id,
                timestamp=_parse_timestamp(raw["timestamp"]),
                line_sel=str(raw["line_sel"]),
                machine_id=str(raw["machine_id"]),
                vrms=float(raw["vrms"]),
                arms=float(raw["arms"]),
                apeak=float(raw["apeak"]),
                crest=float(raw["crest"]),
                temp=float(raw["temp"]),
            )
            return SourceRecord(source_id, raw, reading)
        except Exception as exc:
            return SourceRecord(source_id, raw, None, f"could not parse row: {exc}")

    def fetch_after(self, last_source_id: int | None, batch_size: int | None = None) -> list[SourceRecord]:
        self._ensure_connection()
        batch_size = int(batch_size or self.config.batch_size)
        if last_source_id is None:
            query = self._select_sql(where_sql="", limit=True)
            params = (batch_size,)
        else:
            query = self._select_sql(where_sql="WHERE {id} > %s", limit=True)
            params = (last_source_id, batch_size)
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            return [self._to_record(row) for row in cursor.fetchall()]

    def fetch_history_before(self, last_source_id: int, start_timestamp: datetime) -> list[SourceRecord]:
        self._ensure_connection()
        _, sql = _load_psycopg()
        c = self.config.columns
        query = sql.SQL(
            "SELECT {id} AS source_id, {timestamp} AS timestamp, {line} AS line_sel, "
            "{machine} AS machine_id, {vrms} AS vrms, {arms} AS arms, {apeak} AS apeak, "
            "{crest} AS crest, {temp} AS temp FROM {schema}.{table} "
            "WHERE {id} <= %s AND {timestamp} >= %s ORDER BY {id} ASC"
        ).format(
            id=sql.Identifier(c.source_id), timestamp=sql.Identifier(c.timestamp),
            line=sql.Identifier(c.line_sel), machine=sql.Identifier(c.machine_id),
            vrms=sql.Identifier(c.vrms), arms=sql.Identifier(c.arms), apeak=sql.Identifier(c.apeak),
            crest=sql.Identifier(c.crest), temp=sql.Identifier(c.temp),
            schema=sql.Identifier(self.config.schema), table=sql.Identifier(self.config.table),
        )
        with self.connection.cursor() as cursor:
            cursor.execute(query, (last_source_id, start_timestamp))
            return [self._to_record(row) for row in cursor.fetchall()]

    def fetch_recent(self, hours: float) -> list[SourceRecord]:
        self._ensure_connection()
        _, sql = _load_psycopg()
        c = self.config.columns
        query = sql.SQL(
            "SELECT {id} AS source_id, {timestamp} AS timestamp, {line} AS line_sel, "
            "{machine} AS machine_id, {vrms} AS vrms, {arms} AS arms, {apeak} AS apeak, "
            "{crest} AS crest, {temp} AS temp FROM {schema}.{table} "
            "WHERE {timestamp} >= NOW() - (%s * INTERVAL '1 hour') ORDER BY {id} ASC"
        ).format(
            id=sql.Identifier(c.source_id), timestamp=sql.Identifier(c.timestamp),
            line=sql.Identifier(c.line_sel), machine=sql.Identifier(c.machine_id),
            vrms=sql.Identifier(c.vrms), arms=sql.Identifier(c.arms), apeak=sql.Identifier(c.apeak),
            crest=sql.Identifier(c.crest), temp=sql.Identifier(c.temp),
            schema=sql.Identifier(self.config.schema), table=sql.Identifier(self.config.table),
        )
        with self.connection.cursor() as cursor:
            cursor.execute(query, (float(hours),))
            return [self._to_record(row) for row in cursor.fetchall()]

    def reconnect_with_backoff(self, delay_seconds: float = 2.0) -> None:
        self.close()
        time.sleep(max(0.0, delay_seconds))
        self.connect()
