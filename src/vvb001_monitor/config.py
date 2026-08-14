from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ColumnMap:
    source_id: str = "id"
    timestamp: str = "timestamp"
    line_sel: str = "line_sel"
    machine_id: str = "machine_id"
    vrms: str = "vrms"
    arms: str = "arms"
    apeak: str = "apeak"
    crest: str = "crest"
    temp: str = "temp"

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ColumnMap":
        value = value or {}
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown column mapping keys: {sorted(unknown)}")
        result = cls(**{k: str(v) for k, v in value.items()})
        for key, item in result.__dict__.items():
            if not item.strip():
                raise ValueError(f"Column mapping {key!r} cannot be empty")
        return result


@dataclass(frozen=True)
class PostgresConfig:
    source_name: str = "vvb001"
    dsn_env: str = "VVB001_POSTGRES_DSN"
    dsn: str | None = None
    schema: str = "public"
    table: str = "vvb001_readings"
    columns: ColumnMap = field(default_factory=ColumnMap)
    poll_seconds: float = 5.0
    batch_size: int = 1000

    def resolved_dsn(self) -> str:
        if self.dsn:
            return self.dsn
        value = os.getenv(self.dsn_env, "")
        if not value:
            raise RuntimeError(
                f"Environment variable {self.dsn_env!r} is not set. "
                "Set it to a PostgreSQL DSN before starting monitoring."
            )
        return value

    @property
    def source_identity(self) -> str:
        c = self.columns
        return (
            f"{self.source_name}:{self.schema}.{self.table}:"
            f"id={c.source_id};timestamp={c.timestamp};line={c.line_sel};"
            f"machine={c.machine_id};vrms={c.vrms};arms={c.arms};"
            f"apeak={c.apeak};crest={c.crest};temp={c.temp}"
        )


@dataclass(frozen=True)
class SensorConfig:
    acceleration_unit: str = "m_s2"
    history_hours: float = 24.0
    feature_windows_minutes: tuple[int, ...] = (15, 60, 360, 1440)
    ewma_alpha: float = 0.1
    baseline_hours: float = 3.0
    baseline_min_points: int = 10
    baseline_std_floor_fraction: float = 0.03
    crest_consistency_relative_tolerance: float = 0.25
    crest_consistency_min_arms: float = 1e-9

    def validate(self) -> None:
        if self.acceleration_unit not in {"m_s2", "g"}:
            raise ValueError("acceleration_unit must be 'm_s2' or 'g'")
        if self.history_hours <= 0:
            raise ValueError("history_hours must be positive")
        if not self.feature_windows_minutes or min(self.feature_windows_minutes) <= 0:
            raise ValueError("feature_windows_minutes must contain positive values")
        if max(self.feature_windows_minutes) / 60.0 > self.history_hours:
            raise ValueError("history_hours must cover the largest feature window")
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.baseline_hours <= 0 or self.baseline_hours > self.history_hours:
            raise ValueError("baseline_hours must be positive and no greater than history_hours")
        if self.baseline_min_points < 2:
            raise ValueError("baseline_min_points must be at least 2")
        if self.baseline_std_floor_fraction <= 0:
            raise ValueError("baseline_std_floor_fraction must be positive")
        if self.crest_consistency_relative_tolerance < 0:
            raise ValueError("crest consistency tolerance cannot be negative")


@dataclass(frozen=True)
class AppConfig:
    postgres: PostgresConfig
    sensor: SensorConfig

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Config root must be a JSON object")
        allowed = {"postgres", "sensor"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        pg_raw = dict(payload.get("postgres") or {})
        columns = ColumnMap.from_dict(pg_raw.pop("columns", None))
        pg = PostgresConfig(columns=columns, **pg_raw)
        sensor_raw = dict(payload.get("sensor") or {})
        if "feature_windows_minutes" in sensor_raw:
            sensor_raw["feature_windows_minutes"] = tuple(int(x) for x in sensor_raw["feature_windows_minutes"])
        sensor = SensorConfig(**sensor_raw)
        sensor.validate()
        if pg.poll_seconds < 0:
            raise ValueError("poll_seconds cannot be negative")
        if pg.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not pg.schema.strip() or not pg.table.strip():
            raise ValueError("PostgreSQL schema/table cannot be empty")
        return cls(postgres=pg, sensor=sensor)
