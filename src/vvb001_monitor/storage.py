from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import ProcessedReading, SourceRecord, ValidationResult


class LocalStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS readings (
                source_id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                line_sel TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                vrms REAL NOT NULL,
                arms REAL NOT NULL,
                apeak REAL NOT NULL,
                crest REAL NOT NULL,
                temp REAL NOT NULL,
                quality_status TEXT NOT NULL,
                quality_reasons TEXT NOT NULL,
                features_json TEXT NOT NULL,
                predicted_status TEXT,
                prediction_probabilities_json TEXT,
                degradation_score REAL,
                prediction_policy_reason TEXT,
                prediction_state_source TEXT,
                prediction_state_held INTEGER,
                sensor_quality_status TEXT,
                sensor_quality_reasons TEXT,
                sensor_quality_held_sensors TEXT,
                sensor_quality_extreme_raw_override INTEGER,
                estimated_hours_to_warning REAL,
                warning_rul_lower_hours REAL,
                warning_rul_upper_hours REAL,
                estimated_hours_to_critical REAL,
                critical_rul_lower_hours REAL,
                critical_rul_upper_hours REAL,
                rul_reliability TEXT,
                rul_reason TEXT,
                rul_trend_score_per_hour REAL,
                rul_trend_r2 REAL,
                rul_history_hours REAL,
                rul_trusted_points INTEGER,
                rul_state_source TEXT,
                rul_forecastability_state TEXT,
                rul_forecastability_score REAL,
                rul_serviceable_intent INTEGER,
                rul_hard_eligible INTEGER,
                rul_selector_active INTEGER,
                rul_withholding_reason_code TEXT,
                rul_withholding_reasons TEXT,
                rul_support_distance REAL,
                rul_neighbor_dispersion_hours REAL,
                rul_model_disagreement_hours REAL,
                warning_rul_forecastability_state TEXT,
                warning_rul_forecastability_score REAL,
                warning_rul_serviceable_intent INTEGER,
                warning_rul_hard_eligible INTEGER,
                warning_rul_selector_active INTEGER,
                warning_rul_withholding_reason_code TEXT,
                warning_rul_withholding_reasons TEXT,
                warning_rul_raw_point_hours REAL,
                warning_rul_corrected_point_hours REAL,
                warning_rul_calibration_stratum TEXT,
                critical_rul_forecastability_state TEXT,
                critical_rul_forecastability_score REAL,
                critical_rul_serviceable_intent INTEGER,
                critical_rul_hard_eligible INTEGER,
                critical_rul_selector_active INTEGER,
                critical_rul_withholding_reason_code TEXT,
                critical_rul_withholding_reasons TEXT,
                critical_rul_raw_point_hours REAL,
                critical_rul_corrected_point_hours REAL,
                critical_rul_calibration_stratum TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_readings_machine_time
                ON readings(line_sel, machine_id, timestamp);
            CREATE TABLE IF NOT EXISTS invalid_rows (
                source_id INTEGER PRIMARY KEY,
                raw_json TEXT NOT NULL,
                reasons TEXT NOT NULL
            );
            """
        )
        cols = {row[1] for row in self.db.execute("PRAGMA table_info(readings)")}
        if "predicted_status" not in cols:
            self.db.execute("ALTER TABLE readings ADD COLUMN predicted_status TEXT")
        if "prediction_probabilities_json" not in cols:
            self.db.execute("ALTER TABLE readings ADD COLUMN prediction_probabilities_json TEXT")
        if "degradation_score" not in cols:
            self.db.execute("ALTER TABLE readings ADD COLUMN degradation_score REAL")
        if "prediction_policy_reason" not in cols:
            self.db.execute("ALTER TABLE readings ADD COLUMN prediction_policy_reason TEXT")
        if "prediction_state_source" not in cols:
            self.db.execute("ALTER TABLE readings ADD COLUMN prediction_state_source TEXT")
        if "prediction_state_held" not in cols:
            self.db.execute("ALTER TABLE readings ADD COLUMN prediction_state_held INTEGER")
        if "sensor_quality_status" not in cols:
            self.db.execute("ALTER TABLE readings ADD COLUMN sensor_quality_status TEXT")
        if "sensor_quality_reasons" not in cols:
            self.db.execute("ALTER TABLE readings ADD COLUMN sensor_quality_reasons TEXT")
        if "sensor_quality_held_sensors" not in cols:
            self.db.execute("ALTER TABLE readings ADD COLUMN sensor_quality_held_sensors TEXT")
        if "sensor_quality_extreme_raw_override" not in cols:
            self.db.execute("ALTER TABLE readings ADD COLUMN sensor_quality_extreme_raw_override INTEGER")
        for column, sql_type in (
            ("estimated_hours_to_warning", "REAL"),
            ("warning_rul_lower_hours", "REAL"),
            ("warning_rul_upper_hours", "REAL"),
            ("estimated_hours_to_critical", "REAL"),
            ("critical_rul_lower_hours", "REAL"),
            ("critical_rul_upper_hours", "REAL"),
            ("rul_reliability", "TEXT"),
            ("rul_reason", "TEXT"),
            ("rul_trend_score_per_hour", "REAL"),
            ("rul_trend_r2", "REAL"),
            ("rul_history_hours", "REAL"),
            ("rul_trusted_points", "INTEGER"),
            ("rul_state_source", "TEXT"),
            ("rul_forecastability_state", "TEXT"),
            ("rul_forecastability_score", "REAL"),
            ("rul_serviceable_intent", "INTEGER"),
            ("rul_hard_eligible", "INTEGER"),
            ("rul_selector_active", "INTEGER"),
            ("rul_withholding_reason_code", "TEXT"),
            ("rul_withholding_reasons", "TEXT"),
            ("rul_support_distance", "REAL"),
            ("rul_neighbor_dispersion_hours", "REAL"),
            ("rul_model_disagreement_hours", "REAL"),
            ("warning_rul_forecastability_state", "TEXT"),
            ("warning_rul_forecastability_score", "REAL"),
            ("warning_rul_serviceable_intent", "INTEGER"),
            ("warning_rul_hard_eligible", "INTEGER"),
            ("warning_rul_selector_active", "INTEGER"),
            ("warning_rul_withholding_reason_code", "TEXT"),
            ("warning_rul_withholding_reasons", "TEXT"),
            ("warning_rul_raw_point_hours", "REAL"),
            ("warning_rul_corrected_point_hours", "REAL"),
            ("warning_rul_calibration_stratum", "TEXT"),
            ("critical_rul_forecastability_state", "TEXT"),
            ("critical_rul_forecastability_score", "REAL"),
            ("critical_rul_serviceable_intent", "INTEGER"),
            ("critical_rul_hard_eligible", "INTEGER"),
            ("critical_rul_selector_active", "INTEGER"),
            ("critical_rul_withholding_reason_code", "TEXT"),
            ("critical_rul_withholding_reasons", "TEXT"),
            ("critical_rul_raw_point_hours", "REAL"),
            ("critical_rul_corrected_point_hours", "REAL"),
            ("critical_rul_calibration_stratum", "TEXT"),
        ):
            if column not in cols:
                self.db.execute(f"ALTER TABLE readings ADD COLUMN {column} {sql_type}")
        self.db.commit()

    def save_processed(self, item: ProcessedReading) -> None:
        r = item.reading
        self.db.execute(
            f"""
            INSERT OR REPLACE INTO readings(
                source_id,timestamp,line_sel,machine_id,vrms,arms,apeak,crest,temp,
                quality_status,quality_reasons,features_json,predicted_status,
                prediction_probabilities_json,degradation_score,prediction_policy_reason,
                prediction_state_source,prediction_state_held,sensor_quality_status,
                sensor_quality_reasons,sensor_quality_held_sensors,sensor_quality_extreme_raw_override,
                estimated_hours_to_warning,warning_rul_lower_hours,warning_rul_upper_hours,
                estimated_hours_to_critical,critical_rul_lower_hours,critical_rul_upper_hours,
                rul_reliability,rul_reason,rul_trend_score_per_hour,rul_trend_r2,
                rul_history_hours,rul_trusted_points,rul_state_source,rul_forecastability_state,
                rul_forecastability_score,rul_serviceable_intent,rul_hard_eligible,
                rul_selector_active,rul_withholding_reason_code,rul_withholding_reasons,
                rul_support_distance,rul_neighbor_dispersion_hours,rul_model_disagreement_hours,
                warning_rul_forecastability_state,warning_rul_forecastability_score,
                warning_rul_serviceable_intent,warning_rul_hard_eligible,warning_rul_selector_active,
                warning_rul_withholding_reason_code,warning_rul_withholding_reasons,
                warning_rul_raw_point_hours,warning_rul_corrected_point_hours,warning_rul_calibration_stratum,
                critical_rul_forecastability_state,critical_rul_forecastability_score,
                critical_rul_serviceable_intent,critical_rul_hard_eligible,critical_rul_selector_active,
                critical_rul_withholding_reason_code,critical_rul_withholding_reasons,
                critical_rul_raw_point_hours,critical_rul_corrected_point_hours,critical_rul_calibration_stratum
            ) VALUES({','.join('?' for _ in range(65))})
            """,
            (
                r.source_id, r.timestamp.isoformat(), r.line_sel, r.machine_id,
                r.vrms, r.arms, r.apeak, r.crest, r.temp,
                item.quality_status, json.dumps(item.quality_reasons),
                json.dumps(item.features, sort_keys=True, allow_nan=False),
                item.predicted_status,
                json.dumps(item.prediction_probabilities, sort_keys=True) if item.prediction_probabilities else None,
                item.degradation_score,
                item.prediction_policy_reason,
                item.prediction_state_source,
                int(item.prediction_state_held),
                item.sensor_quality_status,
                json.dumps(item.sensor_quality_reasons),
                json.dumps(item.sensor_quality_held_sensors),
                int(item.sensor_quality_extreme_raw_override),
                item.estimated_hours_to_warning,
                item.warning_rul_lower_hours,
                item.warning_rul_upper_hours,
                item.estimated_hours_to_critical,
                item.critical_rul_lower_hours,
                item.critical_rul_upper_hours,
                item.rul_reliability,
                item.rul_reason,
                item.rul_trend_score_per_hour,
                item.rul_trend_r2,
                item.rul_history_hours,
                item.rul_trusted_points,
                item.rul_state_source,
                item.rul_forecastability_state,
                item.rul_forecastability_score,
                int(item.rul_serviceable_intent),
                int(item.rul_hard_eligible),
                int(item.rul_selector_active),
                item.rul_withholding_reason_code,
                json.dumps(item.rul_withholding_reasons),
                item.rul_support_distance,
                item.rul_neighbor_dispersion_hours,
                item.rul_model_disagreement_hours,
                item.warning_rul_forecastability_state,
                item.warning_rul_forecastability_score,
                int(item.warning_rul_serviceable_intent),
                int(item.warning_rul_hard_eligible),
                int(item.warning_rul_selector_active),
                item.warning_rul_withholding_reason_code,
                json.dumps(item.warning_rul_withholding_reasons),
                item.warning_rul_raw_point_hours,
                item.warning_rul_corrected_point_hours,
                item.warning_rul_calibration_stratum,
                item.critical_rul_forecastability_state,
                item.critical_rul_forecastability_score,
                int(item.critical_rul_serviceable_intent),
                int(item.critical_rul_hard_eligible),
                int(item.critical_rul_selector_active),
                item.critical_rul_withholding_reason_code,
                json.dumps(item.critical_rul_withholding_reasons),
                item.critical_rul_raw_point_hours,
                item.critical_rul_corrected_point_hours,
                item.critical_rul_calibration_stratum,
            ),
        )
        self.db.commit()

    def save_invalid(self, validation: ValidationResult, record: SourceRecord) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO invalid_rows(source_id,raw_json,reasons) VALUES(?,?,?)",
            (record.source_id, json.dumps(record.raw, default=str, sort_keys=True), json.dumps(validation.reasons)),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def export_csv(database: str | Path, output: str | Path) -> int:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(database)
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM readings ORDER BY source_id ASC").fetchall()
    feature_keys: set[str] = set()
    parsed_features: list[dict[str, Any]] = []
    for row in rows:
        features = json.loads(row["features_json"])
        parsed_features.append(features)
        feature_keys.update(features)
    base_fields = [
        "source_id", "timestamp", "line_sel", "machine_id", "vrms", "arms", "apeak", "crest", "temp",
        "quality_status", "quality_reasons", "predicted_status", "prediction_probabilities_json", "degradation_score",
        "prediction_policy_reason", "prediction_state_source", "prediction_state_held",
        "sensor_quality_status", "sensor_quality_reasons", "sensor_quality_held_sensors", "sensor_quality_extreme_raw_override",
        "estimated_hours_to_warning", "warning_rul_lower_hours", "warning_rul_upper_hours",
        "estimated_hours_to_critical", "critical_rul_lower_hours", "critical_rul_upper_hours",
        "rul_reliability", "rul_reason", "rul_trend_score_per_hour", "rul_trend_r2",
        "rul_history_hours", "rul_trusted_points", "rul_state_source", "rul_forecastability_state",
        "rul_forecastability_score",
        "rul_serviceable_intent", "rul_hard_eligible", "rul_selector_active",
        "rul_withholding_reason_code", "rul_withholding_reasons", "rul_support_distance",
        "rul_neighbor_dispersion_hours", "rul_model_disagreement_hours",
        "warning_rul_forecastability_state", "warning_rul_forecastability_score",
        "warning_rul_serviceable_intent", "warning_rul_hard_eligible", "warning_rul_selector_active",
        "warning_rul_withholding_reason_code", "warning_rul_withholding_reasons",
        "warning_rul_raw_point_hours", "warning_rul_corrected_point_hours", "warning_rul_calibration_stratum",
        "critical_rul_forecastability_state", "critical_rul_forecastability_score",
        "critical_rul_serviceable_intent", "critical_rul_hard_eligible", "critical_rul_selector_active",
        "critical_rul_withholding_reason_code", "critical_rul_withholding_reasons",
        "critical_rul_raw_point_hours", "critical_rul_corrected_point_hours", "critical_rul_calibration_stratum",
    ]
    ordered_features = sorted(feature_keys)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields + ordered_features)
        writer.writeheader()
        for row, features in zip(rows, parsed_features):
            record = {key: row[key] for key in base_fields}
            record.update(features)
            writer.writerow(record)
    db.close()
    return len(rows)
