from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from spindle_monitor.config import load_config
from spindle_monitor.ml_forecaster import MLForecaster
from spindle_monitor.model_registry import TARGET_FILES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_verification(domain: str, metadata_boundaries: set[str] | None = None) -> dict:
    csv_path = ROOT / "output" / domain / "replay.csv"
    database_path = ROOT / "output" / domain / "monitor.db"
    states: Counter[str] = Counter()
    first: dict[str, str] | None = None
    row_count = 0
    boundary_states: Counter[str] = Counter()
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            first = first or row
            row_count += 1
            states[row["lifecycle_state"]] += 1
            if metadata_boundaries and row["timestamp"] in metadata_boundaries:
                boundary_states[row["lifecycle_state"]] += 1
    connection = sqlite3.connect(database_path)
    try:
        database_rows = connection.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        database_states = dict(connection.execute(
            "SELECT lifecycle_state, COUNT(*) FROM readings GROUP BY lifecycle_state"
        ))
        database_first = connection.execute(
            "SELECT timestamp, source_sampling_interval_seconds, sampling_gap_seconds, lifecycle_state "
            "FROM readings ORDER BY id LIMIT 1"
        ).fetchone()
        lifecycle_count = connection.execute("SELECT COUNT(*) FROM lifecycles").fetchone()[0]
    finally:
        connection.close()
    return {
        "csv_rows": row_count,
        "sqlite_rows": database_rows,
        "csv_sqlite_row_count_match": row_count == database_rows,
        "csv_states": dict(states),
        "sqlite_states": database_states,
        "csv_sqlite_state_counts_match": dict(states) == database_states,
        "first_csv": {
            "timestamp": first["timestamp"] if first else None,
            "source_sampling_interval_seconds": first["source_sampling_interval_seconds"] if first else None,
            "lifecycle_state": first["lifecycle_state"] if first else None,
        },
        "first_sqlite": {
            "timestamp": database_first[0],
            "source_sampling_interval_seconds": database_first[1],
            "sampling_gap_seconds": database_first[2],
            "lifecycle_state": database_first[3],
        },
        "completed_lifecycles": lifecycle_count,
        "metadata_boundary_count": len(metadata_boundaries or ()),
        "boundary_state_counts": dict(boundary_states),
    }


def main() -> None:
    metadata = json.loads((ROOT / "data" / "realistic_spindle_mock_metadata.json").read_text(encoding="utf-8"))
    boundaries = {item["maintenance_record_timestamp"] for item in metadata["lifecycles"]}
    hash_evidence = json.loads((ROOT / "output" / "model_environment_update.json").read_text(encoding="utf-8"))
    models: dict[str, dict] = {}
    config = load_config(ROOT / "config" / "thresholds.json")
    for domain in ("mock", "realistic"):
        candidate = ROOT / "models" / domain / "candidate"
        model_metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forecaster = MLForecaster(config, model_metadata["feature_names"], ROOT / "models" / domain)
            forecaster.predict_many([{name: 0.0 for name in model_metadata["feature_names"]}])
        hashes = {filename: _sha256(candidate / filename) for filename in TARGET_FILES.values()}
        expected_hashes = hash_evidence["models"][domain]["artifact_sha256_after"]
        models[domain] = {
            "loaded_targets": sorted(forecaster.models),
            "loaded_count": len(forecaster.models),
            "warning_count": len(caught),
            "warning_messages": [str(item.message) for item in caught],
            "joblib_hashes_match_pre_correction_binaries": hashes == expected_hashes,
            "feature_contains_lifecycle_state": "lifecycle_state" in model_metadata["feature_names"],
            "feature_contains_elapsed_lifecycle_hours": "elapsed_lifecycle_hours" in model_metadata["feature_names"],
        }
    profiles = {}
    for name in ("out_of_range", "impossible_jump", "invalid_status", "duplicate_timestamp"):
        value = json.loads((ROOT / "output" / "profile_cases" / f"{name}.json").read_text(encoding="utf-8"))
        profiles[name] = {
            "valid_rows": value["input_validation"]["valid_rows"],
            "invalid_rows": value["input_validation"]["invalid_rows"],
            "invalid_reason_counts": value["input_validation"]["invalid_reason_counts"],
            "replay_suitable": value["suitability"]["replay"],
        }
    promotions = {
        domain: json.loads((ROOT / "output" / domain / "promotion_refusal.json").read_text(encoding="utf-8"))
        for domain in ("mock", "realistic")
    }
    report = {
        "realistic_replay": _replay_verification("realistic", boundaries),
        "mock_replay": _replay_verification("mock"),
        "model_loading": models,
        "profile_validation_cases": profiles,
        "promotion_refusals": {
            domain: {
                "status": value["status"],
                "promoted": value["promoted"],
                "reason": value["promotion_refusal_reason"],
                "environment_compatible": value["environment_compatibility"]["compatible"],
            }
            for domain, value in promotions.items()
        },
        "no_retraining_performed_for_final_correction": True,
        "anomaly_detection_added": True,
        "anomaly_evaluation": json.loads(
            (ROOT / "output" / "anomaly_evaluation.json").read_text(encoding="utf-8")
        ),
        "anomaly_defaults_plant_validated": False,
        "model_retraining_required_for_anomaly_layer": False,
        "anomaly_correction_verification": json.loads(
            (ROOT / "output" / "anomaly_correction_verification.json").read_text(encoding="utf-8")
        ),
    }
    output = ROOT / "output" / "final_verification.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
