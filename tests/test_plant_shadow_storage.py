from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from vvb001_monitor.plant_shadow.contracts import (
    EndpointClass,
    EndpointPrecision,
    ObservationDisposition,
    OperatingState,
    OperatingStateSource,
    PlantObservation,
)
from vvb001_monitor.plant_shadow.storage import EvidenceStore, ProcessingCommit


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def observation(row_id: str, hour: int = 0, *, source: str = "source-a") -> PlantObservation:
    return PlantObservation(
        source, row_id, BASE + timedelta(hours=hour), "LINE", "M1",
        1.0, 2.0, 3.0, 1.5, 30.0,
        raw_payload={"id": row_id},
        operating_state=OperatingState.RUNNING,
        operating_state_source=OperatingStateSource.SYNTHETIC_FIXTURE,
        operating_state_confidence=1.0,
    )


def prediction(point: float = 5.0) -> dict[str, object]:
    return {
        "deployment_id": "deployment-test",
        "prediction_timestamp": BASE.isoformat(),
        "health_state_model": "NORMAL",
        "health_state_source": "MODEL",
        "connectivity_state": "ONLINE",
        "forecast_state": "AVAILABLE",
        "warning_point_hours": point,
        "warning_lower_hours": point - 1,
        "warning_upper_hours": point + 1,
        "warning_serviceable": True,
        "critical_point_hours": point + 5,
        "critical_lower_hours": point + 4,
        "critical_upper_hours": point + 6,
        "critical_serviceable": True,
        "model_version": "v2.7",
        "model_artifact_sha256": "a" * 64,
        "runtime_manifest_id": "manifest-test",
        "inference_latency_ms": 1.0,
    }


def test_atomic_processing_is_idempotent_and_append_only(tmp_path):
    store = EvidenceStore(tmp_path / "shadow.db")
    calls = 0

    def processor(_ingestion_id, _lifecycle_id):
        nonlocal calls
        calls += 1
        return ProcessingCommit(ObservationDisposition.PROCESSED, "VALID", prediction=prediction())

    first = store.process_observation(observation("1"), processor)
    duplicate = store.process_observation(observation("1"), processor)
    assert calls == 1
    assert first.prediction_id
    assert duplicate.duplicate is True
    assert store.overview() == {
        "sources": 1,
        "machines": 1,
        "machines_running": 1,
        "predictions": 1,
        "lifecycles": 1,
        "plant_production_authorized": False,
    }
    with pytest.raises(sqlite3.IntegrityError, match="append-only evidence"):
        store.db.execute("UPDATE prediction_attempts SET model_version='changed'")
    store.close()


def test_late_row_is_preserved_but_not_injected_into_runtime(tmp_path):
    store = EvidenceStore(tmp_path / "shadow.db")
    calls = []
    processor = lambda ingestion_id, lifecycle_id: (
        calls.append(ingestion_id)
        or ProcessingCommit(ObservationDisposition.PROCESSED, "VALID", prediction=prediction())
    )
    store.process_observation(observation("2", 2), processor)
    result = store.process_observation(observation("1", 1), processor)
    assert result.disposition == ObservationDisposition.LATE_QUARANTINED
    assert len(calls) == 1
    assert store.db.execute("SELECT COUNT(*) FROM raw_observations").fetchone()[0] == 2
    store.close()


def test_equal_timestamp_integer_row_ids_use_numeric_watermark_order(tmp_path):
    store = EvidenceStore(tmp_path / "shadow.db")
    store.save_source_config(
        "source-a",
        {"row_id_kind": "integer"},
        actor="test",
        enabled=True,
    )
    calls = []
    processor = lambda ingestion_id, lifecycle_id: (
        calls.append(ingestion_id)
        or ProcessingCommit(ObservationDisposition.PROCESSED, "VALID", prediction=prediction())
    )

    first = store.process_observation(observation("9"), processor)
    second = store.process_observation(observation("10"), processor)
    late = store.process_observation(observation("8"), processor)

    assert first.disposition == ObservationDisposition.PROCESSED
    assert second.disposition == ObservationDisposition.PROCESSED
    assert late.disposition == ObservationDisposition.LATE_QUARANTINED
    assert len(calls) == 2
    watermark = store.db.execute(
        "SELECT source_row_id FROM source_watermarks WHERE source_key='source-a'"
    ).fetchone()
    assert watermark[0] == "10"
    store.close()


def test_rollback_removes_raw_row_and_marks_runtime_advanced(tmp_path):
    store = EvidenceStore(tmp_path / "shadow.db")

    def processor(_ingestion_id, _lifecycle_id):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError) as caught:
        store.process_observation(observation("1"), processor)
    assert caught.value.plant_shadow_runtime_advanced is True
    assert store.db.execute("SELECT COUNT(*) FROM raw_observations").fetchone()[0] == 0
    store.close()


def test_warning_truth_and_critical_censoring_are_independent(tmp_path):
    store = EvidenceStore(tmp_path / "shadow.db")
    commit = lambda ingestion_id, lifecycle_id: ProcessingCommit(
        ObservationDisposition.PROCESSED, "VALID", prediction=prediction(),
    )
    processed = store.process_observation(observation("1", 0), commit)
    warning_time = BASE + timedelta(hours=6)
    evidence = store.add_endpoint_evidence(
        source_system="cmms",
        external_event_id="warning-1",
        machine_uid="source-a::LINE::M1",
        lifecycle_id=processed.lifecycle_id,
        precision=EndpointPrecision.EXACT_TIMESTAMP,
        event_time_lower=warning_time,
        event_time_upper=warning_time,
        details={"kind": "independent alarm"},
        actor="engineer",
    )
    result = store.confirm_endpoint_classification(evidence, EndpointClass.WARNING_ONSET, actor="engineer")
    assert result["truth_rows_created"] == 1
    truth = store.db.execute("SELECT target,true_hours FROM target_truth").fetchone()
    assert tuple(truth) == ("WARNING", 6.0)

    maintenance_time = BASE + timedelta(hours=8)
    maintenance = store.add_endpoint_evidence(
        source_system="cmms",
        external_event_id="maintenance-1",
        machine_uid="source-a::LINE::M1",
        lifecycle_id=processed.lifecycle_id,
        precision=EndpointPrecision.EXACT_TIMESTAMP,
        event_time_lower=maintenance_time,
        event_time_upper=maintenance_time,
        details={"kind": "preventive replacement"},
        actor="engineer",
    )
    store.confirm_endpoint_classification(maintenance, EndpointClass.PREVENTIVE_MAINTENANCE, actor="engineer")
    critical = store.db.execute(
        "SELECT eligibility,censoring FROM target_truth_eligibility WHERE target='CRITICAL' ORDER BY decided_at DESC LIMIT 1"
    ).fetchone()
    assert tuple(critical) == ("CENSORED", "RIGHT_CENSORED")
    assert store.db.execute("SELECT COUNT(*) FROM target_truth WHERE target='CRITICAL'").fetchone()[0] == 0
    store.close()
