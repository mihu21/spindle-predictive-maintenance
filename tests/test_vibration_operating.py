from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from vvb001_monitor.config import AppConfig
from vvb001_monitor.plant_shadow.api import create_app
from vvb001_monitor.plant_shadow.contracts import (
    ForecastState,
    ObservationDisposition,
    OperatingState,
    OperatingStateSource,
    PlantObservation,
)
from vvb001_monitor.plant_shadow.runtime import FrozenRuntimeRouter
from vvb001_monitor.plant_shadow.service import PlantShadowService
from vvb001_monitor.plant_shadow.storage import EvidenceStore
from vvb001_monitor.plant_shadow.vibration_operating import (
    VibrationOperatingConfig,
    VibrationOperatingDetector,
)


BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)


def config() -> VibrationOperatingConfig:
    return VibrationOperatingConfig(
        short_window_seconds=20.0,
        calibration_history_seconds=3_600.0,
        calibration_min_samples=40,
        calibration_min_span_seconds=120.0,
        labeled_min_samples_per_class=3,
        activation_seconds=20.0,
        activation_confidence=0.95,
    )


def observation(
    row_id: int,
    seconds: float,
    energy: str,
    *,
    state: OperatingState = OperatingState.UNKNOWN,
    source: OperatingStateSource = OperatingStateSource.UNAVAILABLE,
) -> PlantObservation:
    if energy == "quiet":
        vrms, arms, crest = 0.05, 0.04, 3.0
    elif energy == "running":
        vrms, arms, crest = 2.0, 2.0, 3.0
    elif energy == "impulsive":
        vrms, arms, crest = 39.0, 20.0, 3.0
    else:
        raise ValueError(energy)
    return PlantObservation(
        source_key="source-a",
        source_row_id=str(row_id),
        event_timestamp=BASE + timedelta(seconds=seconds),
        line_sel="LINE",
        machine_id="M1",
        vrms=vrms,
        arms=arms,
        apeak=arms * crest,
        crest=crest,
        temp=31.0,
        operating_state=state,
        operating_state_source=source,
        operating_state_confidence=1.0 if source != OperatingStateSource.UNAVAILABLE else 0.0,
    )


def calibrate_unsupervised(detector: VibrationOperatingDetector) -> tuple[int, float]:
    row_id = 0
    seconds = 0.0
    for _ in range(30):
        row_id += 1
        detector.observe(observation(row_id, seconds, "quiet"))
        seconds += 10.0
    for _ in range(30):
        row_id += 1
        detector.observe(observation(row_id, seconds, "running"))
        seconds += 10.0
    return row_id, seconds


def test_unsupervised_bimodal_calibration_confirms_only_persistent_running():
    detector = VibrationOperatingDetector(config())
    row_id, seconds = calibrate_unsupervised(detector)

    decisions = []
    for _ in range(4):
        row_id += 1
        resolved, decision = detector.observe(observation(row_id, seconds, "running"))
        decisions.append((resolved, decision))
        seconds += 10.0

    resolved, decision = decisions[-1]
    assert decision.calibration_state == "UNSUPERVISED_BIMODAL"
    assert decision.calibration_artifact_sha256
    assert decision.classification == "RUNNING_CONFIRMED"
    assert decision.confidence >= 0.95
    assert resolved.operating_state == OperatingState.RUNNING
    assert resolved.operating_state_source == OperatingStateSource.VIBRATION_INFERENCE
    assert resolved.confirmed_running


def test_quiet_and_impulsive_vibration_fail_closed_after_calibration():
    detector = VibrationOperatingDetector(config())
    row_id, seconds = calibrate_unsupervised(detector)

    row_id += 1
    quiet, quiet_decision = detector.observe(observation(row_id, seconds, "quiet"))
    assert quiet.operating_state == OperatingState.UNKNOWN
    assert quiet.operating_state_source == OperatingStateSource.VIBRATION_INFERENCE
    assert quiet_decision.classification in {"QUIET_NOT_RUNNING", "UNKNOWN_NOVEL"}

    row_id += 1
    impulsive, impulsive_decision = detector.observe(
        observation(row_id, seconds + 10.0, "impulsive")
    )
    assert impulsive.operating_state == OperatingState.UNKNOWN
    assert impulsive_decision.classification == "UNKNOWN_IMPULSIVE"
    assert impulsive_decision.reason_code == "VIBRATION_IMPULSIVE_OR_EXTREME"


def test_authoritative_conflict_cannot_be_overridden_by_running_vibration():
    detector = VibrationOperatingDetector(config())
    row_id, seconds = calibrate_unsupervised(detector)
    conflicted = replace(
        observation(row_id + 1, seconds, "running"),
        operating_context_block_reason="CONTRADICTORY_LOCAL_OPERATING_EVIDENCE",
    )
    resolved, decision = detector.observe(conflicted)
    assert resolved.operating_state == OperatingState.UNKNOWN
    assert resolved.operating_state_source == OperatingStateSource.UNAVAILABLE
    assert decision.classification == "AUTHORITATIVE_BLOCKED"
    assert decision.reason_code == "CONTRADICTORY_LOCAL_OPERATING_EVIDENCE"


def test_authoritative_context_wins_and_can_seed_commissioning_calibration():
    detector = VibrationOperatingDetector(config())
    row_id = 0
    for state, level in (
        (OperatingState.OFF, "quiet"),
        (OperatingState.RUNNING, "running"),
    ):
        for _ in range(3):
            row_id += 1
            resolved, decision = detector.observe(
                observation(
                    row_id,
                    row_id * 10.0,
                    level,
                    state=state,
                    source=OperatingStateSource.OPERATOR,
                )
            )
            assert resolved.operating_state == state
            assert resolved.operating_state_source == OperatingStateSource.OPERATOR
            assert decision.classification == f"AUTHORITATIVE_{state.value}"

    final = None
    for _ in range(5):
        row_id += 1
        final = detector.observe(observation(row_id, row_id * 10.0, "running"))
    assert final is not None
    resolved, decision = final
    assert decision.calibration_state == "COMMISSIONED_LABELS"
    assert resolved.operating_state == OperatingState.RUNNING


def test_single_regime_never_manufactures_a_running_calibration():
    detector = VibrationOperatingDetector(config())
    final = None
    for index in range(80):
        final = detector.observe(observation(index + 1, index * 10.0, "quiet"))
    assert final is not None
    resolved, decision = final
    assert resolved.operating_state == OperatingState.UNKNOWN
    assert decision.classification == "CALIBRATING"
    assert decision.calibration_artifact_sha256 is None


def test_default_calibration_supports_ten_minute_plant_cadence():
    detector = VibrationOperatingDetector()
    row_id = 0
    seconds = 0.0
    final = None
    for level, count in (("quiet", 40), ("running", 44)):
        for _ in range(count):
            row_id += 1
            final = detector.observe(observation(row_id, seconds, level))
            seconds += 600.0
    assert final is not None
    resolved, decision = final
    assert decision.calibration_state == "UNSUPERVISED_BIMODAL"
    assert decision.classification == "RUNNING_CONFIRMED"
    assert resolved.confirmed_running


def test_service_persists_inference_pauses_preproduction_and_rebuilds_deterministically(tmp_path):
    database = tmp_path / "shadow.db"
    sensor = AppConfig.load("config/vvb001.json").sensor
    router = FrozenRuntimeRouter(
        "models/rul_v2_7_full_cadence.joblib",
        sensor,
        deployment_id="vibration-operating-test",
        runtime_manifest_id="vibration-operating-test",
        vibration_operating_config=config(),
    )
    with EvidenceStore(database) as store:
        service = PlantShadowService(store, router)
        row_id = 0
        seconds = 0.0
        results = []
        for level, count in (("quiet", 30), ("running", 34)):
            for _ in range(count):
                row_id += 1
                results.append(service.process(observation(row_id, seconds, level)))
                seconds += 10.0

        assert results[0].disposition == ObservationDisposition.HELD_OPERATING_CONTEXT
        assert any(result.lifecycle_id is not None for result in results)
        assert store.db.execute("SELECT COUNT(*) FROM vibration_operating_inferences").fetchone()[0] == 64
        assert store.db.execute("SELECT COUNT(*) FROM lifecycle_records").fetchone()[0] == 1
        earliest = store.db.execute(
            "SELECT forecast_state,warning_point_hours,critical_point_hours "
            "FROM prediction_attempts ORDER BY ingestion_id LIMIT 1"
        ).fetchone()
        assert tuple(earliest) == (ForecastState.PAUSED, None, None)

        before = store.db.execute("SELECT COUNT(*) FROM vibration_operating_inferences").fetchone()[0]
        duplicate = service.process(observation(row_id, seconds - 10.0, "running"))
        assert duplicate.duplicate
        assert store.db.execute("SELECT COUNT(*) FROM vibration_operating_inferences").fetchone()[0] == before

        rebuilt = FrozenRuntimeRouter(
            "models/rul_v2_7_full_cadence.joblib",
            sensor,
            deployment_id="vibration-operating-test",
            runtime_manifest_id="vibration-operating-test",
            vibration_operating_config=config(),
        )
        rebuilt.rebuild_source("source-a", store.committed_observations("source-a"))
        next_observation = observation(row_id + 1, seconds, "running")
        original_resolved = router.resolve_operating_context(next_observation)
        rebuilt_resolved = rebuilt.resolve_operating_context(next_observation)
        assert original_resolved.operating_state == rebuilt_resolved.operating_state
        assert original_resolved.vibration_inference == rebuilt_resolved.vibration_inference

    with TestClient(create_app(database)) as client:
        overview = client.get("/api/v1/overview").json()
        assert overview["machines_vibration_inferred_running"] == 1
        response = client.get("/api/v1/vibration-operating")
        assert response.status_code == 200
        assert response.json()["items"][0]["classification"] in {
            "RUNNING_CONFIRMED", "PRODUCTION_PENDING"
        }
