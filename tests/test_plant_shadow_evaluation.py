from __future__ import annotations

from datetime import datetime, timedelta, timezone

from vvb001_monitor.plant_shadow.contracts import (
    EndpointClass,
    EndpointPrecision,
    ObservationDisposition,
    OperatingState,
    OperatingStateSource,
    PlantObservation,
)
from vvb001_monitor.plant_shadow.evaluation import SupportPolicy, evaluate_plant
from vvb001_monitor.plant_shadow.storage import EvidenceStore, ProcessingCommit


def test_evaluation_is_support_gated_and_target_specific(tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = EvidenceStore(tmp_path / "shadow.db")
    obs = PlantObservation(
        "a", "1", base, "L", "M", 1, 2, 3, 1.5, 30,
        operating_state=OperatingState.RUNNING,
        operating_state_source=OperatingStateSource.SYNTHETIC_FIXTURE,
        operating_state_confidence=1.0,
    )
    result = store.process_observation(
        obs,
        lambda *_: ProcessingCommit(
            ObservationDisposition.PROCESSED,
            "VALID",
            prediction={
                "deployment_id": "d",
                "prediction_timestamp": base.isoformat(),
                "health_state_model": "NORMAL",
                "health_state_source": "MODEL",
                "connectivity_state": "ONLINE",
                "forecast_state": "AVAILABLE",
                "warning_point_hours": 5.0,
                "warning_lower_hours": 4.0,
                "warning_upper_hours": 7.0,
                "warning_serviceable": True,
                "critical_serviceable": False,
                "model_artifact_sha256": "a" * 64,
                "runtime_manifest_id": "m",
            },
        ),
    )
    endpoint = base + timedelta(hours=6)
    evidence = store.add_endpoint_evidence(
        source_system="truth",
        external_event_id="w1",
        machine_uid=obs.uid,
        lifecycle_id=result.lifecycle_id,
        precision=EndpointPrecision.EXACT_TIMESTAMP,
        event_time_lower=endpoint,
        event_time_upper=endpoint,
        details={},
        actor="test",
    )
    store.confirm_endpoint_classification(evidence, EndpointClass.WARNING_ONSET, actor="test")

    gated = evaluate_plant(store.db)
    assert gated["warning"]["status"] == "INSUFFICIENT_EVIDENCE"
    supported = evaluate_plant(
        store.db,
        policy=SupportPolicy(min_lifecycles=1, min_serviceable_predictions=1, min_intervals=1),
    )
    assert supported["warning"]["point_metrics"]["mae_hours"] == 1.0
    assert supported["warning"]["interval_metrics"]["coverage"] == 1.0
    assert supported["critical"]["status"] == "INSUFFICIENT_EVIDENCE"
    store.close()
