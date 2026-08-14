from __future__ import annotations

from datetime import datetime, timezone

import pytest

from vvb001_monitor.plant_shadow.contracts import ObservationDisposition, PlantObservation
from vvb001_monitor.plant_shadow.service import PlantShadowService
from vvb001_monitor.plant_shadow.storage import EvidenceStore, ProcessingCommit


class FakeRouter:
    def __init__(self):
        self.processed = 0
        self.rebuilds: list[tuple[str, list[tuple[int, PlantObservation]]]] = []

    def process(self, observation, ingestion_id, lifecycle_id):
        self.processed += 1
        return ProcessingCommit(ObservationDisposition.PROCESSED, "VALID")

    def resolve_operating_context(self, observation):
        return observation

    def rebuild_source(self, source_key, observations):
        self.rebuilds.append((source_key, list(observations)))


def test_service_rebuilds_source_runtime_after_post_inference_commit_failure(tmp_path, monkeypatch):
    store = EvidenceStore(tmp_path / "shadow.db")
    router = FakeRouter()
    service = PlantShadowService(store, router)
    observation = PlantObservation(
        "source-a", "1", datetime(2026, 1, 1, tzinfo=timezone.utc), "L", "M",
        1, 2, 3, 1.5, 30,
    )
    monkeypatch.setattr(store, "_audit", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("commit path failed")))
    with pytest.raises(RuntimeError, match="commit path failed"):
        service.process(observation)
    assert router.processed == 1
    assert router.rebuilds == [("source-a", [])]
    assert store.db.execute("SELECT COUNT(*) FROM raw_observations").fetchone()[0] == 0
    store.close()
