from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from vvb001_monitor.plant_shadow.api import create_app
from vvb001_monitor.plant_shadow.contracts import OperatingState, OperatingStateSource
from vvb001_monitor.plant_shadow.storage import EvidenceStore


def test_api_is_read_only_and_redacts_secret_references(tmp_path):
    database = tmp_path / "shadow.db"
    with EvidenceStore(database) as store:
        store.save_source_config(
            "source-a",
            {"dsn_env": "PLANT_SECRET_DSN", "schema": "public", "table": "readings"},
            actor="engineer",
        )
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        store.add_operating_state_evidence(
            source_system="cmms",
            external_event_id="maintenance-1",
            machine_uid="source-a::LINE::M1",
            operating_state=OperatingState.MAINTENANCE,
            operating_state_source=OperatingStateSource.CMMS,
            confidence=1.0,
            effective_from=start,
            effective_to=start + timedelta(hours=2),
            maintenance_event_id="work-1",
            details={"approved": True},
            actor="engineer",
        )
    with TestClient(create_app(database)) as client:
        overview = client.get("/api/v1/overview")
        assert overview.status_code == 200
        assert overview.json()["plant_production_authorized"] is False
        sources = client.get("/api/v1/sources")
        assert sources.status_code == 200
        text = sources.text
        assert "PLANT_SECRET_DSN" not in text
        assert '"access":"READ_ONLY"' in text
        operating = client.get("/api/v1/operating-state-evidence")
        assert operating.status_code == 200
        assert operating.json()["items"][0]["operating_state"] == "MAINTENANCE"
        assert operating.json()["items"][0]["details"] == {"approved": True}
        assert client.post("/api/v1/sources", json={}).status_code == 405
    renamed = database.with_suffix(".closed.db")
    database.rename(renamed)
    assert renamed.is_file()
