from __future__ import annotations

from vvb001_monitor.config import AppConfig
from vvb001_monitor.plant_shadow.golden import verify_golden_replay


def test_frozen_v27_golden_replay_matches_fixture():
    config = AppConfig.load("config/vvb001.json")
    result = verify_golden_replay(
        "config/plant_shadow_golden_replay.json",
        "models/rul_v2_7_full_cadence.joblib",
        config.sensor,
    )
    assert result["row_count"] == 24
