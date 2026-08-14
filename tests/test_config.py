import json
from pathlib import Path

import pytest

from vvb001_monitor.config import AppConfig, ColumnMap, SensorConfig


def test_default_column_map_matches_real_table_names():
    c = ColumnMap()
    assert (c.source_id, c.timestamp, c.line_sel, c.machine_id) == ("id", "timestamp", "line_sel", "machine_id")
    assert (c.vrms, c.arms, c.apeak, c.crest, c.temp) == ("vrms", "arms", "apeak", "crest", "temp")


def test_sensor_config_requires_supported_acceleration_unit():
    with pytest.raises(ValueError):
        SensorConfig(acceleration_unit="mg").validate()


def test_config_loads_column_names_with_spaces(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "postgres": {"columns": {"line_sel": "line sel"}},
        "sensor": {"acceleration_unit": "g"}
    }))
    cfg = AppConfig.load(path)
    assert cfg.postgres.columns.line_sel == "line sel"
    assert cfg.sensor.acceleration_unit == "g"
