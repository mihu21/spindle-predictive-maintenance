from __future__ import annotations

import pytest

from datetime import datetime, timedelta, timezone

from vvb001_monitor.plant_shadow.contracts import OperatingState, OperatingStateSource
from vvb001_monitor.plant_shadow.source import PlantPostgresConfig, PlantPostgresSource


def test_source_config_is_redacted_and_requires_environment_secret(monkeypatch):
    config = PlantPostgresConfig.from_dict({
        "source_key": "source-a",
        "display_name": "Source A",
        "dsn_env": "PLANT_SOURCE_A_DSN",
    })
    assert "password" not in str(config.redacted_dict()).lower()
    monkeypatch.delenv("PLANT_SOURCE_A_DSN", raising=False)
    with pytest.raises(RuntimeError, match="secret environment variable"):
        config.resolved_dsn()


def test_source_config_rejects_ctid():
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_UNSAFE"):
        PlantPostgresConfig.from_dict({
            "source_key": "source-a",
            "display_name": "Source A",
            "dsn_env": "PLANT_SOURCE_A_DSN",
            "columns": {"source_row_id": "ctid"},
        })


def test_source_watermark_row_id_is_restored_to_declared_database_type():
    integer = PlantPostgresConfig.from_dict({
        "source_key": "source-a", "display_name": "A", "dsn_env": "DSN", "row_id_kind": "integer",
    })
    from vvb001_monitor.plant_shadow.source import PlantPostgresSource
    assert PlantPostgresSource(integer)._typed_row_id("42") == 42
    text = PlantPostgresConfig.from_dict({
        "source_key": "source-b", "display_name": "B", "dsn_env": "DSN", "row_id_kind": "text",
    })
    assert PlantPostgresSource(text)._typed_row_id("00042") == "00042"


def test_source_config_rejects_negative_lateness_window():
    with pytest.raises(ValueError, match="lateness_seconds"):
        PlantPostgresConfig.from_dict({
            "source_key": "source-a",
            "display_name": "A",
            "dsn_env": "DSN",
            "lateness_seconds": -1,
        })


def test_source_config_bounds_direct_database_bootstrap_and_query_time():
    config = PlantPostgresConfig.from_dict({
        "source_key": "source-a",
        "display_name": "A",
        "dsn_env": "DSN",
        "initial_lookback_hours": 24,
        "statement_timeout_seconds": 45,
    })
    source = PlantPostgresSource(config)
    latest = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)
    assert source._bootstrap_lower_bound(latest) == latest - timedelta(hours=24)
    assert config.redacted_dict()["require_ordering_index"] is True
    assert config.redacted_dict()["statement_timeout_seconds"] == 45


@pytest.mark.parametrize(
    "values,match",
    [
        ({"initial_lookback_hours": 0}, "initial_lookback_hours"),
        ({"statement_timeout_seconds": 0}, "statement_timeout_seconds"),
    ],
)
def test_source_config_rejects_unbounded_runtime_values(values, match):
    with pytest.raises(ValueError, match=match):
        PlantPostgresConfig.from_dict({
            "source_key": "source-a", "display_name": "A", "dsn_env": "DSN", **values,
        })


def test_source_maps_explicit_operating_context_and_defaults_missing_context_to_unknown():
    config = PlantPostgresConfig.from_dict({
        "source_key": "source-a",
        "display_name": "A",
        "dsn_env": "DSN",
    })
    source = PlantPostgresSource(config)
    base = (1, datetime(2026, 1, 1, tzinfo=timezone.utc), "L", "M", 1, 2, 3, 1.5, 30)
    unknown = source._observation((*base, None, None, None, None))
    assert unknown.operating_state == OperatingState.UNKNOWN
    assert unknown.operating_state_source == OperatingStateSource.UNAVAILABLE
    assert unknown.operating_state_confidence == 0.0

    running = source._observation((*base, "RUNNING", "PLC", 0.98, None))
    assert running.operating_state == OperatingState.RUNNING
    assert running.operating_state_source == OperatingStateSource.PLC
    assert running.operating_state_confidence == 0.98
