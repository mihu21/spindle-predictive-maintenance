from __future__ import annotations

import json

import pytest

from vvb001_monitor.cli import build_parser
from vvb001_monitor.plant_shadow.manifest import build_manifest
from vvb001_monitor.plant_shadow.storage import EvidenceStore


def test_runtime_manifest_covers_frozen_model_and_integration_modules():
    value = build_manifest(".", deployment_id="test-deployment")
    assert value["v2_7_model_sha256"] == "ecad8f4f704129a3f0456c3c8dd47aabeb94ea6dfbdfd4c264313aea46076b9a"
    assert value["plant_production_authorized"] is False
    assert "src/vvb001_monitor/plant_shadow/service.py" in value["runtime_module_hashes"]
    assert "src/vvb001_monitor/plant_shadow/source.py" in value["runtime_module_hashes"]


def test_cli_rejects_non_local_api_host():
    parser = build_parser()
    args = parser.parse_args(["plant-shadow", "serve-api", "--host", "0.0.0.0"])
    from vvb001_monitor.plant_shadow.commands import run_plant_shadow_command
    with pytest.raises(ValueError, match="localhost"):
        run_plant_shadow_command(args)


def test_source_configuration_is_versioned_and_api_secret_reference_can_be_redacted(tmp_path):
    with EvidenceStore(tmp_path / "shadow.db") as store:
        first = store.save_source_config(
            "source-a",
            {"dsn_env": "SECRET_ENV", "schema": "public"},
            actor="engineer",
        )
        second = store.save_source_config(
            "source-a",
            {"dsn_env": "SECRET_ENV", "schema": "sensor"},
            actor="engineer",
        )
        values = store.latest_source_configs()
        assert first != second
        assert len(values) == 1
        assert values[0]["version_number"] == 2
        assert values[0]["config"]["schema"] == "sensor"
