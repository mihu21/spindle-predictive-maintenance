from __future__ import annotations

import pytest

from vvb001_monitor.plant_shadow.native_postgres import (
    dsn_for_database,
    validate_dedicated_database_name,
    validate_loopback_dsn,
)


def test_native_e2e_refuses_remote_admin_dsn():
    with pytest.raises(ValueError, match="loopback"):
        validate_loopback_dsn("postgresql://admin:secret@plant-db.example/vvb001")


def test_native_e2e_refuses_non_dedicated_database_name():
    for name in ("postgres", "vvb001", "production", "vvb001_e2e_bad-name"):
        with pytest.raises(ValueError, match="dedicated"):
            validate_dedicated_database_name(name)


def test_native_e2e_builds_separate_reader_dsn_on_dedicated_database():
    result = dsn_for_database(
        "postgresql://fixture_admin:admin_secret@127.0.0.1:5432/postgres",
        "vvb001_e2e_safe",
        user="vvb001_e2e_safe_reader",
        password="reader_secret",
    )
    values = validate_loopback_dsn(result)
    assert values["dbname"] == "vvb001_e2e_safe"
    assert values["user"] == "vvb001_e2e_safe_reader"
