from pathlib import Path


def test_postgres_runtime_contains_no_write_sql():
    text = (Path(__file__).parents[1] / "src" / "vvb001_monitor" / "postgres_source.py").read_text().upper()
    forbidden = ("INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE", "ALTER TABLE", "DROP TABLE", "TRUNCATE")
    for phrase in forbidden:
        assert phrase not in text
    assert "DEFAULT_TRANSACTION_READ_ONLY" in text
    assert "SELECT " in text


def test_plant_shadow_postgres_connector_contains_no_write_sql():
    text = (Path(__file__).parents[1] / "src" / "vvb001_monitor" / "plant_shadow" / "source.py").read_text().upper()
    forbidden = ("INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE", "ALTER TABLE", "DROP TABLE", "TRUNCATE")
    for phrase in forbidden:
        assert phrase not in text
    assert "DEFAULT_TRANSACTION_READ_ONLY" in text
    assert "SELECT " in text
