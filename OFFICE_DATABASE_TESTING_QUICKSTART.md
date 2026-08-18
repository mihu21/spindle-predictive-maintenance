# VVB001 office database testing quickstart

This transfer kit is for read-only plant-shadow testing against the office PostgreSQL sensor database. It contains no database credentials, no plant data, no Python virtual environment, and no `node_modules` directory. The frozen v2.7 model remains observational only: plant-production authorization is `false`.

## 1. Prepare the office computer

Install 64-bit Python 3.14 and a current Node.js LTS release. Then open PowerShell in the extracted
kit directory and run the automated setup:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_office_laptop.ps1
```

The setup script creates `.venv`, installs the Python dependencies using the validated
`requirements-office-lock.txt` constraints, installs the exact frontend dependency lock with
`npm ci`, tests and builds the dashboard, creates the ignored local source
template, and runs offline plant-shadow verification. It does not connect to PostgreSQL and does not
store credentials.

For a backend-only setup, use `-SkipFrontend`. The normal dashboard launcher requires the frontend
setup to have completed.

You can rerun the offline readiness checks at any time:

```powershell
.\test_office_readiness.ps1
```

Use `-Full` to run all Python tests. Test scratch data is deliberately placed inside the project,
which avoids failures on office laptops whose system temp folder is restricted.

## 2. Verify the frozen runtime before connecting

```powershell
.venv\Scripts\python.exe main.py plant-shadow verify-golden
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from vvb001_monitor.plant_shadow.manifest import verify_manifest; print(verify_manifest('.', 'output/plant_shadow/plant_shadow_runtime_manifest.json')['deployment_id'])"
.venv\Scripts\python.exe -c "from pathlib import Path; import hashlib; p=Path('models/rul_v2_7_full_cadence.joblib'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

The final command must print:

```text
ecad8f4f704129a3f0456c3c8dd47aabeb94ea6dfbdfd4c264313aea46076b9a
```

These checks are already included in both setup and `test_office_readiness.ps1`; the individual
commands remain documented for manual auditing.

## 3. Obtain a SELECT-only database account

Use a dedicated PostgreSQL account with only `CONNECT`, `USAGE` on the required schema, and `SELECT` on the sensor table. Do not use a database owner, schema owner, writer, fixture-admin, or personal administrator account for plant ingestion.

Ask the DBA to confirm an index whose leading columns are the mapped timestamp and durable row-ID columns, in that order. For example:

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS vvb001_readings_shadow_order_idx
ON public.vvb001_readings (timestamp, id);
```

The application checks this ordering index but never creates or modifies it.

## 4. Configure the source without storing its password

Copy the example to the ignored local filename:

```powershell
Copy-Item config\plant_shadow_sources.example.json config\plant_shadow_sources.local.json
```

Edit only the local copy to match the real schema, table, stable row ID, timestamp, machine identity, and five sensor columns. Keep unavailable operating-state and maintenance columns as `null`; the causal vibration-only detector will infer only persistent high-confidence `RUNNING` and will pause exact RUL otherwise.

Keep the password in the environment variable named by `dsn_env`, never in JSON:

```powershell
$env:VVB001_PLANT_SOURCE_A_DSN = "postgresql://select_only_user:REPLACE_ME@database-host/database-name"
```

For a very large table, retain the safe defaults unless there is a measured reason to change them:

- `initial_lookback_hours`: `24.0`
- `batch_size`: `1000`
- `statement_timeout_seconds`: `30.0`
- `require_ordering_index`: `true`

## 5. Run a bounded first connection

The safest automated first connection performs the same redacted source save, exactly one bounded
ingestion cycle, and a local status read:

```powershell
.\test_office_readiness.ps1 -IncludeDatabase -Actor "your-name"
```

It refuses to connect if a configured `dsn_env` variable is missing and never prints the secret.
Alternatively, run the equivalent commands individually:

```powershell
.venv\Scripts\python.exe main.py plant-shadow source-save `
    --sources config\plant_shadow_sources.local.json `
    --actor "your-name"

.venv\Scripts\python.exe main.py plant-shadow ingest `
    --sources config\plant_shadow_sources.local.json `
    --once

.venv\Scripts\python.exe main.py plant-shadow status
```

The first run reads only the latest configured source-relative lookback, then advances through bounded batches using the composite `(timestamp, durable_row_id)` watermark. It does not export the database, train on plant rows, or write to PostgreSQL. Evidence and checkpoints are written only to the local SQLite ledger.

If the bounded run succeeds, repeat it and verify that the durable watermark resumes without duplicates. Only then consider continuous shadow ingestion by removing `--once`.

Start the localhost-only API and dashboard after the bounded check:

```powershell
.\run_plant_shadow.ps1 -Actor "your-name"
```

The launcher intentionally does not start ingestion. Keep using explicit `--once` cycles until the
machine-state evidence, machine isolation, watermark, and local audit records have been inspected.

## 6. Database-focused verification

The default readiness script runs the connector, storage, lifecycle, operating-state, API,
manifest, and golden-replay tests:

```powershell
.\test_office_readiness.ps1
```

The equivalent direct pytest command is:

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp_office_readiness `
    tests\test_config.py `
    tests\test_readonly_contract.py `
    tests\test_plant_shadow_contracts.py `
    tests\test_plant_shadow_source.py `
    tests\test_plant_shadow_storage.py `
    tests\test_plant_shadow_service.py `
    tests\test_plant_shadow_operating_context.py `
    tests\test_vibration_operating.py `
    tests\test_plant_shadow_api.py `
    tests\test_plant_shadow_evaluation.py `
    tests\test_plant_shadow_manifest_cli.py `
    tests\test_plant_shadow_golden.py `
    tests\test_plant_shadow_native_postgres.py `
    tests\test_native_postgres_lifecycle.py
```

The disposable PostgreSQL E2E harness is for a loopback test server only; it deliberately refuses a remote plant database. Do not set `VVB001_E2E_ADMIN_DSN` to a plant database. If the office computer has an approved local PostgreSQL test server or native server binaries, follow the isolated instructions in `README.md`.

## 7. Finish safely

Capture the console output and the redacted `plant-shadow status` result. Do not copy passwords into screenshots, logs, issues, or reports. Remove the session credential when finished:

```powershell
Remove-Item Env:VVB001_PLANT_SOURCE_A_DSN
```

Do not weaken the ordering-index check, manifest verification, golden replay, read-only privilege contract, operating-state gate, model thresholds, calibration, manufacturer safety logic, or plant-production authorization to make a connection succeed.
