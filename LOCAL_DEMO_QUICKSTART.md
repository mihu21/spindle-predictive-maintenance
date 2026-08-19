# VVB001 deterministic local plant-shadow demo

## Purpose

This workflow exercises the existing plant-shadow service, frozen v2.7 runtime, evidence ledger, read-only FastAPI API, and React UI on a laptop. All observations are deterministic synthetic demo evidence. This is local offline testing only and is not plant validation.

## Quick start

From the repository root in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_local_demo.ps1 -Actor "Michael"
```

The launcher defaults to eight machines, 72 hours, a 10-minute cadence, seed 42, and the fixed start `2026-01-01T00:00:00+00:00`. A six-machine compact demo combines the sensor-quality and support-gating behaviors with other profiles:

```powershell
.\run_local_demo.ps1 -Machines 6 -Hours 72 -Seed 42 -Actor "Michael"
```

URLs:

```text
API:      http://127.0.0.1:8000
Frontend: http://127.0.0.1:5173
```

The launcher rebuilds the demo database before starting two hidden local server processes and prints their PIDs. Use `-SkipBrowser` to avoid opening the browser.

## Generate evidence only

```powershell
.venv\Scripts\python.exe main.py plant-shadow generate-demo `
    --machines 6 `
    --hours 72 `
    --seed 42 `
    --start-time 2026-01-01T00:00:00+00:00 `
    --actor "Michael"
```

The command atomically rebuilds `output\plant_shadow_demo\plant_shadow_demo.db`. It rejects the normal `output\plant_shadow\plant_shadow.db` path. The generator imports no PostgreSQL observations, requires no DSN, and stores an `OFFLINE_DEMO_COMPLETE` source-health event.

## What it does

- Creates an isolated SQLite evidence ledger.
- Produces deterministic, globally ordered `PlantObservation` values.
- Processes every observation through `PlantShadowService` and `FrozenRuntimeRouter` with the frozen v2.7 model loaded once per demo source.
- Preserves raw telemetry, operating-context decisions, sensor-quality decisions, forecasts, lifecycle events, and audit evidence.
- Uses the existing endpoint-evidence/classification APIs to confirm component replacement and close the old lifecycle.
- Starts the normal read-only FastAPI application and existing React frontend against the demo ledger.
- Marks API responses and the UI as `LOCAL DEMO MODE`, synthetic evidence, with production authorization disabled.

## What it does not do

- It does not connect to PostgreSQL or start plant ingestion.
- It does not read or create office credentials and requires no DSN.
- It does not modify `output\plant_shadow\plant_shadow.db`.
- It does not retrain, replace, tune, or promote v2.7.
- It does not use the sealed holdout for generation or tuning.
- It does not authorize plant production or turn synthetic metrics into plant validation.

## Scenario machines

| Machine | Scenario | Intended evidence |
|---|---|---|
| `DEMO_HEALTHY` | Healthy RUNNING | Low degradation, normal history, real support-gating outcomes |
| `DEMO_DEGRADING` | Progressive degradation | Smooth physical degradation; compact six-machine mode also includes dropout/recovery |
| `DEMO_IDLE` | RUNNING to IDLE to RUNNING | Raw rows retained, runtime and operating clock paused, then resumed |
| `DEMO_OFF` | RUNNING to OFF to RUNNING | OFF remains distinct, runtime pauses, then resumes |
| `DEMO_UNKNOWN` | Missing operating evidence | UNKNOWN fail-closed state, no runtime admission, explicit reason |
| `DEMO_MAINTENANCE` | Maintenance and replacement | Paused maintenance, confirmed component replacement, closed lifecycle, clean lifecycle 2 |
| `DEMO_SENSOR_FAULT` | Dedicated quality fault | Dropout/stuck evidence and recovery; included when `--machines` is at least 7 |
| `DEMO_WITHHELD` | Dedicated support gating | Valid atypical telemetry evaluated by genuine frozen support gates; included when `--machines` is at least 8 |

Additional requested machines are deterministic independent healthy replicas. Scenario names live only in raw/audit metadata and are never model feature inputs. A particular RUL or health state is never forced; the actual frozen runtime decides it.

## Inspect the demo

```powershell
.venv\Scripts\python.exe main.py plant-shadow status `
    --database output\plant_shadow_demo\plant_shadow_demo.db
```

The API exposes overview, machines, sensor history, forecasts, operating context, lifecycles, model identity, system health, audit records, and `/api/v1/demo` metadata using the same SQLite query path as normal plant shadow.

## Reset only the demo

Regenerating is the safest reset because the command builds a replacement database and swaps it into place only after successful completion:

```powershell
.venv\Scripts\python.exe main.py plant-shadow generate-demo --machines 8 --hours 72 --seed 42
```

To remove only local demo evidence while both demo servers are stopped:

```powershell
Remove-Item -LiteralPath .\output\plant_shadow_demo\plant_shadow_demo.db -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\output\plant_shadow_demo\plant_shadow_demo.db-wal -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\output\plant_shadow_demo\plant_shadow_demo.db-shm -ErrorAction SilentlyContinue
```

These paths are separate from the normal plant-shadow database.

## Verification

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp .pytest_tmp_local_demo
Set-Location frontend
npm.cmd test
npm.cmd run build
Set-Location ..
.venv\Scripts\python.exe main.py plant-shadow verify-golden
```

The workspace-local pytest base temp avoids a known Windows `%TEMP%` permission issue on some laptops; it does not change product behavior or weaken cleanup.

