# VVB001 Plant Shadow Implementation Summary

## Outcome

The revised schema-v3 plant-shadow foundation is implemented around the frozen v2.7 bundle. It is
shadow-only and fail-closed. It does not authorize production, retrain a model, write to plant
PostgreSQL, create work orders, or control machinery.

## Implemented

- Distinct `source_key`, lossless `source_row_id`, local `ingestion_id`, and `machine_uid` contracts.
- Durable source identity validation; continuous monitoring rejects missing identities and `ctid`.
- A read-only PostgreSQL connector with parameterized identifiers, composite ordering, schema
  validation, timezone enforcement, and environment-held DSNs.
- A separate SQLite WAL evidence ledger with foreign keys, busy timeout, one-writer semantics,
  versioned source configuration, append-only evidence triggers, and supersession records.
- Atomic raw observation, validation disposition, online prediction, lifecycle event, audit, and
  watermark commits.
- Forced runtime discard/rebuild from committed causal history after a post-inference commit failure.
- One frozen v2.7 runtime per source, leaving the internal `line::machine` state key unchanged.
- Immutable available, withheld, and error-aware prediction payloads. Exact target values are cleared
  when the frozen serviceability contract does not make them available.
- Separate manager state and lifecycle record status, left-censored startup, conservative lifecycle
  creation, and explicit `MODEL_STATE_RESET_GAP` events that do not close lifecycles.
- Independent endpoint evidence, classification, precision, WARNING/CRITICAL eligibility, exact
  truth, right censoring, plant-validation locks, and confirmed maintenance closure.
- Preventive maintenance/replacement does not become CRITICAL onset truth. `CLOSED_INFERRED` is
  excluded from exact plant accuracy.
- Support-gated, target-specific online plant metrics with separate point and interval support.
- A transitive runtime manifest and deterministic 24-row v2.7 golden replay.
- A versioned, read-only FastAPI route implementation restricted by CLI to localhost.
- A React/TypeScript monitoring frontend for overview, machines, lifecycles, sources, plant evidence,
  model identity, system health, and audit evidence.
- A fail-closed operating-context gate that admits model updates only for high-confidence `RUNNING`
  evidence, pauses RUL while idle/off/under maintenance/unknown, and preserves immediate raw-sensor
  manufacturer CRITICAL handling without feeding non-running samples into the model.
- A causal per-machine operating-exposure clock. RUL aging, lifecycle truth, recovery, and runtime
  rebuild use accumulated confirmed-running time rather than unattended wall-clock time.
- Append-only local PLC/CMMS/operator operating-state evidence, conflict handling, maintenance IDs,
  context decisions, transition audits, and frontend/API visibility.
- A causal per-machine vibration eligibility detector for installations with no machine-state log.
  It learns distinct quiet/production regimes, recognizes only persistent high-confidence RUNNING,
  and fails closed for calibration gaps, quiet, unstable, impulsive, extreme, or novel vibration.
- Append-only vibration calibration/decision evidence, artifact hashes, API/frontend visibility,
  duplicate/late protection, and deterministic detector rebuild from committed pre-lifecycle rows.
- Deterministic duty-cycled synthetic runtime fixtures containing idle, shutdown, maintenance, and
  unknown periods, including a no-visible-state vibration-inference variant. Training rejects both
  fixture classes so runtime-gate test data cannot enter model fit.
- Local CLI commands for manifest/golden verification, source configuration, ingestion, status,
  endpoint evidence, endpoint classification, evaluation, and API serving.
- A self-cleaning native PostgreSQL E2E harness with two overlapping source identities, a real
  database-level read-only application role, bounded lateness, restart recovery, endpoint semantics,
  API checks, and operational measurements.
- Direct large-table startup from the latest 24 source-hours instead of an unbounded lifetime
  backfill, followed by durable composite-watermark continuation in bounded batches.
- A required PostgreSQL `(timestamp,durable_row_id)` ordering index, 30-second statement timeout,
  and append-only recording of the exact source-tail bootstrap boundary. The application remains
  SELECT-only and never creates the index itself.
- No GPU stack or plant-data retraining path. The frozen v2.7 model remains unchanged; indexed
  streaming addresses the database bottleneck without manufacturing unlabeled plant targets.

## Verification

```text
Full Python suite                 197 passed, 1 skipped
Accepted pre-change baseline      144 tests retained
Vibration/operating/API target      24 passed
FastAPI/API test                    1 passed
Golden replay                       PASS
Golden canonical output SHA-256    dfed03ffb3d083561a9ed8bc40c7f4baf80c2a3e3adcbf46abca6e0e9699ac43
Runtime manifest                    PASS
Shadow schema version               plant_shadow_schema_v3_vibration_operating_inference
Frozen model SHA-256               ecad8f4f704129a3f0456c3c8dd47aabeb94ea6dfbdfd4c264313aea46076b9a
Native PostgreSQL E2E               PASS (PostgreSQL 18 / Windows / 25.6s)
Real plant PostgreSQL E2E            NOT RUN (correctly deferred)
Frontend tests                      2 passed
TypeScript project check            PASS
Production frontend build           PASS (89 modules transformed)
Plant validation                    NOT_STARTED
Plant-production authorization      FALSE
```

Full test command:

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp=.test-tmp\large-db-full-final
# 197 passed, 1 skipped, 10288 warnings in 113.70s
```

### Dependency state and commands

The declared backend and frontend dependencies were installed with:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Set-Location frontend
npm install
```

The initial Python resolver paired FastAPI 0.125.0 with Pydantic 1.10.26, which failed at import
because that FastAPI build imports Pydantic's v2 `TypeAdapter`. `requirements.txt` now declares
`pydantic>=2.12,<3`; Pydantic 2.13.4 was installed. This is API dependency correction only and does
not change the model or prognostics contract. `pip check` reports no broken requirements.

Verified installed Python versions:

```text
Python             3.14.6
psycopg            3.3.4
psycopg-binary     3.3.4
numpy              2.5.2
scikit-learn       1.9.0
joblib             1.5.3
pytest             8.4.2
fastapi            0.125.0
pydantic           2.13.4
pydantic-core      2.46.4
starlette          0.50.0
uvicorn            0.52.1
httpx              0.28.1
```

Verified direct frontend versions:

```text
@tanstack/react-query       5.101.4
@testing-library/jest-dom   6.9.1
@testing-library/react      16.3.2
@types/react                19.2.18
@types/react-dom            19.2.4
@vitejs/plugin-react        5.2.0
jsdom                       26.1.0
react                       19.2.8
react-dom                   19.2.8
react-router-dom            7.18.2
typescript                  5.9.3
vite                        7.3.6
vitest                      3.2.7
```

`npm install` added 166 packages, audited 167 packages, and reported zero vulnerabilities. The
resolved `frontend/package-lock.json` SHA-256 is
`aa097ce1574e193df26ef429b393c00b7a0e89d6d3fa25b8a9dc092156eafca1`.

Commands and exact results from the earlier dependency verification are retained below. The final
vibration-operating update verification follows it.

```powershell
.venv\Scripts\python.exe -m pip check
# No broken requirements found.

.venv\Scripts\python.exe -m pytest -q tests\test_plant_shadow_api.py --basetemp=.pytest_tmp_api_verified
# 1 passed in 0.43s

.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp_plant_shadow_final_dependencies_2
# 163 passed, 2216 warnings in 112.04s

.venv\Scripts\python.exe -m pytest -q tests\test_plant_shadow_storage.py tests\test_plant_shadow_source.py tests\test_plant_shadow_native_postgres.py tests\test_plant_shadow_postgres_e2e.py --basetemp=.pytest_tmp_native_pg_targeted_3
# 12 passed, 1 skipped in 0.45s

.venv\Scripts\python.exe -m pytest -q tests\test_plant_shadow_operating_context.py tests\test_plant_shadow_storage.py tests\test_plant_shadow_api.py --basetemp=.pytest_tmp_operating_context_migration
# 15 passed, 5045 warnings in 3.60s

.venv\Scripts\python.exe -m pytest -q tests\test_plant_shadow_api.py tests\test_native_postgres_lifecycle.py tests\test_plant_shadow_native_postgres.py tests\test_plant_shadow_postgres_e2e.py --basetemp=.pytest_tmp_pg_hang_targeted_3
# 12 passed, 1 skipped in 0.55s

.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp_pg_hang_full_final
# 186 passed, 1 skipped, 7261 warnings in 112.08s

Remove-Item Env:VVB001_E2E_ADMIN_DSN -ErrorAction SilentlyContinue
.venv\Scripts\python.exe tests\e2e\run_plant_shadow_postgres_e2e.py --pg-bin-dir "C:\Program Files\PostgreSQL\18\bin" --port 55432
# [E2E] PASS; exit 0; 21.6s; port released; owned PID terminated

npm test
# 1 test file passed; 1 test passed; duration 525 ms

npx tsc -b
# PASS; no TypeScript errors

npm run build
# PASS; Vite 7.3.6; 89 modules transformed; built in 1.13 s
# dist/index.html                  0.45 kB (gzip 0.29 kB)
# dist/assets/index-ZCPGawgU.css   4.13 kB
# dist/assets/index-DOdkRejO.js  276.68 kB

.venv\Scripts\python.exe main.py plant-shadow verify-golden
# PASS; 24 rows; canonical output SHA-256 dfed03ffb3d083561a9ed8bc40c7f4baf80c2a3e3adcbf46abca6e0e9699ac43

.venv\Scripts\python.exe main.py plant-shadow create-manifest --deployment-id vvb001-plant-shadow-v1 --output output\plant_shadow\plant_shadow_runtime_manifest.json
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from vvb001_monitor.plant_shadow.manifest import verify_manifest; print(verify_manifest('.', 'output/plant_shadow/plant_shadow_runtime_manifest.json')['deployment_id'])"
# PASS; vvb001-plant-shadow-v1; dependency lock updated for Pydantic 2;
# plant_production_authorized=false
```

Final vibration-operating verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_vibration_operating.py tests\test_plant_shadow_operating_context.py tests\test_plant_shadow_storage.py tests\test_plant_shadow_api.py tests\test_plant_shadow_service.py -q --basetemp=.test-tmp\vibration-cadence-final
# 24 passed, 8072 warnings in 5.27s

.venv\Scripts\python.exe -m pytest -q --basetemp=.test-tmp\vibration-cadence-full-final
# 194 passed, 1 skipped, 10288 warnings in 186.23s

npm test -- --run
# 1 test file passed; 2 tests passed; duration 1.52s

npx tsc -b
# PASS; no TypeScript errors

npm run build
# PASS; Vite 7.3.6; 89 modules transformed; built in 873ms
# dist/index.html                  0.45 kB (gzip 0.29 kB)
# dist/assets/index-5y-Rw-hA.css   4.28 kB (gzip 1.62 kB)
# dist/assets/index-DAeSaFm6.js  277.51 kB (gzip 86.97 kB)

.venv\Scripts\python.exe main.py plant-shadow verify-golden
# PASS; 24 rows; canonical output SHA-256 dfed03ffb3d083561a9ed8bc40c7f4baf80c2a3e3adcbf46abca6e0e9699ac43

.venv\Scripts\python.exe main.py plant-shadow create-manifest --deployment-id vvb001-plant-shadow-v1 --output output\plant_shadow\plant_shadow_runtime_manifest.json
# PASS; schema v3; vibration detector included transitively; plant_production_authorized=false

.venv\Scripts\python.exe tests\e2e\run_plant_shadow_postgres_e2e.py --pg-bin-dir "C:\Program Files\PostgreSQL\18\bin" --port 55432
# [E2E] PASS; exit 0; 25.1s; port released; owned PID 7236 terminated
```

Final direct large-database verification:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_plant_shadow_source.py tests\test_plant_shadow_operating_context.py tests\test_vibration_operating.py -q --basetemp=.test-tmp\large-db-targeted
# 25 passed, 8072 warnings in 4.27s

.venv\Scripts\python.exe -m pytest -q --basetemp=.test-tmp\large-db-full-final
# 197 passed, 1 skipped, 10288 warnings in 113.70s

.venv\Scripts\python.exe tests\e2e\run_plant_shadow_postgres_e2e.py --pg-bin-dir "C:\Program Files\PostgreSQL\18\bin" --port 55432
# [E2E] PASS; source-tail boundary asserted; ordering_index=true; exit 0; 25.6s;
# port released; owned PID 13004 terminated
```

Production build hashes:

```text
dist/index.html                  5db1ff42f6eb5480975bc600c63e07a209afc9ad3194714b7a5de34bdc9c7356
dist/assets/index-5y-Rw-hA.css  6023c4eb6b36949782174f2bf2c554df8b77edaa2682025d4a0692bb7ed26459
dist/assets/index-DAeSaFm6.js   231415c4a563e04264b844af51ec98837c542714efb330fe461f134b1531dd64
```

## Evidence identities

```text
Plant-shadow deployment ID          vvb001-plant-shadow-v1
Plant-shadow manifest SHA-256        d69923adf4c5a7bc7863e93ba6b5ec7509f254735cc11341671273a8c52946d4
Native PostgreSQL E2E report SHA-256 5e6fa10964ada2abde8d1038ad2427770a5b2f71260057fdb41cc104bd03eae9
Dependency lock hash                 a77256928b7d0726ba56152def6a6a469f91dbe0f52d43cea8f6b7ea25ae05c7
Accepted freeze manifest SHA-256    37a8a9907e3ceafacd8366968829dc2a09d5b96859c8a9bf5ae2a5c4ed8f6497
Consumed evidence registry SHA-256  56b69200445512adadbad322082b13dc53631ee66bd162b2212214f4bf473b66
Sealed holdout report SHA-256        8aa9a28cdff32dcb1597d75ea5b3dfcc0ad831c9b361f665342b1de423590fdb
```

Additional read-only historical evidence hashes observed during this verification:

```text
Development acceptance report       104967a084cb7e8e8b6bf776bcfc66f61ecab008d673f7d17e21c2856d5fe373
Acceptance runtime report            38ada2f99119d5601afffc587845710ab1c058ac62a358c29c240f44e2d153fb
Sealed holdout manifest file          b9db4f53139b6c3633478c81b44b8fe4457a4a6c463cb267bc2f0f4290d18f8b
Sealed holdout runtime report         89105607a9791e7653c219a9f6bd44c7e95f7b2de684e2612ec54bd9411bb395
Registered acceptance corpus hash     d4a6d32e81475e6aa9a044af017ea978476f4e44af9d3c387e2461199aed832d
Registered sealed corpus hash         fde79d4abdd203a8aabd48251054bebf3d390f3e30d65a3779aa9008b57cc587
Registered sealed report hash         8aa9a28cdff32dcb1597d75ea5b3dfcc0ad831c9b361f665342b1de423590fdb
```

No command in this continuation opened these files for writing. The pre/post hashes captured for the
model, accepted freeze, consumed registry, and sealed report were byte-identical.

These hashes matched the pre-install snapshot exactly after verification. Registry semantics also
remain unchanged:

```text
Development acceptance consumed   TRUE / PASS
Sealed holdout consumed            TRUE / PASS
Synthetic candidate locked         TRUE
Plant validation                   NOT_STARTED
Plant-production authorization     FALSE
```

## Native PostgreSQL E2E remediation result

The disposable PostgreSQL E2E harness is implemented at
`tests/e2e/run_plant_shadow_postgres_e2e.py`. PostgreSQL 18.4 is now installed at
`C:\Program Files\PostgreSQL\18\bin`, and the `postgresql-x64-18` service is running on the default
port. The disposable cluster completed independently on port 55432 without using or changing the
installed service credentials:

```powershell
Remove-Item Env:VVB001_E2E_ADMIN_DSN -ErrorAction SilentlyContinue
.venv\Scripts\python.exe tests\e2e\run_plant_shadow_postgres_e2e.py `
    --pg-bin-dir "C:\Program Files\PostgreSQL\18\bin" `
    --port 55432
```

Exact result:

```text
[E2E] PostgreSQL ready (owned PID 13004).
[E2E] Temporary PostgreSQL stopped and port released.
[E2E] PASS
Exit code: 0
Elapsed time: 25.6 seconds
Post-test port 55432: RELEASED
Owned PID 13004: TERMINATED
```

The original hang was caused by executing `pg_ctl start` with `capture_output=True`. On Windows,
the detached `cmd.exe -> postgres.exe` process tree retained captured pipe handles; Python waited
for pipe EOF even after PostgreSQL was ready. Startup now uses `pg_ctl -W start` through `Popen`
with regular-file output, followed by an explicit bounded `pg_isready` loop.

The completed E2E also identified and corrected three previously masked integration defects:

- PostgreSQL 18 does not accept a bind parameter in `CREATE ROLE ... PASSWORD`; fixture DDL now
  uses psycopg escaped SQL composition without logging the secret.
- `information_schema.table_constraints` hides constraint rows from a SELECT-only reader. Durable
  identity is now verified through read-only `pg_catalog.pg_index` metadata without granting writes.
- API SQLite connections and `TestClient` lifecycle are now closed deterministically on Windows.

Final direct-source target result: `25 passed`. Final full suite result:
`197 passed, 1 skipped`. The native report status is `PASS`, including source-tail bootstrap and
ordering-index assertions.

## Required work before initial plant observation

Alternatively, run it against a dedicated database on the local service using a real, intentionally
provisioned fixture-admin credential:

```powershell
$env:VVB001_E2E_ADMIN_DSN = "postgresql://fixture_admin:password@127.0.0.1:5432/postgres"
.venv\Scripts\python.exe tests\e2e\run_plant_shadow_postgres_e2e.py
```

Alternatively, install native PostgreSQL binaries and pass `--pg-bin-dir`; the harness will create a
temporary `initdb` cluster on a non-default loopback port and stop/remove it afterward. It refuses
remote admin DSNs and non-`vvb001_e2e_*` target names. It creates separate fixture/admin and
SELECT-only application credentials, and the production adapter receives only the reader DSN. No
Docker, Testcontainers, or orchestration dependency is present. No approved plant source is used.

With that local E2E report passed:

1. Create a local source configuration from the example without embedding secrets.
2. Confirm the approved source account is database-level read-only and the row identity is durable.
3. Run one-shot ingestion and inspect source health, watermark, lifecycle, and prediction evidence.
4. Run operational load tests at the declared plant cadence.
5. Expand frontend/API behavioral coverage beyond the current smoke tests before wider deployment.

These are deployment-verification steps, not permission to use the system for plant control.
