# VVB001 PostgreSQL Plant-Shadow E2E Hang Remediation

## Root cause

The runner called `pg_ctl start` through `subprocess.run(..., capture_output=True)`. On Windows,
`pg_ctl` launches a detached `cmd.exe -> postgres.exe` process tree. That long-running tree retained
the captured pipe handles, so Python waited for pipe EOF after `pg_ctl` handed off a server that was
already accepting connections. The runner never reached database provisioning.

## Implementation

- `pg_ctl -W start` is launched through `Popen` with stdout/stderr redirected to a regular file,
  never a pipe, and only the finite launcher is awaited.
- `pg_isready` polls `127.0.0.1:<port>` independently with a 60-second startup deadline.
- `initdb`, the launcher, readiness probes, SQL connects/statements, and shutdown are bounded.
- Failures identify the operation, timeout, host, port, temporary data directory, log path, and log
  tail without printing credentials.
- A port preflight refuses collisions before initialization.
- The postmaster PID is read from the owned data directory. Cleanup uses `pg_ctl -D` for that exact
  cluster; its fallback signals only that recorded PID, never every `postgres.exe` process.
- Cleanup runs through context-manager `finally` behavior after success, assertion failure,
  exceptions, and `KeyboardInterrupt`.
- Concise `[E2E]` stages are flushed throughout initialization, readiness, provisioning, ingestion,
  validation, and shutdown.

## Integration corrections exposed by the remediated run

Once startup progressed, the real PostgreSQL 18 run exposed fixture and read-only integration bugs
that unit mocks had not reached:

1. `CREATE ROLE ... PASSWORD $1` is invalid PostgreSQL DDL. The fixture uses psycopg SQL literal
   composition, which escapes the generated test credential and never emits it to diagnostics.
2. `information_schema.table_constraints` does not expose table constraints to a SELECT-only user.
   The production adapter now verifies a valid one-column, non-partial, non-expression unique index
   through read-only `pg_catalog` metadata. No extra source privilege is granted.
3. The intended late row arrived in the same ordered poll as newer rows and therefore was not late.
   It is now inserted in a second poll inside the configured lookback after the watermark advances.
4. Read-only API SQLite connections and the E2E `TestClient` are explicitly closed, allowing Windows
   to remove the temporary evidence database.

## Verification

```text
Targeted lifecycle/API tests       12 passed, 1 skipped
Complete Python suite             186 passed, 1 skipped
Native PostgreSQL 18 E2E           PASS
Native E2E elapsed                 21.6 seconds
Native E2E exit code               0
Temporary port 55432               RELEASED
Owned postmaster PID 18896         TERMINATED
Plant source used                  false
Plant-production authorization     false
```

Exact successful command:

```powershell
Remove-Item Env:VVB001_E2E_ADMIN_DSN -ErrorAction SilentlyContinue
.venv\Scripts\python.exe tests\e2e\run_plant_shadow_postgres_e2e.py `
    --pg-bin-dir "C:\Program Files\PostgreSQL\18\bin" `
    --port 55432
```

Evidence hashes:

```text
Runtime manifest                 e577f3d1ad622a95a976a99b3549cac4b1962f190e023e6b09865f579e3a7282
Native PostgreSQL E2E report     6cd9f579f93a27f9dc1faa540b8d9a23c4c9c09e085027560b6997562b0c0629
Frozen v2.7 model                ecad8f4f704129a3f0456c3c8dd47aabeb94ea6dfbdfd4c264313aea46076b9a
Accepted freeze manifest         37a8a9907e3ceafacd8366968829dc2a09d5b96859c8a9bf5ae2a5c4ed8f6497
Consumed evidence registry       56b69200445512adadbad322082b13dc53631ee66bd162b2212214f4bf473b66
Sealed holdout report            8aa9a28cdff32dcb1597d75ea5b3dfcc0ad831c9b361f665342b1de423590fdb
```

This is a disposable local synthetic integration result, not plant validation. It does not change
or authorize model promotion, production database writes, machine control, or plant production.
