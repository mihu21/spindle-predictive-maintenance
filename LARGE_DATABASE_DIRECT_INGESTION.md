# Direct Large-Database Plant-Shadow Ingestion

## Outcome

The plant-shadow worker connects directly to the live sensor PostgreSQL database. It does not
export the database, copy the full table, or use plant rows to retrain the frozen v2.7 model.

The initial connection now starts from a bounded source-relative tail rather than the oldest row.
The default reads at most the latest 24 hours by event timestamp, then continues forward through
durable SQLite watermarks in batches of 1,000 rows. The exact initial lower bound is written to
local source-health evidence.

## Why GPU training was not added

- The accepted v2.7 Extra Trees artifact is frozen and scikit-learn Extra Trees is CPU-based.
- The live sensor stream has no independent WARNING/CRITICAL endpoint labels suitable for
  supervised retraining.
- Vibration operating calibration is a bounded causal median/variability calculation, not a deep
  learning training workload.
- GPU libraries would add a new model stack while leaving PostgreSQL scan, ordering, network, and
  SQLite commit costs unchanged.

The correct optimization is bounded indexed streaming. No CUDA, cuML, XGBoost, Docker, or other GPU
dependency was added.

## Source configuration

```json
"batch_size": 1000,
"poll_seconds": 5.0,
"lateness_seconds": 300.0,
"initial_lookback_hours": 24.0,
"statement_timeout_seconds": 30.0,
"require_ordering_index": true
```

`initial_lookback_hours` is measured backward from the latest source timestamp, not the workstation
clock. Once the first rows commit, the SQLite watermark is authoritative and later polls use the
normal composite continuation query.

Setting `initial_lookback_hours` to JSON `null` explicitly requests full-history ingestion. Do not
do this for a large production table without a separate reviewed migration plan.

## Required PostgreSQL index

Continuous ingestion requires a B-tree index whose first key columns are the timestamp and durable
row ID in that order. The SELECT-only application checks the PostgreSQL catalog and refuses to run
if it is absent.

Ask the database administrator to review a command equivalent to:

```sql
CREATE INDEX CONCURRENTLY vvb001_shadow_order_idx
ON public.vvb001_readings (timestamp, id);
```

Use the actual schema, table, timestamp, and durable-ID names. The plant-shadow application never
executes this DDL and never receives index-creation privileges.

## Operational behavior

- Every SELECT has a configured statement timeout.
- Each fetch has a hard row limit.
- Queries use `(timestamp, durable_row_id)` ordering and watermark continuation.
- Source credentials remain database-level SELECT-only.
- Rows are processed causally; duplicates and late rows cannot advance runtime state.
- Historical rows outside the configured source tail are neither read nor represented as plant
  validation evidence.
- No plant-production authorization is granted.

The 24-hour default is long enough for the vibration detector's bounded calibration history while
avoiding an unbounded lifetime backfill. If the detector lacks both quiet and normal-running regimes
within that window, it remains `CALIBRATING` rather than broadening the query automatically.
