# VVB001 Plant Shadow, Lifecycle, Evidence, and Frontend Specification — v2

## 0. Status and authority

**Project:** `spindle_prognostics_vvb001`  
**Document version:** 2.0  
**Purpose:** implementation contract for the next plant-shadow release  
**Supersedes:** `VVB001_PLANT_SHADOW_LIFECYCLE_FRONTEND_SPEC.md`  
**Production authorization:** false

This revision incorporates the review of the original specification. If this document conflicts with the original, this document governs.

Current state that must be preserved:

```text
Synthetic development validation      COMPLETE / PASS
Synthetic sealed-holdout validation   COMPLETE / PASS
Frozen candidate                      v2.7
Frozen model                          models/rul_v2_7_full_cadence.joblib
Frozen model SHA-256                  ecad8f4f704129a3f0456c3c8dd47aabeb94ea6dfbdfd4c264313aea46076b9a
Sealed evidence                       CONSUMED_PASS
Synthetic candidate                   LOCKED
Plant shadow validation               NOT_STARTED
Plant-production authorization        FALSE
Automatic retraining                  DISABLED
Automatic maintenance actions         DISABLED
```

The implementation must preserve the frozen v2.7 artifact, historical reports, evidence registries, synthetic validation semantics, and immediate manufacturer safety behavior.

This release collects observational plant evidence. It does not validate the model for production control.

---

## 1. Objective

Build a local, read-only plant-shadow system that can:

1. Read one or more approved PostgreSQL sensor sources.
2. Normalize rows without changing their source evidence.
3. process each source through an isolated frozen v2.7 runtime;
4. persist predictions exactly as issued online;
5. manage per-machine lifecycles without confusing gaps or runtime resets with physical replacement;
6. ingest independent endpoint evidence;
7. determine WARNING and CRITICAL truth eligibility independently;
8. handle censoring and timestamp precision explicitly;
9. evaluate only eligible stored online predictions;
10. monitor drift, source health, data quality, forecast availability, and runtime health;
11. expose a read-mostly local FastAPI service; and
12. provide a React and TypeScript monitoring frontend.

The system must state everywhere:

```text
Synthetic validation success != plant validation != plant-production authorization
```

---

## 2. Non-negotiable boundaries

### 2.1 Plant PostgreSQL is read-only

Allowed operations are approved reads and metadata inspection required for those reads. Plant connections must set and verify read-only transaction behavior where supported.

Forbidden against the plant database:

```text
INSERT, UPDATE, DELETE, CREATE, ALTER, DROP
triggers, indexes, stored procedures, advisory writes
ML metadata or watermarks stored in the plant database
```

Use a database account with `SELECT`-only privileges. Warn if a privilege audit finds broader access. All application state belongs in the local application database or another separately approved store.

### 2.2 Frozen v2.7 is an external boundary

Do not retrain, recalibrate, retune, rewrite, or silently replace v2.7. Preserve its:

- model artifact and estimator identity;
- target-specific selectors and correction logic;
- feature contract and causal ordering;
- asymmetric calibration;
- serviceability and withholding rules;
- manufacturer safety behavior;
- synthetic acceptance reports and consumed registries.

Do not change the frozen runtime merely to add `source_key` or support the UI. Integration occurs through adapters and isolated runtime instances.

### 2.3 Shadow outputs are advisory

Shadow outputs must not automatically stop equipment, command a PLC, create or schedule a work order, acknowledge physical alarms, alter manufacturer status, write to the plant database, authorize production, or promote a model.

Manufacturer WARNING and CRITICAL behavior remains immediate and independent of ML serviceability.

### 2.4 No automatic learning

Plant observations and eligible outcomes may be archived for future governed work. They must not automatically train, tune, calibrate, select, or replace any model.

### 2.5 Predictions are never truth

Never derive endpoint evidence or labels from predicted RUL, intervals, model health, serviceability, anomaly outputs, or predicted health state.

---

## 3. Target and endpoint contract

This section is a blocking contract. Code that collapses endpoint evidence into one generic `T_end` is non-conforming.

### 3.1 Separate prediction targets

v2.7 exposes two different targets:

```text
WARNING target  = time until independently established WARNING onset
CRITICAL target = time until independently established CRITICAL onset
```

WARNING and CRITICAL truth, censoring, eligibility, metrics, and support counts must be stored and evaluated separately. A lifecycle need not have both targets.

### 3.2 Evidence is not automatically truth

Use this pipeline:

```text
endpoint evidence
    -> endpoint classification
    -> target-specific truth eligibility
    -> retrospective target truth or censoring record
```

Required endpoint classes:

```text
WARNING_ONSET
CRITICAL_ONSET
PREVENTIVE_MAINTENANCE
COMPONENT_REPLACEMENT
PHYSICAL_FAILURE
MACHINE_REPLACEMENT
SENSOR_REPLACEMENT
RESTART
OTHER
UNKNOWN
```

Store the original evidence and its provenance separately from the derived classification and eligibility decision.

### 3.3 Default eligibility matrix

| Independent event | WARNING truth | CRITICAL truth | Other endpoint |
|---|---:|---:|---|
| Independently observed WARNING onset | eligible after quality checks | none | none |
| Independently observed CRITICAL onset | none unless WARNING onset also exists | eligible after quality checks | none |
| Preventive maintenance before CRITICAL | possible only with separate WARNING onset | right-censored | maintenance endpoint |
| Component replacement | not automatic | not automatic; normally censored if before onset | replacement endpoint |
| Physical failure | not automatic | not automatic unless contract maps an independently observed CRITICAL onset | failure endpoint |
| Machine replacement | not automatic | not automatic | replacement endpoint and lifecycle boundary |
| Sensor replacement | no | no | sensor event; not a machine lifecycle boundary by itself |
| Restart | no | no | supporting evidence only |

An operator may not override this matrix by simply marking a replacement as CRITICAL. Any policy exception requires a new truth-policy version, documented rationale, and prospective application.

### 3.4 Censoring

Represent target-specific censoring explicitly:

```text
NOT_CENSORED
LEFT_CENSORED
RIGHT_CENSORED
INTERVAL_CENSORED
UNKNOWN
```

Examples:

- first observation occurs after lifecycle start: lifecycle is left-censored;
- preventive replacement occurs before observed CRITICAL: CRITICAL is right-censored at replacement;
- WARNING onset is independently observed but CRITICAL is prevented: WARNING may be exact while CRITICAL is censored;
- uncertain event time within a shift: endpoint is interval-censored.

Right-censored rows must not enter exact MAE or signed-bias calculations as if the censoring time were onset truth. Censoring-aware analyses may be added later and must be labeled separately.

### 3.5 Endpoint precision

Store endpoint precision:

```text
EXACT_TIMESTAMP
BOUNDED_INTERVAL
DATE_ONLY
UNKNOWN
```

Also store `endpoint_time_lower`, `endpoint_time_upper`, timezone, evidence source, confirmation actor, confirmation time, and policy version.

Only `EXACT_TIMESTAMP`, or a predeclared bounded tolerance suitable for a named metric, may contribute to exact point-error metrics. `DATE_ONLY` and `UNKNOWN` never enter exact MAE. Interval endpoints may support separately labeled interval-compatible analysis.

### 3.6 Truth generation

For target `q` with an eligible exact onset `T_q`, the retrospective truth for an online prediction made at event time `t` is:

```text
true_time_to_q_hours(t) = (T_q - t) / 3600
```

Requirements:

- use event time, not ingestion or processing time;
- do not generate negative target values;
- apply documented boundary inclusion rules;
- preserve timezone-aware timestamps;
- retain the endpoint evidence and eligibility-decision IDs;
- store truth separately from predictions;
- never overwrite the online prediction;
- never rerun the model with hindsight to create online metrics.

`CLOSED_INFERRED` lifecycles are not eligible for plant RUL accuracy in this release.

---

## 4. Canonical identities and timestamps

### 4.1 Required names

Use these meanings consistently:

```text
source_key       configured connector identity; stable UUID or immutable string
source_row_id    immutable identity supplied by the source, stored losslessly
ingestion_id     local SQLite integer surrogate key
machine_uid      source_key::line_sel::machine_id
lifecycle_id     local immutable lifecycle identity
```

Do not use `source_id` for connector identity in new plant-shadow code. The existing `VVB001Reading.source_id` means an integer row identity and must not be silently repurposed.

### 4.2 Frozen-runtime adapter

Introduce a plant-shadow observation type containing `source_key`, `source_row_id`, and `ingestion_id`. When adapting into the legacy frozen runtime, use the local integer `ingestion_id` for the legacy row field if needed. Persist the original `source_row_id` independently.

Do not alter the frozen predictor’s state key. Runtime isolation is provided at the source boundary, described in Section 6.

### 4.3 Required timestamps

Persist:

```text
event_timestamp       timestamp asserted by the sensor source
observed_at            time the application first observed/ingested the row
prediction_timestamp   time inference completed and output was issued
```

Use UTC internally while retaining original timezone/offset metadata. Do not rewrite raw event timestamps.

### 4.4 Source activation levels

```text
PROFILE_ONLY
    timestamp-only access may be accepted for limited profiling
    no continuous authoritative shadow evidence

SHADOW_MONITORING
    requires immutable unique source_row_id
    or timestamp plus immutable unique tie-breaker

PLANT_EVIDENCE_ELIGIBLE
    requires the same durable ordering identity
    plus all evidence and quality policies
```

PostgreSQL `ctid` is not a durable row identity. A timestamp alone is insufficient when duplicates are possible. Sources without a stable order key must fail closed for continuous shadow activation with `SOURCE_IDENTITY_UNSAFE`.

---

## 5. Architecture

```text
Plant PostgreSQL sources (read-only)
        |
        v
Source connectors and validators
        |
        v
Canonical observations + stable identities
        |
        v
Outer router keyed by source_key and machine_uid
        |
        +-- Source A -> dedicated frozen runtime A
        |                 +-- Line1::Machine01
        |                 +-- Line1::Machine02
        |
        +-- Source B -> dedicated frozen runtime B
                          +-- Line1::Machine01
        |
        v
Transactional local evidence store
        |
        +-- lifecycle manager and endpoint evidence
        +-- plant evaluation and drift
        +-- read-only FastAPI
                    |
                    v
             React/TypeScript UI
```

### 5.1 One source, one frozen runtime

Each enabled `source_key` receives a dedicated instance of the frozen stateful runtime. Never process rows from different sources through the same runtime instance, even when line and machine identifiers appear unique.

The outer plant-shadow layer keys all persisted and routed state by `machine_uid`. Within a source-isolated runtime, the existing `line_sel::machine_id` key remains unchanged.

Source failures and runtime rebuilds must be isolated. One source may not reset, block, or contaminate another.

### 5.2 One active processing owner per source

Only one worker may own inference advancement for a given `source_key` and deployment. Enforce this with a local lease or equivalent single-owner rule. The initial SQLite deployment uses one writer process.

---

## 6. Freeze, manifest, and configuration governance

### 6.1 Separate immutable identity from mutable deployment configuration

Maintain two related records:

```text
CODE / MODEL FREEZE
    immutable model and execution identity

DEPLOYMENT CONFIGURATION
    versioned and audited operational values
    may change through controlled local administration
```

Changing a PostgreSQL hostname must create a new deployment-configuration version, not falsely report a model-code hash mismatch.

### 6.2 Plant shadow runtime manifest

Create `plant_shadow_runtime_manifest.json` containing at least:

```text
deployment_id
created_at
v2_7_model_sha256
accepted_freeze_manifest_sha256
sealed_evidence_registry_sha256
sealed_holdout_report_sha256
runtime_module_hashes
plant_shadow_orchestrator_hash
source_mapper_hash
python_version
dependency_lock_hash
feature_contract_version
shadow_schema_version
lifecycle_policy_version
truth_policy_version
support_policy_version
```

The transitive runtime hash set must cover every module actually used to produce a live prediction, including predictor, features, validation, anomaly/generalization, sensor quality, RUL modules, model/config loading, and integration adapters.

Do not rewrite the historical v2.7 freeze file. The plant-shadow manifest references it.

### 6.3 Startup behavior

Before source activation:

1. verify the model artifact hash;
2. verify the accepted freeze and consumed registries;
3. verify runtime module hashes;
4. validate schema, lifecycle, truth, and support policy versions;
5. run or verify the golden replay fixture;
6. fail closed on mismatch.

Required reason codes include:

```text
MODEL_ARTIFACT_MISMATCH
ACCEPTED_FREEZE_MISMATCH
EVIDENCE_REGISTRY_MISMATCH
RUNTIME_HASH_MISMATCH
GOLDEN_REPLAY_MISMATCH
SCHEMA_VERSION_UNSUPPORTED
POLICY_VERSION_UNSUPPORTED
```

### 6.4 Golden replay

Freeze a small causal input sequence and expected v2.7 outputs. The same input must produce equivalent serviceability states, target outputs, intervals, withholding reasons, manufacturer state, and ordering before and after plant-shadow integration. Define numeric tolerances explicitly; compare categorical and null states exactly.

---

## 7. Source configuration and read-only validation

Each source configuration includes:

```text
source_key
display_name
host, port, database
username
secret_reference
ssl_mode
schema, table
timestamp_column
immutable_row_id_column or immutable_tie_breaker_column
line_column, machine_column
vrms_column, arms_column, apeak_column, crest_column, temperature_column
timezone interpretation
activation_level
enabled state
configuration_version
```

Secrets are environment or secret-store references. Do not store plaintext passwords in SQLite, logs, API payloads, exports, or frontend state.

Validation is read-only and must check connection, read-only mode, schema, column types, timestamp parsing, identity uniqueness/stability, duplicate frequency, ordering, nulls, numeric validity, plausible sensor ranges, cadence, latest event timestamp, and discovered machines.

An invalid source may remain saved as inactive configuration but cannot become `SHADOW_MONITORING` or `PLANT_EVIDENCE_ELIGIBLE`.

Source deletion means archive/deactivate. Never remove a `source_key` referenced by evidence.

---

## 8. Durable ingestion, ordering, and lateness

### 8.1 Watermark

Use a per-source durable order key such as:

```text
(last_event_timestamp, last_source_row_id)
```

or another validated total order. Requirements are deterministic pagination, restart safety, retry safety, duplicate detection, and no assumption that timestamps are unique.

### 8.2 Processing sequence

For each ordered observation:

1. validate source identity and map canonical fields;
2. begin the logical local transaction;
3. insert the immutable raw observation or establish its existing immutable reference;
4. record validation and lateness disposition;
5. if eligible, run the source-isolated frozen runtime causally;
6. insert the immutable prediction attempt, including withheld/error outcomes;
7. insert any lifecycle transition/event and lifecycle-state version;
8. insert audit records;
9. advance the source watermark;
10. commit.

Unique constraints make retries idempotent. The watermark may advance only in the same commit as the corresponding disposition and prediction attempt.

### 8.3 Failure after in-memory state advancement

Runtime state is not transactional. If inference advances in-memory state and the local commit fails, the worker must not continue with that runtime instance. It must:

1. mark the source worker unhealthy;
2. discard the affected runtime instance;
3. rebuild it from committed causal history/checkpoints;
4. retry from the last committed watermark;
5. verify deterministic replay before resuming.

### 8.4 Late rows

Use a bounded, versioned lateness policy:

```text
within reorder window
    preserve raw row and reorder before stateful processing

beyond reorder window
    preserve raw row
    disposition = LATE_QUARANTINED
    do not inject it retroactively into finalized online state
```

Historical reprocessing must create a separately identified replay/version. It must never overwrite online evidence.

### 8.5 Gaps and reconnects

A disconnect, process restart, or data gap does not create a lifecycle boundary. A long gap that resets the feature engine emits:

```text
MODEL_STATE_RESET_GAP
```

It does not emit `LIFECYCLE_CLOSED` or `LIFECYCLE_STARTED`. Exact RUL remains unavailable during required warm-up, with a structured reason.

---

## 9. Local evidence store

Use a migrated SQLite database initially with:

- WAL mode;
- foreign keys enabled;
- a busy timeout;
- one writer process;
- per-request read connections for the API;
- explicit migrations and schema versioning;
- bounded/paginated queries;
- UTC timestamps and immutable identifiers.

Minimum entities:

```text
deployment_manifests
deployment_config_versions
data_sources
source_column_mappings
source_health_events
source_watermarks
raw_observations
observation_dispositions
machine_registry
machine_state_versions
prediction_attempts
prediction_supersessions
lifecycle_records
lifecycle_manager_state_versions
lifecycle_events
endpoint_evidence
endpoint_classifications
target_truth_eligibility
target_truth
target_censoring
alerts
plant_evaluation_locks
plant_metric_snapshots
drift_snapshots
audit_log
```

### 9.1 Append-only evidence

Raw observations, prediction attempts, lifecycle events, endpoint evidence, classifications, eligibility decisions, target truth, evaluation locks, and audit rows are insert-only.

Do not use `INSERT OR REPLACE` for evidence. Corrections are new rows with:

```text
supersedes_id
correction_reason
corrected_by
corrected_at
```

The current legacy `readings` storage may remain for the legacy monitor but must not serve as the new plant-evidence ledger.

### 9.2 Prediction attempt contract

Persist every valid attempt, including available, withheld, and error states:

```text
prediction_id
deployment_id
source_key, source_row_id, ingestion_id
machine_uid, lifecycle_id
event_timestamp, observed_at, prediction_timestamp
health_state_model
health_state_manufacturer
health_state_independent_truth
connectivity_state
forecast_state
warning point/lower/upper/serviceability/withhold reason
critical point/lower/upper/serviceability/withhold reason
sensor values and quality flags
model version and artifact hash
runtime manifest identity
feature contract version
inference latency
```

Do not force multiple health authorities into one ambiguous field. If a derived display state is needed, persist `health_state_source` and keep the constituent states.

---

## 10. Runtime states and actionability

Keep these dimensions separate:

```text
Health:       NORMAL | WARNING | CRITICAL | UNKNOWN
Connectivity: ONLINE | STALE | OFFLINE
Forecast:     AVAILABLE | WITHHELD | INITIALIZING | ERROR
```

Withholding reasons include:

```text
INSUFFICIENT_HISTORY
MISSING_DATA
STALE_DATA
SENSOR_FAULT
OOD_INPUT
UNSUPPORTED_MACHINE
MODEL_ERROR
FEATURE_ERROR
MODEL_STATE_RESET_GAP
MODEL_ARTIFACT_MISMATCH
RUNTIME_HASH_MISMATCH
UNKNOWN
```

Unavailable or low-confidence exact RUL must not remain actionable. Clear point and interval outputs according to the frozen contract and show the reason. Never render unavailable values as zero, negative one, or infinity.

---

## 11. Lifecycle management

### 11.1 Separate manager state from record disposition

Machine-manager state:

```text
UNINITIALIZED
ACTIVE
RESET_CANDIDATE
CLOSURE_PENDING
QUARANTINED
```

Lifecycle-record status:

```text
ACTIVE
CLOSED_CONFIRMED
CLOSED_INFERRED
CENSORED
QUARANTINED
```

Do not put terminal record statuses into the live manager state machine.

### 11.2 Lifecycle record

Store:

```text
lifecycle_id, machine_uid, sequence_number
observed_start_timestamp, confirmed_start_timestamp
first and latest observation identities
valid/invalid counts and gap statistics
left-censoring and target-specific censoring
record status, closure reason, closure confidence
endpoint precision
manager/lifecycle/truth policy versions
created_at, updated_at
```

### 11.3 Startup

The first seen row is not assumed to be beginning-of-life. New machines begin with `left_censored = true` until an independent boundary is established.

### 11.4 Conservative reset policy

`CRITICAL -> NORMAL` alone never confirms a reset or closure. A gap, restart, sensor recovery, load change, calibration change, or feature-state reset alone also never confirms it.

Create a reset candidate from evidence such as persistent post-event baseline change plus supporting downtime/maintenance evidence. Require persistence and emit versioned evidence. Allow cancellation without rewriting history.

### 11.5 Closure confidence

```text
CONFIRMED   independent evidence reviewed and accepted
HIGH        strong inferred evidence; review only
MEDIUM      ambiguous; not target truth
LOW         weak; not target truth
UNRESOLVED  pending or quarantined
```

Only target-specific eligibility decisions based on independent evidence create plant accuracy truth. `CLOSED_INFERRED`, regardless of confidence, is excluded from initial plant accuracy.

### 11.6 Audit events

Emit append-only events such as:

```text
LIFECYCLE_STARTED
RESET_CANDIDATE_CREATED
RESET_CANDIDATE_CANCELLED
CLOSURE_PENDING
LIFECYCLE_CLOSED_CONFIRMED
LIFECYCLE_CLOSED_INFERRED
LIFECYCLE_CENSORED
LIFECYCLE_QUARANTINED
MODEL_STATE_RESET_GAP
ENDPOINT_EVIDENCE_ATTACHED
ENDPOINT_CLASSIFIED
TRUTH_ELIGIBILITY_DECIDED
```

---

## 12. Independent endpoint evidence workflow

Initial evidence entry and confirmation are local administrative operations, not unauthenticated network API mutations.

Required operations:

```text
add endpoint evidence
attach evidence to machine/lifecycle
classify endpoint
confirm or reject classification
decide WARNING eligibility
decide CRITICAL eligibility/censoring
correct by supersession
audit history
```

Required evidence fields include source system, external event ID, machine UID, event-time bounds, endpoint class candidate, component, maintenance type, operator/actor, attachments or reference, ingestion time, and provenance.

Idempotency uses `(source_system, external_event_id)` or another stable external identity.

---

## 13. Plant evaluation and support policy

### 13.1 Evaluation lock

Once an eligible lifecycle/target is admitted to frozen-v2.7 plant evaluation, write an immutable `PLANT_VALIDATION_LOCKED` record. It cannot be used to tune v2.7. Future-training eligibility is a separate field and is not assigned automatically.

### 13.2 Metric eligibility

Compute WARNING and CRITICAL separately. Exact point metrics use only:

- predictions actually persisted online before the target onset was known;
- serviceable outputs under the frozen contract;
- independently eligible exact target truth;
- non-superseded records;
- policy-compatible event timing;
- no inferred closures or censoring timestamps treated as onset.

Report MAE, median absolute error, signed bias, lifecycle-macro MAE, interval coverage, lifecycle-macro coverage, interval width, availability, service breadth, monotonicity, and predeclared horizon slices where eligible.

Always display prediction count, interval count, lifecycle count, source count, and machine count.

### 13.3 Predeclared support policy

Before the first plant metric is released, commit a versioned support policy defining minimum lifecycles and samples for:

- target-wide metrics;
- interval coverage;
- horizon buckets;
- source, line, and machine slices;
- drift alerts.

Until a metric meets its frozen support requirement, return:

```text
INSUFFICIENT_EVIDENCE
```

Do not choose support thresholds after observing results. Any later policy version applies prospectively and remains audited.

### 13.4 Separate online and replay results

```text
online_shadow_metrics  stored online predictions only
offline_replay_metrics explicitly labeled causal replay
```

Never merge these populations.

### 13.5 Drift

Distinguish:

```text
SYNTHETIC_TO_PLANT_DISTANCE
    frozen synthetic reference versus plant observations

WITHIN_PLANT_DRIFT
    plant baseline versus later plant windows
```

Monitor causal raw/features, missingness, cadence, selector/serviceability scores, predicted RUL, degradation bands, residuals only after eligible truth, and source/machine mix. Drift is diagnostic and does not trigger automatic retraining or threshold changes.

---

## 14. Security and deployment boundary

Initial release:

```text
backend bind            127.0.0.1 only
frontend access         localhost only
API                     read-only monitoring endpoints
source administration   local CLI
endpoint administration local CLI
```

Do not bind to `0.0.0.0` or expose mutation endpoints to a plant LAN without a separate authentication and authorization release.

Also require parameterized queries, validated SQL identifiers, restricted CORS, secret redaction, no arbitrary SQL endpoint, and no model replacement endpoint.

Source "delete" is archive/deactivate. Historical identities and evidence remain.

---

## 15. Read-only API

Expose versioned `/api/v1` read endpoints. Initial API must include:

```text
GET /overview
GET /machines
GET /machines/{machine_uid}
GET /machines/{machine_uid}/sensors
GET /machines/{machine_uid}/forecasts
GET /machines/{machine_uid}/events
GET /lifecycles
GET /lifecycles/{lifecycle_id}
GET /lifecycles/{lifecycle_id}/evidence
GET /alerts
GET /events
GET /sources
GET /sources/{source_key}
GET /sources/{source_key}/health
GET /model
GET /plant-evaluation
GET /drift
GET /system/health
GET /audit
```

Use pagination, bounded time ranges, server-side downsampling, explicit nulls, and stable reason codes. Never return credentials or raw secret references.

Administrative source testing, activation, evidence confirmation, and lifecycle adjudication remain CLI-only initially.

---

## 16. Frontend

Use React, TypeScript, Vite, TanStack Query, React Router, and an appropriate chart library unless repository inspection justifies an equivalent stack.

Initial pages:

```text
Overview
Machines
Machine Detail
Lifecycles
Alerts and Events
Data Sources / Source Health
Data Quality
Model Information
Plant Performance
Drift Monitoring
System Health
Audit
```

### 16.1 Required presentation rules

- Show health, connectivity, and forecast states separately.
- Show manufacturer and model health sources without conflating them.
- Show WARNING and CRITICAL forecasts separately.
- Show withholding reason and clear unavailable exact values.
- Do not connect sensor-chart lines across material data gaps.
- Show event, processing, and prediction times where diagnostically relevant.
- Label synthetic development, sealed synthetic, plant shadow, and production authorization separately.
- Plant performance begins as `INSUFFICIENT EVIDENCE`.
- Never display a generic `VALIDATED` badge that could imply plant validation.

### 16.2 Long history

Do not send unbounded raw history to the browser. Support resolution/downsampling while retaining immutable raw evidence in storage and preserving significant event points where practical.

### 16.3 Live updates

Use polling first. Do not add WebSockets/SSE until the persistence and API path is stable and measured.

### 16.4 Overview and machine priority

The overview must show total/online machines, NORMAL/WARNING/CRITICAL counts, available/withheld forecasts, source health, and active alerts. Its machine table includes source, line, health, connectivity, separate reliable WARNING/CRITICAL RUL summaries, forecast state, current sensors, trend, and last-seen time.

Default ordering is:

```text
manufacturer CRITICAL
then manufacturer WARNING
then lowest reliable target RUL
then NORMAL
then UNKNOWN/offline according to an explicit UI policy
```

An unavailable forecast must never sort as zero RUL.

### 16.5 Machine detail

Show machine/source identity, current lifecycle, health authorities, connectivity, last event/ingestion/prediction times, sensor values, target-specific forecasts and intervals, serviceability/withholding reasons, quality flags, and model/runtime identity.

Sensor charts cover VRMS, ARMS, APEAK, crest, and temperature with 1 h, 6 h, 24 h, 7 d, 30 d, full-lifecycle, and bounded custom ranges. Support zoom, timestamps, gap visualization, and event/lifecycle markers.

Target-specific RUL charts show point, lower/upper interval, withheld regions, event markers, model identity, and eligible retrospective truth only after it exists.

### 16.6 Lifecycle and evidence views

Lifecycle lists and detail views show manager state separately from record status, start/end bounds, censoring, endpoint precision, closure reason/confidence, evidence provenance, eligibility decisions for each target, online prediction history, and inclusion/exclusion reasons for plant metrics.

Timeline events include observed start, independent WARNING/CRITICAL onset, anomaly, forecast withholding/restoration, model-state gap reset, reset candidate, endpoint evidence, closure, censoring, and next lifecycle.

### 16.7 Alerts and events

Separate:

```text
Safety
    manufacturer WARNING/CRITICAL and independent plant safety evidence

Prognostic advisory
    reliable RUL review thresholds, rapid decrease, large uncertainty

Data/system
    disconnect, stale machine, sensor fault, missing rows, artifact mismatch,
    backlog, lifecycle ambiguity, runtime rebuild
```

Never render a model or infrastructure error as a physical machine failure.

### 16.8 Model, plant performance, source, and system pages

Model Information shows the model hash, runtime manifest, feature contract, load time, and separate statuses for synthetic development, sealed synthetic holdout, plant shadow, plant validation, and production authorization.

Plant Performance shows support counts and target-specific metrics or `INSUFFICIENT_EVIDENCE`; it exposes no numeric placeholder for unavailable plant MAE, bias, or coverage.

Source Health shows read-only status, query times, latest source event time, ingestion delay, cadence, machine counts, malformed/duplicate/late rows, gaps, errors, and last error without secrets.

System Health shows API/UI, pollers, source runtimes, lifecycle service, local database, writer queue, disk/memory, active manifest identity, artifact verification, prediction latency, backlog, last committed watermark, and latest successful prediction.

---

## 17. Repository compatibility requirements

Current code has these known constraints:

1. `VVB001Reading.source_id` is an integer row identifier.
2. `machine_key` is based on line and machine, not source.
3. legacy SQLite tables use `source_id INTEGER PRIMARY KEY` and replace-style writes;
4. `PostgresSource` is read-only but assumes a single integer `id > watermark` path;
5. the JSON checkpoint carries `last_source_id` and runtime baseline state separately from SQLite;
6. long feature gaps reset model state;
7. the historical v2.7 freeze does not itself enumerate every plant integration module.

Therefore:

- keep legacy paths compatible and add a distinct plant-shadow evidence store;
- adapt, do not repurpose, existing identifiers;
- reuse read-only query safeguards where appropriate;
- replace JSON-only checkpointing for plant shadow with transactional local state;
- represent feature-gap reset separately from lifecycle reset;
- use a new plant-shadow runtime manifest rather than rewriting historical evidence.

---

## 18. Suggested modules and commands

Adapt to the repository after inspection. A reasonable structure is:

```text
src/vvb001_monitor/plant_shadow/
    contracts.py
    manifest.py
    service.py
    runtime_router.py
    recovery.py
    evaluation.py
    drift.py

src/vvb001_monitor/plant_sources/
    registry.py
    postgres.py
    mapper.py
    validation.py
    polling.py

src/vvb001_monitor/evidence/
    database.py
    migrations/
    repositories.py
    endpoint_policy.py
    truth.py
    audit.py

src/vvb001_monitor/lifecycle/
    state.py
    manager.py
    reset_policy.py

src/vvb001_monitor/api/
    app.py
    routes/

frontend/
    src/api/
    src/components/
    src/pages/
    src/types/
```

Potential local commands:

```text
python main.py shadow-source add|validate|enable|disable|archive
python main.py shadow-run
python main.py shadow-status
python main.py lifecycle-status
python main.py lifecycle-audit
python main.py endpoint-evidence add|classify|confirm|supersede
python main.py finalize-lifecycle
python main.py plant-evaluate
python main.py serve-api --host 127.0.0.1
```

Reuse existing CLI conventions and do not break current commands.

---

## 19. Required tests

### 19.1 Identity and source safety

- duplicate machine IDs across sources remain isolated;
- durable composite ordering and duplicate timestamps;
- missing stable identity blocks activation;
- PostgreSQL `ctid` is rejected as authoritative identity;
- no plant write SQL;
- credentials never appear in API/log output;
- source archive preserves evidence references.

### 19.2 Transaction and recovery

- raw/disposition/prediction/lifecycle/watermark commit atomically;
- commit failure after inference discards and rebuilds runtime;
- retry produces one evidence record;
- crash/restart resumes from committed watermark;
- prediction history is append-only;
- correction supersedes rather than replaces;
- disconnect and restart do not create lifecycle boundaries.

### 19.3 Runtime isolation and causality

- interleaved machines are isolated;
- identical line/machine names in two sources are isolated;
- failure/rebuild of source A does not change source B;
- changing future rows cannot change output at time `t`;
- golden replay matches frozen behavior;
- manufacturer WARNING/CRITICAL remains immediate;
- artifact, manifest, and registry mismatches fail closed.

### 19.4 Lifecycle behavior

Test genuine maintenance, single spike, transient CRITICAL recovery, stuck sensor, communication gap, downtime without maintenance, stable post-maintenance baseline, sensor replacement, partial reset, machine replacement, and model-state gap reset.

Require that transient recovery, restart, gap, and `MODEL_STATE_RESET_GAP` do not create confirmed truth.

### 19.5 Target truth and censoring

- independent WARNING creates WARNING truth only;
- independent CRITICAL creates CRITICAL truth only;
- WARNING exact plus CRITICAL right-censored is valid;
- preventive replacement does not become CRITICAL onset;
- replacement/failure endpoints do not automatically become either target;
- `CLOSED_INFERRED` never enters accuracy;
- exact, bounded, date-only, and unknown precision behave correctly;
- exact fractional hours and timezones are correct;
- negative labels are rejected;
- truth and prediction remain separate;
- eligibility decisions are policy-versioned and deterministic.

### 19.6 Evaluation

- zero evidence returns `INSUFFICIENT_EVIDENCE`;
- support thresholds gate every aggregate and slice;
- WARNING and CRITICAL are separate;
- censored cases do not enter exact MAE/bias;
- lifecycle-macro and row-weighted metrics are distinguishable;
- online and replay metrics never mix;
- synthetic-to-plant distance and within-plant drift are separate.

### 19.7 API and frontend

- all initial API routes are read-only;
- pagination and bounded history work;
- source and machine filters work;
- chart gaps render correctly;
- withholding and unavailable states are explicit;
- evidence precision/censoring is visible;
- validation-domain labels cannot be conflated;
- the UI builds and tests cleanly.

### 19.8 Full regression

Run the complete existing suite, currently 144 tests at the accepted v2.7 baseline, plus all new backend, frontend, integration, and end-to-end tests. Do not weaken existing assertions.

### 19.9 Operational and performance tests

Measure sustained source polling at the declared plant requirement, inference latency, API response latency, writer-queue/backlog behavior, bounded per-machine runtime memory, database and disk growth, restart/rebuild time, source-failure isolation, and downsampled chart response size. Record requirements and results; do not trade evidence integrity for throughput.

---

## 20. End-to-end acceptance scenario

The local fake-source scenario must prove:

1. two sources with overlapping line/machine names are configured;
2. stable row identities and read-only access are validated;
3. each source receives a separate frozen runtime;
4. interleaved rows produce immutable online attempts;
5. one machine degrades while all others remain isolated;
6. a sensor anomaly and a disconnect do not close a lifecycle;
7. reconnect resumes from the committed watermark;
8. a feature gap emits `MODEL_STATE_RESET_GAP`, not a lifecycle reset;
9. independent WARNING onset is recorded and becomes WARNING truth;
10. preventive replacement is recorded before CRITICAL;
11. the lifecycle closes confirmed while CRITICAL is right-censored;
12. WARNING metrics update only if support policy permits;
13. CRITICAL exact MAE remains unavailable for that lifecycle;
14. a second lifecycle with independent CRITICAL onset creates CRITICAL truth;
15. inferred closure is visible but excluded from metrics;
16. a forced local commit failure rebuilds runtime without lost or duplicate evidence;
17. all historical predictions remain immutable;
18. the API and UI show correct state, provenance, support, and validation labels.

---

## 21. Implementation order and gates

Do not implement all phases as one uncontrolled patch. Each gate must pass before the next phase uses its output.

### Phase 0 — Contracts and freeze

- record repository inspection and reuse map;
- implement identity, timestamp, endpoint, censoring, precision, truth, and support-policy contracts;
- create schema and policy versions;
- create the plant-shadow runtime manifest;
- create golden replay fixture and verification;
- verify frozen artifact and evidence identities.

**Gate:** contract tests, hash verification, and golden replay pass. No source activation before this gate.

### Phase 1 — Append-only evidence storage

- migrations, WAL, foreign keys, busy timeout, one-writer rule;
- immutable evidence tables and supersession;
- atomic transaction boundary;
- audit and configuration versions.

**Gate:** transaction, rollback, idempotency, and migration tests pass.

### Phase 2 — One-source ingestion

- local source administration;
- read-only connection/schema/data validation;
- stable ordering, raw persistence, lateness, watermark;
- restart and duplicate handling.

**Gate:** one source survives disconnect/restart without loss, duplicate evidence, or false lifecycle change.

### Phase 3 — Frozen inference integration

- one source, one frozen runtime;
- canonical-to-legacy adapter;
- immutable prediction attempts;
- commit-failure runtime rebuild;
- fail-closed manifest checks.

**Gate:** golden replay and injected-failure recovery pass.

### Phase 4 — Multi-source routing

- `machine_uid` routing;
- source runtime isolation;
- source failure isolation;
- single-owner enforcement.

**Gate:** overlapping identities and interleaved-source tests pass.

### Phase 5 — Lifecycle segmentation without target truth

- manager state and lifecycle record status;
- left censoring;
- reset candidates, gaps, quarantine, audit;
- no target truth generation yet.

**Gate:** lifecycle scenario suite passes without prediction-derived truth.

### Phase 6 — Independent endpoint evidence

- CLI evidence entry/confirmation;
- endpoint classification and precision;
- target-specific eligibility and censoring;
- truth generation and evaluation locks.

**Gate:** eligibility matrix and censoring tests pass.

### Phase 7 — Plant evaluation and drift

- separate target metrics;
- support policy and insufficient-evidence states;
- online/replay separation;
- synthetic-to-plant and within-plant drift.

**Gate:** no unsupported, censored, inferred, or hindsight prediction can enter exact metrics.

### Phase 8 — Read-only API

- read routes, schemas, pagination, downsampling;
- localhost binding and CORS restrictions;
- credential redaction.

**Gate:** API mutation attempts fail and response/security tests pass.

### Phase 9 — Controlled local administration

- finalize source and endpoint CLI workflows;
- archive rather than delete;
- audited corrections/supersession.

**Gate:** every change is attributable, versioned, and recoverable.

### Phase 10 — React frontend

- build vertical slices against stable API contracts;
- monitoring, lifecycle, evidence, plant evaluation, drift, system pages;
- explicit state/source/validation semantics.

**Gate:** frontend unit/integration tests and production build pass.

### Phase 11 — Full verification

- full Python regression;
- frontend tests/build;
- end-to-end scenario;
- performance and restart tests;
- pre/post artifact and evidence hashes;
- documentation and startup commands.

**Gate:** all definition-of-done items pass. The result remains shadow-only.

---

## 22. Definition of done

Initial plant shadow is ready only when all are true:

1. Plant PostgreSQL access is demonstrably read-only.
2. Continuous sources require durable row identity.
3. `source_key`, `source_row_id`, `ingestion_id`, and `machine_uid` have distinct meanings.
4. Each source has an isolated frozen runtime.
5. Raw disposition, prediction attempt, lifecycle transition, audit, and watermark are transactionally consistent.
6. A failed commit after inference forces deterministic runtime rebuild.
7. Evidence and predictions are append-only; corrections supersede.
8. Frozen artifact, accepted evidence, runtime manifest, and golden replay verify at startup.
9. Manufacturer safety behavior is unchanged and immediate.
10. Health, connectivity, forecast, and health-state source remain distinct.
11. Unavailable exact forecasts are cleared and reason-coded.
12. Manager state and lifecycle record status are separate.
13. Runtime gap reset is not a lifecycle boundary.
14. `CRITICAL -> NORMAL` alone cannot confirm closure.
15. Endpoint evidence, classification, eligibility, and truth are separate records.
16. WARNING and CRITICAL truth are target-specific.
17. Preventive maintenance/replacement does not automatically become CRITICAL truth.
18. Precision and censoring are represented explicitly.
19. `CLOSED_INFERRED` is excluded from initial plant accuracy.
20. Metrics use stored online predictions and eligible independent truth only.
21. Support policy produces `INSUFFICIENT_EVIDENCE` when required.
22. Online/replay and synthetic/plant domains never mix.
23. API is read-only and bound to localhost initially.
24. Administrative operations are local, controlled, and audited.
25. Frontend displays all validation and availability distinctions honestly.
26. Existing 144-test baseline plus new tests pass.
27. Frontend tests and production build pass.
28. Frozen v2.7 artifact and historical evidence remain unchanged.
29. Plant-production authorization remains false.
30. No automatic retraining, promotion, threshold tuning, work order, shutdown, or plant write exists.

---

## 23. Deliverables

1. Repository inspection and reuse map.
2. Plant-shadow contracts and versioned policies.
3. Runtime manifest and golden replay fixture.
4. Migrated append-only evidence database.
5. Read-only source adapters and resilient ingestion service.
6. Source-isolated frozen inference router.
7. Persistent lifecycle and endpoint-evidence workflows.
8. Target-specific plant evaluation and drift reports.
9. Read-only FastAPI service.
10. React/TypeScript frontend.
11. Unit, integration, recovery, E2E, and frontend tests.
12. Updated README and exact local startup commands.
13. `PLANT_SHADOW_IMPLEMENTATION_SUMMARY.md` with test and hash evidence.
14. Explicit list of deferred retraining, promotion, LAN authentication, and control functionality.

---

## 24. Explicitly deferred

Do not implement or enable:

```text
automatic retraining or calibration
automatic training-pool ingestion
plant train/validation/test role assignment
candidate training or promotion
automatic model replacement
threshold tuning from plant evidence
online gradient updates
maintenance work-order creation
PLC or machine control
automatic shutdown or scheduling
plant-production authorization
LAN exposure without authentication/authorization
prediction-derived endpoint truth
```

---

## 25. Codex execution instruction

> Implement this v2 specification incrementally from Phase 0 through Phase 11. First produce the repository inspection/reuse map and verify the frozen v2.7 artifact and consumed evidence. Preserve all historical v2.7 files exactly. Keep plant PostgreSQL read-only. Use target-specific WARNING/CRITICAL truth, explicit censoring and precision, durable row identity, source-isolated frozen runtimes, and append-only transactionally consistent local evidence. Bind the initial API to localhost and keep source/evidence mutations in audited local CLI workflows. Do not retrain, recalibrate, retune, promote, authorize production, control machinery, schedule maintenance, write to the plant database, or manufacture truth from model outputs. At every phase gate, run the specified tests and stop on integrity failure. At completion, run the full existing and new suites, frontend tests/build, golden replay, injected recovery tests, and pre/post hash verification, then document exact commands and results.

---

## 26. Final design principle

```text
REAL SENSOR STREAM
    -> FROZEN, SOURCE-ISOLATED v2.7 SHADOW INFERENCE
    -> IMMUTABLE ONLINE PREDICTIONS
    -> PERSISTENT LIFECYCLE TRACKING
    -> INDEPENDENT ENDPOINT EVIDENCE
    -> TARGET-SPECIFIC ELIGIBILITY AND CENSORING
    -> RETROSPECTIVE WARNING / CRITICAL TRUTH
    -> SUPPORT-GATED PLANT EVALUATION
    -> ENGINEERING REVIEW
```

Never:

```text
MODEL OUTPUT -> MODEL-CREATED TRUTH -> AUTOMATIC RETRAINING OR CONTROL
```

The frontend exists to make plant-shadow evidence understandable without weakening causal, scientific, audit, security, or safety boundaries.

---

## 27. Operating-context addendum

The always-powered sensor condition is handled by the implemented
`plant_shadow_schema_v2_operating_context` contract documented in
`OPERATING_CONTEXT_IMPLEMENTATION.md`.

- Exact RUL is admitted only with high-confidence `RUNNING` evidence.
- `IDLE`, `OFF`, `MAINTENANCE`, `UNKNOWN`, missing, stale, conflicting, or low-confidence context
  pauses the forecast and clears actionable exact RUL fields.
- Non-running samples do not update the frozen model or advance lifecycle/RUL time.
- Immediate manufacturer extreme-raw safety behavior remains active in every operating state.
- Runtime aging and retrospective exact truth use confirmed operating exposure, not wall time.
- Duty-cycled mocks are runtime-test fixtures and are rejected by model training.

This addendum changes the plant-shadow observation contract and evidence path only. It does not
change the frozen v2.7 artifact, calibration, selectors, thresholds, manufacturer safety logic, or
plant-production authorization.

---

## 28. Vibration-only operating eligibility addendum

For a plant with no machine-state log, schema v3 implements the conservative fallback documented
in `VIBRATION_OPERATING_INFERENCE_IMPLEMENTATION.md`.

- Calibration is causal, machine-specific, bounded, and requires distinct quiet and
  production-like vibration support.
- The detector may infer only persistent high-confidence `RUNNING`; it never infers exact OFF,
  IDLE, or MAINTENANCE modes from aggregate vibration.
- Calibration gaps, single-regime histories, quiet, impulsive, extreme, unstable, and novel
  patterns fail closed to `UNKNOWN`, pause the runtime, and clear actionable exact RUL.
- PLC, CMMS, operator, and mapped database evidence always takes precedence.
- Every inference and calibration artifact hash is append-only evidence; duplicate and late rows
  cannot advance detector state.
- Pre-lifecycle calibration rows rebuild the detector only and are never replayed into degradation
  features.
- State-hidden duty-cycle fixtures are marked and rejected by training.

This addendum changes only runtime eligibility, evidence, API/frontend visibility, and mock testing.
It does not change the frozen v2.7 artifact, learned calibration, selectors, thresholds,
manufacturer safety behavior, or plant-production authorization.
