# Spindle Prognostics — IFM VVB001

A VVB001-native spindle condition-monitoring project for learning degradation patterns from `vrms`, `arms`, `apeak`, `crest`, and temperature. It uses synthetic lifecycles first to bootstrap and test the learning pipeline, then can run the same feature/model pipeline on the real VVB001 PostgreSQL data.

Expected PostgreSQL columns:

```text
id, timestamp, line_sel, machine_id, vrms, arms, apeak, crest, temp
```

PostgreSQL is **read-only**. All checkpoints, features, predictions, audits, SQLite data, and CSV exports are stored locally.

## Pipeline

```text
BOOTSTRAP DEVELOPMENT

improved synthetic VVB001 lifecycles
  - one line by default
  - multiple machine_id values
  - different healthy baselines per machine
  - bearing / unbalance / lubrication / looseness /
    thermal / sudden-impact / mixed degradation modes
        ↓
VVB001 validation
        ↓
per-machine causal feature engine
  - rolling statistics
  - EWMA / rates / trends
  - per-machine baseline deltas / ratios / z-scores
        ↓
NO NORMAL/WARNING/CRITICAL training labels
        ↓
unsupervised regime discovery
        ↓
early lifecycle used only as a healthy anchor
        ↓
learned regime ordering
        ↓
continuous degradation score 0..1
        ↓
learned NORMAL / WARNING / CRITICAL regimes
        ↓
models/vvb001_bootstrap.joblib

REAL RUNTIME

IFM VVB001 → PostgreSQL → SELECT only
        ↓
same validation + feature engine
        ↓
same learned degradation model
        ↓
smoothed degradation score + status
        ↓
dedicated learned RUL ML v2.1 with v2 + legacy trend fallback
  - hours to WARNING
  - hours to CRITICAL / remaining useful life
  - uncertainty range + reliability
        ↓
local SQLite / CSV / checkpoint only
```

## How the model learns the states

The mock training CSV deliberately has **no `health_status` column**. The learner does not receive fixed sensor thresholds such as `vrms > X = WARNING`.

Instead it:

1. Separates every lifecycle and every `(line_sel, machine_id)` stream.
2. Builds a short healthy reference from the beginning of each lifecycle.
3. Converts absolute readings into relative features such as deviation from that machine's baseline, ratios, z-scores, trends, and rolling variation.
4. Uses unsupervised clustering to discover three recurring operating/degradation regimes.
5. Identifies the healthiest discovered regime from the early-lifecycle anchor samples.
6. Orders the remaining regimes using where they naturally occur in the lifecycle.
7. Produces a continuous degradation score from 0 to 1 and maps the learned regimes to `NORMAL`, `WARNING`, and `CRITICAL`.
8. Smooths the live degradation score using elapsed time (not row count) and applies hysteresis around the learned regime boundaries so single noisy readings do not cause rapid status flicker.

The default early anchor is the first 15% of each bootstrap lifecycle. That is **not** a WARNING/CRITICAL threshold; it is only the assumption that a lifecycle begins after maintenance in a relatively healthy condition.

The synthetic generator also writes `latent_damage_score` and `fault_mode`, but they are **audit fields only**. They are never used to train the regime model.

## 1. Install

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 2. Generate improved mock lifecycles

Recommended default:

```powershell
python main.py generate-mock --lifecycles 50 --machines 6 --seed 42
```

or simply:

```powershell
python main.py generate-mock
```

Default output:

```text
data/vvb001_mock_training.csv
```

Columns:

```text
id
timestamp
line_sel
machine_id
vrms
arms
apeak
crest
temp
lifecycle_id
lifecycle_progress
latent_damage_score
fault_mode
operating_state
operating_state_source
operating_state_confidence
maintenance_event_id
operating_elapsed_hours
```

By default, `line_sel` is always `LINE_1` while multiple `machine_id` values are generated, matching the structure you described in the real database.

Useful options:

```powershell
python main.py generate-mock `
    --lifecycles 80 `
    --machines 8 `
    --line-sel LINE_1 `
    --cadence-seconds 60 `
    --seed 42
```

To generate a runtime-test fixture containing sensor rows while the spindle is idle, off, under
maintenance, or operationally unknown:

```powershell
python main.py generate-mock `
    --output data\vvb001_operating_context_test.csv `
    --lifecycles 8 `
    --machines 4 `
    --cadence-seconds 600 `
    --seed 812 `
    --duty-cycled-operating-context
```

This duty-cycled file is intentionally rejected by `train-mock`. It tests runtime gating and must
not be used to train, calibrate, or select a model.

For the no-machine-log path, add `--vibration-inference-fixture`. The generated CSV leaves visible
operating-state fields blank and retains `synthetic_true_operating_state` only for runtime test
assertions. This fixture is also rejected by training:

```powershell
python main.py generate-mock `
    --output data\vvb001_vibration_operating_test.csv `
    --lifecycles 8 `
    --machines 4 `
    --cadence-seconds 600 `
    --seed 913 `
    --duty-cycled-operating-context `
    --vibration-inference-fixture
```

## 3. Train the bootstrap degradation model

```powershell
python main.py train-mock
```

Outputs:

```text
models/vvb001_bootstrap.joblib
output/bootstrap_training_report.json
```

Or generate and train together:

```powershell
.\run_bootstrap_training.ps1
```

Training now uses three lifecycle-disjoint partitions: **fit**, **validation**, and untouched **test**. Rows from the same `lifecycle_id` are never mixed across these partitions. Fit rows are lifecycle-balanced so long trajectories cannot dominate the unsupervised clustering objective, with a modest `1 + 1.5 * progress^3` tail weight to stabilize the severe regime. `fault_mode` and `latent_damage_score` remain audit-only and do not affect fitting.

The WARNING score boundary remains cluster-derived. The CRITICAL boundary is conservatively calibrated on validation lifecycles only: the code searches downward from the original cluster midpoint and selects the highest boundary that reaches the held-out late-tail target without materially worsening transition reversals. The untouched test lifecycles are evaluated only after calibration.

The report does **not** report true plant FP/FN because the new learner is not trained from fixed NORMAL/WARNING/CRITICAL labels. Instead it reports useful bootstrap diagnostics such as:

- early-anchor alert rate — how often early lifecycle data is assigned WARNING/CRITICAL
- late critical coverage — how often the final 10% of synthetic lifecycles is assigned CRITICAL
- degradation-score vs lifecycle-progress rank correlation
- degradation-score vs hidden simulator damage rank correlation
- regime separation (`silhouette_score`)
- backward status-transition rate
- per-fault-mode early-alert, late-CRITICAL, correlation, and per-lifecycle results

These are still synthetic checks, not real-plant validation.

## 4. Automated generalization evaluation

After training, test the **saved model without retraining it** on fresh synthetic lifecycles generated from unseen random seeds:

```powershell
python main.py evaluate-generalization `
    --model models/vvb001_bootstrap.joblib `
    --report output/generalization_evaluation.json
```

For a model trained on 10-minute (`600` second) data, especially an older model created before cadence metadata was added, specify the cadence explicitly:

```powershell
python main.py evaluate-generalization `
    --model models/model_10min.joblib `
    --cadence-seconds 600 `
    --trials 5 `
    --lifecycles 30 `
    --machines 6 `
    --seed-start 1001 `
    --report output/generalization_10min.json
```

What the command does:

1. Loads the already-trained joblib bundle once.
2. Generates fresh synthetic lifecycle datasets using seeds that are different from the bootstrap-development default.
3. Runs the same validation and feature-engineering pipeline used for training/runtime.
4. Evaluates the fixed model on every unseen lifecycle; it does **not** fit K-means, reorder regimes, or tune boundaries during the test.
5. Writes per-trial metrics, aggregate stability statistics, each individual acceptance check, and an overall `PASS`/`FAIL` field to JSON.
6. Deletes temporary synthetic CSVs automatically unless `--keep-generated-dir` is provided.

Default synthetic-regime consistency gates are:

```text
early_anchor_alert_rate          <= 0.10
late_critical_coverage           >= 0.20
score_progress_spearman          >= 0.65
score_latent_damage_spearman     >= 0.70
transition_reversal_rate         <= 0.20
```

Every trial must satisfy every configured gate for `overall_pass=true`. These gates can be changed from the CLI for research/regression experiments, but do not loosen them merely to force a passing result.

The JSON also contains `aggregate_per_fault_mode`, making it easier to see whether a particular synthetic fault family is responsible for weak late-CRITICAL coverage.

### After upgrading from the previous model

Existing `.joblib` files keep their old score boundaries. **Retrain the model** to use the lifecycle-balanced fit and held-out CRITICAL calibration:

```powershell
python main.py train-mock `
    --input data\mock_10min.csv `
    --model models\model_10min.joblib `
    --report output\eval_10min.json
```

Then rerun the generalization evaluator. Do not reuse the old model file and expect the calibration fix to appear automatically.

Useful options:

```powershell
python main.py evaluate-generalization --help
```

Important limitations:

- This is a **synthetic generalization regression test**, not real-plant validation.
- `latent_damage_score` and `lifecycle_progress` are simulator-only evaluation fields and are never model inputs.
- A synthetic PASS does not establish real false-positive rate, real false-negative rate, maintenance lead time, or production safety.
- For plant validation, collect real lifecycle/reset/maintenance evidence and evaluate the fixed model against that independent evidence.

## 5. Versioned adversarial stress validation

`evaluate-generalization` tests fresh seeds from the bootstrap-generator family. The separate `stress.py` framework tests adversarial trajectories and sensor faults through the **same guarded runtime monitor path** used for live predictions.

`stress_v1`, `stress_v2`, `stress_v3`, and `stress_v4` have all now been consumed as development/acceptance evidence and are retained as **regression suites**. The RUL ML v2.1 update does not change the learned classifier, score smoothing, sensor-quality policy, or status thresholds.

The runtime now has three narrow protections that do not retrain or globally retune the learned model:

- sensor-fault rows quarantined by `SensorQualityGuard` use sanitized feature values;
- the predictor captures a **pre-suspect checkpoint** and serves that trusted state for the whole quarantine episode, then rolls back to it on sensor recovery;
- trusted sustained thermal evidence may bypass only the **upward hysteresis margin** after the learned degradation score has already reached the model's calibrated CRITICAL boundary. The learned boundary itself is not lowered.

Run consumed suites only as regressions:

```powershell
python main.py evaluate-stress `
    --model models\model_10min.joblib `
    --suite-dir data\stress\stress_v1 `
    --report output\stress\stress_v1_regression.json

python main.py evaluate-stress `
    --model models\model_10min.joblib `
    --suite-dir data\stress\stress_v2 `
    --report output\stress\stress_v2_regression.json

python main.py evaluate-stress `
    --model models\model_10min.joblib `
    --suite-dir data\stress\stress_v3 `
    --report output\stress\stress_v3_regression.json
```

`stress_v4` was previously run as the untouched acceptance suite before this RUL-only addition. Its frozen data remain included for classifier/status regression:

- generator: `independent_adversarial_v4`
- seed: `20260907`
- cadence: 600 seconds
- scenarios: 16
- lifecycles: 32
- rows: 12,888
- current evidence role in this snapshot: `development_regression_consumed_v4`

Run v4 as a regression check for the unchanged classifier/status path:

```powershell
python main.py evaluate-stress `
    --model models\model_10min.joblib `
    --suite-dir data\stress\stress_v4 `
    --report output\stress\stress_v4_acceptance.json
```

The model-facing sensor CSV never contains hidden true state/damage/scenario/fault labels. SHA-256 hashes in every manifest are verified before evaluation. The prior v4 acceptance result applies to the unchanged classifier/status path; it does not validate the newly added RUL hours. Real RUL accuracy requires plant maintenance/failure timing evidence. None of these synthetic suites establish plant accuracy. See `STRESS_TEST.md`, `THERMAL_BIAS_REMEDIATION.md`, `PREDICTION_CHECKPOINT_REMEDIATION.md`, and `RUL_ESTIMATION.md`.

## 6. Per-machine baseline behavior

Every machine is isolated using:

```text
(line_sel, machine_id)
```

For example:

```text
LINE_1::MACHINE_01
LINE_1::MACHINE_02
LINE_1::MACHINE_03
```

Each stream has separate rolling history, EWMA state, and baseline statistics. A naturally higher-vibration machine therefore does not automatically inherit another machine's baseline.

The local checkpoint also stores the frozen per-machine baseline statistics, so restarting the program does not redefine the baseline from scratch. Recent PostgreSQL history is replayed only to rebuild rolling/EWMA state.

## 7. Configure PostgreSQL

Edit:

```text
config/vvb001.json
```

Default mapping:

```json
{
  "source_id": "id",
  "timestamp": "timestamp",
  "line_sel": "line_sel",
  "machine_id": "machine_id",
  "vrms": "vrms",
  "arms": "arms",
  "apeak": "apeak",
  "crest": "crest",
  "temp": "temp"
}
```

If the real column names differ, change only this mapping.

Set the acceleration unit correctly:

```json
"acceleration_unit": "m_s2"
```

or:

```json
"acceleration_unit": "g"
```

The model and live runtime must use the same feature configuration used during training.

Keep credentials outside Git:

```powershell
$env:VVB001_POSTGRES_DSN = "postgresql://READ_ONLY_USER:PASSWORD@HOST:5432/DATABASE"
```

Use a PostgreSQL account with `SELECT` permission only when possible. The application additionally enables a read-only PostgreSQL session and contains no PostgreSQL write path.

## 8. Run on the real database

With the bootstrap model:

```powershell
python main.py monitor-postgres `
    --config config/vvb001.json `
    --model models/vvb001_bootstrap.joblib
```

or:

```powershell
.\run_vvb001_monitor.ps1
```

The runtime warns that a synthetic-trained model is provisional.

To collect real VVB001 data without using any model:

```powershell
python main.py monitor-postgres --collect-only
```

To catch up currently available rows once and exit:

```powershell
python main.py monitor-postgres --once
```

For the first deployment, if you intentionally want to process all historical rows and there is no checkpoint yet:

```powershell
python main.py monitor-postgres --start-from-beginning --once
```

## 9. Local outputs

Default files:

```text
output/vvb001_monitor.db
output/vvb001_checkpoint.json
```

The local SQLite database stores:

- original VVB001 readings
- validation status
- engineered features
- learned `NORMAL/WARNING/CRITICAL` regime
- class-like regime probabilities
- continuous `degradation_score`
- estimated hours to WARNING
- estimated hours to CRITICAL / operational RUL
- RUL lower/upper ranges, reliability, trend diagnostics and availability reason
- sensor-quality/prediction-state audit fields
- invalid-row audit data

The checkpoint stores the last PostgreSQL `id` and the per-machine baseline state locally.

## 10. Inspect real collected data

Latest status and RUL for every monitored machine:

```powershell
python main.py show-latest
```

Example:

```text
LINE_1::VVB001 @ 2026-08-11T04:00:00+00:00
  Status: NORMAL; degradation_score=0.22
  Estimated time to WARNING: 12.5 h
  Estimated time to CRITICAL / RUL: 31.0 h
  RUL reliability: MEDIUM
```

`RUL` here is produced by the dedicated learned RUL ML v2.1 in newly trained bundles, using only causal sensor/degradation history. V2 artifacts retain their original behavior, while older bundles without learned RUL fall back to the trend estimator. It is **not yet a plant-validated physical lifetime prediction**. Sensor-quality quarantined rows cannot advance learned RUL state. See `RUL_ESTIMATION.md`.

Aggregate and per-machine profile:

```powershell
python main.py profile-local
```

Export local readings/features/predictions/RUL:

```powershell
python main.py export-csv
```

Default export:

```text
output/vvb001_training_features.csv
```

## 11. Moving from synthetic bootstrap to real learning

The bootstrap model proves the architecture and provides an initial degradation representation. It should not be treated as a validated maintenance decision system.

The intended next progression is:

```text
synthetic bootstrap lifecycles
        ↓
learn/test degradation representation
        ↓
connect to real PostgreSQL read-only
        ↓
collect historical VVB001 data per machine
        ↓
identify real lifecycle/reset/maintenance boundaries
        ↓
fit the same lifecycle-aware learner on real lifecycles
        ↓
compare learned regimes with maintenance/failure evidence
        ↓
validate real false alarms, missed faults, and warning lead time
        ↓
promote a real-data model
```

A statistical regime can be discovered without manual labels, but calling it operationally `CRITICAL` still requires plant/maintenance validation before production use.


## Test RUL before PostgreSQL

**Retrain after this update.** Existing joblib bundles remain loadable but use the legacy trend RUL fallback because they do not contain the learned RUL v2 artifact.

A live database is not required to validate the new RUL output. Generate a fresh unseen synthetic
lifecycle set and evaluate the fixed model without retraining:

```powershell
python main.py generate-mock `
    --lifecycles 30 `
    --machines 6 `
    --cadence-seconds 600 `
    --seed 12001 `
    --output data\rul_test_12001.csv

python main.py evaluate-rul `
    --model models\model_10min.joblib `
    --input data\rul_test_12001.csv `
    --report output\rul\rul_test_12001.json `
    --predictions output\rul\rul_test_12001_predictions.csv
```

See `RUL_EVALUATION.md` for metrics, stress-suite evaluation, truth isolation, and interpretation.

## RUL v2.6 target-specific development workflow

RUL v2.6 keeps the v2.5 point models frozen and separates WARNING and CRITICAL
forecastability, bias correction, asymmetric calibration, and runtime state. Its acceptance file
is generated only after an exact final-artifact serialize/reload calibration replay passes:

```powershell
python main.py generate-rul-v2-6-preacceptance
python main.py train-rul-v2-6
python main.py generate-rul-v2-6-acceptance
python main.py evaluate-rul-v2-6-development
```

Do not run the acceptance evaluation more than once. It permanently consumes that evidence.
The full-cadence remediation passed exact final-artifact parity, target coverage, interval width,
availability, breadth, support, and every horizon-bias gate except CRITICAL `<=6 h` (`+7.876 h`
against `+/-6 h`). Its acceptance remains physically ungenerated and no sealed holdout is
authorized. See `RUL_V2_6_IMPLEMENTATION_SUMMARY.md`.

## RUL v2.7 identity-first CRITICAL remediation

RUL v2.7 freezes the v2.6 Extra Trees estimators, target selectors, WARNING pipeline, features,
generator, runtime state machine, and manufacturer overrides. CRITICAL identity is compared with
one zero-anchored bounded correction and identity wins unless the learned alternative dominates
across every supported horizon:

```powershell
python main.py generate-rul-v2-7-preacceptance
python main.py train-rul-v2-7
python main.py generate-rul-v2-7-acceptance
python main.py evaluate-rul-v2-7-development
```

Fresh full-cadence evidence selected identity. Exact parity passed across 21,749 calibration rows,
all preacceptance gates passed, and the one-time 19,534-row development acceptance passed. Final
acceptance CRITICAL MAE was `3.534 h`, `<=6 h` bias was `+0.500 h`, coverage was `94.22%`, and
monotonicity was `98.64%`. WARNING, CRITICAL, and system sealed-holdout creation are authorized,
and the authorized sealed holdout has now been consumed exactly once:

```powershell
python main.py generate-rul-v2-7-sealed-holdout
python main.py evaluate-rul-v2-7-sealed-holdout
```

The 8-batch, 64-lifecycle, 41,239-row sealed realistic-synthetic holdout passed every target and
runtime gate. Sealed WARNING MAE was `2.756 h`; sealed CRITICAL MAE was `3.582 h`, with `+0.521 h`
bias at `<=6 h`. The synthetic candidate is locked and both development acceptance and sealed
evidence are permanently consumed. Plant shadow validation has not started and plant-production
use remains unauthorized. See `RUL_V2_7_IMPLEMENTATION_SUMMARY.md` and
`RUL_V2_7_SEALED_HOLDOUT_SUMMARY.md`.

## Plant shadow, lifecycle evidence, API, and frontend

The plant-shadow path wraps the frozen v2.7 artifact without changing its learned parameters. It uses a separate
append-only SQLite evidence ledger, one stateful runtime per `source_key`, independent WARNING and
CRITICAL truth policies, explicit censoring, and a localhost-only read API. It remains observational:
plant-production authorization is false and there is no automatic retraining or machine control.

Sensor connectivity and machine operation are separate contracts. Every raw sensor row is retained,
but only `RUNNING` observations from a declared source with confidence at least 0.90 enter the
feature/predictor/RUL runtime. `IDLE`, `OFF`, `MAINTENANCE`, low-confidence, contradictory, and
`UNKNOWN` rows produce a `PAUSED` forecast with exact RUL values cleared. Extreme raw safety evidence
remains immediate and does not advance degradation state. RUL values are explicitly operating hours,
not calendar hours.

When no machine-state database exists, leave the optional operating-context source mappings as
`null`, as shown in `config/plant_shadow_sources.example.json`. A causal machine-specific detector
then learns distinct quiet and production-like vibration regimes. It may infer only high-confidence,
persistent `RUNNING`; uncalibrated, quiet, unstable, impulsive, extreme, or unfamiliar patterns stay
`UNKNOWN` and keep RUL paused. Authoritative PLC/CMMS/operator evidence always wins. Vibration alone
never claims `OFF`, `IDLE`, or `MAINTENANCE`, because those modes are not identifiable reliably from
the five aggregate sensor fields.

Install backend/API and frontend dependencies:

```powershell
.\setup_office_laptop.ps1
```

On Windows this creates the Python 3.14 virtual environment, uses the validated Python constraints
in `requirements-office-lock.txt`, installs locked frontend dependencies, tests/builds the dashboard,
and runs the offline office plant-shadow checks. See
`OFFICE_DATABASE_TESTING_QUICKSTART.md` for the bounded live-database workflow. The test script uses
project-local pytest scratch space for compatibility with restricted corporate temp directories.

Verify the immutable runtime boundary before adding a source:

```powershell
.venv\Scripts\python.exe main.py plant-shadow verify-golden
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from vvb001_monitor.plant_shadow.manifest import verify_manifest; print(verify_manifest('.', 'output/plant_shadow/plant_shadow_runtime_manifest.json')['deployment_id'])"
```

Copy `config/plant_shadow_sources.example.json` to a local untracked configuration file. Keep the
DSN itself in the named environment variable; do not put a password in JSON or SQLite:

```powershell
$env:VVB001_PLANT_SOURCE_A_DSN = "postgresql://readonly_user:password@host/database"

.venv\Scripts\python.exe main.py plant-shadow source-save `
    --sources config\plant_shadow_sources.local.json `
    --actor "engineer-name"

.venv\Scripts\python.exe main.py plant-shadow ingest `
    --sources config\plant_shadow_sources.local.json `
    --once
```

Remove `--once` only after the source passes read-only schema and durable-identity validation.
The service verifies the manifest and golden replay before connecting, resumes from a composite
timestamp/row watermark, and rebuilds a source runtime from committed evidence after a failed
local transaction.

For a large live database, no export or lifetime backfill is required. The default source
configuration reads only the latest 24 source-hours on first connection, in bounded 1,000-row
batches, then continues from the durable watermark. Every query has a 30-second statement timeout.
The connector requires a PostgreSQL index beginning with `(timestamp, durable_row_id)`; it reports a
clear error if the index is absent and never creates it with the SELECT-only account. See
`LARGE_DATABASE_DIRECT_INGESTION.md`.

GPU training is intentionally not part of this path. The accepted v2.7 Extra Trees artifact remains
frozen, the unlabeled plant stream is not training data, and the operating detector uses bounded
causal statistics. Database indexing and bounded streaming address the actual large-table
bottleneck without adding a second model stack.

Operating state may come from optional read-only PostgreSQL columns configured in
`plant_shadow_sources.example.json`, or from a separate local PLC/CMMS/operator interval. For
example, record a confirmed maintenance window without modifying the sensor table:

```powershell
.venv\Scripts\python.exe main.py plant-shadow operating-state-add `
    --source-system CMMS `
    --external-event-id WO-2026-0814-001 `
    --machine-uid "plant-source-a::LINE_1::MACHINE_1" `
    --operating-state MAINTENANCE `
    --operating-state-source CMMS `
    --confidence 1.0 `
    --effective-from "2026-08-14T08:00:00+08:00" `
    --effective-to "2026-08-14T12:00:00+08:00" `
    --maintenance-event-id WO-2026-0814-001 `
    --actor "engineer-name"
```

Overlapping local evidence with contradictory states resolves to `UNKNOWN` and withholds RUL.
Confirmed component replacement closes the old lifecycle; the next confirmed-running row starts a
clean lifecycle and operating clock. Inspection, idle time, shutdown, and unclassified maintenance
do not silently reset spindle life.

Independent endpoint evidence is entered locally. An exact timestamp needs equal lower and upper
values. Classification applies target-specific truth/censoring policy; replacement does not
automatically become CRITICAL truth:

```powershell
.venv\Scripts\python.exe main.py plant-shadow evidence-add `
    --source-system CMMS `
    --external-event-id EVENT-123 `
    --machine-uid "plant-source-a::LINE_1::MACHINE_1" `
    --lifecycle-id "lc_..." `
    --precision EXACT_TIMESTAMP `
    --time-lower "2026-08-13T10:00:00+08:00" `
    --time-upper "2026-08-13T10:00:00+08:00" `
    --actor "engineer-name"

.venv\Scripts\python.exe main.py plant-shadow evidence-classify `
    --evidence-id "evidence_..." `
    --endpoint-class PREVENTIVE_MAINTENANCE `
    --actor "engineer-name"

.venv\Scripts\python.exe main.py plant-shadow evaluate
```

Start the read-only local dashboard services:

```powershell
.venv\Scripts\python.exe main.py plant-shadow serve-api --host 127.0.0.1
Set-Location frontend
npm.cmd run dev
```

The API refuses a non-loopback host in this release. Source and endpoint mutations are CLI-only.
See `VVB001_PLANT_SHADOW_LIFECYCLE_FRONTEND_SPEC_V2.md`,
`VIBRATION_OPERATING_INFERENCE_IMPLEMENTATION.md`, `PLANT_SHADOW_REPOSITORY_INSPECTION.md`, and
`PLANT_SHADOW_IMPLEMENTATION_SUMMARY.md`.

### Disposable PostgreSQL end-to-end verification

Before any approved plant connection, run the isolated two-source native PostgreSQL scenario. The
harness uses either a dedicated database on an existing loopback PostgreSQL server or a disposable
native `initdb`/`pg_ctl` cluster on `127.0.0.1:55432`. It creates a separate SELECT-only application
role and uses overlapping line/machine names across two schemas.

Existing local server mode:

```powershell
$env:VVB001_E2E_ADMIN_DSN = "postgresql://fixture_admin:password@127.0.0.1:5432/postgres"
.venv\Scripts\python.exe tests\e2e\run_plant_shadow_postgres_e2e.py
Remove-Item Env:VVB001_E2E_ADMIN_DSN
```

The supplied admin must be limited to the local E2E server and capable of creating/dropping the
strictly named `vvb001_e2e_*` fixture database and its dedicated reader role. The harness refuses
remote hosts, broad database names, an existing target database, or an existing derived reader role.
It drops only the database and role it created unless `--keep-database` is explicitly supplied.

Disposable native-cluster mode requires installed PostgreSQL server binaries. They are detected from
`PATH` and standard Windows PostgreSQL directories, or may be supplied explicitly:

```powershell
.venv\Scripts\python.exe tests\e2e\run_plant_shadow_postgres_e2e.py `
    --pg-bin-dir "C:\Program Files\PostgreSQL\18\bin" `
    --port 55432
```

Startup is non-blocking and bounded: `pg_ctl -W start` returns after handoff, `pg_isready` polls for
at most 60 seconds, and all utility/database operations have finite timeouts. Stage messages are
flushed to the console. The harness refuses an occupied port, records the temporary cluster PID and
data directory, and stops only that owned cluster on success, failure, or interruption.

The harness verifies real PostgreSQL schema/identity checks, composite polling, bounded late-row
capture, duplicate idempotency, source-isolated frozen runtimes, restart rebuild, typed operating
state, OFF/MAINTENANCE exclusion, pause/resume exposure time, feature-gap versus lifecycle semantics,
independent WARNING/CRITICAL evidence, preventive-maintenance censoring,
support-gated metrics, read-only API state, and source-level failure isolation. Fixture provisioning
uses the fixture/admin connection; the application itself uses only the generated SELECT-only reader,
whose attempted write must fail. No Docker, Testcontainers, or container dependency is used.

To include this scenario in pytest explicitly:

```powershell
$env:VVB001_RUN_NATIVE_POSTGRES_E2E = "1"
.venv\Scripts\python.exe -m pytest -q tests\test_plant_shadow_postgres_e2e.py
Remove-Item Env:VVB001_RUN_NATIVE_POSTGRES_E2E
```

The report is written to `output/plant_shadow/postgres_e2e_report.json`. This is strictly a
disposable local integration test and is not plant validation. The PostgreSQL 18 Windows run on
port 55432 completed with `[E2E] PASS`; the owned PID terminated and the port was released.
