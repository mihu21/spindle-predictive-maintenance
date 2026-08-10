# Realistic-data readiness remediation report

## Current model-generalization attempt (2026-08-07)

The authoritative current model-quality report is
`docs/model_generalization_remediation.md`. The research candidate was rejected
and isolated; `models/realistic` was not replaced. The only current required
probability targets are `probability_warning_12h` and
`probability_critical_24h`.

## Forecast FP/FN and contract remediation (2026-08-06)

This pass supersedes the older all-target-mandatory forecast addenda below.
The authoritative report, exact metrics, limitations, and commands are in
`docs/forecast_policy_remediation.md`.

Root causes: the historical 87.561% FP was an actionable urgency/recommendation
rate while every manufacturer/status layer was NORMAL; the policy admitted
unapproved forecast sources. The current null-threshold failure was a metadata
layout mismatch plus an all-eight-target mandatory contract. Runtime now uses
trainer-produced metadata schema 3.0 / policy contract 3.0, numeric per-target
thresholds, safe deterministic legacy migration, two explicitly configured
required planning targets, and isolated optional-target failures.

Modified implementation files and purpose:

| File | Change |
|---|---|
| `config/ml.json`, `config.py` | Explicit required probability/ETA target configuration and validation |
| `policy_contract.py` | Single schema/version/required-target contract |
| `forecast_policy.py` | Safe legacy normalization, numeric evidence validation, per-target eligibility, required-only global gating |
| `retraining.py` | Validation-only FN-first/FP-second threshold selection, event evidence, full counts/metrics, ETA evidence, schema serialization |
| `model_registry.py` | Canonical normalization on load and schema propagation during promotion |
| `ml_forecaster.py`, `models.py` | Independent target evidence and raw/reconciled/status-constrained probability layers |
| `monitor.py` | Probability and ETA action rules require their own target eligibility |
| `storage.py` | Stable audit serialization for all probability layers and contract versions |
| `evaluate_monitoring.py` | External label alignment, model-only versus strict operational metrics, event lead time, false-alert episodes, CSV/JSON reports |
| `validate_forecast_contract.py` | Standalone machine-readable candidate contract validator |
| `tests/test_forecast_contract_v3.py` and existing policy tests | Migration, schema rejection, serialization, optional isolation, reconciliation, elapsed-time labels, withheld-as-miss regressions |
| `models/realistic/candidate/*` | Retrained candidate artifacts and trainer-produced schema-3.0 metadata |

Verification: 155 tests and 2 subtests pass; compilation passes; the metadata
contract validator passes; both final stress replays contain 100,000 valid and
zero invalid rows; synthetic-to-plant promotion is refused. Immediate
manufacturer status FN remains 0%. The synthetic model is not production-ready:
model-only NORMAL-stress FP and short-horizon FN remain high and are reported,
not hidden by operational withholding.

## Forecast-policy hardening addendum (2026-08-06)

The probability decision boundary is now versioned in model metadata
(`probability_thresholds.schema_version=1.0`) and is the only boundary used by
live urgency logic. Selection applies `FN rate <= configured maximum`; amongst
valid candidates it chooses lowest FP, then nearest configured default, then
lower threshold. A target with no valid threshold or an FN ceiling miss is
ineligible and cannot create an actionable recommendation. The live policy is
fail-closed for non-deployed/non-production/ineligible/wrong-domain/revoked
metadata, malformed thresholds, invalid probability/ETA values, inconsistent
warning/critical ETAs, OOD, and immature feature windows. Statistical forecasts
remain diagnostics-only (`statistical_fallback_actionable=false`).

`evaluate_monitoring.py` now reports current-status FP using NORMAL rows and FN
using WARNING/CRITICAL rows, preserves source-label precedence, fails when no
ground truth exists, and separates predictive false alerts, misses, episodes,
and trusted/withheld coverage. The bundled synthetic candidates remain
non-actionable and are excluded from this delivery package; no plant data or
generated evaluation output is included.

### Artifact completeness and portable-package addendum

Production readiness now compares `required_model_targets` with successfully
loaded estimators. Missing, unreadable, or mismatched artifacts fail closed and
are retained as structured audit reasons. `artifact_targets` records expected
file-to-target identity. Threshold selection is validation-only; the unchanged
selected threshold must also meet the FN ceiling (`<=`) on the untouched test
split or the required target is rejected as `test_fn_ceiling_not_met`.

Detailed CSV output now contains stable JSON policy audit fields, including
required/loaded/missing targets, selected thresholds, target eligibility,
physical and feature readiness reasons, and explicit recommendation actionability.
The evaluator fails closed when any trusted-coverage audit field is absent.
Predictive false alerts use NORMAL rows as denominator and missed unsafe rows
use WARNING/CRITICAL rows. Episodes split at lifecycle boundaries and gaps over
ten minutes; repeated rows in an uninterrupted episode count once.

Final validation: source workspace `unittest discover` passed 118 tests and
`compileall src evaluate_monitoring.py` passed. A fresh extracted clean archive
also passed all 118 tests, compilation, and `import spindle_monitor`. This is
evidence of fail-closed packaging and policy behavior, not predictive accuracy.

### Final auditability hardening

No-estimator loading no longer bypasses policy evaluation: valid metadata is
retained, every required artifact is listed as missing, and the resulting
forecast contains the model/version/domain audit context. Forecast decisions
retain a primary withholding reason for concise display plus the complete
deduplicated reason list through replay storage. Per-target ineligibility is a
mapping sourced from target metadata, not a copy of model-level reasons.

Trusted evaluator rows now parse and cross-check artifact lists, threshold and
target mappings, policy/schema versions, validation fields, runtime guardrails,
withholding state, and ML recommendation identity. Contradictory or malformed
evidence is untrusted. These controls and fixtures validate policy mechanics;
horizon-aware plant-performance metrics still require chronological labelled
plant data and no withholding result demonstrates predictive accuracy.

### Production-evidence and horizon-evaluation completion (2026-08-06)

Root causes were missing explicit deployment-state evidence, Boolean-only trust
decisions, alphabetical primary reasons, and same-row unsafe "miss" counting.
Runtime forecasts now carry revoked, superseded, corrupted, metadata-valid,
artifact-valid, schema-compatible and feature-schema-compatible state plus all
load failures. CSV keeps legacy delimited reasons and adds deterministic JSON
arrays/objects for every complete reason and per-target evidence collection.

Primary reason priority is safety-oriented: corruption, load failure, missing
required artifact, invalid metadata, production ineligibility, revocation,
supersession, invalid probability/ETA, ETA inconsistency, OOD, support,
disagreement, feature schema, sample count, window maturity, then coverage.
Unknown reasons remain preserved after known reasons in lexical order.

Trusted coverage independently parses and cross-checks deployment state,
artifact set inclusion, empty load/missing lists, schemas, thresholds, per-target
eligibility/reasons, validation and untouched-test FN gates, physical/readiness
evidence, OOD/support/disagreement, withholding state and actionable ML identity.
Absent or malformed evidence is untrusted and reported separately.

WARNING and CRITICAL ground-truth episodes are constructed separately and split
at state transitions, lifecycle changes and gaps over the configured maximum.
For H=6/12/24 hours, an episode is eligible only with H hours of continuous
labelled pre-onset history. The earliest trusted actionable ML prediction in
`[onset-H, onset)` determines lead time; onset/post-onset, statistical, withheld,
untrusted, cross-lifecycle and cross-gap rows never count. Insufficient-history
episodes are excluded rather than called misses. False-alert episodes are
contiguous trusted actionable ML rows during actual NORMAL, split by lifecycle,
gap or any non-actionable row.

The added adversarial tests cover deployment contradictions, corrupted/revoked/
superseded evidence, load-set inconsistency, target FN evidence, deterministic
reason priority, horizon matching, onset exclusion, insufficient history and
lifecycle-aware false-alert grouping. These policy fixtures do not establish
plant-model accuracy; synthetic models remain non-actionable, zero trusted
coverage is not zero FN, and meaningful horizon metrics require chronological
labelled plant data.

### Historical / superseded: all-eight-target mandatory pass

This section describes an older contract and is not current. The ETA-only bypass existed because model-controlled metadata could shrink the
required target list. `mandatory_production_targets()` now defines both ETA
targets and all six 6/12/24-hour probability targets in trusted code. Effective
requirements are the union of that set and metadata additions. Missing metadata
declarations, estimators, threshold mappings or target evidence each fail closed
and remain visible in configured/metadata/effective/missing audit arrays. Partial
registry promotion may preserve artifacts for diagnostics, but cannot mark the
deployment production-eligible until the mandatory contract is complete.

Probability evidence is now numerically checked: every threshold/FP/FN/ceiling
must be finite in [0,1], FN gates are recomputed with the exact `rate <= ceiling`
rule, recorded Booleans must agree, validation/test thresholds must match within
1e-12, and baseline/calibration/eligibility/reason evidence must all pass.

WARNING onset is strictly NORMAL to WARNING; CRITICAL onset is NORMAL or WARNING
to CRITICAL; combined unsafe onset is only NORMAL to WARNING/CRITICAL. WARNING
and combined candidates must come from actual NORMAL rows. CRITICAL uses
escalation mode (NORMAL or WARNING, never CRITICAL). A return to an allowed safer
state confirms closure; EOF, timestamp gaps and lifecycle boundaries censor open
episodes with an explicit reason. Censored onsets remain recall-eligible when
continuous same-lifecycle history is sufficient, but duration/recovery inference
is not made. Per-horizon detail reports available history and whether a gap,
lifecycle boundary or simple insufficient history caused ineligibility.

The evaluator now reports explicit current-status FP/FN counts, denominators and
rates, plus probability availability, threshold crossing, trusted actionability,
withholding, missing-model and ineligible-target coverage by target/horizon.
Policy fixtures validate mechanics only: synthetic models remain non-actionable,
censored episodes lack confirmed closure, zero trust is not zero FN, and plant
accuracy requires chronological labelled plant evidence.

### Runtime-ceiling, strict-audit and chronology pass

The blocking `_005` defect was that model metadata controlled its own acceptable
FN ceiling. Runtime and plant promotion now use
`min(runtime_configured_ceiling, model_recorded_ceiling)`. Metadata may be
stricter, never looser; a recorded ceiling above the live maximum is explicitly
rejected. Recorded Booleans are still checked against recorded rates, while
actual acceptance is recomputed against the effective ceiling. Per-target audit
includes runtime, model-recorded and effective ceiling mappings.

Every effective required target must carry the actual JSON Boolean
`target_required=true`. Required/configured/metadata/effective/loaded/missing
sets are recomputed and cross-checked. Per-target reasons must be arrays of
strings, eligibility values actual Booleans, thresholds finite numeric values,
and complete reason fields arrays of strings. Production thresholds use the
consistent strict range `0 < threshold < 1`.

Timestamp continuity now fails closed on missing, malformed, duplicate and
non-monotonic values. Such rows split the segment, and history on opposite sides
cannot reconnect. Evaluation reports invalid/missing/duplicate/non-monotonic
counts, broken segments, and horizon exclusions with the continuity reason.

`recommendation_trigger_targets` records only eligible probability targets that
actually crossed their selected threshold during a policy-passed NORMAL row.
Coverage now separates any trusted actionable recommendation from target-
triggered actionable rows; another target or an ETA cannot be attributed to all
probability horizons. Statistical diagnostics never create ML trigger targets.

Tests include loose-ceiling runtime and registry rejection, strict required
flags, boundary thresholds, contradictory required lists, malformed reason
mappings, invalid/duplicate/reversed timestamps and target-specific attribution.
Manufacturer status authority and all previous fail-closed behavior remain.
These controls prove policy mechanics, not plant accuracy; synthetic models are
non-actionable and zero trusted coverage is not evidence of zero FN.

Date: 2026-08-05 (Asia/Taipei)  
Scope: confirmed findings only; fixed five-column input, manufacturer safety authority, domain separation, lifecycle-disjoint splits, registry audit history, and synthetic-to-plant refusal are preserved.

## Finding-to-fix traceability

| Finding | Root cause | Files changed | Fix implemented | Test proving the fix | Remaining limitation |
|---|---|---|---|---|---|
| Raw warning excursion became a target | Generator wrote the first raw crossing into legacy target fields | `realistic_generator.py`, `models.py`, `offline.py`, `lifecycle_detector.py`, `lifecycle_store.py` | Added separate raw/confirmed timestamps; legacy target aliases now mean confirmed; retrained from scratch | `test_ten_minute_warning_is_raw_but_not_confirmed`, generator metadata regression, pre-event target tests | Confirmation describes configured software semantics, not a plant-validated event definition |
| Critical event confirmation was immediate | Event policy special-cased critical as confirmed on its first row | `status_policy.py`, `monitor.py` | Added independent critical timer while raw critical protection remains immediate | `test_warning_and_critical_confirm_at_configured_elapsed_time`, `test_raw_critical_protection_is_immediate_while_event_is_unconfirmed`, existing manufacturer tests | No plant-approved critical confirmation duration exists |
| Resampler was isolated | `prepare_offline_replay` returned raw validations directly | `resampling.py`, `offline.py`, `monitor.py`, `cli.py`, `retraining.py`, `models.py` | Integrated safe resampling into offline preparation, training, guardrail evaluation, and replay; normal-only interpolation, reset/outage barriers, provenance/cadence flags | long-gap, reset-boundary, offline-interpolation, audit tests | Live online input is deliberately not interpolated |
| Profiler suitability contradicted findings | Replay boolean ignored duplicates and unknown statuses | `data_profile.py`, `validation.py` | Central machine-readable refusal list covers schema, timestamps, order, status, numeric/missing sensors, and intervals | duplicate and unknown-status refusal tests | Plausibility limits beyond configured sensor bounds remain project-specific |
| Lifecycle timing config was not fully enforced | Critical confirmation unused; minimum critical reset duration defaulted to zero | `status_policy.py`, `lifecycle_detector.py`, `config/lifecycle.json` | Enforced warning/critical confirmation, gap reset, recovery confirmation, and independent 30-minute critical/reset hold | timing and minimum-critical-duration tests plus existing reset regressions | Warning-only reset remains the documented medium-confidence path |
| Normal replay CSV was reduced | Full fields were limited to SQLite/optional detailed CSV | `storage.py`, `models.py`, `monitor.py`, `cli.py` | Expanded the normal CSV and SQLite columns/payload with provenance, domain/hash/version, three statuses, lifecycle/censoring, maturity, cadence/OOD, raw predictions/targets, confidence/support, guardrail and fallback reason | CSV schema and SQLite audit tests; regenerated row-count/header audit | Per-sensor/per-feature diagnostics still require `--detailed-csv` to avoid making the normal file even larger |
| Long windows appeared mature with one row | Feature availability was a gap-only boolean | `feature_engineering.py`, `storage.py` | Added duration, coverage, maturity, count, and discontinuity per window plus 24-hour summary columns | `test_early_24_hour_window_is_partial_and_immature` | Partial-window values remain allowed inputs by design |
| Target support checked only the maximum | Runtime and guardrail comparison ignored the lower bound | `ml_forecaster.py`, `monitor.py`, `guardrail_evaluation.py`, `models.py`, `storage.py` | Checks both bounds, retains raw output, marks negative values physical/non-actionable, and falls back or withholds operationally | lower/upper support test and guardrail regression | Support is empirical synthetic training support, not a physical operating envelope |
| Split report hid representation limits | Only IDs and critical stratification label were stored | `retraining.py` | Added representation counts for duration, regime, event reach, censoring and family; reports sparse-stratification limitation | existing disjoint split test and regenerated metadata inspection | Joint stratification is impossible for sparse category combinations; all realistic lifecycles reached both events |
| Generator controls/noise were misleading | Fractions were unused, common noise was sampled per sensor, total lifecycle SD stood in for noise | `realistic_generator.py`, `data_profile.py` | Removed unused fractions; separated shared load, once-per-row shared measurement noise and per-sensor noise; added robust first-difference scale; documented inferred/manual assumptions | deterministic generator and metadata tests | Generator assumptions are not validated plant physics |

## Requirement completion status

| Requirement | Status | Evidence / note |
|---|---|---|
| 1. Raw versus confirmed events and confirmed training targets | Fully completed | Version-2 metadata, training metadata hash/version, event regressions, models retrained from scratch |
| 2. Safe resampling in real workflows | Fully completed | Shared offline entry point reaches training/replay/evaluation; provenance and cadence audited; online behavior documented |
| 3. Correct suitability decisions | Fully completed | Machine-readable refusal codes and refusal regressions |
| 4. Lifecycle timing rules | Fully completed | Warning/critical/gap/recovery/reset rules enforced; default minimum critical duration is 30 minutes |
| 5. Replay/audit schema | Fully completed | Normal CSV and SQLite verified against required fields and equal row counts |
| 6. Feature maturity | Fully completed | Per-window duration/coverage/maturity/count/gap plus early-life regression |
| 7. Both target support ends | Fully completed | Lower/upper/negative flags, raw-versus-operational separation, tests |
| 8. Split reporting | Fully completed | Representation tables retained in candidate metadata; stratification limitation explicit |
| 9. Generator cleanup | Fully completed for requested controls | Unused controls removed; common/independent components separated; robust noise proxy; assumptions documented |
| 10. Clean packages | Fully completed | Source/reproduction and evidence ZIPs are built separately; hashes/sizes recorded below |

## Regenerated artifacts and honest results

Plant profile: 10,000 rows, exact 60-second cadence, 166.65 hours, zero completed lifecycles, one censored `open_critical` lifecycle. Replay and feature extraction are suitable; training, validation, and promotion evaluation are refused. SHA-256 `d7713aec5c04473f2c9cb30508641b3d386b78f9bf0338ffa307ece36341992b`.

Realistic generator: `realistic_spindle_v2`, seed 42, 264,556 rows, 30 explicit lifecycles, SHA-256 `37baaac3187fd37fbb30194c177b02c53065e105b58214ba7e8e26938cc24f7c`. All raw warning/critical timestamps precede their confirmed counterparts. The source trajectory supplies cadence, duration scale, and robust first-difference noise estimates; degradation/reset/load/noise-mixture assumptions remain manual.

Realistic model `ml_20260805T014606Z`: 20/5/5 disjoint split. Untouched-test macro-lifecycle MAE is 4.909 hours warning and 8.876 hours critical; statistical baselines are 40.239 and 83.421 hours. Target support is 0.017–108.150 hours warning and 0.017–170.767 hours critical. These are synthetic integration metrics only.

Accelerated model `ml_20260805T020356Z`: 16/4/4 disjoint split from 24 usable lifecycles. Untouched-test macro-lifecycle MAE is 2.107 hours warning and 2.223 hours critical. This domain is deliberately accelerated and provides no plant-accuracy evidence.

Guardrail evaluation selected realistic validation candidates of 23.464004 hours warning absolute, 76.987250 hours critical absolute, and 0.997358 relative. The report records untouched-test coverage, error, unsafe-late, duration bucket, OOD, support, withholding, and per-lifecycle results. Selected values are evaluation evidence and are not silently written into the shared cross-domain configuration.

Full replay verification:

| Domain | CSV rows | SQLite rows | Completed lifecycles | Invalid rows |
|---|---:|---:|---:|---:|
| realistic_synthetic | 264,556 | 264,556 | 30 | 0 |
| accelerated_mock | 65,286 | 65,286 | 24 | 0 |

Both prohibited promotion attempts were refused: `Cannot promote realistic_synthetic model to plant_production.` and `Cannot promote accelerated_mock model to plant_production.` Refusals are in `output/*/promotion_refusal.json` and registry audit JSONL.

## Verification status

- Complete `unittest` suite: passed (final output saved in `output/test_output.txt`).
- Static compilation: passed (saved in `output/compile_output.txt`).
- Plant profile/refusal: passed.
- Fixed-seed realistic regeneration and timestamp-order audit: passed.
- Warning and critical retraining: passed for realistic and accelerated domains.
- Untouched-test and duration/guardrail evaluation: passed.
- Realistic and accelerated replay: passed; CSV/SQLite fields and counts verified.
- Synthetic promotion refusal: passed for both synthetic domains.

## Remaining limitations

The realistic generator is calibrated from one censored trajectory and manually invents complete maintenance lifecycles. Every realistic lifecycle reaches warning and critical, so event-reach representation is homogeneous and could not be stratified. The normal realistic replay CSV is about 295 MB and SQLite database about 1.25 GB; these are reproducible evidence, not source-package contents. The unsafe-late definition has no plant-approved tolerance. No result in this report establishes plant accuracy, calibration, safety validation, or deployment readiness.

Package sizes, entry counts, and SHA-256 values are recorded after the build in `output/package_manifest.json` and summarized in the final delivery response. They are intentionally not embedded in this report because doing so would change the source ZIP hash recursively.

## Final focused correction pass

This pass did not add anomaly detection, alter manufacturer thresholds, change
the five-column input contract, retrain a model, or change training inputs,
labels, lifecycle IDs, or explicit maintenance boundaries.

| Correction | Final evidence |
|---|---|
| Profiler/replay validation parity | `profile-data` now calls the replay `InputValidator`; plant has 10,000/10,000 valid rows, while each shipped out-of-range, jump, invalid-status, and duplicate fixture has one exact rejected row and replay suitability `false` |
| Causal realistic lifecycle audit | 264,556 rows: 120,783 healthy, 25,127 degrading, 92,340 warning, 24,900 critical, 1,376 recovery-confirmation, and 30 reset-complete; all reset rows match the 30 metadata maintenance timestamps |
| Model-impact review | `lifecycle_state` is absent from model features; `elapsed_lifecycle_hours` remains present and unchanged, so no retraining was required |
| Deterministic runtime | CPython 3.14.6 and all direct/transitive numerical dependencies are exactly pinned in `requirements-lock.txt`; metadata and evaluation reports record the same environment |
| Runtime compatibility | Material Python/scikit-learn/NumPy/joblib mismatches refuse model loading with an audit record; missing provenance warns; all 16 bundled targets load and execute with zero warnings in the pinned runtime |
| First-row cadence | Source cadence is blank in CSV and `NULL` in SQLite; sampling gap is `0.0`, not a fabricated one-second interval |
| Artifact parity | Realistic CSV/SQLite both contain 264,556 rows with identical state counts; accelerated CSV/SQLite both contain 65,286 rows |
| Promotion safety | Realistic-synthetic and accelerated-mock plant promotions remain explicitly refused |
| Model preservation | All 16 joblib hashes match the pre-correction binaries recorded in `output/model_environment_update.json` |
| Portable packaging | Builder rejects backslash ZIP entries and writes a SHA-256 manifest beside both archives and under `output` |

The consolidated machine-readable proof is
`output/final_verification.json`. The remaining limitation is unchanged: the
realistic dataset and candidates are synthetic integration evidence, not plant
accuracy, calibration, safety, or deployment-readiness evidence.
# Conservative anomaly/data-quality layer (2026-08-05)

Implemented an additive, opt-in causal anomaly observer with stable structured
enums and consistent CSV/JSON/SQLite serialization. Implemented invalid/malformed
data mapping, duplicate/out-of-order reasons, short/long gaps, isolated spike
observation and later resolution, persistent single-channel changes, correlated
multi-sensor deterioration, context-backed possible/suspected stuck sensors,
external-only stuck confirmation, clipping, robust excessive noise,
single-sensor disagreement, and conservative drift reporting.

Raw manufacturer warning/critical evaluation remains independent. Critical raw
values always retain immediate critical action and zero threshold time, including
possible spike, disagreement, clipping, invalid/out-of-range input, and confirmed
sensor-diagnostic cases. Anomaly policy can exclude a current trend update, hold,
reduce confidence, suspend, or refuse operational RUL while leaving raw ML output
available in audit. No input is imputed and no fallback model was trained.

Constancy safeguards require exact full-precision repetition plus meaningful
related context response; low variance, repeated rounded values, and stable
operation alone remain normal and training-eligible. Suspicion additionally needs
multiple context changes and expiration of the configured response delay.
Confirmation requires explicit external evidence. Per-sensor resolution,
precision, response delays, physical bounds, durations, related channels, and all
detector thresholds are in `config/anomaly.json`. Their origin is a conservative
engineering assumption requiring plant calibration, not plant validation.

Training filters honor `training_eligible`; suspected/confirmed faults, invalid
rows, clipping/noise/drift, interpolated short-gap rows, post-long-gap recovery,
and abrupt event rows are not automatically mixed into gradual-degradation
training. Constant valid rows remain eligible. Existing model inputs and labels
were not intentionally changed, so no retraining was required and existing
joblib binaries were not modified. Domain separation and synthetic-to-plant
promotion refusal remain unchanged.

Files added/changed: `config/anomaly.json`, `spindle_monitor/anomaly.py`,
`spindle_monitor/anomaly_fixtures.py`, models/config/monitor/storage/CLI/validation/
retraining integration, deterministic fixture catalog, anomaly regression tests,
README, this report, detailed anomaly documentation, packaging inputs, evaluation
JSON, SQLite migration, and CSV schema. The compatibility regression now derives
a guaranteed different runtime major rather than hard-coding Python 3.13.

Deterministic evaluation covers constant temperature/current/rounded vibration,
isolated/persistent/multi-sensor changes, stuck-through-load changes, temperature
response delay/failure, clipping, noise, short/long dropout, invalid data, drift,
and genuine gradual degradation. `tests/test_anomaly_layer.py` proves safety,
false-positive protections, evidence escalation, causal resolution, no imputation,
gap recovery, retraining eligibility, and CSV/SQLite parity. Existing manufacturer,
lifecycle, profiler, resampling, evaluation, registry, domain and promotion tests
remain in the full suite.

Deferred: cyberattack/spoofing, deep classifiers, exact component/root-cause
diagnosis, automatic gain/timestamp corrections, multiple independent simultaneous
sensor failures, recalibration, long-period reconstruction, and cross-sensor value
replacement. Remaining plant work requires representative uncensored multi-machine
data, resolution verification, operating-state context, controlled response tests,
manufacturer/manual review, calibration, and site validation. No plant safety,
threshold validation, or deployment readiness is claimed.

# Focused anomaly integration correction pass (2026-08-05)

| Finding | Root cause | Files changed | Corrected behavior | Test proving the correction | Direct runtime evidence | Remaining limitation |
|---|---|---|---|---|---|---|
| Held values contaminated future features | `FeatureGenerator.update` committed before checking `compute`; smoother also mutated first | `models.py`, `feature_engineering.py`, `smoothing.py`, `monitor.py`, anomaly tests | Explicit commit/hold/discard action gates both rolling trackers and smoother state; no retroactive backfill | `test_held_spike_never_contaminates_subsequent_rolling_features` and raw-audit/persistent-policy tests | Correction verification reports next mean/max/std/slope/crossings as `1/1/0/0/0` after a held critical spike | Excluded rows create a deliberate sampling discontinuity; they are not reconstructed |
| Short gaps disappeared when interpolation was off | Detector only handled generated interpolated rows or long gaps | `anomaly.py`, models/config/storage schemas, fixtures/tests | Cadence/tolerance independently classify short gaps and record source/expected interval, two missing samples for 180 s, duration, and interpolation flags | Disabled/enabled interpolation and CSV/SQLite gap parity tests | 180-second fixture is `SHORT_DATA_GAP` with two missing samples and no fabricated row | One-channel missing CSV data remains invalid under the preserved fixed schema |
| Correlation window was unused | Persistent and new abrupt sensor lists were unioned without timestamp separation | `anomaly.py`, fixtures/tests | Per-sensor first/latest timestamps expire from correlation; maximum separation must be within configured 300 s | Within/outside window, expiry, and temperature non-correlation tests | Correction verification shows in-window separation and no correlated classification one hour later | Delayed thermal correlation is not implemented; strict main window applies |
| Standard training could not screen | Training config always loaded anomaly handling disabled | `cli.py`, `retraining.py`, training tests | Plant training without `--anomaly-screening` is refused; reproduction mode remains explicit; screening counts and configuration hash enter metrics/metadata | Plant refusal, parser, eligibility, and screening-count tests | 10,000-row preparation retained 9,935 and excluded 65 by five explicit reasons; status states no model training | Supplied trajectory has no completed lifecycle, so this audit did not create trainable examples |
| Stuck timestamps/confirmation metadata were inaccurate | Result defaulted first observation to decision time and confirmation only followed machine event quality | `models.py`, `anomaly.py`, storage schemas/tests | Constant-interval start, suspicion decision, external confirmation time/source/evidence/actor, and resolution are distinct | Stuck timestamp/external confirmation and spike resolution tests | External controlled-test fixture records actor `tech-17` and a non-null confirmation timestamp | External systems must supply trustworthy actor/source metadata |
| Interval export emitted event rows | CLI read `anomaly_events` directly | `storage.py`, `cli.py`, interval tests | Maintained logical intervals consolidate episode rows and expose duration, maxima, actions, state, final class, and evidence; raw event export is separate | Consolidation, separate spikes, active/null end, duration, JSON/CSV/SQLite parity tests | Plant audit: 65 event rows consolidated into 33 resolved intervals | Cross-process concurrent writers are outside this single-process SQLite workflow |
| State inspection was stale | Normal rows were absent from event table | `storage.py`, `cli.py`, state tests | Per-sensor `anomaly_state` updates every reading; normal recovery resolves state while history remains; restart reads persisted state | Active, recovery, restart, history, and independent-sensor tests | Final plant state has zero active sensors and zero active intervals | State represents the latest completed database write, not uncommitted external telemetry |
| Configured observation/precision/penalties were partly unused | Pending spike resolved immediately; correlation/precision fields were audit-only; multipliers were hardcoded | `config.py`, `anomaly.py`, configuration tests/docs | Observation period, exact-value weakness, decimal precision, all sensor timings, correlation window, configurable penalties, and floor are enforced and audited | Observation timing, precision-limited repetition, penalty override, stuck delay, noise/clipping tests | Changing suspected penalty to `0.42` changes runtime multiplier to `0.42` | Delayed thermal correlation is reserved and not configured/implemented |

Model retraining was not performed. Both mock and realistic candidate binary hashes
remain identical to the pre-correction hash evidence. Training targets, model
schemas, domain separation, lifecycle/confirmed-event behavior, manufacturer
critical protection, and synthetic-to-plant promotion refusal remain unchanged.

Feature-history contamination is fully resolved for held/discarded readings.
Short-gap detection is independent from interpolation. Correlation windows are
enforced. Plant training is guarded by anomaly screening. Current state and
logical interval reporting are maintained and restart-safe. Every anomaly
configuration field is enforced or used as explicit audit metadata; no field is
silently unused. Delayed thermal correlation is the only reserved behavior and is
not represented as an active configuration field.

The complete 10,000-row supplied trajectory produced no stuck suspicion or stuck
confirmation. It produced 65 causal anomaly rows, 33 resolved intervals, zero
active final state, and exact CSV/SQLite reading and anomaly-count agreement.
These counts are engineering audit evidence only; they do not establish plant
calibration, plant accuracy, safety validation, or deployment readiness.

## 2026-08-06 promotion and audit hardening

The registry now uses `validate_model_for_production_promotion` as its single plant-production gate. A complete candidate requires the current FN ceiling and the effective required-target contract (currently WARNING 12h and CRITICAL 24h), plus valid evidence and the required staged artifacts before activation. Metadata can tighten but never relax the current safety policy. Promotion stages and reloads every requested artifact before swapping the production directory; activation failure restores the previous directory.

Runtime audit now separates threshold crossings from the probability/ETA inputs and rules that actually caused urgency. A warning-6h crossing alone is recorded as a crossing but cannot claim it caused a recommendation. ETA-caused recommendations identify their ETA target. Evaluator boolean parsing is non-fatal and malformed rows are excluded and counted. False-alert normal periods and alerts use the same timestamp/lifecycle segmentation; duration values are null when unavailable rather than NaN. The pure shared policy contract supplies target names and threshold boundaries to runtime and evaluator.

These policy/evaluator tests do not establish plant-model accuracy. Synthetic or legacy diagnostic fixtures remain non-actionable, and zero trusted coverage does not imply zero false negatives.

## 2026-08-06 final-decision and active-artifact pass

Anomaly withholding is now the final predictive decision: during manufacturer NORMAL it clears recommendation urgency and every predictive trigger/rule, while preserving immediate manufacturer WARNING and CRITICAL actions. Plant activation rejects a target subset, verifies all required staged artifacts, and revalidates staged metadata before directory activation. The replay now transports threshold crossings, trigger causes/rules, ETA evidence, shared policy versions, and separate validation/test FN-ceiling mappings. Evaluator schema validation now fails closed for malformed ETA evidence or unsupported policy-contract evidence.

## 2026-08-06 attribution and legacy-production pass

Replay validation now treats probability and ETA triggers separately and recomputes the rule-derived expected trigger set. The backward-compatible combined list must be the union of typed triggers. Recommendation source is taken from the final urgency rule rather than the critical forecast source, so warning-driven ML recommendations correctly serialize as `ml`. Anomaly withholding clears the nested ML trigger trace as well as the top-level result. Separate validation/test FN mappings are cross-checked against evidence. Legacy partial plant candidates are rejected rather than being written under a production label.

## 2026-08-07 generalization v4 targeted remediation

The rejected v3 model was not deployed. The next research code path now performs locked-v3 feature/joint-shift/attribution diagnosis before retraining, restricts newly trained probability estimators to causal rate/change features, balances safe operating load independently from healthy/degrading outcome, and stratifies duration bands separately in both outcome cohorts. Validation lifecycle calibration and threshold-selection subsets are explicitly disjoint; test and external stress data remain non-selection inputs.

The compact large-replay evaluator now parses distinct audit JSON once, performs timestamp segmentation once, reuses event context across targets, and vectorizes false-alert episodes. Controlled regression tests verify model-quality metric equivalence with the legacy evaluator. Full verification in the packaged workspace is 168 passed plus 2 subtests; compilation passes. No production artifacts were modified and no promotion occurred.

The archive supplied for this continuation omits the large local v3 development/stress/acceptance data, so no v4 research model was trained and no v4 FP/FN acceptance metrics are claimed here. See `docs/model_generalization_v4.md` and `run_generalization_v4.ps1` for the local locked workflow.

## Generalization v5.2

V5.2 addresses the remaining v5.1 overlap between healthy hard negatives and early/slow degradation without generating more large synthetic data. It adds noise-normalized relative-state residuals and multi-window trend signal-to-noise derived features while keeping raw absolute level out of the probability estimator. Required WARNING-12h/CRITICAL-24h targets additionally use training-only lifecycle GroupKFold hard-example mining, upweighting high-scoring negatives and difficult early positives before final fitting. A more regularized ExtraTrees candidate and optional training-OOF sigmoid calibration are available. The full regression suite passes (178 tests, 5 expected warnings, 2 subtests), and reduced end-to-end serialization plus hard-mining integration smoke tests pass. Production models and anomaly/manufacturer safety logic remain unchanged.
