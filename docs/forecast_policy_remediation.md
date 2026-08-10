# Forecast FP/FN remediation and policy contract

Date: 2026-08-06 (Asia/Taipei)

## Outcome

Immediate manufacturer protection is unchanged and passed the 100,000-row FN
stress replay with 0 FP and 0 FN for `status`, `raw_status`, and
`raw_safety_status`. Forecast policy is now schema-versioned, trainer-produced,
per-target, and fail-closed. The realistic-synthetic candidate is **not
production-ready**: it remains a candidate in the wrong deployment domain, and
its model-only stress generalization has substantial row FN and FP despite good
event detection for two horizons.

## Root causes

### Historical 87% FP

`output/fp_test/replay.csv` reproduces the earlier result exactly: all five
status layers have 0 FP, but 87,561/100,000 NORMAL rows (87.561%) have an
actionable maintenance urgency (77,315 `MAINTENANCE_RECOMMENDED_SOON`, 10,188
`PLAN_INSPECTION`, 58 `PLAN_MAINTENANCE`). The old urgency path used candidate
probabilities/ETAs or a statistical primary without enforcing deployment,
domain, target eligibility, feature maturity, OOD, support, disagreement, or
withholding. It was a real recommendation-layer FP problem, not an immediate
manufacturer-status FP problem and not merely a denominator mistake.

The verified current FP replay has 0 actionable recommendations, 0
recommendation false-alert episodes, and 0 FP in every status layer. No
manufacturer threshold or immediate safety action was weakened.

### Null thresholds and all-target withholding

The prior `models/realistic/candidate/metadata.json` stored thresholds under
`probability_targets.<target>.validation_metrics.classification_threshold`,
while runtime accepted only `probability_thresholds.targets.<target>`. It also
required new evidence aliases and marked all eight artifacts mandatory. That
made every selected runtime threshold null and allowed one weak target to block
all others.

The trainer now emits model metadata schema `3.0`, forecast-policy contract
`3.0`, threshold schema `1.0`, numeric selected thresholds, validation/test
evidence, counts, split IDs, label timing, baseline/calibration evidence, and
per-target required/eligible/reason fields. Legacy unversioned metadata is
migrated only when field mapping is unambiguous; missing FN ceilings remain
missing and fail closed. Unsupported versions are rejected.

## Contract and runtime architecture

- Required product targets are explicitly configured in `config/ml.json`:
  `probability_warning_12h` and `probability_critical_24h`.
- All other probability and ETA targets are optional and independently loaded,
  available, eligible, withheld, and actionable.
- Required target failure blocks production policy; optional failure is
  isolated and retained in the per-target reason mapping.
- Threshold selection uses validation lifecycles only. It first enforces row and
  event FN `<= 0.10`, then minimizes FP, then event FN, distance from default,
  and threshold value deterministically. Test data is evaluated once afterward
  and can reject eligibility but cannot change the threshold.
- Runtime stores raw estimator, eligible-only monotonic-reconciled, and
  manufacturer status-constrained probability layers separately. Reconciliation
  uses cumulative maximum only across eligible horizons and never overwrites raw
  values.
- Probability recommendations check the selected threshold and the specific
  target's eligibility. ETA recommendations independently require ETA evidence.
- Manufacturer WARNING/CRITICAL is evaluated first and cannot be delayed,
  downgraded, or suppressed by forecast, anomaly, OOD, or withholding logic.
- Confirmed reset/lifecycle boundaries continue clearing smoothing, event,
  feature, anomaly, and trend state; reset rows expose explicit suppression.

## Trainer evidence (locked lifecycle split)

Model `ml_20260806T070137Z`, realistic-synthetic, 20 training / 5 validation /
5 untouched-test lifecycles:

| Target | Required | Threshold | Validation FN / FP | Test FN / FP | Eligible |
|---|---:|---:|---:|---:|---|
| WARNING 6h | no | 0.109493 | 9.633% / 2.585% | 1.105% / 2.932% | yes |
| WARNING 12h | yes | 0.246046 | 10.000% / 1.214% | 2.204% / 4.165% | yes |
| WARNING 24h | no | 0.997515 | 8.142% / 2.457% | 14.030% / 0.000% | no: test FN ceiling |
| CRITICAL 6h | no | 0.015597 | 9.859% / 0.906% | 26.786% / 0.491% | no: sparse target and test FN ceiling |
| CRITICAL 12h | no | 0.017141 | 10.000% / 0.805% | 7.207% / 0.753% | no: target nearly constant |
| CRITICAL 24h | yes | 0.076571 | 9.747% / 1.439% | 2.703% / 0.307% | yes |

All validation event FN values are 0% on five validation lifecycles. This small
synthetic event count is integration evidence, not plant validation.

## Verified stress results

Final evidence is in `output/remediation_20260806/fp_verified` and
`output/remediation_20260806/fn_verified`. Both clean replays processed exactly
100,000 valid rows with zero invalid rows. FN source labels were joined from
`health_status` with exact row-count and timestamp equality.

### Status and operations

| Layer | FN-stress FN | FN-stress FP | NORMAL-stress FP |
|---|---:|---:|---:|
| `status` | 0% | 0% | 0% |
| `raw_status` | 0% | 0% | 0% |
| `raw_safety_status` | 0% | 0% | 0% |
| `stabilized_status` | 0.1090% | 0.1216% | 0% |
| `event_status` | 1.6353% | 0% | 0% |
| actionable recommendation | predictive layer not applicable to current unsafe status; strict misses reported below | 0% | 0% |

FN replay probability availability is 98.841%; 84.901% of rows are operationally
withheld. FP replay probability availability is 100%; 100% is operationally
withheld because the candidate is synthetic, not deployed, not plant-domain,
and not production-eligible. Strict operational target FN is therefore 100%
and strict target FP is 0%; this is fail-closed behavior, not forecast accuracy.

### FN/event stress: model-only raw probabilities

| Target | Row FN | Row FP | Event FN | False-alert episodes |
|---|---:|---:|---:|---:|
| WARNING 6h | 70.847% | 31.867% | 60% | 25 |
| WARNING 12h | 54.556% | 42.875% | 0% | 20 |
| WARNING 24h | 72.507% | 50.316% | 20% | 0 |
| CRITICAL 6h | 85.278% | 3.115% | 80% | 0 |
| CRITICAL 12h | 71.736% | 10.210% | 60% | 0 |
| CRITICAL 24h | 34.142% | 13.747% | 0% | 3 |

WARNING 12h and CRITICAL 24h detect every event in this stress dataset but have
poor row-level discrimination. WARNING 6h, CRITICAL 6h, and CRITICAL 12h remain
weak. Alerts at or after event onset receive no pre-event credit.

### NORMAL-only stress: model-only FP

| Target | Raw row FP | Reconciled row FP | False-alert episodes |
|---|---:|---:|---:|
| WARNING 6h | 23.837% | 23.837% | 179 |
| WARNING 12h | 28.993% | 29.743% | 127 |
| WARNING 24h | 19.230% | 19.230% | 349 |
| CRITICAL 6h | 0% | 0% | 0 |
| CRITICAL 12h | 0% | 0% | 0 |
| CRITICAL 24h | 2.239% | 2.239% | 24 |

These numbers prove that the underlying synthetic model still generalizes
poorly to several hard-NORMAL patterns. The remediation prevents those values
from becoming operational recommendations, but does not claim the raw model FP
problem is solved. The NORMAL stress file is a single open lifecycle, so using
part of it for training and another part for acceptance would violate the
lifecycle-disjoint requirement. Hard-negative retraining is deferred until
independent NORMAL lifecycles are available.

## Exact PowerShell reproduction

```powershell
$python = ".\.venv-win\Scripts\python.exe"

& $python -m pytest -q
& $python -m compileall -q src tests main.py evaluate_monitoring.py evaluate_fp_rate.py validate_forecast_contract.py

& $python main.py train-model --input data\realistic_spindle_mock.csv --data-domain realistic_synthetic --models-root models\realistic --metrics output\remediation_20260806\realistic_model_metrics.json --database output\remediation_20260806\training.db
& $python validate_forecast_contract.py --metadata models\realistic\candidate\metadata.json --models-directory models\realistic\candidate --output output\remediation_20260806\contract_validation.json
& $python main.py evaluate-model --models-root models\realistic --output output\remediation_20260806\model_evaluation.json --database output\remediation_20260806\monitor.db
& $python main.py evaluate-model --models-root models\realistic --promote --output output\remediation_20260806\promotion_refusal.json --database output\remediation_20260806\monitor.db

& $python main.py replay --input data\spindle_fp_rate_stress_test_100000.csv --data-domain realistic_synthetic --models-root models\realistic --csv output\remediation_20260806\fp_verified\replay.csv --lifecycles-csv output\remediation_20260806\fp_verified\lifecycles.csv --invalid-csv output\remediation_20260806\fp_verified\invalid_rows.csv --database output\remediation_20260806\fp_verified\monitor.db --progress-every 50000
& $python evaluate_monitoring.py --replay output\remediation_20260806\fp_verified\replay.csv --all-normal --json-output output\remediation_20260806\fp_verified\evaluation.json --csv-output output\remediation_20260806\fp_verified\evaluation.csv

& $python main.py replay --input data\spindle_fn_rate_stress_test_100000.csv --data-domain realistic_synthetic --models-root models\realistic --csv output\remediation_20260806\fn_verified\replay.csv --lifecycles-csv output\remediation_20260806\fn_verified\lifecycles.csv --invalid-csv output\remediation_20260806\fn_verified\invalid_rows.csv --database output\remediation_20260806\fn_verified\monitor.db --progress-every 50000
& $python evaluate_monitoring.py --replay output\remediation_20260806\fn_verified\replay.csv --labels-file data\spindle_fn_rate_stress_test_100000.csv --labels-column health_status --json-output output\remediation_20260806\fn_verified\evaluation.json --csv-output output\remediation_20260806\fn_verified\evaluation.csv

& $python evaluate_fp_rate.py --replay output\fp_test\replay.csv --prediction-column status --manifest data\spindle_fp_rate_stress_test_manifest.csv
```

Use new empty output directories for a fresh clean replay. Do not run the two
wide-file evaluators concurrently on a memory-constrained workstation.

## Remaining limitations

- No representative labelled plant lifecycles are available; promotion is
  refused and no production-readiness claim is made.
- The hard-NORMAL stress data has only one lifecycle and cannot safely serve as
  both hard-negative training and held-out lifecycle evidence.
- Short-horizon stress FN and WARNING stress FP remain high.
- Strict operational event FN is 100% for this candidate because every
  predictive action is correctly withheld; manufacturer safety FN remains 0%.
- Reset rows intentionally have no restored forecast artifact state and are
  reported as 1,159 reset-suppressed/missing-prediction rows in the FN replay.
- Full causal replay and row-wise audit validation are slow and memory-heavy;
  sequential evaluation is required on this machine.
