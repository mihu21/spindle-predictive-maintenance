# V6.0.2 anomaly-withholding remediation

## What changed

- Bumped anomaly detector audit version to `1.1.0`.
- Persistent physically valid single-sensor changes now use `USE_WITH_REDUCED_CONFIDENCE` instead of `SUSPEND_PREDICTION`.
- Correlated physically valid multi-sensor possible-machine events now use `USE_WITH_REDUCED_CONFIDENCE` instead of `SUSPEND_PREDICTION`.
- Changed the machine-event confidence multiplier from `0.0` to `0.55`.
- The initial causal abrupt-reading confirmation window still uses `HOLD_FOR_CONFIRMATION`.
- Invalid input, confirmed sensor faults, and long data gaps still fail closed and suspend/refuse prediction.
- Persistent/machine-event rows remain excluded from automatic retraining pending review.
- When manufacturer status is NORMAL, a reduced-confidence anomaly forecast remains visible but cannot create an actionable maintenance recommendation.
- Added regression tests for thermal-led variable-load behavior.
- Included the corrected V6.0.1 prognostics evaluator patch in the full snapshot.

## Verification

Baseline before changes:

- 185 tests passed
- 5 expected RuntimeWarnings
- 2 subtests passed

After changes:

- 186 tests passed
- 5 expected RuntimeWarnings
- 2 subtests passed

Actual replay anomaly-only verification on the three previously missed thermal-led variable-load lifecycles:

| Lifecycle | Before: prediction disabled | After: prediction disabled |
|---|---:|---:|
| lifecycle_0007 | 95.3052% | 0.0206% |
| lifecycle_0015 | 98.2316% | 0.00895% |
| lifecycle_0023 | 98.9643% | 0.00566% |

Only the initial 3-row causal confirmation window remains prediction-disabled in each of those lifecycles.

## Important next validation

This patch fixes the systematic withholding mechanism. It does **not** by itself prove improved WARNING-12h / CRITICAL-24h recall or precision. Re-run the normal replay and `evaluate_prognostics_v6.py` on the same dataset after applying the patch, then compare event recall, FP episodes, ETA error, and total withholding rate.
