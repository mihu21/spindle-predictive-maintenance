Spindle prognostics V6.0.2 - anomaly withholding remediation

Changes
- Detector version bumped to 1.1.0.
- Persistent physically valid single-sensor changes no longer suspend prediction.
- Correlated physically valid multi-sensor possible-machine events no longer suspend prediction.
- Both continue with reduced confidence (machine_event multiplier 0.55).
- Initial isolated abrupt samples still use HOLD_FOR_CONFIRMATION.
- Invalid data, confirmed sensor faults, and long data gaps remain fail-closed.
- Reduced-confidence NORMAL-state forecasts are informational only and cannot produce actionable maintenance recommendations.
- Persistent/machine-event rows remain excluded from automatic retraining.
- Added thermal-led variable-load regression coverage.
- Included the V6.0.1 corrected prognostics evaluator.

Verification
- Full test suite: 186 passed, 5 expected RuntimeWarnings, 2 subtests passed.
- Previous 185-test baseline passed before modification.
- Actual replay anomaly-only check on thermal-led variable-load lifecycles:
  lifecycle_0007 prediction-disabled: ~95.3% -> 0.0206%
  lifecycle_0015 prediction-disabled: ~98.2% -> 0.00895%
  lifecycle_0023 prediction-disabled: ~99.0% -> 0.00566%

Important
This removes the systematic withholding failure but does not prove new prognostic
recall/precision. Re-run the full replay/evaluator to measure WARNING-12h and
CRITICAL-24h event performance after the policy change. Synthetic evidence is
not plant production validation.
