# Realistic-data readiness

## Status and safety semantics

The preserved input contract is exactly `timestamp`, `vibration_mps2`, `temperature_c`, `current_ampere`, and `health_status`.

- `raw_status` is computed immediately from manufacturer thresholds. A raw critical sample immediately forces the operational critical protection path; no confirmation delay can weaken this rule.
- `stabilized_status` is the causal vote/recovery debounce used by the operational monitor.
- `event_status` is the independent time-confirmed lifecycle state. Warning and critical confirmation use `warning_confirmation_minutes` and `critical_confirmation_minutes`, and a gap longer than `maximum_confirmation_gap_minutes` resets the pending confirmation.
- Lifecycle records contain `first_raw_warning_timestamp`, `first_confirmed_warning_timestamp`, `first_raw_critical_timestamp`, and `first_confirmed_critical_timestamp`. Backward-compatible `first_warning_timestamp` and `first_critical_timestamp` mean the confirmed timestamps.
- Reset confirmation remains independent. Once critical is confirmed, `minimum_critical_duration_before_reset_minutes` must elapse before a reset candidate can begin. The shipped value is 30 minutes, not zero.

Training warning/critical remaining-life targets use confirmed timestamps only. The generator's approximately ten-minute warning excursion remains visible as a raw event but cannot become a confirmed warning or target under the 30-minute rule.

## Resampling and gaps

`config/data.json` controls target cadence, interpolation opt-in, maximum interpolation gap, and `large_gap_policy`. `prepare_offline_replay()` applies the shared safe resampler, so the same behavior reaches offline preparation, training, guardrail evaluation, and replay.

Interpolation is permitted only when explicitly enabled, inside a short normal-to-normal span. It cannot cross an explicit/inferred lifecycle boundary, a long outage, or a warning/critical span. Every row records original/interpolated provenance, source sampling interval, effective resampling interval, and feature availability. `mark_unavailable` leaves the first post-outage row in the audit but marks affected features unavailable; `reject` refuses the stream. Rolling gap flags remain until the discontinuity leaves the relevant window.

The online monitor does not resample or interpolate. It processes samples causally as received; this avoids fabricating an online safety/event observation and matches the existing architecture.

## Feature maturity and support

Each 5-minute through 24-hour rolling window reports:

- `available_history_duration_seconds`
- `coverage_fraction`
- `window_mature`
- `sample_count`
- `gap_or_discontinuity`
- the existing `feature_available` and sampling-gap summaries

Partial windows may still be model inputs, but they are explicit. A single first-lifecycle sample has a small coverage fraction and `window_mature = 0`, including for the 24-hour window.

Regression metadata records lower and upper target support. Raw predictions are never clipped. Below-minimum and above-maximum predictions are marked as target-support violations; negative predictions are additionally physically invalid/non-actionable and receive a fallback or withholding outcome while the raw estimate remains in audit output.

## Profiler refusals

Replay and feature suitability now refuse missing/extra/reordered contract columns, invalid timestamps, duplicate/non-monotonic timestamps, blank/unknown statuses, non-numeric or missing required sensor values, invalid sampling intervals, configured sensor-bound violations, and impossible sensor jumps. The profiler invokes the authoritative replay `InputValidator`; `input_validation` contains exact valid/invalid counts, reason counts, and representative row/timestamp examples. `suitability_refusal_reasons` is machine-readable and grouped by use. Training, validation, and promotion add lifecycle-count refusal codes.

The provided plant trajectory has 10,000 valid rows at exactly 60 seconds over 166.65 hours. It has zero completed lifecycles and one censored `open_critical` lifecycle. It remains suitable for replay and feature extraction, but training, validation, and promotion evaluation are refused. Input SHA-256: `d7713aec5c04473f2c9cb30508641b3d386b78f9bf0338ffa307ece36341992b`.

## Regenerated realistic evidence (seed 42)

- Generator: `realistic_spindle_v2`
- Rows/lifecycles: 264,556 / 30
- Dataset SHA-256: `37baaac3187fd37fbb30194c177b02c53065e105b58214ba7e8e26938cc24f7c`
- Model version: `ml_20260805T014606Z`
- Split: 20 training, 5 validation, 5 untouched test; all IDs are disjoint
- Warning target support: 0.017 to 108.150 hours
- Critical target support: 0.017 to 170.767 hours
- Untouched-test macro-lifecycle MAE: 4.909 hours warning, 8.876 hours critical
- Statistical-baseline untouched-test macro-lifecycle MAE: 40.239 hours warning, 83.421 hours critical
- Full replay: 264,556 CSV rows and 264,556 SQLite readings; 30 lifecycle records; zero invalid rows

Every generated lifecycle reached confirmed warning and critical. Consequently, critical-reach stratification was impossible and is reported as such. Split representation still reports duration, regime, warning/critical reach, censoring, and degradation family. The validation-selected guardrail report chose 23.464004 hours warning absolute, 76.987250 hours critical absolute, and 0.997358 relative. These values are evaluation output; they are not silently copied into the shared cross-domain config.

## Generator scope and limitations

The real censored trajectory supplies cadence, total-duration scale, and a robust first-difference measurement-noise estimate. The generator manually assumes maintenance boundaries, power-curve degradation families, temporary excursions, operating regimes, shared process/load variation, and the shared/independent measurement-noise mixture. Shared measurement noise is sampled once per timestamp; independent noise is sampled per sensor. Total lifecycle standard deviation is not used as measurement-noise sigma.

These assumptions are software-integration fixtures, not validated plant physics. Synthetic metrics cannot establish plant accuracy, calibration, safety, alert tolerances, or deployment readiness. Plant use still requires completed independent plant lifecycles, confirmed maintenance records, approved tolerances, prospective validation, drift monitoring, and human-controlled promotion.

## Audit and domain separation

The normal replay CSV contains the complete audit schema; `--detailed-csv` only adds per-sensor/per-feature detail. SQLite uses additive migrations and retains registry/training/evaluation history when replay readings are cleared. Synthetic domains remain separated under `models/mock`, `models/realistic`, and `models/plant`. Attempts to promote `realistic_synthetic` or `accelerated_mock` candidates to plant production are refused and recorded in CLI JSON and registry audit JSONL.

Explicit realistic maintenance timestamps remain authoritative lifecycle
boundaries. At those fixed boundaries, replay now derives audit-only state from
the same smoothing, stabilized-status, elapsed event confirmation, recovery,
and lifecycle-detector semantics used by the monitor. The final realistic
evidence contains 120,783 `HEALTHY`, 25,127 `DEGRADING`, 92,340 `WARNING`,
24,900 `CRITICAL`, 1,376 `RECOVERY_CONFIRMATION`, and 30 `RESET_COMPLETE`
rows. All 30 reset rows match metadata maintenance timestamps. CSV and SQLite
state counts are identical. This correction does not affect inputs, labels,
elapsed lifecycle time, boundary discovery, or model behavior;
`lifecycle_state` is absent from `feature_names`.

The first replay row has no prior observation from which a source cadence can
be measured. It therefore records blank/`NULL` source cadence and a zero
sampling gap. Later rows retain measured source cadence and the configured
effective resampling interval.

## Deterministic environment and packages

The tested reproduction environment is CPython 3.14.6 with exact pins in
`requirements-lock.txt`, including NumPy 2.5.1, pandas 3.0.3,
scikit-learn 1.9.0, and joblib 1.5.3. Training metadata, model metrics,
candidate evaluation, promotion refusal, and guardrail evaluation record the
training/runtime environments. Runtime loading refuses material compatibility
mismatches and warns when environment provenance is absent. All eight mock and
eight realistic candidate targets load and execute without warnings in the
pinned environment.

No model was retrained for this final correction. The joblib hashes in
`output/model_environment_update.json` prove the binaries were unchanged while
environment metadata was added. `output/final_verification.json` records the
final replay, model-loading, feature-impact, profiler, and promotion checks.
`build_packages.ps1` uses an explicit source allowlist, excludes virtual
environments/caches/temporary files, validates forward-slash ZIP entry names,
and places `packages/package_manifest.json` beside the source and evidence
archives.
