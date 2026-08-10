# Conservative anomaly and sensor-fault handling

## Scope and safety boundary

The anomaly detector is an independent prediction-usability observer. It does not
own manufacturer status, stabilized status, confirmed event status, or lifecycle
state. Raw manufacturer thresholds are evaluated first and remain authoritative.
A raw critical value always produces `IMMEDIATE_MANUFACTURER_CRITICAL_ACTION`,
zero immediate threshold time, and `IMMEDIATE_MAINTENANCE_REQUIRED`, even when
the same row is a possible spike, disagreement, clipping event, invalid value, or
suspected sensor fault. Uncertainty is never evidence that the machine is safe.

The layer is opt-in for replay (`--anomaly-audit`). The configuration file keeps
it disabled by default so previously reproduced replay and training behavior can
be regenerated exactly. Enabling it changes only anomaly audit, feature/model
usability policy, operational forecast qualification, and future-training
eligibility. Existing model binaries, labels, threshold rules, lifecycle
boundaries, registry behavior, and the five-column input contract are unchanged.

## Structured decision

Every accepted row has one `AnomalyResult`, serialized consistently to normal and
detailed CSV, `readings.details_json`, indexed SQLite reading columns, and the
SQLite `anomaly_events` table. Invalid rows use the same schema in the invalid CSV
and `anomaly_events`. The result records quality, detailed anomaly types, severity,
confidence, uncertain origin, affected channels, causal timestamps, active state,
raw safety and model actions, feature/prediction/retraining eligibility,
confidence multiplier, evidence, configuration snapshot, detector version,
original causal decision, and later offline classification.

High-level quality values are stable: `NORMAL`, `VALID_WITH_OBSERVATION`,
`SUSPECTED_SENSOR_ANOMALY`, `INVALID_SENSOR_DATA`, `POSSIBLE_MACHINE_EVENT`,
`CONFIRMED_MACHINE_CRITICAL`, and `DATA_UNAVAILABLE`. Detailed reasons remain
separate. Several reasons may coexist; for example a first abrupt vibration value
can be `SINGLE_SAMPLE_SPIKE|SENSOR_DISAGREEMENT|ORIGIN_UNCERTAIN` without being
declared a sensor failure.

## Causal behavior

Online decisions use only current and prior values. The first isolated abrupt
reading is held with `HOLD_OUTSIDE_FEATURE_HISTORY`, excluded from both rolling
trackers and smoothing state, and excluded from future
automatic retraining. A later reading can resolve it as a transient or establish a
persistent change. The initial decision is immutable in its row; the resolution
row records `RESOLVED_TRANSIENT_SPIKE` as the offline classification. Persistent
single-channel changes retain an uncertain machine/sensor/regime origin.
Coherent multi-channel changes become `POSSIBLE_MACHINE_EVENT`. Because these
accepted values are still physically valid and may represent genuine machine
degradation or a load/regime shift, both persistent single-channel changes and
coherent multi-channel machine events now **continue forecasting with reduced
confidence** instead of automatically suspending prediction. They remain excluded
from automatic retraining pending review. Only strong sensor/data-integrity
failures (for example invalid input, confirmed sensor failure, or a long data gap)
remain fail-closed for prediction.

Feature history has an explicit audit action. `COMMIT_TO_FEATURE_HISTORY` updates
the rolling trackers and prediction smoother. `HOLD_OUTSIDE_FEATURE_HISTORY`
keeps a pending value outside history. `DISCARD_FROM_FEATURE_HISTORY` permanently
excludes values that are not safe to use as features. Initial abrupt readings are
still held causally during the confirmation window; once a physically valid
persistent machine/regime change is established, current and future readings are
committed so real degradation is not hidden from prognostics. The online system
does not retroactively backfill the originally held rows or rewrite historical
predictions. The raw reading remains in CSV and SQLite.

Short normal-to-normal gaps continue to use the existing safe resampler. Generated
rows record interpolation and original/effective intervals and receive reduced
confidence. Interpolation never crosses lifecycle boundaries, long outages, or
warning/critical spans. Long outages do not create recovery or maintenance; they
mark data unavailable, preserve the lifecycle, suspend RUL, and require the
configured number of valid recovery samples before confidence begins to return.

Gap detection is independent from interpolation. Expected cadence comes from
`target_sampling_interval_seconds`; tolerance comes from
`sampling_interval_tolerance_seconds`. A source interval with missing samples that
is below `long_gap_seconds` is `SHORT_DATA_GAP` even when interpolation is off.
Audit fields include expected/source interval, estimated missing samples, gap
duration, interpolation configuration, and whether interpolation occurred.

Abrupt timestamps are retained per sensor. Correlation requires the maximum
separation between included first-abrupt timestamps to be no greater than
`multi_sensor_correlation_window_seconds`. Expired evidence remains auditable as a
persistent/uncertain single-sensor episode but cannot be reused in a later
correlated event. Delayed thermal correlation is not implemented; temperature is
subject to the same strict main window, while its stuck-response delay remains
sensor-specific.

Invalid/malformed input continues through authoritative `InputValidator` rules.
Raw values and exact reasons are retained where parsing permits. Invalid values do
not enter features or training. If a complete parsed but out-of-range value is
also manufacturer-critical, its immediate critical safety action is preserved.


## Forecast-availability remediation (detector 1.1.0)

A large realistic-synthetic replay exposed a systematic failure: thermal-led
variable-load degradation was being classified as an origin-uncertain persistent
change and prediction was suspended for roughly 95-99% of the affected
lifecycles. Detector 1.1.0 changes the policy boundary rather than weakening
manufacturer safety:

- invalid input, confirmed sensor failure, and long data gaps still suspend/refuse prediction;
- first isolated abrupt readings remain held for causal confirmation;
- persistent physically valid changes continue with `USE_WITH_REDUCED_CONFIDENCE`;
- coherent multi-sensor possible machine events continue with `USE_WITH_REDUCED_CONFIDENCE`;
- these uncertain intervals remain ineligible for automatic retraining; and
- while raw manufacturer status is NORMAL, reduced-confidence forecasts are
  informational only: maintenance recommendations are forced non-actionable.

An anomaly-only replay over the three previously missed `thermal_leads` /
`variable_mixed_load` lifecycles reduced prediction-disabled fractions from
approximately 95.3%, 98.2%, and 99.0% to 0.021%, 0.009%, and 0.006%, respectively.
A full prognostic replay is still required to measure the resulting WARNING/CRITICAL
event recall and false-alert trade-off.

## Constant-sensor false-positive protection

Constancy, low variance, rounded repetition, and stable operating state are never
sufficient evidence of a stuck sensor. Exact raw repetition must exceed the
sensor-specific observation time and outlast prior normal constant behavior, and
at least one related sensor must exhibit a meaningful context change before a
weak `POSSIBLE_STUCK_SENSOR` observation is emitted. That weak observation uses
the value normally and applies no confidence or training penalty.

`STUCK_SENSOR_SUSPECTED` additionally requires continued repetition, multiple
context events, expiration of the physical response delay, and behavior unusual
for configured resolution/history. Temperature defaults use a four-hour
observation period, eight-hour suspicion period, and one-hour response delay.
Vibration and current use one-/two-hour observation/suspicion periods, with
two-minute and five-minute response delays. These are conservative engineering
assumptions, not plant-calibrated values.

`STUCK_SENSOR_CONFIRMED` has no purely statistical transition. It is reachable
only through an explicit diagnostic, calibration, technician, communication, or
controlled-test confirmation. Confirmed channels are not fabricated. With no
validated fallback model in this project, RUL is suspended while remaining raw
safety signals continue to be monitored.

## Other implemented observations

- Clipping requires repeated readings at or very near configured physical limits;
  stable values away from a boundary are not clipping and values beyond the limit
  are never reconstructed.
- Excessive noise uses robust detrended residual MAD and direction changes. It
  reduces confidence and excludes the interval from retraining without aggressive
  smoothing. A monotonic degradation trend is not noise merely because its raw
  variance rises.
- Single-sensor disagreement preserves uncertain origin; it never replaces a
  reading from correlated sensor predictions.
- Conservative drift reporting requires a long, mostly one-direction movement.
  It never subtracts a drift estimate, alters lifecycle state, or self-calibrates.
- Training eligibility is explicit. Invalid, interpolated, suspected/confirmed
  sensor-fault, clipping/noise/drift, and abrupt-machine-event rows are excluded or
  held for review. Constant valid data remains eligible.

## Configuration and calibration limits

`config/anomaly.json` contains all important windows, tolerances, duration rules,
confidence penalties, sensor resolution/precision, response delay, physical
bounds, exact-value availability, and related channels. Every current default is
labelled `Conservative engineering assumption; requires plant calibration`.
Manufacturer alert thresholds remain separately derived and configured in
`thresholds.json`.

No adequate multi-machine uncensored plant baseline was supplied. The available
trajectory is censored and represents one machine; synthetic data cannot validate
plant thresholds. Consequently typical variance, longest normal constant period,
rate of change, response behavior, and correlations may be described from a
chosen plant dataset in a future calibration study, but they are not promoted
automatically into aggressive thresholds here. Deployment safety and plant
readiness are not claimed.

All current anomaly configuration fields are enforced or audit metadata. This
includes spike observation/recovery/delta, persistence, cadence tolerance,
correlation window, exact-value availability, decimal precision, resolution,
physical boundaries, clipping tolerance, noise/drift windows and multipliers,
stuck durations, response delays, related sensors, confidence penalties, and the
minimum confidence floor. Precision-limited repetition doubles the duration
requirements, requires extra context, and explicitly records weakened evidence.
No configured field is silently ignored. Delayed thermal correlation is a
documented reserved capability, not an active configuration field.

## Training screening

`train` and `train-model` support `--anomaly-screening` and
`--reproduction-mode`. Plant-domain training is refused unless screening is
enabled. Synthetic commands without screening remain a clearly announced legacy
reproduction mode so existing artifacts remain reproducible. Screening metadata
records detector version, canonical configuration SHA-256, eligible/excluded row
counts, exclusion counts by reason, human-review counts, and interpolation policy.
`audit-training-screening` executes preparation and produces these counts without
fitting or saving any model.

## State and interval persistence

SQLite now has three distinct anomaly records. `anomaly_events` is the immutable
causal event-row history. `anomaly_state` is updated on every reading, including
normal recovery, and stores independent current state per sensor. It prevents the
inspection command from treating the last historical event as current. The
`anomaly_intervals` table consolidates only the same logical episode and retains
first decision/confirmation, resolution, duration, maximum severity/confidence,
observed safety/model actions, final classification, event count, and evidence.
Separate incidents remain separate intervals. JSON and optional CSV exports are
generated from this maintained SQLite interval table; raw event export remains
available separately.

## Commands

```powershell
python main.py replay --anomaly-audit --input data/input.csv
python main.py evaluate-anomalies --output output/anomaly_evaluation.json
python main.py summarize-anomalies --database output/monitor.db
python main.py export-anomaly-intervals --database output/monitor.db --output output/anomaly_intervals.json
python main.py export-anomaly-events --database output/monitor.db --output output/anomaly_events.json
python main.py inspect-anomaly-state --database output/monitor.db
python main.py explain-prediction --database output/monitor.db
python main.py audit-training-screening --input data/input.csv --output output/anomaly_training_screening.json
python main.py train-model --data-domain plant --models-root models/plant --anomaly-screening
python main.py train-model --data-domain accelerated_mock --reproduction-mode
```

The evaluation report treats any false positive on the three normal constant-data
fixtures as a serious failure.

## Explicitly deferred

Cyberattack/spoofing detection, deep-learning anomaly classification, exact
component diagnosis, automatic gain correction, automatic timestamp-lag
estimation, multiple independent simultaneous sensor-failure handling, automatic
root-cause diagnosis, sensor recalibration, long-fault reconstruction, and
replacement of one channel from other-channel predictions are deferred. A partial
one-sensor CSV dropout is rejected as invalid because the preserved input schema
requires all three channels; it is not silently converted into all-sensor loss.
