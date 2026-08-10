# Spindle Predictive Maintenance — Model Validation and Forecast Fixes

## Purpose

Use this document as the implementation specification for fixing the current spindle predictive-maintenance project.

The existing project already provides:

- Manufacturer-rule status detection
- Lifecycle/reset detection
- Statistical forecasts
- Machine-learning time-to-warning forecast
- Machine-learning time-to-critical forecast
- Warning and critical probability forecasts
- Candidate and production model registries
- CSV and SQLite output

The current candidate model was not promoted because:

```text
Time-to-critical candidate MAE: 0.831 hours
Time-to-critical baseline MAE:  6.034 hours

Time-to-warning candidate MAE:  3.732 hours
Time-to-warning baseline MAE:   3.225 hours
```

The critical model appears better than its baseline, while the warning model is worse.

There are also validation and target-generation issues that may make the reported results optimistic or misleading.

The goal is to correct these issues without changing the fixed input CSV schema.

---

# 1. Fixed input schema

Do not add, remove, or rename any required input columns.

The runtime input remains:

```text
timestamp
vibration_mps2
temperature_c
current_ampere
health_status
```

`health_status` is only a supplied validation label. Runtime status must continue to be calculated from manufacturer thresholds.

All additional fields must remain derived output fields.

---

# 2. Intended model behavior

The system should answer:

```text
1. What is the machine condition now?
2. How long until the machine reaches WARNING?
3. How long until the machine reaches CRITICAL?
4. What is the probability of reaching WARNING or CRITICAL within configured horizons?
```

The system must not predict when the maintenance team will perform maintenance.

Reset detection is only used to:

- Detect that maintenance likely occurred
- Close the previous lifecycle
- Start a new lifecycle
- Reset stateful filters and feature history
- Generate independent lifecycle data for training and validation

Remove or do not reintroduce:

```text
time-to-reset prediction
ml_time_to_reset_hours
probability of maintenance occurring
prediction of human maintenance timing
```

---

# 3. Manufacturer rules remain authoritative

Manufacturer thresholds remain the immediate safety mechanism.

The current status order is:

```text
NORMAL < WARNING < CRITICAL
```

Required behavior:

```text
Current status NORMAL:
time-to-warning may be forecast
time-to-critical may be forecast

Current status WARNING:
time-to-warning = 0
time-to-critical may be forecast

Current status CRITICAL:
time-to-warning = 0
time-to-critical = 0
maintenance urgency = IMMEDIATE_MAINTENANCE_REQUIRED
```

Machine-learning predictions must never override a raw manufacturer critical condition.

---

# 4. Problem: warning target contains too many zero rows

The current time-to-warning validation result is:

```text
MAE:                    3.7316 hours
Median absolute error:  0.0 hours
Bias:                  -3.7316 hours
```

A zero median combined with a large negative mean error strongly suggests that many training and validation rows occur after warning has already been reached and therefore have:

```text
time_to_warning_hours = 0
```

These rows make the median look good while reducing the model’s ability to forecast warning before it occurs.

## Required fix

Train and validate the time-to-warning regressor only on rows strictly before the first warning timestamp:

```python
row.timestamp < lifecycle.first_warning_timestamp
```

Do not include rows at or after first warning in the warning regressor dataset.

At runtime, manufacturer rules already provide:

```text
WARNING or CRITICAL → time_to_warning = 0
```

Therefore, post-warning zero rows are unnecessary for ML regression.

## Required tests

Add tests proving:

1. No warning-regression sample occurs at or after first warning.
2. Every warning-regression target is greater than zero.
3. Warning and critical runtime statuses still force warning forecast to zero.
4. Pre-warning samples retain the correct remaining-time target.

---

# 5. Problem: critical target contains post-critical zero rows

The same target-construction rule should apply to the critical model.

## Required fix

Train and validate the time-to-critical regressor only on rows strictly before the first critical timestamp:

```python
row.timestamp < lifecycle.first_critical_timestamp
```

Do not include rows after critical with a target of zero.

At runtime:

```text
CRITICAL → time_to_critical = 0
```

must remain enforced by manufacturer status logic.

## Required tests

Add tests proving:

1. No critical-regression sample occurs at or after first critical.
2. Every critical-regression target is greater than zero.
3. Critical runtime status forces both forecasts to zero.
4. Warning rows may still receive a positive critical forecast.

---

# 6. Problem: training and validation lifecycle metadata overlap

The current metadata reports the same lifecycle IDs in both:

```text
training_lifecycle_ids
validation_lifecycle_ids
```

All lifecycle IDs from `lifecycle_0001` through `lifecycle_0024` appear in both lists.

This prevents the results from being treated as a clean unseen-lifecycle evaluation.

## Required fix

Use completely disjoint lifecycle groups.

For 24 completed lifecycles, use approximately:

```text
Training:   16 lifecycles
Validation: 4 lifecycles
Test:       4 lifecycles
```

No lifecycle ID may appear in more than one group.

Example:

```text
training_lifecycle_ids:
lifecycle_0001 through lifecycle_0016

validation_lifecycle_ids:
lifecycle_0017 through lifecycle_0020

test_lifecycle_ids:
lifecycle_0021 through lifecycle_0024
```

The exact IDs may differ if stratification is required.

## Critical lifecycle stratification

Only 18 of the 24 mock lifecycles reach critical.

The split must ensure that critical-ending lifecycles exist in:

- Training
- Validation
- Test

The time-to-critical model must not be evaluated using warning-only lifecycles as though a critical target were observed.

## Reproducibility

The split must:

- Use a configurable random seed
- Be deterministic for the same input
- Be stored in model metadata
- Be performed by lifecycle, never by row

## Required metadata

Add:

```text
training_lifecycle_ids
validation_lifecycle_ids
test_lifecycle_ids
split_random_seed
split_strategy
```

## Required tests

Add tests proving:

1. Training, validation, and test sets are pairwise disjoint.
2. No row from a lifecycle appears in multiple groups.
3. The same seed produces the same split.
4. Critical model groups contain appropriate critical-ending lifecycles.
5. Random row splitting is never used.

---

# 7. Separate model fitting from final evaluation

Use the following workflow:

```text
Training lifecycles:
Fit candidate models

Validation lifecycles:
Choose parameters and promotion thresholds

Test lifecycles:
Final untouched performance report
```

The test group must not be used for:

- Hyperparameter selection
- Probability threshold selection
- Early stopping decisions
- Choosing between candidate configurations
- Promotion-rule tuning

After the candidate passes final testing and is promoted, it may optionally be retrained on training plus validation data.

Do not retrain using the test set before reporting final test metrics.

---

# 8. Problem: promotion treats warning and critical as one bundle

The current evaluation blocks the complete candidate because:

```text
time-to-critical eligible = true
time-to-warning eligible = false
```

This prevents a strong critical model from being used merely because the warning model is weaker.

## Required fix

Evaluate and promote each target independently.

Supported production configurations must include:

```text
Warning forecast:
statistical baseline

Critical forecast:
ML model
```

or:

```text
Warning forecast:
ML model

Critical forecast:
statistical baseline
```

or:

```text
Both ML
```

or:

```text
Both statistical
```

## Model registry design

Production metadata should identify the source for each forecast:

```json
{
  "time_to_warning": {
    "source": "statistical",
    "model_version": null
  },
  "time_to_critical": {
    "source": "ml",
    "model_version": "ml_..."
  }
}
```

Alternatively, maintain independent production files and metadata for each target.

## Promotion logic

For each target:

```text
Eligible when:
- Minimum lifecycle count is satisfied
- Candidate beats the corresponding statistical baseline
- Candidate beats the current production model, if one exists
- Candidate passes validation and untouched test requirements
- Error and safety metrics remain within configured limits
```

One failed target must not block another eligible target.

## Required CLI result

`evaluate-model` should return something similar to:

```json
{
  "status": "partially_eligible",
  "targets": {
    "time_to_warning": {
      "eligible": false,
      "recommended_source": "statistical"
    },
    "time_to_critical": {
      "eligible": true,
      "recommended_source": "ml"
    }
  }
}
```

With `--promote`, only eligible targets should be promoted.

## Required tests

Add tests proving:

1. Critical ML can be promoted while warning remains statistical.
2. Warning ML can be promoted independently.
3. A worse target is not promoted.
4. Existing production models are not deleted when another target is promoted.
5. Runtime loads the correct source independently for each forecast.

---

# 9. Lifecycle-balanced sampling

Long lifecycles must not dominate training merely because they contain more rows.

## Required fix

Use one or both of the following:

### Option A: equal samples per lifecycle

Limit each lifecycle to a configurable maximum number of samples.

Sample across lifecycle stages rather than taking only adjacent rows.

Recommended stages:

```text
Early lifecycle
Middle lifecycle
Late lifecycle
Near warning
Near critical
```

### Option B: lifecycle sample weights

Assign each row:

```python
sample_weight = 1.0 / number_of_training_rows_in_its_lifecycle
```

The total weight contributed by each lifecycle should be approximately equal.

Use deterministic sampling with a configured random seed.

## Required metadata

Store:

```text
sampling_strategy
maximum_samples_per_lifecycle
sample_weighting_enabled
sampling_random_seed
```

---

# 10. Report both row-level and lifecycle-level metrics

Current metrics aggregate sampled rows. This can overrepresent long lifecycles.

## Required metrics

For each target, calculate:

### Micro metrics

Every row has equal weight:

```text
micro_mae_hours
micro_median_absolute_error_hours
micro_bias_hours
micro_p90_absolute_error_hours
```

### Macro lifecycle metrics

Calculate the metric independently for each lifecycle, then average across lifecycles:

```text
macro_lifecycle_mae_hours
median_lifecycle_mae_hours
worst_lifecycle_mae_hours
macro_lifecycle_bias_hours
```

### Per-lifecycle metrics

Store a dictionary or list such as:

```json
{
  "lifecycle_0021": {
    "mae_hours": 0.8,
    "bias_hours": -0.2,
    "sample_count": 400
  }
}
```

Promotion should primarily use:

```text
macro_lifecycle_mae_hours
```

and not only row-weighted MAE.

## Required tests

Add tests proving:

1. A long lifecycle does not dominate the macro metric.
2. Per-lifecycle metrics are stored.
3. Promotion uses the configured primary metric.
4. Micro and macro metrics are both reported.

---

# 11. Warning model improvement

After removing post-warning zero rows, retrain the warning model.

If ML remains worse than the statistical baseline, keep the statistical model in production.

Do not force ML merely because ML exists.

## Optional warning-model improvements

Test these only using training and validation lifecycles:

- Lower `max_leaf_nodes`
- Stronger `l2_regularization`
- Smaller `learning_rate`
- Lifecycle-balanced sampling
- Predicting log remaining time:

```python
target = log1p(time_to_warning_hours)
```

and converting back with:

```python
hours = expm1(prediction)
```

- Separate early-life and near-warning models
- Quantile regression for uncertainty

Do not tune using test lifecycles.

---

# 12. Probability model fixes

The project currently trains:

```text
probability_warning_6h
probability_warning_12h
probability_warning_24h
probability_critical_6h
probability_critical_12h
probability_critical_24h
```

These probability models must also use disjoint lifecycle validation.

## Required changes

For each probability target:

1. Train only on training lifecycles.
2. Tune/calibrate using validation lifecycles.
3. Report final metrics on untouched test lifecycles.
4. Report event balance.
5. Detect nearly constant targets.
6. Avoid promoting a classifier merely because the event is almost always true.

## Required metrics

Add:

```text
brier_score
log_loss
roc_auc, when both classes exist
precision
recall
false_positive_rate
false_negative_rate
observed_event_rate
mean_predicted_probability
calibration_error
```

When only one class exists in a split:

- Do not calculate invalid metrics.
- Record a clear reason.
- Do not claim classifier quality.

## 24-hour target imbalance

The current warning-within-24-hours event rate is approximately:

```text
99.85%
```

This target is nearly constant and may not be useful.

Add configurable minimum and maximum event-rate requirements, for example:

```text
minimum_positive_rate = 0.05
maximum_positive_rate = 0.95
```

When outside this range:

```text
eligible = false
reason = target_is_nearly_constant
```

Do not promote a probability model that only predicts the majority class.

---

# 13. Forecast consistency

The final forecasts must satisfy:

```text
time_to_warning <= time_to_critical
```

when both values exist and the machine is still normal.

If forecasts violate this:

- Do not silently swap them.
- Do not average them.
- Lower confidence.
- Leave inconsistent final forecasts empty.
- Include a clear reason.

Example:

```text
forecast_confidence = LOW
forecast_reason = Warning and critical forecasts are logically inconsistent.
```

## Current-status behavior

```text
NORMAL:
warning forecast may be positive
critical forecast may be positive

WARNING:
warning forecast = 0
critical forecast may be positive

CRITICAL:
warning forecast = 0
critical forecast = 0
```

---

# 14. Low-confidence behavior

Do not always fill final forecast columns.

When:

- ML and statistical forecasts strongly disagree
- Warning and critical forecasts are inconsistent
- Current features are outside the training distribution
- Lifecycle is in `RESET_CANDIDATE`
- Insufficient causal history exists
- Probability target is poorly calibrated

then:

```text
final_time_to_warning_hours = empty, when affected
final_time_to_critical_hours = empty, when affected
forecast_confidence = LOW or UNAVAILABLE
```

Keep the separate statistical and ML diagnostic values visible.

---

# 15. Reset-candidate suppression

While lifecycle state is:

```text
RESET_CANDIDATE
```

derived historical forecasts may be contaminated by pre-maintenance data.

Required behavior:

```text
ml_time_to_warning_hours = empty
ml_time_to_critical_hours = empty
final_time_to_warning_hours = empty
final_time_to_critical_hours = empty
forecast_confidence = UNAVAILABLE
forecast_reason = Reset confirmation is in progress.
```

Manufacturer status must still be calculated from raw sensor values.

After reset confirmation, clear all stateful components:

```text
Rolling-median history
Fast EWMA
Slow EWMA
Per-sensor Kalman state
Combined degradation Kalman state
Feature history
Consecutive warning duration
Consecutive critical duration
Statistical trend history
Stabilized status history
```

---

# 16. Lifecycle boundary consistency

For offline replay and training, implement a two-pass process.

## Pass 1

```text
Read and validate rows
Detect confirmed lifecycle boundaries
Store reset candidate and confirmation information
```

## Pass 2

```text
Replay rows using confirmed lifecycle boundaries
Assign correct lifecycle IDs
Reset all state at each confirmed boundary
Generate final features and forecasts
Write CSV and SQLite
```

The lifecycle boundary used by:

- Reading-level CSV
- Lifecycle CSV
- SQLite readings
- SQLite lifecycle records
- Training data

must be identical.

Do not allow confirmation-period rows to remain assigned to the previous lifecycle when the official boundary is earlier.

---

# 17. Worst-sensor correction

The selected `worst_sensor` must actually belong to the current highest status level.

Required algorithm:

```python
overall_status = maximum sensor status

status_candidates = [
    sensor
    for sensor in sensors
    if sensor.status == overall_status
]

worst_sensor = max(
    status_candidates,
    key=lambda sensor: sensor.normalized_severity,
)
```

Do not select a `NORMAL` sensor as `worst_sensor` when another sensor is causing `WARNING` or `CRITICAL`.

Add threshold-boundary and multi-sensor tests.

---

# 18. Production registry auditing

When a target is promoted, update SQLite and model metadata.

Store:

```text
model_version
target_name
stage
promotion_timestamp
training_lifecycle_ids
validation_lifecycle_ids
test_lifecycle_ids
validation_metrics
test_metrics
baseline_metrics
feature_schema
synthetic_data_only
```

For models trained solely on generated mock data:

```text
synthetic_data_only = true
```

Such a model may be used for software testing but must not be described as validated for real industrial equipment.

The database must distinguish:

```text
candidate
production
archived
```

Promotion must create an auditable production record.

---

# 19. Expected output columns

Do not change the input columns.

The concise output should include:

```text
timestamp
vibration_mps2
temperature_c
current_ampere
status
status_reason
worst_sensor
lifecycle_id
lifecycle_state
elapsed_lifecycle_hours

statistical_time_to_warning_hours
statistical_warning_sensor
ml_time_to_warning_hours

statistical_time_to_critical_hours
statistical_critical_sensor
ml_time_to_critical_hours

probability_warning_6h
probability_warning_12h
probability_warning_24h

probability_critical_6h
probability_critical_12h
probability_critical_24h

final_time_to_warning_hours
final_time_to_critical_hours
forecast_confidence
maintenance_urgency
forecast_reason
```

Do not include:

```text
ml_time_to_reset_hours
predicted maintenance time
invented health percentage
```

---

# 20. Maintenance urgency

Keep maintenance urgency configurable.

Suggested initial rules:

```text
Current status CRITICAL
→ IMMEDIATE_MAINTENANCE_REQUIRED

High probability of critical within 6 hours
→ MAINTENANCE_RECOMMENDED_SOON

High probability of critical within 24 hours
→ PLAN_MAINTENANCE

Warning expected soon
→ PLAN_INSPECTION

Low-confidence or unavailable forecast
→ MONITOR_CLOSELY

Healthy and stable
→ NORMAL_MONITORING
```

Manufacturer status always overrides forecast-based recommendations.

---

# 21. CLI behavior

Preserve the existing commands:

```bash
python main.py replay
python main.py detect-lifecycles
python main.py train
python main.py evaluate-model
python main.py validate
python main.py simulate
```

## `train`

Must:

1. Detect and assign completed lifecycles.
2. Create disjoint training, validation, and test lifecycle groups.
3. Generate pre-event targets only.
4. Train candidate warning and critical models independently.
5. Train probability models.
6. Store honest split metadata.
7. Store validation and final test metrics.
8. Never promote automatically.

## `evaluate-model`

Must:

1. Evaluate each forecast target independently.
2. Compare candidate with the matching baseline.
3. Compare candidate with current production, when available.
4. Use macro lifecycle metrics as the main regression criterion.
5. Check probability calibration and target balance.
6. Return per-target eligibility.

## `evaluate-model --promote`

Must:

1. Promote only eligible targets.
2. Leave ineligible targets on their existing source.
3. Preserve previous production models in the archive.
4. Add production records to SQLite.
5. Report partial promotion clearly.

Example:

```json
{
  "status": "partially_promoted",
  "targets": {
    "time_to_warning": {
      "promoted": false,
      "active_source": "statistical"
    },
    "time_to_critical": {
      "promoted": true,
      "active_source": "ml"
    }
  }
}
```

---

# 22. Required tests

Add or update tests for:

- Fixed input schema compatibility
- Warning targets only before first warning
- Critical targets only before first critical
- No zero-target post-event regression rows
- Disjoint lifecycle train/validation/test splits
- Deterministic lifecycle split
- No future leakage
- Equal lifecycle weighting or deterministic sampling
- Micro and macro lifecycle metrics
- Independent warning and critical promotion
- Partial promotion
- Warning baseline retained when warning ML is worse
- Critical ML promoted when independently eligible
- Probability target imbalance detection
- Probability metrics on unseen lifecycles
- Constant-target classifier handling
- Reset-candidate forecast suppression
- Full state reset after confirmed maintenance
- Consistent lifecycle boundaries across CSV and SQLite
- Correct worst-sensor selection
- Warning forecast not exceeding critical forecast
- Low-confidence final forecast left empty
- Critical manufacturer override
- Production registry SQLite audit record
- Synthetic-only metadata
- Full replay producing one output row per valid input row

---

# 23. Acceptance criteria

The work is complete when:

1. The original fixed input CSV still works.
2. No input columns are added or required.
3. Warning regression uses only pre-warning rows.
4. Critical regression uses only pre-critical rows.
5. Train, validation, and test lifecycle IDs are disjoint.
6. Final test metrics use untouched lifecycle data.
7. Warning and critical models are promoted independently.
8. Warning may remain statistical while critical uses ML.
9. Macro lifecycle metrics are calculated and used.
10. Probability models are validated on unseen lifecycles.
11. Near-constant probability targets are rejected.
12. Reset-candidate forecasts are suppressed.
13. Stateful filters reset after confirmed maintenance.
14. Reading and lifecycle boundaries agree.
15. `worst_sensor` always matches the highest current status.
16. Final warning and critical forecasts are logically consistent.
17. Low-confidence forecasts do not produce misleading final values.
18. Production model promotion is recorded in SQLite.
19. All automated tests pass.
20. The complete mock dataset replays successfully.
21. The candidate is retrained and reevaluated honestly.
22. The README explains the corrected workflow and limitations.
23. The modified project is returned as a downloadable ZIP.

---

# 24. Required final verification

Run:

```bash
python -m unittest discover -s tests -v
```

Then run:

```bash
python main.py replay \
  --input data/spindle_predictive_maintenance_mock.csv \
  --csv output/mock_replay_before_ml.csv \
  --lifecycles-csv output/mock_lifecycles.csv \
  --invalid-csv output/mock_invalid_rows.csv \
  --database output/mock_monitor.db \
  --progress-every 5000
```

Train:

```bash
python main.py train \
  --input data/spindle_predictive_maintenance_mock.csv \
  --database output/mock_monitor.db \
  --metrics output/mock_model_metrics.json
```

Evaluate:

```bash
python main.py evaluate-model \
  --database output/mock_monitor.db \
  --output output/mock_model_evaluation.json
```

Report:

- Training lifecycle IDs
- Validation lifecycle IDs
- Test lifecycle IDs
- Confirmation that they do not overlap
- Warning candidate micro and macro metrics
- Critical candidate micro and macro metrics
- Statistical baseline metrics
- Probability-model metrics
- Per-target eligibility
- Whether partial promotion is available
- Any remaining limitations

Do not claim real-world accuracy from synthetic data.

---

# 25. Implementation instructions for Codex

Before editing:

1. Inspect the complete existing project.
2. Identify current target generation, splitting, model registry, promotion, lifecycle handling, and runtime forecast-selection code.
3. Preserve all working manufacturer safety behavior.
4. Preserve the fixed input format.
5. Modify the existing architecture rather than replacing the project blindly.
6. Add tests before or alongside each correction.
7. Keep training and runtime feature generation identical.
8. Run the complete test suite.
9. Replay the full mock dataset.
10. Return a concise change summary and a downloadable ZIP.

The most important fixes are:

```text
1. Pre-warning and pre-critical targets only
2. Disjoint lifecycle train/validation/test groups
3. Independent per-target promotion
4. Lifecycle-weighted evaluation
5. Honest probability validation
```

Do not force model promotion merely to obtain ML output. A statistical warning forecast combined with an ML critical forecast is an acceptable and expected production configuration when supported by validation.
