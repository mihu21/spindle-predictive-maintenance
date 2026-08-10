# Spindle Predictive Maintenance — V6 Model-Based Prognostics

> **Current primary architecture (v6):** manufacturer safety + sensor-integrity anomaly handling + model-based probabilistic degradation forecasting. The supervised synthetic-lifecycle classifier from v4/v5.x is retained only for explicit research/audit comparison and is no longer the primary runtime forecast. See [docs/model_prognostics_v6.md](docs/model_prognostics_v6.md).

V6 does **not** require generation of another large synthetic dataset or supervised failure-model training. It estimates persistent multi-timescale degradation rate and uncertainty directly from the observed sensor history, then forecasts manufacturer threshold first-passage time to produce WARNING/CRITICAL ETA and 6/12/24h probabilities. Raw manufacturer WARNING/CRITICAL behavior remains immediate and authoritative.

Quick start on the current one-machine CSV:

```powershell
Unblock-File .\run_prognostics_v6.ps1
.\run_prognostics_v6.ps1
```

The runner enables anomaly auditing by default, skips SQLite unless `-WithDatabase` is supplied, and does not run legacy ML unless `-LegacyMLAudit` is supplied.

---

## Historical v5.x supervised-model documentation

This project combines authoritative manufacturer rules, causal statistical forecasts, and optional lifecycle-trained ML. It forecasts physical progression to `WARNING` and `CRITICAL`; it never predicts maintenance or reset timing.

The current forecast metadata/policy contract is schema 3.0 with independent
per-target eligibility. The full FP/FN remediation report, verified stress
metrics, and exact PowerShell commands are in
[docs/forecast_policy_remediation.md](docs/forecast_policy_remediation.md).

## Fixed input and safety contract

The required input CSV remains exactly `timestamp`, `vibration_mps2`, `temperature_c`, `current_ampere`, and `health_status`. The label is validation-only. Runtime status comes from the raw sensors and manufacturer thresholds. Raw critical forces both primary and conservative times to zero and `IMMEDIATE_MAINTENANCE_REQUIRED`; ML cannot weaken it. Reset-candidate suppression and confirmed-boundary state clearing remain unchanged.

An opt-in conservative anomaly/data-quality layer now keeps safety status separate from model usability. Use `replay --anomaly-audit` to audit malformed data, gaps, causal spike/persistence, context-backed stuck suspicion, clipping, robust excessive noise, disagreement, correlated abrupt deterioration, and conservative drift. Raw critical protection remains immediate under every anomaly action. Naturally constant sensors remain valid unless exact repetition persists through meaningful related operating changes and the sensor-specific response delay. See [docs/anomaly_handling.md](docs/anomaly_handling.md). Defaults are engineering assumptions requiring plant calibration; this is not a plant-safety or deployment-readiness claim.

Detector 1.1.0 distinguishes **prediction-invalid sensor/data failures** from **physically valid machine/regime changes**. Persistent or correlated machine-change observations now keep prognostic outputs available at reduced confidence, are excluded from retraining pending review, and cannot generate an actionable ML maintenance recommendation while raw manufacturer status remains NORMAL.

Held anomaly rows now remain outside rolling and smoothing feature history, with an explicit `feature_history_action`; raw values remain permanently audited. Short gaps are detected from expected cadence even when interpolation is disabled. Multi-sensor abrupt correlation enforces its configured time window. SQLite maintains independent current state and consolidated logical intervals in addition to immutable event rows, so normal recovery clears current state without deleting history.

## Lifecycle and model validation

Offline replay and training use the same two-pass confirmed boundaries. `raw_status` is the immediate manufacturer result, `stabilized_status` is the operational debounce result, and `event_status` is the independent elapsed-time-confirmed lifecycle result. Lifecycle metadata stores `first_raw_*` and `first_confirmed_*` timestamps separately. Warning and critical regression targets use only confirmed timestamps; a temporary raw crossing is never a target. Raw critical protection remains immediate.

Training examples are strictly pre-event with positive targets. Lifecycle groups are deterministic and disjoint; critical stratification is attempted only when both classes have enough lifecycles. Split reports always show duration buckets, operating regime, warning/critical reach, censoring, and degradation family and explicitly state when joint stratification is impossible. Models fit training lifecycles, selection uses validation lifecycles, and test lifecycles remain untouched until one final report.

## Runtime forecast concepts

Four concepts are intentionally separate:

- `primary_forecast` is the best estimated remaining time. Valid, in-distribution, promoted ML is primary.
- `conservative_alert_forecast` is an optional earlier safety value. When strong disagreement has ML later than statistics, ML remains primary and the statistical value is exposed here. When ML is earlier, ML is already conservative. Values are never silently averaged.
- `confidence` describes trust in the primary estimate. Strong disagreement downgrades confidence but does not erase valid ML.
- `disagreement` is the absolute and relative difference between methods. Disagreement with a weaker baseline is evidence to audit, not automatically an ML failure.

Model uncertainty is distinct from disagreement. Missing ML, incompatible schemas, and out-of-distribution features concern whether ML is usable. OOD is defined by the stored training feature envelope; an acceptable statistical forecast is used as fallback, otherwise the target is withheld with `distribution_check`. Reset suppression, missing ML, distribution failure, and logical consistency have explicit withholding causes. Disagreement alone does not withhold.

For each target:

```text
absolute = abs(ML - statistical)
relative = absolute / max(abs(ML), abs(statistical), 1e-6)
strong = absolute > target_absolute_threshold
         AND relative > disagreement_relative_threshold
```

The current realistic-synthetic validation-selected configuration is:

```text
warning_disagreement_absolute_hours = 23.464004
critical_disagreement_absolute_hours = 76.987250
disagreement_relative_threshold = 0.997358
```

Candidates came from the 50th, 75th, and 90th percentiles of training-lifecycle disagreement only. Validation selected among those candidates by retaining at least 90% of the maximum validation recall for unsafe-late cases, then minimizing low-confidence rows and withholding. The selected configuration was evaluated once on the untouched test lifecycles. The JSON report includes coverage, MAE, bias, p90 error, low-confidence and withheld percentages, unsafe-late rates, and per-lifecycle results.

Warning primary time may never exceed critical primary time when both exist. A logical inconsistency remains a separate failure: both primary values are withheld without swapping or averaging. Offline replay batches estimator calls for performance only; feature generation, lifecycle state, manufacturer rules, and row-level policy remain causal and ordered.

## Commands

Create the deterministic tested environment with CPython 3.14.6. The broad
`requirements.txt` remains useful for development, while the lock file is the
authoritative reproduction environment:

```powershell
py -3.14 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install --requirement requirements-lock.txt
.\.venv-win\Scripts\python.exe -m pip check
```

The 2026-08-07 long-lifecycle model-generalization research candidate is
documented in `docs/model_generalization_remediation.md`. It failed required
target and external long-normal acceptance evidence, remains isolated under
`models/generalization_v3/realistic`, and did not replace the verified baseline.

Candidate metadata and evaluation reports record Python, platform, NumPy,
pandas, SciPy, scikit-learn, joblib, threadpoolctl, python-dateutil, six, and
tzdata versions. Loading refuses material Python/scikit-learn/NumPy/joblib
incompatibilities and warns when legacy metadata has no training environment.

```bash
python -m unittest discover -s tests -v

python main.py evaluate-guardrails \
  --input data/spindle_predictive_maintenance_mock.csv \
  --models-root models \
  --output output/mock_guardrail_evaluation.json

python main.py replay \
  --input data/spindle_predictive_maintenance_mock.csv \
  --csv output/mock_replay_revised_guardrail.csv \
  --lifecycles-csv output/mock_lifecycles_revised_guardrail.csv \
  --invalid-csv output/mock_invalid_rows_revised_guardrail.csv \
  --database output/mock_monitor_revised_guardrail.db \
  --progress-every 5000
```

Training and explicit promotion remain:

```bash
python main.py train --input data/spindle_predictive_maintenance_mock.csv \
  --database output/mock_monitor.db --metrics output/mock_model_metrics.json

python main.py evaluate-model --database output/mock_monitor.db \
  --output output/mock_model_evaluation_after_promotion.json --promote
```

## Limitations

The bundled data is synthetic. These metrics validate software behavior and a reproducible backtest, not real spindle accuracy, calibration, safety, or deployment readiness. “Unsafe-late rate” currently means any positive prediction error, with no plant-approved tolerance, so it is deliberately sensitive. Only four validation and four test lifecycles are available. Real deployment needs representative plant regimes, confirmed labels, approved alert tolerances, drift monitoring, prospective validation, and human-reviewed promotion.

## Data domains and realistic-data readiness

`profile-data` calls the same `InputValidator` as replay and offline training.
Its `input_validation` block reports valid/invalid row counts, exact reason
counts, and representative row/timestamp examples. Those failures are also
carried into suitability refusals. The shipped negative fixtures cover an
out-of-range sensor, an impossible jump, an unknown status, and a duplicate
timestamp.

For realistic-synthetic files with explicit maintenance metadata, lifecycle
IDs and boundaries remain metadata-authoritative while the replay audit state
is assigned causally as `HEALTHY`, `DEGRADING`, `WARNING`, `CRITICAL`,
`RECOVERY_CONFIRMATION`, or `RESET_COMPLETE`. `lifecycle_state` is audit-only:
it is not a model feature or label. `elapsed_lifecycle_hours` remains in the
runtime feature schema for audit/support evidence, but new ETA and probability
estimators exclude absolute lifecycle age from predictive inputs; its boundary
calculation is unchanged. The first observation has
unknown source cadence (blank CSV / SQL `NULL`) and a zero elapsed sampling gap;
it is never reported as a fabricated one-second source interval.

The repository has three auditable domains. `accelerated_mock` is for fast software correctness and keeps the existing accelerated dataset. `realistic_synthetic` is for realistic-duration integration and candidate-model testing. `plant` is reserved for actual plant evidence. Their roots are `models/mock`, `models/realistic`, `models/plant` and `output/mock`, `output/realistic`, `output/plant`. Synthetic candidates cannot become `plant_production`; refusal reasons are written to CLI JSON, registry audit JSONL, and SQLite evaluation records.

Sampling rate and degradation rate are different. A one-minute sample rate describes observation cadence; it does not imply that a spindle degrades in minutes. Accelerated degradation is useful for quick reset, replay, and safety-policy tests, but its accuracy metrics are not real-world accuracy. The 6 h, 12 h, and 24 h probability names are event horizons, not maximum lifecycle lengths.

The fixed input contract remains exactly `timestamp`, `vibration_mps2`, `temperature_c`, `current_ampere`, and `health_status`. Lifecycle completion requires a confirmed reset or explicit maintenance record. A file ending in normal, warning, or critical state is `open_normal`, `open_warning`, or `open_critical` and censored. Open/censored rows are not exact remaining-time regression labels. Shipped training defaults require 10 completed lifecycles, at least 4 validation lifecycles, and at least 4 untouched test lifecycles; plant promotion evaluation requires 20.

Features use causal 5 min, 15 min, 30 min, 1 h, 3 h, 6 h, 12 h, and 24 h timestamp windows. Every window reports available history duration, coverage fraction, maturity, sample count, and gap/discontinuity. Gap/interpolation behavior is configured in `config/data.json`. Offline preparation, training, and replay share `safe_resample`; interpolation is off by default and, when enabled, is limited to short normal-to-normal spans. It never crosses explicit/inferred lifecycle boundaries, long outages, or warning/critical spans. Large gaps either mark features unavailable or reject input according to `large_gap_policy`. The live online monitor does not interpolate; it processes observations causally as received.

The normal replay CSV is the audit export (no hidden detailed flag is required). Stable columns include row provenance; domain, dataset, generator, stage and model version; raw/stabilized/event status; lifecycle and censoring state; feature/history maturity; source/effective cadence and sampling OOD; raw ML predictions and confirmed targets; confidence/support/physical-validity flags; guardrail outcome; and fallback/refusal reason. SQLite stores the same audit payload plus indexed operational columns.

With anomaly auditing enabled, the same CSV and SQLite rows also include `quality_status`, detailed `anomaly_type`, safety/model actions, affected sensors, confidence multiplier, causal and offline decisions, evidence/configuration snapshots, and training eligibility. Invalid rows carry parallel fields. SQLite migration is additive and includes `anomaly_events`. `evaluate-anomalies`, `summarize-anomalies`, `export-anomaly-intervals`, `inspect-anomaly-state`, and `explain-prediction` provide deterministic evaluation and audit inspection.

Future plant training requires `--anomaly-screening`; an unscreened plant command is refused. `--reproduction-mode` intentionally preserves existing synthetic reproduction without changing model artifacts. `audit-training-screening` reports exclusions, human-review rows, interpolation policy, detector version, and configuration hash without training. Interval export supports JSON plus optional CSV, and `export-anomaly-events` retains a separate raw causal event export.

Runtime predictions are never silently clipped. Both lower and upper training target support are checked. A negative raw prediction is retained for audit, marked outside support and physically invalid, assigned low confidence, and replaced operationally by an acceptable statistical fallback or withheld.

Windows activation on Windows:

```powershell
.\.venv-win\Scripts\Activate.ps1
```

Exact realistic workflow:

```powershell
python -m unittest discover -s tests -v
python main.py profile-data --input data/spindle_predictive_maintenance_10000_unlabeled.csv --output output/realistic/actual_data_profile.json --models-root models/realistic
python main.py generate-realistic-mock --profile output/realistic/actual_data_profile.json --lifecycles 30 --output data/realistic_spindle_mock.csv --metadata-output data/realistic_spindle_mock_metadata.json --seed 42
python main.py train-model --input data/realistic_spindle_mock.csv --data-domain realistic_synthetic --models-root models/realistic --metrics output/realistic/model_metrics.json --database output/realistic/training.db
python main.py evaluate-model --models-root models/realistic --output output/realistic/model_evaluation.json --database output/realistic/training.db
python main.py evaluate-guardrails --input data/realistic_spindle_mock.csv --models-root models/realistic --output output/realistic/guardrail_evaluation.json
python main.py replay --input data/realistic_spindle_mock.csv --data-domain realistic_synthetic --models-root models/realistic --csv output/realistic/replay.csv --lifecycles-csv output/realistic/lifecycles.csv --invalid-csv output/realistic/invalid_rows.csv --database output/realistic/monitor.db --progress-every 5000
python main.py evaluate-model --models-root models/realistic --output output/realistic/promotion_refusal.json --database output/realistic/training.db --promote
```

The existing `train` command remains as a backward-compatible alias of `train-model`. Full one-minute realistic training/replay is intentionally much slower than accelerated testing. `reproduce.ps1` installs the exact lock and runs the verified workflow. `build_packages.ps1` builds the clean source ZIP and separate large evidence bundle, rejects archive entries containing backslashes, and writes `package_manifest.json` beside both ZIPs as well as under `output`. See `docs/realistic_data_readiness.md` for configuration, audit details, actual metrics, and limitations.

## Generalization v4 research remediation

The rejected `generalization_v3` candidate remains isolated and `models/realistic` is unchanged. The v4 code path removes absolute/rolling level proxies from newly trained probability estimators, expands healthy high-load support with outcome-stratified duration coverage, adds locked-v3 multivariate/attribution diagnostics, and replaces the slow wide acceptance metric path with a narrow cached/vectorized evaluator. The runtime feature schema and manufacturer safety logic are unchanged.

Run `run_generalization_v4.ps1` with no switches for diagnosis only. Research retraining requires the explicit `-RunResearchTraining` switch and writes only to `models/generalization_v4/realistic`; the script never promotes. See `docs/model_generalization_v4.md` for the exact workflow, safety constraints, and current limitation that v4 accuracy has not been measured in the packaged snapshot because the large local datasets/replays are not included.

### Generalization v5.2 research pass

`run_generalization_v5_2.ps1` reuses the v4 development CSV and adds noise-normalized relative-state residuals, persistent trend signal-to-noise features, and training-only lifecycle-held-out hard-example mining for the required probability targets. It writes only to `models/generalization_v5_2/realistic` and `output/generalization_v5_2`; production promotion remains disabled. See `docs/model_generalization_v5_2.md`.
