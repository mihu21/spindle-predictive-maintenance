# V6 — Hybrid Model-Based Probabilistic Prognostics

## Decision

V6 stops using the synthetic-lifecycle supervised probability classifier as the primary runtime forecast. The rest of the project is preserved: input validation, manufacturer rules, anomaly handling, lifecycle/reset logic, feature history, replay, SQLite/CSV audit, and legacy model tooling.

The legacy ML stack remains available only for explicit research comparison. It cannot override the v6 primary output.

## Why the method changed

The v4→v5.2 experiments showed a recurring sensitivity/specificity tradeoff and large validation→test shifts even as classifier ranking improved. The fundamental limitation is that the training evidence contains only a small number of independent synthetic failure trajectories. Additional synthetic rows increase row count but do not create empirical evidence about how the physical spindle fails.

The available plant signal is better matched by a first-passage/degradation problem:

> Given current sensor state, persistent degradation rate, and uncertainty, when will vibration, temperature, or current cross the manufacturer WARNING or CRITICAL threshold?

That formulation naturally produces ETA and 6/12/24-hour probabilities without supervised failure labels.

## Runtime architecture

```text
raw sensor row
    ↓
input validation
    ↓
sensor-integrity anomaly layer
    ↓
manufacturer threshold status ─────────→ immediate safety result
    ↓
causal smoothing + rolling feature history
    ↓
trusted-NORMAL adaptive healthy baseline
    ↓
robust multi-timescale degradation-rate distribution
    ↓
manufacturer threshold first-passage calculation
    ↓
WARNING ETA + interval
CRITICAL ETA + interval
P(WARNING <= 6/12/24h)
P(CRITICAL <= 6/12/24h)
    ↓
time-persistent advisory policy
```

## Robust degradation-rate estimate

For each sensor, v6 uses causal trend windows configured in `config/prognostics.json`:

- 60 min
- 180 min
- 360 min
- 720 min
- 1440 min

Only windows with enough samples, coverage, history, and no large gap are used. Their slopes are combined with a weighted median. Longer windows receive more evidence weight, but only proportional to the square root of duration so a 24-hour window cannot completely suppress a real recent trend.

The median absolute deviation of the accepted slopes estimates cross-timescale rate uncertainty. Inconsistent/noisy slopes are shrunk toward zero and the removed amount increases uncertainty. This is intentionally simpler and more interpretable than the 141-feature synthetic classifier.

## Threshold-crossing probability

Let:

- `x` = current smoothed sensor state
- `a` = manufacturer target threshold
- `d = a - x` = remaining distance
- `S` = uncertain persistent degradation rate
- `h` = forecast horizon

A crossing by horizon `h` occurs if:

```text
S >= d / h
```

V6 approximates `S` with a Normal distribution whose mean is the robust/shrunk multi-window slope and whose standard deviation comes from slope disagreement plus a physical floor.

Therefore:

```text
P(cross by h) = P(S >= d/h)
```

This gives monotonic probabilities by construction. The machine-level probability is the maximum per-sensor probability rather than an independence union, because vibration/temperature/current are correlated and an independence assumption would overstate risk.

## ETA and uncertainty interval

If the median persistent rate is positive, median ETA is:

```text
ETA = distance / median_rate
```

An uncertainty interval is calculated using lower/upper rate quantiles. If the lower credible rate is non-positive, the latest ETA is intentionally left unavailable rather than fabricated.

## Healthy baseline

V6 keeps an adaptive machine-specific healthy baseline for the multivariate health-deviation score. It updates only when:

- raw manufacturer status is NORMAL;
- the anomaly layer permits feature use;
- no human-review anomaly is active;
- the observation is sufficiently close to the current healthy baseline.

The final condition is important: once a persistent degradation state has moved sufficiently far from the healthy reference, it cannot slowly redefine itself as healthy.

The baseline supports deviation/confidence interpretation; manufacturer thresholds remain authoritative.

## Anomaly vs degradation

V6 explicitly keeps two concepts separate.

### Sensor-integrity anomaly

Examples:

- malformed/out-of-range reading
- spike
- clipping
- stuck sensor
- excessive noise
- data gap
- impossible discontinuity

These can reduce confidence or withhold prediction.

### Equipment degradation

Examples:

- persistent vibration increase
- persistent temperature increase
- sustained current increase
- multi-timescale trend toward WARNING/CRITICAL

These increase threshold-crossing risk and must not be filtered away as sensor faults merely because they are unusual.

## Recommendation persistence

Raw forecast probabilities are never modified for audit/calibration. However, an advisory probability threshold must remain crossed for the configured persistence duration (30 minutes by default) before it becomes a recommendation trigger. A release hysteresis fraction prevents rapid on/off chatter.

This separates:

- **probability estimate**: instantaneous model belief;
- **operational advisory**: time-persistent decision policy.

Automated action remains disabled by default. Manufacturer safety output is still immediate.

## Legacy ML behavior

The old `MLForecaster` is retained for research, regression tests, and historical artifact compatibility. `main.py replay` does not run it unless `--legacy-ml-audit` is supplied. Even then, `apply_ml_forecast()` attaches the result only as an audit field; it cannot replace v6 primary ETA/probabilities.

Legacy `train`, `train-model`, and `evaluate-model` commands remain present because removing them would destroy historical research reproducibility. They are not part of the v6 primary method.

## Evaluation changes

`evaluate_prognostics_v6.py` emphasizes event-oriented behavior:

- row FN/FP remains available;
- confirmed-event detection rate;
- first alert lead time;
- raw probability false-alert episodes;
- operational persistent false-alert episodes per 100 eligible hours;
- Brier score;
- expected calibration error;
- ETA absolute error within the requested horizon.

`event_status` is used for confirmed machine-event evaluation. One-sample raw manufacturer crossings remain immediate safety events but are not treated as confirmed degradation labels.

## Development sanity check

During implementation, the existing 10,000-row trajectory was replayed only as a non-independent sanity check. At the configured 0.5 probability threshold, the event-oriented evaluator found both the confirmed WARNING-12h and CRITICAL-24h events within their requested horizons. This is useful evidence that the implementation behaves coherently on the available trajectory, but it is **not** independent accuracy evidence because the trajectory had already been inspected during design.

Do not use those development numbers as a production-performance claim.

## Memory/runtime behavior

V6 does not generate a new large synthetic dataset and does not train a classifier. Runtime state is bounded by the existing rolling histories and small per-sensor baseline state. Threshold probability calculation is analytical rather than Monte Carlo, so replay remains lightweight.

SQLite is optional in `run_prognostics_v6.ps1`; it is disabled by default to reduce disk usage.

## Run

```powershell
Unblock-File .\run_prognostics_v6.ps1
.\run_prognostics_v6.ps1
```

For research-only legacy ML comparison:

```powershell
.\run_prognostics_v6.ps1 -LegacyMLAudit
```

For SQLite audit storage:

```powershell
.\run_prognostics_v6.ps1 -WithDatabase
```

## Validation status at implementation

- full regression suite: 183 passed;
- existing manufacturer immediate safety behavior preserved;
- v6-specific flat/rising/probability-persistence/legacy-ML-isolation tests added;
- no production model promotion performed;
- no synthetic training dataset generated.
