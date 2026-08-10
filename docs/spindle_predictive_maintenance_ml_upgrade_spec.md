# Forecasting Design Addendum

This document supersedes the earlier reset-time-prediction design.

## Core rule

Maintenance/reset timing is a human decision and is not a machine-health target. The monitoring system must not predict a reset time, maintenance time, or whether a human will intervene before a critical event.

## Forecast targets

For each inferred lifecycle, models may be trained only against observed machine-condition events:

```text
time_to_warning_hours  = first_warning_timestamp - current_timestamp
time_to_critical_hours = first_critical_timestamp - current_timestamp
```

Only strictly pre-event readings are regression samples: their timestamp must precede the first warning or critical event and their remaining-time target must be positive. Event and post-event rows are excluded rather than assigned a zero target. Warning models use lifecycles that reached warning; critical models use lifecycles that reached critical. A lifecycle maintained before critical is right-censored for the critical target and is not relabelled as a critical event.

The ML registry contains independent regressors for the two time targets and classifiers for:

```text
probability_warning_6h, probability_warning_12h, probability_warning_24h
probability_critical_6h, probability_critical_12h, probability_critical_24h
```

## Lifecycle resets

Reset detection remains enabled solely to segment the data after sustained multi-sensor recovery. It clears historical filter and feature state only when confirmed. A `RESET_CANDIDATE` suppresses statistical, ML, and final forecasts until the boundary is accepted or rejected.

## Final forecast policy

Manufacturer status remains authoritative. A current warning forces time-to-warning to zero; a current critical forces both forecasts to zero. A low-confidence forecast or a strongly conflicting ML/statistical estimate is not copied into the final forecast columns. The separate raw model/baseline outputs are retained for audit.
