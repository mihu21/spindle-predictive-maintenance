SPINDLE PREDICTIVE MAINTENANCE — V6 REDESIGN
============================================

V6 retires the supervised synthetic failure classifier as the PRIMARY forecasting method.
The existing manufacturer safety, anomaly handling, lifecycle, replay, storage, and audit
infrastructure is preserved.

PRIMARY V6 METHOD
-----------------
1. Manufacturer raw WARNING/CRITICAL thresholds remain authoritative and immediate.
2. Sensor-integrity anomalies stay separate from equipment degradation.
3. A healthy baseline is learned only from trusted NORMAL operation and is gated so
   persistent degradation cannot be absorbed into the healthy reference.
4. Causal 1h/3h/6h/12h/24h trend windows estimate a robust persistent degradation rate.
5. Disagreement between time-scale slopes becomes rate uncertainty.
6. Threshold-crossing probabilities are computed directly from the uncertain rate:
      crossing by horizon h <=> degradation_rate >= (threshold - state) / h
7. WARNING/CRITICAL ETA, uncertainty intervals, and 6/12/24h probabilities are produced.
8. Probability recommendations require time persistence/hysteresis to avoid alert flicker.
9. Legacy supervised ML can be recorded with --legacy-ml-audit but cannot override v6.

NO LARGE TRAINING DATA REQUIRED
-------------------------------
V6 does not need realistic_spindle_long_v4.csv, the v5.x training database, or a trained
probability classifier for normal operation. Synthetic generators remain useful for
software/stress testing only.

RUN
---
PowerShell:
    Unblock-File .\run_prognostics_v6.ps1
    .\run_prognostics_v6.ps1

Default input:
    data\spindle_predictive_maintenance_10000_unlabeled.csv

Default output:
    output\prognostics_v6\replay.csv
    output\prognostics_v6\evaluation.json
    output\prognostics_v6\lifecycles.csv
    output\prognostics_v6\invalid_rows.csv

SQLite is disabled by default to reduce disk usage. Add -WithDatabase if wanted.
Anomaly auditing is enabled by the runner unless -DisableAnomalyAudit is supplied.

RESEARCH-ONLY LEGACY ML
-----------------------
To record the old classifier beside v6 without letting it affect the primary forecast:
    .\run_prognostics_v6.ps1 -LegacyMLAudit

VALIDATION
----------
Full regression suite at creation time:
    183 passed, 5 expected legacy-model environment warnings, 2 subtests passed.

A replay of the already-inspected 10,000-row plant trajectory was used only as a sanity
check. It is NOT independent validation and must not be presented as real-world accuracy.
