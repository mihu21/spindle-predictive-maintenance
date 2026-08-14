# VVB001 Vibration-Derived Operating Eligibility

## Outcome

The plant-shadow runtime can now operate safely when the sensor database has no machine-state
columns and the sensor continues recording while the spindle is idle, off, or under maintenance.
It does not infer exact equipment modes from vibration. It learns a machine-specific,
production-like vibration regime and admits the frozen prognostics runtime only after that regime
is stable, familiar, high-confidence, and persistent.

All other cases remain `UNKNOWN`; exact WARNING and CRITICAL RUL values are cleared and the
forecast is `PAUSED`. Raw telemetry and immediate manufacturer safety handling remain active.
The frozen v2.7 model, calibration, selectors, thresholds, and plant authorization are unchanged.

## Decision hierarchy

1. A bounded local PLC, CMMS, or operator interval, or mapped authoritative source column, wins.
2. Without authoritative evidence, the vibration detector may emit only inferred `RUNNING`.
3. Quiet, uncalibrated, unstable, impulsive, extreme, or unfamiliar vibration emits `UNKNOWN`.
4. Contradictory authoritative evidence also emits `UNKNOWN` and cannot be overridden by vibration.

Vibration alone never claims `OFF`, `IDLE`, or `MAINTENANCE`. Those modes are not reliably
identifiable from aggregate `vrms`, `arms`, `apeak`, `crest`, and temperature values.

## Causal detector

The detector is independent for each `source_key::line_sel::machine_id` and uses only present and
past rows. Its default contract is:

- 30-minute short stability window, supporting plant cadences through ten minutes;
- up to 24 hours of bounded calibration history;
- at least 60 stable calibration samples spanning at least four hours;
- a clearly separated quiet and production-like energy regime;
- a minimum 1.8x energy ratio between learned regimes;
- crest-factor consistency, variability, impulse, novelty, and raw-extreme rejection;
- at least two minutes of continuous production-like evidence;
- at least 0.95 activation confidence.

If only one regime is observed, calibration remains incomplete indefinitely rather than inventing
a RUNNING threshold. The learned centers, separation, sample support, decision reason, confidence,
and calibration artifact SHA-256 are persisted for every committed observation.

Known commissioning labels can seed the two regimes through short operator intervals. This is
optional; an unsupervised bimodal calibration is used when sufficiently distinct quiet and
production periods are naturally present.

## Runtime and recovery

Rows used to establish calibration are preserved even before a lifecycle exists. They rebuild only
the vibration detector; they are not replayed into degradation features. The prognostic monitor
starts only at the first confirmed-running lifecycle row. Duplicate and late rows never advance the
detector. After a failed SQLite commit, the detector, operating clock, and frozen model runtime are
rebuilt from committed causal evidence.

The SQLite schema is
`plant_shadow_schema_v3_vibration_operating_inference`. The append-only
`vibration_operating_inferences` table and read-only API/frontend expose calibration and decisions.

## Source configuration without machine logs

Leave all four optional operating-context mappings as JSON `null`:

```json
"operating_state": null,
"operating_state_source": null,
"operating_state_confidence": null,
"maintenance_event_id": null
```

The supplied `config/plant_shadow_sources.example.json` now uses this configuration. If a reliable
state source is added later, map those columns or add bounded local evidence; it automatically takes
priority over inference.

## Synthetic runtime fixture

Generate a fixture whose visible operating-state fields are deliberately blank while hidden truth
is retained only for test assertions:

```powershell
.venv\Scripts\python.exe main.py generate-mock `
    --output data\vvb001_vibration_operating_test.csv `
    --lifecycles 8 `
    --machines 4 `
    --cadence-seconds 600 `
    --seed 913 `
    --duty-cycled-operating-context `
    --vibration-inference-fixture
```

The training loader rejects this fixture. Neither hidden operating truth nor runtime-generated
eligibility decisions can enter model fitting.

## Important limitation

Aggregate vibration cannot guarantee that a stable power tool used during maintenance will never
resemble production. Record known maintenance windows with `plant-shadow operating-state-add` when
possible. Such an override immediately pauses RUL and is always preferred to vibration inference.
Do not treat vibration eligibility as a personnel-safety interlock or machine-control signal.
