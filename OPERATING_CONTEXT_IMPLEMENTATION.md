# VVB001 Operating Context and Exposure-Time Implementation

## Outcome

The plant-shadow runtime no longer treats an online sensor as proof that the spindle is operating.
All telemetry remains immutable evidence, while degradation features and RUL advance only during
confirmed operation. The frozen v2.7 model artifact is unchanged.

## Contract

The supported equipment states are:

- `RUNNING`
- `IDLE`
- `OFF`
- `MAINTENANCE`
- `UNKNOWN`

Each observation records the state, source, confidence, optional maintenance event, admission
decision, cumulative operating seconds, effective operating timestamp, reason code, and policy
version. Confirmed runtime admission requires `RUNNING`, a non-`UNAVAILABLE` source, and confidence
of at least 0.90.

State can be supplied by optional read-only PostgreSQL columns or by a local bounded evidence
interval from PLC, CMMS, or an operator. Contradictory overlapping local intervals fail closed to
`UNKNOWN`.

If no state source exists, schema v3 adds a machine-specific vibration eligibility detector. It
uses only causal `vrms`, `arms`, `apeak`, and `crest` stability evidence to learn distinct quiet and
production-like regimes. It may emit only inferred `RUNNING`; every uncalibrated, quiet, impulsive,
extreme, unstable, or novel pattern remains `UNKNOWN`. Authoritative intervals always take
precedence. See `VIBRATION_OPERATING_INFERENCE_IMPLEMENTATION.md` for the calibration and recovery
contract.

When an existing v1 SQLite ledger is opened, historical raw rows are retained and receive an
explicit `LEGACY_CONTEXT_UNAVAILABLE` decision. They are not guessed to have been running and are
not admitted into rebuilt model state.

## Runtime behavior

| State | Raw row retained | Model state advances | Exact RUL |
|---|---:|---:|---:|
| RUNNING, confirmed | Yes | Yes | Subject to frozen v2.7 serviceability |
| IDLE | Yes | No | Cleared / PAUSED |
| OFF | Yes | No | Cleared / PAUSED |
| MAINTENANCE | Yes | No | Cleared / PAUSED |
| UNKNOWN or low confidence | Yes | No | Cleared / PAUSED |

The causal clock accumulates time only between consecutive confirmed-running observations. The
first running row after a pause is a resume warm-up row when no operating time has advanced. The
next confirmed-running row resumes model inference using an effective timestamp that excludes the
pause. Output units are `OPERATING_HOURS`.

Extreme raw safety telemetry remains visible immediately during a pause, but does not contaminate
the feature engine, baseline, predictor hysteresis, or RUL estimator.

## Lifecycle and truth behavior

- OFF/IDLE/MAINTENANCE/UNKNOWN observations cannot create a lifecycle.
- The first confirmed-running observation creates the observed left-censored lifecycle.
- Ordinary maintenance pauses but does not silently renew the spindle.
- Confirmed component replacement closes the old lifecycle; the next confirmed-running row resets
  runtime state and starts a clean operating clock.
- Exact target truth uses cumulative operating exposure when context exists at or after the endpoint.
  Legacy evidence without that coverage retains its wall-clock calculation for backward compatibility.

## Synthetic verification

`generate-mock --duty-cycled-operating-context` injects deterministic idle, shutdown, maintenance,
and unknown periods. Maintenance rows deliberately include disruptive but valid tool vibration.
The training loader refuses any such non-running fixture, preventing accidental model contamination.

Adding `--vibration-inference-fixture` blanks the visible state fields and retains hidden state only
for test assertions. Training rejects this fixture independently of the duty-cycle marker.

## Native integration verification

The updated PostgreSQL 18 E2E scenario completed on Windows using the disposable native cluster.
It verified the operating-state pause/resume path through the production read-only adapter and the
SQLite evidence ledger, then terminated its owned server and released port 55432. This remains a
local synthetic plant-shadow integration test and does not authorize production use.
