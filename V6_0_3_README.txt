Spindle Prognostics V6.0.3 - Alert Chatter Hardening

Goal
- Preserve the V6.0.2 anomaly-policy correction that restored raw event detection to 14/14 WARNING and 14/14 CRITICAL on the latest evaluation.
- Reduce repeated operational probability-alert episodes without delaying initial activation.
- Make evaluation distinguish raw probability detection from actual operational alert-policy detection.

Runtime policy changes
1. Activation persistence remains 30 minutes.
   This is deliberately unchanged because the latest WARNING evaluation included a lifecycle with only 0.5 h raw lead time.
2. Release hysteresis is strengthened:
   - probability_release_fraction: 0.70 (was 0.80)
   - probability_release_persistence_minutes: 60.0 (new)
3. Once an alert target is operationally active, a brief probability dip does not release it.
   Release occurs only after probability remains below threshold * 0.70 for 60 continuous minutes.
4. A recovery above the lower release boundary cancels a pending release.
5. Withheld forecasts and lifecycle resets still clear all alert state.
6. The V6.0.2 anomaly logic is unchanged.

Evaluator changes
- Existing raw event detection metrics remain for backward compatibility.
- Adds operational_detected_event_count and operational_event_false_negative_rate using probability_threshold_crossings.
- Adds operational event lead-time metrics.
- Adds raw/operational false-alert hours.
- Adds operational false-alert time fraction.
- Adds median/p90/maximum false-alert episode duration.

Why this matters
The prior evaluator reported 14/14 detection from raw probability crossings while operational false-alert episodes were measured after persistence. V6.0.3 exposes both layers so alert-policy tuning cannot hide lost operational detections.

Verification
- Full regression suite: 190 passed, 5 warnings, 2 subtests passed.
- Warnings are the existing model-environment compatibility warnings.

Important
The latest post-V6.0.2 replay.csv was not available while building this patch, so the reduction from the previously measured 14.17 WARNING operational false-alert episodes per 100 healthy hours has NOT yet been measured. Re-run the same replay with V6.0.3 and inspect the new operational metrics before changing activation persistence or probability thresholds.
