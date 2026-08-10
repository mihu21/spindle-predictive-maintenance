Spindle Prognostics V6.0.4

Purpose
- Restore WARNING operational event sensitivity lost in V6.0.3 while preserving its large reduction in alert chatter.

Changes
- WARNING probability activation persistence: 10 minutes.
- CRITICAL probability activation persistence: unchanged at 30 minutes.
- Release hysteresis remains unchanged: release boundary = 70% of threshold and 60 minutes sustained below that boundary.
- V6.0.2 anomaly/degradation handling remains unchanged.
- Added regression tests for target-specific activation and brief-warning-spike rejection.

Rationale
- V6.0.3 achieved 14/14 CRITICAL operational detections and reduced WARNING false-alert episodes to ~1.02/100 healthy hours, but WARNING operational detection fell to 12/14.
- The shorter WARNING activation gate addresses that specific bottleneck without weakening CRITICAL behavior or release debounce.

Validation
- Run: python -m pytest -q
- Then replay the same dataset and inspect operational_event_false_negative_rate and operational_false_alert_episodes_per_100_hours.
