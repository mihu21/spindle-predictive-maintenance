from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from .config import ProjectConfig
from .io import read_csv_records
from .lifecycle_detector import LifecycleDetector
from .models import LifecycleRecord, LifecycleSnapshot, Status, ValidationResult
from .resampling import safe_resample
from .rules import severity_for_value, status_for_value
from .smoothing import SignalSmoother
from .status_policy import StabilizedStatusPolicy, TimeBasedEventPolicy


@dataclass(frozen=True)
class OfflineLifecyclePlan:
    """Confirmed boundaries discovered before the authoritative offline replay."""

    first_timestamp: datetime | None
    records: tuple[LifecycleRecord, ...]
    lifecycle_states: dict[datetime, str] = field(default_factory=dict)
    last_timestamp: datetime | None = None
    open_lifecycle_state: str = "open_normal"

    @property
    def boundary_timestamps(self) -> tuple[datetime, ...]:
        return tuple(record.end_timestamp for record in self.records)

    def is_boundary(self, timestamp: datetime) -> bool:
        return any(record.end_timestamp == timestamp for record in self.records)

    def snapshot(self, timestamp: datetime) -> LifecycleSnapshot:
        prior = [record for record in self.records if record.end_timestamp <= timestamp]
        lifecycle_number = len(prior) + 1
        lifecycle_id = f"lifecycle_{lifecycle_number:04d}"
        start = prior[-1].end_timestamp if prior else self.first_timestamp or timestamp
        elapsed = max(0.0, (timestamp - start).total_seconds() / 3600.0)

        for record in self.records:
            if record.end_timestamp <= timestamp < record.inferred_reset_timestamp:
                return LifecycleSnapshot(
                    lifecycle_id,
                    "RESET_CANDIDATE",
                    elapsed,
                    reset_reason="Reset confirmation is in progress.",
                )
            if timestamp == record.inferred_reset_timestamp:
                return LifecycleSnapshot(
                    lifecycle_id,
                    self.lifecycle_states.get(timestamp, "RESET_COMPLETE"),
                    elapsed,
                    reset_confidence=record.reset_confidence,
                    reset_reason=record.reset_reason,
                    completed_lifecycle=record,
                )

        state = self.lifecycle_states.get(timestamp, "HEALTHY")
        if self.last_timestamp is not None and timestamp == self.last_timestamp:
            state = self.open_lifecycle_state
        return LifecycleSnapshot(
            lifecycle_id,
            state,
            elapsed,
        )


def read_validated_input(
    input_path: str | Path,
    config: ProjectConfig,
    *,
    valid_limit: int | None = None,
) -> list[ValidationResult]:
    validations: list[ValidationResult] = []
    accepted = 0
    for validation in read_csv_records(input_path, config):
        validations.append(validation)
        if validation.valid:
            accepted += 1
            if valid_limit not in (None, 0) and accepted >= valid_limit:
                break
    return validations


def discover_lifecycle_plan(
    validations: list[ValidationResult],
    config: ProjectConfig,
    models_root: str | Path,
) -> OfflineLifecyclePlan:
    # Imported lazily to avoid a monitor/offline import cycle.
    from .monitor import ConditionMonitor

    # Boundary discovery depends only on manufacturer rules and lifecycle state.
    # Loading or running a production model here is both wasteful and a source of
    # accidental coupling between lifecycle assignment and model availability.
    monitor = ConditionMonitor(
        config, models_root=models_root, enable_ml=False, enable_features=False
    )
    first_timestamp: datetime | None = None
    lifecycle_states: dict[datetime, str] = {}
    last_result = None
    for validation in validations:
        if not validation.valid or validation.reading is None:
            continue
        first_timestamp = first_timestamp or validation.reading.timestamp
        result = monitor.process(validation.reading)
        last_result = result
        lifecycle_states[validation.reading.timestamp] = result.lifecycle_state
    return OfflineLifecyclePlan(
        first_timestamp,
        tuple(monitor.lifecycle.completed),
        lifecycle_states,
        last_timestamp=last_result.timestamp if last_result else None,
        open_lifecycle_state=(f"open_{last_result.raw_status.name.lower()}" if last_result else "invalid"),
    )


def _causal_metadata_states(
    validations: list[ValidationResult],
    records: list[LifecycleRecord],
    config: ProjectConfig,
) -> dict[datetime, str]:
    """Reuse monitor policies to assign audit-only states at fixed boundaries.

    The explicit metadata controls IDs/boundaries.  These states do not feed
    features, labels, elapsed lifecycle time, or reset discovery.
    """
    boundaries = {record.end_timestamp for record in records}
    states: dict[datetime, str] = {}
    smoother = SignalSmoother(config)
    stabilized = StabilizedStatusPolicy(config.monitoring)
    events = TimeBasedEventPolicy(config.lifecycle)
    detector = LifecycleDetector(config)
    previous_timestamp: datetime | None = None
    for validation in validations:
        if not validation.valid or validation.reading is None:
            continue
        reading = validation.reading
        at_boundary = reading.timestamp in boundaries
        if at_boundary:
            smoother = SignalSmoother(config)
            stabilized = StabilizedStatusPolicy(config.monitoring)
            events = TimeBasedEventPolicy(config.lifecycle)
            detector = LifecycleDetector(config)
            previous_timestamp = None
        values = reading.values()
        dt_seconds = 1.0 if previous_timestamp is None else max(
            1e-6, (reading.timestamp - previous_timestamp).total_seconds()
        )
        previous_timestamp = reading.timestamp
        signals = smoother.update(values, dt_seconds)
        raw_status = max(
            status_for_value(values[key], sensor, config)
            for key, sensor in config.sensors.items()
        )
        effective_status = stabilized.update(raw_status)
        event_status = events.update(reading.timestamp, raw_status)
        maximum_severity = max(
            severity_for_value(signals[key].kalman_level, sensor)
            for key, sensor in config.sensors.items()
        )
        snapshot = detector.update(
            timestamp=reading.timestamp,
            status=effective_status,
            event_status=event_status,
            smoothed_values={key: signals[key].kalman_level for key in signals},
            maximum_severity=maximum_severity,
        )
        if at_boundary:
            states[reading.timestamp] = "RESET_COMPLETE"
        elif snapshot.lifecycle_state == "RESET_CANDIDATE":
            states[reading.timestamp] = "RECOVERY_CONFIRMATION"
        elif event_status >= Status.CRITICAL:
            states[reading.timestamp] = "CRITICAL"
        elif event_status >= Status.WARNING:
            states[reading.timestamp] = "WARNING"
        elif snapshot.lifecycle_state == "DEGRADING":
            states[reading.timestamp] = "DEGRADING"
        else:
            states[reading.timestamp] = "HEALTHY"
    return states


def _metadata_lifecycle_plan(
    input_path: str | Path, validations: list[ValidationResult], config: ProjectConfig
) -> OfflineLifecyclePlan | None:
    path = Path(input_path)
    candidates = [
        path.with_name(f"{path.stem}_metadata.json"),
        path.with_suffix(".metadata.json"),
    ]
    metadata_path = next((value for value in candidates if value.exists()), None)
    if metadata_path is None:
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("data_domain") != "realistic_synthetic" or not metadata.get("lifecycles"):
        return None
    records: list[LifecycleRecord] = []
    for value in metadata["lifecycles"]:
        maintenance = datetime.fromisoformat(value["maintenance_record_timestamp"])
        records.append(LifecycleRecord(
            lifecycle_id=value["lifecycle_id"],
            start_timestamp=datetime.fromisoformat(value["start_timestamp"]),
            end_timestamp=datetime.fromisoformat(value["end_timestamp"]),
            duration_hours=float(value["duration_hours"]),
            highest_status=value["highest_status"],
            first_warning_timestamp=datetime.fromisoformat(value["first_warning_timestamp"]) if value.get("first_warning_timestamp") else None,
            first_critical_timestamp=datetime.fromisoformat(value["first_critical_timestamp"]) if value.get("first_critical_timestamp") else None,
            critical_reached=bool(value.get("critical_reached")),
            inferred_reset_timestamp=maintenance,
            reset_confidence="EXPLICIT",
            reset_reason="Explicit maintenance record from realistic-synthetic generator metadata.",
            pre_reset_values={}, post_reset_values={},
            lifecycle_state="completed_with_maintenance_record",
            censored=False,
            first_raw_warning_timestamp=datetime.fromisoformat(value["first_raw_warning_timestamp"]) if value.get("first_raw_warning_timestamp") else None,
            first_confirmed_warning_timestamp=datetime.fromisoformat(value["first_confirmed_warning_timestamp"]) if value.get("first_confirmed_warning_timestamp") else None,
            first_raw_critical_timestamp=datetime.fromisoformat(value["first_raw_critical_timestamp"]) if value.get("first_raw_critical_timestamp") else None,
            first_confirmed_critical_timestamp=datetime.fromisoformat(value["first_confirmed_critical_timestamp"]) if value.get("first_confirmed_critical_timestamp") else None,
            operating_regime=str(value.get("operating_regime", "unknown")),
            degradation_family=str(value.get("degradation_family", "unknown")),
        ))
    first = next((item.reading.timestamp for item in validations if item.valid and item.reading), None)
    states = _causal_metadata_states(validations, records, config)
    last = next((item.reading for item in reversed(validations) if item.valid and item.reading), None)
    label = next((item.source_label for item in reversed(validations) if item.valid and item.reading), "normal") or "normal"
    return OfflineLifecyclePlan(
        first, tuple(records), states,
        last_timestamp=last.timestamp if last else None,
        open_lifecycle_state=f"open_{label.lower()}" if label.lower() in {"normal", "warning", "critical"} else "censored",
    )


def prepare_offline_replay(
    input_path: str | Path,
    config: ProjectConfig,
    models_root: str | Path,
    *,
    valid_limit: int | None = None,
) -> tuple[list[ValidationResult], OfflineLifecyclePlan]:
    validations = read_validated_input(input_path, config, valid_limit=valid_limit)
    metadata_plan = _metadata_lifecycle_plan(input_path, validations, config)
    plan = metadata_plan or discover_lifecycle_plan(validations, config, models_root)
    accepted = [value for value in validations if value.valid and value.reading is not None]
    if not accepted:
        return validations, plan
    processed = safe_resample(
        [value.reading for value in accepted if value.reading is not None],
        target_interval_seconds=config.target_sampling_interval_seconds,
        interpolation_enabled=config.interpolation_enabled,
        maximum_interpolation_gap_minutes=config.maximum_interpolation_gap_minutes,
        large_gap_policy=config.large_gap_policy,
        lifecycle_boundary_timestamps=plan.boundary_timestamps,
        source_statuses=[value.source_label for value in accepted],
    )
    originals = {value.reading.timestamp: value for value in accepted if value.reading is not None}
    replay_rows: list[ValidationResult] = [value for value in validations if not value.valid]
    for value in processed:
        original = originals.get(value.processed.timestamp)
        if original is not None and not value.interpolated:
            replay_rows.append(ValidationResult(
                valid=True,
                reason=original.reason,
                row_number=original.row_number,
                raw_row=original.raw_row,
                reading=original.reading,
                source_label=original.source_label,
                interpolated=False,
                source_sampling_interval_seconds=value.source_sampling_interval_seconds,
                effective_resampling_interval_seconds=value.effective_resampling_interval_seconds,
                features_available=value.features_available,
            ))
            continue
        replay_rows.append(ValidationResult(
            valid=True,
            reason="safely interpolated normal-state row",
            reading=value.processed,
            source_label="normal",
            interpolated=True,
            source_sampling_interval_seconds=value.source_sampling_interval_seconds,
            effective_resampling_interval_seconds=value.effective_resampling_interval_seconds,
            features_available=value.features_available,
        ))
    replay_rows.sort(key=lambda value: (
        value.reading.timestamp if value.reading is not None else datetime.min,
        0 if not value.interpolated else 1,
    ))
    return replay_rows, plan
