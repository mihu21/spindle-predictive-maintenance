from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from statistics import median

from .config import ProjectConfig
from .models import LifecycleRecord, LifecycleSnapshot, Status


class LifecycleDetector:
    """Infer lifecycle resets from sustained, multi-sensor recovery patterns.

    The detector deliberately calls these inferred resets. It never treats a
    one-reading WARNING-to-NORMAL transition as confirmed maintenance.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.lifecycle_number = 1
        self.lifecycle_id = self._format_id(self.lifecycle_number)
        self.start_timestamp: datetime | None = None
        self.state = "HEALTHY"
        self.degraded_since: datetime | None = None
        self.highest_status = Status.NORMAL
        self.first_warning_timestamp: datetime | None = None
        self.first_critical_timestamp: datetime | None = None
        self.first_raw_warning_timestamp: datetime | None = None
        self.first_raw_critical_timestamp: datetime | None = None
        self.candidate_start: datetime | None = None
        self.candidate_pre_values: dict[str, float] = {}
        self.candidate_post: list[tuple[datetime, dict[str, float], float]] = []
        self.candidate_last_timestamp: datetime | None = None
        self.history: deque[tuple[datetime, dict[str, float], float]] = deque()
        self.last_reset_timestamp: datetime | None = None
        self.completed: list[LifecycleRecord] = []

    @staticmethod
    def _format_id(number: int) -> str:
        return f"lifecycle_{number:04d}"

    def _trim_history(self, timestamp: datetime) -> None:
        minutes = max(
            self.config.lifecycle.pre_reset_window_minutes,
            self.config.lifecycle.post_reset_window_minutes,
        )
        cutoff = timestamp - timedelta(minutes=max(minutes * 2.0, 60.0))
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def _window_medians(
        self,
        points: list[tuple[datetime, dict[str, float], float]],
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for key in self.config.sensors:
            sensor_values = [values[key] for _, values, _ in points]
            result[key] = float(median(sensor_values)) if sensor_values else 0.0
        return result

    def _pre_points(self, timestamp: datetime) -> list[tuple[datetime, dict[str, float], float]]:
        cutoff = timestamp - timedelta(minutes=self.config.lifecycle.pre_reset_window_minutes)
        return [point for point in self.history if point[0] >= cutoff]

    def _recovery_metrics(
        self,
        pre: dict[str, float],
        post: dict[str, float],
        post_severity: float,
    ) -> tuple[int, float, bool, list[str]]:
        improved = 0
        notes: list[str] = []
        pre_severities: list[float] = []
        close_to_healthy = True
        for key, sensor in self.config.sensors.items():
            span = max(sensor.critical - sensor.healthy_baseline, 1e-9)
            drop_fraction = (pre[key] - post[key]) / span
            if drop_fraction >= self.config.lifecycle.minimum_sensor_drop_fraction:
                improved += 1
            pre_severities.append(max(0.0, (pre[key] - sensor.healthy_baseline) / span))
            healthy_ceiling = sensor.healthy_baseline + (
                self.config.lifecycle.healthy_baseline_tolerance
                * (sensor.warning - sensor.healthy_baseline)
            )
            if post[key] > healthy_ceiling:
                close_to_healthy = False
            notes.append(f"{key} drop={drop_fraction:.3f}")
        pre_severity = max(pre_severities) if pre_severities else 0.0
        severity_drop = pre_severity - post_severity
        return improved, severity_drop, close_to_healthy, notes

    def _minimum_lifecycle_reached(self, timestamp: datetime) -> bool:
        if self.start_timestamp is None:
            return False
        hours = (timestamp - self.start_timestamp).total_seconds() / 3600.0
        return hours >= self.config.lifecycle.minimum_lifecycle_hours

    def _cooldown_reached(self, timestamp: datetime) -> bool:
        if self.last_reset_timestamp is None:
            return True
        hours = (timestamp - self.last_reset_timestamp).total_seconds() / 3600.0
        return hours >= self.config.lifecycle.reset_cooldown_hours

    def _minimum_critical_duration_reached(self, timestamp: datetime) -> bool:
        if self.first_critical_timestamp is None:
            return True
        minutes = (timestamp - self.first_critical_timestamp).total_seconds() / 60.0
        return minutes >= self.config.lifecycle.minimum_critical_duration_before_reset_minutes

    def _start_candidate(self, timestamp: datetime) -> None:
        pre_points = self._pre_points(timestamp)
        if not pre_points:
            return
        self.candidate_start = timestamp
        self.candidate_pre_values = self._window_medians(pre_points)
        self.candidate_post = []
        self.candidate_last_timestamp = timestamp
        self.state = "RESET_CANDIDATE"

    def _reject_candidate(self, reason: str) -> LifecycleSnapshot:
        self.candidate_start = None
        self.candidate_pre_values = {}
        self.candidate_post = []
        self.candidate_last_timestamp = None
        self.state = "DEGRADING"
        assert self.start_timestamp is not None
        return LifecycleSnapshot(
            self.lifecycle_id,
            self.state,
            0.0,
            reset_confidence="REJECTED",
            reset_reason=reason,
        )

    def _confirm_candidate(
        self,
        timestamp: datetime,
        confidence: str,
        reason: str,
    ) -> LifecycleSnapshot:
        assert self.start_timestamp is not None
        assert self.candidate_start is not None
        post_values = self._window_medians(self.candidate_post)
        duration_hours = max(
            0.0,
            (self.candidate_start - self.start_timestamp).total_seconds() / 3600.0,
        )
        record = LifecycleRecord(
            lifecycle_id=self.lifecycle_id,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.candidate_start,
            duration_hours=duration_hours,
            highest_status=self.highest_status.name,
            first_warning_timestamp=self.first_warning_timestamp,
            first_critical_timestamp=self.first_critical_timestamp,
            critical_reached=self.first_critical_timestamp is not None,
            inferred_reset_timestamp=timestamp,
            reset_confidence=confidence,
            reset_reason=reason,
            pre_reset_values=dict(self.candidate_pre_values),
            post_reset_values=post_values,
            first_raw_warning_timestamp=self.first_raw_warning_timestamp,
            first_confirmed_warning_timestamp=self.first_warning_timestamp,
            first_raw_critical_timestamp=self.first_raw_critical_timestamp,
            first_confirmed_critical_timestamp=self.first_critical_timestamp,
        )
        self.completed.append(record)
        new_start = self.candidate_start
        self.lifecycle_number += 1
        self.lifecycle_id = self._format_id(self.lifecycle_number)
        self.start_timestamp = new_start
        self.state = "HEALTHY"
        self.degraded_since = None
        self.highest_status = Status.NORMAL
        self.first_warning_timestamp = None
        self.first_critical_timestamp = None
        self.first_raw_warning_timestamp = None
        self.first_raw_critical_timestamp = None
        # Preserve only the confirmed healthy recovery for the new lifecycle.
        # Pre-reset history must not be used when the next reset candidate is
        # evaluated, otherwise the old degradation can leak across a boundary.
        self.history = deque(self.candidate_post)
        self.candidate_start = None
        self.candidate_pre_values = {}
        self.candidate_post = []
        self.candidate_last_timestamp = None
        self.last_reset_timestamp = timestamp
        elapsed = max(0.0, (timestamp - new_start).total_seconds() / 3600.0)
        return LifecycleSnapshot(
            self.lifecycle_id,
            self.state,
            elapsed,
            reset_confidence=confidence,
            reset_reason=reason,
            completed_lifecycle=record,
        )

    def update(
        self,
        *,
        timestamp: datetime,
        status: Status,
        event_status: Status | None = None,
        smoothed_values: dict[str, float],
        maximum_severity: float,
    ) -> LifecycleSnapshot:
        if self.start_timestamp is None:
            self.start_timestamp = timestamp

        self._trim_history(timestamp)
        observed_status = status if event_status is None else event_status
        if status >= Status.WARNING and self.first_raw_warning_timestamp is None:
            self.first_raw_warning_timestamp = timestamp
        if status >= Status.CRITICAL and self.first_raw_critical_timestamp is None:
            self.first_raw_critical_timestamp = timestamp
        self.highest_status = max(self.highest_status, observed_status)
        if observed_status >= Status.WARNING and self.first_warning_timestamp is None:
            self.first_warning_timestamp = timestamp
        if observed_status >= Status.CRITICAL and self.first_critical_timestamp is None:
            self.first_critical_timestamp = timestamp

        if self.state != "RESET_CANDIDATE":
            if status >= Status.WARNING:
                self.degraded_since = self.degraded_since or timestamp
                degraded_minutes = (
                    timestamp - self.degraded_since
                ).total_seconds() / 60.0
                if degraded_minutes >= self.config.lifecycle.minimum_degraded_minutes:
                    self.state = "DEGRADING"
            elif self.state == "HEALTHY":
                self.degraded_since = None

            if (
                self.state == "DEGRADING"
                and status == Status.NORMAL
                and self._minimum_lifecycle_reached(timestamp)
                and self._cooldown_reached(timestamp)
                and self._minimum_critical_duration_reached(timestamp)
            ):
                self._start_candidate(timestamp)

        if self.state == "RESET_CANDIDATE":
            assert self.candidate_start is not None
            if self.candidate_last_timestamp is not None:
                gap_minutes = (timestamp - self.candidate_last_timestamp).total_seconds() / 60.0
                if gap_minutes > self.config.lifecycle.maximum_confirmation_gap_minutes:
                    snapshot = self._reject_candidate(
                        "Reset confirmation was interrupted by a sampling gap larger than the configured maximum."
                    )
                    elapsed = (timestamp - self.start_timestamp).total_seconds() / 3600.0
                    return LifecycleSnapshot(snapshot.lifecycle_id, snapshot.lifecycle_state, max(0.0, elapsed), snapshot.reset_confidence, snapshot.reset_reason)
            self.candidate_last_timestamp = timestamp
            if status != Status.NORMAL:
                snapshot = self._reject_candidate(
                    "Normal recovery did not persist; degradation resumed before confirmation."
                )
                self.history.append((timestamp, dict(smoothed_values), maximum_severity))
                elapsed = (timestamp - self.start_timestamp).total_seconds() / 3600.0
                return LifecycleSnapshot(
                    snapshot.lifecycle_id,
                    snapshot.lifecycle_state,
                    max(0.0, elapsed),
                    snapshot.reset_confidence,
                    snapshot.reset_reason,
                )

            self.candidate_post.append((timestamp, dict(smoothed_values), maximum_severity))
            candidate_minutes = (timestamp - self.candidate_start).total_seconds() / 60.0
            high_confidence = self.first_critical_timestamp is not None
            required_minutes = min(
                self.config.lifecycle.normal_confirmation_minutes,
                self.config.lifecycle.normal_reset_confirmation_minutes,
            ) * (
                1.0 if high_confidence else self.config.lifecycle.medium_confirmation_multiplier
            )
            if candidate_minutes >= required_minutes:
                post_values = self._window_medians(self.candidate_post)
                post_severity = float(median([point[2] for point in self.candidate_post]))
                improved, severity_drop, close_to_healthy, notes = self._recovery_metrics(
                    self.candidate_pre_values,
                    post_values,
                    post_severity,
                )
                conditions_met = (
                    improved >= self.config.lifecycle.minimum_improved_sensor_count
                    and severity_drop >= self.config.lifecycle.minimum_overall_severity_drop
                    and close_to_healthy
                )
                if conditions_met:
                    confidence = "HIGH" if high_confidence else "MEDIUM"
                    reason = (
                        f"Sustained normal recovery after {'critical' if high_confidence else 'warning'} "
                        f"degradation; {improved} sensors improved, severity drop={severity_drop:.3f}; "
                        + ", ".join(notes)
                    )
                    snapshot = self._confirm_candidate(timestamp, confidence, reason)
                    return snapshot
                if candidate_minutes >= max(required_minutes * 2.0, required_minutes + 1.0):
                    snapshot = self._reject_candidate(
                        "Recovery persisted but did not show the configured multi-sensor step drop "
                        "and healthy-baseline return."
                    )
                    self.history.append((timestamp, dict(smoothed_values), maximum_severity))
                    elapsed = (timestamp - self.start_timestamp).total_seconds() / 3600.0
                    return LifecycleSnapshot(
                        snapshot.lifecycle_id,
                        snapshot.lifecycle_state,
                        max(0.0, elapsed),
                        snapshot.reset_confidence,
                        snapshot.reset_reason,
                    )

        self.history.append((timestamp, dict(smoothed_values), maximum_severity))
        elapsed = max(0.0, (timestamp - self.start_timestamp).total_seconds() / 3600.0)
        return LifecycleSnapshot(self.lifecycle_id, self.state, elapsed)
