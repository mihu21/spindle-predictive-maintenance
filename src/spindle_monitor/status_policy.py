from __future__ import annotations

from collections import deque
from datetime import datetime

from .config import LifecycleConfig, MonitoringConfig
from .models import Status


class StabilizedStatusPolicy:
    """Legacy operational stabilization retained for accelerated-mock compatibility."""

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.history: deque[Status] = deque(maxlen=config.warning_vote_window)
        self.current: Status | None = None
        self.normal_streak = 0
        self.noncritical_streak = 0

    def update(self, raw_status: Status) -> Status:
        self.history.append(raw_status)
        if self.current is None:
            self.current = raw_status
            return self.current
        if raw_status == Status.CRITICAL:
            self.current = Status.CRITICAL
            self.noncritical_streak = self.normal_streak = 0
            return self.current
        self.noncritical_streak += 1
        self.normal_streak = self.normal_streak + 1 if raw_status == Status.NORMAL else 0
        if self.current == Status.CRITICAL:
            if self.noncritical_streak < self.config.critical_recovery_readings:
                return self.current
            self.current = Status.WARNING if raw_status == Status.WARNING else Status.NORMAL
            self.noncritical_streak = 0
            return self.current
        warning_votes = sum(status >= Status.WARNING for status in self.history)
        if self.current == Status.NORMAL:
            if warning_votes >= self.config.warning_votes_required:
                self.current = Status.WARNING
            return self.current
        if self.current == Status.WARNING and self.normal_streak >= self.config.normal_recovery_readings:
            self.current = Status.NORMAL
        return self.current

    def reset(self, raw_status: Status | None = None) -> None:
        self.history.clear()
        self.current = raw_status
        self.normal_streak = self.noncritical_streak = 0


class TimeBasedEventPolicy:
    """Causally confirm lifecycle events without delaying raw safety protection.

    This policy is intentionally not the manufacturer protection policy.  A raw
    critical reading remains available to the monitor immediately while this
    class separately records when warning/critical persistence is long enough
    to count as a confirmed lifecycle event.
    """

    def __init__(self, config: LifecycleConfig) -> None:
        self.config = config
        self.warning_since: datetime | None = None
        self.critical_since: datetime | None = None
        self.last_timestamp: datetime | None = None
        self.current = Status.NORMAL

    def update(self, timestamp: datetime, raw_status: Status) -> Status:
        if self.last_timestamp is not None:
            gap = (timestamp - self.last_timestamp).total_seconds() / 60.0
            if gap <= 0:
                raise ValueError("Event timestamps must be strictly increasing")
            if gap > self.config.maximum_confirmation_gap_minutes:
                self.warning_since = None
                self.critical_since = None
        self.last_timestamp = timestamp

        if raw_status >= Status.WARNING:
            self.warning_since = self.warning_since or timestamp
        else:
            self.warning_since = None

        if raw_status >= Status.CRITICAL:
            self.critical_since = self.critical_since or timestamp
        else:
            self.critical_since = None

        warning_confirmed = bool(
            self.warning_since is not None
            and (timestamp - self.warning_since).total_seconds() / 60.0
            >= self.config.warning_confirmation_minutes
        )
        critical_confirmed = bool(
            self.critical_since is not None
            and (timestamp - self.critical_since).total_seconds() / 60.0
            >= self.config.critical_confirmation_minutes
        )
        self.current = (
            Status.CRITICAL if critical_confirmed
            else Status.WARNING if warning_confirmed
            else Status.NORMAL
        )
        return self.current

    def reset(self) -> None:
        self.warning_since = self.critical_since = self.last_timestamp = None
        self.current = Status.NORMAL
