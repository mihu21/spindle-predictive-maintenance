from __future__ import annotations

from .models import MonitorResult


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unavailable"
    if seconds <= 0:
        return "now"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def best_forecast_text(result: MonitorResult) -> str:
    available = []
    for assessment in result.assessments.values():
        forecast = assessment.forecast
        if forecast and forecast.eta_seconds is not None:
            available.append((forecast.eta_seconds, assessment.display_name, forecast))
    if not available:
        worst = result.assessments[result.worst_sensor].forecast
        reason = worst.confidence if worst else "unavailable"
        return f"forecast={reason}"
    _, name, forecast = min(available, key=lambda item: item[0])
    if forecast.eta_seconds == 0:
        return f"forecast={name}->{forecast.target_status} threshold reached"
    return (
        f"forecast={name}->{forecast.target_status} in "
        f"{format_duration(forecast.eta_seconds)} ({forecast.confidence})"
    )


def print_result(result: MonitorResult) -> None:
    values = " ".join(
        f"{assessment.display_name.lower()}={assessment.raw_value:.2f}{assessment.unit}"
        for assessment in result.assessments.values()
    )
    print(
        f"{result.timestamp.isoformat(sep=' ')} | raw={result.raw_status.name:<8} "
        f"effective={result.effective_status.name:<8} health={result.health_percent:6.2f}% "
        f"worst={result.worst_sensor_display:<11} {values} | {best_forecast_text(result)}"
    )
