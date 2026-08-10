from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.config import load_config
from spindle_monitor.io import read_csv_readings
from spindle_monitor.monitor import ConditionMonitor
from spindle_monitor.storage import _best_critical_forecast

DATASET = ROOT / "data" / "spindle_predictive_maintenance_10000_unlabeled.csv"
PREVIOUS_AI = ROOT / "comparison" / "previous_result_ai.csv"
PREVIOUS_REBUILT = ROOT / "comparison" / "previous_rebuilt_replay_results.csv"
REPORT = ROOT / "output" / "validation_report.txt"
STATUSES = ("NORMAL", "WARNING", "CRITICAL")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _normalise(value: str | None) -> str:
    return (value or "").strip().upper()


def _accuracy(expected: list[str], predicted: list[str]) -> float:
    return sum(a == b for a, b in zip(expected, predicted)) / len(expected)


def _confusion(expected: list[str], predicted: list[str]) -> dict[str, dict[str, int]]:
    table = {actual: {guess: 0 for guess in STATUSES} for actual in STATUSES}
    for actual, guess in zip(expected, predicted):
        table[actual][guess] += 1
    return table


def _format_confusion(table: dict[str, dict[str, int]]) -> list[str]:
    lines = ["actual\\predicted  NORMAL  WARNING  CRITICAL"]
    for actual in STATUSES:
        row = table[actual]
        lines.append(
            f"{actual:<16} {row['NORMAL']:>6}  {row['WARNING']:>7}  {row['CRITICAL']:>8}"
        )
    return lines


def _summary(errors: list[float]) -> dict[str, float]:
    absolute = sorted(abs(value) for value in errors)
    return {
        "count": float(len(errors)),
        "mae": statistics.fmean(absolute),
        "median": statistics.median(absolute),
        "p90": absolute[int(0.90 * (len(absolute) - 1))],
        "bias": statistics.fmean(errors),
    }


def main() -> None:
    config = load_config(ROOT / "config" / "thresholds.json")
    data = list(read_csv_readings(DATASET))
    readings = [item[0] for item in data]
    expected = [_normalise(item[1]) for item in data]
    times = [reading.timestamp for reading in readings]

    first_critical_index = expected.index("CRITICAL")
    first_critical_time = times[first_critical_index]

    monitor = ConditionMonitor(config)
    predicted_statuses: list[str] = []
    new_forecast_errors: list[float] = []
    forecast_count_after_warmup = 0

    for index, reading in enumerate(readings):
        result = monitor.process(reading)
        predicted_statuses.append(result.raw_status.name)

        if index >= first_critical_index:
            continue
        critical = _best_critical_forecast(result)
        if critical is None:
            continue
        forecast = critical[1]
        if forecast.critical_eta_seconds is None:
            continue

        predicted_hours = forecast.critical_eta_seconds / 3600.0
        actual_hours = (first_critical_time - reading.timestamp).total_seconds() / 3600.0
        new_forecast_errors.append(predicted_hours - actual_hours)
        forecast_count_after_warmup += 1


    previous_ai_rows = _rows(PREVIOUS_AI)
    previous_ai_statuses = [_normalise(row.get("status_ai")) for row in previous_ai_rows]

    previous_rebuilt_rows = _rows(PREVIOUS_REBUILT)
    previous_rebuilt_errors: list[float] = []
    for index, row in enumerate(previous_rebuilt_rows[:first_critical_index]):
        value = (row.get("time_to_maintenance_hours") or "").strip()
        if not value:
            continue
        predicted_hours = float(value)
        actual_hours = (first_critical_time - times[index]).total_seconds() / 3600.0
        previous_rebuilt_errors.append(predicted_hours - actual_hours)

    previous_ai_lifetime_errors_days: list[float] = []
    for index, row in enumerate(previous_ai_rows[:first_critical_index]):
        predicted_days = float(row["remaining_days"])
        actual_days = (
            first_critical_time - times[index]
        ).total_seconds() / 86400.0
        previous_ai_lifetime_errors_days.append(predicted_days - actual_days)

    current_confusion = _confusion(expected, predicted_statuses)
    old_confusion = _confusion(expected, previous_ai_statuses)
    new_summary = _summary(new_forecast_errors)
    previous_rebuilt_summary = _summary(previous_rebuilt_errors)
    old_lifetime_summary = _summary(previous_ai_lifetime_errors_days)

    baseline_warmup_points = max(3, min(config.monitoring.minimum_forecast_points, 30))
    warmup_eligible = max(first_critical_index - baseline_warmup_points + 1, 0)
    coverage = (
        len(new_forecast_errors) / warmup_eligible * 100.0
        if warmup_eligible
        else 0.0
    )
    previous_coverage = (
        len(previous_rebuilt_errors) / first_critical_index * 100.0
        if first_critical_index
        else 0.0
    )
    label_changes = sum(a != b for a, b in zip(expected, expected[1:]))

    lines = [
        "CORRECTED SPINDLE MONITOR VALIDATION",
        "=" * 52,
        f"Dataset rows: {len(expected)}",
        f"Label counts: {dict(Counter(expected))}",
        f"First critical-labelled reading: row {first_critical_index + 1} at {first_critical_time.isoformat()}",
        f"Label changes in one seven-day trajectory: {label_changes}",
        "",
        "1. CURRENT STATUS",
        f"Direct manufacturer-rule accuracy: {_accuracy(expected, predicted_statuses) * 100:.2f}%",
        f"Previous Autoencoder + K-means accuracy: {_accuracy(expected, previous_ai_statuses) * 100:.2f}%",
        "",
        "Current rule confusion matrix:",
        *_format_confusion(current_confusion),
        "",
        "Previous AI confusion matrix:",
        *_format_confusion(old_confusion),
        "",
        "2. TIME-TO-CRITICAL STATISTICAL BASELINE",
        "Validation target: time remaining until the first critical-labelled reading.",
        f"Kalman level/rate forecast count: {int(new_summary['count'])}",
        f"Kalman forecast coverage after warm-up: {coverage:.2f}%",
        f"Kalman forecast MAE: {new_summary['mae']:.2f} hours",
        f"Kalman forecast median absolute error: {new_summary['median']:.2f} hours",
        f"Kalman forecast 90th-percentile absolute error: {new_summary['p90']:.2f} hours",
        f"Kalman forecast mean bias: {new_summary['bias']:+.2f} hours",
        "",
        f"Previous rebuilt deterministic forecast count: {int(previous_rebuilt_summary['count'])}",
        f"Previous rebuilt forecast coverage: {previous_coverage:.2f}%",
        f"Previous rebuilt forecast MAE: {previous_rebuilt_summary['mae']:.2f} hours",
        f"Previous rebuilt forecast median absolute error: {previous_rebuilt_summary['median']:.2f} hours",
        f"Previous rebuilt forecast 90th-percentile absolute error: {previous_rebuilt_summary['p90']:.2f} hours",
        f"Previous rebuilt forecast mean bias: {previous_rebuilt_summary['bias']:+.2f} hours",
        "",
        "3. ORIGINAL PROJECT REMAINING DAYS",
        f"Original remaining-days MAE versus first critical event: {old_lifetime_summary['mae']:.2f} days",
        f"Original remaining-days median absolute error: {old_lifetime_summary['median']:.2f} days",
        f"Original remaining-days mean bias: {old_lifetime_summary['bias']:+.2f} days",
        "",
        "4. IMPORTANT LIMITATIONS",
        "- The status result is strongly validated because the labels are generated from manufacturer thresholds.",
        "- The forecast is only a backtest on one supplied degradation trajectory; it is not proof of accuracy on a real machine.",
        "- The data contains hundreds of label switches caused by noisy one-reading threshold crossings.",
        "- The current statistical baseline extrapolates each per-sensor Kalman level and rate to its critical threshold.",
        "- ML metrics are intentionally unavailable because this dataset contains no completed reset lifecycle.",
        "- A true component lifetime model still requires multiple real maintenance cycles and confirmed inspection outcomes.",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nValidation report written to: {REPORT}")


if __name__ == "__main__":
    main()
