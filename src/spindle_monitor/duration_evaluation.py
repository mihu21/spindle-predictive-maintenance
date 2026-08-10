from __future__ import annotations

from typing import Any, Iterable

import numpy as np


LATE_TOLERANCES_MINUTES = (5, 15, 30, 60, 180)


def _bucket_name(duration: float, boundaries: tuple[float, ...]) -> str:
    if len(boundaries) == 3:
        first, second, third = boundaries
        if duration < first:
            return "short"
        if duration < second:
            return "medium"
        if duration < third:
            return "long"
        return "very_long"
    lower = 0.0
    for upper in boundaries:
        if duration < upper:
            return f"[{lower:g},{upper:g})h"
        lower = upper
    return f"[{lower:g},inf)h"


def _metrics(
    actual: np.ndarray, predicted: np.ndarray, lifecycle_ids: list[str], *, include_per_lifecycle: bool = True
) -> dict[str, Any]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    coverage = float(np.mean(valid)) if len(valid) else 0.0
    actual, predicted = actual[valid], predicted[valid]
    ids = [value for value, keep in zip(lifecycle_ids, valid) if keep]
    if not len(actual):
        return {"sample_count": 0, "forecast_coverage": 0.0}
    error = predicted - actual
    absolute = np.abs(error)
    late = np.maximum(error, 0.0)
    result: dict[str, Any] = {
        "lifecycle_count": len(set(ids)), "sample_count": len(actual), "forecast_coverage": coverage,
        "mae_hours": float(np.mean(absolute)), "median_absolute_error_hours": float(np.median(absolute)),
        "bias_hours": float(np.mean(error)), "p90_absolute_error_hours": float(np.quantile(absolute, .9)),
        "maximum_absolute_error_hours": float(np.max(absolute)), "p90_late_error_hours": float(np.quantile(late, .9)),
        "maximum_late_error_hours": float(np.max(late)), "early_prediction_rate": float(np.mean(error < 0)),
        "out_of_distribution_rate": None, "beyond_target_support_rate": None,
        "withholding_rate": float(1.0 - coverage), "low_confidence_rate": None,
    }
    result["late_by_more_than"] = {
        f"{minutes}_minutes": float(np.mean(error > minutes / 60.0))
        for minutes in LATE_TOLERANCES_MINUTES
    }
    if include_per_lifecycle:
        result["per_lifecycle"] = {
            lifecycle_id: _metrics(
                actual[np.asarray(ids) == lifecycle_id], predicted[np.asarray(ids) == lifecycle_id],
                [lifecycle_id] * int(np.sum(np.asarray(ids) == lifecycle_id)), include_per_lifecycle=False,
            )
            for lifecycle_id in sorted(set(ids))
        }
    return result


def duration_bucket_evaluation(
    actual: Iterable[float], predicted: Iterable[float], lifecycle_ids: list[str],
    lifecycle_durations: dict[str, float], boundaries: tuple[float, ...] = (72.0, 168.0, 336.0),
) -> dict[str, Any]:
    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    buckets: dict[str, Any] = {}
    names = (
        ("short", "medium", "long", "very_long")
        if len(boundaries) == 3 else tuple(
            [f"[{0 if index == 0 else boundaries[index - 1]:g},{upper:g})h" for index, upper in enumerate(boundaries)]
            + [f"[{boundaries[-1]:g},inf)h"]
        )
    )
    for name in names:
        mask = np.asarray([
            _bucket_name(float(lifecycle_durations.get(value, 0.0)), boundaries) == name
            for value in lifecycle_ids
        ])
        buckets[name] = _metrics(
            actual_array[mask], predicted_array[mask],
            [value for value, keep in zip(lifecycle_ids, mask) if keep],
        )
    return {"bucket_boundaries_hours": list(boundaries), "buckets": buckets}
