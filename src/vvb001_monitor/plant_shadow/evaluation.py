from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable

from .contracts import SUPPORT_POLICY_VERSION, TargetName


@dataclass(frozen=True)
class SupportPolicy:
    version: str = SUPPORT_POLICY_VERSION
    min_lifecycles: int = 5
    min_serviceable_predictions: int = 100
    min_intervals: int = 100

    def __post_init__(self) -> None:
        if min(self.min_lifecycles, self.min_serviceable_predictions, self.min_intervals) < 1:
            raise ValueError("support thresholds must be positive and predeclared")


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_target(
    connection: sqlite3.Connection,
    target: TargetName,
    *,
    policy: SupportPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or SupportPolicy()
    connection.row_factory = sqlite3.Row
    prefix = "warning" if target == TargetName.WARNING else "critical"
    rows = connection.execute(
        f"""
        SELECT t.true_hours,p.prediction_id,p.lifecycle_id,p.machine_uid,
               p.{prefix}_point_hours AS point_hours,
               p.{prefix}_lower_hours AS lower_hours,
               p.{prefix}_upper_hours AS upper_hours,
               p.{prefix}_serviceable AS serviceable,
               r.source_key
        FROM target_truth t
        JOIN prediction_attempts p ON p.prediction_id=t.prediction_id
        JOIN raw_observations r ON r.ingestion_id=p.ingestion_id
        JOIN lifecycle_records l ON l.lifecycle_id=p.lifecycle_id
        WHERE t.target=?
          AND l.status != 'CLOSED_INFERRED'
          AND NOT EXISTS (
              SELECT 1 FROM prediction_supersessions s
              WHERE s.original_prediction_id=p.prediction_id
          )
        ORDER BY p.lifecycle_id,r.event_timestamp
        """,
        (target,),
    ).fetchall()
    lifecycle_ids = {str(row["lifecycle_id"]) for row in rows}
    serviceable = [
        row for row in rows
        if bool(row["serviceable"]) and _finite(row["point_hours"])
    ]
    intervals = [
        row for row in serviceable
        if _finite(row["lower_hours"]) and _finite(row["upper_hours"])
    ]
    counts = {
        "truth_predictions": len(rows),
        "serviceable_predictions": len(serviceable),
        "intervals": len(intervals),
        "lifecycles": len(lifecycle_ids),
        "machines": len({str(row["machine_uid"]) for row in rows}),
        "sources": len({str(row["source_key"]) for row in rows}),
    }
    support = {
        "policy_version": policy.version,
        "min_lifecycles": policy.min_lifecycles,
        "min_serviceable_predictions": policy.min_serviceable_predictions,
        "min_intervals": policy.min_intervals,
    }
    point_supported = (
        counts["lifecycles"] >= policy.min_lifecycles
        and counts["serviceable_predictions"] >= policy.min_serviceable_predictions
    )
    interval_supported = point_supported and counts["intervals"] >= policy.min_intervals
    result: dict[str, Any] = {
        "target": target,
        "status": "SUPPORTED" if point_supported else "INSUFFICIENT_EVIDENCE",
        "counts": counts,
        "support": support,
        "availability": len(serviceable) / len(rows) if rows else None,
        "point_metrics": None,
        "interval_metrics": None,
    }
    if not point_supported:
        return result

    errors = [float(row["point_hours"]) - float(row["true_hours"]) for row in serviceable]
    absolute = [abs(value) for value in errors]
    by_lifecycle: dict[str, list[float]] = {}
    for row, error in zip(serviceable, absolute):
        by_lifecycle.setdefault(str(row["lifecycle_id"]), []).append(error)
    result["point_metrics"] = {
        "mae_hours": _mean(absolute),
        "median_absolute_error_hours": median(absolute),
        "signed_bias_hours": _mean(errors),
        "lifecycle_macro_mae_hours": _mean(_mean(values) for values in by_lifecycle.values()),
    }
    if interval_supported:
        covered = [
            float(row["lower_hours"]) <= float(row["true_hours"]) <= float(row["upper_hours"])
            for row in intervals
        ]
        widths = [float(row["upper_hours"]) - float(row["lower_hours"]) for row in intervals]
        lifecycle_coverage: dict[str, list[bool]] = {}
        for row, value in zip(intervals, covered):
            lifecycle_coverage.setdefault(str(row["lifecycle_id"]), []).append(value)
        result["interval_metrics"] = {
            "coverage": sum(covered) / len(covered),
            "lifecycle_macro_coverage": _mean(sum(values) / len(values) for values in lifecycle_coverage.values()),
            "mean_width_hours": _mean(widths),
            "median_width_hours": median(widths),
        }
    else:
        result["interval_status"] = "INSUFFICIENT_EVIDENCE"
    return result


def evaluate_plant(connection: sqlite3.Connection, *, policy: SupportPolicy | None = None) -> dict[str, Any]:
    return {
        "validation_domain": "plant_shadow",
        "plant_production_authorized": False,
        "warning": evaluate_target(connection, TargetName.WARNING, policy=policy),
        "critical": evaluate_target(connection, TargetName.CRITICAL, policy=policy),
    }
