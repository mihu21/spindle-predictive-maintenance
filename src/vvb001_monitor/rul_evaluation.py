from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import SensorConfig
from .features import FeatureEngine
from .models import SourceRecord, VVB001Reading
from .monitor import VVB001Monitor
from .predictor import VVB001Predictor
from .sensor_quality import SensorQualityGuard
from .validation import VVB001Validator


@dataclass(frozen=True)
class RULEvaluationCriteria:
    """Synthetic-only development gates for the online RUL estimator.

    These are intentionally not production acceptance limits.  They are sanity checks for whether
    the causal RUL output is directionally useful on simulator lifecycles with hidden future truth.
    """

    max_critical_mae_hours: float = 12.0
    max_critical_median_abs_error_hours: float = 10.0
    min_critical_estimate_availability: float = 0.95
    min_rul_monotonicity: float = 0.95
    min_interval_coverage: float = 0.80
    min_macro_lifecycle_interval_coverage: float = 0.75
    max_within_24h_mae_hours: float = 6.0
    max_within_12h_mae_hours: float = 5.0
    max_within_6h_mae_hours: float = 4.0
    max_premature_short_rul_rate: float = 0.10
    max_sensor_fault_large_rul_jump_rate: float = 0.25

    def validate(self) -> None:
        for name in (
            "max_critical_mae_hours",
            "max_critical_median_abs_error_hours",
            "max_within_24h_mae_hours",
            "max_within_12h_mae_hours",
            "max_within_6h_mae_hours",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "min_critical_estimate_availability",
            "min_rul_monotonicity",
            "min_interval_coverage",
            "min_macro_lifecycle_interval_coverage",
            "max_premature_short_rul_rate",
            "max_sensor_fault_large_rul_jump_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def _parse_dt(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _safe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _safe_rate(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def _safe_p90(values: list[float]) -> float | None:
    if not values:
        return None
    try:
        return float(np.quantile(np.asarray(values, dtype=float), 0.90, method="higher"))
    except TypeError:
        return float(np.quantile(np.asarray(values, dtype=float), 0.90, interpolation="higher"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(y) != len(x):
        return None
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(ya)):
        return None
    rx = _rankdata(xa)
    ry = _rankdata(ya)
    if float(np.std(rx)) <= 1e-12 or float(np.std(ry)) <= 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _load_truth(sensor_rows: list[dict[str, str]], truth_path: str | Path | None) -> dict[int, dict[str, Any]]:
    if truth_path is not None:
        truth: dict[int, dict[str, Any]] = {}
        with Path(truth_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "lifecycle_id", "true_state"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"RUL ground-truth CSV missing columns: {sorted(missing)}")
            for raw in reader:
                truth[int(raw["id"])] = {
                    "lifecycle_id": raw["lifecycle_id"],
                    "true_state": raw["true_state"].strip().upper(),
                    "true_damage": float(raw["true_damage"]) if raw.get("true_damage") not in {None, ""} else None,
                    "is_sensor_fault": str(raw.get("is_sensor_fault", "")).strip().lower() in {"1", "true", "yes", "y"},
                }
        return truth

    fields = set(sensor_rows[0]) if sensor_rows else set()
    if "latent_damage_score" not in fields and "true_state" not in fields:
        raise ValueError(
            "RUL evaluation requires hidden synthetic truth. Provide an input CSV containing latent_damage_score/true_state "
            "or pass --ground-truth with a separate hidden-truth CSV."
        )

    truth = {}
    for raw in sensor_rows:
        if raw.get("true_state"):
            state = raw["true_state"].strip().upper()
            damage = float(raw["true_damage"]) if raw.get("true_damage") not in {None, ""} else None
        else:
            damage = float(raw["latent_damage_score"])
            if damage >= 0.75:
                state = "CRITICAL"
            elif damage >= 0.35:
                state = "WARNING"
            else:
                state = "NORMAL"
        truth[int(raw["id"])] = {
            "lifecycle_id": raw["lifecycle_id"],
            "true_state": state,
            "true_damage": damage,
            "is_sensor_fault": str(raw.get("is_sensor_fault", "")).strip().lower() in {"1", "true", "yes", "y"},
        }
    return truth


def _summary_from_errors(errors: list[float]) -> dict[str, float | int | None]:
    abs_errors = [abs(x) for x in errors]
    return {
        "count": len(errors),
        "mae_hours": _safe_mean(abs_errors),
        "median_abs_error_hours": _safe_median(abs_errors),
        "mean_signed_error_hours": _safe_mean(errors),
        "rmse_hours": float(math.sqrt(statistics.mean([x * x for x in errors]))) if errors else None,
    }


def evaluate_rul(
    model_path: str | Path,
    input_path: str | Path,
    report_path: str | Path,
    predictions_path: str | Path,
    sensor_config: SensorConfig,
    *,
    ground_truth_path: str | Path | None = None,
    criteria: RULEvaluationCriteria | None = None,
) -> dict[str, Any]:
    """Replay synthetic lifecycles causally and compare online RUL against hidden future onset times."""

    model_path = Path(model_path)
    input_path = Path(input_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"RUL evaluation input not found: {input_path}")
    criteria = criteria or RULEvaluationCriteria()
    criteria.validate()
    model_sha256_before = _sha256_file(model_path)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        sensor_rows = list(csv.DictReader(handle))
    if not sensor_rows:
        raise ValueError("RUL evaluation input contains no rows")
    required = {"id", "timestamp", "line_sel", "machine_id", "vrms", "arms", "apeak", "crest", "temp", "lifecycle_id"}
    missing = required - set(sensor_rows[0])
    if missing:
        raise ValueError(f"RUL evaluation input missing columns: {sorted(missing)}")

    truth = _load_truth(sensor_rows, ground_truth_path)
    for raw in sensor_rows:
        if int(raw["id"]) not in truth:
            raise ValueError(f"Missing hidden truth for sensor row id={raw['id']}")

    # Future onset timestamps are computed only from hidden evaluator truth, never passed to monitor/model.
    lifecycle_timestamps: dict[str, list[tuple[datetime, str]]] = {}
    for raw in sensor_rows:
        row_id = int(raw["id"])
        lifecycle = raw["lifecycle_id"]
        lifecycle_timestamps.setdefault(lifecycle, []).append((_parse_dt(raw["timestamp"]), truth[row_id]["true_state"]))
    onset: dict[str, dict[str, datetime | None]] = {}
    for lifecycle, items in lifecycle_timestamps.items():
        items.sort(key=lambda x: x[0])
        warning = next((ts for ts, state in items if state in {"WARNING", "CRITICAL"}), None)
        critical = next((ts for ts, state in items if state == "CRITICAL"), None)
        onset[lifecycle] = {"WARNING": warning, "CRITICAL": critical}

    predictor = VVB001Predictor(model_path, sensor_config)
    monitor = VVB001Monitor(
        VVB001Validator(sensor_config),
        FeatureEngine(sensor_config),
        predictor,
        SensorQualityGuard(sensor_config),
    )

    previous_lifecycle_by_machine: dict[str, str] = {}
    out_rows: list[dict[str, Any]] = []

    for raw in sensor_rows:
        reading = VVB001Reading(
            source_id=int(raw["id"]),
            timestamp=_parse_dt(raw["timestamp"]),
            line_sel=raw["line_sel"],
            machine_id=raw["machine_id"],
            vrms=float(raw["vrms"]),
            arms=float(raw["arms"]),
            apeak=float(raw["apeak"]),
            crest=float(raw["crest"]),
            temp=float(raw["temp"]),
        )
        lifecycle = raw["lifecycle_id"]
        prior = previous_lifecycle_by_machine.get(reading.machine_key)
        if prior is not None and prior != lifecycle:
            monitor.reset_machine(reading.machine_key)
        previous_lifecycle_by_machine[reading.machine_key] = lifecycle

        inference_started = time.perf_counter()
        validation, item = monitor.process(SourceRecord(reading.source_id, dict(raw), reading))
        inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
        if item is None or item.predicted_status is None or item.degradation_score is None:
            raise RuntimeError(f"RUL evaluation row {reading.source_id} could not be scored: {validation.reasons}")

        hidden = truth[reading.source_id]
        warning_onset = onset[lifecycle]["WARNING"]
        critical_onset = onset[lifecycle]["CRITICAL"]
        true_warn = max(0.0, (warning_onset - reading.timestamp).total_seconds() / 3600.0) if warning_onset else None
        true_crit = max(0.0, (critical_onset - reading.timestamp).total_seconds() / 3600.0) if critical_onset else None
        pred_warn = item.estimated_hours_to_warning
        pred_crit = item.estimated_hours_to_critical

        out_rows.append({
            "id": reading.source_id,
            "timestamp": reading.timestamp.isoformat(),
            "line_sel": reading.line_sel,
            "machine_id": reading.machine_id,
            "lifecycle_id": lifecycle,
            "batch_id": raw.get("batch_id", ""),
            "development_role": raw.get("development_role", ""),
            "lifecycle_progress": raw.get("lifecycle_progress", ""),
            "true_state": hidden["true_state"],
            "true_damage": hidden.get("true_damage"),
            "is_sensor_fault": bool(hidden.get("is_sensor_fault")),
            "degradation_score": item.degradation_score,
            "predicted_status": item.predicted_status,
            "sensor_quality_status": item.sensor_quality_status,
            "prediction_state_held": item.prediction_state_held,
            "true_hours_to_warning": true_warn,
            "estimated_hours_to_warning": pred_warn,
            "warning_error_hours": (pred_warn - true_warn) if pred_warn is not None and true_warn is not None else None,
            "warning_lower_hours": item.warning_rul_lower_hours,
            "warning_upper_hours": item.warning_rul_upper_hours,
            "true_hours_to_critical": true_crit,
            "estimated_hours_to_critical": pred_crit,
            "critical_error_hours": (pred_crit - true_crit) if pred_crit is not None and true_crit is not None else None,
            "critical_lower_hours": item.critical_rul_lower_hours,
            "critical_upper_hours": item.critical_rul_upper_hours,
            "rul_reliability": item.rul_reliability,
            "rul_reason": item.rul_reason,
            "rul_state_source": item.rul_state_source,
            "rul_trend_score_per_hour": item.rul_trend_score_per_hour,
            "rul_trend_r2": item.rul_trend_r2,
            "rul_history_hours": item.rul_history_hours,
            "rul_trusted_points": item.rul_trusted_points,
            "rul_available": pred_crit is not None,
            "rul_method": item.rul_method,
            "rul_calibration_method": item.rul_calibration_method,
            "rul_calibration_bucket": item.rul_calibration_bucket,
            "rul_forecastability_state": item.rul_forecastability_state,
            "rul_forecastability_score": item.rul_forecastability_score,
            "rul_serviceable_intent": item.rul_serviceable_intent,
            "rul_hard_eligible": item.rul_hard_eligible,
            "rul_selector_active": item.rul_selector_active,
            "rul_withholding_reason_code": item.rul_withholding_reason_code,
            "rul_withholding_reasons": json.dumps(item.rul_withholding_reasons),
            "rul_support_distance": item.rul_support_distance,
            "rul_neighbor_dispersion_hours": item.rul_neighbor_dispersion_hours,
            "rul_model_disagreement_hours": item.rul_model_disagreement_hours,
            "warning_rul_forecastability_state": item.warning_rul_forecastability_state,
            "warning_rul_forecastability_score": item.warning_rul_forecastability_score,
            "warning_rul_serviceable_intent": item.warning_rul_serviceable_intent,
            "warning_rul_hard_eligible": item.warning_rul_hard_eligible,
            "warning_rul_selector_active": item.warning_rul_selector_active,
            "warning_rul_withholding_reason_code": item.warning_rul_withholding_reason_code,
            "warning_rul_withholding_reasons": json.dumps(item.warning_rul_withholding_reasons),
            "warning_rul_raw_point_hours": item.warning_rul_raw_point_hours,
            "warning_rul_corrected_point_hours": item.warning_rul_corrected_point_hours,
            "warning_rul_calibration_stratum": item.warning_rul_calibration_stratum,
            "critical_rul_forecastability_state": item.critical_rul_forecastability_state,
            "critical_rul_forecastability_score": item.critical_rul_forecastability_score,
            "critical_rul_serviceable_intent": item.critical_rul_serviceable_intent,
            "critical_rul_hard_eligible": item.critical_rul_hard_eligible,
            "critical_rul_selector_active": item.critical_rul_selector_active,
            "critical_rul_withholding_reason_code": item.critical_rul_withholding_reason_code,
            "critical_rul_withholding_reasons": json.dumps(item.critical_rul_withholding_reasons),
            "critical_rul_raw_point_hours": item.critical_rul_raw_point_hours,
            "critical_rul_corrected_point_hours": item.critical_rul_corrected_point_hours,
            "critical_rul_calibration_stratum": item.critical_rul_calibration_stratum,
            "rul_inference_latency_ms": inference_latency_ms,
        })

    critical_eligible = [r for r in out_rows if r["true_hours_to_critical"] is not None]
    pre_onset_eligible = [r for r in critical_eligible if float(r["true_hours_to_critical"]) > 0.0]
    already_critical_rows = [r for r in critical_eligible if float(r["true_hours_to_critical"]) == 0.0]
    warning_eligible = [r for r in out_rows if r["true_hours_to_warning"] is not None]

    def interval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        intervals = [
            row for row in rows
            if row["estimated_hours_to_critical"] is not None
            and row["critical_lower_hours"] is not None
            and row["critical_upper_hours"] is not None
        ]
        hits = [
            row for row in intervals
            if float(row["critical_lower_hours"])
            <= float(row["true_hours_to_critical"])
            <= float(row["critical_upper_hours"])
        ]
        widths = [float(row["critical_upper_hours"]) - float(row["critical_lower_hours"]) for row in intervals]
        lifecycle_conditional: list[float] = []
        lifecycle_effective: list[float] = []
        per_lifecycle_interval: dict[str, Any] = {}
        for lifecycle in sorted({str(row["lifecycle_id"]) for row in rows}):
            eligible = [row for row in rows if str(row["lifecycle_id"]) == lifecycle]
            local_intervals = [row for row in intervals if str(row["lifecycle_id"]) == lifecycle]
            local_hits = [row for row in hits if str(row["lifecycle_id"]) == lifecycle]
            conditional = _safe_rate(len(local_hits), len(local_intervals))
            effective = _safe_rate(len(local_hits), len(eligible))
            if conditional is not None:
                lifecycle_conditional.append(conditional)
            if effective is not None:
                lifecycle_effective.append(effective)
            per_lifecycle_interval[lifecycle] = {
                "eligible_rows": len(eligible),
                "interval_rows": len(local_intervals),
                "availability": _safe_rate(len(local_intervals), len(eligible)),
                "conditional_coverage": conditional,
                "effective_coverage": effective,
                "mean_width_hours": _safe_mean([
                    float(row["critical_upper_hours"]) - float(row["critical_lower_hours"])
                    for row in local_intervals
                ]),
            }
        return {
            "interval_rows": len(intervals),
            "interval_hits": len(hits),
            "interval_coverage": _safe_rate(len(hits), len(intervals)),
            "covered_eligible_rate": _safe_rate(len(hits), len(rows)),
            "mean_interval_width_hours": _safe_mean(widths),
            "median_interval_width_hours": _safe_median(widths),
            "p90_interval_width_hours": _safe_p90(widths),
            "evaluated_lifecycles": len(per_lifecycle_interval),
            # Effective coverage counts unavailable rows as misses and is the macro acceptance metric.
            "macro_lifecycle_coverage": _safe_mean(lifecycle_effective),
            "macro_lifecycle_conditional_coverage": _safe_mean(lifecycle_conditional),
            "median_lifecycle_coverage": _safe_median(lifecycle_effective),
            "worst_lifecycle_coverage": min(lifecycle_effective) if lifecycle_effective else None,
            "best_lifecycle_coverage": max(lifecycle_effective) if lifecycle_effective else None,
            "per_lifecycle": per_lifecycle_interval,
        }

    def critical_summary_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
        estimated = [row for row in rows if row["estimated_hours_to_critical"] is not None]
        errors = [float(row["critical_error_hours"]) for row in estimated]
        return {
            "eligible_rows": len(rows),
            "estimated_rows": len(estimated),
            "availability": _safe_rate(len(estimated), len(rows)),
            **_summary_from_errors(errors),
            **interval_summary(rows),
        }

    critical_summary = critical_summary_for(critical_eligible)
    pre_onset_summary = critical_summary_for(pre_onset_eligible)
    already_critical_summary = critical_summary_for(already_critical_rows)
    warning_estimated = [r for r in warning_eligible if r["estimated_hours_to_warning"] is not None]
    warning_errors = [float(r["warning_error_hours"]) for r in warning_estimated]

    horizon_metrics: dict[str, Any] = {}
    for horizon in (48.0, 24.0, 12.0, 6.0):
        rows = [r for r in pre_onset_eligible if float(r["true_hours_to_critical"]) <= horizon]
        horizon_metrics[f"within_{int(horizon)}h"] = critical_summary_for(rows)

    true_rul_regions = {
        "gt_72h": [r for r in pre_onset_eligible if float(r["true_hours_to_critical"]) > 72.0],
        "48_72h": [r for r in pre_onset_eligible if 48.0 < float(r["true_hours_to_critical"]) <= 72.0],
        "24_48h": [r for r in pre_onset_eligible if 24.0 < float(r["true_hours_to_critical"]) <= 48.0],
        "12_24h": [r for r in pre_onset_eligible if 12.0 < float(r["true_hours_to_critical"]) <= 24.0],
        "6_12h": [r for r in pre_onset_eligible if 6.0 < float(r["true_hours_to_critical"]) <= 12.0],
        "le_6h": [r for r in pre_onset_eligible if float(r["true_hours_to_critical"]) <= 6.0],
    }
    coverage_by_true_rul_region = {
        region: critical_summary_for(rows) for region, rows in true_rul_regions.items()
    }

    anchor_metrics: dict[str, Any] = {}
    for target in (48.0, 24.0, 12.0, 6.0):
        selected: list[dict[str, Any]] = []
        for lifecycle in sorted({str(r["lifecycle_id"]) for r in critical_eligible}):
            candidates = [
                r for r in critical_eligible
                if r["lifecycle_id"] == lifecycle and float(r["true_hours_to_critical"]) > 0.0
            ]
            if not candidates:
                continue
            selected.append(min(candidates, key=lambda r: abs(float(r["true_hours_to_critical"]) - target)))
        estimated = [r for r in selected if r["estimated_hours_to_critical"] is not None]
        errors = [float(r["critical_error_hours"]) for r in estimated]
        anchor_metrics[f"at_{int(target)}h"] = {
            "eligible_lifecycles": len(selected),
            "estimated_lifecycles": len(estimated),
            "availability": _safe_rate(len(estimated), len(selected)),
            **_summary_from_errors(errors),
        }

    monotonic_pairs = 0
    monotonic_good = 0
    per_lifecycle: dict[str, dict[str, Any]] = {}
    for lifecycle in sorted({str(r["lifecycle_id"]) for r in out_rows}):
        rows = [r for r in out_rows if r["lifecycle_id"] == lifecycle]
        estimates = [
            r for r in rows
            if r["estimated_hours_to_critical"] is not None
            and r["true_hours_to_critical"] is not None
            and float(r["true_hours_to_critical"]) > 0.0
        ]
        local_pairs = 0
        local_good = 0
        for prev, cur in zip(estimates, estimates[1:]):
            if float(cur["true_hours_to_critical"]) > float(prev["true_hours_to_critical"]):
                continue
            local_pairs += 1
            if float(cur["estimated_hours_to_critical"]) <= float(prev["estimated_hours_to_critical"]) + 1e-9:
                local_good += 1
        monotonic_pairs += local_pairs
        monotonic_good += local_good
        local_errors = [float(r["critical_error_hours"]) for r in estimates]
        per_lifecycle[lifecycle] = {
            "rows": len(rows),
            "critical_truth_available": onset[lifecycle]["CRITICAL"] is not None,
            "estimated_critical_rows": len(estimates),
            "critical_mae_hours": _safe_mean([abs(x) for x in local_errors]),
            "critical_median_abs_error_hours": _safe_median([abs(x) for x in local_errors]),
            "rul_monotonicity": _safe_rate(local_good, local_pairs),
            "pre_onset_interval": pre_onset_summary["per_lifecycle"].get(lifecycle),
        }

    pre_onset_summary["monotonicity"] = _safe_rate(monotonic_good, monotonic_pairs)
    critical_summary["monotonicity"] = pre_onset_summary["monotonicity"]
    critical_estimated = [r for r in critical_eligible if r["estimated_hours_to_critical"] is not None]

    premature_reference = [r for r in critical_eligible if float(r["true_hours_to_critical"]) > 24.0]
    premature_short = [
        r for r in premature_reference
        if r["estimated_hours_to_critical"] is not None and float(r["estimated_hours_to_critical"]) <= 6.0
    ]
    premature_rate = _safe_rate(len(premature_short), len(premature_reference))

    sensor_fault_rows = [r for r in out_rows if r["is_sensor_fault"]]
    large_sensor_fault_jumps = 0
    sensor_fault_comparable = 0
    previous_by_lifecycle: dict[str, dict[str, Any]] = {}
    for row in out_rows:
        lifecycle = str(row["lifecycle_id"])
        prev = previous_by_lifecycle.get(lifecycle)
        if row["is_sensor_fault"] and prev is not None:
            if row["estimated_hours_to_critical"] is not None and prev["estimated_hours_to_critical"] is not None:
                sensor_fault_comparable += 1
                if abs(float(row["estimated_hours_to_critical"]) - float(prev["estimated_hours_to_critical"])) > 6.0:
                    large_sensor_fault_jumps += 1
        previous_by_lifecycle[lifecycle] = row

    pre_onset_estimated = [r for r in pre_onset_eligible if r["estimated_hours_to_critical"] is not None]
    true_vals = [float(r["true_hours_to_critical"]) for r in pre_onset_estimated]
    pred_vals = [float(r["estimated_hours_to_critical"]) for r in pre_onset_estimated]
    pre_onset_summary["spearman_predicted_vs_true_rul"] = _spearman(pred_vals, true_vals)
    pre_onset_summary["premature_short_rul_rate"] = premature_rate
    critical_summary["spearman_predicted_vs_true_rul"] = pre_onset_summary["spearman_predicted_vs_true_rul"]
    critical_summary["premature_short_rul_rate"] = premature_rate
    warning_summary = {
        "eligible_rows": len(warning_eligible),
        "estimated_rows": len(warning_estimated),
        "availability": _safe_rate(len(warning_estimated), len(warning_eligible)),
        **_summary_from_errors(warning_errors),
    }
    sensor_fault_summary = {
        "annotated_sensor_fault_rows": len(sensor_fault_rows),
        "comparable_rows": sensor_fault_comparable,
        "large_rul_jump_threshold_hours": 6.0,
        "large_rul_jumps": large_sensor_fault_jumps,
        "large_rul_jump_rate": _safe_rate(large_sensor_fault_jumps, sensor_fault_comparable),
    }

    checks = {
        "critical_mae": pre_onset_summary["mae_hours"] is not None and float(pre_onset_summary["mae_hours"]) <= criteria.max_critical_mae_hours,
        "critical_median_abs_error": pre_onset_summary["median_abs_error_hours"] is not None and float(pre_onset_summary["median_abs_error_hours"]) <= criteria.max_critical_median_abs_error_hours,
        "critical_estimate_availability": pre_onset_summary["availability"] is not None and float(pre_onset_summary["availability"]) >= criteria.min_critical_estimate_availability,
        "rul_monotonicity": pre_onset_summary["monotonicity"] is not None and float(pre_onset_summary["monotonicity"]) >= criteria.min_rul_monotonicity,
        "interval_coverage": pre_onset_summary["interval_coverage"] is not None and float(pre_onset_summary["interval_coverage"]) >= criteria.min_interval_coverage,
        "macro_lifecycle_interval_coverage": pre_onset_summary["macro_lifecycle_coverage"] is not None and float(pre_onset_summary["macro_lifecycle_coverage"]) >= criteria.min_macro_lifecycle_interval_coverage,
        "within_24h_mae": horizon_metrics["within_24h"]["mae_hours"] is not None and float(horizon_metrics["within_24h"]["mae_hours"]) <= criteria.max_within_24h_mae_hours,
        "within_12h_mae": horizon_metrics["within_12h"]["mae_hours"] is not None and float(horizon_metrics["within_12h"]["mae_hours"]) <= criteria.max_within_12h_mae_hours,
        "within_6h_mae": horizon_metrics["within_6h"]["mae_hours"] is not None and float(horizon_metrics["within_6h"]["mae_hours"]) <= criteria.max_within_6h_mae_hours,
        "premature_short_rul_rate": pre_onset_summary["premature_short_rul_rate"] is not None and float(pre_onset_summary["premature_short_rul_rate"]) <= criteria.max_premature_short_rul_rate,
    }
    if sensor_fault_comparable:
        checks["sensor_fault_large_rul_jump_rate"] = (
            sensor_fault_summary["large_rul_jump_rate"] is not None
            and float(sensor_fault_summary["large_rul_jump_rate"]) <= criteria.max_sensor_fault_large_rul_jump_rate
        )

    width_warning_threshold = max(48.0, 2.0 * (_safe_median(true_vals) or 0.0))
    width_warning = bool(
        pre_onset_summary["interval_coverage"] is not None
        and float(pre_onset_summary["interval_coverage"]) >= criteria.min_interval_coverage
        and pre_onset_summary["mean_interval_width_hours"] is not None
        and float(pre_onset_summary["mean_interval_width_hours"]) > width_warning_threshold
    )

    learned_artifact = getattr(predictor, "rul_model_artifact", None)
    learned_rul = bool(learned_artifact)
    artifact_version = str((learned_artifact or {}).get("version") or "")
    rul_method = (
        "supervised_learned_rul_v2_7_identity_first_critical"
        if artifact_version == "supervised_quantile_rul_v2_7_identity_first_critical"
        else
        "supervised_learned_rul_v2_6_target_specific_corrected"
        if artifact_version == "supervised_quantile_rul_v2_6_target_specific_corrected"
        else "supervised_learned_rul_v2_5_active_population_calibrated"
        if artifact_version == "supervised_quantile_rul_v2_5_active_population_calibrated"
        else "supervised_learned_rul_v2_4_state_selective"
        if artifact_version == "supervised_quantile_rul_v2_4_state_selective"
        else "supervised_learned_rul_v2_3_forecastability_aware"
        if artifact_version == "supervised_quantile_rul_v2_3_forecastability_aware"
        else "supervised_learned_rul_v2_2"
        if artifact_version == "supervised_quantile_rul_v2_2"
        else "supervised_learned_rul_v2_1"
        if artifact_version == "supervised_quantile_rul_v2_1"
        else "supervised_learned_rul_v2"
        if learned_rul
        else "legacy_causal_trend_extrapolation"
    )
    model_sha256_after = _sha256_file(model_path)
    if model_sha256_after != model_sha256_before:
        raise RuntimeError("evaluate-rul mutated the serialized model artifact")
    report = {
        "evaluation_type": "synthetic_rul_development_evaluation",
        "bootstrap_only": True,
        "rul_method": rul_method,
        "rul_artifact_version": artifact_version or None,
        "model_path": str(model_path.resolve()),
        "input_path": str(input_path.resolve()),
        "ground_truth_path": str(Path(ground_truth_path).resolve()) if ground_truth_path is not None else None,
        "truth_definition": {
            "warning": "first hidden true_state WARNING/CRITICAL; embedded latent_damage_score uses >=0.35",
            "critical": "first hidden true_state CRITICAL; embedded latent_damage_score uses >=0.75",
            "leakage_contract": "hidden future truth is used only after runtime inference for evaluation and is never passed to feature engineering, sensor quality, predictor, or RUL estimator",
            "acceptance_population": "pre-onset rows with true hours to CRITICAL > 0; already-CRITICAL rows are reported separately",
        },
        "evaluation_isolation": {
            "model_sha256_before": model_sha256_before,
            "model_sha256_after": model_sha256_after,
            "model_mutated": False,
            "recalibration_performed": False,
        },
        "criteria": asdict(criteria),
        "critical_rul": critical_summary,
        "pre_onset_critical_rul": pre_onset_summary,
        "already_critical_rul": already_critical_summary,
        "warning_rul": warning_summary,
        "horizon_metrics": horizon_metrics,
        "coverage_by_true_rul_region": coverage_by_true_rul_region,
        "anchor_metrics": anchor_metrics,
        "sensor_fault_stability": sensor_fault_summary,
        "per_lifecycle": per_lifecycle,
        "interval_width_safeguard": {
            "warning": width_warning,
            "diagnostic_only": True,
            "mean_width_warning_threshold_hours": width_warning_threshold,
            "reason": "high_coverage_with_excessive_mean_width" if width_warning else None,
        },
        "checks": checks,
        "overall_pass": bool(all(checks.values())),
        "limitations": [
            "Synthetic hidden-damage time is not a plant-validated failure or maintenance timestamp.",
            (
                "The learned RUL model is supervised only by synthetic completed-lifecycle onset times and does not establish physical component survival time."
                if learned_rul
                else "The legacy evaluator measures trend extrapolation to the learned CRITICAL regime, not physical bearing/component survival time."
            ),
            "Production RUL accuracy must later be measured against real maintenance/failure events.",
        ],
    }

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    return report
