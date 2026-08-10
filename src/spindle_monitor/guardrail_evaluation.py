from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .environment import environment_compatibility
from .model_registry import ModelRegistry
from .retraining import LifecycleExamples, build_training_examples, split_lifecycles


@dataclass(frozen=True)
class GuardrailThresholds:
    warning_absolute_hours: float
    critical_absolute_hours: float
    relative: float

    def to_dict(self) -> dict[str, float]:
        return {
            "warning_disagreement_absolute_hours": self.warning_absolute_hours,
            "critical_disagreement_absolute_hours": self.critical_absolute_hours,
            "disagreement_relative_threshold": self.relative,
        }


def _outside_distribution(
    row: np.ndarray, metadata: dict[str, Any], config: ProjectConfig
) -> tuple[bool, float]:
    distribution = metadata.get("feature_distribution", {})
    lower = distribution.get("lower_quantile_01")
    upper = distribution.get("upper_quantile_99")
    if not isinstance(lower, list) or not isinstance(upper, list):
        return False, 0.0
    if len(lower) != len(row) or len(upper) != len(row):
        return True, 1.0
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    span = np.maximum(upper_array - lower_array, 1e-9)
    margin = config.ml.feature_distribution_margin_fraction * span
    outside = (row < lower_array - margin) | (row > upper_array + margin)
    fraction = float(np.mean(outside))
    return fraction > config.ml.feature_distribution_max_outside_fraction, fraction


def _records(
    examples: list[LifecycleExamples],
    models: dict[str, Any],
    metadata: dict[str, Any],
    config: ProjectConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in examples:
        feature_matrix = np.asarray(example.feature_rows, dtype=float)
        predictions = {
            target: np.asarray(model.predict(feature_matrix), dtype=float)
            for target, model in models.items()
        }
        for index, timestamp in enumerate(example.timestamps):
            feature_row = feature_matrix[index]
            ood, outside_fraction = _outside_distribution(feature_row, metadata, config)
            record: dict[str, Any] = {
                "lifecycle_id": example.lifecycle_id,
                "timestamp": timestamp.isoformat(),
                "out_of_distribution": ood,
                "outside_feature_fraction": outside_fraction,
                "lifecycle_duration_hours": example.duration_hours,
            }
            for target in ("warning", "critical"):
                event = getattr(example, f"first_{target}_timestamp")
                actual = (
                    (event - timestamp).total_seconds() / 3600.0
                    if event is not None and timestamp < event
                    else None
                )
                baseline = getattr(example, f"baseline_{target}")[index]
                prediction = predictions.get(target)
                ml = None if prediction is None else float(prediction[index])
                support = metadata.get("target_support", {}).get(f"time_to_{target}", {})
                minimum = support.get("minimum")
                maximum = support.get("maximum")
                beyond = bool(ml is not None and (
                    (minimum is not None and ml < float(minimum))
                    or (maximum is not None and ml > float(maximum))
                ))
                record[target] = {
                    "actual": actual, "ml": ml, "statistical": baseline,
                    "beyond_target_support": beyond,
                    "physically_invalid": bool(ml is not None and ml < 0.0),
                }
            records.append(record)
    return records


def _target_choice(
    values: dict[str, float | None],
    target: str,
    out_of_distribution: bool,
    thresholds: GuardrailThresholds,
) -> dict[str, Any]:
    ml = values["ml"]
    statistical = values["statistical"]
    if values.get("beyond_target_support", False):
        return {
            "primary": None,
            "source": "unavailable",
            "strong": False,
            "withheld": True,
            "absolute": None,
            "relative": None,
            "reason": "target_support_violation",
        }
    if out_of_distribution:
        return {
            "primary": None,
            "source": "unavailable",
            "strong": False,
            "withheld": True,
            "absolute": None,
            "relative": None,
        }
    if ml is None:
        return {
            "primary": None,
            "source": "statistical_monitoring" if statistical is not None else "unavailable",
            "strong": False,
            "withheld": True,
            "absolute": None,
            "relative": None,
        }
    absolute = relative = None
    strong = False
    if statistical is not None:
        absolute = abs(ml - statistical)
        relative = absolute / max(abs(ml), abs(statistical), 1e-6)
        absolute_threshold = (
            thresholds.warning_absolute_hours
            if target == "warning"
            else thresholds.critical_absolute_hours
        )
        strong = absolute > absolute_threshold and relative > thresholds.relative
    return {
        "primary": None if strong else ml,
        "source": "unavailable" if strong else "ml",
        "strong": strong,
        "withheld": strong,
        "absolute": absolute,
        "relative": relative,
    }


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _target_metrics(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    eligible = [row for row in rows if row[target]["actual"] is not None]
    covered = [row for row in eligible if row[f"{target}_decision"]["primary"] is not None]
    errors = np.asarray([
        row[f"{target}_decision"]["primary"] - row[target]["actual"] for row in covered
    ], dtype=float)
    strong_count = sum(row[f"{target}_decision"]["strong"] for row in eligible)
    withheld_count = len(eligible) - len(covered)
    unsafe_count = int(np.sum(errors > 0.0)) if len(errors) else 0
    unflagged_unsafe = sum(
        row[f"{target}_decision"]["primary"] is not None
        and row[f"{target}_decision"]["primary"] > row[target]["actual"]
        and not row[f"{target}_decision"]["strong"]
        for row in eligible
    )
    per_lifecycle: dict[str, Any] = {}
    for lifecycle_id in sorted({row["lifecycle_id"] for row in eligible}):
        life = [row for row in eligible if row["lifecycle_id"] == lifecycle_id]
        life_covered = [row for row in life if row[f"{target}_decision"]["primary"] is not None]
        life_errors = np.asarray([
            row[f"{target}_decision"]["primary"] - row[target]["actual"]
            for row in life_covered
        ], dtype=float)
        per_lifecycle[lifecycle_id] = {
            "sample_count": len(life),
            "forecast_coverage_pct": _percent(len(life_covered), len(life)),
            "mae_hours": float(np.mean(np.abs(life_errors))) if len(life_errors) else None,
            "bias_hours": float(np.mean(life_errors)) if len(life_errors) else None,
            "p90_absolute_error_hours": (
                float(np.percentile(np.abs(life_errors), 90)) if len(life_errors) else None
            ),
            "low_confidence_pct": _percent(
                sum(row[f"{target}_decision"]["strong"] for row in life), len(life)
            ),
            "withheld_pct": _percent(len(life) - len(life_covered), len(life)),
            "unsafe_late_prediction_rate_pct": _percent(
                int(np.sum(life_errors > 0.0)) if len(life_errors) else 0,
                len(life_covered),
            ),
        }
    return {
        "sample_count": len(eligible),
        "forecast_coverage_pct": _percent(len(covered), len(eligible)),
        "mae_hours": float(np.mean(np.abs(errors))) if len(errors) else None,
        "bias_hours": float(np.mean(errors)) if len(errors) else None,
        "p90_absolute_error_hours": float(np.percentile(np.abs(errors), 90)) if len(errors) else None,
        "low_confidence_pct": _percent(strong_count, len(eligible)),
        "withheld_pct": _percent(withheld_count, len(eligible)),
        "unsafe_late_prediction_rate_pct": _percent(unsafe_count, len(covered)),
        "unflagged_unsafe_late_prediction_rate_pct": _percent(
            unflagged_unsafe, len(covered)
        ),
        "median_absolute_error_hours": float(np.median(np.abs(errors))) if len(errors) else None,
        "maximum_absolute_error_hours": float(np.max(np.abs(errors))) if len(errors) else None,
        "p90_late_error_hours": float(np.quantile(np.maximum(errors, 0), .9)) if len(errors) else None,
        "maximum_late_error_hours": float(np.max(np.maximum(errors, 0))) if len(errors) else None,
        "early_prediction_rate_pct": _percent(int(np.sum(errors < 0)), len(errors)),
        "late_by_more_than_pct": {
            f"{minutes}_minutes": _percent(int(np.sum(errors > minutes / 60.0)), len(errors))
            for minutes in (5, 15, 30, 60, 180)
        },
        "out_of_distribution_pct": _percent(sum(row["out_of_distribution"] for row in eligible), len(eligible)),
        "beyond_target_support_pct": _percent(sum(row[target].get("beyond_target_support", False) for row in eligible), len(eligible)),
        "per_lifecycle": per_lifecycle,
    }


def evaluate_thresholds(
    records: list[dict[str, Any]], thresholds: GuardrailThresholds,
    duration_boundaries: tuple[float, ...] = (72.0, 168.0, 336.0),
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        for target in ("warning", "critical"):
            row[f"{target}_decision"] = _target_choice(
                row[target], target, row["out_of_distribution"], thresholds
            )
        evaluated.append(row)
    row_low = sum(
        row["warning_decision"]["strong"] or row["critical_decision"]["strong"]
        for row in evaluated
    )
    row_withheld = sum(
        row["warning_decision"]["withheld"] or row["critical_decision"]["withheld"]
        for row in evaluated
    )
    boundaries = duration_boundaries
    def bucket(duration: float) -> str:
        return "short" if duration < boundaries[0] else "medium" if duration < boundaries[1] else "long" if duration < boundaries[2] else "very_long"
    target_reports = {}
    for target in ("warning", "critical"):
        target_report = _target_metrics(evaluated, target)
        target_report["duration_buckets"] = {
            name: _target_metrics([row for row in evaluated if bucket(float(row.get("lifecycle_duration_hours", 0.0))) == name], target)
            for name in ("short", "medium", "long", "very_long")
        }
        target_reports[target] = target_report
    return {
        "thresholds": thresholds.to_dict(),
        "row_count": len(evaluated),
        "row_low_confidence_pct": _percent(row_low, len(evaluated)),
        "row_withheld_pct": _percent(row_withheld, len(evaluated)),
        "out_of_distribution_pct": _percent(
            sum(row["out_of_distribution"] for row in evaluated), len(evaluated)
        ),
        "warning": target_reports["warning"],
        "critical": target_reports["critical"],
    }


def _quantile_candidates(values: list[float]) -> list[float]:
    if not values:
        return [0.0]
    candidates = np.quantile(np.asarray(values, dtype=float), [0.50, 0.75, 0.90])
    return sorted({round(float(value), 6) for value in candidates})


def evaluate_guardrails(
    input_path: str | Path,
    config: ProjectConfig,
    models_root: str | Path,
) -> dict[str, Any]:
    feature_names, examples, data_summary = build_training_examples(
        input_path, config, models_root
    )
    training, validation, test = split_lifecycles(examples, config)
    registry = ModelRegistry(models_root)
    stage = "production" if registry.load_metadata("production") else "candidate"
    models: dict[str, Any] = {}
    metadata: dict[str, Any] | None = None
    for target in ("warning", "critical"):
        loaded = registry.load_model(
            f"time_to_{target}",
            stage=stage,
            expected_features=feature_names,
            expected_schema_version=config.ml.schema_version,
            expected_threshold_config_version=config.threshold_config_version,
            expected_lifecycle_config_version=config.lifecycle_config_version,
        )
        if loaded is None:
            raise RuntimeError(f"No compatible {stage} time_to_{target} model is available")
        model, target_metadata = loaded
        models[target] = model
        metadata = target_metadata
    assert metadata is not None

    training_records = _records(training, models, metadata, config)
    validation_records = _records(validation, models, metadata, config)
    test_records = _records(test, models, metadata, config)
    disagreements: dict[str, list[float]] = {"warning": [], "critical": [], "relative": []}
    for row in training_records:
        if row["out_of_distribution"]:
            continue
        for target in ("warning", "critical"):
            ml = row[target]["ml"]
            statistical = row[target]["statistical"]
            if ml is None or statistical is None:
                continue
            absolute = abs(ml - statistical)
            relative = absolute / max(abs(ml), abs(statistical), 1e-6)
            disagreements[target].append(absolute)
            disagreements["relative"].append(relative)

    candidates: list[GuardrailThresholds] = []
    for warning_absolute in _quantile_candidates(disagreements["warning"]):
        for critical_absolute in _quantile_candidates(disagreements["critical"]):
            for relative in _quantile_candidates(disagreements["relative"]):
                candidates.append(
                    GuardrailThresholds(warning_absolute, critical_absolute, relative)
                )
    validation_results = [evaluate_thresholds(validation_records, value) for value in candidates]

    def unsafe_recall(report: dict[str, Any]) -> float:
        unsafe = sum(
            report[target]["unsafe_late_prediction_rate_pct"]
            * report[target]["sample_count"] / 100.0
            for target in ("warning", "critical")
        )
        unflagged = sum(
            report[target]["unflagged_unsafe_late_prediction_rate_pct"]
            * report[target]["sample_count"] / 100.0
            for target in ("warning", "critical")
        )
        return 1.0 if unsafe <= 0 else max(0.0, (unsafe - unflagged) / unsafe)

    maximum_recall = max(unsafe_recall(report) for report in validation_results)
    recall_floor = 0.90 * maximum_recall
    selectable = [
        report for report in validation_results if unsafe_recall(report) >= recall_floor
    ]
    selected_validation = min(
        selectable,
        key=lambda report: (
            report["row_low_confidence_pct"],
            report["row_withheld_pct"],
            -unsafe_recall(report),
        ),
    )
    selected_thresholds = GuardrailThresholds(
        selected_validation["thresholds"]["warning_disagreement_absolute_hours"],
        selected_validation["thresholds"]["critical_disagreement_absolute_hours"],
        selected_validation["thresholds"]["disagreement_relative_threshold"],
    )
    # The untouched test group is evaluated exactly once, after validation-only selection.
    final_test = evaluate_thresholds(test_records, selected_thresholds)
    return {
        "status": "guardrail_selected",
        "model_stage": stage,
        "model_version": metadata.get("model_version"),
        "training_environment": metadata.get("training_environment"),
        "environment_compatibility": environment_compatibility(
            metadata.get("training_environment")
        ),
        **data_summary,
        "training_lifecycle_ids": [value.lifecycle_id for value in training],
        "validation_lifecycle_ids": [value.lifecycle_id for value in validation],
        "test_lifecycle_ids": [value.lifecycle_id for value in test],
        "candidate_generation": {
            "source": "training_lifecycle_disagreement_quantiles_only",
            "quantiles": [0.50, 0.75, 0.90],
            "warning_absolute_candidates_hours": _quantile_candidates(disagreements["warning"]),
            "critical_absolute_candidates_hours": _quantile_candidates(disagreements["critical"]),
            "relative_candidates": _quantile_candidates(disagreements["relative"]),
        },
        "selection": {
            "source": "validation_lifecycles_only",
            "rule": (
                "Among candidates retaining at least 90% of the maximum validation recall "
                "for unsafe-late predictions, minimize low-confidence rows, then withholding."
            ),
            "maximum_unsafe_late_recall": maximum_recall,
            "minimum_accepted_unsafe_late_recall": recall_floor,
            "selected_thresholds": selected_thresholds.to_dict(),
            "selected_validation_metrics": selected_validation,
        },
        "validation_candidate_results": validation_results,
        "untouched_test": {
            "evaluation_count": 1,
            "evaluated_after_selection": True,
            "metrics": final_test,
        },
    }
