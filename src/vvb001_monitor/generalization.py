from __future__ import annotations

import json
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from .config import SensorConfig
from .predictor import sensor_contract
from .synthetic import SyntheticConfig, generate_mock_csv
from .training import build_training_matrix, evaluate_regime_model


@dataclass(frozen=True)
class GeneralizationCriteria:
    """Synthetic bootstrap acceptance gates for unseen-lifecycle evaluation.

    These gates are deliberately about consistency of the learned degradation representation,
    not plant safety or production readiness. They use only simulator/lifecycle diagnostics.
    """

    max_early_anchor_alert_rate: float = 0.10
    min_late_critical_coverage: float = 0.20
    min_score_progress_spearman: float = 0.65
    min_score_latent_damage_spearman: float = 0.70
    max_transition_reversal_rate: float = 0.20

    def validate(self) -> None:
        rate_fields = {
            "max_early_anchor_alert_rate": self.max_early_anchor_alert_rate,
            "min_late_critical_coverage": self.min_late_critical_coverage,
            "max_transition_reversal_rate": self.max_transition_reversal_rate,
        }
        for name, value in rate_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name, value in {
            "min_score_progress_spearman": self.min_score_progress_spearman,
            "min_score_latent_damage_spearman": self.min_score_latent_damage_spearman,
        }.items():
            if not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")


def _metric_passes(metrics: dict[str, Any], criteria: GeneralizationCriteria) -> dict[str, bool]:
    progress_corr = metrics.get("score_progress_spearman")
    latent_corr = metrics.get("score_latent_damage_spearman")
    return {
        "early_anchor_alert_rate": (
            metrics.get("early_anchor_alert_rate") is not None
            and float(metrics["early_anchor_alert_rate"]) <= criteria.max_early_anchor_alert_rate
        ),
        "late_critical_coverage": (
            metrics.get("late_critical_coverage") is not None
            and float(metrics["late_critical_coverage"]) >= criteria.min_late_critical_coverage
        ),
        "score_progress_spearman": (
            progress_corr is not None and float(progress_corr) >= criteria.min_score_progress_spearman
        ),
        "score_latent_damage_spearman": (
            latent_corr is not None and float(latent_corr) >= criteria.min_score_latent_damage_spearman
        ),
        "transition_reversal_rate": (
            metrics.get("transition_reversal_rate") is not None
            and float(metrics["transition_reversal_rate"]) <= criteria.max_transition_reversal_rate
        ),
    }


def _summary(values: Iterable[float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not clean:
        return {"mean": None, "min": None, "max": None, "stdev": None}
    return {
        "mean": float(statistics.mean(clean)),
        "min": float(min(clean)),
        "max": float(max(clean)),
        "stdev": float(statistics.pstdev(clean)) if len(clean) > 1 else 0.0,
    }


def evaluate_generalization(
    model_path: str | Path,
    report_path: str | Path,
    sensor_config: SensorConfig,
    *,
    trials: int = 5,
    lifecycles: int = 30,
    machines: int = 6,
    cadence_seconds: int | None = None,
    seed_start: int = 1001,
    seeds: list[int] | None = None,
    line_sel: str = "LINE_1",
    criteria: GeneralizationCriteria | None = None,
    keep_generated_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate an already-trained model on fresh synthetic lifecycles without retraining it.

    Fresh synthetic datasets are generated from deterministic unseen seeds. The loaded model is
    held fixed for every trial. This is a synthetic generalization regression test only; it does
    not establish real-plant FP/FN, maintenance lead time, or production safety.
    """

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if trials < 1:
        raise ValueError("trials must be at least 1")
    if lifecycles < 3:
        raise ValueError("lifecycles must be at least 3")
    if machines < 2:
        raise ValueError("machines must be at least 2")
    if seed_start < 0:
        raise ValueError("seed_start cannot be negative")

    criteria = criteria or GeneralizationCriteria()
    criteria.validate()

    bundle = joblib.load(model_path)
    if bundle.get("sensor_contract") != sensor_contract(sensor_config):
        raise RuntimeError(
            "Model feature contract does not match the evaluator sensor configuration. "
            "Use the same acceleration unit, history window, feature windows, EWMA and baseline settings used for training."
        )
    if "model" not in bundle or "feature_names" not in bundle:
        raise ValueError("Model bundle is missing required model/feature_names fields")

    metadata = dict(bundle.get("metadata") or {})
    baseline_anchor_fraction = float(metadata.get("baseline_anchor_fraction", 0.15))
    if cadence_seconds is None:
        stored_cadence = metadata.get("training_cadence_seconds")
        cadence_seconds = int(round(float(stored_cadence))) if stored_cadence else 60
    if cadence_seconds < 10:
        raise ValueError("cadence_seconds must be at least 10")

    if seeds is None:
        trial_seeds = [seed_start + offset for offset in range(trials)]
    else:
        if not seeds:
            raise ValueError("seeds cannot be empty")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if any(seed < 0 for seed in seeds):
            raise ValueError("seeds cannot be negative")
        trial_seeds = [int(seed) for seed in seeds]

    keep_dir = Path(keep_generated_dir) if keep_generated_dir is not None else None
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)

    trial_reports: list[dict[str, Any]] = []

    def run_trials(work_dir: Path) -> None:
        for trial_index, seed in enumerate(trial_seeds, start=1):
            csv_path = work_dir / f"generalization_seed_{seed}.csv"
            rows = generate_mock_csv(
                csv_path,
                SyntheticConfig(
                    lifecycles=lifecycles,
                    machines=machines,
                    cadence_seconds=cadence_seconds,
                    seed=seed,
                    line_sel=line_sel,
                ),
            )
            X, groups, progress, latent, fault_modes, timestamps, feature_names = build_training_matrix(
                csv_path, sensor_config
            )
            expected_features = list(bundle["feature_names"])
            if feature_names != expected_features:
                raise RuntimeError(
                    "Generalization feature schema does not match the trained model. "
                    f"Expected {len(expected_features)} features, got {len(feature_names)}."
                )

            indices = np.arange(len(X), dtype=int)
            metrics = evaluate_regime_model(
                bundle["model"],
                X,
                groups,
                progress,
                latent,
                fault_modes,
                timestamps,
                indices,
                baseline_anchor_fraction=baseline_anchor_fraction,
            )
            checks = _metric_passes(metrics, criteria)
            trial_reports.append(
                {
                    "trial": trial_index,
                    "seed": seed,
                    "rows": rows,
                    "lifecycles": lifecycles,
                    "machines": machines,
                    "cadence_seconds": cadence_seconds,
                    "metrics": metrics,
                    "checks": checks,
                    "passed": bool(all(checks.values())),
                    "generated_csv": str(csv_path.resolve()) if keep_dir is not None else None,
                }
            )

    if keep_dir is not None:
        run_trials(keep_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="vvb001_generalization_") as temp_dir:
            run_trials(Path(temp_dir))

    metric_names = (
        "early_anchor_alert_rate",
        "late_critical_coverage",
        "score_progress_spearman",
        "score_latent_damage_spearman",
        "transition_reversal_rate",
        "silhouette_score",
    )
    aggregate = {
        name: _summary(trial["metrics"].get(name) for trial in trial_reports)
        for name in metric_names
    }
    fault_modes = sorted({
        mode
        for trial in trial_reports
        for mode in trial["metrics"].get("per_fault_mode", {})
    })
    aggregate_per_fault_mode = {
        mode: {
            "early_anchor_alert_rate": _summary(
                trial["metrics"].get("per_fault_mode", {}).get(mode, {}).get("early_anchor_alert_rate")
                for trial in trial_reports
            ),
            "late_critical_coverage": _summary(
                trial["metrics"].get("per_fault_mode", {}).get(mode, {}).get("late_critical_coverage")
                for trial in trial_reports
            ),
            "score_progress_spearman": _summary(
                trial["metrics"].get("per_fault_mode", {}).get(mode, {}).get("score_progress_spearman")
                for trial in trial_reports
            ),
            "score_latent_damage_spearman": _summary(
                trial["metrics"].get("per_fault_mode", {}).get(mode, {}).get("score_latent_damage_spearman")
                for trial in trial_reports
            ),
        }
        for mode in fault_modes
    }
    pass_count = sum(1 for trial in trial_reports if trial["passed"])
    report: dict[str, Any] = {
        "evaluation_type": "unseen_synthetic_lifecycle_generalization",
        "bootstrap_only": True,
        "model_path": str(model_path.resolve()),
        "model_metadata": metadata,
        "model_feature_count": len(bundle["feature_names"]),
        "baseline_anchor_fraction": baseline_anchor_fraction,
        "configuration": {
            "trial_count": len(trial_seeds),
            "seeds": trial_seeds,
            "lifecycles_per_trial": lifecycles,
            "machines_per_trial": machines,
            "cadence_seconds": cadence_seconds,
            "line_sel": line_sel,
            "generated_csvs_retained": keep_dir is not None,
            "generated_csv_directory": str(keep_dir.resolve()) if keep_dir is not None else None,
        },
        "criteria": asdict(criteria),
        "trials": trial_reports,
        "aggregate_metrics": aggregate,
        "aggregate_per_fault_mode": aggregate_per_fault_mode,
        "passed_trials": pass_count,
        "failed_trials": len(trial_reports) - pass_count,
        "overall_pass": bool(pass_count == len(trial_reports)),
        "interpretation": (
            "PASS means the fixed bootstrap model met the configured synthetic-regime consistency gates on every fresh seed. "
            "It does not mean the model is validated for plant maintenance decisions."
        ),
        "limitations": [
            "The evaluator uses the same synthetic generator family as bootstrap development, but with fresh seeds and lifecycles.",
            "latent_damage_score and lifecycle_progress are simulator-only evaluation fields and are never model inputs.",
            "No retraining, threshold fitting, or regime re-ordering is performed during evaluation.",
            "Real plant false-positive/false-negative rates require real maintenance/failure evidence and are not inferred here.",
        ],
    }

    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
