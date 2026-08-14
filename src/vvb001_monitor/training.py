from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupShuffleSplit

from .config import SensorConfig
from .features import FeatureEngine
from .models import SourceRecord, VVB001Reading
from .monitor import VVB001Monitor
from .predictor import sensor_contract
from .regime_model import LearnedRegimeModel, STATUSES
from .rul_ml import (
    RUL_MODEL_VERSION,
    RUL_TARGET_THRESHOLDS,
    build_rul_matrix,
    calibrate_rul_targets_temporally,
    causal_status_signals,
    derive_time_to_onset_targets,
    evaluate_quantile_rul_target,
    fit_quantile_rul_target,
    validate_rul_feature_names,
)
from .validation import VVB001Validator


EXTERNAL_RUL_EVALUATION_FILENAMES = frozenset({
    "rul_test_12001.csv",
    "rul_holdout_14001.csv",
})
EXTERNAL_RUL_EVALUATION_SHA256 = frozenset({
    "657a6ea261a626b2107d5d32d47ec66e84f91ee079437c17df8d9dbb7291db7b",
    "685a3867b369b237b2c57783df9b18b74f66cd7ceccee896f304d875bb2103be",
})


def _parse_dt(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _lifecycle_id_hash(values: list[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_not_external_rul_evaluation_input(csv_path: str | Path) -> None:
    """Fail loudly if the frozen external RUL regression dataset is passed to training."""
    path = Path(csv_path)
    if path.name.lower() in EXTERNAL_RUL_EVALUATION_FILENAMES:
        raise ValueError(f"External RUL evaluation dataset cannot be used for training: {path}")
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest().lower()
        if digest in EXTERNAL_RUL_EVALUATION_SHA256:
            raise ValueError(
                "External RUL evaluation dataset content hash cannot be used for training, "
                "calibration, or model selection"
            )


def _numeric_feature_names(features: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key, value in features.items():
        if key in {"machine_key", "acceleration_unit", "baseline_ready", "state_reset", "elapsed_since_previous_hours", "crest_recomputed", "crest_relative_error"}:
            continue
        # Counts encode elapsed warm-up time and absolute levels can encode machine identity.
        # The regime learner therefore uses relative-to-baseline and dynamic features.
        if key.endswith("_count") or key.endswith("_baseline_ready"):
            continue
        if key.endswith("_raw") or key.endswith("_ewma") or key.endswith("_baseline_mean") or key.endswith("_baseline_std"):
            continue
        if key.endswith("m_mean"):
            continue
        if isinstance(value, (int, float)) or value is None:
            names.append(key)
    return sorted(names)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    corr = np.corrcoef(ra, rb)[0, 1]
    return float(corr) if np.isfinite(corr) else None


def build_training_matrix(csv_path: str | Path, sensor_config: SensorConfig):
    validator = VVB001Validator(sensor_config)
    engine = FeatureEngine(sensor_config)
    monitor = VVB001Monitor(validator, engine)
    X_rows: list[list[float]] = []
    groups: list[str] = []
    progresses: list[float] = []
    latent_damage: list[float] = []
    fault_modes: list[str] = []
    timestamps: list[datetime] = []
    feature_names: list[str] | None = None
    previous_lifecycle_by_machine: dict[str, str] = {}

    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "id", "timestamp", "line_sel", "machine_id", "vrms", "arms", "apeak",
            "crest", "temp", "lifecycle_id", "lifecycle_progress",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Training CSV missing columns: {sorted(missing)}")
        for raw in reader:
            if str(raw.get("vibration_inference_fixture") or "0").strip().lower() in {
                "1", "true", "yes",
            }:
                raise ValueError(
                    "Vibration operating-inference fixtures are runtime tests, not training data"
                )
            operating_state = str(raw.get("operating_state") or "RUNNING").upper()
            if operating_state != "RUNNING":
                raise ValueError(
                    "Duty-cycled operating-context fixtures are runtime tests, not training data; "
                    "train only on confirmed RUNNING exposure rows with an audited operating-time contract"
                )
            reading = VVB001Reading(
                source_id=int(raw["id"]), timestamp=_parse_dt(raw["timestamp"]),
                line_sel=raw["line_sel"], machine_id=raw["machine_id"],
                vrms=float(raw["vrms"]), arms=float(raw["arms"]), apeak=float(raw["apeak"]),
                crest=float(raw["crest"]), temp=float(raw["temp"]),
            )
            lifecycle_id = raw["lifecycle_id"]
            if previous_lifecycle_by_machine.get(reading.machine_key) not in {None, lifecycle_id}:
                engine.reset_machine(reading.machine_key)
            previous_lifecycle_by_machine[reading.machine_key] = lifecycle_id

            validation, processed = monitor.process(SourceRecord(reading.source_id, dict(raw), reading))
            if processed is None:
                raise ValueError(f"Synthetic row {reading.source_id} failed validation: {validation.reasons}")
            if feature_names is None:
                feature_names = _numeric_feature_names(processed.features)
            X_rows.append([
                float(processed.features.get(name))
                if isinstance(processed.features.get(name), (int, float)) and processed.features.get(name) is not None
                else np.nan
                for name in feature_names
            ])
            groups.append(lifecycle_id)
            progresses.append(float(raw["lifecycle_progress"]))
            latent_damage.append(float(raw.get("latent_damage_score") or "nan"))
            fault_modes.append(str(raw.get("fault_mode") or "unknown"))
            timestamps.append(reading.timestamp)

    if not X_rows or feature_names is None:
        raise ValueError("Training CSV contains no rows")
    return (
        np.asarray(X_rows, dtype=float),
        np.asarray(groups, dtype=object),
        np.asarray(progresses, dtype=float),
        np.asarray(latent_damage, dtype=float),
        np.asarray(fault_modes, dtype=object),
        np.asarray(timestamps, dtype=object),
        feature_names,
    )


def evaluate_regime_model(
    model: LearnedRegimeModel,
    X: np.ndarray,
    groups: np.ndarray,
    progress: np.ndarray,
    latent: np.ndarray,
    fault_modes: np.ndarray,
    timestamps: np.ndarray,
    indices: np.ndarray,
    *,
    baseline_anchor_fraction: float,
) -> dict[str, Any]:
    Xi = X[indices]
    gi = groups[indices]
    pi = progress[indices]
    li = latent[indices]
    fi = fault_modes[indices]
    ti = timestamps[indices]
    raw_pred = model.predict(Xi)
    raw_score = model.degradation_score(Xi)
    smoothing_tau_hours = 0.25
    hysteresis_margin = 0.04
    pred = np.empty(len(Xi), dtype=object)
    score = np.empty(len(Xi), dtype=float)
    for lifecycle in sorted(set(gi.tolist())):
        positions = np.flatnonzero(gi == lifecycle)
        previous_score = None
        previous_status = None
        previous_ts = None
        for pos in positions:
            value = float(raw_score[pos])
            if previous_score is None or previous_ts is None:
                smoothed = value
            else:
                dt_h = max(0.0, (ti[pos] - previous_ts).total_seconds() / 3600.0)
                alpha = 1.0 - np.exp(-dt_h / smoothing_tau_hours) if dt_h > 0 else 0.0
                smoothed = float(alpha * value + (1.0 - alpha) * previous_score)
            status = model.status_from_score(smoothed, previous_status=previous_status, hysteresis=hysteresis_margin)
            score[pos] = smoothed
            pred[pos] = status
            previous_score = smoothed
            previous_status = status
            previous_ts = ti[pos]

    early = pi <= baseline_anchor_fraction
    late = pi >= 0.90
    severity = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    pred_severity = np.asarray([severity[str(s)] for s in pred], dtype=int)

    lifecycle_metrics: dict[str, Any] = {}
    reversal_count = 0
    transition_count = 0
    for lifecycle in sorted(set(gi.tolist())):
        mask = gi == lifecycle
        seq = pred_severity[mask]
        diffs = np.diff(seq)
        reversal_count += int(np.sum(diffs < 0))
        transition_count += int(np.sum(diffs != 0))
        lifecycle_metrics[str(lifecycle)] = {
            "rows": int(np.sum(mask)),
            "score_progress_spearman": _spearman(score[mask], pi[mask]),
            "start_score": float(np.mean(score[mask][: max(1, min(20, len(score[mask])))])),
            "end_score": float(np.mean(score[mask][-max(1, min(20, len(score[mask]))):])),
        }

    transformed = model._transform(Xi)
    cluster_labels = np.asarray([model.status_to_cluster[str(s)] for s in raw_pred])
    sil = None
    if len(set(cluster_labels.tolist())) >= 2 and len(transformed) >= 10:
        sample_n = min(1500, len(transformed))
        sample_idx = np.linspace(0, len(transformed) - 1, sample_n, dtype=int)
        try:
            sil = float(silhouette_score(transformed[sample_idx], cluster_labels[sample_idx]))
        except ValueError:
            sil = None

    by_fault: dict[str, Any] = {}
    for mode in sorted(set(fi.tolist())):
        mask = fi == mode
        mode_early = mask & early
        mode_late = mask & late
        by_fault[str(mode)] = {
            "rows": int(np.sum(mask)),
            "mean_degradation_score": float(np.mean(score[mask])),
            "early_anchor_alert_rate": float(np.mean(pred[mode_early] != "NORMAL")) if np.any(mode_early) else None,
            "late_critical_coverage": float(np.mean(pred[mode_late] == "CRITICAL")) if np.any(mode_late) else None,
            "score_progress_spearman": _spearman(score[mask], pi[mask]),
            "score_latent_damage_spearman": _spearman(score[mask], li[mask]) if np.all(np.isfinite(li[mask])) else None,
        }

    return {
        "rows": int(len(indices)),
        "lifecycles": sorted(set(gi.tolist())),
        "regime_counts": {status: int(np.sum(pred == status)) for status in STATUSES},
        "raw_regime_counts": {status: int(np.sum(raw_pred == status)) for status in STATUSES},
        "prediction_smoothing_tau_hours": smoothing_tau_hours,
        "prediction_hysteresis_margin": hysteresis_margin,
        "early_anchor_alert_rate": float(np.mean(pred[early] != "NORMAL")) if np.any(early) else None,
        "late_critical_coverage": float(np.mean(pred[late] == "CRITICAL")) if np.any(late) else None,
        "score_progress_spearman": _spearman(score, pi),
        "score_latent_damage_spearman": _spearman(score, li) if np.all(np.isfinite(li)) else None,
        "transition_reversal_rate": (reversal_count / transition_count) if transition_count else 0.0,
        "silhouette_score": sil,
        "per_fault_mode": by_fault,
        "per_lifecycle": lifecycle_metrics,
    }



def _bootstrap_sample_weights(groups: np.ndarray, progress: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Balance lifecycle contribution and give late-stage rows modest extra influence.

    The weights use only lifecycle identity/progress from the synthetic bootstrap generator.
    fault_mode and latent_damage_score remain audit-only and never influence fitting.
    """
    gi = groups[indices]
    pi = progress[indices]
    weights = np.ones(len(indices), dtype=float)
    for lifecycle in set(gi.tolist()):
        mask = gi == lifecycle
        count = max(1, int(np.sum(mask)))
        # Equalize lifecycle mass so long trajectories cannot dominate the clustering objective.
        weights[mask] *= 1.0 / count
    # Preserve broad unsupervised structure while giving the tail enough influence to form a
    # stable severe regime. Maximum multiplier is 2.5x at lifecycle completion.
    weights *= 1.0 + 1.5 * np.power(np.clip(pi, 0.0, 1.0), 3)
    weights *= len(weights) / np.sum(weights)
    return weights


def _calibrate_critical_boundary(
    model: LearnedRegimeModel,
    X: np.ndarray,
    groups: np.ndarray,
    progress: np.ndarray,
    latent: np.ndarray,
    fault_modes: np.ndarray,
    timestamps: np.ndarray,
    validation_indices: np.ndarray,
    *,
    baseline_anchor_fraction: float,
    target_late_coverage: float = 0.30,
) -> dict[str, Any]:
    """Conservatively calibrate only the CRITICAL score boundary on held-out lifecycles.

    The WARNING boundary is frozen. Candidate CRITICAL cutoffs are evaluated on validation
    lifecycles only. We choose the highest (most conservative) boundary that reaches the
    desired late-tail coverage without materially worsening transition reversals.
    """
    original_warning, original_critical = model.score_boundaries
    base = evaluate_regime_model(
        model, X, groups, progress, latent, fault_modes, timestamps, validation_indices,
        baseline_anchor_fraction=baseline_anchor_fraction,
    )
    base_reversal = float(base.get("transition_reversal_rate") or 0.0)
    reversal_limit = min(0.20, base_reversal + 0.05)
    minimum_gap = 0.06
    lower = min(original_critical, original_warning + minimum_gap)
    if lower >= original_critical - 1e-9:
        return {
            "applied": False,
            "reason": "no_boundary_room",
            "original_critical_boundary": original_critical,
            "selected_critical_boundary": original_critical,
            "validation_before": base,
            "validation_after": base,
            "target_late_coverage": target_late_coverage,
        }

    # Search from the original boundary downward. The first passing candidate is therefore
    # the most conservative threshold that fixes the held-out late-stage under-coverage.
    candidates = np.linspace(original_critical, lower, 31)
    selected = original_critical
    selected_metrics = base
    best_fallback = (float(base.get("late_critical_coverage") or 0.0), -original_critical, original_critical, base)
    for candidate in candidates[1:]:
        model.set_score_boundaries(original_warning, float(candidate))
        metrics = evaluate_regime_model(
            model, X, groups, progress, latent, fault_modes, timestamps, validation_indices,
            baseline_anchor_fraction=baseline_anchor_fraction,
        )
        late = float(metrics.get("late_critical_coverage") or 0.0)
        reversal = float(metrics.get("transition_reversal_rate") or 0.0)
        fallback_key = (late, -float(candidate), float(candidate), metrics)
        if fallback_key[:2] > best_fallback[:2] and reversal <= reversal_limit:
            best_fallback = fallback_key
        if late >= target_late_coverage and reversal <= reversal_limit:
            selected = float(candidate)
            selected_metrics = metrics
            break
    else:
        selected = float(best_fallback[2])
        selected_metrics = best_fallback[3]

    model.set_score_boundaries(
        original_warning,
        selected,
        calibration={
            "method": "heldout_lifecycle_tail_calibration_v1",
            "warning_boundary_frozen": original_warning,
            "original_critical_boundary": original_critical,
            "selected_critical_boundary": selected,
            "target_late_coverage": target_late_coverage,
            "validation_lifecycle_count": len(set(groups[validation_indices].tolist())),
            "validation_late_coverage_before": base.get("late_critical_coverage"),
            "validation_late_coverage_after": selected_metrics.get("late_critical_coverage"),
            "validation_reversal_before": base.get("transition_reversal_rate"),
            "validation_reversal_after": selected_metrics.get("transition_reversal_rate"),
            "reversal_limit": reversal_limit,
        },
    )
    return {
        "applied": bool(selected < original_critical - 1e-9),
        "original_critical_boundary": original_critical,
        "selected_critical_boundary": selected,
        "target_late_coverage": target_late_coverage,
        "validation_before": base,
        "validation_after": selected_metrics,
    }

def train_bootstrap_model(
    csv_path: str | Path,
    model_path: str | Path,
    report_path: str | Path,
    sensor_config: SensorConfig,
    *,
    seed: int = 42,
    baseline_anchor_fraction: float = 0.15,
) -> dict[str, Any]:
    assert_not_external_rul_evaluation_input(csv_path)
    if not 0.05 <= baseline_anchor_fraction <= 0.35:
        raise ValueError("baseline_anchor_fraction must be between 0.05 and 0.35")
    X, groups, progress, latent, fault_modes, timestamps, feature_names = build_training_matrix(csv_path, sensor_config)
    unique_groups = sorted(set(groups.tolist()))
    cadence_samples: list[float] = []
    for lifecycle in unique_groups:
        positions = np.flatnonzero(groups == lifecycle)
        for left, right in zip(positions[:-1], positions[1:]):
            delta = (timestamps[right] - timestamps[left]).total_seconds()
            if delta > 0:
                cadence_samples.append(float(delta))
    training_cadence_seconds = float(np.median(cadence_samples)) if cadence_samples else None
    if len(unique_groups) < 4:
        raise ValueError("At least 4 lifecycles are required for fit/validation/test lifecycle separation")

    # Test lifecycles remain untouched. A second lifecycle-disjoint split is used only for
    # conservative CRITICAL-boundary calibration.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    development_idx, test_idx = next(outer.split(X, groups=groups))
    development_groups = groups[development_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    fit_local, validation_local = next(inner.split(X[development_idx], groups=development_groups))
    fit_idx = development_idx[fit_local]
    validation_idx = development_idx[validation_local]

    fit_lifecycles = sorted(set(groups[fit_idx].tolist()))
    validation_lifecycles = sorted(set(groups[validation_idx].tolist()))
    test_lifecycles = sorted(set(groups[test_idx].tolist()))
    split_overlaps = {
        "fit_validation": sorted(set(fit_lifecycles) & set(validation_lifecycles)),
        "fit_test": sorted(set(fit_lifecycles) & set(test_lifecycles)),
        "validation_test": sorted(set(validation_lifecycles) & set(test_lifecycles)),
    }
    if any(split_overlaps.values()):
        raise RuntimeError(f"Lifecycle leakage detected in train/validation/test split: {split_overlaps}")

    fit_baseline = progress[fit_idx] <= baseline_anchor_fraction
    fit_weights = _bootstrap_sample_weights(groups, progress, fit_idx)

    model = LearnedRegimeModel(seed=seed)
    model.fit(X[fit_idx], progress[fit_idx], fit_baseline, sample_weight=fit_weights)
    calibration = _calibrate_critical_boundary(
        model, X, groups, progress, latent, fault_modes, timestamps, validation_idx,
        baseline_anchor_fraction=baseline_anchor_fraction,
    )

    # RUL ML v2.1 is trained after the status model is frozen. Its inputs are the same causal
    # sensor-derived feature contract plus degradation/status history available online. Hidden
    # simulator truth is used only below to construct time-to-onset labels.
    causal_scores, causal_statuses = causal_status_signals(
        model,
        X,
        groups,
        timestamps,
        smoothing_tau_hours=0.25,
        hysteresis_margin=0.04,
    )
    X_rul, rul_feature_names = build_rul_matrix(
        X, feature_names, groups, timestamps, causal_scores, causal_statuses
    )
    validate_rul_feature_names(rul_feature_names)
    warning_rul_target = derive_time_to_onset_targets(
        groups, timestamps, latent, threshold=RUL_TARGET_THRESHOLDS["warning"]
    )
    critical_rul_target = derive_time_to_onset_targets(
        groups, timestamps, latent, threshold=RUL_TARGET_THRESHOLDS["critical"]
    )
    warning_rul_artifact, warning_rul_validation = fit_quantile_rul_target(
        X_rul, warning_rul_target, groups, fit_idx, validation_idx,
        seed=seed + 1000, target_name="warning",
    )
    critical_rul_artifact, critical_rul_validation = fit_quantile_rul_target(
        X_rul, critical_rul_target, groups, fit_idx, validation_idx,
        seed=seed + 2000, target_name="critical",
    )
    temporal_calibration_metrics = calibrate_rul_targets_temporally(
        {"warning": warning_rul_artifact, "critical": critical_rul_artifact},
        X_rul,
        {"warning": warning_rul_target, "critical": critical_rul_target},
        groups,
        timestamps,
        validation_idx,
        target_coverage=0.80,
        target_macro_lifecycle_coverage=0.75,
        minimum_bucket_rows=50,
        minimum_bucket_lifecycles=8,
        min_history_hours=0.5,
        min_points=6,
        max_upward_jump_floor_hours=0.5,
        max_upward_jump_per_elapsed_hour=1.5,
        upward_revision_cooldown_hours=6.0,
        upward_revision_trigger_hours=4.0,
    )
    warning_rul_validation = {
        **warning_rul_validation,
        **temporal_calibration_metrics["warning"],
    }
    critical_rul_validation = {
        **critical_rul_validation,
        **temporal_calibration_metrics["critical"],
    }
    rul_artifact = {
        "version": RUL_MODEL_VERSION,
        "feature_names": rul_feature_names,
        "base_feature_names": list(feature_names),
        "targets": {
            "warning": warning_rul_artifact,
            "critical": critical_rul_artifact,
        },
        "target_definition": {
            "warning": "first hidden synthetic damage >= 0.35; label only, never runtime input",
            "critical": "first hidden synthetic damage >= 0.75; label only, never runtime input",
        },
        "forbidden_runtime_fields": [
            "latent_damage_score", "true_damage", "true_state", "fault_mode",
            "future WARNING/CRITICAL timestamp", "lifecycle_progress", "actual_rul",
            "true_rul", "remaining_life_target", "target_rul",
        ],
        "forbidden_feature_contract_version": "rul_future_label_denylist_v2_1",
        "external_evaluation_training_guard": {
            "protected_filenames": sorted(EXTERNAL_RUL_EVALUATION_FILENAMES),
            "protected_sha256": sorted(EXTERNAL_RUL_EVALUATION_SHA256),
        },
        "fit_lifecycles": fit_lifecycles,
        "validation_lifecycles": validation_lifecycles,
        "test_lifecycles": test_lifecycles,
        "lifecycle_audit_hashes": {
            "fit": _lifecycle_id_hash(fit_lifecycles),
            "validation": _lifecycle_id_hash(validation_lifecycles),
            "test": _lifecycle_id_hash(test_lifecycles),
        },
        "min_history_hours": 0.5,
        "min_points": 6,
        "max_upward_jump_floor_hours": 0.5,
        "max_upward_jump_per_elapsed_hour": 1.5,
        "upward_revision_cooldown_hours": 6.0,
        "upward_revision_trigger_hours": 4.0,
        "interval_semantics": "train-OOF raw bounds plus validation-only lifecycle-aware split-conformal calibration, synthetic-only",
    }
    warning_rul_test = evaluate_quantile_rul_target(
        warning_rul_artifact, X_rul, warning_rul_target, groups, test_idx
    )
    critical_rul_test = evaluate_quantile_rul_target(
        critical_rul_artifact, X_rul, critical_rul_target, groups, test_idx
    )

    report: dict[str, Any] = {
        "bootstrap_only": True,
        "learning_mode": "lifecycle_anchored_unsupervised_regimes",
        "warning": (
            "The model was trained on synthetic lifecycles without NORMAL/WARNING/CRITICAL labels. "
            "The three status names are assigned to learned regimes using early-lifecycle healthy anchors and lifecycle ordering. "
            "The CRITICAL score boundary is calibrated only on lifecycle-disjoint synthetic validation tails. "
            "Synthetic metrics are not evidence of real-plant production accuracy."
        ),
        "rows": int(len(groups)),
        "feature_count": len(feature_names),
        "model_features_exclude_window_counts": True,
        "baseline_anchor_fraction": baseline_anchor_fraction,
        "training_cadence_seconds": training_cadence_seconds,
        "fit_lifecycles": fit_lifecycles,
        "validation_lifecycles": validation_lifecycles,
        "test_lifecycles": test_lifecycles,
        "split_overlap_check": {
            "passed": True,
            "overlaps": split_overlaps,
            "fit_count": len(fit_lifecycles),
            "validation_count": len(validation_lifecycles),
            "test_count": len(test_lifecycles),
        },
        # Compatibility alias: older consumers expected train_lifecycles.
        "train_lifecycles": sorted(set(groups[development_idx].tolist())),
        "fit_weighting": {
            "lifecycle_equalization": True,
            "late_stage_multiplier": "1 + 1.5 * progress^3",
            "uses_fault_mode": False,
            "uses_latent_damage": False,
        },
        "critical_boundary_calibration": calibration,
        "regime_model": model.metadata(),
        "fit_evaluation": evaluate_regime_model(
            model, X, groups, progress, latent, fault_modes, timestamps, fit_idx,
            baseline_anchor_fraction=baseline_anchor_fraction,
        ),
        "validation_evaluation": evaluate_regime_model(
            model, X, groups, progress, latent, fault_modes, timestamps, validation_idx,
            baseline_anchor_fraction=baseline_anchor_fraction,
        ),
        "test_evaluation": evaluate_regime_model(
            model, X, groups, progress, latent, fault_modes, timestamps, test_idx,
            baseline_anchor_fraction=baseline_anchor_fraction,
        ),
        "rul_model": {
            "version": RUL_MODEL_VERSION,
            "feature_count": len(rul_feature_names),
            "feature_names": rul_feature_names,
            "forbidden_feature_check_passed": True,
            "external_evaluation_training_guard_passed": True,
            "fit_lifecycles": fit_lifecycles,
            "validation_lifecycles": validation_lifecycles,
            "test_lifecycles": test_lifecycles,
            "split_overlap_check": {"passed": True, "overlaps": split_overlaps},
            "warning_validation": warning_rul_validation,
            "critical_validation": critical_rul_validation,
            "warning_untouched_test": warning_rul_test,
            "critical_untouched_test": critical_rul_test,
            "sample_weighting": {
                "lifecycle_equalization": True,
                "within_48h": 2.0,
                "within_24h": 3.0,
                "within_12h": 4.5,
                "within_6h": 6.0,
            },
            "interval_calibration": "train-group-OOF raw bounds plus validation-only lifecycle-aware bucketed split conformal",
        },
        "metric_notes": {
            "early_anchor_alert_rate": "Proxy only: fraction of early-lifecycle anchor rows assigned WARNING/CRITICAL.",
            "late_critical_coverage": "Proxy only: fraction of final 10% lifecycle rows assigned CRITICAL.",
            "score_latent_damage_spearman": "Generator-only audit. latent_damage_score is never used for training or calibration.",
            "transition_reversal_rate": "Fraction of predicted status transitions that move backward in severity; lower is more stable.",
        },
    }

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "classes": list(STATUSES),
        "rul_model": rul_artifact,
        "sensor_contract": sensor_contract(sensor_config),
        "metadata": {
            "training_source": "synthetic_vvb001_unlabelled_bootstrap",
            "bootstrap_only": True,
            "learning_mode": "lifecycle_anchored_unsupervised_regimes",
            "seed": seed,
            "rows": int(len(groups)),
            "baseline_anchor_fraction": baseline_anchor_fraction,
            "training_cadence_seconds": training_cadence_seconds,
            "prediction_smoothing_tau_hours": 0.25,
            "prediction_hysteresis_margin": 0.04,
            "critical_boundary_calibration": model.metadata().get("score_boundary_calibration"),
            "fit_weighting_version": "lifecycle_balanced_late_weight_v1",
            "rul_model_version": RUL_MODEL_VERSION,
            "rul_training_mode": "supervised_quantile_completed_synthetic_lifecycles_lifecycle_disjoint",
            "rul_interval_calibration": "lifecycle_hierarchical_post_stabilization_split_conformal_v2_1",
        },
    }
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output)
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
