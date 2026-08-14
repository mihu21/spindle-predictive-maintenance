from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import SensorConfig
from .predictor import sensor_contract
from .rul_ml import (
    RUL_MODEL_VERSION_V2_2,
    RUL_TARGET_THRESHOLDS,
    build_rul_matrix,
    calibrate_rul_targets_temporally,
    causal_status_signals,
    compare_rul_point_candidates,
    derive_time_to_onset_targets,
    diagnostic_horizon_bucket,
    evaluate_rul_targets_temporally,
    fit_quantile_rul_target,
    rul_sample_weights_v2_2,
    validate_rul_feature_names,
)
from .synthetic import SyntheticConfig, generate_mock_csv
from .training import (
    EXTERNAL_RUL_EVALUATION_FILENAMES,
    EXTERNAL_RUL_EVALUATION_SHA256,
    assert_not_external_rul_evaluation_input,
    build_training_matrix,
)


V2_2_CORPUS_VERSION = "rul_v2_2_multiseed_causal_hazard_corpus_v1"
V2_2_SPLIT_POLICY = "seed_disjoint_fit_calibration_acceptance_60_20_20_v1"


@dataclass(frozen=True)
class RULV22AcceptanceCriteria:
    max_overall_mae_hours: float = 12.0
    min_interval_coverage: float = 0.80
    min_macro_lifecycle_coverage: float = 0.75
    min_gt48_interval_coverage: float = 0.75
    max_abs_gt48_bias_hours: float = 8.0
    min_availability: float = 0.95
    min_monotonicity: float = 0.95
    minimum_bucket_lifecycles: int = 8
    max_width_ratio_vs_v2_1: float = 1.25
    minimum_long_horizon_mae_improvement_fraction: float = 0.10


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: Iterable[float], q: float) -> float | None:
    data = np.asarray(list(values), dtype=float)
    return float(np.quantile(data, q)) if len(data) else None


def _corpus_distribution(rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, int]]:
    by_lifecycle: dict[str, list[dict[str, str]]] = {}
    lifecycle_to_seed: dict[str, int] = {}
    for row in rows:
        lifecycle = row["lifecycle_id"]
        by_lifecycle.setdefault(lifecycle, []).append(row)
        lifecycle_to_seed[lifecycle] = int(row["generation_seed"])
    durations: list[float] = []
    maximum_critical_rul: dict[str, float] = {}
    mode_counts: dict[str, int] = {}
    for lifecycle, local in by_lifecycle.items():
        local.sort(key=lambda row: datetime.fromisoformat(row["timestamp"]))
        timestamps = [datetime.fromisoformat(row["timestamp"]) for row in local]
        durations.append((timestamps[-1] - timestamps[0]).total_seconds() / 3600.0)
        critical = next(
            (ts for row, ts in zip(local, timestamps) if float(row["latent_damage_score"]) >= 0.75),
            None,
        )
        maximum_critical_rul[lifecycle] = (
            (critical - timestamps[0]).total_seconds() / 3600.0 if critical is not None else 0.0
        )
        mode = str(local[0].get("fault_mode") or "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    distribution = {
        "lifecycle_count": len(by_lifecycle),
        "seed_count": len(set(lifecycle_to_seed.values())),
        "duration_hours": {
            "min": min(durations) if durations else None,
            "median": _quantile(durations, 0.50),
            "p90": _quantile(durations, 0.90),
            "max": max(durations) if durations else None,
        },
        "lifecycles_contributing_true_rul_gt_24h": sum(value > 24.0 for value in maximum_critical_rul.values()),
        "lifecycles_contributing_true_rul_gt_48h": sum(value > 48.0 for value in maximum_critical_rul.values()),
        "lifecycles_contributing_true_rul_gt_72h": sum(value > 72.0 for value in maximum_critical_rul.values()),
        "lifecycles_contributing_true_rul_gt_96h": sum(value > 96.0 for value in maximum_critical_rul.values()),
        "fault_mode_lifecycle_counts": mode_counts,
    }
    return distribution, lifecycle_to_seed


def generate_rul_v2_2_development_corpus(
    output_csv: str | Path,
    manifest_path: str | Path,
    *,
    seeds: Iterable[int],
    lifecycles_per_seed: int = 20,
    machines: int = 6,
    cadence_seconds: int = 600,
    line_sel: str = "LINE_1",
) -> dict[str, Any]:
    """Generate a multi-seed corpus with globally unique identities and an immutable manifest."""
    seed_values = [int(seed) for seed in seeds]
    if len(seed_values) < 5 or len(set(seed_values)) != len(seed_values):
        raise ValueError("v2.2 development requires at least five unique seeds")
    if {12001, 14001} & set(seed_values):
        raise ValueError("Protected external-evaluation seeds cannot be reused for development")
    output = Path(output_csv)
    manifest_output = Path(manifest_path)
    assert_not_external_rul_evaluation_input(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    batch_dir = output.parent / f"{output.stem}_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, str]] = []
    batch_reports: list[dict[str, Any]] = []
    next_id = 1
    for batch_index, seed in enumerate(seed_values):
        prefix = f"dev_s{seed}"
        batch_path = batch_dir / f"{prefix}.csv"
        generate_mock_csv(
            batch_path,
            SyntheticConfig(
                lifecycles=lifecycles_per_seed,
                cadence_seconds=cadence_seconds,
                seed=seed,
                machines=machines,
                line_sel=line_sel,
                start_time=datetime(2027, 1, 1, tzinfo=timezone.utc) + timedelta(days=370 * batch_index),
                identity_prefix=prefix,
                duration_min_hours=54.0,
                duration_max_hours=156.0,
                causal_hazard_coupling=True,
            ),
        )
        with batch_path.open("r", encoding="utf-8", newline="") as handle:
            batch_rows = list(csv.DictReader(handle))
        for row in batch_rows:
            row["id"] = str(next_id)
            row["generation_seed"] = str(seed)
            row["development_batch"] = prefix
            next_id += 1
            combined.append(row)
        batch_reports.append({
            "seed": seed,
            "batch": prefix,
            "path": str(batch_path.resolve()),
            "sha256": _sha256(batch_path),
            "rows": len(batch_rows),
            "lifecycles": lifecycles_per_seed,
        })
    fields = list(combined[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    distribution, lifecycle_to_seed = _corpus_distribution(combined)
    manifest = {
        "corpus_version": V2_2_CORPUS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "protected_external_datasets_used": False,
        "output_csv": str(output.resolve()),
        "output_sha256": _sha256(output),
        "rows": len(combined),
        "seeds": seed_values,
        "lifecycles_per_seed": lifecycles_per_seed,
        "cadence_seconds": cadence_seconds,
        "generator_contract": {
            "globally_unique_lifecycle_and_machine_ids": True,
            "causal_hazard_coupling": True,
            "eventual_duration_or_onset_exposed_to_model": False,
            "duration_range_hours_before_wear_rate": [54.0, 156.0],
        },
        "distribution": distribution,
        "lifecycle_to_seed": lifecycle_to_seed,
        "batches": batch_reports,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _seed_disjoint_indices(
    groups: np.ndarray, lifecycle_to_seed: dict[str, int], *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    seeds = sorted(set(int(value) for value in lifecycle_to_seed.values()))
    if len(seeds) < 5:
        raise ValueError("At least five independent seeds are required for v2.2 splits")
    rng = np.random.default_rng(seed)
    shuffled = [int(value) for value in rng.permutation(seeds)]
    fit_count = max(3, int(math.floor(0.60 * len(shuffled))))
    calibration_count = max(1, int(math.floor(0.20 * len(shuffled))))
    if fit_count + calibration_count >= len(shuffled):
        fit_count = len(shuffled) - 2
        calibration_count = 1
    fit_seeds = set(shuffled[:fit_count])
    calibration_seeds = set(shuffled[fit_count:fit_count + calibration_count])
    acceptance_seeds = set(shuffled[fit_count + calibration_count:])
    row_seeds = np.asarray([lifecycle_to_seed[str(group)] for group in groups], dtype=int)
    fit_idx = np.flatnonzero(np.isin(row_seeds, list(fit_seeds)))
    calibration_idx = np.flatnonzero(np.isin(row_seeds, list(calibration_seeds)))
    acceptance_idx = np.flatnonzero(np.isin(row_seeds, list(acceptance_seeds)))
    audit = {
        "policy": V2_2_SPLIT_POLICY,
        "fit_seeds": sorted(fit_seeds),
        "calibration_seeds": sorted(calibration_seeds),
        "acceptance_seeds": sorted(acceptance_seeds),
        "seed_overlap": {
            "fit_calibration": sorted(fit_seeds & calibration_seeds),
            "fit_acceptance": sorted(fit_seeds & acceptance_seeds),
            "calibration_acceptance": sorted(calibration_seeds & acceptance_seeds),
        },
    }
    if any(audit["seed_overlap"].values()):
        raise RuntimeError(f"Seed leakage detected: {audit['seed_overlap']}")
    return fit_idx, calibration_idx, acceptance_idx, audit


def _weight_audit(groups: np.ndarray, target: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    weights = rul_sample_weights_v2_2(groups, target, indices)
    gi = groups[indices]
    yi = target[indices]
    buckets = np.asarray([diagnostic_horizon_bucket(value) for value in yi], dtype=object)
    return {
        "total_weight_by_horizon_bucket": {
            bucket: float(np.sum(weights[buckets == bucket]))
            for bucket in sorted(set(buckets.tolist()))
        },
        "total_weight_by_lifecycle": {
            str(lifecycle): float(np.sum(weights[gi == lifecycle]))
            for lifecycle in sorted(set(gi.tolist()))
        },
        "minimum_row_weight": float(np.min(weights)),
        "maximum_row_weight": float(np.max(weights)),
    }


def _identifiability_audit(
    X: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    candidate_report: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    idx = np.asarray([i for i in indices if np.isfinite(target[i]) and target[i] > 48.0], dtype=int)
    rng = np.random.default_rng(seed)
    if len(idx) > 2500:
        idx = np.sort(rng.choice(idx, size=2500, replace=False))
    if len(idx) < 2:
        return {"supported": False, "reason": "fewer_than_two_high_rul_rows"}
    Xi = np.asarray(X[idx], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(Xi, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    Xi = np.where(np.isfinite(Xi), Xi, medians)
    scale = np.std(Xi, axis=0)
    scale[scale < 1e-9] = 1.0
    Xi = (Xi - np.mean(Xi, axis=0)) / scale
    neighbors = NearestNeighbors(n_neighbors=min(20, len(idx))).fit(Xi)
    _, neighbor_indices = neighbors.kneighbors(Xi)
    target_deltas: list[float] = []
    for row_pos, candidates in enumerate(neighbor_indices):
        other = next(
            (candidate for candidate in candidates[1:] if groups[idx[candidate]] != groups[idx[row_pos]]),
            None,
        )
        if other is not None:
            target_deltas.append(abs(float(target[idx[row_pos]] - target[idx[other]])))
    selected = candidate_report["candidates"][candidate_report["selected"]]
    unconditional = float(np.median(target[indices][np.isfinite(target[indices])]))
    long_truth = target[indices][target[indices] > 48.0]
    unconditional_long_mae = float(np.mean(np.abs(long_truth - unconditional))) if len(long_truth) else None
    selected_long_mae = selected["long_horizon_mae_hours"]
    improvement = (
        1.0 - float(selected_long_mae) / unconditional_long_mae
        if unconditional_long_mae and selected_long_mae is not None
        else None
    )
    return {
        "supported": len(set(groups[idx].tolist())) >= 8,
        "sampled_high_rul_rows": int(len(idx)),
        "sampled_high_rul_lifecycles": len(set(groups[idx].tolist())),
        "nearest_other_lifecycle_target_delta_median_hours": _quantile(target_deltas, 0.50),
        "nearest_other_lifecycle_target_delta_p90_hours": _quantile(target_deltas, 0.90),
        "unconditional_median_baseline_hours": unconditional,
        "unconditional_high_rul_mae_hours": unconditional_long_mae,
        "selected_candidate_high_rul_mae_hours": selected_long_mae,
        "selected_candidate_improvement_vs_unconditional_fraction": improvement,
        "causal_signal_supported": bool(improvement is not None and improvement >= 0.05),
        "decision_rule": "selected grouped-OOF >48h MAE must improve at least 5% over unconditional-median baseline",
        "note": "Nearest-neighbor target dispersion estimates residual ambiguity; simulator truth and seed are excluded from X.",
    }


def _acceptance_checks(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    criteria: RULV22AcceptanceCriteria,
) -> tuple[dict[str, bool | None], dict[str, Any]]:
    long_combined = dict(current.get("gt48_combined_diagnostic") or {})
    long_supported = bool(long_combined.get("supported"))
    long_coverage = long_combined.get("interval_coverage") if long_supported else None
    long_bias = long_combined.get("mean_signed_error_hours") if long_supported else None
    width_ratio = None
    point_improvement = None
    if baseline and baseline.get("mean_interval_width_hours"):
        width_ratio = current["mean_interval_width_hours"] / baseline["mean_interval_width_hours"]
        current_long_mae = long_combined.get("mae_hours") if long_supported else None
        baseline_long = dict(baseline.get("gt48_combined_diagnostic") or {})
        baseline_long_mae = baseline_long.get("mae_hours") if baseline_long.get("supported") else None
        if current_long_mae is not None and baseline_long_mae and baseline_long_mae > 0:
            point_improvement = 1.0 - current_long_mae / float(baseline_long_mae)
    width_pass = (
        width_ratio is not None
        and (width_ratio <= criteria.max_width_ratio_vs_v2_1 or (point_improvement or -math.inf) >= criteria.minimum_long_horizon_mae_improvement_fraction)
    )
    checks: dict[str, bool | None] = {
        "overall_mae": current.get("mae_hours") is not None and current["mae_hours"] <= criteria.max_overall_mae_hours,
        "interval_coverage": current.get("interval_coverage") is not None and current["interval_coverage"] >= criteria.min_interval_coverage,
        "macro_lifecycle_coverage": current.get("macro_lifecycle_coverage") is not None and current["macro_lifecycle_coverage"] >= criteria.min_macro_lifecycle_coverage,
        "gt48_interval_coverage": None if long_coverage is None else long_coverage >= criteria.min_gt48_interval_coverage,
        "gt48_signed_bias": None if long_bias is None else abs(long_bias) <= criteria.max_abs_gt48_bias_hours,
        "availability": current.get("availability") is not None and current["availability"] >= criteria.min_availability,
        "monotonicity": current.get("monotonicity") is not None and current["monotonicity"] >= criteria.min_monotonicity,
        "interval_width_guard": width_pass if baseline is not None else False,
    }
    diagnostics = {
        "gt48_supported": long_supported,
        "gt48_lifecycles": long_combined.get("lifecycles"),
        "gt48_coverage": long_coverage,
        "gt48_signed_bias_hours": long_bias,
        "width_ratio_vs_frozen_v2_1": width_ratio,
        "long_horizon_mae_improvement_fraction": point_improvement,
    }
    return checks, diagnostics


def train_rul_v2_2_model(
    corpus_csv: str | Path,
    manifest_path: str | Path,
    frozen_status_model_path: str | Path,
    model_path: str | Path,
    report_path: str | Path,
    candidate_report_path: str | Path,
    horizon_report_path: str | Path,
    sensor_config: SensorConfig,
    *,
    seed: int = 2200,
    criteria: RULV22AcceptanceCriteria | None = None,
) -> dict[str, Any]:
    """Train only RUL v2.2 while preserving the frozen status/safety artifact."""
    criteria = criteria or RULV22AcceptanceCriteria()
    assert_not_external_rul_evaluation_input(corpus_csv)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("corpus_version") != V2_2_CORPUS_VERSION:
        raise ValueError("Unsupported or missing v2.2 development corpus manifest")
    if _sha256(corpus_csv) != manifest.get("output_sha256"):
        raise ValueError("Development corpus SHA-256 does not match its manifest")
    lifecycle_to_seed = {str(key): int(value) for key, value in manifest["lifecycle_to_seed"].items()}
    frozen_path = Path(frozen_status_model_path)
    frozen_sha_before = _sha256(frozen_path)
    frozen_bundle = joblib.load(frozen_path)
    if frozen_bundle.get("sensor_contract") != sensor_contract(sensor_config):
        raise RuntimeError("Frozen status model sensor contract does not match v2.2 training config")

    X, groups, _progress, latent, _fault_modes, timestamps, feature_names = build_training_matrix(corpus_csv, sensor_config)
    missing_provenance = sorted(set(groups.tolist()) - set(lifecycle_to_seed))
    if missing_provenance:
        raise ValueError(f"Manifest lacks lifecycle provenance: {missing_provenance[:3]}")
    if list(feature_names) != list(frozen_bundle["feature_names"]):
        raise RuntimeError("v2.2 base feature contract differs from the frozen status model")
    fit_idx, calibration_idx, acceptance_idx, split_audit = _seed_disjoint_indices(
        groups, lifecycle_to_seed, seed=seed
    )
    status_model = frozen_bundle["model"]
    frozen_status_object_hash = joblib.hash(status_model)
    causal_scores, causal_statuses = causal_status_signals(
        status_model, X, groups, timestamps, smoothing_tau_hours=0.25, hysteresis_margin=0.04
    )
    X_rul, rul_feature_names = build_rul_matrix(
        X, feature_names, groups, timestamps, causal_scores, causal_statuses
    )
    validate_rul_feature_names(rul_feature_names)
    targets = {
        name: derive_time_to_onset_targets(groups, timestamps, latent, threshold=threshold)
        for name, threshold in RUL_TARGET_THRESHOLDS.items()
    }
    selected, candidate_report = compare_rul_point_candidates(
        X_rul, targets["critical"], groups, fit_idx, seed=seed + 3000,
        minimum_long_horizon_lifecycles=criteria.minimum_bucket_lifecycles,
    )
    identifiability = _identifiability_audit(
        X_rul, targets["critical"], groups, fit_idx, candidate_report, seed=seed + 4000
    )
    artifacts: dict[str, dict[str, Any]] = {}
    fit_metrics: dict[str, Any] = {}
    for offset, name in enumerate(("warning", "critical")):
        artifact, metrics = fit_quantile_rul_target(
            X_rul, targets[name], groups, fit_idx, calibration_idx,
            seed=seed + 5000 + 1000 * offset,
            target_name=name,
            estimator_name=selected,
            weighting_version="v2_2",
            minimum_bucket_lifecycles=criteria.minimum_bucket_lifecycles,
        )
        artifacts[name] = artifact
        fit_metrics[name] = metrics
    temporal_calibration = calibrate_rul_targets_temporally(
        artifacts, X_rul, targets, groups, timestamps, calibration_idx,
        target_coverage=criteria.min_interval_coverage,
        target_macro_lifecycle_coverage=criteria.min_macro_lifecycle_coverage,
        minimum_bucket_lifecycles=criteria.minimum_bucket_lifecycles,
    )
    for name in artifacts:
        fit_metrics[name] = {**fit_metrics[name], **temporal_calibration[name]}

    acceptance = evaluate_rul_targets_temporally(
        artifacts, X_rul, targets, groups, timestamps, acceptance_idx,
        minimum_bucket_lifecycles=criteria.minimum_bucket_lifecycles,
    )
    baseline_acceptance = None
    existing_rul = frozen_bundle.get("rul_model")
    if existing_rul and list(existing_rul.get("feature_names") or []) == list(rul_feature_names):
        baseline_acceptance = evaluate_rul_targets_temporally(
            existing_rul["targets"], X_rul, targets, groups, timestamps, acceptance_idx,
            minimum_bucket_lifecycles=criteria.minimum_bucket_lifecycles,
        )
    checks, gate_diagnostics = _acceptance_checks(
        acceptance["critical"],
        baseline_acceptance["critical"] if baseline_acceptance else None,
        criteria,
    )
    supported_checks = [value for value in checks.values() if value is not None]
    gates_passed = bool(supported_checks and all(supported_checks))
    artifact = {
        "version": RUL_MODEL_VERSION_V2_2,
        "feature_names": rul_feature_names,
        "base_feature_names": list(feature_names),
        "targets": artifacts,
        "target_definition": {
            "warning": "first hidden synthetic damage >= 0.35; label only, never runtime input",
            "critical": "first hidden synthetic damage >= 0.75; label only, never runtime input",
        },
        "forbidden_runtime_fields": [
            "latent_damage_score", "true_damage", "true_state", "fault_mode", "generation_seed",
            "development_batch", "future WARNING/CRITICAL timestamp", "lifecycle_progress",
            "actual_rul", "true_rul", "remaining_life_target", "target_rul",
        ],
        "external_evaluation_training_guard": {
            "protected_filenames": sorted(EXTERNAL_RUL_EVALUATION_FILENAMES),
            "protected_sha256": sorted(EXTERNAL_RUL_EVALUATION_SHA256),
        },
        "selected_point_estimator": selected,
        "sample_weighting": "equal_lifecycle_hard_constraint_with_damped_horizon_balance_v2_2",
        "split_policy": V2_2_SPLIT_POLICY,
        "fit_seeds": split_audit["fit_seeds"],
        "calibration_seeds": split_audit["calibration_seeds"],
        "acceptance_seeds": split_audit["acceptance_seeds"],
        "min_history_hours": 0.5,
        "min_points": 6,
        "max_upward_jump_floor_hours": 0.5,
        "max_upward_jump_per_elapsed_hour": 1.5,
        "upward_revision_cooldown_hours": 6.0,
        "upward_revision_trigger_hours": 4.0,
        "development_gates_passed": gates_passed,
        "synthetic_only_not_production_ready": True,
    }
    bundle = {
        **frozen_bundle,
        "rul_model": artifact,
        "metadata": {
            **dict(frozen_bundle.get("metadata") or {}),
            "rul_model_version": RUL_MODEL_VERSION_V2_2,
            "rul_training_mode": "multiseed_seed_disjoint_frozen_status_development_only",
            "frozen_status_source_sha256": frozen_sha_before,
            "rul_development_gates_passed": gates_passed,
            "production_ready": False,
        },
    }
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output)
    model_sha = _sha256(model_output)
    persisted_bundle = joblib.load(model_output)
    persisted_status_object_hash = joblib.hash(persisted_bundle["model"])
    if persisted_status_object_hash != frozen_status_object_hash:
        raise RuntimeError("Persisted v2.2 bundle changed the frozen status model object")
    if list(persisted_bundle["feature_names"]) != list(frozen_bundle["feature_names"]):
        raise RuntimeError("Persisted v2.2 bundle changed the status feature contract")
    if _sha256(frozen_path) != frozen_sha_before:
        raise RuntimeError("v2.2 training mutated the frozen status source artifact")

    weight_audit = _weight_audit(groups, targets["critical"], fit_idx[np.isfinite(targets["critical"][fit_idx])])
    report = {
        "version": RUL_MODEL_VERSION_V2_2,
        "development_only": True,
        "production_ready": False,
        "warning": "Synthetic development evidence only; representative plant lifecycle validation remains required.",
        "corpus_manifest": str(Path(manifest_path).resolve()),
        "corpus_sha256": manifest["output_sha256"],
        "corpus_distribution": manifest["distribution"],
        "protected_external_datasets_used": False,
        "protected_dataset_guards": artifact["external_evaluation_training_guard"],
        "frozen_status_model": {
            "source_path": str(frozen_path.resolve()),
            "source_sha256_before": frozen_sha_before,
            "source_sha256_after": _sha256(frozen_path),
            "mutated": False,
            "status_object_hash_before": frozen_status_object_hash,
            "status_object_hash_after_persistence": persisted_status_object_hash,
            "base_feature_contract_unchanged": True,
        },
        "split_audit": split_audit,
        "candidate_comparison": candidate_report,
        "selected_model_rationale": candidate_report["selection_rule"],
        "identifiability_audit": identifiability,
        "sample_weight_audit": weight_audit,
        "calibration": fit_metrics,
        "acceptance_runtime_replay": acceptance,
        "frozen_v2_1_reference_runtime_replay": baseline_acceptance,
        "acceptance_criteria": asdict(criteria),
        "acceptance_checks": checks,
        "acceptance_gate_diagnostics": gate_diagnostics,
        "development_gates_passed": gates_passed,
        "sealed_holdout_authorized": bool(gates_passed and identifiability.get("causal_signal_supported")),
        "model_sha256": model_sha,
    }
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    candidate_output = Path(candidate_report_path)
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_output.write_text(json.dumps(candidate_report, indent=2), encoding="utf-8")
    horizon_output = Path(horizon_report_path)
    horizon_output.parent.mkdir(parents=True, exist_ok=True)
    horizon_output.write_text(json.dumps({
        "selected_candidate_oof": candidate_report["candidates"][selected]["horizon_diagnostics"],
        "acceptance_runtime_replay": acceptance["critical"]["horizon_diagnostics"],
        "frozen_v2_1_reference": (
            baseline_acceptance["critical"]["horizon_diagnostics"] if baseline_acceptance else None
        ),
    }, indent=2), encoding="utf-8")
    return report
