from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SensorConfig
from .predictor import sensor_contract
from .rul_ml import (
    RUL_DIAGNOSTIC_BUCKETS,
    RUL_MODEL_VERSION_V2_3,
    RUL_TARGET_THRESHOLDS,
    _fit_point_model,
    _stabilize_prediction_pair_values,
    build_rul_matrix,
    calibrate_rul_targets_temporally,
    causal_status_signals,
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


V2_3_CORPUS_VERSION = "rul_v2_3_role_locked_forecastability_corpus_v1"
V2_3_GENERATOR_VERSION = "causal_hazard_accumulated_stress_v2_2_unchanged"
V2_3_OOF_POLICY = "generation_batch_grouped_four_fold_runtime_point_replay_v1"
ROLE_ORDER = ("fit", "calibration", "acceptance")
ACTIVATION_THRESHOLD_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
BASELINE_NAMES = (
    "unconditional_training_median",
    "lifecycle_age_only_ridge",
    "current_severity_only_ridge",
    "lifecycle_age_plus_current_severity_ridge",
)


@dataclass(frozen=True)
class RULV23AcceptanceCriteria:
    max_active_mae_hours: float = 12.0
    min_active_interval_coverage: float = 0.80
    min_active_macro_lifecycle_coverage: float = 0.75
    min_active_monotonicity: float = 0.95
    max_abs_far_bias_hours: float = 8.0
    min_far_coverage: float = 0.70
    minimum_lifecycles: int = 8
    minimum_batches: int = 4
    min_active_region_availability: float = 0.95
    forecastable_improvement_fraction: float = 0.10
    max_bias_degradation_vs_baseline_hours: float = 2.0
    max_unidentifiable_active_rate: float = 0.05
    width_ratio_limit: float = 1.15
    width_absolute_increase_limit_hours: float = 6.0
    width_exception_macro_mae_improvement: float = 0.10


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: Iterable[float], q: float) -> float | None:
    data = np.asarray(list(values), dtype=float)
    return float(np.quantile(data, q)) if len(data) else None


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _batch_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_lifecycle: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_lifecycle[row["lifecycle_id"]].append(row)
    durations: list[float] = []
    onset_hours: list[float] = []
    rows_per_lifecycle: dict[str, int] = {}
    mode_counts: dict[str, int] = defaultdict(int)
    maximum_rul: dict[str, float] = {}
    for lifecycle, local in by_lifecycle.items():
        local.sort(key=lambda row: _parse_dt(row["timestamp"]))
        timestamps = [_parse_dt(row["timestamp"]) for row in local]
        duration = (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0
        durations.append(duration)
        rows_per_lifecycle[lifecycle] = len(local)
        mode_counts[str(local[0].get("fault_mode") or "unknown")] += 1
        critical = next(
            (ts for row, ts in zip(local, timestamps) if float(row["latent_damage_score"]) >= 0.75),
            None,
        )
        onset = (critical - timestamps[0]).total_seconds() / 3600.0 if critical else 0.0
        onset_hours.append(onset)
        maximum_rul[lifecycle] = onset
    return {
        "lifecycle_ids": sorted(by_lifecycle),
        "lifecycle_count": len(by_lifecycle),
        "rows": len(rows),
        "rows_per_lifecycle": rows_per_lifecycle,
        "duration_hours": {
            "min": min(durations), "median": _quantile(durations, 0.50),
            "p90": _quantile(durations, 0.90), "max": max(durations),
        },
        "critical_onset_hours_audit_only": {
            "min": min(onset_hours), "median": _quantile(onset_hours, 0.50),
            "p90": _quantile(onset_hours, 0.90), "max": max(onset_hours),
        },
        "fault_mode_lifecycle_counts": dict(sorted(mode_counts.items())),
        "horizon_support_lifecycles": {
            "gt_24h": sum(value > 24.0 for value in maximum_rul.values()),
            "gt_48h": sum(value > 48.0 for value in maximum_rul.values()),
            "gt_72h": sum(value > 72.0 for value in maximum_rul.values()),
            "gt_96h": sum(value > 96.0 for value in maximum_rul.values()),
        },
    }


def generate_rul_v2_3_development_corpus(
    output_csv: str | Path,
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    *,
    role_seeds: dict[str, Iterable[int]],
    lifecycles_per_batch: int = 12,
    machines: int = 6,
    cadence_seconds: int = 600,
    line_sel: str = "LINE_1",
) -> dict[str, Any]:
    """Generate fresh, permanently role-locked batches for v2.3 development."""
    normalized = {role: [int(seed) for seed in role_seeds.get(role, [])] for role in ROLE_ORDER}
    if any(len(normalized[role]) < 4 for role in ROLE_ORDER):
        raise ValueError("v2.3 requires at least four independent batches for every development role")
    all_seeds = [seed for role in ROLE_ORDER for seed in normalized[role]]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("A seed may belong to only one permanent development role")
    v22_manifest_path = Path("output/rul_v2_2/corpus_manifest.json")
    consumed_seeds = {12001, 14001}
    consumed_batches: list[dict[str, Any]] = []
    if v22_manifest_path.exists():
        old = json.loads(v22_manifest_path.read_text(encoding="utf-8"))
        consumed_seeds.update(int(seed) for seed in old.get("seeds", []))
        consumed_batches.extend({
            "batch_id": row.get("batch"), "seed": row.get("seed"),
            "role": "historical_v2_2_consumed", "sha256": row.get("sha256"),
        } for row in old.get("batches", []))
    overlap = sorted(consumed_seeds & set(all_seeds))
    if overlap:
        raise ValueError(f"Consumed/protected seeds cannot be reused in v2.3: {overlap}")

    output = Path(output_csv)
    manifest_output = Path(manifest_path)
    registry_output = Path(consumed_registry_path)
    assert_not_external_rul_evaluation_input(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    batch_dir = output.parent / f"{output.stem}_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, str]] = []
    batches: list[dict[str, Any]] = []
    lifecycle_to_batch: dict[str, str] = {}
    lifecycle_to_role: dict[str, str] = {}
    next_id = 1
    batch_ordinal = 0
    for role in ROLE_ORDER:
        role_prefix = {"fit": "devfit", "calibration": "devcal", "acceptance": "devacc"}[role]
        for role_index, seed in enumerate(normalized[role], start=1):
            batch_ordinal += 1
            batch_id = f"{role_prefix}_b{role_index:03d}_s{seed}"
            path = batch_dir / f"{batch_id}.csv"
            generate_mock_csv(
                path,
                SyntheticConfig(
                    lifecycles=lifecycles_per_batch,
                    cadence_seconds=cadence_seconds,
                    seed=seed,
                    machines=machines,
                    line_sel=line_sel,
                    start_time=datetime(2035, 1, 1, tzinfo=timezone.utc) + timedelta(days=370 * batch_ordinal),
                    identity_prefix=batch_id,
                    duration_min_hours=54.0,
                    duration_max_hours=156.0,
                    causal_hazard_coupling=True,
                ),
            )
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["id"] = str(next_id)
                row["generation_seed"] = str(seed)
                row["batch_id"] = batch_id
                row["development_role"] = role
                if (
                    row["lifecycle_id"] in lifecycle_to_batch
                    and lifecycle_to_batch[row["lifecycle_id"]] != batch_id
                ):
                    raise RuntimeError(f"Lifecycle identity collision across batches: {row['lifecycle_id']}")
                lifecycle_to_batch[row["lifecycle_id"]] = batch_id
                lifecycle_to_role[row["lifecycle_id"]] = role
                combined.append(row)
                next_id += 1
            summary = _batch_summary(rows)
            batches.append({
                "role": role,
                "batch_id": batch_id,
                "seed": seed,
                "scenario_family": "causal_hazard_accumulated_stress",
                "generator_version": V2_3_GENERATOR_VERSION,
                "generator_configuration_hash": hashlib.sha256(json.dumps({
                    "lifecycles": lifecycles_per_batch, "machines": machines,
                    "cadence_seconds": cadence_seconds, "duration": [54.0, 156.0],
                    "causal_hazard_coupling": True,
                }, sort_keys=True).encode()).hexdigest(),
                "source_file": str(path.resolve()),
                "source_sha256": _sha256(path),
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                **summary,
            })
    fields = list(combined[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    role_summary = {
        role: {
            "batch_count": sum(row["role"] == role for row in batches),
            "lifecycle_count": sum(row["lifecycle_count"] for row in batches if row["role"] == role),
            "rows": sum(row["rows"] for row in batches if row["role"] == role),
        }
        for role in ROLE_ORDER
    }
    manifest = {
        "corpus_version": V2_3_CORPUS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "role_assignments_permanent": True,
        "protected_external_datasets_used": False,
        "generator_version": V2_3_GENERATOR_VERSION,
        "output_csv": str(output.resolve()),
        "output_sha256": _sha256(output),
        "rows": len(combined),
        "role_summary": role_summary,
        "batches": batches,
        "lifecycle_to_batch": lifecycle_to_batch,
        "lifecycle_to_role": lifecycle_to_role,
        "provenance_fields_excluded_from_features": [
            "generation_seed", "batch_id", "development_role", "latent_damage_score",
            "fault_mode", "lifecycle_progress",
        ],
        "code_identifier": {
            "git_available": False,
            "synthetic_py_sha256": _sha256(Path(__file__).with_name("synthetic.py")),
            "rul_ml_py_sha256": _sha256(Path(__file__).with_name("rul_ml.py")),
        },
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = {
        "registry_version": "permanent_consumed_rul_evidence_registry_v1",
        "protected_external": [
            {
                "filename": "rul_test_12001.csv",
                "sha256": "657a6ea261a626b2107d5d32d47ec66e84f91ee079437c17df8d9dbb7291db7b",
                "role": "permanent_external",
            },
            {
                "filename": "rul_holdout_14001.csv",
                "sha256": "685a3867b369b237b2c57783df9b18b74f66cd7ceccee896f304d875bb2103be",
                "role": "permanent_external",
            },
        ],
        "historical_consumed_batches": consumed_batches,
        "v2_3_role_locked_batches": [
            {"batch_id": row["batch_id"], "seed": row["seed"], "role": row["role"], "sha256": row["source_sha256"]}
            for row in batches
        ],
        "role_reassignment_permitted": False,
    }
    registry_output.parent.mkdir(parents=True, exist_ok=True)
    registry_output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return manifest


RAW_HISTORY_FEATURE_NAMES = (
    "elapsed_lifecycle_hours",
    "vrms_current", "arms_current", "apeak_current", "crest_current", "temp_current",
    "vrms_delta_start", "arms_delta_start", "apeak_delta_start", "crest_delta_start", "temp_delta_start",
    "vrms_12h_mean", "arms_12h_mean", "apeak_12h_mean", "crest_12h_mean", "temp_12h_mean",
    "vrms_12h_slope", "arms_12h_slope", "apeak_12h_slope", "crest_12h_slope", "temp_12h_slope",
)


def _raw_causal_history_matrix(csv_path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Build an audit-only raw-history representation using no future observations."""
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    histories: dict[str, deque[tuple[datetime, np.ndarray]]] = defaultdict(deque)
    starts: dict[str, tuple[datetime, np.ndarray]] = {}
    result: list[list[float]] = []
    sensors = ("vrms", "arms", "apeak", "crest", "temp")
    for row in rows:
        lifecycle = row["lifecycle_id"]
        ts = _parse_dt(row["timestamp"])
        values = np.asarray([float(row[name]) for name in sensors], dtype=float)
        start_ts, start_values = starts.setdefault(lifecycle, (ts, values.copy()))
        history = histories[lifecycle]
        history.append((ts, values))
        cutoff = ts - timedelta(hours=12.0)
        while history and history[0][0] < cutoff:
            history.popleft()
        hist_values = np.asarray([item[1] for item in history], dtype=float)
        elapsed = max(0.0, (ts - start_ts).total_seconds() / 3600.0)
        means = np.mean(hist_values, axis=0)
        if len(history) >= 2:
            hours = np.asarray([(item[0] - history[0][0]).total_seconds() / 3600.0 for item in history])
            centered = hours - np.mean(hours)
            denominator = float(np.sum(centered ** 2))
            slopes = (
                np.sum(centered[:, None] * (hist_values - np.mean(hist_values, axis=0)), axis=0) / denominator
                if denominator > 1e-12
                else np.zeros(5)
            )
        else:
            slopes = np.zeros(5)
        result.append([elapsed, *values, *(values - start_values), *means, *slopes])
    return np.asarray(result, dtype=float), list(RAW_HISTORY_FEATURE_NAMES)


def _runtime_stabilize_points(
    points: np.ndarray,
    groups: np.ndarray,
    timestamps: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Replay critical point predictions through the exact v2.2 temporal revision policy."""
    output = np.full(len(points), np.nan, dtype=float)
    local_groups = groups[indices]
    for lifecycle in sorted(set(local_groups.tolist())):
        local = np.flatnonzero(local_groups == lifecycle)
        local = local[np.argsort(np.asarray([timestamps[indices[pos]].timestamp() for pos in local]))]
        started = timestamps[indices[local[0]]]
        recent: deque[datetime] = deque()
        previous_mid: float | None = None
        previous_ts: datetime | None = None
        last_revision: datetime | None = None
        for pos in local:
            ts = timestamps[indices[pos]]
            recent.append(ts)
            cutoff = ts - timedelta(hours=12.0)
            while recent and recent[0] < cutoff:
                recent.popleft()
            history_hours = (ts - started).total_seconds() / 3600.0
            if len(recent) < 6 or history_hours < 0.5:
                previous_ts = ts
                continue
            value = max(0.0, float(points[pos]))
            elapsed_h = max(0.0, (ts - previous_ts).total_seconds() / 3600.0) if previous_ts else 0.0
            _warning, critical, last_revision = _stabilize_prediction_pair_values(
                (value, value, value),
                (value, value, value),
                previous_warning_mid=None,
                previous_critical_mid=previous_mid,
                timestamp=ts,
                elapsed_h=elapsed_h,
                last_revision_at=last_revision,
                max_upward_jump_floor_hours=0.5,
                max_upward_jump_per_elapsed_hour=1.5,
                upward_revision_cooldown_hours=6.0,
                upward_revision_trigger_hours=4.0,
            )
            output[pos] = critical[1]
            previous_mid = critical[1]
            previous_ts = ts
    return output


def _fit_ridge(X: np.ndarray, y: np.ndarray, weights: np.ndarray):
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    model.fit(X, y, ridge__sample_weight=weights)
    return model


def _batch_grouped_runtime_oof(
    X_derived: np.ndarray,
    X_raw: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    timestamps: np.ndarray,
    causal_scores: np.ndarray,
    fit_indices: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    finite_fit = np.asarray([idx for idx in fit_indices if np.isfinite(target[idx])], dtype=int)
    unique_batches = sorted(set(batches[finite_fit].tolist()))
    if len(unique_batches) < 4:
        raise ValueError("v2.3 OOF requires at least four independent fit batches")
    folds = min(4, len(unique_batches))
    splitter = GroupKFold(n_splits=folds)
    names = ("derived_extra_trees", "raw_history_extra_trees", *BASELINE_NAMES)
    predictions = {name: np.full(len(target), np.nan, dtype=float) for name in names}
    fold_assignments: list[dict[str, Any]] = []
    placeholder = np.zeros((len(finite_fit), 1), dtype=float)
    for fold, (train_local, holdout_local) in enumerate(
        splitter.split(placeholder, target[finite_fit], groups=batches[finite_fit])
    ):
        train_idx = finite_fit[train_local]
        hold_idx = finite_fit[holdout_local]
        weights = rul_sample_weights_v2_2(groups, target, train_idx)
        derived_model = _fit_point_model(
            X_derived[train_idx], target[train_idx], weights, seed + fold,
            estimator_name="extra_trees_v2_1", shared_v2_2_preprocessing=True,
        )
        raw_model = _fit_point_model(
            X_raw[train_idx], target[train_idx], weights, seed + 100 + fold,
            estimator_name="extra_trees_v2_1", shared_v2_2_preprocessing=True,
        )
        age_train = X_raw[train_idx][:, [0]]
        age_hold = X_raw[hold_idx][:, [0]]
        severity_train = causal_scores[train_idx].reshape(-1, 1)
        severity_hold = causal_scores[hold_idx].reshape(-1, 1)
        ridge_inputs = {
            "lifecycle_age_only_ridge": (age_train, age_hold),
            "current_severity_only_ridge": (severity_train, severity_hold),
            "lifecycle_age_plus_current_severity_ridge": (
                np.column_stack([age_train, severity_train]),
                np.column_stack([age_hold, severity_hold]),
            ),
        }
        raw_fold_predictions: dict[str, np.ndarray] = {
            "derived_extra_trees": derived_model.predict(X_derived[hold_idx]),
            "raw_history_extra_trees": raw_model.predict(X_raw[hold_idx]),
            "unconditional_training_median": np.full(
                len(hold_idx), float(np.median(target[train_idx])), dtype=float
            ),
        }
        for name, (train_x, hold_x) in ridge_inputs.items():
            raw_fold_predictions[name] = _fit_ridge(train_x, target[train_idx], weights).predict(hold_x)
        for name, point in raw_fold_predictions.items():
            stabilized = _runtime_stabilize_points(point, groups, timestamps, hold_idx)
            predictions[name][hold_idx] = stabilized
        fold_assignments.append({
            "fold": fold,
            "training_batches": sorted(set(batches[train_idx].tolist())),
            "holdout_batches": sorted(set(batches[hold_idx].tolist())),
            "batch_overlap": sorted(set(batches[train_idx].tolist()) & set(batches[hold_idx].tolist())),
            "training_lifecycles": len(set(groups[train_idx].tolist())),
            "holdout_lifecycles": len(set(groups[hold_idx].tolist())),
        })
    return {
        "predictions": predictions,
        "indices": finite_fit,
        "folds": fold_assignments,
        "policy": V2_3_OOF_POLICY,
    }


def _error_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    use = mask & np.isfinite(truth) & np.isfinite(prediction)
    errors = prediction[use] - truth[use]
    local_groups = groups[use]
    local_batches = batches[use]
    lifecycle_mae = [
        float(np.mean(np.abs(errors[local_groups == lifecycle])))
        for lifecycle in sorted(set(local_groups.tolist()))
    ]
    return {
        "rows": int(np.sum(use)),
        "lifecycles": len(set(local_groups.tolist())),
        "batches": len(set(local_batches.tolist())),
        "mae_hours": float(np.mean(np.abs(errors))) if len(errors) else None,
        "median_abs_error_hours": float(np.median(np.abs(errors))) if len(errors) else None,
        "p90_abs_error_hours": _quantile(np.abs(errors), 0.90),
        "mean_signed_error_hours": float(np.mean(errors)) if len(errors) else None,
        "median_signed_error_hours": float(np.median(errors)) if len(errors) else None,
        "macro_lifecycle_mae_hours": float(np.mean(lifecycle_mae)) if lifecycle_mae else None,
    }


def _bootstrap_improvement_interval(
    truth: np.ndarray,
    ml: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    use = mask & np.isfinite(truth) & np.isfinite(ml) & np.isfinite(baseline)
    lifecycles = sorted(set(groups[use].tolist()))
    if len(lifecycles) < 2:
        return {"samples": 0, "p05": None, "median": None, "p95": None}
    ml_mae = {
        lifecycle: float(np.mean(np.abs(ml[use][groups[use] == lifecycle] - truth[use][groups[use] == lifecycle])))
        for lifecycle in lifecycles
    }
    baseline_mae = {
        lifecycle: float(np.mean(np.abs(baseline[use][groups[use] == lifecycle] - truth[use][groups[use] == lifecycle])))
        for lifecycle in lifecycles
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(1000):
        sample = rng.choice(lifecycles, size=len(lifecycles), replace=True)
        base = float(np.mean([baseline_mae[item] for item in sample]))
        model = float(np.mean([ml_mae[item] for item in sample]))
        if base > 1e-12:
            values.append(1.0 - model / base)
    return {
        "samples": len(values),
        "p05": _quantile(values, 0.05),
        "median": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
    }


def _cross_batch_neighbor_audit(
    X: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    fit_indices: np.ndarray,
    *,
    seed: int,
    representation: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    finite = np.asarray([idx for idx in fit_indices if np.isfinite(target[idx]) and target[idx] > 0.0], dtype=int)
    reference = (
        np.sort(rng.choice(finite, size=5000, replace=False))
        if len(finite) > 5000 else finite
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    imputer.fit(np.asarray(X[finite], dtype=float))
    reference_X = scaler.fit_transform(imputer.transform(np.asarray(X[reference], dtype=float)))
    for bucket_index, bucket in enumerate(RUL_DIAGNOSTIC_BUCKETS):
        candidates = finite[np.asarray([diagnostic_horizon_bucket(target[idx]) == bucket for idx in finite])]
        lifecycles = len(set(groups[candidates].tolist()))
        batch_count = len(set(batches[candidates].tolist()))
        if len(candidates) > 300:
            query = np.sort(rng.choice(candidates, size=300, replace=False))
        else:
            query = candidates
        if len(query) < 2 or batch_count < 2:
            result[bucket] = {
                "representation": representation, "rows": int(len(candidates)),
                "lifecycles": lifecycles, "batches": batch_count, "supported": False,
            }
            continue
        query_X = scaler.transform(imputer.transform(np.asarray(X[query], dtype=float)))
        neighbors = NearestNeighbors(n_neighbors=min(100, len(reference))).fit(reference_X)
        distances, neighbor_positions = neighbors.kneighbors(query_X)
        nearest_distances: list[float] = []
        nearest_deltas: list[float] = []
        conditional_variances: list[float] = []
        conditional_stds: list[float] = []
        local_dispersions: list[float] = []
        ambiguity: list[dict[str, Any]] = []
        represented_lifecycles: set[str] = set()
        represented_batches: set[str] = set()
        for query_idx, ds, positions in zip(query, distances, neighbor_positions):
            valid: list[tuple[float, int]] = []
            for distance, position in zip(ds, positions):
                candidate_idx = int(reference[position])
                if candidate_idx == int(query_idx):
                    continue
                if groups[candidate_idx] == groups[query_idx] or batches[candidate_idx] == batches[query_idx]:
                    continue
                valid.append((float(distance), candidate_idx))
                if len(valid) >= 10:
                    break
            if not valid:
                continue
            distance, neighbor_idx = valid[0]
            delta = abs(float(target[query_idx] - target[neighbor_idx]))
            nearest_distances.append(distance)
            nearest_deltas.append(delta)
            represented_lifecycles.update([str(groups[query_idx]), str(groups[neighbor_idx])])
            represented_batches.update([str(batches[query_idx]), str(batches[neighbor_idx])])
            neighbor_truth = np.asarray([target[idx] for _, idx in valid], dtype=float)
            conditional_variances.append(float(np.var(neighbor_truth)))
            conditional_stds.append(float(np.std(neighbor_truth)))
            local_dispersions.append(float(np.median(np.abs(neighbor_truth - target[query_idx]))))
            ambiguity.append({
                "query_lifecycle": str(groups[query_idx]),
                "query_batch": str(batches[query_idx]),
                "query_true_rul_hours": float(target[query_idx]),
                "neighbor_lifecycle": str(groups[neighbor_idx]),
                "neighbor_batch": str(batches[neighbor_idx]),
                "neighbor_true_rul_hours": float(target[neighbor_idx]),
                "standardized_distance": distance,
                "absolute_rul_delta_hours": delta,
            })
        ambiguity.sort(key=lambda row: (-row["absolute_rul_delta_hours"], row["standardized_distance"]))
        result[bucket] = {
            "representation": representation,
            "rows": int(len(candidates)),
            "sampled_queries": len(query),
            "cross_batch_reference_rows": int(len(reference)),
            "valid_cross_batch_neighbors": len(nearest_deltas),
            "lifecycles": lifecycles,
            "batches": batch_count,
            "independent_lifecycles_represented": len(represented_lifecycles),
            "independent_batches_represented": len(represented_batches),
            "supported": lifecycles >= 8 and batch_count >= 4,
            "median_nearest_neighbor_distance": _quantile(nearest_distances, 0.50),
            "conditional_rul_variance_mean": float(np.mean(conditional_variances)) if conditional_variances else None,
            "conditional_rul_std_mean": float(np.mean(conditional_stds)) if conditional_stds else None,
            "median_absolute_target_dispersion": _quantile(local_dispersions, 0.50),
            "nearest_neighbor_abs_rul_delta_median": _quantile(nearest_deltas, 0.50),
            "nearest_neighbor_abs_rul_delta_p90": _quantile(nearest_deltas, 0.90),
            "ambiguity_examples": ambiguity[:8],
            "same_lifecycle_neighbors_excluded": True,
            "same_batch_neighbors_excluded": True,
        }
    return result


def _forecastability_report(
    oof: dict[str, Any],
    target: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    raw_neighbor: dict[str, Any],
    derived_neighbor: dict[str, Any],
    criteria: RULV23AcceptanceCriteria,
    *,
    seed: int,
) -> dict[str, Any]:
    predictions = oof["predictions"]
    indices = oof["indices"]
    valid_oof = np.zeros(len(target), dtype=bool)
    valid_oof[indices] = True
    horizons: dict[str, Any] = {}
    for bucket_index, bucket in enumerate(RUL_DIAGNOSTIC_BUCKETS):
        mask = valid_oof & np.asarray([
            np.isfinite(value) and diagnostic_horizon_bucket(value) == bucket for value in target
        ])
        derived_metrics = _error_metrics(
            target, predictions["derived_extra_trees"], groups, batches, mask
        )
        raw_metrics = _error_metrics(
            target, predictions["raw_history_extra_trees"], groups, batches, mask
        )
        baseline_metrics = {
            name: _error_metrics(target, predictions[name], groups, batches, mask)
            for name in BASELINE_NAMES
        }
        supported_baselines = [
            name for name, row in baseline_metrics.items()
            if row["macro_lifecycle_mae_hours"] is not None
        ]
        best = min(
            supported_baselines,
            key=lambda name: (baseline_metrics[name]["macro_lifecycle_mae_hours"], name),
        ) if supported_baselines else None
        best_metrics = baseline_metrics.get(best) if best else None
        improvement = None
        bias_degradation = None
        interval = {"samples": 0, "p05": None, "median": None, "p95": None}
        if best_metrics and best_metrics["macro_lifecycle_mae_hours"]:
            improvement = 1.0 - (
                derived_metrics["macro_lifecycle_mae_hours"]
                / best_metrics["macro_lifecycle_mae_hours"]
            )
            bias_degradation = abs(float(derived_metrics["mean_signed_error_hours"])) - abs(
                float(best_metrics["mean_signed_error_hours"])
            )
            interval = _bootstrap_improvement_interval(
                target,
                predictions["derived_extra_trees"],
                predictions[best],
                groups,
                mask,
                seed=seed + bucket_index,
            )
        supported = (
            derived_metrics["lifecycles"] >= criteria.minimum_lifecycles
            and derived_metrics["batches"] >= criteria.minimum_batches
        )
        raw_improvement = None
        if best_metrics and best_metrics["macro_lifecycle_mae_hours"]:
            raw_improvement = 1.0 - (
                raw_metrics["macro_lifecycle_mae_hours"] / best_metrics["macro_lifecycle_mae_hours"]
            )
        dispersion = derived_neighbor.get(bucket, {}).get("nearest_neighbor_abs_rul_delta_median")
        if not supported:
            classification = "UNSUPPORTED"
            rationale = "minimum independent lifecycle/batch support not met"
        elif (
            improvement is not None
            and improvement >= criteria.forecastable_improvement_fraction
            and bias_degradation is not None
            and bias_degradation <= criteria.max_bias_degradation_vs_baseline_hours
            and interval.get("p05") is not None
            and interval["p05"] > 0.0
        ):
            classification = "FORECASTABLE"
            rationale = "derived causal representation beats the best simple baseline with positive lifecycle-bootstrap evidence"
        elif improvement is not None and improvement > 0.0:
            classification = "WEAKLY_FORECASTABLE"
            rationale = "causal representation improves on baseline but misses the predeclared strength or bias requirement"
        elif raw_improvement is not None and raw_improvement > 0.0:
            classification = "WEAKLY_FORECASTABLE"
            rationale = "raw causal history contains signal that the current derived feature representation does not preserve"
        elif dispersion is not None and dispersion >= 12.0:
            classification = "UNIDENTIFIABLE"
            rationale = "cross-batch similar causal histories retain large target dispersion and neither representation beats simple baselines"
        else:
            classification = "WEAKLY_FORECASTABLE"
            rationale = "evidence is inconclusive; exact RUL must not be described as strongly forecastable"
        horizons[bucket] = {
            "classification": classification,
            "supported": supported,
            "rationale": rationale,
            "derived_runtime_oof": derived_metrics,
            "raw_history_runtime_oof": raw_metrics,
            "simple_baselines": baseline_metrics,
            "best_simple_baseline": best,
            "lifecycle_macro_mae_improvement_fraction": improvement,
            "raw_history_improvement_fraction": raw_improvement,
            "absolute_bias_degradation_vs_baseline_hours": bias_degradation,
            "lifecycle_bootstrap_improvement_interval": interval,
            "raw_history_neighbor_audit": raw_neighbor.get(bucket),
            "derived_feature_neighbor_audit": derived_neighbor.get(bucket),
            "feature_information_loss_hours": (
                derived_metrics["macro_lifecycle_mae_hours"] - raw_metrics["macro_lifecycle_mae_hours"]
                if derived_metrics["macro_lifecycle_mae_hours"] is not None
                and raw_metrics["macro_lifecycle_mae_hours"] is not None
                else None
            ),
        }
    return {
        "classification_rule": {
            "forecastable_min_macro_mae_improvement_fraction": criteria.forecastable_improvement_fraction,
            "maximum_bias_degradation_vs_baseline_hours": criteria.max_bias_degradation_vs_baseline_hours,
            "requires_positive_lifecycle_bootstrap_p05": True,
            "minimum_lifecycles": criteria.minimum_lifecycles,
            "minimum_batches": criteria.minimum_batches,
        },
        "horizons": horizons,
        "oof_policy": oof["policy"],
        "oof_folds": oof["folds"],
        "generator_diagnosis": {
            "generator_version": V2_3_GENERATOR_VERSION,
            "v2_2_causal_hazard_coupling_retained": True,
            "observable_precursor": "elapsed accumulated operating stress affects vibration/current/temperature from early life",
            "remaining_hidden_randomness": "nominal duration, onset fraction, exponent, and fault mode retain stochastic variation",
            "generator_changed_in_v2_3": False,
            "reason": "v2.3 first audits the existing causal generator regime; it does not tune the generator against acceptance evidence",
        },
        "probability_forecasts": {
            "implemented": False,
            "reason": "deferred separate model family; probabilities are not fabricated from RUL intervals",
        },
    }


def _activation_policy_from_fit_oof(
    prediction: np.ndarray,
    target: np.ndarray,
    causal_scores: np.ndarray,
    causal_statuses: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    fit_indices: np.ndarray,
    *,
    criteria: RULV23AcceptanceCriteria,
) -> dict[str, Any]:
    base = np.zeros(len(target), dtype=bool)
    base[fit_indices] = True
    base &= np.isfinite(prediction) & np.isfinite(target) & (target > 0.0)
    candidates: list[dict[str, Any]] = []
    for threshold in ACTIVATION_THRESHOLD_GRID:
        active = base & (
            (causal_scores >= threshold) | (causal_statuses != "NORMAL")
        )
        overall = _error_metrics(target, prediction, groups, batches, active)
        far = active & (target > 48.0)
        far_metrics = _error_metrics(target, prediction, groups, batches, far)
        active_rows = int(np.sum(active))
        pass_policy = bool(
            overall["mae_hours"] is not None
            and overall["mae_hours"] <= criteria.max_active_mae_hours
            and far_metrics["mean_signed_error_hours"] is not None
            and abs(far_metrics["mean_signed_error_hours"]) <= criteria.max_abs_far_bias_hours
            and overall["lifecycles"] >= criteria.minimum_lifecycles
            and overall["batches"] >= criteria.minimum_batches
        )
        overall_mae = overall["mae_hours"] if overall["mae_hours"] is not None else math.inf
        far_bias = far_metrics["mean_signed_error_hours"] if far_metrics["mean_signed_error_hours"] is not None else math.inf
        violation = (
            max(0.0, float(overall_mae) - criteria.max_active_mae_hours)
            + max(0.0, abs(float(far_bias)) - criteria.max_abs_far_bias_hours)
        )
        candidates.append({
            "minimum_degradation_score": threshold,
            "active_rows": active_rows,
            "active_fraction": float(active_rows / np.sum(base)) if np.sum(base) else None,
            "overall": overall,
            "gt48": far_metrics,
            "passes_fit_oof_activation_requirements": pass_policy,
            "normalized_violation_score": violation,
        })
    passing = [row for row in candidates if row["passes_fit_oof_activation_requirements"]]
    selected = (
        max(passing, key=lambda row: (row["active_fraction"], -row["minimum_degradation_score"]))
        if passing
        else min(candidates, key=lambda row: (row["normalized_violation_score"], -row["minimum_degradation_score"]))
    )
    return {
        "candidate_grid_predeclared": list(ACTIVATION_THRESHOLD_GRID),
        "selection_rule": (
            "maximum fit-OOF active fraction among thresholds passing MAE/bias/support; "
            "if none pass, minimum normalized violation with stricter-threshold tie break"
        ),
        "selected_minimum_degradation_score": selected["minimum_degradation_score"],
        "selected_passed_fit_oof_requirements": selected["passes_fit_oof_activation_requirements"],
        "candidates": candidates,
        "acceptance_data_used": False,
    }


def _weighting_diagnostics(
    groups: np.ndarray,
    batches: np.ndarray,
    target: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    idx = np.asarray([item for item in fit_indices if np.isfinite(target[item])], dtype=int)
    weights = rul_sample_weights_v2_2(groups, target, idx)
    truth = target[idx]
    bucket_names = np.asarray([diagnostic_horizon_bucket(value) for value in truth], dtype=object)
    result: dict[str, Any] = {}
    for bucket in RUL_DIAGNOSTIC_BUCKETS:
        mask = bucket_names == bucket
        local = weights[mask]
        result[bucket] = {
            "rows": int(np.sum(mask)),
            "lifecycles": len(set(groups[idx][mask].tolist())),
            "batches": len(set(batches[idx][mask].tolist())),
            "raw_row_weight_sum": float(np.sum(mask)),
            "lifecycle_normalized_weight_sum": float(np.sum(local)),
            "percentage_total_optimization_weight": float(np.sum(local) / np.sum(weights)) if len(local) else 0.0,
            "effective_sample_size": (
                float(np.sum(local) ** 2 / np.sum(local ** 2)) if len(local) and np.sum(local ** 2) > 0 else 0.0
            ),
        }
    lifecycle_masses = {
        str(lifecycle): float(np.sum(weights[groups[idx] == lifecycle]))
        for lifecycle in sorted(set(groups[idx].tolist()))
    }
    return {
        "policy": "equal lifecycle hard constraint with damped horizon balance v2.2 frozen before acceptance",
        "by_horizon": result,
        "lifecycle_mass_min": min(lifecycle_masses.values()),
        "lifecycle_mass_max": max(lifecycle_masses.values()),
        "lifecycle_masses": lifecycle_masses,
        "acceptance_data_used": False,
    }


def _top_feature_rank_associations(
    X: np.ndarray,
    names: list[str],
    target: np.ndarray,
    indices: np.ndarray,
    *,
    maximum: int = 12,
) -> list[dict[str, Any]]:
    use = np.asarray([idx for idx in indices if np.isfinite(target[idx]) and target[idx] > 72.0], dtype=int)
    rows: list[dict[str, Any]] = []
    if len(use) < 3:
        return rows
    target_rank = np.argsort(np.argsort(target[use], kind="mergesort"), kind="mergesort").astype(float)
    for column, name in enumerate(names):
        values = np.asarray(X[use, column], dtype=float)
        finite = np.isfinite(values)
        if np.sum(finite) < 3 or np.all(values[finite] == values[finite][0]):
            continue
        value_rank = np.argsort(np.argsort(values[finite], kind="mergesort"), kind="mergesort").astype(float)
        correlation = float(np.corrcoef(value_rank, target_rank[finite])[0, 1])
        if np.isfinite(correlation):
            rows.append({"feature": name, "spearman_like_rank_association": correlation})
    rows.sort(key=lambda row: (-abs(row["spearman_like_rank_association"]), row["feature"]))
    return rows[:maximum]


def _acceptance_gate_report(
    current: dict[str, Any],
    baseline: dict[str, Any],
    classifications: dict[str, Any],
    criteria: RULV23AcceptanceCriteria,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, observed: Any, threshold: str, passed: bool | None, *, reason: str | None = None) -> None:
        checks[name] = {
            "observed": observed,
            "threshold": threshold,
            "status": "UNSUPPORTED" if passed is None else "PASS" if passed else "FAIL",
            "reason": reason,
        }

    record("active_overall_mae", current.get("mae_hours"), "<= 12 h", current.get("mae_hours") is not None and current["mae_hours"] <= criteria.max_active_mae_hours)
    record("active_interval_coverage", current.get("interval_coverage"), ">= 0.80", current.get("interval_coverage") is not None and current["interval_coverage"] >= criteria.min_active_interval_coverage)
    record("active_macro_lifecycle_coverage", current.get("macro_lifecycle_coverage"), ">= 0.75", current.get("macro_lifecycle_coverage") is not None and current["macro_lifecycle_coverage"] >= criteria.min_active_macro_lifecycle_coverage)
    record("active_monotonicity", current.get("monotonicity"), ">= 0.95", current.get("monotonicity") is not None and current["monotonicity"] >= criteria.min_active_monotonicity)
    record(
        "active_region_availability", current.get("availability_inside_active_region"), ">= 0.95",
        current.get("availability_inside_active_region") is not None
        and current["availability_inside_active_region"] >= criteria.min_active_region_availability,
    )
    for bucket in ("48_72h", "gt_72h"):
        classification = classifications[bucket]["classification"]
        row = current.get("horizon_diagnostics", {}).get(bucket, {})
        support = row.get("lifecycles", 0) >= criteria.minimum_lifecycles and row.get("batches", 0) >= criteria.minimum_batches
        if classification == "UNSUPPORTED":
            record(f"{bucket}_support", {"lifecycles": row.get("lifecycles"), "batches": row.get("batches")}, ">=8 lifecycles and >=4 batches", None, reason="forecastability classification unsupported")
        elif classification == "UNIDENTIFIABLE":
            active_rate = current.get("forecastability_contract_by_true_horizon", {}).get(bucket, {}).get("contract_active_rate")
            record(
                f"{bucket}_unidentifiable_active_rate", active_rate, "<= 0.05",
                active_rate is not None and active_rate <= criteria.max_unidentifiable_active_rate,
                reason="exact RUL should be withheld in a proven unidentifiable region",
            )
        elif not support:
            record(f"{bucket}_active_support", {"lifecycles": row.get("lifecycles"), "batches": row.get("batches")}, ">=8 lifecycles and >=4 batches", None)
        else:
            bias = row.get("mean_signed_error_hours")
            coverage = row.get("interval_coverage")
            record(f"{bucket}_active_bias", bias, "absolute bias <= 8 h", bias is not None and abs(bias) <= criteria.max_abs_far_bias_hours)
            record(f"{bucket}_active_coverage", coverage, ">= 0.70", coverage is not None and coverage >= criteria.min_far_coverage)
            improvement = classifications[bucket].get("lifecycle_macro_mae_improvement_fraction")
            bias_degradation = classifications[bucket].get("absolute_bias_degradation_vs_baseline_hours")
            if classification == "FORECASTABLE":
                record(
                    f"{bucket}_baseline_comparison",
                    {"macro_mae_improvement": improvement, "bias_degradation": bias_degradation},
                    ">=10% macro MAE improvement and <=2 h bias degradation",
                    improvement is not None and improvement >= criteria.forecastable_improvement_fraction
                    and bias_degradation is not None and bias_degradation <= criteria.max_bias_degradation_vs_baseline_hours,
                )
    baseline_width = baseline.get("mean_interval_width_hours")
    current_width = current.get("mean_interval_width_hours")
    baseline_macro = baseline.get("macro_lifecycle_mae_hours")
    current_macro = current.get("macro_lifecycle_mae_hours")
    width_limit = min(
        criteria.width_ratio_limit * baseline_width,
        baseline_width + criteria.width_absolute_increase_limit_hours,
    ) if baseline_width is not None else None
    macro_improvement = (
        1.0 - current_macro / baseline_macro
        if baseline_macro and current_macro is not None else None
    )
    width_pass = bool(
        current_width is not None and width_limit is not None
        and (current_width <= width_limit or (macro_improvement or -math.inf) >= criteria.width_exception_macro_mae_improvement)
    )
    record(
        "interval_width_guard",
        {"new_mean_width": current_width, "baseline_mean_width": baseline_width, "limit": width_limit, "macro_mae_improvement": macro_improvement},
        "<= min(1.15*baseline, baseline+6h) unless >=10% macro MAE improvement",
        width_pass,
    )
    statuses = [row["status"] for row in checks.values()]
    passed = bool(statuses and all(status == "PASS" for status in statuses))
    return {"checks": checks, "all_required_gates_passed": passed}


def train_rul_v2_3_model(
    corpus_csv: str | Path,
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    frozen_status_model_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    sensor_config: SensorConfig,
    *,
    seed: int = 2300,
    criteria: RULV23AcceptanceCriteria | None = None,
) -> dict[str, Any]:
    """Execute the v2.3 forecastability decision tree without using protected evidence."""
    criteria = criteria or RULV23AcceptanceCriteria()
    assert_not_external_rul_evaluation_input(corpus_csv)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    registry = json.loads(Path(consumed_registry_path).read_text(encoding="utf-8"))
    if manifest.get("corpus_version") != V2_3_CORPUS_VERSION:
        raise ValueError("Unsupported v2.3 corpus manifest")
    if _sha256(corpus_csv) != manifest.get("output_sha256"):
        raise ValueError("v2.3 corpus does not match its manifest SHA-256")
    if registry.get("role_reassignment_permitted") is not False:
        raise ValueError("Consumed evidence registry must permanently prohibit role reassignment")

    frozen_path = Path(frozen_status_model_path)
    frozen_sha = _sha256(frozen_path)
    frozen_bundle = joblib.load(frozen_path)
    if frozen_bundle.get("sensor_contract") != sensor_contract(sensor_config):
        raise RuntimeError("Frozen status model sensor contract mismatch")
    frozen_status_hash = joblib.hash(frozen_bundle["model"])
    X, groups, _progress, latent, _fault_modes, timestamps, feature_names = build_training_matrix(corpus_csv, sensor_config)
    X_raw, raw_feature_names = _raw_causal_history_matrix(corpus_csv)
    if len(X_raw) != len(X):
        raise RuntimeError("Raw-history and runtime-derived matrices are not row-aligned")
    lifecycle_to_batch = {str(key): str(value) for key, value in manifest["lifecycle_to_batch"].items()}
    lifecycle_to_role = {str(key): str(value) for key, value in manifest["lifecycle_to_role"].items()}
    batches = np.asarray([lifecycle_to_batch[str(group)] for group in groups], dtype=object)
    roles = np.asarray([lifecycle_to_role[str(group)] for group in groups], dtype=object)
    role_indices = {role: np.flatnonzero(roles == role) for role in ROLE_ORDER}
    role_overlap = {
        f"{left}_{right}": sorted(set(batches[role_indices[left]].tolist()) & set(batches[role_indices[right]].tolist()))
        for pos, left in enumerate(ROLE_ORDER) for right in ROLE_ORDER[pos + 1:]
    }
    if any(role_overlap.values()):
        raise RuntimeError(f"Development role batch leakage detected: {role_overlap}")
    if list(feature_names) != list(frozen_bundle["feature_names"]):
        raise RuntimeError("Frozen status feature contract changed")
    status_model = frozen_bundle["model"]
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
    oof = _batch_grouped_runtime_oof(
        X_rul, X_raw, targets["critical"], groups, batches, timestamps,
        causal_scores, role_indices["fit"], seed=seed + 1000,
    )
    raw_neighbor = _cross_batch_neighbor_audit(
        X_raw, targets["critical"], groups, batches, role_indices["fit"],
        seed=seed + 2000, representation="raw_causal_history",
    )
    derived_neighbor = _cross_batch_neighbor_audit(
        X_rul, targets["critical"], groups, batches, role_indices["fit"],
        seed=seed + 3000, representation="derived_runtime_rul_features",
    )
    identifiability = _forecastability_report(
        oof, targets["critical"], groups, batches, raw_neighbor, derived_neighbor,
        criteria, seed=seed + 4000,
    )
    identifiability["early_precursor_information"] = {
        "raw_history_top_rank_associations_gt72": _top_feature_rank_associations(
            X_raw, raw_feature_names, targets["critical"], role_indices["fit"]
        ),
        "derived_feature_top_rank_associations_gt72": _top_feature_rank_associations(
            X_rul, rul_feature_names, targets["critical"], role_indices["fit"]
        ),
        "interpretation_rule": "raw signal with weaker derived OOF performance indicates feature-information loss; neither space beating causal baselines indicates intrinsic ambiguity",
    }
    activation = _activation_policy_from_fit_oof(
        oof["predictions"]["derived_extra_trees"], targets["critical"],
        causal_scores, causal_statuses, groups, batches, role_indices["fit"], criteria=criteria,
    )
    threshold = float(activation["selected_minimum_degradation_score"])
    artifacts: dict[str, dict[str, Any]] = {}
    training_metrics: dict[str, Any] = {}
    for offset, name in enumerate(("warning", "critical")):
        artifact, metrics = fit_quantile_rul_target(
            X_rul, targets[name], groups, role_indices["fit"], role_indices["calibration"],
            seed=seed + 5000 + offset * 1000,
            target_name=name,
            estimator_name="extra_trees_v2_1",
            weighting_version="v2_2",
            minimum_bucket_lifecycles=criteria.minimum_lifecycles,
        )
        artifacts[name] = artifact
        training_metrics[name] = metrics
    temporal_calibration = calibrate_rul_targets_temporally(
        artifacts, X_rul, targets, groups, timestamps, role_indices["calibration"],
        target_coverage=criteria.min_active_interval_coverage,
        target_macro_lifecycle_coverage=criteria.min_active_macro_lifecycle_coverage,
        minimum_bucket_lifecycles=criteria.minimum_lifecycles,
        activation_scores=causal_scores,
        activation_statuses=causal_statuses,
        minimum_activation_score=threshold,
    )
    for name in artifacts:
        training_metrics[name] = {**training_metrics[name], **temporal_calibration[name]}
    acceptance = evaluate_rul_targets_temporally(
        artifacts, X_rul, targets, groups, timestamps, role_indices["acceptance"],
        minimum_bucket_lifecycles=criteria.minimum_lifecycles,
        activation_scores=causal_scores,
        activation_statuses=causal_statuses,
        minimum_activation_score=threshold,
        batch_labels=batches,
    )
    baseline_rul = frozen_bundle.get("rul_model")
    if not baseline_rul or list(baseline_rul.get("feature_names") or []) != list(rul_feature_names):
        raise RuntimeError("Frozen compatible RUL baseline is required for v2.3 width comparison")
    baseline_acceptance = evaluate_rul_targets_temporally(
        baseline_rul["targets"], X_rul, targets, groups, timestamps, role_indices["acceptance"],
        minimum_bucket_lifecycles=criteria.minimum_lifecycles,
        activation_scores=causal_scores,
        activation_statuses=causal_statuses,
        minimum_activation_score=threshold,
        batch_labels=batches,
    )
    gate_report = _acceptance_gate_report(
        acceptance["critical"], baseline_acceptance["critical"],
        identifiability["horizons"], criteria,
    )
    gates_passed = bool(gate_report["all_required_gates_passed"])
    artifact = {
        "version": RUL_MODEL_VERSION_V2_3,
        "feature_names": rul_feature_names,
        "base_feature_names": list(feature_names),
        "targets": artifacts,
        "forecastability_contract": {
            "mode": "degradation_evidence_activation_v1",
            "minimum_degradation_score": threshold,
            "low_confidence_degradation_score": 0.5 * threshold,
            "normal_below_threshold_behavior": "clear point and interval; exact RUL unavailable",
            "warning_behavior": "current WARNING remains immediate; active RUL permitted",
            "critical_behavior": "current CRITICAL immediately emits zero for both endpoints",
            "probability_forecasts_implemented": False,
            "selection_source": "fit-batch grouped runtime OOF only",
        },
        "target_definition": {
            "warning": "first hidden synthetic damage >=0.35; label only",
            "critical": "first hidden synthetic damage >=0.75; label only",
        },
        "forbidden_runtime_fields": [
            "latent_damage_score", "true_damage", "true_state", "fault_mode", "generation_seed",
            "batch_id", "development_role", "future onset", "eventual duration",
            "lifecycle_progress", "true_rul", "target_rul",
        ],
        "external_evaluation_training_guard": {
            "protected_filenames": sorted(EXTERNAL_RUL_EVALUATION_FILENAMES),
            "protected_sha256": sorted(EXTERNAL_RUL_EVALUATION_SHA256),
        },
        "fit_batches": sorted(set(batches[role_indices["fit"]].tolist())),
        "calibration_batches": sorted(set(batches[role_indices["calibration"]].tolist())),
        "acceptance_batches": sorted(set(batches[role_indices["acceptance"]].tolist())),
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
            "rul_model_version": RUL_MODEL_VERSION_V2_3,
            "rul_training_mode": "role_locked_batch_disjoint_forecastability_aware_development",
            "frozen_status_source_sha256": frozen_sha,
            "rul_development_gates_passed": gates_passed,
            "production_ready": False,
        },
    }
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output)
    model_sha = _sha256(model_output)
    persisted = joblib.load(model_output)
    persisted_status_hash = joblib.hash(persisted["model"])
    if _sha256(frozen_path) != frozen_sha or persisted_status_hash != frozen_status_hash:
        raise RuntimeError("Frozen status model changed during v2.3 training/persistence")
    weighting = _weighting_diagnostics(
        groups, batches, targets["critical"], role_indices["fit"]
    )
    runtime_contract = {
        "forecastability_contract": artifact["forecastability_contract"],
        "fit_oof_activation_policy": activation,
        "acceptance": {
            "overall_exact_rul_availability": acceptance["critical"]["availability"],
            "availability_inside_forecastable_active_region": acceptance["critical"]["availability_inside_active_region"],
            "active_rows": acceptance["critical"]["forecastability_contract_active_rows"],
            "withheld_rows": acceptance["critical"]["forecastability_contract_withheld_rows"],
            "active_region_metrics": acceptance["critical"],
        },
        "withholding_clears_point_interval_and_actionability": True,
        "manufacturer_warning_critical_precedence_unchanged": True,
        "probability_outputs": "deferred; not derived from RUL interval",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "version": RUL_MODEL_VERSION_V2_3,
        "development_only": True,
        "production_ready": False,
        "protected_external_datasets_used": False,
        "corpus_sha256": manifest["output_sha256"],
        "model_sha256": model_sha,
        "frozen_status_model_sha256": frozen_sha,
        "frozen_status_object_hash_before": frozen_status_hash,
        "frozen_status_object_hash_after_persistence": persisted_status_hash,
        "role_overlap": role_overlap,
    }
    identifiability_report = {**common, **identifiability}
    horizon_report = {
        **common,
        "fit_oof_forecastability": identifiability["horizons"],
        "acceptance_runtime_active_predictions": acceptance["critical"]["horizon_diagnostics"],
        "combined_gt48_acceptance": acceptance["critical"]["gt48_combined_diagnostic"],
    }
    acceptance_report = {
        **common,
        "criteria": asdict(criteria),
        "runtime_active_prediction_metrics": acceptance,
        "frozen_baseline_same_active_cohort": baseline_acceptance,
        **gate_report,
        "sealed_holdout_authorized": gates_passed,
        "sealed_holdout_generated_or_evaluated": False,
    }
    reports = {
        "identifiability_report.json": identifiability_report,
        "horizon_diagnostics.json": horizon_report,
        "weighting_diagnostics.json": {**common, **weighting},
        "runtime_contract_report.json": {**common, **runtime_contract},
        "development_acceptance_report.json": acceptance_report,
        "training_report.json": {
            **common,
            "corpus_manifest": str(Path(manifest_path).resolve()),
            "consumed_registry": str(Path(consumed_registry_path).resolve()),
            "training_and_calibration": training_metrics,
            "activation_policy": activation,
            "forecastability_classifications": {
                bucket: row["classification"] for bucket, row in identifiability["horizons"].items()
            },
            "development_gates_passed": gates_passed,
            "sealed_holdout_authorized": gates_passed,
            "warning": "Synthetic development evidence only; not production ready.",
        },
    }
    for name, payload in reports.items():
        (output / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return reports["training_report.json"]
