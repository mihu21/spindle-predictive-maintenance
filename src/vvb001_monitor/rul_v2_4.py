from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .config import SensorConfig
from .predictor import sensor_contract
from .rul_evaluation import evaluate_rul
from .rul_features_v2_4 import (
    V2_4_CAUSAL_FEATURE_NAMES,
    V2_4_RATE_FEATURES,
    V2_4_STRESS_FEATURES,
    V24CausalFeatureBuilder,
)
from .rul_ml import (
    ColumnSubsetRegressor,
    RUL_DIAGNOSTIC_BUCKETS,
    RUL_MODEL_VERSION_V2_4,
    RUL_TARGET_THRESHOLDS,
    _fit_point_model,
    build_rul_matrix,
    calibrate_rul_targets_temporally,
    causal_status_signals,
    derive_time_to_onset_targets,
    diagnostic_horizon_bucket,
    fit_quantile_rul_target,
    rul_sample_weights_v2_2,
    rul_sample_weights_v2_4,
    v2_4_selector_matrix,
    v2_4_support_signals,
    validate_rul_feature_names,
)
from .rul_v2_3 import (
    BASELINE_NAMES,
    RULV23AcceptanceCriteria,
    _batch_grouped_runtime_oof,
    _batch_summary,
    _cross_batch_neighbor_audit,
    _error_metrics,
    _forecastability_report,
    _raw_causal_history_matrix,
    _runtime_stabilize_points,
)
from .synthetic import SyntheticConfig, generate_mock_csv
from .training import (
    EXTERNAL_RUL_EVALUATION_FILENAMES,
    EXTERNAL_RUL_EVALUATION_SHA256,
    assert_not_external_rul_evaluation_input,
    build_training_matrix,
)


V2_4_CORPUS_VERSION = "rul_v2_4_physically_role_separated_corpus_v1"
V2_4_GENERATOR_VERSION = "causal_hazard_accumulated_stress_v2_2_unchanged"
V2_4_PROTOCOL_VERSION = "nested_batch_crossfit_state_selective_protocol_v1"
ROLE_ORDER = ("fit", "calibration", "acceptance")
SELECTOR_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SELECTOR_LABEL_MAX_ABS_ERROR_HOURS = 12.0
SELECTOR_LABEL_MAX_NEIGHBOR_DISPERSION_HOURS = 18.0
SELECTOR_HYSTERESIS_MARGIN = 0.05
SUPPORT_SIGNAL_NAMES = (
    "nearest_standardized_distance",
    "neighbor_target_std_hours",
    "neighbor_target_mad_hours",
    "neighbor_batch_diversity",
)


@dataclass(frozen=True)
class RULV24AcceptanceCriteria:
    max_active_mae_hours: float = 7.0
    max_active_macro_mae_hours: float = 8.0
    min_active_interval_coverage: float = 0.80
    min_active_macro_lifecycle_coverage: float = 0.75
    min_active_monotonicity: float = 0.95
    min_active_region_availability: float = 0.95
    max_abs_horizon_bias_hours: float = 8.0
    max_unidentifiable_active_rate: float = 0.05
    max_false_issuance_risk: float = 0.15
    minimum_active_row_fraction: float = 0.20
    minimum_active_lifecycle_fraction: float = 0.75
    minimum_median_lifecycle_active_fraction: float = 0.10
    minimum_p10_lifecycle_active_fraction: float = 0.01
    minimum_lifecycles: int = 8
    minimum_batches: int = 4
    max_single_lifecycle_active_share: float = 0.25
    max_single_batch_active_share: float = 0.40
    max_state_oscillation_rate: float = 0.10
    width_ratio_limit: float = 1.15
    width_absolute_increase_limit_hours: float = 6.0
    width_exception_macro_mae_improvement: float = 0.10
    max_runtime_p95_ms_per_row: float = 50.0


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: str | Path, rows: list[dict[str, str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _consumed_seed_evidence() -> tuple[set[int], list[dict[str, Any]]]:
    seeds = {12001, 14001}
    evidence: list[dict[str, Any]] = []
    for version in ("rul_v2_2", "rul_v2_3"):
        manifest_path = Path("output") / version / "corpus_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for batch in manifest.get("batches", []):
            if batch.get("seed") is not None:
                seeds.add(int(batch["seed"]))
            evidence.append({
                "source_version": version,
                "batch_id": batch.get("batch_id") or batch.get("batch"),
                "seed": batch.get("seed"),
                "role": batch.get("role"),
                "sha256": batch.get("source_sha256") or batch.get("sha256"),
            })
    return seeds, evidence


def generate_rul_v2_4_development_corpus(
    data_dir: str | Path,
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    *,
    role_seeds: dict[str, Iterable[int]],
    lifecycles_per_batch: int = 12,
    machines: int = 6,
    cadence_seconds: int = 600,
    line_sel: str = "LINE_1",
) -> dict[str, Any]:
    """Generate physically separate, immutable fit/calibration/acceptance evidence."""
    normalized = {role: [int(seed) for seed in role_seeds.get(role, [])] for role in ROLE_ORDER}
    if any(len(normalized[role]) < 4 for role in ROLE_ORDER):
        raise ValueError("v2.4 requires at least four batches for every immutable role")
    all_seeds = [seed for role in ROLE_ORDER for seed in normalized[role]]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("A generation seed may belong to only one v2.4 evidence role")
    consumed_seeds, historical = _consumed_seed_evidence()
    overlap = sorted(consumed_seeds & set(all_seeds))
    if overlap:
        raise ValueError(f"Protected or consumed seeds cannot be reused in v2.4: {overlap}")

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    batch_dir = root / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    lifecycle_seen: set[str] = set()
    machine_seen_by_role: dict[str, set[str]] = defaultdict(set)
    batches: list[dict[str, Any]] = []
    role_files: dict[str, dict[str, Any]] = {}
    next_id = 1
    ordinal = 0
    generator_config = {
        "lifecycles_per_batch": lifecycles_per_batch,
        "machines": machines,
        "cadence_seconds": cadence_seconds,
        "duration_min_hours": 54.0,
        "duration_max_hours": 156.0,
        "causal_hazard_coupling": True,
    }
    for role in ROLE_ORDER:
        role_rows: list[dict[str, str]] = []
        prefix = {"fit": "v24fit", "calibration": "v24cal", "acceptance": "v24acc"}[role]
        for index, seed in enumerate(normalized[role], start=1):
            ordinal += 1
            batch_id = f"{prefix}_b{index:03d}_s{seed}"
            batch_path = batch_dir / f"{batch_id}.csv"
            generate_mock_csv(
                batch_path,
                SyntheticConfig(
                    lifecycles=lifecycles_per_batch,
                    cadence_seconds=cadence_seconds,
                    seed=seed,
                    machines=machines,
                    line_sel=line_sel,
                    start_time=datetime(2055, 1, 1, tzinfo=timezone.utc) + timedelta(days=370 * ordinal),
                    identity_prefix=batch_id,
                    duration_min_hours=54.0,
                    duration_max_hours=156.0,
                    causal_hazard_coupling=True,
                ),
            )
            rows = _read_rows(batch_path)
            for row in rows:
                if row["lifecycle_id"] in lifecycle_seen:
                    raise RuntimeError(f"Lifecycle identity collision: {row['lifecycle_id']}")
                row["id"] = str(next_id)
                row["generation_seed"] = str(seed)
                row["batch_id"] = batch_id
                row["development_role"] = role
                role_rows.append(row)
                machine_seen_by_role[role].add(row["machine_id"])
                next_id += 1
            lifecycle_seen.update({row["lifecycle_id"] for row in rows})
            batches.append({
                "role": role,
                "batch_id": batch_id,
                "seed": seed,
                "generator_version": V2_4_GENERATOR_VERSION,
                "generator_configuration_hash": _json_hash(generator_config),
                "source_file": str(batch_path.resolve()),
                "source_sha256": _sha256(batch_path),
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                **_batch_summary(rows),
            })
        role_path = root / f"{role}.csv"
        assert_not_external_rul_evaluation_input(role_path)
        _write_rows(role_path, role_rows)
        role_files[role] = {
            "path": str(role_path.resolve()),
            "sha256": _sha256(role_path),
            "rows": len(role_rows),
            "batches": len(normalized[role]),
            "lifecycles": len({row["lifecycle_id"] for row in role_rows}),
            "machines": len(machine_seen_by_role[role]),
        }
    if any(machine_seen_by_role[left] & machine_seen_by_role[right] for left in ROLE_ORDER for right in ROLE_ORDER if left < right):
        raise RuntimeError("Machine identities must be globally role-disjoint")

    manifest = {
        "corpus_version": V2_4_CORPUS_VERSION,
        "protocol_version": V2_4_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "role_assignments_permanent": True,
        "physical_role_files": True,
        "protected_external_datasets_used": False,
        "generator_version": V2_4_GENERATOR_VERSION,
        "generator_changed_from_v2_3": False,
        "role_files": role_files,
        "batches": batches,
        "provenance_fields_excluded_from_features": [
            "generation_seed", "batch_id", "development_role", "latent_damage_score",
            "fault_mode", "lifecycle_progress", "machine_id", "lifecycle_id",
        ],
        "code_identifier": {
            "synthetic_py_sha256": _sha256(Path(__file__).with_name("synthetic.py")),
            "rul_ml_py_sha256": _sha256(Path(__file__).with_name("rul_ml.py")),
            "rul_features_v2_4_py_sha256": _sha256(Path(__file__).with_name("rul_features_v2_4.py")),
        },
    }
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = {
        "registry_version": "permanent_consumed_rul_evidence_registry_v2",
        "role_reassignment_permitted": False,
        "protected_external": [
            {"filename": name, "sha256": digest, "role": "permanent_external"}
            for name, digest in (
                ("rul_test_12001.csv", "657a6ea261a626b2107d5d32d47ec66e84f91ee079437c17df8d9dbb7291db7b"),
                ("rul_holdout_14001.csv", "685a3867b369b237b2c57783df9b18b74f66cd7ceccee896f304d875bb2103be"),
            )
        ],
        "historical_consumed_batches": historical,
        "v2_4_role_locked_batches": [
            {"batch_id": row["batch_id"], "seed": row["seed"], "role": row["role"], "sha256": row["source_sha256"]}
            for row in batches
        ],
        "v2_4_acceptance_opened": False,
        "v2_4_acceptance_consumed": False,
    }
    registry_output = Path(consumed_registry_path)
    registry_output.parent.mkdir(parents=True, exist_ok=True)
    registry_output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return manifest


def _build_v2_4_extra_matrix(csv_path: str | Path, sensor_config: SensorConfig) -> np.ndarray:
    rows = _read_rows(csv_path)
    builder = V24CausalFeatureBuilder(
        baseline_hours=sensor_config.baseline_hours,
        baseline_minimum_points=sensor_config.baseline_min_points,
        baseline_std_floor_fraction=sensor_config.baseline_std_floor_fraction,
    )
    previous_lifecycle: dict[str, str] = {}
    output: list[list[float]] = []
    for row in rows:
        machine_key = f"{row['line_sel']}::{row['machine_id']}"
        lifecycle = row["lifecycle_id"]
        reset = previous_lifecycle.get(machine_key) not in {None, lifecycle}
        previous_lifecycle[machine_key] = lifecycle
        values = {sensor: float(row[sensor]) for sensor in ("vrms", "arms", "apeak", "crest", "temp")}
        features = builder.update(
            machine_key,
            datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
            values,
            state_reset=reset,
        )
        output.append([features[name] for name in V2_4_CAUSAL_FEATURE_NAMES])
    return np.asarray(output, dtype=float)


def _role_arrays(
    csv_path: str | Path,
    frozen_bundle: dict[str, Any],
    sensor_config: SensorConfig,
) -> dict[str, Any]:
    X, groups, _progress, latent, _fault, timestamps, base_names = build_training_matrix(csv_path, sensor_config)
    if list(base_names) != list(frozen_bundle["feature_names"]):
        raise RuntimeError("Frozen status feature contract changed in v2.4")
    scores, statuses = causal_status_signals(frozen_bundle["model"], X, groups, timestamps)
    X_v23, v23_names = build_rul_matrix(X, base_names, groups, timestamps, scores, statuses)
    X_extra = _build_v2_4_extra_matrix(csv_path, sensor_config)
    if len(X_extra) != len(X_v23):
        raise RuntimeError("v2.4 causal feature rows do not align with the frozen runtime matrix")
    X_full = np.column_stack([X_v23, X_extra])
    full_names = list(v23_names) + list(V2_4_CAUSAL_FEATURE_NAMES)
    validate_rul_feature_names(full_names)
    rows = _read_rows(csv_path)
    return {
        "X_v23": X_v23,
        "X_full": X_full,
        "X_raw": _raw_causal_history_matrix(csv_path)[0],
        "v23_names": list(v23_names),
        "full_names": full_names,
        "groups": groups,
        "batches": np.asarray([row["batch_id"] for row in rows], dtype=object),
        "machines": np.asarray([row["machine_id"] for row in rows], dtype=object),
        "timestamps": timestamps,
        "scores": scores,
        "statuses": statuses,
        "targets": {
            name: derive_time_to_onset_targets(groups, timestamps, latent, threshold=threshold)
            for name, threshold in RUL_TARGET_THRESHOLDS.items()
        },
    }


def _feature_set_indices(names: list[str], v23_count: int) -> dict[str, np.ndarray]:
    index = {name: pos for pos, name in enumerate(names)}
    rate = [index[name] for name in V2_4_RATE_FEATURES]
    stress_core_names = [
        name for name in V2_4_STRESS_FEATURES
        if "cumulative" in name or "elevated_duration" in name
    ]
    stress_core = [index[name] for name in stress_core_names]
    base = list(range(v23_count))
    return {
        "v2_3_feature_set": np.asarray(base, dtype=int),
        "v2_3_plus_degradation_rate": np.asarray(base + rate, dtype=int),
        "v2_3_plus_cumulative_stress": np.asarray(base + stress_core, dtype=int),
        "v2_3_plus_rate_and_stress": np.asarray(base + rate + stress_core, dtype=int),
        "v2_3_plus_all_approved_v2_4_causal": np.arange(len(names), dtype=int),
    }


def _batch_folds(indices: np.ndarray, batches: np.ndarray, *, folds: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = sorted(set(batches[indices].tolist()))
    if len(unique) < folds:
        raise ValueError(f"Nested v2.4 cross-fitting requires at least {folds} independent fit batches")
    splitter = GroupKFold(n_splits=folds)
    placeholder = np.zeros((len(indices), 1), dtype=float)
    return [
        (indices[train_local], indices[hold_local])
        for train_local, hold_local in splitter.split(placeholder, groups=batches[indices])
    ]


def _lifecycle_balanced_fit_sample(
    target: np.ndarray,
    groups: np.ndarray,
    *,
    maximum_rows_per_lifecycle: int = 120,
) -> np.ndarray:
    """Deterministically retain chronological coverage from every fit lifecycle."""
    chosen: list[int] = []
    for lifecycle in sorted(set(groups.tolist())):
        local = np.flatnonzero((groups == lifecycle) & np.isfinite(target) & (target > 0.0))
        if len(local) > maximum_rows_per_lifecycle:
            local = local[np.unique(np.linspace(0, len(local) - 1, maximum_rows_per_lifecycle, dtype=int))]
        chosen.extend(local.tolist())
    return np.asarray(sorted(chosen), dtype=int)


def _fit_support_bank(
    X: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    indices: np.ndarray,
    *,
    seed: int,
    maximum_reference_rows: int = 3000,
) -> dict[str, Any]:
    finite = np.asarray([idx for idx in indices if np.isfinite(target[idx]) and target[idx] > 0.0], dtype=int)
    if len(finite) < 20:
        raise ValueError("Insufficient training reference rows for v2.4 support diagnostics")
    rng = np.random.default_rng(seed)
    if len(finite) > maximum_reference_rows:
        lifecycles = sorted(set(groups[finite].tolist()))
        per_lifecycle = max(1, maximum_reference_rows // len(lifecycles))
        chosen: list[int] = []
        for lifecycle in lifecycles:
            local = finite[groups[finite] == lifecycle]
            take = min(per_lifecycle, len(local))
            chosen.extend(rng.choice(local, size=take, replace=False).tolist())
        reference = np.asarray(sorted(chosen[:maximum_reference_rows]), dtype=int)
    else:
        reference = finite
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    imputer.fit(X[finite])
    transformed = scaler.fit_transform(imputer.transform(X[reference]))
    return {
        "imputer": imputer,
        "scaler": scaler,
        "reference_X": transformed,
        "reference_targets": target[reference],
        "reference_batches": batches[reference],
        "reference_lifecycles": groups[reference],
        "reference_rows": int(len(reference)),
        "neighbors": 10,
        "source_roles": ["fit"],
    }


def _fit_classifier(X: np.ndarray, labels: np.ndarray, *, seed: int):
    labels = np.asarray(labels, dtype=int)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    transformed = imputer.fit_transform(X)
    if len(set(labels.tolist())) < 2:
        model = DummyClassifier(strategy="constant", constant=int(labels[0])).fit(transformed, labels)
        return _ImputedClassifier(imputer, model)
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_leaf=20,
        max_features=0.7,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    model.fit(transformed, labels)
    return _ImputedClassifier(imputer, model)


class _ImputedClassifier:
    def __init__(self, imputer: SimpleImputer, model: Any) -> None:
        self.imputer = imputer
        self.model = model
        self.classes_ = model.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.imputer.transform(np.asarray(X, dtype=float)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.imputer.transform(np.asarray(X, dtype=float)))


def _positive_probability(model: Any, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(X), dtype=float)


def _horizon_mask(target: np.ndarray, bucket: str) -> np.ndarray:
    return np.asarray([
        np.isfinite(value) and value > 0.0 and diagnostic_horizon_bucket(value) == bucket
        for value in target
    ], dtype=bool)


def _apply_hysteresis(
    probability: np.ndarray,
    groups: np.ndarray,
    timestamps: np.ndarray,
    *,
    activation_threshold: float,
    deactivation_threshold: float,
) -> np.ndarray:
    active = np.zeros(len(probability), dtype=bool)
    for lifecycle in sorted(set(groups.tolist())):
        positions = np.flatnonzero(groups == lifecycle)
        positions = positions[np.argsort(np.asarray([timestamps[pos].timestamp() for pos in positions]))]
        state = False
        for pos in positions:
            threshold = deactivation_threshold if state else activation_threshold
            state = bool(np.isfinite(probability[pos]) and probability[pos] >= threshold)
            active[pos] = state
    return active


def _lifecycle_macro_rate(numerator: np.ndarray, denominator: np.ndarray, groups: np.ndarray) -> float | None:
    values: list[float] = []
    for lifecycle in sorted(set(groups[denominator].tolist())):
        local = denominator & (groups == lifecycle)
        if np.any(local):
            values.append(float(np.sum(numerator & local) / np.sum(local)))
    return float(np.mean(values)) if values else None


def _selection_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    active: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    eligible_override: np.ndarray | None = None,
) -> dict[str, Any]:
    eligible = np.isfinite(truth) & (truth > 0.0)
    if eligible_override is None:
        eligible &= np.isfinite(prediction)
    else:
        eligible &= np.asarray(eligible_override, dtype=bool)
    issued = eligible & active & np.isfinite(prediction)
    errors = prediction[issued] - truth[issued]
    active_lifecycles = sorted(set(groups[issued].tolist()))
    active_batches = sorted(set(batches[issued].tolist()))
    lifecycle_mae = [
        float(np.mean(np.abs(prediction[issued & (groups == lifecycle)] - truth[issued & (groups == lifecycle)])))
        for lifecycle in active_lifecycles
    ]
    unreliable = eligible & ~labels
    false_issued = issued & unreliable
    forecastable = eligible & labels
    withheld = eligible & ~active
    per_lifecycle_active = [
        float(np.sum(issued & (groups == lifecycle)) / np.sum(eligible & (groups == lifecycle)))
        for lifecycle in sorted(set(groups[eligible].tolist()))
    ]
    life_counts = {lifecycle: int(np.sum(issued & (groups == lifecycle))) for lifecycle in active_lifecycles}
    batch_counts = {batch: int(np.sum(issued & (batches == batch))) for batch in active_batches}
    horizons: dict[str, Any] = {}
    for bucket in RUL_DIAGNOSTIC_BUCKETS:
        region = eligible & _horizon_mask(truth, bucket)
        region_active = region & active
        horizons[bucket] = {
            **_error_metrics(truth, prediction, groups, batches, region_active),
            "eligible_rows": int(np.sum(region)),
            "active_rows": int(np.sum(region_active)),
            "active_rate": float(np.sum(region_active) / np.sum(region)) if np.sum(region) else None,
        }
    return {
        "eligible_rows": int(np.sum(eligible)),
        "active_rows": int(np.sum(issued)),
        "active_row_fraction": float(np.sum(issued) / np.sum(eligible)) if np.sum(eligible) else None,
        "active_lifecycles": len(active_lifecycles),
        "eligible_lifecycles": len(set(groups[eligible].tolist())),
        "active_lifecycle_fraction": (
            len(active_lifecycles) / len(set(groups[eligible].tolist())) if np.sum(eligible) else None
        ),
        "active_batches": len(active_batches),
        "eligible_batches": len(set(batches[eligible].tolist())),
        "active_batch_fraction": (
            len(active_batches) / len(set(batches[eligible].tolist())) if np.sum(eligible) else None
        ),
        "mae_hours": float(np.mean(np.abs(errors))) if len(errors) else None,
        "macro_lifecycle_mae_hours": float(np.mean(lifecycle_mae)) if lifecycle_mae else None,
        "mean_signed_error_hours": float(np.mean(errors)) if len(errors) else None,
        "p90_abs_error_hours": float(np.quantile(np.abs(errors), 0.90)) if len(errors) else None,
        "false_issuance_risk": float(np.sum(false_issued) / np.sum(issued)) if np.sum(issued) else None,
        "false_activation_population_rate": (
            float(np.sum(false_issued) / np.sum(eligible)) if np.sum(eligible) else None
        ),
        "false_withholding_rate": (
            float(np.sum(withheld & forecastable) / np.sum(forecastable)) if np.sum(forecastable) else None
        ),
        "macro_false_issuance_risk": _lifecycle_macro_rate(false_issued, issued, groups),
        "median_lifecycle_active_fraction": (
            float(np.median(per_lifecycle_active)) if per_lifecycle_active else None
        ),
        "p10_lifecycle_active_fraction": (
            float(np.quantile(per_lifecycle_active, 0.10)) if per_lifecycle_active else None
        ),
        "maximum_single_lifecycle_active_share": (
            max(life_counts.values()) / np.sum(issued) if life_counts and np.sum(issued) else None
        ),
        "maximum_single_batch_active_share": (
            max(batch_counts.values()) / np.sum(issued) if batch_counts and np.sum(issued) else None
        ),
        "horizons": horizons,
    }


def _feature_ablation(
    X_full: np.ndarray,
    feature_sets: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    timestamps: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    finite = _lifecycle_balanced_fit_sample(target, groups)
    folds = _batch_folds(finite, batches)
    candidates: dict[str, Any] = {}
    for offset, (name, columns) in enumerate(feature_sets.items()):
        prediction = np.full(len(target), np.nan, dtype=float)
        fold_rows: list[dict[str, Any]] = []
        for fold, (train_idx, hold_idx) in enumerate(folds):
            weights = rul_sample_weights_v2_4(groups, target, train_idx)
            model = _fit_point_model(
                X_full[train_idx][:, columns],
                target[train_idx],
                weights,
                seed + offset * 100 + fold,
                estimator_name="extra_trees_v2_1",
                shared_v2_2_preprocessing=True,
            )
            raw = np.asarray(model.predict(X_full[hold_idx][:, columns]), dtype=float)
            prediction[hold_idx] = _runtime_stabilize_points(raw, groups, timestamps, hold_idx)
            fold_rows.append({
                "fold": fold,
                "training_batches": sorted(set(batches[train_idx].tolist())),
                "holdout_batches": sorted(set(batches[hold_idx].tolist())),
                "batch_overlap": sorted(set(batches[train_idx].tolist()) & set(batches[hold_idx].tolist())),
            })
        mask = np.isfinite(prediction) & np.isfinite(target) & (target > 0.0)
        overall = _error_metrics(target, prediction, groups, batches, mask)
        horizons = {
            bucket: _error_metrics(target, prediction, groups, batches, mask & _horizon_mask(target, bucket))
            for bucket in RUL_DIAGNOSTIC_BUCKETS
        }
        bias_48 = horizons["48_72h"].get("mean_signed_error_hours")
        bias_72 = horizons["gt_72h"].get("mean_signed_error_hours")
        selection_score = (
            float(overall["macro_lifecycle_mae_hours"] or math.inf)
            + 0.10 * abs(float(bias_48 or 0.0))
            + 0.10 * abs(float(bias_72 or 0.0))
        )
        candidates[name] = {
            "feature_count": int(len(columns)),
            "overall": overall,
            "horizons": horizons,
            "folds": fold_rows,
            "selection_score": selection_score,
            "predictions": prediction,
        }
    selected = min(candidates, key=lambda name: (candidates[name]["selection_score"], name))
    report_candidates = {
        name: {key: value for key, value in row.items() if key != "predictions"}
        for name, row in candidates.items()
    }
    return {
        "policy": "same four batch folds, fixed Extra Trees, fixed v2.4 flat weights; minimum predeclared bias-aware score",
        "selected_feature_set": selected,
        "candidates": report_candidates,
        "prediction_arrays": {name: row["predictions"] for name, row in candidates.items()},
    }


def _nested_selector_and_candidates(
    X_full: np.ndarray,
    X_v23: np.ndarray,
    selected_columns: np.ndarray,
    selector_base_indices: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    timestamps: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    finite = _lifecycle_balanced_fit_sample(target, groups)
    outer_folds = _batch_folds(finite, batches)
    anchor_raw = np.full(len(target), np.nan, dtype=float)
    alternate_raw = np.full(len(target), np.nan, dtype=float)
    support = np.full((len(target), len(SUPPORT_SIGNAL_NAMES)), np.nan, dtype=float)
    labels = np.zeros(len(target), dtype=bool)
    selector_probability = np.full(len(target), np.nan, dtype=float)
    candidate_names = (
        "global_v2_3_extra_trees",
        "v2_4_features_extra_trees",
        "forecastability_weighted_extra_trees",
        "forecastability_weighted_log1p_extra_trees",
    )
    candidate_predictions = {
        name: np.full(len(target), np.nan, dtype=float) for name in candidate_names
    }
    fold_report: list[dict[str, Any]] = []

    for outer_fold, (outer_train, outer_hold) in enumerate(outer_folds):
        inner_anchor = np.full(len(target), np.nan, dtype=float)
        inner_alternate = np.full(len(target), np.nan, dtype=float)
        inner_support = np.full((len(target), len(SUPPORT_SIGNAL_NAMES)), np.nan, dtype=float)
        inner_labels = np.zeros(len(target), dtype=bool)
        inner_folds = _batch_folds(outer_train, batches, folds=3)
        inner_report: list[dict[str, Any]] = []
        for inner_fold, (inner_train, inner_hold) in enumerate(inner_folds):
            anchor_weights = rul_sample_weights_v2_2(groups, target, inner_train)
            anchor_model = _fit_point_model(
                X_v23[inner_train], target[inner_train], anchor_weights,
                seed + outer_fold * 1000 + inner_fold,
                estimator_name="extra_trees_v2_1", shared_v2_2_preprocessing=True,
            )
            alternate_weights = rul_sample_weights_v2_4(groups, target, inner_train)
            alternate_model = _fit_point_model(
                X_full[inner_train][:, selected_columns], target[inner_train], alternate_weights,
                seed + 100 + outer_fold * 1000 + inner_fold,
                estimator_name="hist_gradient_boosting_v2_2", shared_v2_2_preprocessing=True,
            )
            bank = _fit_support_bank(
                X_full[:, selected_columns], target, groups, batches, inner_train,
                seed=seed + 200 + outer_fold * 1000 + inner_fold,
            )
            inner_anchor[inner_hold] = anchor_model.predict(X_v23[inner_hold])
            inner_alternate[inner_hold] = alternate_model.predict(X_full[inner_hold][:, selected_columns])
            inner_support[inner_hold] = v2_4_support_signals(
                bank, X_full[inner_hold][:, selected_columns]
            )
            stabilized = _runtime_stabilize_points(
                inner_anchor[inner_hold], groups, timestamps, inner_hold
            )
            inner_labels[inner_hold] = (
                np.abs(stabilized - target[inner_hold]) <= SELECTOR_LABEL_MAX_ABS_ERROR_HOURS
            ) & (
                inner_support[inner_hold, 2] <= SELECTOR_LABEL_MAX_NEIGHBOR_DISPERSION_HOURS
            )
            inner_report.append({
                "inner_fold": inner_fold,
                "training_batches": sorted(set(batches[inner_train].tolist())),
                "holdout_batches": sorted(set(batches[inner_hold].tolist())),
                "batch_overlap": sorted(set(batches[inner_train].tolist()) & set(batches[inner_hold].tolist())),
                "support_reference_roles": ["outer_training_fit_only"],
            })
        if not np.all(np.isfinite(inner_anchor[outer_train])):
            raise RuntimeError("Nested selector labels did not cover every outer-training row")
        inner_selector_X = v2_4_selector_matrix(
            X_full[outer_train], inner_anchor[outer_train], inner_alternate[outer_train],
            inner_support[outer_train], selector_base_indices,
        )
        selector_model = _fit_classifier(
            inner_selector_X, inner_labels[outer_train], seed=seed + 5000 + outer_fold
        )

        anchor_weights = rul_sample_weights_v2_2(groups, target, outer_train)
        outer_anchor_model = _fit_point_model(
            X_v23[outer_train], target[outer_train], anchor_weights,
            seed + 6000 + outer_fold,
            estimator_name="extra_trees_v2_1", shared_v2_2_preprocessing=True,
        )
        selected_weights = rul_sample_weights_v2_4(groups, target, outer_train)
        outer_alternate_model = _fit_point_model(
            X_full[outer_train][:, selected_columns], target[outer_train], selected_weights,
            seed + 6100 + outer_fold,
            estimator_name="hist_gradient_boosting_v2_2", shared_v2_2_preprocessing=True,
        )
        outer_bank = _fit_support_bank(
            X_full[:, selected_columns], target, groups, batches, outer_train,
            seed=seed + 6200 + outer_fold,
        )
        anchor_raw[outer_hold] = outer_anchor_model.predict(X_v23[outer_hold])
        alternate_raw[outer_hold] = outer_alternate_model.predict(
            X_full[outer_hold][:, selected_columns]
        )
        support[outer_hold] = v2_4_support_signals(
            outer_bank, X_full[outer_hold][:, selected_columns]
        )
        stabilized_anchor = _runtime_stabilize_points(
            anchor_raw[outer_hold], groups, timestamps, outer_hold
        )
        labels[outer_hold] = (
            np.abs(stabilized_anchor - target[outer_hold]) <= SELECTOR_LABEL_MAX_ABS_ERROR_HOURS
        ) & (support[outer_hold, 2] <= SELECTOR_LABEL_MAX_NEIGHBOR_DISPERSION_HOURS)
        outer_selector_X = v2_4_selector_matrix(
            X_full[outer_hold], anchor_raw[outer_hold], alternate_raw[outer_hold],
            support[outer_hold], selector_base_indices,
        )
        selector_probability[outer_hold] = _positive_probability(selector_model, outer_selector_X)
        candidate_predictions["global_v2_3_extra_trees"][outer_hold] = stabilized_anchor

        multipliers = np.where(inner_labels[outer_train], 1.0, 0.35)
        candidate_specs = (
            ("v2_4_features_extra_trees", "extra_trees_v2_1", np.ones(len(outer_train))),
            ("forecastability_weighted_extra_trees", "extra_trees_v2_1", multipliers),
            ("forecastability_weighted_log1p_extra_trees", "extra_trees_log1p_v2_4", multipliers),
        )
        for offset, (name, estimator, local_multiplier) in enumerate(candidate_specs):
            weights = selected_weights * local_multiplier
            weights *= len(weights) / max(float(np.sum(weights)), 1e-12)
            model = _fit_point_model(
                X_full[outer_train][:, selected_columns], target[outer_train], weights,
                seed + 7000 + outer_fold * 10 + offset,
                estimator_name=estimator, shared_v2_2_preprocessing=True,
            )
            raw = model.predict(X_full[outer_hold][:, selected_columns])
            candidate_predictions[name][outer_hold] = _runtime_stabilize_points(
                raw, groups, timestamps, outer_hold
            )
        fold_report.append({
            "outer_fold": outer_fold,
            "training_batches": sorted(set(batches[outer_train].tolist())),
            "holdout_batches": sorted(set(batches[outer_hold].tolist())),
            "batch_overlap": sorted(set(batches[outer_train].tolist()) & set(batches[outer_hold].tolist())),
            "inner_crossfit": inner_report,
            "selector_training_labels_from_outer_training_inner_oof_only": True,
        })
    return {
        "finite_indices": finite,
        "anchor_raw": anchor_raw,
        "alternate_raw": alternate_raw,
        "support": support,
        "labels": labels,
        "selector_probability": selector_probability,
        "candidate_predictions": candidate_predictions,
        "folds": fold_report,
    }


def _choose_selector_threshold(
    nested: dict[str, Any],
    target: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    timestamps: np.ndarray,
    criteria: RULV24AcceptanceCriteria,
) -> dict[str, Any]:
    anchor = nested["candidate_predictions"]["global_v2_3_extra_trees"]
    candidates: list[dict[str, Any]] = []
    for threshold in SELECTOR_THRESHOLD_GRID:
        active = _apply_hysteresis(
            nested["selector_probability"], groups, timestamps,
            activation_threshold=threshold,
            deactivation_threshold=max(0.0, threshold - SELECTOR_HYSTERESIS_MARGIN),
        )
        metrics = _selection_metrics(
            target, anchor, active, nested["labels"], groups, batches
        )
        horizon_48 = metrics["horizons"]["48_72h"]
        horizon_72 = metrics["horizons"]["gt_72h"]
        far_supported = (
            horizon_72["active_rows"] == 0
            or (
                horizon_72["lifecycles"] >= criteria.minimum_lifecycles
                and horizon_72["batches"] >= criteria.minimum_batches
                and abs(float(horizon_72["mean_signed_error_hours"])) <= criteria.max_abs_horizon_bias_hours
            )
        )
        passed = bool(
            metrics["mae_hours"] is not None
            and metrics["mae_hours"] <= criteria.max_active_mae_hours
            and metrics["macro_lifecycle_mae_hours"] is not None
            and metrics["macro_lifecycle_mae_hours"] <= criteria.max_active_macro_mae_hours
            and metrics["false_issuance_risk"] is not None
            and metrics["false_issuance_risk"] <= criteria.max_false_issuance_risk
            and metrics["active_row_fraction"] >= criteria.minimum_active_row_fraction
            and metrics["active_lifecycle_fraction"] >= criteria.minimum_active_lifecycle_fraction
            and horizon_48["active_rate"] is not None
            and horizon_48["active_rate"] <= criteria.max_unidentifiable_active_rate
            and far_supported
        )
        violation = (
            max(0.0, float(metrics["mae_hours"] or math.inf) - criteria.max_active_mae_hours)
            + max(0.0, float(metrics["macro_lifecycle_mae_hours"] or math.inf) - criteria.max_active_macro_mae_hours)
            + 20.0 * max(0.0, float(metrics["false_issuance_risk"] or 1.0) - criteria.max_false_issuance_risk)
            + 20.0 * max(0.0, criteria.minimum_active_row_fraction - float(metrics["active_row_fraction"] or 0.0))
            + 20.0 * max(0.0, float(horizon_48["active_rate"] or 0.0) - criteria.max_unidentifiable_active_rate)
            + (0.0 if far_supported else 10.0)
        )
        candidates.append({
            "activation_threshold": threshold,
            "deactivation_threshold": max(0.0, threshold - SELECTOR_HYSTERESIS_MARGIN),
            "metrics": metrics,
            "passes_fit_oof_requirements": passed,
            "normalized_violation_score": violation,
        })
    passing = [row for row in candidates if row["passes_fit_oof_requirements"]]
    selected = (
        max(passing, key=lambda row: (row["metrics"]["active_row_fraction"], -row["activation_threshold"]))
        if passing else min(candidates, key=lambda row: (row["normalized_violation_score"], -row["activation_threshold"]))
    )
    return {
        "threshold_grid_predeclared": list(SELECTOR_THRESHOLD_GRID),
        "selection_rule": "maximum fit-only nested-OOF breadth among fully passing thresholds; otherwise minimum normalized violation",
        "selected_activation_threshold": selected["activation_threshold"],
        "selected_deactivation_threshold": selected["deactivation_threshold"],
        "selected_passed_fit_oof_requirements": selected["passes_fit_oof_requirements"],
        "candidates": candidates,
        "acceptance_data_used": False,
    }


def _choose_rul_candidate(
    nested: dict[str, Any],
    active: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    criteria: RULV24AcceptanceCriteria,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, prediction in nested["candidate_predictions"].items():
        metrics = _selection_metrics(
            target, prediction, active, nested["labels"], groups, batches
        )
        horizon_pass = True
        decision_bias: dict[str, Any] = {}
        for bucket, horizon in metrics["horizons"].items():
            if horizon["active_rows"] == 0:
                decision_bias[bucket] = {"status": "WITHHELD", "bias": None}
            elif horizon["lifecycles"] < criteria.minimum_lifecycles or horizon["batches"] < criteria.minimum_batches:
                horizon_pass = False
                decision_bias[bucket] = {"status": "UNSUPPORTED", "bias": horizon["mean_signed_error_hours"]}
            else:
                passed = abs(float(horizon["mean_signed_error_hours"])) <= criteria.max_abs_horizon_bias_hours
                horizon_pass &= passed
                decision_bias[bucket] = {
                    "status": "PASS" if passed else "FAIL",
                    "bias": horizon["mean_signed_error_hours"],
                }
        passed = bool(
            metrics["mae_hours"] is not None
            and metrics["mae_hours"] <= criteria.max_active_mae_hours
            and metrics["macro_lifecycle_mae_hours"] is not None
            and metrics["macro_lifecycle_mae_hours"] <= criteria.max_active_macro_mae_hours
            and horizon_pass
        )
        score = (
            float(metrics["macro_lifecycle_mae_hours"] or math.inf)
            + 0.10 * sum(
                abs(float(item["bias"]))
                for item in decision_bias.values() if item["bias"] is not None
            )
        )
        rows[name] = {
            "metrics": metrics,
            "decision_horizon_bias": decision_bias,
            "passes_fit_oof_requirements": passed,
            "selection_score": score,
        }
    passing = [name for name, row in rows.items() if row["passes_fit_oof_requirements"]]
    selected = min(passing or list(rows), key=lambda name: (rows[name]["selection_score"], name))
    return {
        "candidate_set_predeclared": list(rows),
        "selection_rule": "minimum bias-aware lifecycle-macro score among fully passing candidates; otherwise minimum score and remain development-ineligible",
        "selected_candidate": selected,
        "selected_passed_fit_oof_requirements": rows[selected]["passes_fit_oof_requirements"],
        "candidates": rows,
        "acceptance_data_used": False,
    }


def _classifier_metrics(
    probability: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    batches: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    finite = np.isfinite(probability)
    prediction = probability >= threshold
    tp = int(np.sum(finite & prediction & labels))
    fp = int(np.sum(finite & prediction & ~labels))
    fn = int(np.sum(finite & ~prediction & labels))
    lifecycle_precision: list[float] = []
    lifecycle_recall: list[float] = []
    for lifecycle in sorted(set(groups[finite].tolist())):
        local = finite & (groups == lifecycle)
        local_tp = int(np.sum(local & prediction & labels))
        local_fp = int(np.sum(local & prediction & ~labels))
        local_fn = int(np.sum(local & ~prediction & labels))
        if local_tp + local_fp:
            lifecycle_precision.append(local_tp / (local_tp + local_fp))
        if local_tp + local_fn:
            lifecycle_recall.append(local_tp / (local_tp + local_fn))
    batch_metrics = {}
    for batch in sorted(set(batches[finite].tolist())):
        local = finite & (batches == batch)
        local_tp = int(np.sum(local & prediction & labels))
        local_fp = int(np.sum(local & prediction & ~labels))
        local_fn = int(np.sum(local & ~prediction & labels))
        batch_metrics[str(batch)] = {
            "precision": local_tp / (local_tp + local_fp) if local_tp + local_fp else None,
            "recall": local_tp / (local_tp + local_fn) if local_tp + local_fn else None,
            "activation_rate": float(np.mean(prediction[local])) if np.any(local) else None,
        }
    return {
        "row_precision": tp / (tp + fp) if tp + fp else None,
        "row_recall": tp / (tp + fn) if tp + fn else None,
        "lifecycle_macro_precision": float(np.mean(lifecycle_precision)) if lifecycle_precision else None,
        "lifecycle_macro_recall": float(np.mean(lifecycle_recall)) if lifecycle_recall else None,
        "false_activation_definition": "issued and fixed-anchor nested-OOF reliability label is false",
        "false_issuance_risk": fp / (tp + fp) if tp + fp else None,
        "false_withholding_rate": fn / (tp + fn) if tp + fn else None,
        "activation_rate": float(np.mean(prediction[finite])) if np.any(finite) else None,
        "batch_metrics": batch_metrics,
    }


def _historical_v2_3_false_activation_report() -> dict[str, Any]:
    acceptance_path = Path("output/rul_v2_3/development_acceptance_report.json")
    training_path = Path("output/rul_v2_3/training_report.json")
    if not acceptance_path.exists() or not training_path.exists():
        return {"supported": False, "reason": "v2.3 reports unavailable"}
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    critical = acceptance["runtime_active_prediction_metrics"]["critical"]
    return {
        "supported": True,
        "historical_diagnostic_only": True,
        "may_not_select_v2_4_features_or_thresholds": True,
        "v2_3_model_sha256": acceptance.get("model_sha256"),
        "v2_3_activation_contract": training.get("activation_policy"),
        "48_72h": critical["forecastability_contract_by_true_horizon"].get("48_72h"),
        "gt_72h": critical["horizon_diagnostics"].get("gt_72h"),
        "root_cause": (
            "v2.3 used only degradation score/status thresholding; it had no fold-local support, "
            "neighbor dispersion, model disagreement, or learned reliability decision"
        ),
        "true_rul_and_error_usage": "audit only; never supplied to the v2.4 runtime selector",
    }


def _weighting_report(
    groups: np.ndarray,
    batches: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    finite = np.asarray([idx for idx in indices if np.isfinite(target[idx]) and target[idx] > 0.0], dtype=int)
    policies = {
        "v2_3_weighting": rul_sample_weights_v2_2(groups, target, finite),
        "v2_4_lifecycle_flat_moderate_near_failure": rul_sample_weights_v2_4(groups, target, finite),
    }
    output: dict[str, Any] = {}
    for policy, weights in policies.items():
        buckets: dict[str, Any] = {}
        for bucket in RUL_DIAGNOSTIC_BUCKETS:
            mask = _horizon_mask(target[finite], bucket)
            buckets[bucket] = {
                "rows": int(np.sum(mask)),
                "lifecycles": len(set(groups[finite][mask].tolist())),
                "batches": len(set(batches[finite][mask].tolist())),
                "raw_weight_mass": float(np.sum(mask)),
                "lifecycle_normalized_weight_mass": float(np.sum(weights[mask])),
                "percentage_optimization_weight": float(np.sum(weights[mask]) / np.sum(weights)),
            }
        output[policy] = buckets
    return {
        "policies": output,
        "selected_policy": "v2_4_lifecycle_flat_moderate_near_failure",
        "maximum_near_failure_multiplier": 1.5,
        "acceptance_data_used": False,
    }


def train_rul_v2_4_candidate(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    frozen_status_model_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    freeze_manifest_path: str | Path,
    sensor_config: SensorConfig,
    *,
    seed: int = 2400,
    criteria: RULV24AcceptanceCriteria | None = None,
) -> dict[str, Any]:
    """Select, fit, and calibrate v2.4 without opening acceptance evidence."""
    criteria = criteria or RULV24AcceptanceCriteria()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    registry = json.loads(Path(consumed_registry_path).read_text(encoding="utf-8-sig"))
    if manifest.get("corpus_version") != V2_4_CORPUS_VERSION:
        raise ValueError("Unsupported v2.4 corpus manifest")
    if manifest.get("physical_role_files") is not True:
        raise ValueError("v2.4 requires physically separate evidence-role files")
    if registry.get("role_reassignment_permitted") is not False:
        raise ValueError("Consumed evidence registry must fail closed on role reassignment")
    if registry.get("v2_4_acceptance_opened") is not False:
        raise RuntimeError("Acceptance evidence was already opened; start a new versioned iteration")
    fit_path = Path(manifest["role_files"]["fit"]["path"])
    calibration_path = Path(manifest["role_files"]["calibration"]["path"])
    for role, path in (("fit", fit_path), ("calibration", calibration_path)):
        assert_not_external_rul_evaluation_input(path)
        if _sha256(path) != manifest["role_files"][role]["sha256"]:
            raise ValueError(f"v2.4 {role} evidence does not match its frozen hash")

    frozen_path = Path(frozen_status_model_path)
    frozen_sha = _sha256(frozen_path)
    frozen_bundle = joblib.load(frozen_path)
    if frozen_bundle.get("sensor_contract") != sensor_contract(sensor_config):
        raise RuntimeError("Frozen manufacturer/status model sensor contract mismatch")
    frozen_object_hash = joblib.hash(frozen_bundle["model"])

    fit = _role_arrays(fit_path, frozen_bundle, sensor_config)
    calibration = _role_arrays(calibration_path, frozen_bundle, sensor_config)
    if fit["full_names"] != calibration["full_names"]:
        raise RuntimeError("Fit/calibration runtime feature contracts differ")
    full_names = fit["full_names"]
    v23_count = len(fit["v23_names"])
    feature_sets = _feature_set_indices(full_names, v23_count)
    feature_ablation = _feature_ablation(
        fit["X_full"], feature_sets, fit["targets"]["critical"], fit["groups"],
        fit["batches"], fit["timestamps"], seed=seed + 100,
    )
    selected_feature_set = feature_ablation["selected_feature_set"]
    selected_columns = feature_sets[selected_feature_set]
    selector_base_indices = np.asarray([
        index for index, name in enumerate(full_names)
        if name in {
            "degradation_score", "predicted_status_severity", "elapsed_lifecycle_hours",
            "score_rate_1h", "score_rate_6h", "score_rate_12h",
            "score_delta_6h", "score_delta_12h", "score_std_6h", "score_std_12h",
        } or name.startswith("v24_")
    ], dtype=int)
    nested = _nested_selector_and_candidates(
        fit["X_full"], fit["X_v23"], selected_columns, selector_base_indices,
        fit["targets"]["critical"], fit["groups"], fit["batches"], fit["timestamps"],
        seed=seed + 1000,
    )
    threshold_report = _choose_selector_threshold(
        nested, fit["targets"]["critical"], fit["groups"], fit["batches"],
        fit["timestamps"], criteria,
    )
    activation_threshold = float(threshold_report["selected_activation_threshold"])
    deactivation_threshold = float(threshold_report["selected_deactivation_threshold"])
    fit_active = _apply_hysteresis(
        nested["selector_probability"], fit["groups"], fit["timestamps"],
        activation_threshold=activation_threshold,
        deactivation_threshold=deactivation_threshold,
    )
    candidate_report = _choose_rul_candidate(
        nested, fit_active, fit["targets"]["critical"], fit["groups"], fit["batches"], criteria,
    )
    selected_candidate = candidate_report["selected_candidate"]

    selector_training_X = v2_4_selector_matrix(
        fit["X_full"], nested["anchor_raw"], nested["alternate_raw"], nested["support"],
        selector_base_indices,
    )
    selector_model = _fit_classifier(
        selector_training_X[nested["finite_indices"]],
        nested["labels"][nested["finite_indices"]],
        seed=seed + 2000,
    )
    critical_fit_indices = nested["finite_indices"]
    anchor_weights = rul_sample_weights_v2_2(
        fit["groups"], fit["targets"]["critical"], critical_fit_indices
    )
    final_anchor = _fit_point_model(
        fit["X_v23"][critical_fit_indices], fit["targets"]["critical"][critical_fit_indices],
        anchor_weights, seed + 2100,
        estimator_name="extra_trees_v2_1", shared_v2_2_preprocessing=True,
    )
    alternate_weights = rul_sample_weights_v2_4(
        fit["groups"], fit["targets"]["critical"], critical_fit_indices
    )
    final_alternate = _fit_point_model(
        fit["X_full"][critical_fit_indices][:, selected_columns],
        fit["targets"]["critical"][critical_fit_indices], alternate_weights, seed + 2200,
        estimator_name="hist_gradient_boosting_v2_2", shared_v2_2_preprocessing=True,
    )
    final_support = _fit_support_bank(
        fit["X_full"][:, selected_columns], fit["targets"]["critical"], fit["groups"],
        fit["batches"], critical_fit_indices, seed=seed + 2300,
    )

    calibration_anchor = final_anchor.predict(calibration["X_v23"])
    calibration_alternate = final_alternate.predict(calibration["X_full"][:, selected_columns])
    calibration_support = v2_4_support_signals(
        final_support, calibration["X_full"][:, selected_columns]
    )
    calibration_selector_X = v2_4_selector_matrix(
        calibration["X_full"], calibration_anchor, calibration_alternate,
        calibration_support, selector_base_indices,
    )
    calibration_probability = _positive_probability(selector_model, calibration_selector_X)
    calibration_active = _apply_hysteresis(
        calibration_probability, calibration["groups"], calibration["timestamps"],
        activation_threshold=activation_threshold,
        deactivation_threshold=deactivation_threshold,
    )

    X_combined = np.vstack([fit["X_full"], calibration["X_full"]])
    groups_combined = np.concatenate([fit["groups"], calibration["groups"]])
    batches_combined = np.concatenate([fit["batches"], calibration["batches"]])
    timestamps_combined = np.concatenate([fit["timestamps"], calibration["timestamps"]])
    fit_idx = np.arange(len(fit["groups"]), dtype=int)
    calibration_idx = np.arange(len(fit["groups"]), len(groups_combined), dtype=int)
    active_calibration_idx = calibration_idx[calibration_active]
    if len(active_calibration_idx) < 20:
        raise RuntimeError("Frozen selector produced insufficient active calibration rows")
    active_combined = np.zeros(len(groups_combined), dtype=bool)
    active_combined[active_calibration_idx] = True
    selector_label_multipliers = np.ones(len(groups_combined), dtype=float)
    selector_label_multipliers[nested["finite_indices"]] = np.where(
        nested["labels"][nested["finite_indices"]], 1.0, 0.35
    )

    candidate_configuration = {
        "global_v2_3_extra_trees": (feature_sets["v2_3_feature_set"], "extra_trees_v2_1", "v2_2", False),
        "v2_4_features_extra_trees": (selected_columns, "extra_trees_v2_1", "v2_4_flat", False),
        "forecastability_weighted_extra_trees": (selected_columns, "extra_trees_v2_1", "v2_4_flat", True),
        "forecastability_weighted_log1p_extra_trees": (selected_columns, "extra_trees_log1p_v2_4", "v2_4_flat", True),
    }
    model_columns, estimator_name, weighting_version, active_weighted = candidate_configuration[selected_candidate]
    targets_combined = {
        name: np.concatenate([fit["targets"][name], calibration["targets"][name]])
        for name in RUL_TARGET_THRESHOLDS
    }
    target_artifacts: dict[str, dict[str, Any]] = {}
    training_metrics: dict[str, Any] = {}
    for offset, name in enumerate(("warning", "critical")):
        artifact, metrics = fit_quantile_rul_target(
            X_combined[:, model_columns], targets_combined[name], groups_combined,
            fit_idx, active_calibration_idx,
            seed=seed + 3000 + offset * 100,
            target_name=name,
            estimator_name=estimator_name,
            weighting_version=weighting_version,
            row_weight_multipliers=selector_label_multipliers if active_weighted else None,
            oof_split_groups=batches_combined,
            desired_interval_coverage=criteria.min_active_interval_coverage,
            desired_macro_lifecycle_coverage=criteria.min_active_macro_lifecycle_coverage,
        )
        artifact["model"] = ColumnSubsetRegressor(artifact["model"], model_columns)
        target_artifacts[name] = artifact
        training_metrics[name] = metrics
    temporal = calibrate_rul_targets_temporally(
        target_artifacts, X_combined, targets_combined, groups_combined, timestamps_combined,
        calibration_idx,
        target_coverage=criteria.min_active_interval_coverage,
        target_macro_lifecycle_coverage=criteria.min_active_macro_lifecycle_coverage,
        activation_mask=active_combined,
    )
    for name in target_artifacts:
        training_metrics[name] = {**training_metrics[name], **temporal[name]}

    identifiability_indices = _lifecycle_balanced_fit_sample(
        fit["targets"]["critical"], fit["groups"]
    )
    raw_oof = _batch_grouped_runtime_oof(
        fit["X_full"][:, selected_columns], fit["X_raw"], fit["targets"]["critical"],
        fit["groups"], fit["batches"], fit["timestamps"], fit["scores"],
        identifiability_indices, seed=seed + 4000,
    )
    raw_neighbors = _cross_batch_neighbor_audit(
        fit["X_raw"], fit["targets"]["critical"], fit["groups"], fit["batches"],
        identifiability_indices, seed=seed + 4100, representation="raw_causal_history",
    )
    derived_neighbors = _cross_batch_neighbor_audit(
        fit["X_full"][:, selected_columns], fit["targets"]["critical"], fit["groups"],
        fit["batches"], identifiability_indices, seed=seed + 4200,
        representation="selected_v2_4_runtime_features",
    )
    identifiability = _forecastability_report(
        raw_oof, fit["targets"]["critical"], fit["groups"], fit["batches"],
        raw_neighbors, derived_neighbors, RULV23AcceptanceCriteria(), seed=seed + 4300,
    )

    classifier_report = {
        "protocol_version": V2_4_PROTOCOL_VERSION,
        "anchor_label_model": "frozen-candidate global v2.3 Extra Trees feature/weight contract",
        "label_definition": {
            "anchor_nested_oof_absolute_error_hours_max": SELECTOR_LABEL_MAX_ABS_ERROR_HOURS,
            "neighbor_target_mad_hours_max": SELECTOR_LABEL_MAX_NEIGHBOR_DISPERSION_HOURS,
            "cohort_bias_is_not_a_row_label": True,
        },
        "nested_outer_inner_folds": nested["folds"],
        "classifier_metrics": _classifier_metrics(
            nested["selector_probability"], nested["labels"], fit["groups"], fit["batches"],
            activation_threshold,
        ),
        "risk_versus_coverage": threshold_report,
        "selected_feature_set": selected_feature_set,
        "selector_base_feature_names": [full_names[index] for index in selector_base_indices],
        "support_signal_names": list(SUPPORT_SIGNAL_NAMES),
        "acceptance_data_used": False,
    }
    weighting = _weighting_report(
        fit["groups"], fit["batches"], fit["targets"]["critical"],
        np.arange(len(fit["groups"])),
    )

    selector_contract = {
        "model": selector_model,
        "anchor_model": final_anchor,
        "alternate_model": final_alternate,
        "support_bank": final_support,
        "base_feature_indices": selector_base_indices.tolist(),
        "anchor_feature_indices": list(range(v23_count)),
        "alternate_feature_indices": selected_columns.tolist(),
        "support_feature_indices": selected_columns.tolist(),
    }
    rul_artifact = {
        "version": RUL_MODEL_VERSION_V2_4,
        "feature_names": full_names,
        "base_feature_names": list(frozen_bundle["feature_names"]),
        "targets": target_artifacts,
        "forecastability_contract": {
            "mode": "nested_crossfit_learned_reliability_selector_v1",
            "selector": selector_contract,
            "activation_threshold": activation_threshold,
            "deactivation_threshold": deactivation_threshold,
            "low_confidence_score": 0.5 * activation_threshold,
            "normal_withheld_behavior": "clear point, interval, RUL actionability, trigger and recommendation evidence",
            "warning_behavior": "current WARNING endpoint is zero; future CRITICAL may still be withheld",
            "critical_behavior": "current CRITICAL remains immediate with both endpoints zero",
            "selection_source": "fit-only nested batch OOF",
            "calibration_population": "frozen selector active rows only",
            "probability_forecasts_implemented": False,
        },
        "selected_feature_set": selected_feature_set,
        "selected_rul_candidate": selected_candidate,
        "fit_batches": sorted(set(fit["batches"].tolist())),
        "calibration_batches": sorted(set(calibration["batches"].tolist())),
        "acceptance_batches": [],
        "min_history_hours": 0.5,
        "min_points": 6,
        "max_upward_jump_floor_hours": 0.5,
        "max_upward_jump_per_elapsed_hour": 1.5,
        "upward_revision_cooldown_hours": 6.0,
        "upward_revision_trigger_hours": 4.0,
        "forbidden_runtime_fields": [
            "true_rul", "true_error", "future_timestamp", "latent_damage_score", "fault_mode",
            "generation_seed", "batch_id", "development_role", "lifecycle_progress",
        ],
        "external_evaluation_training_guard": {
            "protected_filenames": sorted(EXTERNAL_RUL_EVALUATION_FILENAMES),
            "protected_sha256": sorted(EXTERNAL_RUL_EVALUATION_SHA256),
        },
        "development_gates_passed": False,
        "synthetic_only_not_production_ready": True,
    }
    bundle = {
        **frozen_bundle,
        "rul_model": rul_artifact,
        "metadata": {
            **dict(frozen_bundle.get("metadata") or {}),
            "rul_model_version": RUL_MODEL_VERSION_V2_4,
            "rul_training_mode": V2_4_PROTOCOL_VERSION,
            "frozen_status_source_sha256": frozen_sha,
            "rul_development_gates_passed": False,
            "production_ready": False,
        },
    }
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output)
    model_sha = _sha256(model_output)
    persisted = joblib.load(model_output)
    persisted_status_hash = joblib.hash(persisted["model"])
    if _sha256(frozen_path) != frozen_sha or persisted_status_hash != frozen_object_hash:
        raise RuntimeError("Frozen status model changed during v2.4 candidate persistence")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "version": RUL_MODEL_VERSION_V2_4,
        "protocol_version": V2_4_PROTOCOL_VERSION,
        "development_only": True,
        "production_ready": False,
        "protected_external_datasets_used": False,
        "model_sha256": model_sha,
        "frozen_status_model_sha256": frozen_sha,
        "frozen_status_object_hash_before": frozen_object_hash,
        "frozen_status_object_hash_after_persistence": persisted_status_hash,
        "acceptance_file_opened": False,
        "sealed_holdout_authorized": False,
    }
    reports = {
        "false_activation_diagnostics.json": {**common, **_historical_v2_3_false_activation_report()},
        "feature_ablation_report.json": {
            **common,
            **{key: value for key, value in feature_ablation.items() if key != "prediction_arrays"},
        },
        "forecastability_classifier_report.json": {**common, **classifier_report},
        "active_rul_candidate_comparison.json": {**common, **candidate_report},
        "identifiability_report.json": {**common, **identifiability},
        "horizon_diagnostics.json": {
            **common,
            "fit_nested_oof_selected_candidate": candidate_report["candidates"][selected_candidate]["metrics"]["horizons"],
        },
        "weighting_diagnostics.json": {**common, **weighting},
        "runtime_contract_report.json": {
            **common,
            "forecastability_contract": {
                key: value for key, value in rul_artifact["forecastability_contract"].items()
                if key != "selector"
            },
            "calibration_active_rows": int(np.sum(calibration_active)),
            "calibration_withheld_rows": int(len(calibration_active) - np.sum(calibration_active)),
            "manufacturer_safety_precedence_unchanged": True,
            "warning_future_critical_may_be_withheld": True,
        },
        "training_report.json": {
            **common,
            "manifest_sha256": _sha256(manifest_path),
            "fit_sha256": manifest["role_files"]["fit"]["sha256"],
            "calibration_sha256": manifest["role_files"]["calibration"]["sha256"],
            "acceptance_sha256_frozen_but_not_opened": manifest["role_files"]["acceptance"]["sha256"],
            "selected_feature_set": selected_feature_set,
            "selected_rul_candidate": selected_candidate,
            "fit_selector_passed": threshold_report["selected_passed_fit_oof_requirements"],
            "fit_rul_candidate_passed": candidate_report["selected_passed_fit_oof_requirements"],
            "training_and_active_calibration": training_metrics,
            "development_gates_passed": False,
            "development_acceptance_status": "PENDING_UNOPENED",
            "warning": "Synthetic development candidate only; acceptance remains unopened and sealed holdout remains prohibited.",
        },
    }
    for name, payload in reports.items():
        (output / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    freeze_manifest = {
        "freeze_version": "rul_v2_4_preacceptance_freeze_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_output.resolve()),
        "model_sha256": model_sha,
        "status_model_sha256": frozen_sha,
        "status_object_hash": frozen_object_hash,
        "manifest_sha256": _sha256(manifest_path),
        "fit_sha256": manifest["role_files"]["fit"]["sha256"],
        "calibration_sha256": manifest["role_files"]["calibration"]["sha256"],
        "acceptance_sha256": manifest["role_files"]["acceptance"]["sha256"],
        "feature_contract_sha256": _json_hash(full_names),
        "selected_feature_set": selected_feature_set,
        "selected_rul_candidate": selected_candidate,
        "activation_threshold": activation_threshold,
        "deactivation_threshold": deactivation_threshold,
        "criteria": asdict(criteria),
        "runtime_code_sha256": {
            "rul_v2_4.py": _sha256(Path(__file__)),
            "rul_ml.py": _sha256(Path(__file__).with_name("rul_ml.py")),
            "rul_features_v2_4.py": _sha256(Path(__file__).with_name("rul_features_v2_4.py")),
        },
        "acceptance_file_opened_before_freeze": False,
        "acceptance_results_used_for_selection": False,
    }
    freeze_output = Path(freeze_manifest_path)
    freeze_output.parent.mkdir(parents=True, exist_ok=True)
    freeze_output.write_text(json.dumps(freeze_manifest, indent=2), encoding="utf-8")
    reports["training_report.json"]["freeze_manifest_sha256"] = _sha256(freeze_output)
    (output / "training_report.json").write_text(
        json.dumps(reports["training_report.json"], indent=2), encoding="utf-8"
    )
    return reports["training_report.json"]


def _optional_float(value: Any) -> float | None:
    if value in {None, "", "None"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _interval_metrics_from_runtime_rows(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    eligible = [row for row in rows if (_optional_float(row.get("true_hours_to_critical")) or 0.0) > 0.0]
    intervals = [
        row for row in eligible
        if _optional_float(row.get("estimated_hours_to_critical")) is not None
        and _optional_float(row.get("critical_lower_hours")) is not None
        and _optional_float(row.get("critical_upper_hours")) is not None
    ]
    hits = [
        row for row in intervals
        if float(row["critical_lower_hours"])
        <= float(row["true_hours_to_critical"])
        <= float(row["critical_upper_hours"])
    ]
    widths = [float(row["critical_upper_hours"]) - float(row["critical_lower_hours"]) for row in intervals]
    lifecycle_coverage: list[float] = []
    for lifecycle in sorted({row["lifecycle_id"] for row in eligible}):
        local = [row for row in intervals if row["lifecycle_id"] == lifecycle]
        local_hits = [row for row in hits if row["lifecycle_id"] == lifecycle]
        if local:
            lifecycle_coverage.append(len(local_hits) / len(local))
    return {
        "eligible_rows": len(eligible),
        "interval_rows": len(intervals),
        "coverage": len(hits) / len(intervals) if intervals else None,
        "macro_lifecycle_coverage": float(np.mean(lifecycle_coverage)) if lifecycle_coverage else None,
        "mean_width_hours": float(np.mean(widths)) if widths else None,
        "median_width_hours": float(np.median(widths)) if widths else None,
    }


def _state_stability(rows: list[dict[str, str]]) -> dict[str, Any]:
    transitions: dict[str, int] = defaultdict(int)
    flips = pairs = 0
    dwell_lengths: list[int] = []
    for lifecycle in sorted({row["lifecycle_id"] for row in rows}):
        local = [row for row in rows if row["lifecycle_id"] == lifecycle]
        states = [row.get("rul_forecastability_state") or "RUL_UNAVAILABLE" for row in local]
        if not states:
            continue
        dwell = 1
        for left, right in zip(states, states[1:]):
            pairs += 1
            if left != right:
                transitions[f"{left}->{right}"] += 1
                flips += 1
                dwell_lengths.append(dwell)
                dwell = 1
            else:
                dwell += 1
        dwell_lengths.append(dwell)
    return {
        "transition_counts": dict(sorted(transitions.items())),
        "oscillation_rate": flips / pairs if pairs else 0.0,
        "transition_pairs": pairs,
        "median_dwell_rows": float(np.median(dwell_lengths)) if dwell_lengths else None,
        "active_unavailable_flip_count": sum(
            count for name, count in transitions.items()
            if "RUL_ACTIVE" in name and "RUL_UNAVAILABLE" in name
        ),
        "hysteresis_applied": True,
    }


def evaluate_rul_v2_4_development_acceptance(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    freeze_manifest_path: str | Path,
    model_path: str | Path,
    frozen_v2_3_model_path: str | Path,
    output_dir: str | Path,
    sensor_config: SensorConfig,
    *,
    criteria: RULV24AcceptanceCriteria | None = None,
) -> dict[str, Any]:
    """Open fresh acceptance exactly once after verifying the preacceptance freeze."""
    criteria = criteria or RULV24AcceptanceCriteria()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    registry_path = Path(consumed_registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    freeze = json.loads(Path(freeze_manifest_path).read_text(encoding="utf-8"))
    model_path = Path(model_path)
    acceptance_path = Path(manifest["role_files"]["acceptance"]["path"])
    assert_not_external_rul_evaluation_input(acceptance_path)
    if registry.get("v2_4_acceptance_opened") is not False:
        raise RuntimeError("v2.4 acceptance is already consumed; a rerun requires a new versioned iteration")
    integrity = {
        "model": _sha256(model_path) == freeze.get("model_sha256"),
        "manifest": _sha256(manifest_path) == freeze.get("manifest_sha256"),
        "acceptance": _sha256(acceptance_path) == freeze.get("acceptance_sha256"),
        "fit": manifest["role_files"]["fit"]["sha256"] == freeze.get("fit_sha256"),
        "calibration": manifest["role_files"]["calibration"]["sha256"] == freeze.get("calibration_sha256"),
        "rul_v2_4_code": _sha256(Path(__file__)) == freeze.get("runtime_code_sha256", {}).get("rul_v2_4.py"),
        "rul_ml_code": _sha256(Path(__file__).with_name("rul_ml.py")) == freeze.get("runtime_code_sha256", {}).get("rul_ml.py"),
        "rul_features_v2_4_code": _sha256(Path(__file__).with_name("rul_features_v2_4.py")) == freeze.get("runtime_code_sha256", {}).get("rul_features_v2_4.py"),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"Preacceptance freeze integrity failed: {integrity}")

    registry["v2_4_acceptance_opened"] = True
    registry["v2_4_acceptance_opened_at"] = datetime.now(timezone.utc).isoformat()
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime_report_path = output / "runtime_acceptance_replay_report.json"
    runtime_predictions_path = output / "runtime_acceptance_predictions.csv"
    started = time.perf_counter()
    runtime_report = evaluate_rul(
        model_path,
        acceptance_path,
        runtime_report_path,
        runtime_predictions_path,
        sensor_config,
    )
    replay_seconds = time.perf_counter() - started
    runtime_rows = _read_rows(runtime_predictions_path)

    bundle = joblib.load(model_path)
    artifact = bundle["rul_model"]
    status_hash = joblib.hash(bundle["model"])
    acceptance = _role_arrays(acceptance_path, bundle, sensor_config)
    if acceptance["full_names"] != list(artifact["feature_names"]):
        raise RuntimeError("Acceptance runtime feature contract differs from frozen v2.4 artifact")
    selector = artifact["forecastability_contract"]["selector"]
    anchor_columns = np.asarray(selector["anchor_feature_indices"], dtype=int)
    alternate_columns = np.asarray(selector["alternate_feature_indices"], dtype=int)
    support_columns = np.asarray(selector["support_feature_indices"], dtype=int)
    anchor_raw = selector["anchor_model"].predict(acceptance["X_full"][:, anchor_columns])
    alternate_raw = selector["alternate_model"].predict(acceptance["X_full"][:, alternate_columns])
    support = v2_4_support_signals(
        selector["support_bank"], acceptance["X_full"][:, support_columns]
    )
    selector_X = v2_4_selector_matrix(
        acceptance["X_full"], anchor_raw, alternate_raw, support,
        np.asarray(selector["base_feature_indices"], dtype=int),
    )
    selector_probability = _positive_probability(selector["model"], selector_X)
    stabilized_anchor = _runtime_stabilize_points(
        anchor_raw, acceptance["groups"], acceptance["timestamps"],
        np.arange(len(anchor_raw), dtype=int),
    )
    reliability_labels = (
        np.abs(stabilized_anchor - acceptance["targets"]["critical"])
        <= SELECTOR_LABEL_MAX_ABS_ERROR_HOURS
    ) & (support[:, 2] <= SELECTOR_LABEL_MAX_NEIGHBOR_DISPERSION_HOURS)

    runtime_by_id = {int(row["id"]): row for row in runtime_rows}
    source_rows = _read_rows(acceptance_path)
    if len(source_rows) != len(runtime_rows):
        raise RuntimeError("Runtime acceptance replay did not preserve every source row")
    prediction = np.full(len(source_rows), np.nan, dtype=float)
    active = np.zeros(len(source_rows), dtype=bool)
    forecastability_states: list[str] = []
    for pos, source in enumerate(source_rows):
        row = runtime_by_id[int(source["id"])]
        value = _optional_float(row.get("estimated_hours_to_critical"))
        if value is not None and acceptance["targets"]["critical"][pos] > 0.0:
            prediction[pos] = value
            active[pos] = True
        forecastability_states.append(row.get("rul_forecastability_state") or "RUL_UNAVAILABLE")
    metrics = _selection_metrics(
        acceptance["targets"]["critical"], prediction, active, reliability_labels,
        acceptance["groups"], acceptance["batches"],
        eligible_override=np.ones(len(prediction), dtype=bool),
    )
    intervals = _interval_metrics_from_runtime_rows(runtime_rows)
    stability = _state_stability(runtime_rows)
    latencies = [
        value for row in runtime_rows
        if (value := _optional_float(row.get("rul_inference_latency_ms"))) is not None
    ]
    latency = {
        "mean_ms_per_row": float(np.mean(latencies)) if latencies else None,
        "p95_ms_per_row": float(np.quantile(latencies, 0.95)) if latencies else None,
        "replay_wall_seconds": replay_seconds,
        "rows": len(runtime_rows),
    }
    contract_active = _apply_hysteresis(
        selector_probability, acceptance["groups"], acceptance["timestamps"],
        activation_threshold=float(artifact["forecastability_contract"]["activation_threshold"]),
        deactivation_threshold=float(artifact["forecastability_contract"]["deactivation_threshold"]),
    )
    active_region_availability = (
        float(np.sum(active & contract_active) / np.sum(contract_active))
        if np.sum(contract_active) else None
    )

    baseline_report_path = output / "frozen_v2_3_same_acceptance_report.json"
    baseline_predictions_path = output / "frozen_v2_3_same_acceptance_predictions.csv"
    baseline_report = evaluate_rul(
        frozen_v2_3_model_path,
        acceptance_path,
        baseline_report_path,
        baseline_predictions_path,
        sensor_config,
    )
    baseline_pre = baseline_report["pre_onset_critical_rul"]
    baseline_width = baseline_pre.get("mean_interval_width_hours")
    baseline_macro_mae = np.mean([
        row["critical_mae_hours"]
        for row in baseline_report.get("per_lifecycle", {}).values()
        if row.get("critical_mae_hours") is not None
    ]) if baseline_report.get("per_lifecycle") else None
    width_limit = (
        min(criteria.width_ratio_limit * baseline_width, baseline_width + criteria.width_absolute_increase_limit_hours)
        if baseline_width is not None else None
    )
    macro_improvement = (
        1.0 - float(metrics["macro_lifecycle_mae_hours"]) / float(baseline_macro_mae)
        if metrics["macro_lifecycle_mae_hours"] is not None and baseline_macro_mae
        else None
    )
    width_pass = bool(
        intervals["mean_width_hours"] is not None
        and width_limit is not None
        and (
            intervals["mean_width_hours"] <= width_limit
            or (macro_improvement is not None and macro_improvement >= criteria.width_exception_macro_mae_improvement)
        )
    )

    checks: dict[str, Any] = {}
    def record(name: str, observed: Any, threshold: str, passed: bool | None, reason: str | None = None) -> None:
        normalized = None if passed is None else bool(passed)
        checks[name] = {
            "observed": observed,
            "threshold": threshold,
            "status": "PASS" if normalized is True else "FAIL" if normalized is False else "UNSUPPORTED",
            "reason": reason,
        }

    frozen_training = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    record(
        "fit_selector_preacceptance",
        frozen_training.get("fit_selector_passed"),
        "must be true before acceptance",
        frozen_training.get("fit_selector_passed") is True,
    )
    record(
        "fit_rul_candidate_preacceptance",
        frozen_training.get("fit_rul_candidate_passed"),
        "must be true before acceptance",
        frozen_training.get("fit_rul_candidate_passed") is True,
        "unsupported active fit-OOF horizon blocks authorization" if frozen_training.get("fit_rul_candidate_passed") is not True else None,
    )
    record("active_overall_mae", metrics["mae_hours"], "<= 7 h", metrics["mae_hours"] is not None and metrics["mae_hours"] <= criteria.max_active_mae_hours)
    record("active_macro_lifecycle_mae", metrics["macro_lifecycle_mae_hours"], "<= 8 h", metrics["macro_lifecycle_mae_hours"] is not None and metrics["macro_lifecycle_mae_hours"] <= criteria.max_active_macro_mae_hours)
    record("active_interval_coverage", intervals["coverage"], ">= 0.80", intervals["coverage"] is not None and intervals["coverage"] >= criteria.min_active_interval_coverage)
    record("active_macro_lifecycle_coverage", intervals["macro_lifecycle_coverage"], ">= 0.75", intervals["macro_lifecycle_coverage"] is not None and intervals["macro_lifecycle_coverage"] >= criteria.min_active_macro_lifecycle_coverage)
    monotonicity = runtime_report["pre_onset_critical_rul"].get("monotonicity")
    record("active_monotonicity", monotonicity, ">= 0.95", monotonicity is not None and monotonicity >= criteria.min_active_monotonicity)
    record("active_region_availability", active_region_availability, ">= 0.95", active_region_availability is not None and active_region_availability >= criteria.min_active_region_availability)
    record("false_issuance_risk", metrics["false_issuance_risk"], "<= 0.15", metrics["false_issuance_risk"] is not None and metrics["false_issuance_risk"] <= criteria.max_false_issuance_risk)
    record("minimum_active_row_fraction", metrics["active_row_fraction"], ">= 0.20", metrics["active_row_fraction"] is not None and metrics["active_row_fraction"] >= criteria.minimum_active_row_fraction)
    record("minimum_active_lifecycle_fraction", metrics["active_lifecycle_fraction"], ">= 0.75", metrics["active_lifecycle_fraction"] is not None and metrics["active_lifecycle_fraction"] >= criteria.minimum_active_lifecycle_fraction)
    record("median_lifecycle_active_fraction", metrics["median_lifecycle_active_fraction"], ">= 0.10", metrics["median_lifecycle_active_fraction"] is not None and metrics["median_lifecycle_active_fraction"] >= criteria.minimum_median_lifecycle_active_fraction)
    record("p10_lifecycle_active_fraction", metrics["p10_lifecycle_active_fraction"], ">= 0.01", metrics["p10_lifecycle_active_fraction"] is not None and metrics["p10_lifecycle_active_fraction"] >= criteria.minimum_p10_lifecycle_active_fraction)
    record("lifecycle_concentration", metrics["maximum_single_lifecycle_active_share"], "<= 0.25", metrics["maximum_single_lifecycle_active_share"] is not None and metrics["maximum_single_lifecycle_active_share"] <= criteria.max_single_lifecycle_active_share)
    record("batch_concentration", metrics["maximum_single_batch_active_share"], "<= 0.40", metrics["maximum_single_batch_active_share"] is not None and metrics["maximum_single_batch_active_share"] <= criteria.max_single_batch_active_share)
    horizon_48 = metrics["horizons"]["48_72h"]
    record("48_72h_unidentifiable_active_rate", horizon_48["active_rate"], "<= 0.05", horizon_48["active_rate"] is not None and horizon_48["active_rate"] <= criteria.max_unidentifiable_active_rate)
    for bucket, horizon in metrics["horizons"].items():
        if horizon["active_rows"] == 0:
            record(f"{bucket}_active_bias", None, "withheld or abs bias <= 8 h with support", True, "exact RUL fully withheld")
        elif horizon["lifecycles"] < criteria.minimum_lifecycles or horizon["batches"] < criteria.minimum_batches:
            record(f"{bucket}_active_bias", horizon["mean_signed_error_hours"], ">=8 lifecycles, >=4 batches, abs bias <=8 h", None, "active horizon unsupported")
        else:
            record(f"{bucket}_active_bias", horizon["mean_signed_error_hours"], "abs bias <= 8 h", abs(float(horizon["mean_signed_error_hours"])) <= criteria.max_abs_horizon_bias_hours)
    record("forecastability_state_stability", stability["oscillation_rate"], "<= 0.10", stability["oscillation_rate"] <= criteria.max_state_oscillation_rate)
    record("runtime_p95_latency", latency["p95_ms_per_row"], "<= 50 ms", latency["p95_ms_per_row"] is not None and latency["p95_ms_per_row"] <= criteria.max_runtime_p95_ms_per_row)
    record("interval_width_guard", {"new": intervals["mean_width_hours"], "baseline": baseline_width, "limit": width_limit, "macro_mae_improvement": macro_improvement}, "<= min(1.15*baseline, baseline+6h) unless macro MAE improves >=10%", width_pass)

    statuses = [row["status"] for row in checks.values()]
    all_passed = bool(statuses and all(status == "PASS" for status in statuses))
    common = {
        "version": RUL_MODEL_VERSION_V2_4,
        "protocol_version": V2_4_PROTOCOL_VERSION,
        "development_only": True,
        "production_ready": False,
        "protected_external_datasets_used": False,
        "model_sha256": _sha256(model_path),
        "frozen_status_model_sha256": freeze["status_model_sha256"],
        "frozen_status_object_hash": status_hash,
        "preacceptance_integrity": integrity,
    }
    acceptance_report = {
        **common,
        "criteria": asdict(criteria),
        "runtime_active_prediction_metrics": metrics,
        "interval_metrics": intervals,
        "active_region_availability": active_region_availability,
        "forecastability_classifier_acceptance": _classifier_metrics(
            selector_probability, reliability_labels, acceptance["groups"], acceptance["batches"],
            float(artifact["forecastability_contract"]["activation_threshold"]),
        ),
        "state_stability": stability,
        "runtime_performance": latency,
        "frozen_v2_3_same_acceptance_cohort": {
            "model_sha256": _sha256(frozen_v2_3_model_path),
            "mean_interval_width_hours": baseline_width,
            "macro_lifecycle_mae_hours": baseline_macro_mae,
        },
        "checks": checks,
        "all_required_gates_passed": all_passed,
        "sealed_holdout_authorized": all_passed,
        "sealed_holdout_generated_or_evaluated": False,
    }
    (output / "development_acceptance_report.json").write_text(
        json.dumps(acceptance_report, indent=2, default=str), encoding="utf-8"
    )
    (output / "horizon_diagnostics.json").write_text(json.dumps({
        **common,
        "acceptance_runtime": metrics["horizons"],
        "combined_gt48": _error_metrics(
            acceptance["targets"]["critical"], prediction, acceptance["groups"], acceptance["batches"],
            active & (acceptance["targets"]["critical"] > 48.0),
        ),
    }, indent=2), encoding="utf-8")
    runtime_contract_path = output / "runtime_contract_report.json"
    prior_contract = json.loads(runtime_contract_path.read_text(encoding="utf-8"))
    runtime_contract_path.write_text(json.dumps({
        **prior_contract,
        "acceptance_runtime_metrics": metrics,
        "forecastability_state_stability": stability,
        "runtime_performance": latency,
        "warning_and_critical_safety_regression": runtime_report.get("current_state_checks"),
        "sealed_holdout_authorized": all_passed,
    }, indent=2, default=str), encoding="utf-8")
    training_path = output / "training_report.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training.update({
        "development_acceptance_status": "PASS" if all_passed else "FAIL",
        "development_gates_passed": all_passed,
        "sealed_holdout_authorized": all_passed,
        "acceptance_opened_after_freeze": True,
    })
    training_path.write_text(json.dumps(training, indent=2), encoding="utf-8")

    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    registry["v2_4_acceptance_consumed"] = True
    registry["v2_4_acceptance_consumed_at"] = datetime.now(timezone.utc).isoformat()
    registry["v2_4_acceptance_model_sha256"] = _sha256(model_path)
    registry["v2_4_acceptance_result"] = "PASS" if all_passed else "FAIL"
    registry["sealed_holdout_authorized"] = all_passed
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return acceptance_report


def refresh_rul_v2_4_acceptance_reporting(
    output_dir: str | Path,
    consumed_registry_path: str | Path,
) -> dict[str, Any]:
    """Correct deterministic report aggregation without reevaluating or retuning a candidate."""
    output = Path(output_dir)
    report_path = output / "development_acceptance_report.json"
    predictions_path = output / "runtime_acceptance_predictions.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _read_rows(predictions_path)
    interval = _interval_metrics_from_runtime_rows(rows)
    report["interval_metrics"] = interval
    checks = report["checks"]
    macro_threshold = float(report["criteria"]["min_active_macro_lifecycle_coverage"])
    checks["active_macro_lifecycle_coverage"].update({
        "observed": interval["macro_lifecycle_coverage"],
        "status": (
            "PASS" if interval["macro_lifecycle_coverage"] is not None
            and interval["macro_lifecycle_coverage"] >= macro_threshold else "FAIL"
        ),
        "reason": "conditional across active interval rows; withheld rows are reported separately",
    })
    for name in ("lifecycle_concentration", "batch_concentration"):
        observed = checks[name]["observed"]
        limit = (
            float(report["criteria"]["max_single_lifecycle_active_share"])
            if name == "lifecycle_concentration"
            else float(report["criteria"]["max_single_batch_active_share"])
        )
        checks[name]["status"] = "PASS" if observed is not None and float(observed) <= limit else "FAIL"
    statuses = [row["status"] for row in checks.values()]
    passed = bool(statuses and all(status == "PASS" for status in statuses))
    report["all_required_gates_passed"] = passed
    report["sealed_holdout_authorized"] = passed
    report["post_evaluation_reporting_correction"] = {
        "applied": True,
        "candidate_predictions_recomputed": False,
        "acceptance_rerun": False,
        "model_or_threshold_changed": False,
        "corrections": [
            "macro coverage denominator changed from all eligible rows to active interval rows",
            "NumPy boolean concentration results normalized to PASS/FAIL",
        ],
        "current_reporting_code_sha256": _sha256(Path(__file__)),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    training_path = output / "training_report.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["development_acceptance_status"] = "PASS" if passed else "FAIL"
    training["development_gates_passed"] = passed
    training["sealed_holdout_authorized"] = passed
    training_path.write_text(json.dumps(training, indent=2), encoding="utf-8")
    registry_path = Path(consumed_registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    registry["v2_4_acceptance_result"] = "PASS" if passed else "FAIL"
    registry["sealed_holdout_authorized"] = passed
    registry["post_evaluation_reporting_correction"] = True
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return report
