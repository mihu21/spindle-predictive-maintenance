from __future__ import annotations

import copy
import json
import math
import os
import platform
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from .config import SensorConfig
from .rul_evaluation import evaluate_rul
from .rul_ml import (
    RUL_CALIBRATION_METHOD_V2_1,
    RUL_CALIBRATION_METHOD_V2_5,
    RUL_DIAGNOSTIC_BUCKETS,
    RUL_MODEL_VERSION_V2_4,
    RUL_MODEL_VERSION_V2_5,
    _finite_sample_conformal_quantile,
    _lifecycle_conformal_margin,
    diagnostic_horizon_bucket,
)
from .rul_v2_4 import (
    ROLE_ORDER,
    V2_4_GENERATOR_VERSION,
    _batch_summary,
    _interval_metrics_from_runtime_rows,
    _json_hash,
    _optional_float,
    _read_rows,
    _sha256,
    _state_stability,
    _write_rows,
)
from .synthetic import SyntheticConfig, generate_mock_csv
from .training import assert_not_external_rul_evaluation_input


V2_5_CORPUS_VERSION = "rul_v2_5_physically_role_separated_corpus_v1"
V2_5_PROTOCOL_VERSION = "frozen_point_final_active_calibration_protocol_v1"
V2_5_GENERATOR_VERSION = V2_4_GENERATOR_VERSION
V2_5_CALIBRATION_CANDIDATES = (
    "global_active",
    "support_stratified",
    "selector_confidence_stratified",
    "asymmetric_active",
)
TARGETS = ("warning", "critical")


@dataclass(frozen=True)
class RULV25AcceptanceCriteria:
    max_active_mae_hours: float = 7.0
    max_active_macro_mae_hours: float = 8.0
    min_active_interval_coverage: float = 0.80
    min_active_macro_lifecycle_coverage: float = 0.75
    min_active_region_availability: float = 0.95
    minimum_active_row_fraction: float = 0.20
    minimum_active_lifecycle_fraction: float = 0.75
    minimum_active_batch_fraction: float = 1.0
    minimum_median_lifecycle_active_fraction: float = 0.10
    minimum_0_24h_active_fraction: float = 0.80
    minimum_24_48h_active_fraction: float = 0.20
    breadth_noninferiority_tolerance: float = 0.05
    minimum_lifecycles_per_horizon: int = 8
    minimum_batches_per_horizon: int = 4
    width_ratio_limit: float = 1.20
    width_absolute_increase_limit_hours: float = 8.0
    width_exception_ratio_hard_limit: float = 1.35
    width_exception_absolute_hard_limit_hours: float = 16.0
    max_rolling_24h_active_unavailable_oscillations: int = 2
    max_runtime_p95_ms_per_row: float = 100.0
    max_runtime_p99_ms_per_row: float = 250.0
    max_incremental_memory_mb: float = 500.0
    max_artifact_bytes: int = 250 * 1024 * 1024


HORIZON_COVERAGE_GATES = {
    "le_6h": 0.80,
    "6_12h": 0.80,
    "12_24h": 0.80,
    "24_48h": 0.75,
}
HORIZON_BIAS_GATES = {
    "le_6h": 2.0,
    "6_12h": 3.0,
    "12_24h": 4.0,
    "24_48h": 6.0,
}


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_quantile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    return float(np.quantile(array, q)) if len(array) else None


def _object_hash(value: Any) -> str:
    return str(joblib.hash(value))


def _estimator_contract(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return {
            "type": "ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": __import__("hashlib").sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, (list, tuple)):
        return [_estimator_contract(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _estimator_contract(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in {"neighbor_model", "_lock", "memory"}
        }
    if hasattr(value, "get_params"):
        state = {
            key: item for key, item in vars(value).items()
            if key.endswith("_") or key in {"indices", "steps", "value"}
        }
        return {
            "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "params": _estimator_contract(value.get_params(deep=False)),
            "learned_state": _estimator_contract(state),
        }
    if hasattr(value, "__dict__"):
        return {
            "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "state": _estimator_contract(vars(value)),
        }
    return repr(value)


def _stable_estimator_configuration(model: Any) -> Any:
    output: dict[str, Any] = {
        "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
    }
    if hasattr(model, "indices"):
        output["indices"] = np.asarray(model.indices, dtype=int).tolist()
    if hasattr(model, "steps"):
        output["steps"] = [
            {"name": name, "estimator": _stable_estimator_configuration(estimator)}
            for name, estimator in model.steps
        ]
    if hasattr(model, "model") and model is not getattr(model, "model", None):
        output["model"] = _stable_estimator_configuration(model.model)
    if hasattr(model, "get_params"):
        params: dict[str, Any] = {}
        for key, value in model.get_params(deep=False).items():
            if value is None or isinstance(value, (str, int, float, bool)):
                params[key] = value
            elif isinstance(value, (list, tuple)) and all(
                item is None or isinstance(item, (str, int, float, bool)) for item in value
            ):
                params[key] = list(value)
        output["primitive_params"] = params
    return output


def _estimator_contract_hash(model: Any, feature_count: int) -> dict[str, Any]:
    rng = np.random.default_rng(2505)
    probe = np.vstack([
        np.zeros((1, feature_count), dtype=float),
        np.ones((1, feature_count), dtype=float),
        np.linspace(-1.0, 1.0, feature_count).reshape(1, -1),
        rng.normal(size=(254, feature_count)),
    ])
    predictions = np.asarray(model.predict(probe), dtype=float)
    return {
        "configuration_hash": _json_hash(_stable_estimator_configuration(model)),
        "probe_prediction_hash": __import__("hashlib").sha256(predictions.tobytes()).hexdigest(),
        "probe_rows": len(probe),
        "probe_prediction_summary": {
            "min": float(np.min(predictions)),
            "median": float(np.median(predictions)),
            "max": float(np.max(predictions)),
        },
    }


def _point_contract_hashes(bundle: dict[str, Any]) -> dict[str, Any]:
    artifact = bundle["rul_model"]
    output: dict[str, Any] = {
        "feature_names_hash": _json_hash(list(artifact["feature_names"])),
        "base_feature_names_hash": _json_hash(list(artifact["base_feature_names"])),
        "stabilization_hash": _json_hash({
            key: artifact.get(key)
            for key in (
                "max_upward_jump_floor_hours",
                "max_upward_jump_per_elapsed_hour",
                "upward_revision_cooldown_hours",
                "upward_revision_trigger_hours",
            )
        }),
    }
    for target in TARGETS:
        row = artifact["targets"][target]
        calibration = dict(row.get("calibration") or {})
        output[target] = {
            "estimator_contract": _estimator_contract_hash(
                row["model"], len(artifact["feature_names"]),
            ),
            "point_bias_hours": float(calibration.get("point_bias_hours", 0.0)),
            "residual_p10_hours": float(calibration.get("residual_p10_hours", 0.0)),
            "residual_p90_hours": float(calibration.get("residual_p90_hours", 0.0)),
        }
    return output


def _consumed_seed_evidence() -> tuple[set[int], list[dict[str, Any]]]:
    seeds = {12001, 14001}
    evidence: list[dict[str, Any]] = []
    for version in ("rul_v2_2", "rul_v2_3", "rul_v2_4"):
        manifest_path = Path("output") / version / "corpus_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for batch in manifest.get("batches", []):
            seed = batch.get("seed")
            if seed is not None:
                seeds.add(int(seed))
            evidence.append({
                "source_version": version,
                "batch_id": batch.get("batch_id") or batch.get("batch"),
                "seed": seed,
                "role": batch.get("role"),
                "sha256": batch.get("source_sha256") or batch.get("sha256"),
                "acceptance_is_consumed": batch.get("role") == "acceptance",
            })
    return seeds, evidence


def generate_rul_v2_5_development_corpus(
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
    normalized = {role: [int(seed) for seed in role_seeds.get(role, [])] for role in ROLE_ORDER}
    if any(len(normalized[role]) < 4 for role in ROLE_ORDER):
        raise ValueError("v2.5 requires at least four batches for every immutable evidence role")
    all_seeds = [seed for role in ROLE_ORDER for seed in normalized[role]]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("A seed may belong to only one v2.5 evidence role")
    consumed_seeds, historical = _consumed_seed_evidence()
    overlap = sorted(consumed_seeds & set(all_seeds))
    if overlap:
        raise ValueError(f"Protected or consumed seeds cannot be reused in v2.5: {overlap}")

    root = Path(data_dir)
    batch_dir = root / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    lifecycle_seen: set[str] = set()
    machine_seen_by_role: dict[str, set[str]] = defaultdict(set)
    batches: list[dict[str, Any]] = []
    role_files: dict[str, dict[str, Any]] = {}
    next_id = 1
    ordinal = 0
    generator_configuration = {
        "lifecycles_per_batch": lifecycles_per_batch,
        "machines": machines,
        "cadence_seconds": cadence_seconds,
        "duration_min_hours": 54.0,
        "duration_max_hours": 156.0,
        "causal_hazard_coupling": True,
    }
    prefixes = {"fit": "v25fit", "calibration": "v25cal", "acceptance": "v25acc"}
    for role in ROLE_ORDER:
        role_rows: list[dict[str, str]] = []
        for index, seed in enumerate(normalized[role], start=1):
            ordinal += 1
            batch_id = f"{prefixes[role]}_b{index:03d}_s{seed}"
            batch_path = batch_dir / f"{batch_id}.csv"
            generate_mock_csv(
                batch_path,
                SyntheticConfig(
                    lifecycles=lifecycles_per_batch,
                    cadence_seconds=cadence_seconds,
                    seed=seed,
                    machines=machines,
                    line_sel=line_sel,
                    start_time=datetime(2095, 1, 1, tzinfo=timezone.utc) + timedelta(days=370 * ordinal),
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
                "generator_version": V2_5_GENERATOR_VERSION,
                "generator_configuration_hash": _json_hash(generator_configuration),
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
    for left in ROLE_ORDER:
        for right in ROLE_ORDER:
            if left < right and machine_seen_by_role[left] & machine_seen_by_role[right]:
                raise RuntimeError("Machine identities must be globally role-disjoint")

    manifest = {
        "corpus_version": V2_5_CORPUS_VERSION,
        "protocol_version": V2_5_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "role_assignments_permanent": True,
        "physical_role_files": True,
        "acceptance_content_parsed_before_freeze": False,
        "acceptance_generation_and_streaming_hash_only": True,
        "protected_external_datasets_used": False,
        "generator_version": V2_5_GENERATOR_VERSION,
        "generator_changed_from_v2_4": False,
        "role_files": role_files,
        "batches": batches,
        "provenance_fields_excluded_from_features": [
            "generation_seed", "batch_id", "development_role", "latent_damage_score",
            "fault_mode", "lifecycle_progress", "machine_id", "lifecycle_id",
        ],
    }
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = {
        "registry_version": "permanent_consumed_rul_evidence_registry_v3",
        "role_reassignment_permitted": False,
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
        "historical_consumed_batches": historical,
        "v2_5_role_locked_batches": [
            {"batch_id": row["batch_id"], "seed": row["seed"], "role": row["role"], "sha256": row["source_sha256"]}
            for row in batches
        ],
        "v2_5_acceptance_opened": False,
        "v2_5_acceptance_consumed": False,
        "sealed_holdout_authorized": False,
    }
    registry_output = Path(consumed_registry_path)
    registry_output.parent.mkdir(parents=True, exist_ok=True)
    registry_output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return manifest


def _prepare_v2_5_bundle(frozen: dict[str, Any], *, provisional: bool) -> dict[str, Any]:
    # Mutate a freshly deserialized bundle in place. Deep-copying sklearn tree internals changes
    # non-semantic object identity and makes object-level digests unstable despite identical
    # parameters and predictions.
    bundle = frozen
    artifact = bundle["rul_model"]
    if artifact.get("version") != RUL_MODEL_VERSION_V2_4:
        raise ValueError("v2.5 requires the frozen v2.4 state-selective artifact")
    point_hashes = _point_contract_hashes(frozen)
    artifact["version"] = RUL_MODEL_VERSION_V2_5
    contract = dict(artifact.get("forecastability_contract") or {})
    contract.update({
        "mode": "frozen_v2_4_selector_canonical_v2_5_serviceability",
        "minimum_active_confirmations": 3,
        "minimum_active_dwell": 3,
        "maximum_exact_rul_horizon_hours": 48.0,
        "final_active_definition": "serviceable_intent AND complete_finite_point_and_interval",
        "serviceable_intent_definition": "hard_eligible AND selector_active AND hysteresis_confirmed",
        "support_and_disagreement_role": "frozen learned selector inputs; not new independent hard filters",
        "point_contract_hashes": point_hashes,
        "normal_withheld_behavior": "clear all exact RUL outputs and actionability",
        "anomaly_withheld_behavior": "clear exact RUL; never hold stale exact RUL",
    })
    artifact["forecastability_contract"] = contract
    for target in TARGETS:
        calibration = dict(artifact["targets"][target].get("calibration") or {})
        if provisional:
            calibration.update({
                "method": RUL_CALIBRATION_METHOD_V2_1,
                "global_margin_hours": 0.0,
                "buckets": {},
                "minimum_interval_widths": {},
                "v2_5_provisional_uncalibrated": True,
            })
        artifact["targets"][target]["calibration"] = calibration
    artifact["v2_5_point_contract_hashes"] = point_hashes
    artifact["v2_5_development_only"] = True
    return bundle


def _active_target_rows(rows: list[dict[str, str]], target: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    truth_key = f"true_hours_to_{target}"
    point_key = f"estimated_hours_to_{target}"
    lo_key = f"{target}_lower_hours"
    hi_key = f"{target}_upper_hours"
    for row in rows:
        truth = _optional_float(row.get(truth_key))
        point = _optional_float(row.get(point_key))
        lo = _optional_float(row.get(lo_key))
        hi = _optional_float(row.get(hi_key))
        if truth is None or truth <= 0.0 or not _bool(row.get("rul_serviceable_intent")):
            continue
        if point is None or lo is None or hi is None:
            continue
        output.append({
            **row,
            "truth": truth,
            "point": point,
            "raw_lower": lo,
            "raw_upper": hi,
            "selector_score": _optional_float(row.get("rul_forecastability_score")),
            "support_distance": _optional_float(row.get("rul_support_distance")),
            "neighbor_dispersion_hours": _optional_float(row.get("rul_neighbor_dispersion_hours")),
            "model_disagreement_hours": _optional_float(row.get("rul_model_disagreement_hours")),
        })
    return output


def _stratum(row: dict[str, Any], candidate: str) -> str:
    if candidate == "selector_confidence_stratified":
        score = row.get("selector_score")
        return "SELECTOR_HIGH" if score is not None and float(score) >= 0.95 else "SELECTOR_MODERATE"
    if candidate == "support_stratified":
        dispersion = row.get("neighbor_dispersion_hours")
        return "SUPPORT_HIGH" if dispersion is not None and float(dispersion) <= 12.0 else "SUPPORT_MARGINAL"
    return "GLOBAL"


def _margin(scores: np.ndarray, groups: np.ndarray, *, coverage: float, macro: float) -> tuple[float, dict[str, Any]]:
    micro, rank, clipped = _finite_sample_conformal_quantile(scores, coverage)
    lifecycle, metadata = _lifecycle_conformal_margin(
        scores, groups, within_lifecycle_coverage=coverage, lifecycle_coverage=macro,
    )
    return max(float(micro), float(lifecycle)), {
        "micro_margin_hours": float(micro),
        "lifecycle_margin_hours": float(lifecycle),
        "finite_sample_rank": int(rank),
        "rank_clipped": bool(clipped),
        "lifecycle_conformal": metadata,
    }


def _fit_calibration(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No final-serviceable rows are available for v2.5 calibration")
    groups = np.asarray([row["lifecycle_id"] for row in rows], dtype=object)
    truth = np.asarray([row["truth"] for row in rows], dtype=float)
    raw_lo = np.asarray([row["raw_lower"] for row in rows], dtype=float)
    raw_hi = np.asarray([row["raw_upper"] for row in rows], dtype=float)
    strata_values = np.asarray([_stratum(row, candidate) for row in rows], dtype=object)
    symmetric_scores = np.maximum.reduce([raw_lo - truth, truth - raw_hi, np.zeros(len(rows))])
    global_margin, global_meta = _margin(symmetric_scores, groups, coverage=0.80, macro=0.75)
    global_row = {
        "margin_hours": global_margin,
        "lower_margin_hours": global_margin,
        "upper_margin_hours": global_margin,
        "rows": len(rows),
        "lifecycles": len(set(groups.tolist())),
        "batches": len({row["batch_id"] for row in rows}),
        "fallback_used": False,
        **global_meta,
    }
    result = {
        "method": RUL_CALIBRATION_METHOD_V2_5,
        "candidate": candidate,
        "selector_high_threshold": 0.95,
        "support_mad_threshold_hours": 12.0,
        "minimum_stratum_lifecycles": 8,
        "minimum_stratum_batches": 4,
        "strata": {"GLOBAL": global_row},
        "fallback_rules": "unsupported strata use GLOBAL",
    }
    for name in sorted(set(strata_values.tolist()) - {"GLOBAL"}):
        mask = strata_values == name
        local_groups = groups[mask]
        local_batches = {rows[pos]["batch_id"] for pos in np.flatnonzero(mask)}
        supported = len(set(local_groups.tolist())) >= 8 and len(local_batches) >= 4
        if supported:
            local_margin, local_meta = _margin(
                symmetric_scores[mask], local_groups, coverage=0.80, macro=0.75,
            )
            result["strata"][name] = {
                "margin_hours": local_margin,
                "lower_margin_hours": local_margin,
                "upper_margin_hours": local_margin,
                "rows": int(np.sum(mask)),
                "lifecycles": len(set(local_groups.tolist())),
                "batches": len(local_batches),
                "fallback_used": False,
                **local_meta,
            }
        else:
            result["strata"][name] = {
                **global_row,
                "rows": int(np.sum(mask)),
                "lifecycles": len(set(local_groups.tolist())),
                "batches": len(local_batches),
                "fallback_used": True,
                "fallback_reason": "fewer_than_8_lifecycles_or_4_batches",
            }
    if candidate == "asymmetric_active":
        lower_scores = np.maximum(raw_lo - truth, 0.0)
        upper_scores = np.maximum(truth - raw_hi, 0.0)
        lower_margin, lower_meta = _margin(lower_scores, groups, coverage=0.90, macro=0.875)
        upper_margin, upper_meta = _margin(upper_scores, groups, coverage=0.90, macro=0.875)
        result["strata"]["GLOBAL"].update({
            "lower_margin_hours": lower_margin,
            "upper_margin_hours": upper_margin,
            "lower_tail_conformal": lower_meta,
            "upper_tail_conformal": upper_meta,
        })
    return result


def _apply_calibration_config(row: dict[str, Any], config: dict[str, Any]) -> tuple[float, float, bool]:
    name = _stratum(row, config["candidate"])
    strata = config["strata"]
    selected = strata.get(name) or strata["GLOBAL"]
    fallback = name not in strata or bool(selected.get("fallback_used"))
    lo = max(0.0, float(row["raw_lower"]) - float(selected["lower_margin_hours"]))
    hi = min(720.0, float(row["raw_upper"]) + float(selected["upper_margin_hours"]))
    return lo, hi, fallback


def _candidate_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    truth = np.asarray([row["truth"] for row in rows], dtype=float)
    point = np.asarray([row["point"] for row in rows], dtype=float)
    groups = np.asarray([row["lifecycle_id"] for row in rows], dtype=object)
    batches = np.asarray([row["batch_id"] for row in rows], dtype=object)
    applied = [_apply_calibration_config(row, config) for row in rows]
    lo = np.asarray([value[0] for value in applied], dtype=float)
    hi = np.asarray([value[1] for value in applied], dtype=float)
    hits = (lo <= truth) & (truth <= hi)
    macro_coverage = [float(np.mean(hits[groups == lifecycle])) for lifecycle in sorted(set(groups.tolist()))]
    macro_mae = [float(np.mean(np.abs(point[groups == lifecycle] - truth[groups == lifecycle]))) for lifecycle in sorted(set(groups.tolist()))]
    return {
        "rows": len(rows),
        "lifecycles": len(set(groups.tolist())),
        "batches": len(set(batches.tolist())),
        "coverage": float(np.mean(hits)) if len(rows) else None,
        "macro_lifecycle_coverage": float(np.mean(macro_coverage)) if macro_coverage else None,
        "mean_width_hours": float(np.mean(hi - lo)) if len(rows) else None,
        "median_width_hours": float(np.median(hi - lo)) if len(rows) else None,
        "p90_width_hours": float(np.quantile(hi - lo, 0.90)) if len(rows) else None,
        "mae_hours": float(np.mean(np.abs(point - truth))) if len(rows) else None,
        "macro_lifecycle_mae_hours": float(np.mean(macro_mae)) if macro_mae else None,
        "fallback_rate": float(np.mean([value[2] for value in applied])) if applied else None,
    }


def _asymmetry_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_batch: dict[str, tuple[float, float]] = {}
    for batch in sorted({row["batch_id"] for row in rows}):
        errors = np.asarray([row["point"] - row["truth"] for row in rows if row["batch_id"] == batch], dtype=float)
        by_batch[batch] = (
            float(np.quantile(np.maximum(errors, 0.0), 0.90)),
            float(np.quantile(np.maximum(-errors, 0.0), 0.90)),
        )
    all_errors = np.asarray([row["point"] - row["truth"] for row in rows], dtype=float)
    over = float(np.quantile(np.maximum(all_errors, 0.0), 0.90))
    under = float(np.quantile(np.maximum(-all_errors, 0.0), 0.90))
    direction = math.copysign(1.0, over - under) if over != under else 0.0
    stable = sum(
        1 for left, right in by_batch.values()
        if direction and math.copysign(1.0, left - right) == direction and abs(left - right) >= 1.0
    )
    justified = abs(over - under) >= 3.0 and stable >= 4
    return {
        "p90_overprediction_hours": over,
        "p90_underprediction_hours": under,
        "absolute_tail_difference_hours": abs(over - under),
        "batches_with_consistent_direction": stable,
        "minimum_required_batches": 4,
        "justified": justified,
        "per_batch": {key: {"over": value[0], "under": value[1]} for key, value in by_batch.items()},
    }


def _compare_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    asymmetry = _asymmetry_diagnostics(rows)
    batches = sorted({row["batch_id"] for row in rows})
    results: dict[str, Any] = {}
    for candidate in V2_5_CALIBRATION_CANDIDATES:
        if candidate == "asymmetric_active" and not asymmetry["justified"]:
            results[candidate] = {"status": "NOT_JUSTIFIED", "crossfit": None}
            continue
        predicted: list[dict[str, Any]] = []
        for held_batch in batches:
            train = [row for row in rows if row["batch_id"] != held_batch]
            held = [row for row in rows if row["batch_id"] == held_batch]
            config = _fit_calibration(train, candidate)
            for row in held:
                lo, hi, fallback = _apply_calibration_config(row, config)
                predicted.append({**row, "raw_lower": lo, "raw_upper": hi, "candidate_fallback": fallback})
        identity = {
            "candidate": "global_active",
            "strata": {"GLOBAL": {"lower_margin_hours": 0.0, "upper_margin_hours": 0.0}},
        }
        metrics = _candidate_metrics(predicted, identity)
        results[candidate] = {"status": "EVALUATED", "crossfit": metrics}
    baseline_width = results["global_active"]["crossfit"]["mean_width_hours"]
    normal_width_limit = min(1.20 * baseline_width, baseline_width + 8.0)
    hard_width_limit = min(1.35 * baseline_width, baseline_width + 16.0)
    eligible: list[str] = []
    for candidate, report in results.items():
        metrics = report.get("crossfit")
        if metrics is None:
            report["preacceptance_pass"] = False
            continue
        macro_improvement = metrics["macro_lifecycle_coverage"] - results["global_active"]["crossfit"]["macro_lifecycle_coverage"]
        width_pass = metrics["mean_width_hours"] <= normal_width_limit or (
            macro_improvement >= 0.10 and metrics["mean_width_hours"] <= hard_width_limit
        )
        report["width_guard"] = {
            "baseline_mean_width_hours": baseline_width,
            "normal_limit_hours": normal_width_limit,
            "hard_exception_limit_hours": hard_width_limit,
            "macro_coverage_improvement": macro_improvement,
            "pass": width_pass,
        }
        report["preacceptance_pass"] = bool(
            metrics["coverage"] >= 0.80
            and metrics["macro_lifecycle_coverage"] >= 0.75
            and width_pass
        )
        if report["preacceptance_pass"]:
            eligible.append(candidate)
    selected = (
        min(eligible, key=lambda name: results[name]["crossfit"]["mean_width_hours"])
        if eligible
        else max(
            [name for name, value in results.items() if value.get("crossfit")],
            key=lambda name: (
                results[name]["crossfit"]["macro_lifecycle_coverage"],
                results[name]["crossfit"]["coverage"],
                -results[name]["crossfit"]["mean_width_hours"],
            ),
        )
    )
    return {
        "protocol_version": V2_5_PROTOCOL_VERSION,
        "selection_evidence_role": "fit_design_only",
        "calibration_evidence_used_for_candidate_selection": False,
        "acceptance_evidence_used": False,
        "asymmetry_diagnostics": asymmetry,
        "candidates": results,
        "selected_candidate": selected,
        "selection_passed": selected in eligible,
    }


def _alignment_report(populations: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, rows in populations.items():
        counts: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        lifecycle_rates: list[float] = []
        by_lifecycle: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            selector = _bool(row.get("rul_selector_active"))
            active = row.get("rul_forecastability_state") == "RUL_ACTIVE"
            key = f"selector_{'active' if selector else 'inactive'}__service_{'active' if active else 'withheld'}"
            counts[key] += 1
            if selector and not active:
                reasons[row.get("rul_withholding_reason_code") or "OTHER"] += 1
            by_lifecycle[row.get("lifecycle_id") or "UNKNOWN"].append(row)
        for local in by_lifecycle.values():
            denominator = sum(_bool(row.get("rul_selector_active")) for row in local)
            numerator = sum(
                _bool(row.get("rul_selector_active"))
                and row.get("rul_forecastability_state") != "RUL_ACTIVE"
                for row in local
            )
            if denominator:
                lifecycle_rates.append(numerator / denominator)
        result[name] = {
            "rows": len(rows),
            "confusion_counts": dict(counts),
            "confusion_percentages": {key: value / len(rows) for key, value in counts.items()} if rows else {},
            "selector_active_final_withheld_reason_counts": dict(reasons),
            "lifecycle_macro_selector_active_final_withheld_rate": (
                float(np.mean(lifecycle_rates)) if lifecycle_rates else None
            ),
            "batches": len({row.get("batch_id") for row in rows}),
        }
    return {
        "version": RUL_MODEL_VERSION_V2_5,
        "availability_denominator": "RUL_SERVICEABLE_INTENT before point and interval construction",
        "populations": result,
    }


def _residual_report(populations: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for population, rows in populations.items():
        output[population] = {}
        for target in TARGETS:
            active = _active_target_rows(rows, target)
            errors = np.asarray([row["point"] - row["truth"] for row in active], dtype=float)
            output[population][target] = {
                "rows": len(active),
                "lifecycles": len({row["lifecycle_id"] for row in active}),
                "batches": len({row["batch_id"] for row in active}),
                "mean_signed_error_hours": float(np.mean(errors)) if len(errors) else None,
                "median_signed_error_hours": float(np.median(errors)) if len(errors) else None,
                "mae_hours": float(np.mean(np.abs(errors))) if len(errors) else None,
                "median_ae_hours": float(np.median(np.abs(errors))) if len(errors) else None,
                "p80_ae_hours": _safe_quantile(np.abs(errors), 0.80),
                "p90_ae_hours": _safe_quantile(np.abs(errors), 0.90),
                "p95_ae_hours": _safe_quantile(np.abs(errors), 0.95),
                "asymmetry": _asymmetry_diagnostics(active) if active else None,
            }
    return {"version": RUL_MODEL_VERSION_V2_5, "populations": output}


def _latency_report(populations: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, rows in populations.items():
        values = [
            value for row in rows[100:]
            if (value := _optional_float(row.get("rul_inference_latency_ms"))) is not None
        ]
        result[name] = {
            "warmup_rows_excluded": min(100, len(rows)),
            "measured_rows": len(values),
            "mean_ms": float(np.mean(values)) if values else None,
            "p95_ms": float(np.quantile(values, 0.95)) if values else None,
            "p99_ms": float(np.quantile(values, 0.99)) if values else None,
        }
    return result


def _complete_lifecycle_design_sample(
    source_path: str | Path,
    output_path: str | Path,
    *,
    lifecycles_per_batch: int = 3,
) -> dict[str, Any]:
    """Select whole lifecycles, never partial histories, for calibration-method design."""
    rows = _read_rows(source_path)
    selected: set[str] = set()
    per_batch: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        lifecycle = row["lifecycle_id"]
        batch = row["batch_id"]
        if lifecycle not in per_batch[batch] and len(per_batch[batch]) < lifecycles_per_batch:
            per_batch[batch].append(lifecycle)
            selected.add(lifecycle)
    sampled = [row for row in rows if row["lifecycle_id"] in selected]
    _write_rows(output_path, sampled)
    return {
        "selection": "first three complete lifecycles by stable source order per fit batch",
        "partial_lifecycles_used": False,
        "batches": len(per_batch),
        "lifecycles": len(selected),
        "rows": len(sampled),
        "per_batch_lifecycles": per_batch,
        "sha256": _sha256(output_path),
    }


def train_rul_v2_5_candidate(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    frozen_v2_4_model_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    freeze_manifest_path: str | Path,
    sensor_config: SensorConfig,
    *,
    criteria: RULV25AcceptanceCriteria | None = None,
) -> dict[str, Any]:
    criteria = criteria or RULV25AcceptanceCriteria()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    registry = json.loads(Path(consumed_registry_path).read_text(encoding="utf-8"))
    if manifest.get("corpus_version") != V2_5_CORPUS_VERSION:
        raise ValueError("Unsupported v2.5 corpus manifest")
    if registry.get("v2_5_acceptance_opened") is not False:
        raise RuntimeError("v2.5 acceptance is already consumed; start a new version")
    for role in ("fit", "calibration"):
        path = Path(manifest["role_files"][role]["path"])
        assert_not_external_rul_evaluation_input(path)
        if _sha256(path) != manifest["role_files"][role]["sha256"]:
            raise RuntimeError(f"v2.5 {role} file hash mismatch")

    frozen_path = Path(frozen_v2_4_model_path)
    frozen = joblib.load(frozen_path)
    frozen_point_hashes = _point_contract_hashes(frozen)
    provisional = _prepare_v2_5_bundle(joblib.load(frozen_path), provisional=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    prior_provisional_contract = None
    if model_output.exists():
        prior_bundle = joblib.load(model_output)
        if prior_bundle.get("rul_model", {}).get("version") == RUL_MODEL_VERSION_V2_5:
            prior_provisional_contract = _point_contract_hashes(prior_bundle)
    joblib.dump(provisional, model_output)
    provisional_sha = _sha256(model_output)
    provisional_contract = _point_contract_hashes(provisional)

    fit_design_path = output / "fit_design_complete_lifecycles.csv"
    fit_design = _complete_lifecycle_design_sample(
        manifest["role_files"]["fit"]["path"], fit_design_path,
    )
    populations: dict[str, list[dict[str, str]]] = {}
    for role in ("fit", "calibration"):
        report_path = output / f"{role}_runtime_replay_report.json"
        predictions_path = output / f"{role}_runtime_predictions.csv"
        role_input = fit_design_path if role == "fit" else Path(manifest["role_files"][role]["path"])
        reused = False
        if (
            role in {"fit", "calibration"}
            and prior_provisional_contract == provisional_contract
            and report_path.exists()
            and predictions_path.exists()
        ):
            prior_report = json.loads(report_path.read_text(encoding="utf-8"))
            prior_rows = _read_rows(predictions_path)
            reused = bool(
                Path(prior_report.get("input_path", "")).resolve() == role_input.resolve()
                and prior_report.get("rul_artifact_version") == RUL_MODEL_VERSION_V2_5
                and len(prior_rows) == (
                    fit_design["rows"] if role == "fit" else manifest["role_files"]["calibration"]["rows"]
                )
            )
        if not reused:
            evaluate_rul(
                model_output,
                role_input,
                report_path,
                predictions_path,
                sensor_config,
            )
        populations[role] = _read_rows(predictions_path)
        if role == "fit":
            fit_design["runtime_replay_reused_after_timeout"] = reused
            fit_design["validated_provisional_model_sha256"] = provisional_sha
            fit_design["runtime_predictions_sha256"] = _sha256(predictions_path)
        elif role == "calibration":
            fit_design["calibration_runtime_replay_reused_after_freeze_hash_failure"] = reused
            fit_design["calibration_runtime_predictions_sha256"] = _sha256(predictions_path)

    design_critical = _active_target_rows(populations["fit"], "critical")
    comparison = _compare_candidates(design_critical)
    selected = comparison["selected_candidate"]
    final_bundle = _prepare_v2_5_bundle(joblib.load(frozen_path), provisional=False)
    support_report: dict[str, Any] = {}
    calibration_metrics: dict[str, Any] = {}
    for target in TARGETS:
        rows = _active_target_rows(populations["calibration"], target)
        fitted = _fit_calibration(rows, selected)
        original = dict(final_bundle["rul_model"]["targets"][target].get("calibration") or {})
        original.update(fitted)
        original.update({
            "calibration_population": "complete output among canonical final serviceable intent rows",
            "candidate_selected_on": "fit_design_batch_crossfit_only",
            "calibration_role_used_for_selection": False,
            "true_rul_used_as_runtime_key": False,
            "v2_5_provisional_uncalibrated": False,
        })
        final_bundle["rul_model"]["targets"][target]["calibration"] = original
        calibration_metrics[target] = _candidate_metrics(rows, fitted)
        support_report[target] = fitted["strata"]
    if _point_contract_hashes(final_bundle) != frozen_point_hashes:
        raise RuntimeError("v2.5 modified the frozen point predictor contract")
    joblib.dump(final_bundle, model_output)

    runtime_order = {
        "version": RUL_MODEL_VERSION_V2_5,
        "canonical_helper": "resolve_v2_5_serviceability",
        "order": [
            "read observation and reset lifecycle state if required",
            "manufacturer CRITICAL safety override (cannot be withheld)",
            "update trusted causal history and v2.4 frozen causal features",
            "hard eligibility: trust, finite state, chronological order, features, history, <=48h support contract",
            "frozen v2.4 forecastability selector and diagnostics",
            "three-reading confirmation and three-reading minimum ACTIVE dwell",
            "RUL_SERVICEABLE_INTENT",
            "frozen point prediction and temporal stabilization",
            "selected final-active calibration",
            "complete finite output validation",
            "export RUL_ACTIVE only after complete output; otherwise clear exact RUL",
        ],
        "critical_override_precedes_quality_withholding": True,
        "warning_zero_does_not_force_future_critical": True,
        "support_disagreement_are_selector_inputs_not_duplicate_hard_filters": True,
    }
    alignment = _alignment_report(populations)
    residual = _residual_report(populations)
    latency = _latency_report(populations)
    runtime_contract = {
        "version": RUL_MODEL_VERSION_V2_5,
        "states": ["RUL_ACTIVE", "RUL_LOW_CONFIDENCE", "RUL_UNAVAILABLE"],
        "serviceable_intent_is_internal_denominator": True,
        "exported_active_requires_complete_output": True,
        "withholding_reason_codes": [
            "NO_DEGRADATION_EVIDENCE", "INSUFFICIENT_HISTORY", "OUT_OF_SUPPORT",
            "HIGH_DISAGREEMENT", "UNSTABLE_TREND", "SELECTOR_BELOW_THRESHOLD",
            "HYSTERESIS_NOT_CONFIRMED", "MISSING_REQUIRED_FEATURE", "ANOMALY_WITHHOLD",
            "INVALID_RUNTIME_STATE",
        ],
        "maximum_exact_rul_horizon_hours": 48.0,
        "above_48h": "OUT_OF_CONTRACT; exact RUL withheld",
        "point_contract_hashes": frozen_point_hashes,
    }
    reports = {
        "runtime_order_audit.json": runtime_order,
        "selector_service_alignment_report.json": alignment,
        "residual_distribution_report.json": residual,
        "calibration_candidate_comparison.json": comparison,
        "calibration_support_report.json": {
            "version": RUL_MODEL_VERSION_V2_5,
            "minimum_lifecycles": 8,
            "minimum_batches": 4,
            "targets": support_report,
        },
        "runtime_contract_report.json": runtime_contract,
        "runtime_performance_report.json": {
            "version": RUL_MODEL_VERSION_V2_5,
            "development_replays": latency,
            "hardware": {"platform": platform.platform(), "processor": platform.processor()},
            "software": {"python": sys.version, "numpy": np.__version__},
        },
    }
    for name, value in reports.items():
        (output / name).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

    selector = final_bundle["rul_model"]["forecastability_contract"]["selector"]
    freeze = {
        "freeze_version": "rul_v2_5_preacceptance_freeze_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_output.resolve()),
        "model_sha256": _sha256(model_output),
        "frozen_v2_4_model_sha256": _sha256(frozen_path),
        "frozen_point_contract_hashes": frozen_point_hashes,
        "status_model_object_hash": _object_hash(final_bundle["model"]),
        "selector_model_object_hash": _object_hash(selector["model"]),
        "selector_thresholds_hash": _json_hash({
            key: final_bundle["rul_model"]["forecastability_contract"].get(key)
            for key in (
                "activation_threshold", "deactivation_threshold", "minimum_active_confirmations",
                "minimum_active_dwell", "maximum_exact_rul_horizon_hours",
            )
        }),
        "support_bank_object_hash": _object_hash({
            key: value for key, value in selector["support_bank"].items() if key != "neighbor_model"
        }),
        "manifest_sha256": _sha256(manifest_path),
        "fit_sha256": manifest["role_files"]["fit"]["sha256"],
        "calibration_sha256": manifest["role_files"]["calibration"]["sha256"],
        "acceptance_sha256": manifest["role_files"]["acceptance"]["sha256"],
        "selected_calibration_candidate": selected,
        "calibration_configuration_hash": _json_hash({
            target: final_bundle["rul_model"]["targets"][target]["calibration"] for target in TARGETS
        }),
        "criteria": asdict(criteria),
        "runtime_code_sha256": {
            "rul_v2_5.py": _sha256(Path(__file__)),
            "rul_ml.py": _sha256(Path(__file__).with_name("rul_ml.py")),
            "rul_features_v2_4.py": _sha256(Path(__file__).with_name("rul_features_v2_4.py")),
            "rul_evaluation.py": _sha256(Path(__file__).with_name("rul_evaluation.py")),
            "monitor.py": _sha256(Path(__file__).with_name("monitor.py")),
        },
        "acceptance_content_parsed_before_freeze": False,
        "acceptance_results_used_for_selection": False,
    }
    freeze_output = Path(freeze_manifest_path)
    freeze_output.parent.mkdir(parents=True, exist_ok=True)
    freeze_output.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    training = {
        "version": RUL_MODEL_VERSION_V2_5,
        "protocol_version": V2_5_PROTOCOL_VERSION,
        "development_only": True,
        "production_ready": False,
        "point_predictor_retrained": False,
        "feature_contract_changed": False,
        "generator_changed": False,
        "protected_external_datasets_used": False,
        "selected_calibration_candidate": selected,
        "candidate_selection_passed": comparison["selection_passed"],
        "fit_design_complete_lifecycle_sample": fit_design,
        "calibration_metrics": calibration_metrics,
        "frozen_point_contract_hashes": frozen_point_hashes,
        "model_sha256": freeze["model_sha256"],
        "acceptance_status": "PENDING_UNOPENED",
        "sealed_holdout_authorized": False,
    }
    (output / "training_report.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    return training


def _process_rss_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError):
            return None
    return None


def _run_with_memory_monitor(function: Any) -> tuple[Any, dict[str, Any]]:
    baseline = _process_rss_bytes()
    peak = baseline
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.wait(0.02):
            value = _process_rss_bytes()
            if value is not None:
                peak = value if peak is None else max(peak, value)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        result = function()
    finally:
        stop.set()
        thread.join(timeout=1.0)
    return result, {
        "measurement": "process RSS sampled during end-to-end replay; conservative inference upper bound",
        "baseline_bytes": baseline,
        "peak_bytes": peak,
        "incremental_bytes": (peak - baseline) if peak is not None and baseline is not None else None,
    }


def _target_acceptance_metrics(rows: list[dict[str, str]], target: str) -> dict[str, Any]:
    truth_key = f"true_hours_to_{target}"
    point_key = f"estimated_hours_to_{target}"
    lo_key = f"{target}_lower_hours"
    hi_key = f"{target}_upper_hours"
    eligible = [row for row in rows if (_optional_float(row.get(truth_key)) or 0.0) > 0.0]
    serviceable = [row for row in eligible if _bool(row.get("rul_serviceable_intent"))]
    active = [
        row for row in serviceable
        if _optional_float(row.get(point_key)) is not None
        and _optional_float(row.get(lo_key)) is not None
        and _optional_float(row.get(hi_key)) is not None
        and row.get("rul_forecastability_state") == "RUL_ACTIVE"
    ]
    errors = np.asarray([
        float(row[point_key]) - float(row[truth_key]) for row in active
    ], dtype=float)
    lifecycle_errors: list[float] = []
    lifecycle_coverage: list[float] = []
    hits: list[bool] = []
    widths: list[float] = []
    for lifecycle in sorted({row["lifecycle_id"] for row in active}):
        local = [row for row in active if row["lifecycle_id"] == lifecycle]
        lifecycle_errors.append(float(np.mean([
            abs(float(row[point_key]) - float(row[truth_key])) for row in local
        ])))
        local_hits = [float(row[lo_key]) <= float(row[truth_key]) <= float(row[hi_key]) for row in local]
        lifecycle_coverage.append(float(np.mean(local_hits)))
    for row in active:
        hits.append(float(row[lo_key]) <= float(row[truth_key]) <= float(row[hi_key]))
        widths.append(float(row[hi_key]) - float(row[lo_key]))
    horizons: dict[str, Any] = {}
    for bucket in RUL_DIAGNOSTIC_BUCKETS:
        local = [row for row in active if diagnostic_horizon_bucket(float(row[truth_key])) == bucket]
        local_eligible = [row for row in eligible if diagnostic_horizon_bucket(float(row[truth_key])) == bucket]
        local_hits = [float(row[lo_key]) <= float(row[truth_key]) <= float(row[hi_key]) for row in local]
        local_errors = [float(row[point_key]) - float(row[truth_key]) for row in local]
        horizons[bucket] = {
            "eligible_rows": len(local_eligible),
            "active_rows": len(local),
            "active_fraction": len(local) / len(local_eligible) if local_eligible else None,
            "lifecycles": len({row["lifecycle_id"] for row in local}),
            "batches": len({row["batch_id"] for row in local}),
            "coverage": float(np.mean(local_hits)) if local_hits else None,
            "mean_signed_error_hours": float(np.mean(local_errors)) if local_errors else None,
            "mae_hours": float(np.mean(np.abs(local_errors))) if local_errors else None,
        }
    per_lifecycle_active = []
    for lifecycle in sorted({row["lifecycle_id"] for row in eligible}):
        local_eligible = [row for row in eligible if row["lifecycle_id"] == lifecycle]
        local_active = [row for row in active if row["lifecycle_id"] == lifecycle]
        per_lifecycle_active.append(len(local_active) / len(local_eligible))
    return {
        "eligible_rows": len(eligible),
        "serviceable_intent_rows": len(serviceable),
        "complete_active_rows": len(active),
        "active_region_availability": len(active) / len(serviceable) if serviceable else None,
        "active_row_fraction": len(active) / len(eligible) if eligible else None,
        "active_lifecycles": len({row["lifecycle_id"] for row in active}),
        "eligible_lifecycles": len({row["lifecycle_id"] for row in eligible}),
        "active_lifecycle_fraction": (
            len({row["lifecycle_id"] for row in active}) / len({row["lifecycle_id"] for row in eligible})
            if eligible else None
        ),
        "active_batches": len({row["batch_id"] for row in active}),
        "eligible_batches": len({row["batch_id"] for row in eligible}),
        "active_batch_fraction": (
            len({row["batch_id"] for row in active}) / len({row["batch_id"] for row in eligible})
            if eligible else None
        ),
        "median_lifecycle_active_fraction": float(np.median(per_lifecycle_active)) if per_lifecycle_active else None,
        "mae_hours": float(np.mean(np.abs(errors))) if len(errors) else None,
        "macro_lifecycle_mae_hours": float(np.mean(lifecycle_errors)) if lifecycle_errors else None,
        "coverage": float(np.mean(hits)) if hits else None,
        "macro_lifecycle_coverage": float(np.mean(lifecycle_coverage)) if lifecycle_coverage else None,
        "mean_width_hours": float(np.mean(widths)) if widths else None,
        "median_width_hours": float(np.median(widths)) if widths else None,
        "p90_width_hours": float(np.quantile(widths, 0.90)) if widths else None,
        "horizons": horizons,
    }


def _rolling_oscillation_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    maximum = 0
    dwell: list[int] = []
    for lifecycle in sorted({row.get("lifecycle_id") for row in rows}):
        local = sorted(
            [row for row in rows if row.get("lifecycle_id") == lifecycle],
            key=lambda row: datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
        )
        transitions: list[datetime] = []
        run = 0
        previous_active: bool | None = None
        for row in local:
            active = row.get("rul_forecastability_state") == "RUL_ACTIVE"
            if active:
                run += 1
            elif run:
                dwell.append(run)
                run = 0
            if previous_active is not None and active != previous_active:
                transitions.append(datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")))
            previous_active = active
        if run:
            dwell.append(run)
        window: deque[datetime] = deque()
        for timestamp in transitions:
            window.append(timestamp)
            while window and window[0] < timestamp - timedelta(hours=24):
                window.popleft()
            maximum = max(maximum, len(window))
    return {
        "maximum_active_unavailable_transitions_per_lifecycle_rolling_24h": maximum,
        "active_dwell_rows_median": float(np.median(dwell)) if dwell else None,
        "active_dwell_rows_p10": float(np.quantile(dwell, 0.10)) if dwell else None,
        "active_runs": len(dwell),
    }


def _population_shift(calibration_rows: list[dict[str, str]], acceptance_rows: list[dict[str, str]]) -> dict[str, Any]:
    fields = (
        "rul_forecastability_score", "rul_support_distance", "rul_neighbor_dispersion_hours",
        "rul_model_disagreement_hours", "degradation_score", "estimated_hours_to_critical",
        "rul_history_hours", "critical_rul_lower_hours", "critical_rul_upper_hours",
        "critical_error_hours",
    )
    output: dict[str, Any] = {}
    for name, rows in (("calibration", calibration_rows), ("acceptance", acceptance_rows)):
        active = [row for row in rows if row.get("rul_forecastability_state") == "RUL_ACTIVE"]
        distributions: dict[str, Any] = {}
        for field in fields:
            values = [value for row in active if (value := _optional_float(row.get(field))) is not None]
            distributions[field] = {
                "rows": len(values),
                "mean": float(np.mean(values)) if values else None,
                "p10": _safe_quantile(values, 0.10),
                "median": _safe_quantile(values, 0.50),
                "p90": _safe_quantile(values, 0.90),
            }
        active_ids = {row.get("id") for row in active}
        reasons = Counter(
            row.get("rul_withholding_reason_code") or "NONE"
            for row in rows if row.get("id") not in active_ids
        )
        output[name] = {
            "active_rows": len(active),
            "active_lifecycles": len({row["lifecycle_id"] for row in active}),
            "active_batches": len({row["batch_id"] for row in active}),
            "distributions": distributions,
            "withholding_reason_distribution": dict(reasons),
            "machine_concentration": max(Counter(row["machine_id"] for row in active).values()) / len(active) if active else None,
            "batch_concentration": max(Counter(row["batch_id"] for row in active).values()) / len(active) if active else None,
        }
    return {
        "version": RUL_MODEL_VERSION_V2_5,
        "acceptance_diagnostics_are_post_freeze_and_may_not_tune_v2_5": True,
        "true_rul_horizons_are_audit_only": True,
        "populations": output,
    }


def evaluate_rul_v2_5_development_acceptance(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    freeze_manifest_path: str | Path,
    model_path: str | Path,
    frozen_v2_4_model_path: str | Path,
    output_dir: str | Path,
    sensor_config: SensorConfig,
    *,
    criteria: RULV25AcceptanceCriteria | None = None,
) -> dict[str, Any]:
    criteria = criteria or RULV25AcceptanceCriteria()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    registry_path = Path(consumed_registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    freeze = json.loads(Path(freeze_manifest_path).read_text(encoding="utf-8"))
    model = Path(model_path)
    acceptance = Path(manifest["role_files"]["acceptance"]["path"])
    if registry.get("v2_5_acceptance_opened") is not False:
        raise RuntimeError("v2.5 acceptance is already consumed; it cannot be rerun")
    assert_not_external_rul_evaluation_input(acceptance)
    integrity = {
        "model": _sha256(model) == freeze["model_sha256"],
        "manifest": _sha256(manifest_path) == freeze["manifest_sha256"],
        "acceptance": _sha256(acceptance) == freeze["acceptance_sha256"],
        "fit": manifest["role_files"]["fit"]["sha256"] == freeze["fit_sha256"],
        "calibration": manifest["role_files"]["calibration"]["sha256"] == freeze["calibration_sha256"],
        "rul_v2_5_code": _sha256(Path(__file__)) == freeze["runtime_code_sha256"]["rul_v2_5.py"],
        "rul_ml_code": _sha256(Path(__file__).with_name("rul_ml.py")) == freeze["runtime_code_sha256"]["rul_ml.py"],
        "features_code": _sha256(Path(__file__).with_name("rul_features_v2_4.py")) == freeze["runtime_code_sha256"]["rul_features_v2_4.py"],
        "evaluation_code": _sha256(Path(__file__).with_name("rul_evaluation.py")) == freeze["runtime_code_sha256"]["rul_evaluation.py"],
        "monitor_code": _sha256(Path(__file__).with_name("monitor.py")) == freeze["runtime_code_sha256"]["monitor.py"],
    }
    if not all(integrity.values()):
        raise RuntimeError(f"v2.5 preacceptance integrity failed: {integrity}")
    bundle = joblib.load(model)
    if _point_contract_hashes(bundle) != freeze["frozen_point_contract_hashes"]:
        raise RuntimeError("Frozen point-contract verification failed before acceptance")
    registry["v2_5_acceptance_opened"] = True
    registry["v2_5_acceptance_opened_at"] = datetime.now(timezone.utc).isoformat()
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime_report_path = output / "runtime_acceptance_replay_report.json"
    runtime_predictions_path = output / "runtime_acceptance_predictions.csv"

    def run_candidate() -> dict[str, Any]:
        return evaluate_rul(model, acceptance, runtime_report_path, runtime_predictions_path, sensor_config)

    runtime_report, memory = _run_with_memory_monitor(run_candidate)
    rows = _read_rows(runtime_predictions_path)
    baseline_report_path = output / "frozen_v2_4_same_acceptance_report.json"
    baseline_predictions_path = output / "frozen_v2_4_same_acceptance_predictions.csv"
    evaluate_rul(
        frozen_v2_4_model_path,
        acceptance,
        baseline_report_path,
        baseline_predictions_path,
        sensor_config,
    )
    baseline_rows = _read_rows(baseline_predictions_path)
    target_metrics = {target: _target_acceptance_metrics(rows, target) for target in TARGETS}
    baseline_critical = _target_acceptance_metrics(baseline_rows, "critical")
    critical = target_metrics["critical"]

    active_ids = {
        row["id"] for row in rows
        if row.get("rul_forecastability_state") == "RUL_ACTIVE"
        and _optional_float(row.get("estimated_hours_to_critical")) is not None
    }
    baseline_ids = {
        row["id"] for row in baseline_rows
        if _optional_float(row.get("estimated_hours_to_critical")) is not None
    }
    common = active_ids & baseline_ids
    new_widths = [
        float(row["critical_upper_hours"]) - float(row["critical_lower_hours"])
        for row in rows if row["id"] in common
    ]
    baseline_widths = [
        float(row["critical_upper_hours"]) - float(row["critical_lower_hours"])
        for row in baseline_rows if row["id"] in common
    ]
    baseline_mean = float(np.mean(baseline_widths)) if baseline_widths else None
    new_mean = float(np.mean(new_widths)) if new_widths else None
    normal_limit = min(1.20 * baseline_mean, baseline_mean + 8.0) if baseline_mean is not None else None
    hard_limit = min(1.35 * baseline_mean, baseline_mean + 16.0) if baseline_mean is not None else None
    macro_improvement = (
        critical["macro_lifecycle_coverage"] - baseline_critical["macro_lifecycle_coverage"]
        if critical["macro_lifecycle_coverage"] is not None and baseline_critical["macro_lifecycle_coverage"] is not None
        else None
    )
    width_pass = bool(
        new_mean is not None and normal_limit is not None and (
            new_mean <= normal_limit
            or (macro_improvement is not None and macro_improvement >= 0.10 and new_mean <= hard_limit)
        )
    )

    latency = _latency_report({"acceptance": rows})["acceptance"]
    artifact_bytes = model.stat().st_size
    stability = _state_stability(rows)
    rolling = _rolling_oscillation_metrics(rows)
    checks: dict[str, Any] = {}

    def record(name: str, observed: Any, criterion: str, passed: bool | None, reason: str | None = None) -> None:
        checks[name] = {
            "observed": observed,
            "criterion": criterion,
            "status": "PASS" if passed is True else "FAIL" if passed is False else "UNSUPPORTED",
            "reason": reason,
        }

    record("preacceptance_integrity", integrity, "all true", all(integrity.values()))
    record("frozen_point_contract", _point_contract_hashes(bundle), "exact hash match", _point_contract_hashes(bundle) == freeze["frozen_point_contract_hashes"])
    training = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    record("calibration_candidate_preacceptance", training["candidate_selection_passed"], "true", training["candidate_selection_passed"] is True)
    for target, metrics in target_metrics.items():
        record(f"{target}_active_mae", metrics["mae_hours"], "<=7 h", metrics["mae_hours"] is not None and metrics["mae_hours"] <= criteria.max_active_mae_hours)
        record(f"{target}_macro_active_mae", metrics["macro_lifecycle_mae_hours"], "<=8 h", metrics["macro_lifecycle_mae_hours"] is not None and metrics["macro_lifecycle_mae_hours"] <= criteria.max_active_macro_mae_hours)
        record(f"{target}_active_coverage", metrics["coverage"], ">=0.80", metrics["coverage"] is not None and metrics["coverage"] >= criteria.min_active_interval_coverage)
        record(f"{target}_macro_active_coverage", metrics["macro_lifecycle_coverage"], ">=0.75", metrics["macro_lifecycle_coverage"] is not None and metrics["macro_lifecycle_coverage"] >= criteria.min_active_macro_lifecycle_coverage)
        record(f"{target}_active_region_availability", metrics["active_region_availability"], ">=0.95", metrics["active_region_availability"] is not None and metrics["active_region_availability"] >= criteria.min_active_region_availability)
        for bucket, coverage_gate in HORIZON_COVERAGE_GATES.items():
            horizon = metrics["horizons"][bucket]
            supported = horizon["lifecycles"] >= criteria.minimum_lifecycles_per_horizon and horizon["batches"] >= criteria.minimum_batches_per_horizon
            record(
                f"{target}_{bucket}_coverage",
                horizon["coverage"],
                f">={coverage_gate} with >=8 lifecycles and >=4 batches",
                (horizon["coverage"] is not None and horizon["coverage"] >= coverage_gate) if supported else None,
                None if supported else "decision-relevant active horizon unsupported",
            )
            bias = horizon["mean_signed_error_hours"]
            bias_gate = HORIZON_BIAS_GATES[bucket]
            record(
                f"{target}_{bucket}_bias",
                bias,
                f"abs bias <={bias_gate} h with support",
                (bias is not None and abs(bias) <= bias_gate) if supported else None,
                None if supported else "decision-relevant active horizon unsupported",
            )
    record("critical_active_row_floor", critical["active_row_fraction"], ">=0.20", critical["active_row_fraction"] is not None and critical["active_row_fraction"] >= criteria.minimum_active_row_fraction)
    record("critical_active_lifecycle_floor", critical["active_lifecycle_fraction"], ">=0.75", critical["active_lifecycle_fraction"] is not None and critical["active_lifecycle_fraction"] >= criteria.minimum_active_lifecycle_fraction)
    record("critical_active_batch_floor", critical["active_batch_fraction"], "=1.0", critical["active_batch_fraction"] == criteria.minimum_active_batch_fraction)
    record("critical_median_lifecycle_floor", critical["median_lifecycle_active_fraction"], ">=0.10", critical["median_lifecycle_active_fraction"] is not None and critical["median_lifecycle_active_fraction"] >= criteria.minimum_median_lifecycle_active_fraction)
    active_0_24 = sum(critical["horizons"][bucket]["active_rows"] for bucket in ("le_6h", "6_12h", "12_24h"))
    eligible_0_24 = sum(critical["horizons"][bucket]["eligible_rows"] for bucket in ("le_6h", "6_12h", "12_24h"))
    fraction_0_24 = active_0_24 / eligible_0_24 if eligible_0_24 else None
    fraction_24_48 = critical["horizons"]["24_48h"]["active_fraction"]
    record("critical_0_24h_breadth", fraction_0_24, ">=0.80", fraction_0_24 is not None and fraction_0_24 >= criteria.minimum_0_24h_active_fraction)
    record("critical_24_48h_breadth", fraction_24_48, ">=0.20", fraction_24_48 is not None and fraction_24_48 >= criteria.minimum_24_48h_active_fraction)
    baseline_active_0_24 = sum(
        baseline_critical["horizons"][bucket]["active_rows"]
        for bucket in ("le_6h", "6_12h", "12_24h")
    )
    baseline_eligible_0_24 = sum(
        baseline_critical["horizons"][bucket]["eligible_rows"]
        for bucket in ("le_6h", "6_12h", "12_24h")
    )
    baseline_fraction_0_24 = (
        baseline_active_0_24 / baseline_eligible_0_24 if baseline_eligible_0_24 else None
    )
    for name, observed, baseline in (
        ("overall", critical["active_row_fraction"], baseline_critical["active_row_fraction"]),
        ("0_24h", fraction_0_24, baseline_fraction_0_24),
        ("24_48h", fraction_24_48, baseline_critical["horizons"]["24_48h"]["active_fraction"]),
    ):
        record(f"critical_{name}_breadth_noninferiority", {"v2_5": observed, "v2_4": baseline}, ">= v2.4 same-cohort - 0.05", observed is not None and baseline is not None and observed >= baseline - criteria.breadth_noninferiority_tolerance)
    record("interval_width_guard_common_active", {"v2_5": new_mean, "v2_4": baseline_mean, "normal_limit": normal_limit, "hard_limit": hard_limit, "macro_coverage_improvement": macro_improvement, "common_rows": len(common)}, "normal cap or >=10pp macro improvement plus hard cap", width_pass)
    record("rolling_24h_state_stability", rolling["maximum_active_unavailable_transitions_per_lifecycle_rolling_24h"], "<=2", rolling["maximum_active_unavailable_transitions_per_lifecycle_rolling_24h"] <= criteria.max_rolling_24h_active_unavailable_oscillations)
    record("runtime_p95_latency", latency["p95_ms"], "<=100 ms", latency["p95_ms"] is not None and latency["p95_ms"] <= criteria.max_runtime_p95_ms_per_row)
    record("runtime_p99_latency", latency["p99_ms"], "<=250 ms", latency["p99_ms"] is not None and latency["p99_ms"] <= criteria.max_runtime_p99_ms_per_row)
    incremental_mb = memory["incremental_bytes"] / (1024 * 1024) if memory["incremental_bytes"] is not None else None
    record("runtime_incremental_memory", incremental_mb, "<=500 MB", incremental_mb is not None and incremental_mb <= criteria.max_incremental_memory_mb)
    record("artifact_size", artifact_bytes, "<=250 MB", artifact_bytes <= criteria.max_artifact_bytes)
    statuses = [value["status"] for value in checks.values()]
    passed = bool(statuses and all(status == "PASS" for status in statuses))

    shift = _population_shift(_read_rows(output / "calibration_runtime_predictions.csv"), rows)
    (output / "active_population_shift_report.json").write_text(json.dumps(shift, indent=2), encoding="utf-8")
    residual = _residual_report({"calibration": _read_rows(output / "calibration_runtime_predictions.csv"), "acceptance": rows})
    (output / "residual_distribution_report.json").write_text(json.dumps(residual, indent=2), encoding="utf-8")
    alignment = _alignment_report({"acceptance": rows})
    (output / "selector_service_alignment_report.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    performance = {
        "version": RUL_MODEL_VERSION_V2_5,
        "acceptance_latency_after_warmup": latency,
        "memory": memory,
        "incremental_memory_mb": incremental_mb,
        "artifact_bytes": artifact_bytes,
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "software": {"python": sys.version, "numpy": np.__version__},
    }
    (output / "runtime_performance_report.json").write_text(json.dumps(performance, indent=2), encoding="utf-8")
    report = {
        "version": RUL_MODEL_VERSION_V2_5,
        "protocol_version": V2_5_PROTOCOL_VERSION,
        "development_only": True,
        "production_ready": False,
        "protected_external_datasets_used": False,
        "preacceptance_integrity": integrity,
        "frozen_point_contract_hashes": freeze["frozen_point_contract_hashes"],
        "model_sha256": _sha256(model),
        "criteria": asdict(criteria),
        "target_metrics": target_metrics,
        "frozen_v2_4_same_acceptance_critical": baseline_critical,
        "common_active_width_comparison": {
            "rows": len(common), "v2_5_mean_width_hours": new_mean,
            "v2_4_mean_width_hours": baseline_mean, "normal_limit_hours": normal_limit,
            "hard_exception_limit_hours": hard_limit,
        },
        "state_stability": {"legacy_summary": stability, "rolling_24h": rolling},
        "runtime_performance": performance,
        "checks": checks,
        "all_required_gates_passed": passed,
        "sealed_holdout_authorized": passed,
        "sealed_holdout_generated_or_evaluated": False,
    }
    (output / "development_acceptance_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    training.update({
        "acceptance_status": "PASS" if passed else "FAIL",
        "sealed_holdout_authorized": passed,
        "acceptance_opened_after_freeze": True,
    })
    (output / "training_report.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry.update({
        "v2_5_acceptance_consumed": True,
        "v2_5_acceptance_consumed_at": datetime.now(timezone.utc).isoformat(),
        "v2_5_acceptance_model_sha256": _sha256(model),
        "v2_5_acceptance_result": "PASS" if passed else "FAIL",
        "sealed_holdout_authorized": passed,
    })
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return report
