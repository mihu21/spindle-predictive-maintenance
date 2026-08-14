from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import SensorConfig
from .rul_evaluation import evaluate_rul
from .rul_ml import (
    RUL_CALIBRATION_METHOD_V2_1,
    RUL_CALIBRATION_METHOD_V2_6,
    RUL_MODEL_VERSION_V2_5,
    RUL_MODEL_VERSION_V2_6,
    _finite_sample_conformal_quantile,
    _lifecycle_conformal_margin,
    apply_v2_6_bias_correction,
)
from .rul_v2_4 import (
    ROLE_ORDER,
    V2_4_GENERATOR_VERSION,
    _batch_summary,
    _json_hash,
    _optional_float,
    _read_rows,
    _sha256,
    _write_rows,
)
from .rul_v2_5 import _complete_lifecycle_design_sample, _point_contract_hashes
from .synthetic import SyntheticConfig, generate_mock_csv
from .training import assert_not_external_rul_evaluation_input


TARGETS = ("warning", "critical")
V2_6_CORPUS_VERSION = "rul_v2_6_deferred_acceptance_corpus_v1"
V2_6_PROTOCOL_VERSION = "target_specific_direct_interval_full_cadence_v2"
V2_6_GENERATOR_VERSION = V2_4_GENERATOR_VERSION
SELECTOR_COLUMNS = (
    "rul_forecastability_score", "estimated_hours_to_critical", "rul_support_distance",
    "rul_neighbor_dispersion_hours", "rul_model_disagreement_hours", "degradation_score",
)


@dataclass(frozen=True)
class RULV26AcceptanceCriteria:
    max_critical_mae_hours: float = 7.0
    max_warning_mae_hours: float = 7.0
    min_interval_coverage: float = 0.80
    min_macro_lifecycle_coverage: float = 0.75
    min_active_region_availability: float = 0.95
    max_absolute_horizon_bias_hours: float = 6.0
    max_parity_float_difference: float = 1e-9
    max_artifact_bytes: int = 250 * 1024 * 1024
    max_runtime_p95_ms_per_row: float = 100.0
    max_runtime_p99_ms_per_row: float = 250.0
    max_incremental_memory_mb: float = 500.0
    max_warning_mean_interval_width_hours: float = 24.0
    max_critical_mean_interval_width_hours: float = 42.0
    minimum_active_row_fraction: float = 0.20
    minimum_active_lifecycle_fraction: float = 0.75
    minimum_active_batch_fraction: float = 1.0
    minimum_lifecycles_per_horizon: int = 8
    minimum_batches_per_horizon: int = 4


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _frozen_point_estimator_hashes(bundle: dict[str, Any]) -> dict[str, Any]:
    """Freeze only causal features, stabilization, and point estimators.

    Legacy point bias and residual quantiles are calibration components, not part of the frozen
    Extra Trees estimator contract. v2.6 remediation deliberately replaces them.
    """
    full = _point_contract_hashes(bundle)
    return {
        "feature_names_hash": full["feature_names_hash"],
        "base_feature_names_hash": full["base_feature_names_hash"],
        "stabilization_hash": full["stabilization_hash"],
        **{
            target: {"estimator_contract": full[target]["estimator_contract"]}
            for target in TARGETS
        },
    }


def _consumed_seeds() -> set[int]:
    seeds = {12001, 14001}
    for version in ("rul_v2_2", "rul_v2_3", "rul_v2_4", "rul_v2_5"):
        path = Path("output") / version / "corpus_manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for batch in manifest.get("batches", []):
            if batch.get("seed") is not None:
                seeds.add(int(batch["seed"]))
    return seeds


def _generate_role(
    root: Path,
    role: str,
    seeds: list[int],
    *,
    lifecycles_per_batch: int,
    machines: int,
    cadence_seconds: int,
    line_sel: str,
    ordinal_start: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    role_rows: list[dict[str, str]] = []
    batches: list[dict[str, Any]] = []
    next_id = ordinal_start * 10_000_000 + 1
    for index, seed in enumerate(seeds, start=1):
        batch_id = f"v26{role[:3]}_b{index:03d}_s{seed}"
        batch_path = root / "batches" / f"{batch_id}.csv"
        generate_mock_csv(
            batch_path,
            SyntheticConfig(
                lifecycles=lifecycles_per_batch, cadence_seconds=cadence_seconds, seed=seed,
                machines=machines, line_sel=line_sel,
                start_time=datetime(2105, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=370 * (ordinal_start + index)),
                identity_prefix=batch_id, duration_min_hours=54.0, duration_max_hours=156.0,
                causal_hazard_coupling=True,
            ),
        )
        rows = _read_rows(batch_path)
        for row in rows:
            row["id"] = str(next_id)
            row["generation_seed"] = str(seed)
            row["batch_id"] = batch_id
            row["development_role"] = role
            next_id += 1
        role_rows.extend(rows)
        batches.append({
            "role": role, "batch_id": batch_id, "seed": seed,
            "generator_version": V2_6_GENERATOR_VERSION,
            "source_file": str(batch_path.resolve()), "source_sha256": _sha256(batch_path),
            "generation_timestamp": datetime.now(timezone.utc).isoformat(), **_batch_summary(rows),
        })
    role_path = root / f"{role}.csv"
    assert_not_external_rul_evaluation_input(role_path)
    _write_rows(role_path, role_rows)
    return ({
        "path": str(role_path.resolve()), "sha256": _sha256(role_path), "rows": len(role_rows),
        "batches": len(seeds), "lifecycles": len({row["lifecycle_id"] for row in role_rows}),
        "machines": len({row["machine_id"] for row in role_rows}),
    }, batches)


def generate_rul_v2_6_preacceptance_corpus(
    data_dir: str | Path,
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    *,
    role_seeds: dict[str, Iterable[int]],
    lifecycles_per_batch: int = 8,
    machines: int = 4,
    cadence_seconds: int = 600,
    line_sel: str = "LINE_1",
) -> dict[str, Any]:
    fit = [int(value) for value in role_seeds.get("fit", [])]
    calibration = [int(value) for value in role_seeds.get("calibration", [])]
    acceptance = [int(value) for value in role_seeds.get("acceptance", [])]
    if len(fit) < 4 or len(calibration) < 4 or len(acceptance) < 4:
        raise ValueError("v2.6 requires at least four predeclared batches per role")
    all_seeds = fit + calibration + acceptance
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("A v2.6 seed may belong to only one evidence role")
    overlap = sorted(set(all_seeds) & _consumed_seeds())
    if overlap:
        raise ValueError(f"Protected or consumed seeds cannot be reused in v2.6: {overlap}")
    root = Path(data_dir)
    (root / "batches").mkdir(parents=True, exist_ok=True)
    role_files: dict[str, Any] = {}
    batches: list[dict[str, Any]] = []
    for ordinal, (role, seeds) in enumerate((("fit", fit), ("calibration", calibration)), start=1):
        role_files[role], local = _generate_role(
            root, role, seeds, lifecycles_per_batch=lifecycles_per_batch, machines=machines,
            cadence_seconds=cadence_seconds, line_sel=line_sel, ordinal_start=ordinal * 100,
        )
        batches.extend(local)
    manifest = {
        "corpus_version": V2_6_CORPUS_VERSION, "protocol_version": V2_6_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "development_only": True,
        "generator_version": V2_6_GENERATOR_VERSION, "generator_changed_from_v2_5": False,
        "physical_role_files": True, "role_assignments_permanent": True,
        "role_files": role_files, "batches": batches,
        "acceptance": {
            "status": "PREDECLARED_NOT_GENERATED", "seeds": acceptance,
            "lifecycles_per_batch": lifecycles_per_batch, "machines": machines,
            "cadence_seconds": cadence_seconds, "line_sel": line_sel,
            "content_parsed_before_parity": False,
        },
        "protected_external_datasets_used": False,
    }
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = {
        "registry_version": "permanent_consumed_rul_evidence_registry_v4",
        "role_reassignment_permitted": False, "v2_6_acceptance_generated": False,
        "v2_6_acceptance_opened": False, "v2_6_acceptance_consumed": False,
        "warning_sealed_holdout_authorized": False,
        "critical_sealed_holdout_authorized": False, "system_sealed_holdout_authorized": False,
        "v2_6_role_locked_batches": [
            {"batch_id": row["batch_id"], "seed": row["seed"], "role": row["role"], "sha256": row["source_sha256"]}
            for row in batches
        ],
    }
    registry_path = Path(consumed_registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return manifest


def _selector_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    matrix = []
    for row in rows:
        values = []
        for name in SELECTOR_COLUMNS:
            value = _optional_float(row.get(name))
            if name == "estimated_hours_to_critical" and value is None:
                value = _optional_float(row.get("critical_rul_raw_point_hours"))
            values.append(value)
        matrix.append([0.0 if value is None else value for value in values])
    return np.asarray(matrix, dtype=float)


def _target_design_rows(rows: list[dict[str, str]], target: str) -> list[dict[str, Any]]:
    truth_key = f"true_hours_to_{target}"
    point_key = f"estimated_hours_to_{target}"
    output = []
    for row in rows:
        truth = _optional_float(row.get(truth_key))
        point = _optional_float(row.get(f"{target}_rul_raw_point_hours"))
        if point is None:
            point = _optional_float(row.get(point_key))
        if truth is None or point is None or truth <= 0.0:
            continue
        output.append({**row, "truth": truth, "point": point, "error": point - truth})
    return output


def _selector_label(row: dict[str, Any], target: str) -> int:
    tolerance = 6.0 if target == "warning" else 8.0
    horizon = 24.0 if target == "warning" else 48.0
    return int(row["truth"] <= horizon and abs(row["error"]) <= tolerance)


def _fit_selector(rows: list[dict[str, Any]], target: str) -> Any | None:
    labels = np.asarray([_selector_label(row, target) for row in rows], dtype=int)
    if len(set(labels.tolist())) < 2:
        return None
    model = LogisticRegression(C=0.25, class_weight="balanced", max_iter=500, random_state=2606)
    model.fit(_selector_matrix(rows), labels)
    return model


def _selector_scores(model: Any | None, rows: list[dict[str, Any]]) -> np.ndarray:
    if model is None:
        return np.asarray([_optional_float(row.get("rul_forecastability_score")) or 0.0 for row in rows])
    probabilities = model.predict_proba(_selector_matrix(rows))
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(rows))


def _fit_bias_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"global": {"correction_hours": 0.0}, "bands": [], "max_forecast_hours": 720.0}
    residuals = np.asarray([row["truth"] - row["point"] for row in rows], dtype=float)
    global_correction = float(np.median(residuals))
    bands = []
    for name, lower, upper in (("LE_12", 0.0, 12.0), ("12_24", 12.0, 24.0), ("24_48", 24.0, 48.0), ("GT_48", 48.0, math.inf)):
        local = [row for row in rows if lower <= row["point"] < upper]
        lifecycles = {row["lifecycle_id"] for row in local}
        batches = {row["batch_id"] for row in local}
        supported = len(lifecycles) >= 8 and len(batches) >= 4
        correction = float(np.median([row["truth"] - row["point"] for row in local])) if local else global_correction
        correction = float(np.clip(correction, -12.0, 12.0))
        bands.append({
            "name": name, "lower_hours": lower, "upper_hours": upper,
            "correction_hours": correction, "rows": len(local), "lifecycles": len(lifecycles),
            "batches": len(batches), "supported": supported,
        })
    return {
        "method": "hierarchical_median_residual_horizon_bands_v1",
        "global": {"correction_hours": float(np.clip(global_correction, -12.0, 12.0))},
        "bands": bands, "minimum_lifecycles": 8, "minimum_batches": 4,
        "max_forecast_hours": 720.0,
    }


def _fit_identity_bias_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method": "identity_no_bias_correction",
        "global": {"correction_hours": 0.0}, "bands": [],
        "max_forecast_hours": 720.0,
    }


def _fit_linear_bias_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return _fit_identity_bias_map(rows)
    residuals = np.asarray([row["truth"] - row["point"] for row in rows], dtype=float)
    global_correction = float(np.clip(np.median(residuals), -12.0, 12.0))
    knots = []
    for name, lower, upper, center in (
        ("LE_12", 0.0, 12.0, 6.0),
        ("12_24", 12.0, 24.0, 18.0),
        ("24_48", 24.0, 48.0, 36.0),
    ):
        local = [row for row in rows if lower <= row["point"] < upper]
        lifecycles = {row["lifecycle_id"] for row in local}
        batches = {row["batch_id"] for row in local}
        supported = len(lifecycles) >= 8 and len(batches) >= 4
        local_median = (
            float(np.median([row["truth"] - row["point"] for row in local]))
            if local and supported else global_correction
        )
        shrinkage = len(local) / (len(local) + 50.0) if supported else 0.0
        correction = shrinkage * local_median + (1.0 - shrinkage) * global_correction
        knots.append({
            "name": name, "point_hours": center,
            "correction_hours": float(np.clip(correction, -12.0, 12.0)),
            "local_median_hours": local_median, "shrinkage_weight": shrinkage,
            "rows": len(local), "lifecycles": len(lifecycles), "batches": len(batches),
            "supported": supported,
        })
    return {
        "method": "shrunk_piecewise_linear_residual_correction_v1",
        "global": {"correction_hours": global_correction}, "knots": knots,
        "minimum_lifecycles": 8, "minimum_batches": 4,
        "max_forecast_hours": 720.0,
    }


BIAS_CORRECTION_FITTERS = {
    "identity": _fit_identity_bias_map,
    "fixed_horizon_bands": _fit_bias_map,
    "shrunk_piecewise_linear": _fit_linear_bias_map,
}


def _bias_crossfit(
    rows: list[dict[str, Any]], candidate: str = "fixed_horizon_bands",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fitter = BIAS_CORRECTION_FITTERS[candidate]
    predicted: list[dict[str, Any]] = []
    for held_batch in sorted({row["batch_id"] for row in rows}):
        training = [row for row in rows if row["batch_id"] != held_batch]
        held = [row for row in rows if row["batch_id"] == held_batch]
        bias = fitter(training)
        for row in held:
            corrected, stratum = apply_v2_6_bias_correction(row["point"], bias)
            predicted.append({**row, "corrected": corrected, "bias_stratum": stratum})
    raw = np.asarray([row["point"] - row["truth"] for row in predicted], dtype=float)
    corrected = np.asarray([row["corrected"] - row["truth"] for row in predicted], dtype=float)
    report = {
        "candidate": candidate,
        "rows": len(predicted), "batches": len({row["batch_id"] for row in predicted}),
        "lifecycles": len({row["lifecycle_id"] for row in predicted}),
        "raw_mae_hours": float(np.mean(np.abs(raw))), "corrected_mae_hours": float(np.mean(np.abs(corrected))),
        "raw_bias_hours": float(np.mean(raw)), "corrected_bias_hours": float(np.mean(corrected)),
    }
    report["survives_ablation"] = bool(
        candidate != "identity"
        and abs(report["corrected_bias_hours"]) < abs(report["raw_bias_hours"])
        and report["corrected_mae_hours"] <= report["raw_mae_hours"] + 0.5
    )
    return predicted, report


def _compare_bias_corrections(
    rows: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    candidates: dict[str, Any] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    for name in BIAS_CORRECTION_FITTERS:
        predictions[name], candidates[name] = _bias_crossfit(rows, name)
    survivors = [
        name for name, report in candidates.items() if report["survives_ablation"]
    ]
    selected = (
        min(survivors, key=lambda name: (
            candidates[name]["corrected_mae_hours"],
            abs(candidates[name]["corrected_bias_hours"]),
        ))
        if survivors else "identity"
    )
    return selected, predictions[selected], {
        "selection_role": "fit_outer_batch_crossfit_only",
        "selection_rule": "survive bias and MAE guards, then lowest corrected MAE, then absolute bias",
        "selected_candidate": selected,
        "selected_survives_ablation": bool(candidates[selected]["survives_ablation"]),
        "candidates": candidates,
    }


def _fit_asymmetric_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lower_scores = np.asarray([max(row["raw_lower"] - row["truth"], 0.0) for row in rows], dtype=float)
    upper_scores = np.asarray([max(row["truth"] - row["raw_upper"], 0.0) for row in rows], dtype=float)
    groups = np.asarray([row["lifecycle_id"] for row in rows], dtype=object)
    lower_micro, lower_rank, lower_clipped = _finite_sample_conformal_quantile(lower_scores, 0.90)
    upper_micro, upper_rank, upper_clipped = _finite_sample_conformal_quantile(upper_scores, 0.90)
    lower_macro, lower_macro_meta = _lifecycle_conformal_margin(
        lower_scores, groups, within_lifecycle_coverage=0.90, lifecycle_coverage=0.875,
    )
    upper_macro, upper_macro_meta = _lifecycle_conformal_margin(
        upper_scores, groups, within_lifecycle_coverage=0.90, lifecycle_coverage=0.875,
    )
    lower = max(float(lower_micro), float(lower_macro))
    upper = max(float(upper_micro), float(upper_macro))
    return {
        "method": RUL_CALIBRATION_METHOD_V2_6, "candidate": "asymmetric_corrected_global",
        "strata": {"GLOBAL": {"lower_margin_hours": lower, "upper_margin_hours": upper,
                                "rows": len(rows), "lifecycles": len({row["lifecycle_id"] for row in rows}),
                                "batches": len({row["batch_id"] for row in rows}),
                                "lower_finite_sample_rank": int(lower_rank),
                                "upper_finite_sample_rank": int(upper_rank),
                                "lower_rank_clipped": bool(lower_clipped),
                                "upper_rank_clipped": bool(upper_clipped),
                                "lower_lifecycle_conformal": lower_macro_meta,
                                "upper_lifecycle_conformal": upper_macro_meta}},
        "fallback_rules": "unsupported target strata use GLOBAL", "max_forecast_hours": 720.0,
    }


def _calibration_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    selected = config["strata"]["GLOBAL"]
    lo = np.maximum(0.0, np.asarray([row["raw_lower"] for row in rows]) - selected["lower_margin_hours"])
    hi = np.minimum(720.0, np.asarray([row["raw_upper"] for row in rows]) + selected["upper_margin_hours"])
    truth = np.asarray([row["truth"] for row in rows])
    hit = (lo <= truth) & (truth <= hi)
    lifecycle = [float(np.mean(hit[np.asarray([row["lifecycle_id"] == name for row in rows])])) for name in sorted({row["lifecycle_id"] for row in rows})]
    return {
        "rows": len(rows), "coverage": float(np.mean(hit)) if len(hit) else None,
        "macro_lifecycle_coverage": float(np.mean(lifecycle)) if lifecycle else None,
        "mean_width_hours": float(np.mean(hi - lo)) if len(hit) else None,
    }


def _prepare_v2_6_bundle(frozen: dict[str, Any], contracts: dict[str, Any], *, provisional: bool) -> dict[str, Any]:
    artifact = frozen["rul_model"]
    if artifact.get("version") != RUL_MODEL_VERSION_V2_5:
        raise ValueError("v2.6 requires the frozen v2.5 artifact")
    point_hashes = _frozen_point_estimator_hashes(frozen)
    artifact["version"] = RUL_MODEL_VERSION_V2_6
    artifact["target_contracts"] = contracts
    artifact["v2_6_point_contract_hashes"] = point_hashes
    artifact["v2_6_development_only"] = True
    for target in TARGETS:
        calibration = dict(artifact["targets"][target].get("calibration") or {})
        calibration.update({
            "point_bias_hours": 0.0,
            "residual_p10_hours": 0.0,
            "residual_p90_hours": 0.0,
            "base_interval_source": "corrected_point_degenerate_before_asymmetric_conformal",
            "legacy_v2_5_residual_envelope_reused": False,
        })
        if provisional:
            calibration.update({"method": RUL_CALIBRATION_METHOD_V2_1, "global_margin_hours": 0.0,
                                "buckets": {}, "minimum_interval_widths": {}, "v2_6_provisional": True})
        else:
            calibration.update(contracts[target]["calibration"])
            calibration["v2_6_provisional"] = False
        artifact["targets"][target]["calibration"] = calibration
    forecastability = dict(artifact.get("forecastability_contract") or {})
    forecastability.update({
        "mode": "target_specific_v2_6", "shared_serviceability_decision": False,
        "hard_invalidation": "immediate RUL_UNAVAILABLE and exact outputs cleared",
        "soft_invalidation": "immediate RUL_LOW_CONFIDENCE and exact outputs cleared",
        "reactivation_confirmations": 3,
    })
    artifact["forecastability_contract"] = forecastability
    return frozen


def _population_shift(left: list[dict[str, Any]], right: list[dict[str, Any]], target: str) -> dict[str, Any]:
    fields = ("corrected", "score", "rul_support_distance", "point", "lifecycle_age")
    result = {}
    for field in fields:
        a_values = [_optional_float(row.get(field)) for row in left]
        b_values = [_optional_float(row.get(field)) for row in right]
        a = np.asarray([value for value in a_values if value is not None], dtype=float)
        b = np.asarray([value for value in b_values if value is not None], dtype=float)
        result[field] = {
            "fit_oof_median": float(np.median(a)) if len(a) else None,
            "calibration_median": float(np.median(b)) if len(b) else None,
            "median_shift": float(np.median(b) - np.median(a)) if len(a) and len(b) else None,
        }
    return {"target": target, "selection_evidence_only": True, "distributions": result}


def _enrich(rows: list[dict[str, Any]], target: str, bias: dict[str, Any], selector: Any | None) -> list[dict[str, Any]]:
    scores = _selector_scores(selector, rows)
    output = []
    for row, score in zip(rows, scores):
        corrected, stratum = apply_v2_6_bias_correction(row["point"], bias)
        output.append({
            **row, "corrected": corrected, "bias_stratum": stratum, "score": float(score),
            "lifecycle_age": _optional_float(row.get("rul_history_hours")),
        })
    return output


def _add_base_interval(
    rows: list[dict[str, Any]], target_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    horizon = float(dict(target_artifact.get("calibration") or {}).get("max_forecast_hours", 720.0))
    return [
        {
            **row,
            "raw_lower": float(np.clip(row["corrected"], 0.0, horizon)),
            "raw_upper": float(np.clip(row["corrected"], 0.0, horizon)),
        }
        for row in rows
    ]


def _runtime_active_calibration_rows(
    rows: list[dict[str, str]], target: str,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        status = str(row.get("predicted_status") or "").upper()
        observed_target = status == "CRITICAL" or (target == "warning" and status == "WARNING")
        truth = _optional_float(row.get(f"true_hours_to_{target}"))
        point = _optional_float(row.get(f"estimated_hours_to_{target}"))
        raw_lower = _optional_float(row.get(f"{target}_lower_hours"))
        raw_upper = _optional_float(row.get(f"{target}_upper_hours"))
        if (
            observed_target or truth is None or truth <= 0.0 or point is None
            or raw_lower is None or raw_upper is None
            or not _bool(row.get(f"{target}_rul_serviceable_intent"))
        ):
            continue
        output.append({
            **row, "truth": truth, "point": point, "corrected": point,
            "raw_lower": raw_lower, "raw_upper": raw_upper,
            "score": _optional_float(row.get(f"{target}_rul_forecastability_score")),
            "lifecycle_age": _optional_float(row.get("rul_history_hours")),
        })
    return output


def parity_replay(
    model_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    sensor_config: SensorConfig,
    *,
    tolerance: float = 1e-9,
    report_version: str = RUL_MODEL_VERSION_V2_6,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    first = output / "parity_first.csv"
    second = output / "parity_reloaded.csv"
    evaluate_rul(model_path, input_path, output / "parity_first_report.json", first, sensor_config)
    _ = joblib.load(model_path)
    evaluate_rul(model_path, input_path, output / "parity_reloaded_report.json", second, sensor_config)
    left = _read_rows(first)
    right = _read_rows(second)
    exact_fields = ["id", "lifecycle_id", "batch_id"]
    float_fields: list[str] = []
    for target in TARGETS:
        exact_fields.extend([
            f"{target}_rul_forecastability_state", f"{target}_rul_serviceable_intent",
            f"{target}_rul_withholding_reason_code", f"{target}_rul_calibration_stratum",
        ])
        float_fields.extend([
            f"{target}_rul_forecastability_score", f"{target}_rul_raw_point_hours",
            f"{target}_rul_corrected_point_hours", f"estimated_hours_to_{target}",
            f"{target}_lower_hours", f"{target}_upper_hours",
        ])
    mismatches = []
    covered_status_fields = [f"{target}_covered" for target in TARGETS]
    if len(left) != len(right):
        mismatches.append({"field": "row_count", "left": len(left), "right": len(right)})
    for pos, (a, b) in enumerate(zip(left, right)):
        for field in exact_fields:
            if str(a.get(field, "")) != str(b.get(field, "")):
                mismatches.append({"row": pos, "field": field, "left": a.get(field), "right": b.get(field)})
        for field in float_fields:
            av, bv = _optional_float(a.get(field)), _optional_float(b.get(field))
            if (av is None) != (bv is None) or (av is not None and abs(av - bv) > tolerance):
                mismatches.append({"row": pos, "field": field, "left": av, "right": bv})
        for target in TARGETS:
            def covered(row: dict[str, str]) -> bool | None:
                truth = _optional_float(row.get(f"true_hours_to_{target}"))
                lo = _optional_float(row.get(f"{target}_lower_hours"))
                hi = _optional_float(row.get(f"{target}_upper_hours"))
                return None if truth is None or lo is None or hi is None else bool(lo <= truth <= hi)
            av, bv = covered(a), covered(b)
            if av != bv:
                mismatches.append({"row": pos, "field": f"{target}_covered", "left": av, "right": bv})
        if len(mismatches) >= 100:
            break
    report = {
        "version": report_version, "rows": len(left), "tolerance": tolerance,
        "exact_fields": exact_fields, "float_fields": float_fields,
        "derived_exact_fields": covered_status_fields,
        "mismatch_count": len(mismatches), "first_mismatches": mismatches,
        "passed": len(mismatches) == 0,
        "first_predictions_sha256": _sha256(first), "reloaded_predictions_sha256": _sha256(second),
    }
    (output / "final_artifact_calibration_parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_rul_v2_6_candidate(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    frozen_v2_5_model_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    freeze_manifest_path: str | Path,
    sensor_config: SensorConfig,
    *,
    criteria: RULV26AcceptanceCriteria | None = None,
) -> dict[str, Any]:
    criteria = criteria or RULV26AcceptanceCriteria()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    registry = json.loads(Path(consumed_registry_path).read_text(encoding="utf-8"))
    if manifest.get("corpus_version") != V2_6_CORPUS_VERSION:
        raise ValueError("Unsupported v2.6 corpus manifest")
    if manifest.get("acceptance", {}).get("status") != "PREDECLARED_NOT_GENERATED":
        raise RuntimeError("v2.6 training requires acceptance to remain ungenerated")
    if registry.get("v2_6_acceptance_generated") is not False:
        raise RuntimeError("v2.6 acceptance is already generated")
    for role in ("fit", "calibration"):
        path = Path(manifest["role_files"][role]["path"])
        assert_not_external_rul_evaluation_input(path)
        if _sha256(path) != manifest["role_files"][role]["sha256"]:
            raise RuntimeError(f"v2.6 {role} hash mismatch")
    frozen_path = Path(frozen_v2_5_model_path)
    frozen = joblib.load(frozen_path)
    frozen_hashes = _frozen_point_estimator_hashes(frozen)
    empty_contracts = {
        target: {
            "activation_threshold": 0.5, "deactivation_threshold": 0.45, "confirmations": 3,
            "maximum_exact_rul_horizon_hours": 24.0 if target == "warning" else 48.0,
            "bias_correction": {"global": {"correction_hours": 0.0}, "bands": []},
        }
        for target in TARGETS
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_prepare_v2_6_bundle(joblib.load(frozen_path), empty_contracts, provisional=True), model_output)
    provisional_contract_hash = _json_hash({
        "version": RUL_MODEL_VERSION_V2_6, "point_contract": frozen_hashes,
        "target_contracts": empty_contracts,
    })
    fit_design_path = output / "fit_design_complete_lifecycles.csv"
    fit_design = _complete_lifecycle_design_sample(
        manifest["role_files"]["fit"]["path"], fit_design_path,
        lifecycles_per_batch=3,
    )
    populations = {}
    for role in ("fit",):
        predictions = output / f"{role}_provisional_predictions.csv"
        checkpoint_path = output / f"{role}_provisional_checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
        role_input = fit_design_path
        role_input_sha256 = fit_design["sha256"]
        expected_rows = fit_design["rows"]
        reusable = bool(
            predictions.exists()
            and checkpoint.get("provisional_contract_hash") == provisional_contract_hash
            and checkpoint.get("input_sha256") == role_input_sha256
            and checkpoint.get("prediction_rows") == expected_rows
        )
        if not reusable:
            evaluate_rul(model_output, role_input, output / f"{role}_provisional_report.json", predictions, sensor_config)
            checkpoint_path.write_text(json.dumps({
                "provisional_contract_hash": provisional_contract_hash,
                "input_sha256": role_input_sha256,
                "prediction_sha256": _sha256(predictions),
                "prediction_rows": len(_read_rows(predictions)),
            }, indent=2), encoding="utf-8")
        populations[role] = _read_rows(predictions)
    contracts: dict[str, Any] = {}
    selection_reports: dict[str, Any] = {}
    shift_reports: dict[str, Any] = {}
    fit_enriched_by_target: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        fit_rows = _target_design_rows(populations["fit"], target)
        batches = sorted({row["batch_id"] for row in fit_rows})
        selector_oof = []
        for held_batch in batches:
            training = [row for row in fit_rows if row["batch_id"] != held_batch]
            held = [row for row in fit_rows if row["batch_id"] == held_batch]
            model = _fit_selector(training, target)
            scores = _selector_scores(model, held)
            selector_oof.extend([{**row, "selector_oof_score": float(score)} for row, score in zip(held, scores)])
        selected_fit = [row for row in selector_oof if row["selector_oof_score"] >= 0.5]
        if not selected_fit:
            selected_fit = selector_oof
        selected_bias, crossfit_rows, bias_comparison = _compare_bias_corrections(selected_fit)
        ablation = bias_comparison["candidates"][selected_bias]
        final_selector = _fit_selector(fit_rows, target)
        final_bias = BIAS_CORRECTION_FITTERS[selected_bias](selected_fit)
        zero_calibration = {
            "method": RUL_CALIBRATION_METHOD_V2_6,
            "candidate": "asymmetric_corrected_global",
            "strata": {"GLOBAL": {"lower_margin_hours": 0.0, "upper_margin_hours": 0.0}},
            "max_forecast_hours": 720.0,
        }
        contracts[target] = {
            "selector_model": final_selector, "selector_columns": list(SELECTOR_COLUMNS),
            "activation_threshold": 0.5, "deactivation_threshold": 0.45, "confirmations": 3,
            "maximum_exact_rul_horizon_hours": 24.0 if target == "warning" else 48.0,
            "bias_correction": final_bias, "calibration": zero_calibration,
        }
        fit_enriched = _add_base_interval(
            _enrich(crossfit_rows, target, final_bias, None),
            frozen["rul_model"]["targets"][target],
        )
        selection_reports[target] = {
            "selector_outer_batch_crossfit": True, "bias_inner_oof_batch_disjoint": True,
            "selector_oof_rows": len(selector_oof), "selector_active_rows": len(selected_fit),
            "bias_correction_comparison": bias_comparison,
            "selected_bias_correction": selected_bias,
            "bias_ablation": ablation,
        }
        fit_enriched_by_target[target] = fit_enriched

    corrected_provisional = _prepare_v2_6_bundle(
        joblib.load(frozen_path), contracts, provisional=False,
    )
    joblib.dump(corrected_provisional, model_output)
    corrected_calibration_path = output / "calibration_corrected_uncalibrated_predictions.csv"
    evaluate_rul(
        model_output, manifest["role_files"]["calibration"]["path"],
        output / "calibration_corrected_uncalibrated_report.json",
        corrected_calibration_path, sensor_config,
    )
    corrected_calibration_rows = _read_rows(corrected_calibration_path)
    for target in TARGETS:
        selected_cal = _runtime_active_calibration_rows(corrected_calibration_rows, target)
        if not selected_cal:
            raise RuntimeError(f"No final-runtime-serviceable {target} calibration rows")
        calibration = _fit_asymmetric_calibration(selected_cal)
        contracts[target]["calibration"] = calibration
        selection_reports[target]["calibration_population"] = "exact corrected runtime serviceable intent"
        selection_reports[target]["calibration_metrics"] = _calibration_metrics(selected_cal, calibration)
        shift_reports[target] = _population_shift(
            fit_enriched_by_target[target], selected_cal, target,
        )
    final_bundle = _prepare_v2_6_bundle(joblib.load(frozen_path), contracts, provisional=False)
    if _frozen_point_estimator_hashes(final_bundle) != frozen_hashes:
        raise RuntimeError("v2.6 modified the frozen point predictor contract")
    joblib.dump(final_bundle, model_output)
    parity = parity_replay(
        model_output, manifest["role_files"]["calibration"]["path"], output, sensor_config,
        tolerance=criteria.max_parity_float_difference,
    )
    if not parity["passed"]:
        raise RuntimeError("v2.6 final-artifact calibration parity failed; acceptance remains ungenerated")
    parity_rows = _read_rows(output / "parity_reloaded.csv")
    preacceptance_metrics = {
        target: _acceptance_metrics(parity_rows, target) for target in TARGETS
    }
    preacceptance_checks: dict[str, bool] = {}
    for target in TARGETS:
        row = preacceptance_metrics[target]
        width_limit = (
            criteria.max_warning_mean_interval_width_hours
            if target == "warning" else criteria.max_critical_mean_interval_width_hours
        )
        preacceptance_checks.update({
            f"{target}_bias_component_selection": bool(
                selection_reports[target]["selected_bias_correction"] == "identity"
                or selection_reports[target]["bias_ablation"]["survives_ablation"]
            ),
            f"{target}_availability": row["active_region_availability"] is not None and row["active_region_availability"] >= criteria.min_active_region_availability,
            f"{target}_coverage": row["interval_coverage"] is not None and row["interval_coverage"] >= criteria.min_interval_coverage,
            f"{target}_macro_coverage": row["macro_lifecycle_coverage"] is not None and row["macro_lifecycle_coverage"] >= criteria.min_macro_lifecycle_coverage,
            f"{target}_width": row["mean_interval_width_hours"] is not None and row["mean_interval_width_hours"] <= width_limit,
            f"{target}_row_breadth": row["active_row_fraction"] is not None and row["active_row_fraction"] >= criteria.minimum_active_row_fraction,
            f"{target}_lifecycle_breadth": row["active_lifecycle_fraction"] is not None and row["active_lifecycle_fraction"] >= criteria.minimum_active_lifecycle_fraction,
            f"{target}_batch_breadth": row["active_batch_fraction"] is not None and row["active_batch_fraction"] >= criteria.minimum_active_batch_fraction,
        })
        for horizon, diagnostic in row["horizon_bias"].items():
            if diagnostic["rows"]:
                supported = (
                    diagnostic["lifecycles"] >= criteria.minimum_lifecycles_per_horizon
                    and diagnostic["batches"] >= criteria.minimum_batches_per_horizon
                )
                preacceptance_checks[f"{target}_{horizon}_support"] = supported
                preacceptance_checks[f"{target}_{horizon}_bias"] = bool(
                    supported and abs(diagnostic["bias_hours"]) <= criteria.max_absolute_horizon_bias_hours
                )
    acceptance_generation_authorized = all(preacceptance_checks.values())
    freeze = {
        "freeze_version": "rul_v2_6_preacceptance_freeze_v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "model_path": str(model_output.resolve()),
        "model_sha256": _sha256(model_output), "frozen_v2_5_model_sha256": _sha256(frozen_path),
        "frozen_point_contract_hashes": frozen_hashes, "manifest_sha256": _sha256(manifest_path),
        "fit_sha256": manifest["role_files"]["fit"]["sha256"],
        "calibration_sha256": manifest["role_files"]["calibration"]["sha256"],
        "acceptance_status": "PREDECLARED_NOT_GENERATED", "parity_report": parity,
        "preacceptance_metrics": preacceptance_metrics,
        "preacceptance_checks": preacceptance_checks,
        "acceptance_generation_authorized": acceptance_generation_authorized,
        "target_contract_hash": _json_hash({
            target: {key: value for key, value in contracts[target].items() if key != "selector_model"}
            for target in TARGETS
        }),
        "criteria": asdict(criteria), "acceptance_content_parsed_before_freeze": False,
        "runtime_code_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in ("rul_v2_6.py", "rul_ml.py", "rul_features_v2_4.py", "rul_evaluation.py", "monitor.py")
        },
    }
    freeze_path = Path(freeze_manifest_path)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    (output / "target_specific_selection_report.json").write_text(json.dumps(selection_reports, indent=2), encoding="utf-8")
    (output / "preacceptance_population_shift_report.json").write_text(json.dumps(shift_reports, indent=2), encoding="utf-8")
    report = {
        "version": RUL_MODEL_VERSION_V2_6, "protocol_version": V2_6_PROTOCOL_VERSION,
        "development_only": True, "point_predictor_retrained": False, "generator_changed": False,
        "feature_contract_changed": False, "protected_external_datasets_used": False,
        "target_separation": True, "target_selection": selection_reports,
        "fit_design_complete_lifecycle_sample": fit_design,
        "final_artifact_calibration_parity": parity, "acceptance_status": "NOT_GENERATED",
        "preacceptance_metrics": preacceptance_metrics,
        "preacceptance_checks": preacceptance_checks,
        "preacceptance_candidate_gates_passed": acceptance_generation_authorized,
        "acceptance_generation_authorized": acceptance_generation_authorized,
        "sealed_holdout_authorized": False,
        "model_sha256": freeze["model_sha256"],
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def generate_rul_v2_6_acceptance_after_parity(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    freeze_manifest_path: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry_path = Path(consumed_registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    freeze = json.loads(Path(freeze_manifest_path).read_text(encoding="utf-8"))
    if not freeze.get("parity_report", {}).get("passed"):
        raise RuntimeError("Acceptance generation is forbidden until final-artifact calibration parity passes")
    if not freeze.get("acceptance_generation_authorized"):
        raise RuntimeError("Acceptance generation is forbidden because preacceptance candidate gates failed")
    if freeze.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("Manifest changed after parity freeze")
    if registry.get("v2_6_acceptance_generated"):
        raise RuntimeError("v2.6 acceptance has already been generated")
    spec = manifest["acceptance"]
    if spec.get("status") != "PREDECLARED_NOT_GENERATED":
        raise RuntimeError("Acceptance is not in the predeclared ungenerated state")
    role_file, batches = _generate_role(
        Path(manifest["role_files"]["fit"]["path"]).parent, "acceptance",
        [int(seed) for seed in spec["seeds"]],
        lifecycles_per_batch=int(spec["lifecycles_per_batch"]), machines=int(spec["machines"]),
        cadence_seconds=int(spec["cadence_seconds"]), line_sel=str(spec["line_sel"]), ordinal_start=300,
    )
    manifest["role_files"]["acceptance"] = role_file
    manifest["batches"].extend(batches)
    manifest["acceptance"].update({
        "status": "GENERATED_UNOPENED", "sha256": role_file["sha256"],
        "content_parsed_before_parity": False,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry.update({
        "v2_6_acceptance_generated": True, "v2_6_acceptance_opened": False,
        "v2_6_acceptance_consumed": False,
    })
    registry["v2_6_role_locked_batches"].extend([
        {"batch_id": row["batch_id"], "seed": row["seed"], "role": row["role"], "sha256": row["source_sha256"]}
        for row in batches
    ])
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    freeze["manifest_sha256_after_acceptance_generation"] = _sha256(manifest_path)
    freeze["acceptance_sha256"] = role_file["sha256"]
    freeze["acceptance_generated_after_parity"] = True
    Path(freeze_manifest_path).write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return manifest


def _acceptance_metrics(rows: list[dict[str, str]], target: str) -> dict[str, Any]:
    truth_key = f"true_hours_to_{target}"
    point_key = f"estimated_hours_to_{target}"
    service_key = f"{target}_rul_serviceable_intent"
    maximum_horizon = 24.0 if target == "warning" else 48.0
    observed = [
        row for row in rows
        if (_optional_float(row.get(truth_key)) or 0.0) > 0.0
        and (
            str(row.get("predicted_status") or "").upper() == "CRITICAL"
            or (target == "warning" and str(row.get("predicted_status") or "").upper() == "WARNING")
        )
    ]
    positive_forecast = [
        row for row in rows
        if (_optional_float(row.get(truth_key)) or 0.0) > 0.0 and row not in observed
    ]
    eligible = [
        row for row in positive_forecast
        if float(_optional_float(row.get(truth_key))) <= maximum_horizon
    ]
    serviceable = [row for row in eligible if _bool(row.get(service_key))]
    emitted = [row for row in serviceable if _optional_float(row.get(point_key)) is not None]
    errors = np.asarray([_optional_float(row[point_key]) - _optional_float(row[truth_key]) for row in emitted])
    hits = []
    widths = []
    lifecycle_hits: dict[str, list[bool]] = defaultdict(list)
    for row in emitted:
        truth = _optional_float(row[truth_key])
        lo = _optional_float(row[f"{target}_lower_hours"])
        hi = _optional_float(row[f"{target}_upper_hours"])
        hit = bool(lo is not None and hi is not None and lo <= truth <= hi)
        hits.append(hit)
        if lo is not None and hi is not None:
            widths.append(hi - lo)
        lifecycle_hits[row["lifecycle_id"]].append(hit)
    horizons = {}
    for name, lower, upper in (("le_6h", 0, 6), ("6_12h", 6, 12), ("12_24h", 12, 24), ("24_48h", 24, 48)):
        local = [row for row in emitted if lower < _optional_float(row[truth_key]) <= upper]
        local_errors = np.asarray([_optional_float(row[point_key]) - _optional_float(row[truth_key]) for row in local])
        horizons[name] = {
            "rows": len(local),
            "lifecycles": len({row["lifecycle_id"] for row in local}),
            "batches": len({row["batch_id"] for row in local}),
            "bias_hours": float(np.mean(local_errors)) if len(local_errors) else None,
        }
    eligible_lifecycles = {row["lifecycle_id"] for row in eligible}
    eligible_batches = {row["batch_id"] for row in eligible}
    active_lifecycles = {row["lifecycle_id"] for row in serviceable}
    active_batches = {row["batch_id"] for row in serviceable}
    return {
        "contract_maximum_horizon_hours": maximum_horizon,
        "total_positive_forecast_rows": len(positive_forecast),
        "outside_contract_horizon_rows": len(positive_forecast) - len(eligible),
        "eligible_rows": len(eligible), "observed_target_override_rows": len(observed),
        "serviceable_intent_rows": len(serviceable), "emitted_rows": len(emitted),
        "active_row_fraction": len(serviceable) / len(eligible) if eligible else None,
        "active_lifecycle_fraction": len(active_lifecycles) / len(eligible_lifecycles) if eligible_lifecycles else None,
        "active_batch_fraction": len(active_batches) / len(eligible_batches) if eligible_batches else None,
        "active_region_availability": len(emitted) / len(serviceable) if serviceable else None,
        "mae_hours": float(np.mean(np.abs(errors))) if len(errors) else None,
        "mean_signed_error_hours": float(np.mean(errors)) if len(errors) else None,
        "interval_coverage": float(np.mean(hits)) if hits else None,
        "macro_lifecycle_coverage": float(np.mean([np.mean(value) for value in lifecycle_hits.values()])) if lifecycle_hits else None,
        "mean_interval_width_hours": float(np.mean(widths)) if widths else None,
        "p90_interval_width_hours": float(np.quantile(widths, 0.90)) if widths else None,
        "horizon_bias": horizons,
    }


def _distribution_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "rows": len(array), "p10": float(np.quantile(array, 0.10)) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "p90": float(np.quantile(array, 0.90)) if len(array) else None,
    }


def _runtime_shift_population(rows: list[dict[str, str]], target: str) -> dict[str, Any]:
    service_key = f"{target}_rul_serviceable_intent"
    active = [
        row for row in rows
        if _bool(row.get(service_key))
        and (_optional_float(row.get(f"true_hours_to_{target}")) or 0.0) > 0.0
        and str(row.get("predicted_status") or "").upper() != "CRITICAL"
        and not (target == "warning" and str(row.get("predicted_status") or "").upper() == "WARNING")
        and _optional_float(row.get(f"{target}_rul_corrected_point_hours")) is not None
    ]
    residuals = [
        _optional_float(row[f"{target}_rul_corrected_point_hours"])
        - _optional_float(row[f"true_hours_to_{target}"])
        for row in active
    ]
    numeric = {
        "corrected_residual_hours": residuals,
        "selector_score": [_optional_float(row.get(f"{target}_rul_forecastability_score")) for row in active],
        "support_score": [_optional_float(row.get("rul_support_distance")) for row in active],
        "predicted_rul_hours": [_optional_float(row.get(f"{target}_rul_corrected_point_hours")) for row in active],
        "lifecycle_age_hours": [_optional_float(row.get("rul_history_hours")) for row in active],
    }
    strata: dict[str, int] = defaultdict(int)
    bands: dict[str, int] = defaultdict(int)
    for row in active:
        strata[row.get(f"{target}_rul_calibration_stratum") or "NONE"] += 1
        degradation = _optional_float(row.get("degradation_score"))
        band = "UNKNOWN" if degradation is None else "LOW" if degradation < 0.35 else "MID" if degradation < 0.75 else "HIGH"
        bands[band] += 1
    return {
        "rows": len(active),
        "numeric": {
            key: _distribution_summary([value for value in values if value is not None])
            for key, values in numeric.items()
        },
        "calibration_stratum_mix": {key: value / len(active) for key, value in strata.items()} if active else {},
        "degradation_band_mix": {key: value / len(active) for key, value in bands.items()} if active else {},
    }


def postacceptance_shift_report(
    calibration_rows: list[dict[str, str]], acceptance_rows: list[dict[str, str]],
) -> dict[str, Any]:
    targets = {}
    for target in TARGETS:
        calibration = _runtime_shift_population(calibration_rows, target)
        acceptance = _runtime_shift_population(acceptance_rows, target)
        shifts = {}
        for name in calibration["numeric"]:
            left = calibration["numeric"][name]["median"]
            right = acceptance["numeric"][name]["median"]
            shifts[name] = None if left is None or right is None else right - left
        targets[target] = {
            "calibration": calibration, "acceptance": acceptance,
            "acceptance_minus_calibration_median_shift": shifts,
        }
    return {
        "version": RUL_MODEL_VERSION_V2_6,
        "role": "postacceptance_hypothesis_only_no_reselection_or_retuning",
        "targets": targets,
    }


def evaluate_rul_v2_6_development_acceptance(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    freeze_manifest_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    sensor_config: SensorConfig,
    *,
    criteria: RULV26AcceptanceCriteria | None = None,
    registry_namespace: str = "v2_6",
    report_version: str = RUL_MODEL_VERSION_V2_6,
    metrics_function: Any = _acceptance_metrics,
) -> dict[str, Any]:
    criteria = criteria or RULV26AcceptanceCriteria()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    registry_path = Path(consumed_registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    freeze = json.loads(Path(freeze_manifest_path).read_text(encoding="utf-8"))
    if manifest.get("acceptance", {}).get("status") != "GENERATED_UNOPENED":
        raise RuntimeError("Fresh v2.6 acceptance has not been generated")
    if registry.get(f"{registry_namespace}_acceptance_opened") or registry.get(f"{registry_namespace}_acceptance_consumed"):
        raise RuntimeError(f"{registry_namespace} acceptance can be evaluated only once")
    if freeze.get("model_sha256") != _sha256(model_path):
        raise RuntimeError(f"Frozen {registry_namespace} model hash mismatch")
    if freeze.get("manifest_sha256_after_acceptance_generation") != _sha256(manifest_path):
        raise RuntimeError(f"{registry_namespace} manifest changed after acceptance generation")
    code_hashes = freeze.get("runtime_code_sha256") or {}
    for name, expected in code_hashes.items():
        if _sha256(Path(__file__).with_name(name)) != expected:
            raise RuntimeError(f"Frozen {registry_namespace} runtime code changed: {name}")
    acceptance = Path(manifest["role_files"]["acceptance"]["path"])
    if _sha256(acceptance) != freeze.get("acceptance_sha256"):
        raise RuntimeError(f"Frozen {registry_namespace} acceptance hash mismatch")
    output = Path(output_dir)
    predictions = output / "acceptance_runtime_predictions.csv"
    rss_before = process_rss_bytes()
    evaluate_rul(model_path, acceptance, output / "acceptance_runtime_report.json", predictions, sensor_config)
    rss_after = process_rss_bytes()
    rows = _read_rows(predictions)
    metrics = {target: metrics_function(rows, target) for target in TARGETS}
    checks = {}
    for target in TARGETS:
        max_mae = criteria.max_warning_mae_hours if target == "warning" else criteria.max_critical_mae_hours
        row = metrics[target]
        checks[f"{target}_mae"] = row["mae_hours"] is not None and row["mae_hours"] <= max_mae
        checks[f"{target}_coverage"] = row["interval_coverage"] is not None and row["interval_coverage"] >= criteria.min_interval_coverage
        checks[f"{target}_macro_coverage"] = row["macro_lifecycle_coverage"] is not None and row["macro_lifecycle_coverage"] >= criteria.min_macro_lifecycle_coverage
        checks[f"{target}_availability"] = row["active_region_availability"] is not None and row["active_region_availability"] >= criteria.min_active_region_availability
        checks[f"{target}_active_row_fraction"] = row["active_row_fraction"] is not None and row["active_row_fraction"] >= criteria.minimum_active_row_fraction
        checks[f"{target}_active_lifecycle_fraction"] = row["active_lifecycle_fraction"] is not None and row["active_lifecycle_fraction"] >= criteria.minimum_active_lifecycle_fraction
        checks[f"{target}_active_batch_fraction"] = row["active_batch_fraction"] is not None and row["active_batch_fraction"] >= criteria.minimum_active_batch_fraction
        width_limit = criteria.max_warning_mean_interval_width_hours if target == "warning" else criteria.max_critical_mean_interval_width_hours
        checks[f"{target}_interval_width"] = row["mean_interval_width_hours"] is not None and row["mean_interval_width_hours"] <= width_limit
        if hasattr(criteria, "min_active_monotonicity"):
            checks[f"{target}_active_monotonicity"] = (
                row.get("active_monotonicity") is not None
                and row["active_monotonicity"] >= criteria.min_active_monotonicity
            )
        for horizon, diagnostic in row["horizon_bias"].items():
            if diagnostic["rows"]:
                supported = (
                    diagnostic["lifecycles"] >= criteria.minimum_lifecycles_per_horizon
                    and diagnostic["batches"] >= criteria.minimum_batches_per_horizon
                )
                checks[f"{target}_{horizon}_support"] = supported
                checks[f"{target}_{horizon}_bias"] = supported and abs(diagnostic["bias_hours"]) <= criteria.max_absolute_horizon_bias_hours
    target_pass = {
        target: all(value for key, value in checks.items() if key.startswith(f"{target}_"))
        for target in TARGETS
    }
    passed = all(target_pass.values())
    latencies = [
        value for row in rows[100:]
        if (value := _optional_float(row.get("rul_inference_latency_ms"))) is not None
    ]
    performance = {
        "warmup_rows_excluded": min(100, len(rows)), "measured_rows": len(latencies),
        "p95_ms_per_row": float(np.quantile(latencies, 0.95)) if latencies else None,
        "p99_ms_per_row": float(np.quantile(latencies, 0.99)) if latencies else None,
        "rss_before_bytes": rss_before, "rss_after_bytes": rss_after,
        "incremental_memory_mb": (
            max(0, rss_after - rss_before) / (1024.0 * 1024.0)
            if rss_before is not None and rss_after is not None else None
        ),
        "artifact_bytes": Path(model_path).stat().st_size,
        "environment": runtime_environment_report(),
    }
    checks.update({
        "runtime_p95": performance["p95_ms_per_row"] is not None and performance["p95_ms_per_row"] <= criteria.max_runtime_p95_ms_per_row,
        "runtime_p99": performance["p99_ms_per_row"] is not None and performance["p99_ms_per_row"] <= criteria.max_runtime_p99_ms_per_row,
        "incremental_memory": performance["incremental_memory_mb"] is not None and performance["incremental_memory_mb"] <= criteria.max_incremental_memory_mb,
        "artifact_size": performance["artifact_bytes"] <= criteria.max_artifact_bytes,
    })
    passed = passed and all(checks[key] for key in ("runtime_p95", "runtime_p99", "incremental_memory", "artifact_size"))
    calibration_predictions = output / "parity_reloaded.csv"
    shift = postacceptance_shift_report(
        _read_rows(calibration_predictions) if calibration_predictions.exists() else [], rows,
    )
    (output / "postacceptance_population_shift_report.json").write_text(json.dumps(shift, indent=2), encoding="utf-8")
    report = {
        "version": report_version, "criteria": asdict(criteria), "target_metrics": metrics,
        "checks": checks, "target_authorization": {
            "warning_sealed_holdout_authorized": target_pass["warning"],
            "critical_sealed_holdout_authorized": target_pass["critical"],
            "system_sealed_holdout_authorized": passed,
        },
        "all_required_gates_passed": passed, "sealed_holdout_generated_or_evaluated": False,
        "runtime_performance": performance,
        "postacceptance_shift_analysis_role": "hypothesis_only_no_reselection",
    }
    (output / "development_acceptance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    registry.update({
        f"{registry_namespace}_acceptance_opened": True,
        f"{registry_namespace}_acceptance_consumed": True,
        f"{registry_namespace}_acceptance_consumed_at": datetime.now(timezone.utc).isoformat(),
        f"{registry_namespace}_acceptance_result": "PASS" if passed else "FAIL",
        **report["target_authorization"],
    })
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return report


def process_rss_bytes() -> int | None:
    if os.name != "nt":
        try:
            import resource
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        except (ImportError, OSError):
            return None
    try:
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    except (AttributeError, OSError, ValueError):
        return None


def runtime_environment_report() -> dict[str, Any]:
    return {
        "platform": platform.platform(), "processor": platform.processor(),
        "python": sys.version, "numpy": np.__version__, "rss_bytes": process_rss_bytes(),
    }


def write_postacceptance_diagnostics_from_frozen_outputs(
    output_dir: str | Path, model_path: str | Path,
) -> dict[str, Any]:
    """Postprocess already-consumed runtime outputs without reopening acceptance evidence."""
    output = Path(output_dir)
    acceptance_path = output / "acceptance_runtime_predictions.csv"
    calibration_path = output / "parity_reloaded.csv"
    if not acceptance_path.exists() or not calibration_path.exists():
        raise FileNotFoundError("Frozen calibration and acceptance runtime outputs are required")
    acceptance_rows = _read_rows(acceptance_path)
    calibration_rows = _read_rows(calibration_path)
    shift = postacceptance_shift_report(calibration_rows, acceptance_rows)
    (output / "postacceptance_population_shift_report.json").write_text(
        json.dumps(shift, indent=2), encoding="utf-8",
    )
    latencies = [
        value for row in acceptance_rows[100:]
        if (value := _optional_float(row.get("rul_inference_latency_ms"))) is not None
    ]
    performance = {
        "source": "already_consumed_acceptance_runtime_predictions",
        "acceptance_reopened": False, "acceptance_inference_rerun": False,
        "warmup_rows_excluded": min(100, len(acceptance_rows)), "measured_rows": len(latencies),
        "p95_ms_per_row": float(np.quantile(latencies, 0.95)) if latencies else None,
        "p99_ms_per_row": float(np.quantile(latencies, 0.99)) if latencies else None,
        "artifact_bytes": Path(model_path).stat().st_size,
        "current_process_rss_probe_bytes": process_rss_bytes(),
        "incremental_acceptance_memory_mb": None,
        "incremental_memory_note": "not reconstructed after consumption; future evaluations capture before/after RSS",
        "environment": runtime_environment_report(),
    }
    (output / "postacceptance_runtime_performance.json").write_text(
        json.dumps(performance, indent=2), encoding="utf-8",
    )
    return {"shift": shift, "performance": performance}
