from __future__ import annotations

import json
from collections import defaultdict
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
    RUL_CALIBRATION_METHOD_V2_6,
    RUL_MODEL_VERSION_V2_6,
    RUL_MODEL_VERSION_V2_7,
    apply_v2_6_bias_correction,
)
from .rul_v2_4 import (
    V2_4_GENERATOR_VERSION,
    _batch_summary,
    _json_hash,
    _optional_float,
    _read_rows,
    _sha256,
    _write_rows,
)
from .rul_v2_5 import _complete_lifecycle_design_sample
from .rul_v2_6 import (
    SELECTOR_COLUMNS,
    TARGETS,
    RULV26AcceptanceCriteria,
    _acceptance_metrics,
    _add_base_interval,
    _bool,
    _calibration_metrics,
    _enrich,
    _fit_asymmetric_calibration,
    _fit_identity_bias_map,
    _fit_selector,
    _frozen_point_estimator_hashes,
    _population_shift,
    _prepare_v2_6_bundle,
    _runtime_active_calibration_rows,
    _selector_scores,
    _target_design_rows,
    evaluate_rul_v2_6_development_acceptance,
    parity_replay,
)
from .synthetic import SyntheticConfig, generate_mock_csv
from .training import assert_not_external_rul_evaluation_input


V2_7_CORPUS_VERSION = "rul_v2_7_identity_first_deferred_acceptance_corpus_v1"
V2_7_PROTOCOL_VERSION = "critical_identity_first_horizon_survival_full_cadence_v1"
V2_7_GENERATOR_VERSION = V2_4_GENERATOR_VERSION
HORIZONS = (
    ("le_6h", 0.0, 6.0),
    ("6_12h", 6.0, 12.0),
    ("12_24h", 12.0, 24.0),
    ("24_48h", 24.0, 48.0),
)


@dataclass(frozen=True)
class RULV27AcceptanceCriteria(RULV26AcceptanceCriteria):
    min_active_monotonicity: float = 0.95
    max_candidate_horizon_mae_increase_hours: float = 0.5
    maximum_correction_hours: float = 3.0
    maximum_correction_slope: float = 0.25


def _all_consumed_seeds() -> set[int]:
    seeds = {12001, 14001}
    for path in Path("output").glob("**/corpus_manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for batch in manifest.get("batches", []):
            if batch.get("seed") is not None:
                seeds.add(int(batch["seed"]))
        for seed in manifest.get("acceptance", {}).get("seeds", []):
            seeds.add(int(seed))
    return seeds


def _generate_role_v2_7(
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
        batch_id = f"v27{role[:3]}_b{index:03d}_s{seed}"
        batch_path = root / "batches" / f"{batch_id}.csv"
        generate_mock_csv(
            batch_path,
            SyntheticConfig(
                lifecycles=lifecycles_per_batch,
                cadence_seconds=cadence_seconds,
                seed=seed,
                machines=machines,
                line_sel=line_sel,
                start_time=datetime(2145, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=370 * (ordinal_start + index)),
                identity_prefix=batch_id,
                duration_min_hours=54.0,
                duration_max_hours=156.0,
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
            "role": role,
            "batch_id": batch_id,
            "seed": seed,
            "generator_version": V2_7_GENERATOR_VERSION,
            "source_file": str(batch_path.resolve()),
            "source_sha256": _sha256(batch_path),
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            **_batch_summary(rows),
        })
    role_path = root / f"{role}.csv"
    assert_not_external_rul_evaluation_input(role_path)
    _write_rows(role_path, role_rows)
    return ({
        "path": str(role_path.resolve()),
        "sha256": _sha256(role_path),
        "rows": len(role_rows),
        "batches": len(seeds),
        "lifecycles": len({row["lifecycle_id"] for row in role_rows}),
        "machines": len({row["machine_id"] for row in role_rows}),
    }, batches)


def generate_rul_v2_7_preacceptance_corpus(
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
        raise ValueError("v2.7 requires at least four predeclared batches per role")
    all_seeds = fit + calibration + acceptance
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("A v2.7 seed may belong to only one evidence role")
    overlap = sorted(set(all_seeds) & _all_consumed_seeds())
    if overlap:
        raise ValueError(f"Protected or consumed seeds cannot be reused in v2.7: {overlap}")
    root = Path(data_dir)
    (root / "batches").mkdir(parents=True, exist_ok=True)
    role_files: dict[str, Any] = {}
    batches: list[dict[str, Any]] = []
    for ordinal, (role, seeds) in enumerate((("fit", fit), ("calibration", calibration)), start=1):
        role_files[role], local = _generate_role_v2_7(
            root,
            role,
            seeds,
            lifecycles_per_batch=lifecycles_per_batch,
            machines=machines,
            cadence_seconds=cadence_seconds,
            line_sel=line_sel,
            ordinal_start=ordinal * 100,
        )
        batches.extend(local)
    manifest = {
        "corpus_version": V2_7_CORPUS_VERSION,
        "protocol_version": V2_7_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "generator_version": V2_7_GENERATOR_VERSION,
        "generator_changed_from_v2_6": False,
        "physical_role_files": True,
        "role_assignments_permanent": True,
        "role_files": role_files,
        "batches": batches,
        "acceptance": {
            "status": "PREDECLARED_NOT_GENERATED",
            "seeds": acceptance,
            "lifecycles_per_batch": lifecycles_per_batch,
            "machines": machines,
            "cadence_seconds": cadence_seconds,
            "line_sel": line_sel,
            "content_parsed_before_parity": False,
        },
        "protected_external_datasets_used": False,
        "v2_6_evidence_role": "hypothesis_generation_only_not_reused",
    }
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = {
        "registry_version": "permanent_consumed_rul_evidence_registry_v5",
        "role_reassignment_permitted": False,
        "v2_7_acceptance_generated": False,
        "v2_7_acceptance_opened": False,
        "v2_7_acceptance_consumed": False,
        "warning_sealed_holdout_authorized": False,
        "critical_sealed_holdout_authorized": False,
        "system_sealed_holdout_authorized": False,
        "v2_7_role_locked_batches": [
            {"batch_id": row["batch_id"], "seed": row["seed"], "role": row["role"], "sha256": row["source_sha256"]}
            for row in batches
        ],
    }
    registry_path = Path(consumed_registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return manifest


def _fit_zero_anchored_correction(
    rows: list[dict[str, Any]],
    *,
    maximum_correction_hours: float = 3.0,
    maximum_slope: float = 0.25,
) -> dict[str, Any]:
    knots = [{
        "name": "ZERO_ANCHOR",
        "point_hours": 0.0,
        "correction_hours": 0.0,
        "supported": True,
        "rows": 0,
        "lifecycles": 0,
        "batches": 0,
    }]
    previous_point = 0.0
    previous_correction = 0.0
    for name, lower, upper, point in (
        ("LE_12", 0.0, 12.0, 12.0),
        ("12_24", 12.0, 24.0, 24.0),
        ("24_48", 24.0, 48.0, 48.0),
    ):
        local = [row for row in rows if lower <= row["point"] < upper]
        lifecycles = {row["lifecycle_id"] for row in local}
        batches = {row["batch_id"] for row in local}
        supported = len(lifecycles) >= 8 and len(batches) >= 4
        local_median = float(np.median([row["truth"] - row["point"] for row in local])) if local and supported else 0.0
        shrinkage = len(local) / (len(local) + 50.0) if supported else 0.0
        proposed = float(np.clip(
            shrinkage * local_median,
            -maximum_correction_hours,
            maximum_correction_hours,
        ))
        maximum_delta = maximum_slope * (point - previous_point)
        correction = float(np.clip(
            proposed,
            previous_correction - maximum_delta,
            previous_correction + maximum_delta,
        ))
        knots.append({
            "name": name,
            "point_hours": point,
            "correction_hours": correction,
            "local_median_hours": local_median,
            "shrinkage_weight": shrinkage,
            "supported": supported,
            "rows": len(local),
            "lifecycles": len(lifecycles),
            "batches": len(batches),
        })
        previous_point = point
        previous_correction = correction
    return {
        "method": "zero_anchored_bounded_piecewise_linear_v2_7",
        "knots": knots,
        "correction_at_zero_hours": 0.0,
        "maximum_absolute_correction_hours": maximum_correction_hours,
        "maximum_absolute_slope": maximum_slope,
        "runtime_inputs": ["predicted_critical_rul_hours"],
        "uses_true_horizon_at_runtime": False,
        "max_forecast_hours": 720.0,
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _macro_metric(rows: list[dict[str, Any]], key: str, *, absolute: bool) -> float | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[key]) - float(row["truth"])
        grouped[row["lifecycle_id"]].append(abs(value) if absolute else value)
    return float(np.mean([np.mean(values) for values in grouped.values()])) if grouped else None


def _candidate_diagnostics(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    errors = [float(row["corrected"]) - float(row["truth"]) for row in rows]
    corrections = [float(row["corrected"]) - float(row["point"]) for row in rows]
    horizon_reports: dict[str, Any] = {}
    for horizon, lower, upper in HORIZONS:
        local = [row for row in rows if lower < float(row["truth"]) <= upper]
        local_errors = [float(row["corrected"]) - float(row["truth"]) for row in local]
        local_corrections = [float(row["corrected"]) - float(row["point"]) for row in local]
        by_batch: dict[str, list[float]] = defaultdict(list)
        by_lifecycle: dict[str, list[float]] = defaultdict(list)
        for row, error in zip(local, local_errors):
            by_batch[row["batch_id"]].append(error)
            by_lifecycle[row["lifecycle_id"]].append(error)
        horizon_reports[horizon] = {
            "rows": len(local),
            "lifecycles": len(by_lifecycle),
            "batches": len(by_batch),
            "mean_signed_error_hours": float(np.mean(local_errors)) if local_errors else None,
            "signed_error_quantiles_hours": _quantiles(local_errors),
            "mae_hours": float(np.mean(np.abs(local_errors))) if local_errors else None,
            "lifecycle_macro_mae_hours": _macro_metric(local, "corrected", absolute=True),
            "lifecycle_macro_bias_hours": _macro_metric(local, "corrected", absolute=False),
            "correction_quantiles_hours": _quantiles(local_corrections),
            "fraction_prediction_increased": float(np.mean(np.asarray(local_corrections) > 0.0)) if local_corrections else None,
            "per_batch": {
                batch: {
                    "rows": len(values),
                    "bias_hours": float(np.mean(values)),
                    "mae_hours": float(np.mean(np.abs(values))),
                }
                for batch, values in sorted(by_batch.items())
            },
            "worst_lifecycle_by_mae": (
                max(
                    ({"lifecycle_id": lifecycle, "rows": len(values), "mae_hours": float(np.mean(np.abs(values))), "bias_hours": float(np.mean(values))}
                     for lifecycle, values in by_lifecycle.items()),
                    key=lambda row: row["mae_hours"],
                ) if by_lifecycle else None
            ),
        }
    return {
        "candidate": name,
        "rows": len(rows),
        "lifecycles": len({row["lifecycle_id"] for row in rows}),
        "batches": len({row["batch_id"] for row in rows}),
        "overall_mae_hours": float(np.mean(np.abs(errors))) if errors else None,
        "overall_bias_hours": float(np.mean(errors)) if errors else None,
        "lifecycle_macro_mae_hours": _macro_metric(rows, "corrected", absolute=True),
        "lifecycle_macro_bias_hours": _macro_metric(rows, "corrected", absolute=False),
        "correction_quantiles_hours": _quantiles(corrections),
        "fraction_prediction_increased": float(np.mean(np.asarray(corrections) > 0.0)) if corrections else None,
        "horizons": horizon_reports,
    }


def _crossfit_correction(
    rows: list[dict[str, Any]],
    candidate: str,
    criteria: RULV27AcceptanceCriteria,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predicted: list[dict[str, Any]] = []
    for held_batch in sorted({row["batch_id"] for row in rows}):
        training = [row for row in rows if row["batch_id"] != held_batch]
        held = [row for row in rows if row["batch_id"] == held_batch]
        correction = (
            _fit_identity_bias_map(training)
            if candidate == "identity"
            else _fit_zero_anchored_correction(
                training,
                maximum_correction_hours=criteria.maximum_correction_hours,
                maximum_slope=criteria.maximum_correction_slope,
            )
        )
        for row in held:
            corrected, stratum = apply_v2_6_bias_correction(row["point"], correction)
            predicted.append({**row, "corrected": corrected, "bias_stratum": stratum})
    return predicted, _candidate_diagnostics(predicted, candidate)


def _supported_horizons_pass(report: dict[str, Any], criteria: RULV27AcceptanceCriteria) -> bool:
    for horizon, _, _ in HORIZONS:
        row = report["horizons"][horizon]
        if (
            row["rows"] == 0
            or row["lifecycles"] < criteria.minimum_lifecycles_per_horizon
            or row["batches"] < criteria.minimum_batches_per_horizon
            or abs(row["mean_signed_error_hours"]) > criteria.max_absolute_horizon_bias_hours
        ):
            return False
    return True


def compare_v2_7_critical_corrections(
    rows: list[dict[str, Any]],
    criteria: RULV27AcceptanceCriteria | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    criteria = criteria or RULV27AcceptanceCriteria()
    identity_rows, identity = _crossfit_correction(rows, "identity", criteria)
    constrained_rows, constrained = _crossfit_correction(rows, "zero_anchored", criteria)
    identity["passes_frozen_horizon_support_and_bias"] = _supported_horizons_pass(identity, criteria)
    constrained["passes_frozen_horizon_support_and_bias"] = _supported_horizons_pass(constrained, criteria)
    horizon_mae_noninferior = all(
        constrained["horizons"][name]["mae_hours"]
        <= identity["horizons"][name]["mae_hours"] + criteria.max_candidate_horizon_mae_increase_hours
        for name, _, _ in HORIZONS
    )
    identity_worst_bias = max(abs(identity["horizons"][name]["mean_signed_error_hours"]) for name, _, _ in HORIZONS)
    constrained_worst_bias = max(abs(constrained["horizons"][name]["mean_signed_error_hours"]) for name, _, _ in HORIZONS)
    constrained_survives = bool(
        constrained["passes_frozen_horizon_support_and_bias"]
        and horizon_mae_noninferior
        and constrained["overall_mae_hours"] <= identity["overall_mae_hours"]
        and constrained["lifecycle_macro_mae_hours"] <= identity["lifecycle_macro_mae_hours"]
        and constrained_worst_bias < identity_worst_bias
    )
    constrained.update({
        "horizon_mae_noninferior_to_identity": horizon_mae_noninferior,
        "overall_mae_not_worse_than_identity": constrained["overall_mae_hours"] <= identity["overall_mae_hours"],
        "lifecycle_macro_mae_not_worse_than_identity": constrained["lifecycle_macro_mae_hours"] <= identity["lifecycle_macro_mae_hours"],
        "worst_horizon_bias_improved": constrained_worst_bias < identity_worst_bias,
        "survives_identity_first_rules": constrained_survives,
    })
    selected = "zero_anchored" if constrained_survives else "identity"
    return selected, constrained_rows if constrained_survives else identity_rows, {
        "selection_role": "fresh_v2_7_fit_outer_batch_crossfit_only",
        "selection_rule": "identity wins unless the sole constrained candidate passes every horizon and dominates identity",
        "selected_candidate": selected,
        "selected_passes_frozen_horizon_support_and_bias": (
            constrained["passes_frozen_horizon_support_and_bias"]
            if constrained_survives else identity["passes_frozen_horizon_support_and_bias"]
        ),
        "candidates": {"identity": identity, "zero_anchored": constrained},
    }


def _prepare_v2_7_bundle(
    frozen: dict[str, Any], contracts: dict[str, Any], *, provisional: bool,
) -> dict[str, Any]:
    artifact = frozen["rul_model"]
    if artifact.get("version") != RUL_MODEL_VERSION_V2_6:
        raise ValueError("v2.7 requires the frozen v2.6 target-specific artifact")
    point_hashes = _frozen_point_estimator_hashes(frozen)
    artifact["version"] = RUL_MODEL_VERSION_V2_7
    artifact["target_contracts"] = contracts
    artifact["v2_7_point_contract_hashes"] = point_hashes
    artifact["v2_7_development_only"] = True
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
            calibration.update({
                "method": RUL_CALIBRATION_METHOD_V2_1,
                "global_margin_hours": 0.0,
                "buckets": {},
                "minimum_interval_widths": {},
                "v2_7_provisional": True,
            })
        else:
            calibration.update(contracts[target]["calibration"])
            calibration["v2_7_provisional"] = False
        artifact["targets"][target]["calibration"] = calibration
    forecastability = dict(artifact.get("forecastability_contract") or {})
    forecastability["mode"] = "target_specific_v2_7_identity_first_critical"
    artifact["forecastability_contract"] = forecastability
    return frozen


def _v2_7_metrics(rows: list[dict[str, str]], target: str) -> dict[str, Any]:
    metrics = _acceptance_metrics(rows, target)
    maximum_horizon = 24.0 if target == "warning" else 48.0
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        truth = _optional_float(row.get(f"true_hours_to_{target}"))
        point = _optional_float(row.get(f"estimated_hours_to_{target}"))
        status = str(row.get("predicted_status") or "").upper()
        observed = status == "CRITICAL" or (target == "warning" and status == "WARNING")
        if (
            not observed
            and truth is not None and 0.0 < truth <= maximum_horizon
            and point is not None
            and _bool(row.get(f"{target}_rul_serviceable_intent"))
        ):
            grouped[row["lifecycle_id"]].append(row)
    good = pairs = 0
    for local in grouped.values():
        local.sort(key=lambda row: row.get("timestamp") or "")
        for before, after in zip(local, local[1:]):
            earlier = _optional_float(before.get(f"estimated_hours_to_{target}"))
            later = _optional_float(after.get(f"estimated_hours_to_{target}"))
            if earlier is not None and later is not None:
                pairs += 1
                good += int(later <= earlier + 1e-9)
    metrics["active_monotonicity"] = float(good / pairs) if pairs else None
    metrics["active_monotonic_pairs"] = pairs
    return metrics


def _selector_contract_hashes(bundle: dict[str, Any]) -> dict[str, str]:
    contracts = bundle["rul_model"].get("target_contracts") or {}
    return {
        target: joblib.hash({
            "selector_model": contracts[target].get("selector_model"),
            "selector_columns": contracts[target].get("selector_columns"),
            "activation_threshold": contracts[target].get("activation_threshold"),
            "deactivation_threshold": contracts[target].get("deactivation_threshold"),
            "confirmations": contracts[target].get("confirmations"),
            "maximum_exact_rul_horizon_hours": contracts[target].get("maximum_exact_rul_horizon_hours"),
        })
        for target in TARGETS
    }


def _preacceptance_checks(
    metrics: dict[str, dict[str, Any]],
    selection_reports: dict[str, Any],
    criteria: RULV27AcceptanceCriteria,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for target in TARGETS:
        row = metrics[target]
        width_limit = (
            criteria.max_warning_mean_interval_width_hours
            if target == "warning" else criteria.max_critical_mean_interval_width_hours
        )
        max_mae = criteria.max_warning_mae_hours if target == "warning" else criteria.max_critical_mae_hours
        checks.update({
            f"{target}_correction_selection": bool(selection_reports[target]["selection_passed"]),
            f"{target}_mae": row["mae_hours"] is not None and row["mae_hours"] <= max_mae,
            f"{target}_availability": row["active_region_availability"] is not None and row["active_region_availability"] >= criteria.min_active_region_availability,
            f"{target}_coverage": row["interval_coverage"] is not None and row["interval_coverage"] >= criteria.min_interval_coverage,
            f"{target}_macro_coverage": row["macro_lifecycle_coverage"] is not None and row["macro_lifecycle_coverage"] >= criteria.min_macro_lifecycle_coverage,
            f"{target}_width": row["mean_interval_width_hours"] is not None and row["mean_interval_width_hours"] <= width_limit,
            f"{target}_row_breadth": row["active_row_fraction"] is not None and row["active_row_fraction"] >= criteria.minimum_active_row_fraction,
            f"{target}_lifecycle_breadth": row["active_lifecycle_fraction"] is not None and row["active_lifecycle_fraction"] >= criteria.minimum_active_lifecycle_fraction,
            f"{target}_batch_breadth": row["active_batch_fraction"] is not None and row["active_batch_fraction"] >= criteria.minimum_active_batch_fraction,
            f"{target}_active_monotonicity": row["active_monotonicity"] is not None and row["active_monotonicity"] >= criteria.min_active_monotonicity,
        })
        for horizon, diagnostic in row["horizon_bias"].items():
            if diagnostic["rows"]:
                supported = (
                    diagnostic["lifecycles"] >= criteria.minimum_lifecycles_per_horizon
                    and diagnostic["batches"] >= criteria.minimum_batches_per_horizon
                )
                checks[f"{target}_{horizon}_support"] = supported
                checks[f"{target}_{horizon}_bias"] = bool(
                    supported and abs(diagnostic["bias_hours"]) <= criteria.max_absolute_horizon_bias_hours
                )
    return checks


def train_rul_v2_7_candidate(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    frozen_v2_6_model_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    freeze_manifest_path: str | Path,
    sensor_config: SensorConfig,
    *,
    criteria: RULV27AcceptanceCriteria | None = None,
) -> dict[str, Any]:
    criteria = criteria or RULV27AcceptanceCriteria()
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(Path(consumed_registry_path).read_text(encoding="utf-8"))
    if manifest.get("corpus_version") != V2_7_CORPUS_VERSION:
        raise ValueError("Unsupported v2.7 corpus manifest")
    if manifest.get("acceptance", {}).get("status") != "PREDECLARED_NOT_GENERATED":
        raise RuntimeError("v2.7 training requires acceptance to remain ungenerated")
    if registry.get("v2_7_acceptance_generated") is not False:
        raise RuntimeError("v2.7 acceptance is already generated")
    for role in ("fit", "calibration"):
        path = Path(manifest["role_files"][role]["path"])
        assert_not_external_rul_evaluation_input(path)
        if _sha256(path) != manifest["role_files"][role]["sha256"]:
            raise RuntimeError(f"v2.7 {role} hash mismatch")
    frozen_path = Path(frozen_v2_6_model_path)
    frozen = joblib.load(frozen_path)
    if frozen.get("rul_model", {}).get("version") != RUL_MODEL_VERSION_V2_6:
        raise ValueError("v2.7 requires a frozen v2.6 target-specific artifact")
    frozen_point_hashes = _frozen_point_estimator_hashes(frozen)
    frozen_selector_hashes = _selector_contract_hashes(frozen)
    frozen_contracts = frozen["rul_model"]["target_contracts"]
    zero_calibration = {
        "method": RUL_CALIBRATION_METHOD_V2_6,
        "candidate": "asymmetric_corrected_global",
        "strata": {"GLOBAL": {"lower_margin_hours": 0.0, "upper_margin_hours": 0.0}},
        "max_forecast_hours": 720.0,
    }
    provisional_contracts: dict[str, Any] = {}
    for target in TARGETS:
        source = frozen_contracts[target]
        provisional_contracts[target] = {
            "selector_model": source.get("selector_model"),
            "selector_columns": source.get("selector_columns", list(SELECTOR_COLUMNS)),
            "activation_threshold": source.get("activation_threshold", 0.5),
            "deactivation_threshold": source.get("deactivation_threshold", 0.45),
            "confirmations": source.get("confirmations", 3),
            "maximum_exact_rul_horizon_hours": source.get(
                "maximum_exact_rul_horizon_hours", 24.0 if target == "warning" else 48.0,
            ),
            "bias_correction": _fit_identity_bias_map([]),
            "calibration": zero_calibration,
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_prepare_v2_7_bundle(joblib.load(frozen_path), provisional_contracts, provisional=True), model_output)
    provisional_contract_hash = _json_hash({
        "version": RUL_MODEL_VERSION_V2_7,
        "protocol": V2_7_PROTOCOL_VERSION,
        "point_contract": frozen_point_hashes,
        "selector_contract": frozen_selector_hashes,
    })
    fit_design_path = output / "fit_design_complete_lifecycles.csv"
    fit_design = _complete_lifecycle_design_sample(
        manifest["role_files"]["fit"]["path"],
        fit_design_path,
        lifecycles_per_batch=3,
    )
    predictions = output / "fit_provisional_predictions.csv"
    checkpoint_path = output / "fit_provisional_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    reusable = bool(
        predictions.exists()
        and checkpoint.get("provisional_contract_hash") == provisional_contract_hash
        and checkpoint.get("input_sha256") == fit_design["sha256"]
        and checkpoint.get("prediction_rows") == fit_design["rows"]
    )
    if not reusable:
        evaluate_rul(
            model_output,
            fit_design_path,
            output / "fit_provisional_report.json",
            predictions,
            sensor_config,
        )
        checkpoint_path.write_text(json.dumps({
            "provisional_contract_hash": provisional_contract_hash,
            "input_sha256": fit_design["sha256"],
            "prediction_sha256": _sha256(predictions),
            "prediction_rows": len(_read_rows(predictions)),
        }, indent=2), encoding="utf-8")
    fit_population = _read_rows(predictions)
    contracts: dict[str, Any] = {}
    selection_reports: dict[str, Any] = {}
    fit_enriched_by_target: dict[str, list[dict[str, Any]]] = {}
    for target in TARGETS:
        design_rows = _target_design_rows(fit_population, target)
        source = frozen_contracts[target]
        scores = _selector_scores(source.get("selector_model"), design_rows)
        scored = [{**row, "selector_score": float(score)} for row, score in zip(design_rows, scores)]
        activation = float(source.get("activation_threshold", 0.5))
        selected_fit = [row for row in scored if row["selector_score"] >= activation]
        if not selected_fit:
            raise RuntimeError(f"Frozen v2.6 {target} selector produced no active v2.7 design rows")
        if target == "warning":
            crossfit_rows, identity_report = _crossfit_correction(selected_fit, "identity", criteria)
            selected_name = "identity"
            comparison = {
                "selection_role": "warning_frozen_unchanged",
                "selection_rule": "WARNING correction is frozen to identity",
                "selected_candidate": "identity",
                "selected_passes_frozen_horizon_support_and_bias": True,
                "candidates": {"identity": identity_report},
            }
            selection_passed = True
            final_bias = _fit_identity_bias_map(selected_fit)
        else:
            selected_name, crossfit_rows, comparison = compare_v2_7_critical_corrections(
                selected_fit, criteria,
            )
            selection_passed = bool(comparison["selected_passes_frozen_horizon_support_and_bias"])
            final_bias = (
                _fit_identity_bias_map(selected_fit)
                if selected_name == "identity"
                else _fit_zero_anchored_correction(
                    selected_fit,
                    maximum_correction_hours=criteria.maximum_correction_hours,
                    maximum_slope=criteria.maximum_correction_slope,
                )
            )
        contracts[target] = {
            "selector_model": source.get("selector_model"),
            "selector_columns": source.get("selector_columns", list(SELECTOR_COLUMNS)),
            "activation_threshold": source.get("activation_threshold", 0.5),
            "deactivation_threshold": source.get("deactivation_threshold", 0.45),
            "confirmations": source.get("confirmations", 3),
            "maximum_exact_rul_horizon_hours": source.get(
                "maximum_exact_rul_horizon_hours", 24.0 if target == "warning" else 48.0,
            ),
            "bias_correction": final_bias,
            "calibration": zero_calibration,
        }
        fit_enriched = _add_base_interval([
            {
                **row,
                "score": row.get("selector_score"),
                "lifecycle_age": _optional_float(row.get("rul_history_hours")),
            }
            for row in crossfit_rows
        ], frozen["rul_model"]["targets"][target])
        fit_enriched_by_target[target] = fit_enriched
        selection_reports[target] = {
            "selector_frozen_from_v2_6": True,
            "selector_contract_hash": frozen_selector_hashes[target],
            "selector_design_rows": len(scored),
            "selector_active_rows": len(selected_fit),
            "correction_comparison": comparison,
            "selected_bias_correction": selected_name,
            "selection_passed": selection_passed,
        }
    joblib.dump(_prepare_v2_7_bundle(joblib.load(frozen_path), contracts, provisional=False), model_output)
    corrected_calibration_path = output / "calibration_corrected_uncalibrated_predictions.csv"
    evaluate_rul(
        model_output,
        manifest["role_files"]["calibration"]["path"],
        output / "calibration_corrected_uncalibrated_report.json",
        corrected_calibration_path,
        sensor_config,
    )
    corrected_calibration_rows = _read_rows(corrected_calibration_path)
    shift_reports: dict[str, Any] = {}
    for target in TARGETS:
        selected_calibration = _runtime_active_calibration_rows(corrected_calibration_rows, target)
        if not selected_calibration:
            raise RuntimeError(f"No final-runtime-serviceable {target} calibration rows")
        calibration = _fit_asymmetric_calibration(selected_calibration)
        contracts[target]["calibration"] = calibration
        selection_reports[target]["calibration_population"] = "fresh_v2_7_exact_runtime_serviceable_intent"
        selection_reports[target]["calibration_metrics"] = _calibration_metrics(selected_calibration, calibration)
        shift_reports[target] = _population_shift(
            fit_enriched_by_target[target], selected_calibration, target,
        )
    final_bundle = _prepare_v2_7_bundle(joblib.load(frozen_path), contracts, provisional=False)
    if _frozen_point_estimator_hashes(final_bundle) != frozen_point_hashes:
        raise RuntimeError("v2.7 modified the frozen Extra Trees point estimator contract")
    if _selector_contract_hashes(final_bundle) != frozen_selector_hashes:
        raise RuntimeError("v2.7 modified the frozen v2.6 selector contract")
    joblib.dump(final_bundle, model_output)
    parity = parity_replay(
        model_output,
        manifest["role_files"]["calibration"]["path"],
        output,
        sensor_config,
        tolerance=criteria.max_parity_float_difference,
        report_version=RUL_MODEL_VERSION_V2_7,
    )
    if not parity["passed"]:
        raise RuntimeError("v2.7 final-artifact calibration parity failed; acceptance remains ungenerated")
    parity_rows = _read_rows(output / "parity_reloaded.csv")
    preacceptance_metrics = {target: _v2_7_metrics(parity_rows, target) for target in TARGETS}
    checks = _preacceptance_checks(preacceptance_metrics, selection_reports, criteria)
    acceptance_generation_authorized = all(checks.values())
    freeze = {
        "freeze_version": "rul_v2_7_preacceptance_freeze_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_output.resolve()),
        "model_sha256": _sha256(model_output),
        "frozen_v2_6_model_sha256": _sha256(frozen_path),
        "frozen_point_contract_hashes": frozen_point_hashes,
        "frozen_selector_contract_hashes": frozen_selector_hashes,
        "manifest_sha256": _sha256(manifest_path),
        "fit_sha256": manifest["role_files"]["fit"]["sha256"],
        "calibration_sha256": manifest["role_files"]["calibration"]["sha256"],
        "acceptance_status": "PREDECLARED_NOT_GENERATED",
        "parity_report": parity,
        "preacceptance_metrics": preacceptance_metrics,
        "preacceptance_checks": checks,
        "acceptance_generation_authorized": acceptance_generation_authorized,
        "target_contract_hash": _json_hash({
            target: {key: value for key, value in contracts[target].items() if key != "selector_model"}
            for target in TARGETS
        }),
        "criteria": asdict(criteria),
        "acceptance_content_parsed_before_freeze": False,
        "runtime_code_sha256": {
            name: _sha256(Path(__file__).with_name(name))
            for name in ("rul_v2_7.py", "rul_v2_6.py", "rul_ml.py", "rul_features_v2_4.py", "rul_evaluation.py", "monitor.py")
        },
    }
    freeze_path = Path(freeze_manifest_path)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    (output / "critical_identity_first_selection_report.json").write_text(
        json.dumps(selection_reports, indent=2), encoding="utf-8",
    )
    (output / "preacceptance_population_shift_report.json").write_text(
        json.dumps(shift_reports, indent=2), encoding="utf-8",
    )
    report = {
        "version": RUL_MODEL_VERSION_V2_7,
        "protocol_version": V2_7_PROTOCOL_VERSION,
        "development_only": True,
        "point_predictor_retrained": False,
        "selector_retrained": False,
        "warning_pipeline_changed": False,
        "generator_changed": False,
        "feature_contract_changed": False,
        "manufacturer_safety_behavior_changed": False,
        "protected_external_datasets_used": False,
        "v2_6_evidence_reused_for_selection_or_calibration": False,
        "target_selection": selection_reports,
        "fit_design_complete_lifecycle_sample": fit_design,
        "final_artifact_calibration_parity": parity,
        "acceptance_status": "NOT_GENERATED",
        "preacceptance_metrics": preacceptance_metrics,
        "preacceptance_checks": checks,
        "preacceptance_candidate_gates_passed": acceptance_generation_authorized,
        "acceptance_generation_authorized": acceptance_generation_authorized,
        "sealed_holdout_authorized": False,
        "model_sha256": freeze["model_sha256"],
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def generate_rul_v2_7_acceptance_after_parity(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    freeze_manifest_path: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    registry_path = Path(consumed_registry_path)
    freeze_path = Path(freeze_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("parity_report", {}).get("passed"):
        raise RuntimeError("Acceptance generation is forbidden until v2.7 final-artifact parity passes")
    if not freeze.get("acceptance_generation_authorized"):
        raise RuntimeError("Acceptance generation is forbidden because v2.7 preacceptance gates failed")
    if freeze.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("v2.7 manifest changed after parity freeze")
    if registry.get("v2_7_acceptance_generated"):
        raise RuntimeError("v2.7 acceptance has already been generated")
    spec = manifest["acceptance"]
    if spec.get("status") != "PREDECLARED_NOT_GENERATED":
        raise RuntimeError("Acceptance is not in the predeclared ungenerated state")
    role_file, batches = _generate_role_v2_7(
        Path(manifest["role_files"]["fit"]["path"]).parent,
        "acceptance",
        [int(seed) for seed in spec["seeds"]],
        lifecycles_per_batch=int(spec["lifecycles_per_batch"]),
        machines=int(spec["machines"]),
        cadence_seconds=int(spec["cadence_seconds"]),
        line_sel=str(spec["line_sel"]),
        ordinal_start=300,
    )
    manifest["role_files"]["acceptance"] = role_file
    manifest["batches"].extend(batches)
    manifest["acceptance"].update({
        "status": "GENERATED_UNOPENED",
        "sha256": role_file["sha256"],
        "content_parsed_before_parity": False,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry.update({
        "v2_7_acceptance_generated": True,
        "v2_7_acceptance_opened": False,
        "v2_7_acceptance_consumed": False,
    })
    registry["v2_7_role_locked_batches"].extend([
        {"batch_id": row["batch_id"], "seed": row["seed"], "role": row["role"], "sha256": row["source_sha256"]}
        for row in batches
    ])
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    freeze["manifest_sha256_after_acceptance_generation"] = _sha256(manifest_path)
    freeze["acceptance_sha256"] = role_file["sha256"]
    freeze["acceptance_generated_after_parity"] = True
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return manifest


def evaluate_rul_v2_7_development_acceptance(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    freeze_manifest_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    sensor_config: SensorConfig,
    *,
    criteria: RULV27AcceptanceCriteria | None = None,
) -> dict[str, Any]:
    return evaluate_rul_v2_6_development_acceptance(
        manifest_path,
        consumed_registry_path,
        freeze_manifest_path,
        model_path,
        output_dir,
        sensor_config,
        criteria=criteria or RULV27AcceptanceCriteria(),
        registry_namespace="v2_7",
        report_version=RUL_MODEL_VERSION_V2_7,
        metrics_function=_v2_7_metrics,
    )
