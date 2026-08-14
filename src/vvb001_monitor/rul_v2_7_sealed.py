from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import SensorConfig
from .rul_evaluation import evaluate_rul
from .rul_ml import RUL_MODEL_VERSION_V2_7
from .rul_v2_4 import _read_rows, _sha256
from .rul_v2_6 import postacceptance_shift_report, process_rss_bytes, runtime_environment_report
from .rul_v2_7 import (
    RULV27AcceptanceCriteria,
    TARGETS,
    V2_7_GENERATOR_VERSION,
    _all_consumed_seeds,
    _generate_role_v2_7,
    _preacceptance_checks,
    _v2_7_metrics,
)


SEALED_PROTOCOL_VERSION = "rul_v2_7_one_time_sealed_holdout_v1"
SEALED_EVIDENCE_DOMAIN = "realistic_synthetic"


def _verify_frozen_runtime(freeze: dict[str, Any]) -> None:
    for name, expected in (freeze.get("runtime_code_sha256") or {}).items():
        path = Path(__file__).with_name(name)
        if not path.exists() or _sha256(path) != expected:
            raise RuntimeError(f"Frozen v2.7 runtime code changed: {name}")


def _reserved_seeds() -> set[int]:
    seeds = set(_all_consumed_seeds())
    for path in Path("output").glob("**/sealed_holdout_manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        seeds.update(int(value) for value in manifest.get("sealed_holdout", {}).get("seeds", []))
    return seeds


def generate_rul_v2_7_sealed_holdout(
    data_dir: str | Path,
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    acceptance_freeze_path: str | Path,
    model_path: str | Path,
    *,
    seeds: Iterable[int],
    lifecycles_per_batch: int = 8,
    machines: int = 4,
    cadence_seconds: int = 600,
    line_sel: str = "LINE_1",
    criteria: RULV27AcceptanceCriteria | None = None,
) -> dict[str, Any]:
    criteria = criteria or RULV27AcceptanceCriteria()
    sealed_seeds = [int(value) for value in seeds]
    if len(sealed_seeds) < 8:
        raise ValueError("The v2.7 sealed holdout requires at least eight independent batches")
    if len(sealed_seeds) != len(set(sealed_seeds)):
        raise ValueError("A sealed-holdout seed may be assigned only once")
    overlap = sorted(set(sealed_seeds) & _reserved_seeds())
    if overlap:
        raise ValueError(f"Protected, consumed, or reserved seeds cannot be reused: {overlap}")

    registry_path = Path(consumed_registry_path)
    freeze_path = Path(acceptance_freeze_path)
    model = Path(model_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not (
        registry.get("v2_7_acceptance_consumed")
        and registry.get("v2_7_acceptance_result") == "PASS"
        and registry.get("warning_sealed_holdout_authorized")
        and registry.get("critical_sealed_holdout_authorized")
        and registry.get("system_sealed_holdout_authorized")
    ):
        raise RuntimeError("A consumed passing v2.7 acceptance is required before sealed generation")
    if registry.get("v2_7_sealed_holdout_generated"):
        raise RuntimeError("The v2.7 sealed holdout has already been generated")
    if freeze.get("model_sha256") != _sha256(model):
        raise RuntimeError("Frozen v2.7 model hash mismatch before sealed generation")
    _verify_frozen_runtime(freeze)

    root = Path(data_dir)
    (root / "batches").mkdir(parents=True, exist_ok=True)
    role_file, batches = _generate_role_v2_7(
        root,
        "sealed_holdout",
        sealed_seeds,
        lifecycles_per_batch=lifecycles_per_batch,
        machines=machines,
        cadence_seconds=cadence_seconds,
        line_sel=line_sel,
        ordinal_start=400,
    )
    manifest = {
        "protocol_version": SEALED_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "evidence_domain": SEALED_EVIDENCE_DOMAIN,
        "generator_version": V2_7_GENERATOR_VERSION,
        "generator_changed_from_v2_7": False,
        "artifact_version": RUL_MODEL_VERSION_V2_7,
        "model_path": str(model.resolve()),
        "model_sha256": _sha256(model),
        "acceptance_freeze_path": str(freeze_path.resolve()),
        "acceptance_freeze_sha256": _sha256(freeze_path),
        "acceptance_registry_path": str(registry_path.resolve()),
        "selection_or_calibration_use_prohibited": True,
        "gate_changes_permitted": False,
        "plant_production_authorized": False,
        "criteria": asdict(criteria),
        "runtime_code_sha256": dict(freeze.get("runtime_code_sha256") or {}),
        "sealed_evaluator_code_sha256": _sha256(Path(__file__)),
        "sealed_holdout": {
            "status": "GENERATED_UNOPENED",
            "seeds": sealed_seeds,
            "lifecycles_per_batch": lifecycles_per_batch,
            "machines": machines,
            "cadence_seconds": cadence_seconds,
            "line_sel": line_sel,
            "role_file": role_file,
            "batches": batches,
            "analyst_content_opened_before_evaluation": False,
        },
    }
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry.update({
        "v2_7_sealed_holdout_generated": True,
        "v2_7_sealed_holdout_opened": False,
        "v2_7_sealed_holdout_consumed": False,
        "v2_7_sealed_holdout_result": "PENDING",
        "v2_7_sealed_holdout_manifest_sha256": _sha256(manifest_output),
        "v2_7_sealed_holdout_sha256": role_file["sha256"],
        "v2_7_sealed_role_locked_batches": [
            {"batch_id": row["batch_id"], "seed": row["seed"], "role": row["role"], "sha256": row["source_sha256"]}
            for row in batches
        ],
    })
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return manifest


def _runtime_checks(
    performance: dict[str, Any], criteria: RULV27AcceptanceCriteria,
) -> dict[str, bool]:
    return {
        "runtime_p95": performance["p95_ms_per_row"] is not None
        and performance["p95_ms_per_row"] <= criteria.max_runtime_p95_ms_per_row,
        "runtime_p99": performance["p99_ms_per_row"] is not None
        and performance["p99_ms_per_row"] <= criteria.max_runtime_p99_ms_per_row,
        "incremental_memory": performance["incremental_memory_mb"] is not None
        and performance["incremental_memory_mb"] <= criteria.max_incremental_memory_mb,
        "artifact_size": performance["artifact_bytes"] <= criteria.max_artifact_bytes,
    }


def evaluate_rul_v2_7_sealed_holdout(
    manifest_path: str | Path,
    consumed_registry_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    sensor_config: SensorConfig,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    registry_path = Path(consumed_registry_path)
    model = Path(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != SEALED_PROTOCOL_VERSION:
        raise ValueError("Unsupported v2.7 sealed-holdout manifest")
    if manifest.get("sealed_holdout", {}).get("status") != "GENERATED_UNOPENED":
        raise RuntimeError("The v2.7 sealed holdout is not in its one-time unopened state")
    if not registry.get("v2_7_sealed_holdout_generated"):
        raise RuntimeError("The sealed-holdout registry does not authorize evaluation")
    if registry.get("v2_7_sealed_holdout_opened") or registry.get("v2_7_sealed_holdout_consumed"):
        raise RuntimeError("The v2.7 sealed holdout can be evaluated only once")
    if registry.get("v2_7_sealed_holdout_manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("The sealed-holdout manifest changed after generation")
    if manifest.get("model_sha256") != _sha256(model):
        raise RuntimeError("Frozen v2.7 model hash mismatch before sealed evaluation")
    if manifest.get("sealed_evaluator_code_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("The sealed evaluator changed after holdout generation")
    _verify_frozen_runtime(manifest)
    holdout = Path(manifest["sealed_holdout"]["role_file"]["path"])
    if manifest["sealed_holdout"]["role_file"]["sha256"] != _sha256(holdout):
        raise RuntimeError("The sealed-holdout evidence hash changed after generation")

    # Mark opened before inference. A crash cannot make the evidence reusable.
    registry.update({
        "v2_7_sealed_holdout_opened": True,
        "v2_7_sealed_holdout_opened_at": datetime.now(timezone.utc).isoformat(),
    })
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    criteria = RULV27AcceptanceCriteria(**manifest["criteria"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "sealed_holdout_runtime_predictions.csv"
    model_sha256_before = _sha256(model)
    rss_before = process_rss_bytes()
    evaluate_rul(
        model,
        holdout,
        output / "sealed_holdout_runtime_report.json",
        predictions,
        sensor_config,
    )
    rss_after = process_rss_bytes()
    if _sha256(model) != model_sha256_before:
        raise RuntimeError("Sealed evaluation mutated the frozen v2.7 artifact")
    rows = _read_rows(predictions)
    metrics = {target: _v2_7_metrics(rows, target) for target in TARGETS}
    frozen_selection = {target: {"selection_passed": True} for target in TARGETS}
    checks = _preacceptance_checks(metrics, frozen_selection, criteria)
    latencies = [
        value for row in rows[100:]
        if (value := _optional_latency(row.get("rul_inference_latency_ms"))) is not None
    ]
    performance = {
        "warmup_rows_excluded": min(100, len(rows)),
        "measured_rows": len(latencies),
        "p95_ms_per_row": float(np.quantile(latencies, 0.95)) if latencies else None,
        "p99_ms_per_row": float(np.quantile(latencies, 0.99)) if latencies else None,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "incremental_memory_mb": (
            max(0, rss_after - rss_before) / (1024.0 * 1024.0)
            if rss_before is not None and rss_after is not None else None
        ),
        "artifact_bytes": model.stat().st_size,
        "environment": runtime_environment_report(),
    }
    checks.update(_runtime_checks(performance, criteria))
    target_pass = {
        target: all(value for key, value in checks.items() if key.startswith(f"{target}_"))
        for target in TARGETS
    }
    passed = all(checks.values())
    acceptance_predictions = Path("output/rul_v2_7_full_cadence/acceptance_runtime_predictions.csv")
    shift = (
        postacceptance_shift_report(_read_rows(acceptance_predictions), rows)
        if acceptance_predictions.exists() else {}
    )
    shift_report = {
        "role": "post_sealed_hypothesis_only_no_reselection_no_retuning",
        "comparison": "consumed_development_acceptance_to_consumed_sealed_holdout",
        "underlying_summary": shift,
    }
    (output / "sealed_vs_acceptance_population_shift_report.json").write_text(
        json.dumps(shift_report, indent=2), encoding="utf-8",
    )
    report = {
        "version": RUL_MODEL_VERSION_V2_7,
        "protocol_version": SEALED_PROTOCOL_VERSION,
        "evidence_domain": SEALED_EVIDENCE_DOMAIN,
        "model_sha256": model_sha256_before,
        "criteria": asdict(criteria),
        "target_metrics": metrics,
        "checks": checks,
        "target_result": {"warning": target_pass["warning"], "critical": target_pass["critical"]},
        "all_required_gates_passed": passed,
        "synthetic_development_validation_complete": passed,
        "synthetic_candidate_locked": passed,
        "sealed_holdout_generated_or_evaluated": True,
        "sealed_holdout_consumed": True,
        "plant_production_authorized": False,
        "plant_validation_status": "NOT_STARTED",
        "runtime_performance": performance,
        "post_sealed_analysis_role": "hypothesis_only_no_reselection_no_retuning",
    }
    report_path = output / "sealed_holdout_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry.update({
        "v2_7_sealed_holdout_consumed": True,
        "v2_7_sealed_holdout_consumed_at": datetime.now(timezone.utc).isoformat(),
        "v2_7_sealed_holdout_result": "PASS" if passed else "FAIL",
        "v2_7_sealed_holdout_report_sha256": _sha256(report_path),
        "v2_7_synthetic_candidate_locked": passed,
        "v2_7_synthetic_development_validation_complete": passed,
        "plant_production_authorized": False,
        "plant_validation_status": "NOT_STARTED",
    })
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    manifest["sealed_holdout"].update({
        "status": "CONSUMED_PASS" if passed else "CONSUMED_FAIL",
        "opened_at": registry["v2_7_sealed_holdout_opened_at"],
        "consumed_at": registry["v2_7_sealed_holdout_consumed_at"],
        "runtime_predictions_sha256": _sha256(predictions),
        "report_sha256": _sha256(report_path),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def _optional_latency(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
