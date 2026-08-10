#!/usr/bin/env python3
"""Validate candidate metadata structure and report per-target runtime policy."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from spindle_monitor.config import load_config
from spindle_monitor.forecast_policy import (
    THRESHOLD_SCHEMA_VERSION,
    normalize_metadata_contract,
    probability_target_names,
    production_eligibility,
    runtime_probability_policy,
)
from spindle_monitor.model_registry import TARGET_FILES
from spindle_monitor.policy_contract import (
    FORECAST_POLICY_CONTRACT_VERSION,
    MODEL_METADATA_SCHEMA_VERSION,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--models-directory", required=True)
    parser.add_argument("--config", default="config/thresholds.json")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    path = Path(args.metadata)
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = tuple(config.ml.required_eta_targets + config.ml.required_probability_targets)
    metadata = normalize_metadata_contract(raw, config.ml.forecast_horizons_hours, required_targets=required)
    model_directory = Path(args.models_directory)
    loaded = {target for target, filename in TARGET_FILES.items() if (model_directory / filename).is_file()}
    thresholds, eligibility, policy_reasons = runtime_probability_policy(metadata, config.ml.forecast_horizons_hours)
    structural_failures = []
    if metadata.get("metadata_invalid"): structural_failures.append(str(metadata["metadata_invalid"]))
    if metadata.get("model_metadata_schema_version") != MODEL_METADATA_SCHEMA_VERSION: structural_failures.append("model_metadata_schema_version_mismatch")
    if metadata.get("forecast_policy_contract_version") != FORECAST_POLICY_CONTRACT_VERSION: structural_failures.append("forecast_policy_contract_version_mismatch")
    if (metadata.get("probability_thresholds") or {}).get("schema_version") != THRESHOLD_SCHEMA_VERSION: structural_failures.append("threshold_schema_version_mismatch")
    target_report = {}
    for target in probability_target_names(config.ml.forecast_horizons_hours):
        value = thresholds.get(target)
        record = metadata.get("probability_targets", {}).get(target)
        threshold_numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0 < float(value) < 1
        evidence_present = isinstance(record, dict)
        target_report[target] = {
            "required": target in required,
            "artifact_present": target in loaded,
            "selected_threshold": value,
            "threshold_numeric": threshold_numeric,
            "runtime_eligible": bool(eligibility.get(target)),
            "recorded_ineligibility_reasons": (record.get("target_ineligibility_reasons", []) if evidence_present else ["target_evidence_missing"]),
            "validation_fn_rate": record.get("validation_fn_rate") if evidence_present else None,
            "validation_fp_rate": record.get("validation_fp_rate") if evidence_present else None,
            "test_fn_rate": record.get("test_fn_rate") if evidence_present else None,
            "test_fp_rate": record.get("test_fp_rate") if evidence_present else None,
        }
        if not threshold_numeric: structural_failures.append(f"invalid_threshold:{target}")
        if not evidence_present: structural_failures.append(f"target_evidence_missing:{target}")
    production_passed, production_reasons, _, _ = production_eligibility(
        metadata, config.ml.forecast_horizons_hours, loaded_targets=loaded,
        maximum_fn_rate=config.ml.maximum_probability_false_negative_rate,
        mandatory_targets=required,
    )
    report = {
        "contract_valid": not structural_failures,
        "structural_failures": sorted(set(structural_failures)),
        "metadata_path": str(path.resolve()),
        "models_directory": str(model_directory.resolve()),
        "model_version": metadata.get("model_version"),
        "model_metadata_schema_version": metadata.get("model_metadata_schema_version"),
        "forecast_policy_contract_version": metadata.get("forecast_policy_contract_version"),
        "required_targets": list(required),
        "loaded_targets": sorted(loaded),
        "runtime_probability_policy_reasons": list(policy_reasons),
        "targets": target_report,
        "production_policy_passed": production_passed,
        "production_policy_reasons": list(production_reasons),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output: Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["contract_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
