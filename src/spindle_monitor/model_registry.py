from __future__ import annotations

import json
import shutil
import os
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from .domain import candidate_stage, promotion_refusal
from .environment import environment_compatibility
from .forecast_policy import mandatory_production_targets, effective_required_targets, validate_model_for_production_promotion, ordered_reasons, normalize_metadata_contract


TARGET_FILES = {
    "time_to_warning": "time_to_warning.joblib",
    "time_to_critical": "time_to_critical.joblib",
    "probability_warning_6h": "probability_warning_6h.joblib",
    "probability_warning_12h": "probability_warning_12h.joblib",
    "probability_warning_24h": "probability_warning_24h.joblib",
    "probability_critical_6h": "probability_critical_6h.joblib",
    "probability_critical_12h": "probability_critical_12h.joblib",
    "probability_critical_24h": "probability_critical_24h.joblib",
}


class ModelRegistry:
    def __init__(self, root: str | Path, *, maximum_probability_false_negative_rate: float | None = None, mandatory_targets: tuple[str, ...] | None = None) -> None:
        self.root = Path(root)
        self.production = self.root / "production"
        self.candidate = self.root / "candidate"
        self.archived = self.root / "archived"
        self.audit_log = self.root / "registry_audit.jsonl"
        self.maximum_probability_false_negative_rate = maximum_probability_false_negative_rate
        self.mandatory_targets = mandatory_targets
        for directory in (self.production, self.candidate, self.archived):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _metadata_path(directory: Path) -> Path:
        return directory / "metadata.json"

    def record_audit(self, event: str, details: dict[str, Any] | None = None) -> None:
        """Append a durable, human-readable audit event for model operations."""
        self.root.mkdir(parents=True, exist_ok=True)
        entry = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": event,
            "details": details or {},
        }
        with self.audit_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, sort_keys=True) + "\n")

    def load_metadata(self, stage: str = "production") -> dict[str, Any] | None:
        directory = getattr(self, stage)
        path = self._metadata_path(directory)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            return normalize_metadata_contract(json.load(file), (6, 12, 24), required_targets=self.mandatory_targets)

    def load_model(
        self,
        target: str,
        *,
        stage: str = "production",
        expected_features: list[str] | None = None,
        expected_schema_version: str | None = None,
        expected_threshold_config_version: str | None = None,
        expected_lifecycle_config_version: str | None = None,
    ) -> tuple[Any, dict[str, Any]] | None:
        if target not in TARGET_FILES:
            raise ValueError(f"Unsupported model target: {target}")
        directory = getattr(self, stage)
        metadata = self.load_metadata(stage)
        path = directory / TARGET_FILES[target]
        if metadata is None or not path.exists():
            return None
        compatibility = environment_compatibility(metadata.get("training_environment"))
        if compatibility["compatible"] is False:
            reason = "; ".join(compatibility["reasons"])
            self.record_audit("model_load_rejected", {
                "stage": stage, "target": target,
                "reason": "materially_incompatible_runtime_environment",
                "details": reason,
            })
            warnings.warn(
                f"Model {target!r} was not loaded because its training environment "
                f"is materially incompatible: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        if compatibility["compatible"] is None:
            warnings.warn(
                f"Model {target!r} has no recorded training environment; compatibility is unverified.",
                RuntimeWarning,
                stacklevel=2,
            )
        if expected_features is not None and metadata.get("feature_names") != expected_features:
            self.record_audit("model_load_rejected", {"stage": stage, "target": target, "reason": "feature_schema_mismatch"})
            return None
        if (
            expected_schema_version is not None
            and str(metadata.get("input_schema_version")) != str(expected_schema_version)
        ):
            self.record_audit("model_load_rejected", {"stage": stage, "target": target, "reason": "input_schema_version_mismatch"})
            return None
        if (
            expected_threshold_config_version is not None
            and str(metadata.get("threshold_config_version")) != str(expected_threshold_config_version)
        ):
            self.record_audit("model_load_rejected", {"stage": stage, "target": target, "reason": "threshold_config_version_mismatch"})
            return None
        if (
            expected_lifecycle_config_version is not None
            and str(metadata.get("lifecycle_config_version")) != str(expected_lifecycle_config_version)
        ):
            self.record_audit("model_load_rejected", {"stage": stage, "target": target, "reason": "lifecycle_config_version_mismatch"})
            return None
        target_group = "probability_targets" if target.startswith("probability_") else "targets"
        targets = metadata.get(target_group, {})
        if target not in targets:
            return None
        self.record_audit("model_loaded", {
            "stage": stage, "target": target,
            "model_version": metadata.get("model_version"),
            "environment_compatibility": compatibility["status"],
        })
        # scikit-learn 1.9's tree unpickler still assigns ``ndarray.shape``.
        # NumPy 2.5 deprecates that internal assignment even though the
        # serialized estimator is compatible. Suppress only that exact,
        # dependency-internal deprecation; all other warnings remain visible.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Setting the shape on a NumPy array has been deprecated.*",
                category=DeprecationWarning,
            )
            model = joblib.load(path)
        return model, metadata

    def save_candidate(
        self,
        models: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        metadata = dict(metadata)
        inferred_domain = "accelerated_mock" if metadata.get("synthetic_data_only") else "plant"
        metadata.setdefault("data_domain", inferred_domain)
        metadata.setdefault("model_stage", candidate_stage(metadata["data_domain"]))
        metadata.setdefault("production_eligible", metadata["data_domain"] == "plant")
        for old in self.candidate.glob("*.joblib"):
            old.unlink()
        for target, model in models.items():
            if target in TARGET_FILES and model is not None:
                joblib.dump(model, self.candidate / TARGET_FILES[target])
        with self._metadata_path(self.candidate).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        self.record_audit(
            "candidate_saved",
            {
                "model_version": metadata.get("model_version"),
                "targets": sorted(models),
                "training_lifecycle_ids": metadata.get("training_lifecycle_ids", []),
                "data_domain": metadata.get("data_domain"),
                "model_stage": metadata.get("model_stage"),
            },
        )

    def promote_candidate(self, *, maximum_fn_rate: float | None = None) -> Path:
        metadata = self.load_metadata("candidate")
        if metadata is None:
            raise FileNotFoundError("No candidate metadata exists")
        self.promote_targets([
            *metadata.get("targets", {}),
            *metadata.get("probability_targets", {}),
        ], maximum_fn_rate=maximum_fn_rate)
        return self.production

    def promote_targets(self, targets: list[str], *, maximum_fn_rate: float | None = None) -> dict[str, Any]:
        """Validate, stage, verify, then atomically activate a candidate."""
        metadata = self.load_metadata("candidate")
        if metadata is None:
            raise FileNotFoundError("No candidate metadata exists")
        refusal = promotion_refusal(str(metadata.get("data_domain", "accelerated_mock"))) if self.root.name.lower() in {"mock", "realistic", "plant"} else None
        if refusal:
            self.record_audit("promotion_refused", {"reason": refusal, "data_domain": metadata.get("data_domain")})
            raise ValueError(refusal)
        if metadata.get("data_domain") == "plant":
            ceiling = maximum_fn_rate if maximum_fn_rate is not None else self.maximum_probability_false_negative_rate
            available = {t for t, f in TARGET_FILES.items() if (self.candidate / f).exists()}
            policy = validate_model_for_production_promotion(metadata, available, maximum_fn_rate=ceiling, mandatory_targets=self.mandatory_targets)
            if not policy.passed:
                self.record_audit("promotion_refused", {"reason": "production_policy_validation_failed", "reasons": policy.reasons})
                raise ValueError("Production promotion refused: " + ", ".join(policy.reasons))
        requested = [t for t in dict.fromkeys(targets) if t in TARGET_FILES]
        if metadata.get("data_domain") == "plant" and set(requested) != set(policy.effective_required_targets):
            reason = "partial_production_promotion_not_allowed"
            self.record_audit("promotion_refused", {"reason": reason, "requested_targets": requested, "required_targets": policy.effective_required_targets})
            raise ValueError(reason)
        if not requested or any(not (self.candidate / TARGET_FILES[t]).exists() for t in requested):
            raise ValueError("Promotion refused: missing_required_model")
        now = datetime.now().astimezone(); stamp = now.strftime("%Y%m%d_%H%M%S_%f")
        stage = self.root / f".promotion_stage_{uuid.uuid4().hex}"; backup = self.root / f".promotion_backup_{uuid.uuid4().hex}"
        old_metadata = self.load_metadata("production") or {"targets": {}, "probability_targets": {}, "forecast_sources": {}}
        production_metadata = json.loads(json.dumps(old_metadata))
        try:
            if self.production.exists(): shutil.copytree(self.production, stage)
            else: stage.mkdir(parents=True)
            for target in requested:
                shutil.copy2(self.candidate / TARGET_FILES[target], stage / TARGET_FILES[target])
                joblib.load(stage / TARGET_FILES[target])
                group = "probability_targets" if target.startswith("probability_") else "targets"
                detail = dict(metadata.get(group, {}).get(target, {})); detail.update({"model_version": metadata.get("model_version"), "stage": "production", "promotion_timestamp": now.isoformat()})
                production_metadata.setdefault(group, {})[target] = detail
            active_targets = {target for target, filename in TARGET_FILES.items() if (stage / filename).exists()}
            if metadata.get("data_domain") == "plant":
                missing_staged = set(policy.effective_required_targets) - active_targets
                if missing_staged:
                    raise ValueError("missing_staged_required_model:" + ",".join(sorted(missing_staged)))
            for key in ("model_metadata_schema_version", "forecast_policy_contract_version", "forecast_policy_version", "forecast_policy_contract", "feature_names", "feature_order", "feature_distribution", "input_schema_version", "threshold_config_version", "lifecycle_config_version", "synthetic_data_only", "data_domain", "production_eligible", "target_support", "probability_thresholds", "required_model_targets", "artifact_targets", "eta_target_evidence"):
                if key in metadata: production_metadata[key] = metadata[key]
            production_metadata.update({"model_version": f"production_{stamp}", "source_candidate_model_version": metadata.get("model_version"), "maturity_stage": "deployed", "model_stage": "plant_production", "last_promotion_timestamp": now.isoformat()})
            for eta in ("time_to_warning", "time_to_critical"):
                production_metadata.setdefault("forecast_sources", {}).setdefault(eta, {"source": "statistical", "model_version": None})
                if eta in requested:
                    production_metadata["forecast_sources"][eta] = {"source": "ml", "model_version": metadata.get("model_version")}
            declared, effective, missing = effective_required_targets(production_metadata, (6, 12, 24))
            production_metadata.update({"configured_mandatory_targets": list(mandatory_production_targets((6,12,24))), "effective_required_targets": list(effective), "mandatory_targets_missing_from_metadata": list(missing)})
            with self._metadata_path(stage).open("w", encoding="utf-8") as file: json.dump(production_metadata, file, indent=2, sort_keys=True)
            if metadata.get("data_domain") == "plant":
                active_policy = validate_model_for_production_promotion(production_metadata, active_targets, maximum_fn_rate=ceiling, mandatory_targets=self.mandatory_targets)
                if not active_policy.passed:
                    raise ValueError("staged_production_policy_failed: " + ", ".join(active_policy.reasons))
            if self.production.exists(): os.replace(self.production, backup)
            try: os.replace(stage, self.production)
            except Exception:
                if backup.exists(): os.replace(backup, self.production)
                raise
            if backup.exists(): shutil.rmtree(backup)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            if backup.exists() and not self.production.exists(): os.replace(backup, self.production)
            self.record_audit("promotion_failed", {"reason": "staging_or_activation_failed"})
            raise
        self.record_audit("candidate_targets_promoted", {"model_version": metadata.get("model_version"), "promoted_targets": requested})
        return {"production_path": str(self.production), "promoted_targets": requested, "archived_targets": [], "archive_path": None}
