from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .contracts import (
    FEATURE_CONTRACT_VERSION,
    LIFECYCLE_POLICY_VERSION,
    SCHEMA_VERSION,
    SUPPORT_POLICY_VERSION,
    TRUTH_POLICY_VERSION,
)


DEFAULT_RUNTIME_MODULES = (
    "src/vvb001_monitor/models.py",
    "src/vvb001_monitor/config.py",
    "src/vvb001_monitor/validation.py",
    "src/vvb001_monitor/features.py",
    "src/vvb001_monitor/predictor.py",
    "src/vvb001_monitor/sensor_quality.py",
    "src/vvb001_monitor/generalization.py",
    "src/vvb001_monitor/rul.py",
    "src/vvb001_monitor/rul_ml.py",
    "src/vvb001_monitor/rul_features_v2_4.py",
    "src/vvb001_monitor/rul_v2_5.py",
    "src/vvb001_monitor/rul_v2_6.py",
    "src/vvb001_monitor/rul_v2_7.py",
    "src/vvb001_monitor/monitor.py",
    "src/vvb001_monitor/plant_shadow/contracts.py",
    "src/vvb001_monitor/plant_shadow/storage.py",
    "src/vvb001_monitor/plant_shadow/source.py",
    "src/vvb001_monitor/plant_shadow/operating_context.py",
    "src/vvb001_monitor/plant_shadow/vibration_operating.py",
    "src/vvb001_monitor/plant_shadow/evaluation.py",
    "src/vvb001_monitor/plant_shadow/runtime.py",
    "src/vvb001_monitor/plant_shadow/service.py",
    "src/vvb001_monitor/plant_shadow/golden.py",
    "src/vvb001_monitor/plant_shadow/demo.py",
    "src/vvb001_monitor/plant_shadow/api.py",
    "src/vvb001_monitor/plant_shadow/commands.py",
)


class ManifestMismatch(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_lock_hash(root: str | Path) -> str:
    root_path = Path(root)
    digest = hashlib.sha256()
    for relative in ("requirements.txt", "requirements-dev.txt"):
        path = root_path / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_manifest(
    root: str | Path,
    *,
    deployment_id: str,
    runtime_modules: Iterable[str] = DEFAULT_RUNTIME_MODULES,
) -> dict[str, object]:
    root_path = Path(root).resolve()
    model = root_path / "models/rul_v2_7_full_cadence.joblib"
    accepted = root_path / "output/rul_v2_7_full_cadence/preacceptance_freeze_v2_7.json"
    registry = root_path / "output/rul_v2_7_full_cadence/consumed_evidence_registry.json"
    sealed = root_path / "output/rul_v2_7_sealed/sealed_holdout_report.json"
    golden = root_path / "config/plant_shadow_golden_replay.json"
    paths = {relative: root_path / relative for relative in runtime_modules}
    missing = [relative for relative, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing runtime module(s): " + ", ".join(missing))
    if not golden.is_file():
        raise FileNotFoundError(f"Missing golden replay fixture: {golden}")
    return {
        "manifest_version": "plant_shadow_runtime_manifest_v1",
        "deployment_id": deployment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "v2_7_model_sha256": sha256_file(model),
        "accepted_freeze_manifest_sha256": sha256_file(accepted),
        "sealed_evidence_registry_sha256": sha256_file(registry),
        "sealed_holdout_report_sha256": sha256_file(sealed),
        "golden_replay_fixture_sha256": sha256_file(golden),
        "runtime_module_hashes": {relative: sha256_file(path) for relative, path in sorted(paths.items())},
        "python_version": platform.python_version(),
        "dependency_lock_hash": dependency_lock_hash(root_path),
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "shadow_schema_version": SCHEMA_VERSION,
        "lifecycle_policy_version": LIFECYCLE_POLICY_VERSION,
        "truth_policy_version": TRUTH_POLICY_VERSION,
        "support_policy_version": SUPPORT_POLICY_VERSION,
        "plant_production_authorized": False,
    }


def write_manifest(root: str | Path, output: str | Path, *, deployment_id: str) -> dict[str, object]:
    payload = build_manifest(root, deployment_id=deployment_id)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_manifest(root: str | Path, manifest: str | Path) -> dict[str, object]:
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    expected = build_manifest(
        root,
        deployment_id=str(payload.get("deployment_id", "")),
        runtime_modules=tuple((payload.get("runtime_module_hashes") or {}).keys()),
    )
    ignored = {"created_at"}
    mismatches = {
        key: {"expected": expected.get(key), "actual": payload.get(key)}
        for key in expected
        if key not in ignored and expected.get(key) != payload.get(key)
    }
    if mismatches:
        raise ManifestMismatch(json.dumps(mismatches, sort_keys=True))
    return payload
