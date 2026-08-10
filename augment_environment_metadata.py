from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spindle_monitor.environment import runtime_environment


MODEL_ROOTS = (ROOT / "models" / "mock", ROOT / "models" / "realistic")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    environment = runtime_environment()
    timestamp = datetime.now(timezone.utc).isoformat()
    report: dict[str, object] = {
        "timestamp": timestamp,
        "method": (
            "Metadata-only augmentation from the unchanged interpreter/environment "
            "used to train the bundled candidates; estimator binaries were not rewritten."
        ),
        "training_environment": environment,
        "models": {},
    }
    for model_root in MODEL_ROOTS:
        candidate = model_root / "candidate"
        metadata_path = candidate / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifacts = sorted(candidate.glob("*.joblib"))
        before = {path.name: sha256(path) for path in artifacts}
        existing = metadata.get("training_environment")
        if existing is not None and existing != environment:
            raise RuntimeError(
                f"Refusing to overwrite different training environment in {metadata_path}"
            )
        metadata["training_environment"] = environment
        metadata["environment_metadata_record"] = {
            "recorded_at": timestamp,
            "metadata_only": True,
            "model_binaries_unchanged": True,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        metrics_path = ROOT / "output" / model_root.name / "model_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["training_environment"] = environment
            metrics["environment_metadata_record"] = metadata["environment_metadata_record"]
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        after = {path.name: sha256(path) for path in artifacts}
        if before != after:
            raise RuntimeError(f"Model artifacts changed while updating {metadata_path}")
        report["models"][model_root.name] = {
            "model_version": metadata.get("model_version"),
            "artifact_sha256_before": before,
            "artifact_sha256_after": after,
            "unchanged": before == after,
        }
    destination = ROOT / "output" / "model_environment_update.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
