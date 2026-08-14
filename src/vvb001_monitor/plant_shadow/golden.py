from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config import SensorConfig
from .contracts import OperatingState, OperatingStateSource, PlantObservation
from .manifest import sha256_file
from .runtime import FrozenSourceRuntime


GOLDEN_REPLAY_VERSION = "plant_shadow_golden_replay_v1"


def golden_observations() -> list[PlantObservation]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = []
    for index in range(24):
        progress = index / 23.0
        arms = 5.0 + 1.2 * progress
        apeak = arms * (2.6 + 0.2 * progress)
        result.append(
            PlantObservation(
                source_key="golden-source",
                source_row_id=str(index + 1),
                event_timestamp=start + timedelta(minutes=5 * index),
                line_sel="GOLDEN_LINE",
                machine_id="GOLDEN_MACHINE",
                vrms=2.0 + 1.1 * progress,
                arms=arms,
                apeak=apeak,
                crest=apeak / arms,
                temp=32.0 + 4.0 * progress,
                observed_at=start + timedelta(minutes=5 * index, seconds=2),
                raw_payload={"fixture_index": index},
                operating_state=OperatingState.RUNNING,
                operating_state_source=OperatingStateSource.SYNTHETIC_FIXTURE,
                operating_state_confidence=1.0,
            )
        )
    return result


def _normalize(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def replay_signature(model_path: str | Path, sensor_config: SensorConfig) -> dict[str, Any]:
    runtime = FrozenSourceRuntime(
        "golden-source",
        model_path,
        sensor_config,
        deployment_id="golden-deployment",
        runtime_manifest_id="golden-manifest",
    )
    include = (
        "health_state_model",
        "health_state_manufacturer",
        "health_state_source",
        "forecast_state",
        "warning_point_hours",
        "warning_lower_hours",
        "warning_upper_hours",
        "warning_serviceable",
        "warning_withhold_reason",
        "critical_point_hours",
        "critical_lower_hours",
        "critical_upper_hours",
        "critical_serviceable",
        "critical_withhold_reason",
        "model_state_reset_gap",
        "quality_status",
        "sensor_quality_status",
        "rul_reliability",
        "rul_method",
        "warning_forecastability_state",
        "critical_forecastability_state",
    )
    outputs = []
    for index, observation in enumerate(golden_observations(), start=1):
        commit = runtime.process(observation, index, "golden-lifecycle")
        prediction = commit.prediction or {}
        outputs.append({"row": index, **{key: _normalize(prediction.get(key)) for key in include}})
    canonical = json.dumps(outputs, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "golden_replay_version": GOLDEN_REPLAY_VERSION,
        "model_artifact_sha256": sha256_file(model_path),
        "row_count": len(outputs),
        "canonical_output_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def verify_golden_replay(
    fixture_path: str | Path,
    model_path: str | Path,
    sensor_config: SensorConfig,
) -> dict[str, Any]:
    expected = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    actual = replay_signature(model_path, sensor_config)
    if expected != actual:
        raise RuntimeError(
            "GOLDEN_REPLAY_MISMATCH: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )
    return actual
