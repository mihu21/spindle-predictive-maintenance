from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    source_identity: str
    last_source_id: int | None = None
    last_timestamp: datetime | None = None
    baseline_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "source_identity": self.source_identity,
            "last_source_id": self.last_source_id,
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "baseline_state": self.baseline_state,
        }


class CheckpointStore:
    def __init__(self, path: str | Path, source_identity: str) -> None:
        self.path = Path(path)
        self.source_identity = source_identity

    def load(self) -> Checkpoint:
        if not self.path.exists():
            return Checkpoint(self.source_identity)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("source_identity") != self.source_identity:
            raise RuntimeError(
                "Checkpoint belongs to a different PostgreSQL source/column mapping. "
                "Use a different checkpoint path or remove the old local checkpoint intentionally."
            )
        timestamp = payload.get("last_timestamp")
        return Checkpoint(
            source_identity=self.source_identity,
            last_source_id=int(payload["last_source_id"]) if payload.get("last_source_id") is not None else None,
            last_timestamp=datetime.fromisoformat(timestamp) if timestamp else None,
            baseline_state=dict(payload.get("baseline_state") or {}),
        )

    def save(self, checkpoint: Checkpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
        temp.replace(self.path)
