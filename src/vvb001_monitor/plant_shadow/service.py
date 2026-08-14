from __future__ import annotations

from .contracts import PlantObservation
from .runtime import FrozenRuntimeRouter
from .storage import CommitResult, EvidenceStore


class RuntimeRecoveryError(RuntimeError):
    pass


class PlantShadowService:
    """Coordinates atomic evidence commits with non-transactional runtime recovery."""

    def __init__(self, store: EvidenceStore, router: FrozenRuntimeRouter) -> None:
        self.store = store
        self.router = router

    def process(self, observation: PlantObservation) -> CommitResult:
        observation = self.store.resolve_operating_context(observation)
        resolved: dict[str, PlantObservation] = {"observation": observation}

        def resolve_context(value: PlantObservation) -> PlantObservation:
            inferred = self.router.resolve_operating_context(value)
            resolved["observation"] = inferred
            return inferred

        try:
            return self.store.process_observation(
                observation,
                lambda ingestion_id, lifecycle_id: self.router.process(
                    resolved["observation"],
                    ingestion_id,
                    lifecycle_id,
                ),
                context_resolver=resolve_context,
            )
        except Exception as exc:
            if bool(getattr(exc, "plant_shadow_runtime_advanced", False)):
                try:
                    self.router.rebuild_source(
                        observation.source_key,
                        self.store.committed_observations(observation.source_key),
                    )
                except Exception as rebuild_exc:
                    raise RuntimeRecoveryError(
                        f"source {observation.source_key!r} stopped after commit failure; runtime rebuild failed"
                    ) from rebuild_exc
            raise
