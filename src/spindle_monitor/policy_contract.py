"""Serialization-safe forecast metadata and policy contract.

This module deliberately contains no estimator/runtime code.  It is the one
place where trainer, registry, replay, evaluator, and promotion agree on the
metadata versions and on which targets are required for the product contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


MODEL_METADATA_SCHEMA_VERSION = "3.0"
FORECAST_POLICY_CONTRACT_VERSION = "3.0"
FORECAST_POLICY_VERSION = "2.0"
URGENCY_POLICY_VERSION = "1.0"


@dataclass(frozen=True)
class ForecastPolicyContract:
    # Immediate manufacturer protection is independent of every forecast
    # target.  The production forecast contract therefore requires only the
    # useful long/mid-horizon planning targets.  Short-horizon and ETA targets
    # remain independently loadable/actionable when their own evidence passes.
    required_eta_targets: tuple[str, ...] = ()
    required_probability_targets: tuple[str, ...] = (
        "probability_warning_12h",
        "probability_critical_24h",
    )
    warning_horizons_hours: tuple[int, ...] = (6, 12, 24)
    critical_horizons_hours: tuple[int, ...] = (6, 12, 24)
    threshold_lower_exclusive: float = 0.0
    threshold_upper_exclusive: float = 1.0

    @property
    def probability_targets(self) -> tuple[str, ...]:
        return tuple(f"probability_{kind}_{hours}h" for kind, horizons in
                     (("warning", self.warning_horizons_hours), ("critical", self.critical_horizons_hours))
                     for hours in horizons)

    @property
    def eta_targets(self) -> tuple[str, ...]:
        return ("time_to_warning", "time_to_critical")

    @property
    def all_targets(self) -> tuple[str, ...]:
        return self.eta_targets + self.probability_targets

    @property
    def required_targets(self) -> tuple[str, ...]:
        return self.required_eta_targets + self.required_probability_targets

    # Compatibility aliases for existing audit readers.  "mandatory" now
    # means explicitly required by the product contract, not every artifact.
    @property
    def mandatory_probability_targets(self) -> tuple[str, ...]:
        return self.required_probability_targets

    @property
    def mandatory_eta_targets(self) -> tuple[str, ...]:
        return self.required_eta_targets

    @property
    def mandatory_targets(self) -> tuple[str, ...]:
        return self.required_targets

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["model_metadata_schema_version"] = MODEL_METADATA_SCHEMA_VERSION
        data["forecast_policy_contract_version"] = FORECAST_POLICY_CONTRACT_VERSION
        data["probability_targets"] = list(self.probability_targets)
        data["eta_targets"] = list(self.eta_targets)
        data["all_targets"] = list(self.all_targets)
        data["required_targets"] = list(self.required_targets)
        data["mandatory_probability_targets"] = list(self.mandatory_probability_targets)
        data["mandatory_targets"] = list(self.mandatory_targets)
        return data


CONTRACT = ForecastPolicyContract()

URGENCY_POLICY_PARAMETERS = {
    "critical_probability_6h_target": "probability_critical_6h",
    "critical_probability_24h_target": "probability_critical_24h",
    "warning_probability_12h_target": "probability_warning_12h",
    "critical_eta_soon_hours": 6.0,
    "critical_eta_plan_hours": 24.0,
    "warning_eta_plan_hours": 12.0,
}
