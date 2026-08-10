from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_DOMAINS = {"accelerated_mock", "realistic_synthetic", "plant"}
MODEL_STAGES = {
    "mock_test", "realistic_candidate", "plant_candidate", "plant_production", "archived"
}


@dataclass(frozen=True)
class DomainPolicy:
    data_domain: str
    model_stage: str
    production_eligible: bool


def validate_domain(value: str) -> str:
    if value not in DATA_DOMAINS:
        raise ValueError(f"Unsupported data domain {value!r}; choose {sorted(DATA_DOMAINS)}")
    return value


def candidate_stage(domain: str) -> str:
    validate_domain(domain)
    return {
        "accelerated_mock": "mock_test",
        "realistic_synthetic": "realistic_candidate",
        "plant": "plant_candidate",
    }[domain]


def expected_root_name(domain: str) -> str:
    return {
        "accelerated_mock": "mock",
        "realistic_synthetic": "realistic",
        "plant": "plant",
    }[validate_domain(domain)]


def validate_registry_domain(models_root: str | Path, domain: str) -> None:
    expected = expected_root_name(domain)
    name = Path(models_root).name.lower()
    if name != expected:
        raise ValueError(
            f"Cannot use {name or models_root} registry with {domain} training data; "
            f"use a models root ending in models/{expected}."
        )


def promotion_refusal(data_domain: str, requested_stage: str = "plant_production") -> str | None:
    validate_domain(data_domain)
    if requested_stage != "plant_production":
        return None
    if data_domain != "plant":
        return f"Cannot promote {data_domain} model to plant_production."
    return None

