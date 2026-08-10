from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any


MODEL_ENVIRONMENT_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "joblib",
    "threadpoolctl", "python-dateutil", "six", "tzdata",
)


def runtime_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in MODEL_ENVIRONMENT_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _numeric_prefix(value: str | None, parts: int) -> tuple[int, ...] | None:
    if not value:
        return None
    output: list[int] = []
    for component in value.split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        output.append(int(digits))
        if len(output) == parts:
            return tuple(output)
    return None


def environment_compatibility(training: dict[str, Any] | None) -> dict[str, Any]:
    runtime = runtime_environment()
    if not training:
        return {
            "compatible": None,
            "status": "unverified_training_environment",
            "reasons": ["Model metadata does not record its training environment."],
            "runtime_environment": runtime,
        }
    reasons: list[str] = []
    training_packages = training.get("packages", {})
    checks = (
        ("python", training.get("python_version"), runtime["python_version"], 2),
        ("scikit-learn", training_packages.get("scikit-learn"), runtime["packages"].get("scikit-learn"), 2),
        ("numpy", training_packages.get("numpy"), runtime["packages"].get("numpy"), 1),
        ("joblib", training_packages.get("joblib"), runtime["packages"].get("joblib"), 1),
    )
    for name, trained, current, parts in checks:
        if trained is None or current is None:
            reasons.append(f"{name} version is unavailable (trained={trained!r}, runtime={current!r}).")
        elif _numeric_prefix(str(trained), parts) != _numeric_prefix(str(current), parts):
            reasons.append(
                f"{name} is materially incompatible: trained={trained}, runtime={current}."
            )
    return {
        "compatible": not reasons,
        "status": "compatible" if not reasons else "materially_incompatible",
        "reasons": reasons,
        "training_environment": training,
        "runtime_environment": runtime,
    }
