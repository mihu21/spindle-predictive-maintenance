from __future__ import annotations

from datetime import datetime, timezone

import pytest

from vvb001_monitor.plant_shadow.contracts import (
    ActivationLevel,
    CensoringKind,
    EndpointClass,
    EndpointPrecision,
    PlantObservation,
    SourceDefinition,
    TargetName,
    TruthEligibility,
    default_endpoint_decisions,
)


def test_machine_uid_includes_source_key_and_requires_durable_identity():
    obs = PlantObservation(
        "source-a", "42", datetime(2026, 1, 1, tzinfo=timezone.utc), "LINE", "M1",
        1.0, 2.0, 3.0, 1.5, 30.0,
    )
    assert obs.uid == "source-a::LINE::M1"
    assert obs.to_legacy_reading(7).source_id == 7
    with pytest.raises(ValueError, match="durable non-empty"):
        PlantObservation(
            "source-a", "", datetime(2026, 1, 1, tzinfo=timezone.utc), "LINE", "M1",
            1.0, 2.0, 3.0, 1.5, 30.0,
        )


def test_continuous_source_rejects_ctid_and_timestamp_only_identity():
    for row_id in (None, "ctid"):
        source = SourceDefinition(
            "source-a", "A", "ENV:DSN", "public", "sensor", "timestamp", row_id,
            activation_level=ActivationLevel.SHADOW_MONITORING,
        )
        with pytest.raises(ValueError, match="SOURCE_IDENTITY_UNSAFE"):
            source.validate_activation()


def test_endpoint_policy_is_target_specific_and_censors_preventive_work():
    warning = default_endpoint_decisions(EndpointClass.WARNING_ONSET, EndpointPrecision.EXACT_TIMESTAMP)
    assert warning[0].target == TargetName.WARNING
    assert warning[0].eligibility == TruthEligibility.ELIGIBLE_EXACT
    assert warning[1].eligibility == TruthEligibility.INELIGIBLE

    maintenance = default_endpoint_decisions(
        EndpointClass.PREVENTIVE_MAINTENANCE,
        EndpointPrecision.EXACT_TIMESTAMP,
    )
    assert maintenance[1].target == TargetName.CRITICAL
    assert maintenance[1].eligibility == TruthEligibility.CENSORED
    assert maintenance[1].censoring == CensoringKind.RIGHT_CENSORED


def test_date_only_onset_is_not_exact_truth():
    decisions = default_endpoint_decisions(EndpointClass.CRITICAL_ONSET, EndpointPrecision.DATE_ONLY)
    assert decisions[1].eligibility == TruthEligibility.INELIGIBLE
