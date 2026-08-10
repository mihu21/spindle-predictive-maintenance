from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_prognostics_v6 import _first_target_onset, _labels_to_first_event, evaluate_target


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=10, freq="h")
    # WARNING begins at hour 4, falls back, then re-enters at hour 7.
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "lifecycle_id": ["lifecycle_0001"] * len(timestamps),
            "event_status": [
                "NORMAL", "NORMAL", "NORMAL", "NORMAL", "WARNING",
                "NORMAL", "NORMAL", "WARNING", "WARNING", "WARNING",
            ],
            "probability_warning_12h": [0.0, 0.0, 0.6, 0.7, 1.0, 0.9, 0.9, 1.0, 1.0, 1.0],
            "probability_critical_24h": [0.0] * len(timestamps),
            "prognostic_time_to_warning_hours": [4, 3, 2, 1, 0, np.nan, np.nan, 0, 0, 0],
            "prognostic_time_to_critical_hours": [np.nan] * len(timestamps),
            "probability_threshold_crossings": ["[]", "[]", '["probability_warning_12h"]', '["probability_warning_12h"]', "[]", "[]", "[]", "[]", "[]", "[]"],
        }
    )


def test_first_target_onset_ignores_later_reentry() -> None:
    frame = _frame()
    onset = _first_target_onset(frame, 1)
    assert onset == frame.loc[4, "timestamp"]
    eligible, labels, returned = _labels_to_first_event(frame, 1, 12.0)
    assert returned == onset
    assert eligible.tolist() == [True, True, True, True, False, False, False, False, False, False]
    assert labels.tolist() == [True, True, True, True, False, False, False, False, False, False]


def test_evaluator_counts_one_warning_event_per_lifecycle() -> None:
    report = evaluate_target(_frame(), target="warning", horizon_hours=12, threshold=0.5)
    assert report["event_count"] == 1
    assert report["detected_event_count"] == 1
    assert len(report["events"]) == 1
    assert report["events"][0]["event_timestamp"].startswith("2026-01-01T04:00:00")
    # Rows after the first onset are not eligible for another WARNING prediction window.
    assert report["eligible_row_count"] == 4


def test_evaluator_reports_operational_detection_separately() -> None:
    frame = _frame()
    # Raw probabilities cross before onset, but remove persisted operational crossings.
    frame["probability_threshold_crossings"] = ["[]"] * len(frame)
    report = evaluate_target(frame, target="warning", horizon_hours=12, threshold=0.5)
    assert report["detected_event_count"] == 1
    assert report["event_false_negative_rate"] == 0.0
    assert report["operational_detected_event_count"] == 0
    assert report["operational_event_false_negative_rate"] == 1.0
    assert report["events"][0]["operational_detected_within_horizon"] is False


def test_evaluator_reports_false_alert_time_and_duration() -> None:
    timestamps = pd.date_range("2026-01-01", periods=6, freq="h")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "lifecycle_id": ["healthy"] * 6,
        "event_status": ["NORMAL"] * 6,
        "probability_warning_12h": [0.0, 0.8, 0.8, 0.0, 0.8, 0.0],
        "probability_critical_24h": [0.0] * 6,
        "prognostic_time_to_warning_hours": [np.nan] * 6,
        "prognostic_time_to_critical_hours": [np.nan] * 6,
        "probability_threshold_crossings": ["[]", '["probability_warning_12h"]', '["probability_warning_12h"]', "[]", '["probability_warning_12h"]', "[]"],
    })
    report = evaluate_target(frame, target="warning", horizon_hours=12, threshold=0.5)
    assert report["operational_false_alert_episode_count"] == 2
    assert report["operational_false_alert_hours"] == 3.0
    assert report["operational_false_alert_episode_duration_hours"]["median"] == 1.5
    assert report["operational_false_alert_episode_duration_hours"]["maximum"] == 2.0
